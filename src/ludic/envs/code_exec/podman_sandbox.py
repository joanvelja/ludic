"""
Podman-HPC sandbox implementation for code execution on HPC clusters.

This module provides:
  - PodmanConfig: Configuration for Podman containers
  - PodmanHPCSandbox: Async Podman container sandbox using subprocess
  - PodmanHPCSandboxPool: Pool of Podman sandboxes with caching

Podman-HPC is a daemonless container runtime wrapper commonly available on HPC
clusters (e.g., Isambard/BRiCS). Unlike Docker, it doesn't require a daemon and
integrates with Slurm job scheduling.

Key differences from Docker implementation:
  - Uses asyncio.create_subprocess_exec instead of docker-py SDK
  - File writing via tar pipe (no put_archive API)
  - Container naming includes SLURM_JOB_ID to avoid conflicts between jobs

Usage:
  # On HPC cluster with podman-hpc available
  pool = PodmanHPCSandboxPool(n_workers=4)
  await pool.start()  # Pulls image and creates containers

  sandbox = await pool.checkout()
  result = await sandbox.execute("print('hello')")
  await pool.release(sandbox)

  await pool.shutdown()

HPC Compatibility Note (Isambard/BRiCS - December 2024):
---------------------------------------------------------
On some HPC systems (notably Isambard), podman-hpc converts pulled images to
squashfs format for shared storage. This conversion can break the container's
PATH environment variable, causing commands like `sleep`, `python`, etc. to
fail with "executable file not found in $PATH" errors.

**Workaround**: All commands in this module use absolute paths:
  - /bin/sleep, /bin/mkdir, /bin/sh
  - /usr/local/bin/python (for official Python Docker images)
  - /usr/bin/pkill

If you encounter PATH-related errors on a new HPC system, verify paths with:
  podman-hpc run --rm python:3.11-slim /bin/ls /bin/
  podman-hpc run --rm python:3.11-slim /bin/ls /usr/local/bin/

See: https://docs.isambard.ac.uk/user-documentation/guides/containers/podman-hpc/
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import tarfile
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Union

from .cache import LRUCache
from .pool import BaseSandboxPool
from .sandbox import Sandbox, SandboxPool
from .types import (
    BatchExecutionSpec,
    BatchTestResult,
    CompileResult,
    CompileStatus,
    ExecutionResult,
    RunStatus,
    TestCase,
)

# Import batch runner script using importlib.resources
# This ensures the script is available regardless of installation method
try:
    from importlib.resources import files
    _BATCH_RUNNER_SCRIPT: Optional[str] = None

    def _get_batch_runner_script() -> str:
        """Lazy-load the batch runner script from package resources."""
        global _BATCH_RUNNER_SCRIPT
        if _BATCH_RUNNER_SCRIPT is None:
            _BATCH_RUNNER_SCRIPT = (
                files("ludic.envs.code_exec")
                .joinpath("batch_runner.py")
                .read_text()
            )
        return _BATCH_RUNNER_SCRIPT

except ImportError:
    # Fallback for older Python versions
    import pkg_resources

    def _get_batch_runner_script() -> str:
        return pkg_resources.resource_string(
            "ludic.envs.code_exec", "batch_runner.py"
        ).decode("utf-8")

logger = logging.getLogger(__name__)


@dataclass
class PodmanConfig:
    """Configuration for Podman-HPC sandboxes."""

    memory_limit: str = "256m"
    cpu_quota: Optional[float] = None  # CPU limit (e.g., 0.5 = 50% of one CPU)
    network_disabled: bool = True
    working_dir: str = "/workspace"
    gpu: bool = False  # Pass --gpu flag for GPU access
    extra_args: Optional[list[str]] = None  # Additional podman-hpc run args


def _get_container_name_prefix() -> str:
    """
    Get container name prefix including SLURM_JOB_ID if in a Slurm job.

    Returns:
        Container name prefix like "ludic-sandbox-12345" or "ludic-sandbox-local"
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return f"ludic-sandbox-{slurm_job_id}"
    return "ludic-sandbox-local"


class PodmanHPCSandbox:
    """
    Async Podman-HPC container sandbox for Python code execution.

    Uses persistent containers (sleep infinity) with exec for code execution.
    All operations use asyncio.create_subprocess_exec for non-blocking I/O.

    Podman Concurrency Note:
        Podman has known issues with concurrent operations (deadlock above ~8
        simultaneous exec calls). All sandboxes in a pool share an exec_semaphore
        to prevent overwhelming podman's lock manager.
    """

    def __init__(
        self,
        container_name: str,
        image: str,
        config: PodmanConfig,
        python_version: str = "3.11",
        exec_semaphore: Optional[asyncio.Semaphore] = None,
    ):
        self._container_name = container_name
        self._image = image
        self._config = config
        self._python_version = python_version
        self._exec_semaphore = exec_semaphore  # Shared across all sandboxes in pool
        self._started = False

    @property
    def python_version(self) -> str:
        return self._python_version

    async def start(self) -> None:
        """Create and start the persistent container."""
        if self._started:
            return

        # Remove existing container if present
        await self._run_podman("rm", "-f", self._container_name, check=False)

        # Build run command
        cmd = ["run", "-d", "--name", self._container_name]

        # Resource limits
        if self._config.memory_limit:
            cmd.extend(["--memory", self._config.memory_limit])
        if self._config.cpu_quota:
            cmd.extend(["--cpus", str(self._config.cpu_quota)])
        if self._config.network_disabled:
            cmd.extend(["--network", "none"])
        if self._config.gpu:
            cmd.append("--gpu")
        if self._config.extra_args:
            cmd.extend(self._config.extra_args)

        # Image and command (use full path for HPC compatibility)
        cmd.extend([self._image, "/bin/sleep", "infinity"])

        # Capture stderr to provide useful error messages
        await self._run_podman(*cmd, capture=True)

        # Ensure workspace directory exists (use full path for HPC compatibility)
        await self._run_podman(
            "exec", self._container_name,
            "/bin/mkdir", "-p", self._config.working_dir,
            capture=True,
        )

        self._started = True

    async def stop(self) -> None:
        """Stop and remove the container."""
        if not self._started:
            return

        await self._run_podman("stop", "-t", "2", self._container_name, check=False)
        await self._run_podman("rm", "-f", self._container_name, check=False)
        self._started = False

    async def reset(self) -> None:
        """Clear workspace directory (in-place, no container restart)."""
        if not self._started:
            return

        logger.debug(f"[{self._container_name}] reset() starting podman-hpc exec...")
        start = time.perf_counter()

        await self._run_podman(
            "exec", self._container_name,
            "/bin/sh", "-c", f"rm -rf {self._config.working_dir}/*"
        )

        elapsed = time.perf_counter() - start
        logger.debug(f"[{self._container_name}] reset() completed in {elapsed:.3f}s")

    async def compile(
        self,
        code: str,
        *,
        timeout_s: float = 5.0,
    ) -> CompileResult:
        """Syntax-check code using py_compile."""
        start = time.perf_counter()

        try:
            # Write code to container
            await self._write_file("_check.py", code, timeout_s=timeout_s)

            # Run py_compile (use full path for HPC compatibility)
            proc = await asyncio.wait_for(
                self._run_podman(
                    "exec", self._container_name,
                    "/usr/local/bin/python", "-m", "py_compile",
                    f"{self._config.working_dir}/_check.py",
                    check=False, capture=True
                ),
                timeout=timeout_s,
            )

            duration_ms = (time.perf_counter() - start) * 1000

            if proc.returncode == 0:
                return CompileResult(
                    status=CompileStatus.SUCCESS,
                    duration_ms=duration_ms,
                )

            # Parse error message
            error_msg = proc.stderr or proc.stdout or ""
            line, column, clean_msg = self._parse_syntax_error(error_msg)

            # Classify error type
            status = CompileStatus.SYNTAX_ERROR
            if "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
                status = CompileStatus.IMPORT_ERROR
            elif not clean_msg:
                status = CompileStatus.UNKNOWN_ERROR

            return CompileResult(
                status=status,
                error_message=clean_msg or error_msg,
                error_line=line,
                error_column=column,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            return CompileResult(
                status=CompileStatus.TIMEOUT,
                error_message=f"Compilation timed out after {timeout_s}s",
                duration_ms=duration_ms,
            )

    async def execute(
        self,
        code: str,
        *,
        stdin: str = "",
        skip_compile: bool = False,
        timeout_s: float = 10.0,
        memory_limit_mb: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute code with full resource isolation."""
        # Step 1: Compile
        if skip_compile:
            compile_result = CompileResult(status=CompileStatus.SUCCESS)
        else:
            compile_result = await self.compile(code, timeout_s=timeout_s)

        total_start = time.perf_counter()

        if not compile_result.success:
            total_ms = (time.perf_counter() - total_start) * 1000
            return ExecutionResult(
                compile_result=compile_result,
                run_status=RunStatus.NOT_RUN,
                compile_duration_ms=compile_result.duration_ms,
                total_duration_ms=total_ms,
            )

        # Step 2: Execute
        run_start = time.perf_counter()

        try:
            # Generate unique filename to avoid race conditions
            exec_id = uuid.uuid4().hex[:8]
            exec_filename = f"_exec_{exec_id}.py"

            # Write code to container
            await self._write_file(exec_filename, code, timeout_s=timeout_s)

            # Build exec command
            exec_cmd = ["exec"]
            if stdin:
                exec_cmd.append("-i")

            # Add environment variables
            if env_vars:
                for key, val in env_vars.items():
                    exec_cmd.extend(["-e", f"{key}={val}"])

            exec_cmd.extend([
                self._container_name,
                "/usr/local/bin/python", f"{self._config.working_dir}/{exec_filename}"
            ])

            # Run with timeout
            proc = await asyncio.wait_for(
                self._run_podman(
                    *exec_cmd,
                    check=False,
                    capture=True,
                    input_data=stdin.encode("utf-8") if stdin else None,
                ),
                timeout=timeout_s,
            )

            run_ms = (time.perf_counter() - run_start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000

            # Classify run status
            exit_code = proc.returncode
            if exit_code == 0:
                run_status = RunStatus.SUCCESS
            elif exit_code == 137:  # SIGKILL (OOM)
                run_status = RunStatus.MEMORY_EXCEEDED
            elif exit_code == 143:  # SIGTERM
                run_status = RunStatus.KILLED
            else:
                run_status = RunStatus.RUNTIME_ERROR

            return ExecutionResult(
                compile_result=compile_result,
                run_status=run_status,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=exit_code,
                compile_duration_ms=compile_result.duration_ms,
                run_duration_ms=run_ms,
                total_duration_ms=total_ms,
            )

        except asyncio.TimeoutError:
            run_ms = (time.perf_counter() - run_start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000

            # Best-effort cleanup - goes through exec_semaphore so won't deadlock
            try:
                await self._run_podman(
                    "exec", self._container_name,
                    "/usr/bin/pkill", "-9", "python",
                    check=False, capture=True
                )
            except Exception:
                pass  # Best effort, reset() will clean up anyway

            return ExecutionResult(
                compile_result=compile_result,
                run_status=RunStatus.TIMEOUT,
                stderr=f"Execution timed out after {timeout_s}s",
                compile_duration_ms=compile_result.duration_ms,
                run_duration_ms=run_ms,
                total_duration_ms=total_ms,
            )

    async def _write_file(
        self,
        filename: str,
        content: str,
        *,
        timeout_s: float = 5.0,
    ) -> None:
        """
        Write a file to the container using tar pipe.

        Creates a tar archive in memory and pipes it to container.
        This is more robust than echo for handling special characters.
        """
        # Create tar archive in memory
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            file_data = content.encode("utf-8")
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(file_data)
            tarinfo.mtime = int(time.time())
            tar.addfile(tarinfo, io.BytesIO(file_data))
        tar_buffer.seek(0)

        # Pipe tar to container
        await asyncio.wait_for(
            self._run_podman(
                "exec", "-i", self._container_name,
                "tar", "-xC", self._config.working_dir,
                check=True,
                capture=True,
                input_data=tar_buffer.read(),
            ),
            timeout=timeout_s,
        )

    async def _run_podman(
        self,
        *args: str,
        check: bool = True,
        capture: bool = False,
        input_data: Optional[bytes] = None,
    ) -> "PodmanResult":
        """
        Run a podman-hpc command asynchronously.

        For 'exec' commands, acquires the shared semaphore to prevent
        overwhelming podman's lock manager (which deadlocks above ~8
        concurrent operations).

        Args:
            *args: Command arguments (e.g., "exec", container_name, "python", ...)
            check: Raise exception if command fails
            capture: Capture stdout/stderr
            input_data: Data to pipe to stdin

        Returns:
            PodmanResult with returncode, stdout, stderr
        """
        is_exec = args and args[0] == "exec"

        # Use semaphore for exec commands to prevent podman deadlock
        if is_exec and self._exec_semaphore:
            async with self._exec_semaphore:
                return await self._run_podman_inner(
                    *args, check=check, capture=capture, input_data=input_data
                )
        else:
            return await self._run_podman_inner(
                *args, check=check, capture=capture, input_data=input_data
            )

    async def _run_podman_inner(
        self,
        *args: str,
        check: bool = True,
        capture: bool = False,
        input_data: Optional[bytes] = None,
    ) -> "PodmanResult":
        """Actually run the podman-hpc command (called by _run_podman)."""
        start = time.perf_counter()

        proc = await asyncio.create_subprocess_exec(
            "podman-hpc",
            *args,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
        )

        stdout_bytes, stderr_bytes = await proc.communicate(input=input_data)

        elapsed = time.perf_counter() - start
        if elapsed > 1.0:
            cmd_preview = " ".join(args[:4])
            logger.warning(
                f"[{self._container_name}] SLOW podman-hpc {cmd_preview}... "
                f"took {elapsed:.2f}s"
            )

        result = PodmanResult(
            returncode=proc.returncode or 0,
            stdout=stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else "",
            stderr=stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else "",
        )

        if check and result.returncode != 0:
            raise PodmanError(
                f"podman-hpc {' '.join(args)} failed with exit code {result.returncode}:\n"
                f"{result.stderr}"
            )

        return result

    @staticmethod
    def _parse_syntax_error(error_msg: str) -> tuple[Optional[int], Optional[int], str]:
        """Parse Python syntax error to extract line, column, and clean message."""
        line = None
        column = None
        clean_msg = ""

        # Try to find line number
        line_match = re.search(r'line (\d+)', error_msg)
        if line_match:
            line = int(line_match.group(1))

        # Try to find column number
        col_match = re.search(r'column (\d+)', error_msg)
        if col_match:
            column = int(col_match.group(1))

        # Extract error type and message
        error_type_match = re.search(
            r'(SyntaxError|IndentationError|TabError):\s*(.+)', error_msg
        )
        if error_type_match:
            error_type = error_type_match.group(1)
            msg = error_type_match.group(2).strip()
            clean_msg = f"{error_type}: {msg}"
        else:
            # Fall back to just extracting the last line
            lines = [l.strip() for l in error_msg.split('\n') if l.strip()]
            if lines:
                clean_msg = lines[-1]

        return line, column, clean_msg

    # -------------------------------------------------------------------------
    # Batch execution (reduces semaphore acquisitions from O(N) to O(1))
    # -------------------------------------------------------------------------

    async def execute_batch(
        self,
        spec: BatchExecutionSpec,
    ) -> AsyncIterator[Union[CompileResult, ExecutionResult]]:
        """
        Execute all tests in a single batch with streaming results.

        This method reduces semaphore acquisitions from O(2N+1) to O(3) by:
        1. Bundling code, manifest, and runner into a single tar
        2. Executing the batch runner once, which runs all tests sequentially
        3. Streaming results back as JSONL

        Args:
            spec: Batch execution specification with code, tests, and options

        Yields:
            CompileResult (if compile_first=True), then ExecutionResult for each test
        """
        batch_dir = "_batch"
        batch_start = time.perf_counter()

        # Build manifest for the batch runner
        manifest = {
            "code_file": "solution.py",
            "compile_first": spec.compile_first,
            "timeout_s": spec.timeout_s,
            "stop_on_first_failure": spec.stop_on_first_failure,
            "tests": [
                {"id": t.id or f"test_{i}", "stdin": t.input, "expected": t.expected}
                for i, t in enumerate(spec.tests)
            ],
        }

        # Build tar archive with all files
        tar_data = self._build_batch_tar(
            manifest=manifest,
            code=spec.code,
            runner_script=_get_batch_runner_script(),
            batch_dir=batch_dir,
        )

        # Write tar to container (1 semaphore acquisition)
        await self._write_tar(tar_data, timeout_s=spec.timeout_s)

        # Execute batch runner and stream results (1 semaphore acquisition)
        manifest_path = f"{self._config.working_dir}/{batch_dir}/manifest.json"
        runner_path = f"{self._config.working_dir}/{batch_dir}/batch_runner.py"

        # Track timing and received results
        run_start = time.perf_counter()
        received_done = False
        received_test_ids: set[str] = set()
        compile_result: Optional[CompileResult] = None

        try:
            async with self._exec_semaphore:
                proc = await asyncio.create_subprocess_exec(
                    "podman-hpc", "exec",
                    "--workdir", f"{self._config.working_dir}/{batch_dir}",
                    self._container_name,
                    "python", runner_path, manifest_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Stream results line by line
                async for line_bytes in proc.stdout:
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from batch runner: {line}")
                        continue

                    result_type = result.get("type")

                    if result_type == "compile":
                        compile_result = self._parse_batch_compile_result(result)
                        yield compile_result
                        if not compile_result.success:
                            # Compilation failed, we're done
                            break

                    elif result_type == "test":
                        test_id = result.get("id", "unknown")
                        received_test_ids.add(test_id)
                        exec_result = self._parse_batch_test_result(result, run_start)
                        yield exec_result

                    elif result_type == "done":
                        received_done = True
                        break

                    elif result_type == "error":
                        logger.error(f"Batch runner error: {result.get('message')}")

                # Wait for process to complete
                await proc.wait()

        except asyncio.TimeoutError:
            logger.warning(f"Batch execution timed out after {spec.timeout_s}s")

        except Exception as e:
            logger.warning(f"Batch execution stream broke: {e}")

        # Handle missing tests (stream truncated before "done")
        if not received_done and compile_result is None:
            # No compile result received - emit a failure
            compile_result = CompileResult(
                status=CompileStatus.UNKNOWN_ERROR,
                error_message="Batch execution terminated unexpectedly",
                duration_ms=(time.perf_counter() - batch_start) * 1000,
            )
            yield compile_result

        if not received_done and (compile_result is None or compile_result.success):
            # Some tests may not have been run
            for i, test in enumerate(spec.tests):
                test_id = test.id or f"test_{i}"
                if test_id not in received_test_ids:
                    run_ms = (time.perf_counter() - run_start) * 1000
                    yield ExecutionResult(
                        compile_result=compile_result or CompileResult(
                            status=CompileStatus.SUCCESS
                        ),
                        run_status=RunStatus.SANDBOX_ERROR,
                        stdout="",
                        stderr="Batch execution terminated unexpectedly",
                        exit_code=None,
                        run_duration_ms=run_ms,
                        total_duration_ms=run_ms,
                    )

    def _build_batch_tar(
        self,
        manifest: dict,
        code: str,
        runner_script: str,
        batch_dir: str = "_batch",
    ) -> bytes:
        """Build tar archive containing batch execution files.

        Creates a tar with:
        - {batch_dir}/manifest.json: Test configuration
        - {batch_dir}/solution.py: Code under test
        - {batch_dir}/batch_runner.py: Self-contained test runner

        Args:
            manifest: Test configuration dict
            code: Python code to test
            runner_script: Content of batch_runner.py
            batch_dir: Directory name within workspace

        Returns:
            Tar archive bytes
        """
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            # Create directory entry first
            dir_info = tarfile.TarInfo(name=batch_dir)
            dir_info.type = tarfile.DIRTYPE
            dir_info.mode = 0o755
            dir_info.mtime = int(time.time())
            tar.addfile(dir_info)

            # Add manifest.json
            manifest_data = json.dumps(manifest, indent=2).encode("utf-8")
            info = tarfile.TarInfo(name=f"{batch_dir}/manifest.json")
            info.size = len(manifest_data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(manifest_data))

            # Add solution.py
            code_data = code.encode("utf-8")
            info = tarfile.TarInfo(name=f"{batch_dir}/solution.py")
            info.size = len(code_data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(code_data))

            # Add batch_runner.py
            runner_data = runner_script.encode("utf-8")
            info = tarfile.TarInfo(name=f"{batch_dir}/batch_runner.py")
            info.size = len(runner_data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(runner_data))

        buf.seek(0)
        return buf.read()

    async def _write_tar(
        self,
        tar_data: bytes,
        *,
        timeout_s: float = 5.0,
    ) -> None:
        """Write a tar archive directly to the container.

        Similar to _write_file but takes raw tar bytes.
        """
        await asyncio.wait_for(
            self._run_podman(
                "exec", "-i", self._container_name,
                "tar", "-xC", self._config.working_dir,
                check=True,
                capture=True,
                input_data=tar_data,
            ),
            timeout=timeout_s,
        )

    def _parse_batch_compile_result(self, result: dict) -> CompileResult:
        """Parse compile result from batch runner JSON."""
        status_str = result.get("status", "unknown_error")

        if status_str == "success":
            status = CompileStatus.SUCCESS
        elif status_str == "syntax_error":
            status = CompileStatus.SYNTAX_ERROR
        elif status_str == "timeout":
            status = CompileStatus.TIMEOUT
        else:
            status = CompileStatus.UNKNOWN_ERROR

        return CompileResult(
            status=status,
            error_message=result.get("error_message"),
            error_line=result.get("error_line"),
            error_column=result.get("error_column"),
            duration_ms=result.get("duration_ms", 0.0),
        )

    def _parse_batch_test_result(
        self,
        result: dict,
        run_start: float,
    ) -> ExecutionResult:
        """Parse test result from batch runner JSON."""
        status_str = result.get("status", "runtime_error")

        if status_str == "success":
            run_status = RunStatus.SUCCESS
        elif status_str == "runtime_error":
            run_status = RunStatus.RUNTIME_ERROR
        elif status_str == "timeout":
            run_status = RunStatus.TIMEOUT
        elif status_str == "memory_exceeded":
            run_status = RunStatus.MEMORY_EXCEEDED
        elif status_str == "not_run":
            run_status = RunStatus.NOT_RUN
        elif status_str == "killed":
            run_status = RunStatus.KILLED
        else:
            run_status = RunStatus.RUNTIME_ERROR

        duration_ms = result.get("duration_ms", 0.0)
        total_ms = (time.perf_counter() - run_start) * 1000

        return ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=run_status,
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            exit_code=result.get("exit_code"),
            run_duration_ms=duration_ms,
            total_duration_ms=total_ms,
            cache_key=result.get("id", ""),  # Pass test_id for matching in runner
        )


@dataclass
class PodmanResult:
    """Result of a podman-hpc command."""
    returncode: int
    stdout: str
    stderr: str


class PodmanError(Exception):
    """Error from podman-hpc command."""
    pass


class PodmanHPCSandboxPool(BaseSandboxPool[PodmanHPCSandbox]):
    """
    Pool of persistent Podman-HPC containers with LRU caching.

    Manages container lifecycle, checkout/release, and execution caching.
    Designed for HPC environments with Slurm job scheduling.

    Inherits from BaseSandboxPool to use background reset pattern:
      - checkout() returns pre-reset sandboxes instantly
      - release() spawns background reset task
      - shutdown() waits for pending resets before cleanup
    """

    def __init__(
        self,
        n_workers: int = 4,
        image: str = "python:3.11-slim",
        config: Optional[PodmanConfig] = None,
        cache_size: int = 10000,
        auto_replace_failed: bool = True,
        max_consecutive_failures: int = 5,
        max_concurrent_ops: int = 8,
    ):
        """
        Initialize Podman-HPC sandbox pool.

        Args:
            n_workers: Number of sandboxes to create
            image: Podman image (e.g., "python:3.11-slim")
            config: Podman-specific configuration
            cache_size: Maximum entries in execution cache
            auto_replace_failed: If True, create new sandbox when reset fails
            max_consecutive_failures: Maximum consecutive reset failures before raising
                SandboxPoolExhaustedError (circuit breaker threshold)
            max_concurrent_ops: Maximum concurrent operations (resets, executions)
        """
        super().__init__(
            n_workers=n_workers,
            cache_size=cache_size,
            auto_replace_failed=auto_replace_failed,
            max_consecutive_failures=max_consecutive_failures,
            max_concurrent_ops=max_concurrent_ops,
        )
        self._image = image
        self._config = config or PodmanConfig()
        self._exec_semaphore: Optional[asyncio.Semaphore] = None

        # Extract Python version from image name
        self._python_version = self._parse_python_version(image)

    @property
    def python_version(self) -> str:
        """Python version used by sandboxes in this pool."""
        return self._python_version

    # -------------------------------------------------------------------------
    # Abstract method implementations (backend-specific logic)
    # -------------------------------------------------------------------------

    async def _create_sandboxes(self) -> List[PodmanHPCSandbox]:
        """
        Create and start all Podman-HPC container sandboxes.

        Pulls the image (auto-migrates to shared storage on HPC) and creates
        persistent containers in parallel.

        Returns:
            List of started PodmanHPCSandbox instances
        """
        # Create shared exec semaphore (prevents podman deadlock)
        self._exec_semaphore = asyncio.Semaphore(self._max_concurrent_ops)
        logger.info(
            f"Podman exec semaphore initialized: max_concurrent_ops={self._max_concurrent_ops}"
        )

        # Pull image (podman-hpc pull auto-migrates to shared storage)
        logger.info(f"Pulling image {self._image} (may take a moment for HPC migration)...")
        proc = await asyncio.create_subprocess_exec(
            "podman-hpc", "pull", self._image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Create and start sandboxes in parallel
        container_prefix = _get_container_name_prefix()

        async def _create_and_start(i: int) -> PodmanHPCSandbox:
            container_name = f"{container_prefix}-{i}"
            sandbox = PodmanHPCSandbox(
                container_name=container_name,
                image=self._image,
                config=self._config,
                python_version=self._python_version,
                exec_semaphore=self._exec_semaphore,  # Shared across all sandboxes
            )
            await sandbox.start()
            return sandbox

        sandboxes = await asyncio.gather(*[
            _create_and_start(i) for i in range(self._n_workers)
        ])

        logger.info(f"Podman-HPC sandbox pool ready ({self._n_workers} workers)")
        return sandboxes

    async def _stop_sandbox(self, sandbox: PodmanHPCSandbox) -> None:
        """
        Stop and remove a single Podman container.

        Called during shutdown and when replacing a failed sandbox.
        Handles errors gracefully (logs warnings, doesn't raise).

        Args:
            sandbox: The sandbox to stop
        """
        try:
            await sandbox.stop()
        except Exception as e:
            logger.warning(f"Failed to stop Podman container: {e}")

    async def _create_replacement_sandbox(self) -> Optional[PodmanHPCSandbox]:
        """
        Create a single replacement sandbox for a failed one.

        Creates a new container with the same configuration and starts it.

        Returns:
            New PodmanHPCSandbox instance, or None if creation fails
        """
        try:
            container_prefix = _get_container_name_prefix()
            # Use timestamp to ensure unique container name
            container_name = f"{container_prefix}-replacement-{int(time.time())}"

            sandbox = PodmanHPCSandbox(
                container_name=container_name,
                image=self._image,
                config=self._config,
                python_version=self._python_version,
                exec_semaphore=self._exec_semaphore,  # Use shared semaphore
            )
            await sandbox.start()
            logger.info(f"Created replacement Podman sandbox: {container_name}")
            return sandbox
        except Exception as e:
            logger.error(f"Failed to create replacement Podman sandbox: {e}")
            return None

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_python_version(image: str) -> str:
        """Extract Python version from image name."""
        # Common patterns: python:3.11-slim, python:3.11, ghcr.io/.../python:3.11
        match = re.search(r'python:(\d+\.\d+)', image)
        if match:
            return match.group(1)
        return "3.11"  # Default fallback

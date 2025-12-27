"""
Podman-HPC sandbox implementation for code execution on HPC clusters.

Provides:
  - PodmanConfig: Configuration for Podman containers
  - PodmanHPCSandbox: Async Podman container sandbox using subprocess
  - PodmanHPCSandboxPool: Pool of Podman sandboxes with caching

Podman-HPC is a daemonless container runtime wrapper for HPC clusters (e.g., Isambard).
Uses asyncio.create_subprocess_exec instead of docker-py SDK.

**Important**: On some HPC systems (Isambard), podman-hpc's squashfs conversion
breaks the PATH variable. All commands in this module use absolute paths:
  - /bin/sleep, /bin/mkdir, /bin/sh
  - /usr/local/bin/python
  - /usr/bin/pkill
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import re
import shutil
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Union

from .parsing import (
    get_batch_runner_script,
    parse_batch_compile_result,
    parse_batch_test_result,
    parse_syntax_error,
)
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
        workspace_host_dir: Optional[str] = None,
    ):
        self._container_name = container_name
        self._image = image
        self._config = config
        self._python_version = python_version
        self._exec_semaphore = exec_semaphore  # Shared across all sandboxes in pool
        self._workspace_host_dir = workspace_host_dir
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

        # Add bind mount if workspace_host_dir is set
        if self._workspace_host_dir:
            logger.info(
                f"[{self._container_name}] Using bind mount: "
                f"{self._workspace_host_dir} -> {self._config.working_dir}"
            )
            cmd.extend(
                ["-v", f"{self._workspace_host_dir}:{self._config.working_dir}:rw"]
            )

        # Image and command (use full path for HPC compatibility)
        cmd.extend([self._image, "/bin/sleep", "infinity"])

        # Capture stderr to provide useful error messages
        await self._run_podman(*cmd, capture=True)

        # Ensure workspace directory exists (use full path for HPC compatibility)
        # Skip if using bind mount (host directory should already exist)
        if not self._workspace_host_dir:
            await self._run_podman(
                "exec",
                self._container_name,
                "/bin/mkdir",
                "-p",
                self._config.working_dir,
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

        if self._workspace_host_dir:
            # Direct host filesystem cleanup - no podman exec, no semaphore
            logger.debug(
                f"[{self._container_name}] reset() using direct host cleanup..."
            )
            start = time.perf_counter()

            workspace_path = Path(self._workspace_host_dir)
            for item in workspace_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            elapsed = time.perf_counter() - start
            logger.debug(
                f"[{self._container_name}] reset() completed in {elapsed:.3f}s (direct)"
            )
            return

        logger.debug(f"[{self._container_name}] reset() starting podman-hpc exec...")
        start = time.perf_counter()

        await self._run_podman(
            "exec",
            self._container_name,
            "/bin/sh",
            "-c",
            f"rm -rf {self._config.working_dir}/*",
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
                    "exec",
                    self._container_name,
                    "/usr/local/bin/python",
                    "-m",
                    "py_compile",
                    f"{self._config.working_dir}/_check.py",
                    check=False,
                    capture=True,
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
            line, column, clean_msg = parse_syntax_error(error_msg)

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

            exec_cmd.extend(
                [
                    self._container_name,
                    "/usr/local/bin/python",
                    f"{self._config.working_dir}/{exec_filename}",
                ]
            )

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
                    "exec",
                    self._container_name,
                    "/usr/bin/pkill",
                    "-9",
                    "python",
                    check=False,
                    capture=True,
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
        if self._workspace_host_dir:
            # Direct host filesystem write - no podman exec, no semaphore
            path = Path(self._workspace_host_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return

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
                "exec",
                "-i",
                self._container_name,
                "tar",
                "-xC",
                self._config.working_dir,
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
            stdout=stdout_bytes.decode("utf-8", errors="replace")
            if stdout_bytes
            else "",
            stderr=stderr_bytes.decode("utf-8", errors="replace")
            if stderr_bytes
            else "",
        )

        if check and result.returncode != 0:
            raise PodmanError(
                f"podman-hpc {' '.join(args)} failed with exit code {result.returncode}:\n"
                f"{result.stderr}"
            )

        return result

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
            runner_script=get_batch_runner_script(),
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

        # Calculate aggregate timeout accounting for parallelization in batch_runner
        # With N workers: timeout = (ceil(N_tests / workers) × timeout_per_test) + buffer
        num_workers = 16  # Matches batch_runner.py default for HPC
        parallel_batches = math.ceil(len(spec.tests) / num_workers) if spec.tests else 1
        aggregate_timeout = (
            spec.timeout_s * parallel_batches + 60.0
        )  # 60s buffer for HPC

        try:
            async with self._exec_semaphore:
                proc = await asyncio.create_subprocess_exec(
                    "podman-hpc",
                    "exec",
                    "--workdir",
                    f"{self._config.working_dir}/{batch_dir}",
                    self._container_name,
                    "python",
                    runner_path,
                    manifest_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Results collected from streaming to yield after timeout handling
                streamed_results: list = []

                async def _stream_results():
                    """Stream results from batch runner, updating nonlocal state."""
                    nonlocal received_done, compile_result
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
                            compile_result = parse_batch_compile_result(result)
                            streamed_results.append(("compile", compile_result))
                            if not compile_result.success:
                                # Compilation failed, we're done
                                break

                        elif result_type == "test":
                            test_id = result.get("id", "unknown")
                            received_test_ids.add(test_id)
                            exec_result = parse_batch_test_result(result, run_start)
                            streamed_results.append(("test", exec_result))

                        elif result_type == "done":
                            received_done = True
                            break

                        elif result_type == "error":
                            logger.error(f"Batch runner error: {result.get('message')}")

                    # Wait for process to complete
                    await proc.wait()

                try:
                    await asyncio.wait_for(_stream_results(), timeout=aggregate_timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self._container_name}] Batch timed out after {aggregate_timeout:.1f}s "
                        f"({len(received_test_ids)}/{len(spec.tests)} tests received)"
                    )
                    proc.kill()
                    await proc.wait()

                # Yield all collected results
                for result_type, result in streamed_results:
                    yield result

        except asyncio.TimeoutError:
            logger.warning(f"Batch execution timed out after {aggregate_timeout:.1f}s")

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
                        compile_result=compile_result
                        or CompileResult(status=CompileStatus.SUCCESS),
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
        if self._workspace_host_dir:
            # Extract tar directly to host filesystem - no podman exec
            buf = io.BytesIO(tar_data)
            with tarfile.open(fileobj=buf, mode="r") as tar:
                tar.extractall(path=self._workspace_host_dir)
            return

        await asyncio.wait_for(
            self._run_podman(
                "exec",
                "-i",
                self._container_name,
                "tar",
                "-xC",
                self._config.working_dir,
                check=True,
                capture=True,
                input_data=tar_data,
            ),
            timeout=timeout_s,
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
        workspace_base_dir: str = "auto",
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
            workspace_base_dir: Base directory for bind mounts. Options:
                - "auto" (default): Auto-detect; use /local if on HPC, else None
                - explicit path: Use specified directory for bind mounts
                - None: Disable bind mounts, use tar-based I/O
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

        # Resolve workspace_base_dir
        if workspace_base_dir == "auto":
            # Auto-detect: use /local if on HPC, else None
            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            if (
                slurm_job_id and Path("/home/u5ds/joanv.u5ds").exists()
            ):  # TODO [joan]: Remove hardcoding
                self._workspace_base_dir: Optional[str] = (
                    f"/home/u5ds/joanv.u5ds/sandbox/ludic-{slurm_job_id}"
                )
            else:
                self._workspace_base_dir = None
        else:
            self._workspace_base_dir = workspace_base_dir

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
        logger.info(
            f"Pulling image {self._image} (may take a moment for HPC migration)..."
        )
        proc = await asyncio.create_subprocess_exec(
            "podman-hpc",
            "pull",
            self._image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # If using bind mounts, create base directory
        if self._workspace_base_dir:
            Path(self._workspace_base_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Bind mount enabled: {self._workspace_base_dir}")
        else:
            logger.info("Bind mount disabled, using tar-based I/O")

        # Create and start sandboxes in parallel
        container_prefix = _get_container_name_prefix()

        async def _create_and_start(i: int) -> PodmanHPCSandbox:
            container_name = f"{container_prefix}-{i}"

            # Create per-sandbox host directory if using bind mounts
            workspace_host_dir = None
            if self._workspace_base_dir:
                workspace_host_dir = f"{self._workspace_base_dir}/sandbox-{i}"
                Path(workspace_host_dir).mkdir(parents=True, exist_ok=True)

            sandbox = PodmanHPCSandbox(
                container_name=container_name,
                image=self._image,
                config=self._config,
                python_version=self._python_version,
                exec_semaphore=self._exec_semaphore,  # Shared across all sandboxes
                workspace_host_dir=workspace_host_dir,
            )
            await sandbox.start()
            return sandbox

        sandboxes = await asyncio.gather(
            *[_create_and_start(i) for i in range(self._n_workers)]
        )

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

            # Create per-sandbox host directory if using bind mounts
            workspace_host_dir = None
            if self._workspace_base_dir:
                workspace_host_dir = (
                    f"{self._workspace_base_dir}/sandbox-replacement-{int(time.time())}"
                )
                Path(workspace_host_dir).mkdir(parents=True, exist_ok=True)

            sandbox = PodmanHPCSandbox(
                container_name=container_name,
                image=self._image,
                config=self._config,
                python_version=self._python_version,
                exec_semaphore=self._exec_semaphore,  # Use shared semaphore
                workspace_host_dir=workspace_host_dir,
            )
            await sandbox.start()
            logger.info(f"Created replacement Podman sandbox: {container_name}")
            return sandbox
        except Exception as e:
            logger.error(f"Failed to create replacement Podman sandbox: {e}")
            return None

    async def shutdown(self) -> None:
        """
        Shutdown pool and clean up resources.

        Stops all sandboxes and removes workspace directories if using bind mounts.
        """
        # Call parent shutdown to stop sandboxes
        await super().shutdown()

        # Clean up host workspace directories
        if self._workspace_base_dir:
            workspace_path = Path(self._workspace_base_dir)
            if workspace_path.exists():
                try:
                    shutil.rmtree(self._workspace_base_dir, ignore_errors=True)
                    logger.info(
                        f"Cleaned up workspace directory: {self._workspace_base_dir}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to clean up workspace directory: {e}")

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_python_version(image: str) -> str:
        """Extract Python version from image name."""
        # Common patterns: python:3.11-slim, python:3.11, ghcr.io/.../python:3.11
        match = re.search(r"python:(\d+\.\d+)", image)
        if match:
            return match.group(1)
        return "3.11"  # Default fallback

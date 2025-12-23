"""
Docker-based sandbox implementation for code execution.

This module provides:
  - DockerSandboxConfig: Configuration for Docker containers
  - DockerSandbox: Async Docker container sandbox
  - DockerSandboxPool: Pool of Docker sandboxes with caching

Requires: docker>=7.0.0
Install with: pip install 'ludic[code-exec]'
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    import docker
    from docker.models.containers import Container
except ImportError as e:
    raise ImportError(
        "Docker is not installed. Install it with: pip install 'ludic[code-exec]'"
    ) from e

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
    import pkg_resources

    def _get_batch_runner_script() -> str:
        return pkg_resources.resource_string(
            "ludic.envs.code_exec", "batch_runner.py"
        ).decode("utf-8")


@dataclass
class DockerSandboxConfig:
    """Configuration for Docker-based sandboxes."""

    python_version: str = "3.11"
    base_image: Optional[str] = None
    memory_limit: str = "256m"
    cpu_quota: int = 50000  # 50% of one CPU (out of 100000)
    network_disabled: bool = True
    working_dir: str = "/workspace"

    @property
    def image(self) -> str:
        """Get Docker image name (auto-generated or explicit)."""
        if self.base_image:
            return self.base_image
        return f"python:{self.python_version}-slim"


class DockerSandbox:
    """
    Async Docker container sandbox for Python code execution.

    Uses ThreadPoolExecutor to make docker-py calls non-blocking.
    Implements the Sandbox protocol with full async support.
    """

    _memory_limit_warned = False

    def __init__(
        self,
        container: Container,
        config: DockerSandboxConfig,
        executor: ThreadPoolExecutor,
    ):
        self._container = container
        self._config = config
        self._executor = executor

    @property
    def python_version(self) -> str:
        return self._config.python_version

    async def reset(self) -> None:
        """Clear workspace directory."""

        def _reset():
            # Remove all files in workspace
            self._container.exec_run(
                f"sh -c 'rm -rf {self._config.working_dir}/*'",
                workdir=self._config.working_dir,
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, _reset)

    async def compile(
        self,
        code: str,
        *,
        timeout_s: float = 5.0,
    ) -> CompileResult:
        """
        Syntax-check code using py_compile.

        Returns rich error info including line and column numbers.
        """
        start = time.perf_counter()

        def _compile():
            # Write code to temp file
            self._write_file("_check.py", code)

            # Run py_compile
            result = self._container.exec_run(
                "python -m py_compile _check.py",
                workdir=self._config.working_dir,
                demux=True,
            )
            return result

        loop = asyncio.get_event_loop()
        try:
            # Run with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _compile),
                timeout=timeout_s,
            )

            duration_ms = (time.perf_counter() - start) * 1000

            exit_code = result.exit_code
            stdout, stderr = result.output

            if exit_code == 0:
                return CompileResult(
                    status=CompileStatus.SUCCESS,
                    duration_ms=duration_ms,
                )

            # Parse error message
            error_msg = (stderr or b"").decode("utf-8", errors="replace")
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
        """
        Execute code with full resource isolation and rich metadata.

        Compiles first, then executes if compilation succeeds (unless skip_compile=True).
        """
        # Log warning for memory_limit_mb if provided (only once)
        if memory_limit_mb is not None and not DockerSandbox._memory_limit_warned:
            logger.warning(
                "Per-execution memory limits are not supported by docker exec. "
                "Container-level memory limit (%s) is enforced instead.",
                self._config.memory_limit,
            )
            DockerSandbox._memory_limit_warned = True

        # Step 1: Compile
        if skip_compile:
            compile_result = CompileResult(status=CompileStatus.SUCCESS)
        else:
            compile_result = await self.compile(code, timeout_s=timeout_s)

        total_start = time.perf_counter()

        if not compile_result.success:
            # Return early with compilation failure
            total_ms = (time.perf_counter() - total_start) * 1000
            return ExecutionResult(
                compile_result=compile_result,
                run_status=RunStatus.NOT_RUN,
                compile_duration_ms=compile_result.duration_ms,
                total_duration_ms=total_ms,
            )

        # Step 2: Execute
        run_start = time.perf_counter()

        def _execute():
            # Generate unique execution ID to avoid race conditions
            exec_id = uuid.uuid4().hex[:8]
            exec_file = f"_exec_{exec_id}.py"
            input_file = f"input_{exec_id}.txt"

            # Write code to file
            self._write_file(exec_file, code)

            # Write stdin to file if provided
            if stdin:
                self._write_file(input_file, stdin)
                # Build command with stdin redirection
                cmd = f"python {self._config.working_dir}/{exec_file} < {self._config.working_dir}/{input_file}"
            else:
                # Build command without redirection
                cmd = f"python {self._config.working_dir}/{exec_file}"

            # Prepare environment
            environment = env_vars or {}

            # Run with resource limits
            result = self._container.exec_run(
                cmd,
                workdir=self._config.working_dir,
                demux=True,
                environment=environment,
            )

            return result

        loop = asyncio.get_event_loop()

        try:
            # Run with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _execute),
                timeout=timeout_s,
            )

            run_ms = (time.perf_counter() - run_start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000

            exit_code = result.exit_code
            stdout, stderr = result.output

            stdout_str = (stdout or b"").decode("utf-8", errors="replace")
            stderr_str = (stderr or b"").decode("utf-8", errors="replace")

            # Classify run status
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
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code,
                compile_duration_ms=compile_result.duration_ms,
                run_duration_ms=run_ms,
                total_duration_ms=total_ms,
            )

        except asyncio.TimeoutError:
            run_ms = (time.perf_counter() - run_start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000

            # Try to kill the process
            try:
                await loop.run_in_executor(
                    self._executor,
                    lambda: self._container.exec_run("pkill -9 python"),
                )
            except Exception:
                pass  # Best effort cleanup

            return ExecutionResult(
                compile_result=compile_result,
                run_status=RunStatus.TIMEOUT,
                stderr=f"Execution timed out after {timeout_s}s",
                compile_duration_ms=compile_result.duration_ms,
                run_duration_ms=run_ms,
                total_duration_ms=total_ms,
            )

    def _write_file(self, path: str, content: str) -> None:
        """
        Write a file to the container using tarfile.

        Docker API doesn't have a direct "write file" method,
        so we create a tar archive in memory and extract it.
        """
        # Create tar archive in memory
        tar_buffer = io.BytesIO()
        tar = tarfile.open(fileobj=tar_buffer, mode="w")

        # Add file to archive
        file_data = content.encode("utf-8")
        tarinfo = tarfile.TarInfo(name=path)
        tarinfo.size = len(file_data)
        tarinfo.mtime = time.time()
        tar.addfile(tarinfo, io.BytesIO(file_data))
        tar.close()

        # Extract to container
        tar_buffer.seek(0)
        self._container.put_archive(self._config.working_dir, tar_buffer)

    @staticmethod
    def _parse_syntax_error(error_msg: str) -> tuple[Optional[int], Optional[int], str]:
        """
        Parse Python syntax error to extract line, column, and clean message.

        Python syntax errors typically look like:
          File "_check.py", line 3
            def foo(
                   ^
        SyntaxError: invalid syntax

        Or:
          File "_check.py", line 5, column 10
        SyntaxError: invalid syntax

        Returns:
            (line, column, clean_message)
        """
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
        error_type_match = re.search(r'(SyntaxError|IndentationError|TabError):\s*(.+)', error_msg)
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
    # Batch execution (reduces ThreadPoolExecutor calls from O(N) to O(1))
    # -------------------------------------------------------------------------

    async def execute_batch(
        self,
        spec: BatchExecutionSpec,
    ) -> AsyncIterator[Union[CompileResult, ExecutionResult]]:
        """
        Execute all tests in a single batch with streaming results.

        This method reduces the number of ThreadPoolExecutor calls by:
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
        loop = asyncio.get_event_loop()

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

        # Build and write tar archive
        tar_data = self._build_batch_tar(
            manifest=manifest,
            code=spec.code,
            runner_script=_get_batch_runner_script(),
            batch_dir=batch_dir,
        )

        def _write_tar():
            tar_buffer = io.BytesIO(tar_data)
            self._container.put_archive(self._config.working_dir, tar_buffer)

        await loop.run_in_executor(self._executor, _write_tar)

        # Execute batch runner and stream results
        manifest_path = f"{self._config.working_dir}/{batch_dir}/manifest.json"
        runner_path = f"{self._config.working_dir}/{batch_dir}/batch_runner.py"

        run_start = time.perf_counter()
        received_done = False
        received_test_ids: set[str] = set()
        compile_result: Optional[CompileResult] = None

        def _execute():
            result = self._container.exec_run(
                f"python {runner_path} {manifest_path}",
                workdir=f"{self._config.working_dir}/{batch_dir}",
                demux=True,
            )
            return result

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _execute),
                timeout=spec.timeout_s * len(spec.tests) + 10.0,  # Extra buffer
            )

            stdout, stderr = result.output
            stdout_str = (stdout or b"").decode("utf-8", errors="replace")

            # Parse JSONL output
            for line in stdout_str.strip().split("\n"):
                if not line:
                    continue

                try:
                    result_dict = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from batch runner: {line}")
                    continue

                result_type = result_dict.get("type")

                if result_type == "compile":
                    compile_result = self._parse_batch_compile_result(result_dict)
                    yield compile_result
                    if not compile_result.success:
                        break

                elif result_type == "test":
                    test_id = result_dict.get("id", "unknown")
                    received_test_ids.add(test_id)
                    exec_result = self._parse_batch_test_result(result_dict, run_start)
                    yield exec_result

                elif result_type == "done":
                    received_done = True
                    break

                elif result_type == "error":
                    logger.error(f"Batch runner error: {result_dict.get('message')}")

        except asyncio.TimeoutError:
            logger.warning(f"Batch execution timed out")

        except Exception as e:
            logger.warning(f"Batch execution failed: {e}")

        # Handle missing tests
        if not received_done and compile_result is None:
            compile_result = CompileResult(
                status=CompileStatus.UNKNOWN_ERROR,
                error_message="Batch execution terminated unexpectedly",
                duration_ms=(time.perf_counter() - batch_start) * 1000,
            )
            yield compile_result

        if not received_done and (compile_result is None or compile_result.success):
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
        """Build tar archive containing batch execution files."""
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


class DockerSandboxPool(BaseSandboxPool[DockerSandbox]):
    """
    Pool of Docker sandboxes with LRU caching.

    Manages container lifecycle, checkout/release, and execution caching.
    Inherits background reset pattern from BaseSandboxPool.
    """

    def __init__(
        self,
        n_workers: int = 4,
        config: Optional[DockerSandboxConfig] = None,
        cache_size: int = 10000,
        executor_threads: int = 8,
        auto_replace_failed: bool = False,
        max_consecutive_failures: int = 5,
        max_concurrent_resets: int = 8,
    ):
        # Initialize base pool
        super().__init__(
            n_workers=n_workers,
            cache_size=cache_size,
            auto_replace_failed=auto_replace_failed,
            max_consecutive_failures=max_consecutive_failures,
            max_concurrent_resets=max_concurrent_resets,
        )

        # Docker-specific configuration
        self._config = config or DockerSandboxConfig()
        self._executor = ThreadPoolExecutor(max_workers=executor_threads)
        self._docker_client: Optional[docker.DockerClient] = None

    @property
    def python_version(self) -> str:
        return self._config.python_version

    async def _create_sandboxes(self) -> list[DockerSandbox]:
        """
        Create all Docker containers.

        Pulls the image if needed, creates containers with resource limits.
        Called by base class start() method.
        """
        loop = asyncio.get_event_loop()

        def _start():
            # Create Docker client
            client = docker.from_env()

            # Pull image if not present
            try:
                client.images.get(self._config.image)
            except docker.errors.ImageNotFound:
                print(f"Pulling image {self._config.image}...")
                client.images.pull(self._config.image)

            # Define function to create a single container
            def create_container(i: int):
                # Generate container name with PID for uniqueness
                container_name = f"ludic-sandbox-{self._config.python_version}-{os.getpid()}-{i}"

                # Remove existing container if present
                try:
                    old = client.containers.get(container_name)
                    old.remove(force=True)
                except docker.errors.NotFound:
                    pass

                # Create container with resource limits
                container = client.containers.create(
                    image=self._config.image,
                    name=container_name,
                    detach=True,
                    command="sleep infinity",  # Keep container alive
                    mem_limit=self._config.memory_limit,
                    cpu_quota=self._config.cpu_quota,
                    cpu_period=100000,  # Standard 100ms period
                    network_disabled=self._config.network_disabled,
                    working_dir=self._config.working_dir,
                    auto_remove=False,  # We'll manage cleanup
                )

                # Start container
                container.start()

                # Create sandbox wrapper
                return DockerSandbox(
                    container=container,
                    config=self._config,
                    executor=self._executor,
                )

            # Parallelize container creation
            with ThreadPoolExecutor(max_workers=self._n_workers) as pool:
                sandboxes = list(pool.map(create_container, range(self._n_workers)))

            return client, sandboxes

        # Run container creation in executor
        self._docker_client, sandboxes = await loop.run_in_executor(
            self._executor, _start
        )

        return sandboxes

    async def _stop_sandbox(self, sandbox: DockerSandbox) -> None:
        """
        Stop and remove a single Docker container.

        Called during shutdown and when replacing a failed sandbox.
        Errors are logged but not raised.
        """
        loop = asyncio.get_event_loop()

        def _stop():
            try:
                sandbox._container.stop(timeout=2)
                sandbox._container.remove(force=True)
            except Exception as e:
                print(f"Warning: Failed to remove container: {e}")

        await loop.run_in_executor(self._executor, _stop)

    async def _create_replacement_sandbox(self) -> Optional[DockerSandbox]:
        """
        Create a single replacement Docker container.

        Called when a sandbox fails to reset and auto_replace_failed is True.
        Returns None if container creation fails.
        """
        loop = asyncio.get_event_loop()

        def _create():
            if self._docker_client is None:
                return None

            try:
                # Generate unique container name
                import random
                i = random.randint(10000, 99999)
                container_name = f"ludic-sandbox-{self._config.python_version}-{os.getpid()}-{i}"

                # Remove existing container if present
                try:
                    old = self._docker_client.containers.get(container_name)
                    old.remove(force=True)
                except docker.errors.NotFound:
                    pass

                # Create container with resource limits
                container = self._docker_client.containers.create(
                    image=self._config.image,
                    name=container_name,
                    detach=True,
                    command="sleep infinity",
                    mem_limit=self._config.memory_limit,
                    cpu_quota=self._config.cpu_quota,
                    cpu_period=100000,
                    network_disabled=self._config.network_disabled,
                    working_dir=self._config.working_dir,
                    auto_remove=False,
                )

                # Start container
                container.start()

                # Create sandbox wrapper
                return DockerSandbox(
                    container=container,
                    config=self._config,
                    executor=self._executor,
                )
            except Exception:
                return None

        return await loop.run_in_executor(self._executor, _create)

    async def shutdown(self) -> None:
        """
        Tear down all containers and release resources.

        Waits for pending resets, stops containers, closes Docker client,
        and shuts down executor.
        """
        # Base shutdown handles pending resets and calls _stop_sandbox
        await super().shutdown()

        # Docker-specific cleanup
        loop = asyncio.get_event_loop()

        def _close_client():
            if self._docker_client:
                self._docker_client.close()

        await loop.run_in_executor(self._executor, _close_client)

        # Shutdown executor
        self._executor.shutdown(wait=True)

# src/ludic/sandbox/pool.py
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, List
import aiohttp

from ludic.sandbox.protocol import ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    base_url: str
    timeout_execute_s: float = 60.0
    timeout_reset_s: float = 10.0


class SandboxHandle:
    """
    A connection to a single sandbox container.
    Managed by SandboxPool — do not instantiate directly.
    """

    def __init__(self, config: SandboxConfig, pool_idx: int):
        self.config = config
        self.pool_idx = pool_idx
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def execute(
        self,
        code: str,
        tests: str,
        language: str = "python",
        timeout_s: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute code against tests in the sandbox."""
        session = await self._ensure_session()
        timeout = aiohttp.ClientTimeout(
            total=timeout_s or self.config.timeout_execute_s
        )

        payload = {
            "code": code,
            "tests": tests,
            "language": language,
        }

        try:
            async with session.post(
                f"{self.config.base_url}/execute",
                json=payload,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    return ExecutionResult(
                        status=ExecutionStatus.RUNTIME_ERROR,
                        tests_passed=0,
                        tests_total=0,
                        stderr=f"Sandbox returned HTTP {resp.status}",
                    )
                data = await resp.json()
                return ExecutionResult(
                    status=ExecutionStatus(data.get("status", "runtime_error")),
                    tests_passed=data.get("tests_passed", 0),
                    tests_total=data.get("tests_total", 0),
                    test_results=data.get("test_results", []),
                    compile_output=data.get("compile_output"),
                    stderr=data.get("stderr"),
                    execution_time_ms=data.get("execution_time_ms", 0.0),
                )
        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                tests_passed=0,
                tests_total=0,
                stderr=f"Execution timed out after {timeout_s}s",
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                tests_passed=0,
                tests_total=0,
                stderr=str(e),
            )

    async def reset(self) -> bool:
        """Reset sandbox state (clear temp files, kill lingering processes)."""
        session = await self._ensure_session()
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_reset_s)

        try:
            async with session.post(
                f"{self.config.base_url}/reset",
                timeout=timeout,
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"Sandbox {self.pool_idx} reset failed: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if sandbox is responsive."""
        session = await self._ensure_session()
        try:
            async with session.get(
                f"{self.config.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class SandboxPool:
    """
    Pool of sandbox connections with async acquire/release.

    Usage:
        pool = SandboxPool(base_port=8000, num_sandboxes=8)
        await pool.start()

        # In env:
        handle = await pool.acquire()
        try:
            result = await handle.execute(code, tests)
        finally:
            await pool.release(handle)

        # Shutdown:
        await pool.shutdown()
    """

    def __init__(
        self,
        host: str = "localhost",
        base_port: int = 8000,
        num_sandboxes: int = 8,
        timeout_execute_s: float = 60.0,
        timeout_reset_s: float = 10.0,
    ):
        self.host = host
        self.base_port = base_port
        self.num_sandboxes = num_sandboxes

        self._handles: List[SandboxHandle] = []
        self._available: asyncio.Queue[SandboxHandle] = asyncio.Queue()
        self._started = False

        self._timeout_execute_s = timeout_execute_s
        self._timeout_reset_s = timeout_reset_s

    async def start(self, health_check: bool = True):
        """Initialize connections to all sandboxes."""
        if self._started:
            return

        for i in range(self.num_sandboxes):
            config = SandboxConfig(
                base_url=f"http://{self.host}:{self.base_port + i}",
                timeout_execute_s=self._timeout_execute_s,
                timeout_reset_s=self._timeout_reset_s,
            )
            handle = SandboxHandle(config, pool_idx=i)

            if health_check:
                healthy = await handle.health_check()
                if not healthy:
                    logger.warning(f"Sandbox {i} at {config.base_url} is not healthy")
                    # Continue anyway - might come up later

            self._handles.append(handle)
            await self._available.put(handle)

        self._started = True
        logger.info(f"SandboxPool started with {self.num_sandboxes} sandboxes")

    async def acquire(self, timeout_s: Optional[float] = None) -> SandboxHandle:
        """
        Acquire a sandbox handle from the pool.
        Blocks if none available.
        """
        if not self._started:
            raise RuntimeError(
                "SandboxPool not started. Call await pool.start() first."
            )

        if timeout_s is not None:
            return await asyncio.wait_for(self._available.get(), timeout=timeout_s)
        return await self._available.get()

    async def release(self, handle: SandboxHandle):
        """Release a sandbox handle back to the pool."""
        # Reset the sandbox state before returning to pool
        await handle.reset()
        await self._available.put(handle)

    @property
    def available_count(self) -> int:
        return self._available.qsize()

    async def shutdown(self):
        """Close all sandbox connections."""
        for handle in self._handles:
            await handle.close()
        self._handles.clear()
        self._started = False

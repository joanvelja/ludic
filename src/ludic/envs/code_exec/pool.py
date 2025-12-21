"""
Base sandbox pool with background reset pattern.

Provides shared pool management logic for Docker, Podman, and other backends.
The background reset pattern ensures that sandbox cleanup happens off the
critical path, maximizing throughput for rollout generation.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, Set, TypeVar

from .cache import LRUCache
from .sandbox import Sandbox
from .types import BatchTestResult, SandboxPoolExhaustedError

logger = logging.getLogger(__name__)

# Type variable for sandbox implementations
S = TypeVar("S", bound=Sandbox)


class BaseSandboxPool(ABC, Generic[S]):
    """
    Abstract base class for sandbox pools with background reset.

    Implements the common pool logic:
      - Queue-based checkout/release
      - Background reset on release (off critical path)
      - Pending task tracking for clean shutdown
      - LRU cache for execution results
      - Error handling for failed resets

    Subclasses must implement:
      - _create_sandboxes(): Create backend-specific sandbox instances
      - _stop_sandbox(sandbox): Stop a single sandbox (for shutdown)
      - python_version property: Return the Python version

    Background Reset Pattern:
      When a sandbox is released, instead of blocking the caller with a reset,
      we spawn a background task that:
        1. Resets the sandbox (cleans filesystem, kills processes)
        2. Returns the sandbox to the available queue
        3. On failure, discards the sandbox and optionally creates a replacement

      This means checkout() gets an already-clean sandbox instantly, and the
      reset latency is completely hidden from the rollout generation loop.
    """

    def __init__(
        self,
        n_workers: int = 4,
        cache_size: int = 10000,
        auto_replace_failed: bool = True,
        max_consecutive_failures: int = 5,
    ):
        """
        Initialize the pool.

        Args:
            n_workers: Number of sandboxes to create
            cache_size: Maximum entries in the execution cache
            auto_replace_failed: If True, create new sandbox when reset fails
            max_consecutive_failures: Maximum consecutive reset failures before raising
                SandboxPoolExhaustedError (circuit breaker threshold)
        """
        self._n_workers = n_workers
        self._cache = LRUCache(max_size=cache_size)
        self._auto_replace_failed = auto_replace_failed
        self._max_consecutive_failures = max_consecutive_failures

        self._sandboxes: List[S] = []
        self._queue: Optional[asyncio.Queue[S]] = None
        self._pending_resets: Set[asyncio.Task] = set()
        self._started = False
        self._shutting_down = False
        self._consecutive_failures = 0

    # -------------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def python_version(self) -> str:
        """Python version used by sandboxes in this pool."""
        ...

    @abstractmethod
    async def _create_sandboxes(self) -> List[S]:
        """
        Create all sandbox instances.

        Called by start(). Should create n_workers sandboxes, start them,
        and return the list. This is where backend-specific logic lives
        (Docker container creation, Podman-HPC setup, etc.).

        Returns:
            List of started sandbox instances
        """
        ...

    @abstractmethod
    async def _stop_sandbox(self, sandbox: S) -> None:
        """
        Stop and cleanup a single sandbox.

        Called during shutdown and when replacing a failed sandbox.
        Should handle errors gracefully (log warnings, don't raise).

        Args:
            sandbox: The sandbox to stop
        """
        ...

    async def _create_replacement_sandbox(self) -> Optional[S]:
        """
        Create a single replacement sandbox.

        Called when a sandbox fails to reset and auto_replace_failed is True.
        Default implementation returns None (no replacement). Override in
        subclass if dynamic sandbox creation is supported.

        Returns:
            New sandbox instance, or None if replacement not supported
        """
        return None

    # -------------------------------------------------------------------------
    # Pool lifecycle
    # -------------------------------------------------------------------------

    @property
    def available(self) -> int:
        """Number of sandboxes currently available for checkout."""
        if self._queue is None:
            return 0
        return self._queue.qsize()

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Cache statistics (hits, misses, size, max_size)."""
        return self._cache.stats

    @property
    def pending_resets(self) -> int:
        """Number of background reset tasks currently running."""
        return len(self._pending_resets)

    async def start(self) -> None:
        """
        Initialize the pool.

        Creates all sandboxes and makes them available for checkout.
        Idempotent - calling multiple times has no effect.
        """
        if self._started:
            return

        # Create sandboxes (backend-specific)
        self._sandboxes = await self._create_sandboxes()

        # Create queue and populate with all sandboxes
        self._queue = asyncio.Queue()
        for sandbox in self._sandboxes:
            await self._queue.put(sandbox)

        self._started = True

    async def shutdown(self) -> None:
        """
        Tear down all sandboxes and release resources.

        Waits for all pending reset tasks to complete before stopping
        sandboxes, ensuring clean shutdown without orphaned tasks.
        """
        if not self._started:
            return

        self._shutting_down = True

        # Wait for all pending reset tasks to complete
        if self._pending_resets:
            logger.debug(f"Waiting for {len(self._pending_resets)} pending resets...")
            await asyncio.gather(*self._pending_resets, return_exceptions=True)

        # Stop all sandboxes
        for sandbox in self._sandboxes:
            await self._stop_sandbox(sandbox)

        self._sandboxes.clear()
        self._started = False
        self._queue = None
        self._shutting_down = False

    async def drain_pending_resets(self, timeout_s: float = 60.0) -> int:
        """
        Wait for all pending reset tasks to complete.

        Call this before switching between high-concurrency phases
        (e.g., before eval after training step) to ensure all sandboxes
        are available in the queue.

        Args:
            timeout_s: Maximum time to wait for resets to complete

        Returns:
            Number of resets that completed
        """
        if not self._pending_resets:
            return 0

        count = len(self._pending_resets)
        logger.info(f"Draining {count} pending sandbox resets...")

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._pending_resets, return_exceptions=True),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout draining resets after {timeout_s}s. "
                f"Remaining: {len(self._pending_resets)}"
            )

        logger.info(f"Drain complete. Pool available: {self.available}/{self._n_workers}")
        return count

    # -------------------------------------------------------------------------
    # Checkout / Release with background reset
    # -------------------------------------------------------------------------

    async def checkout(self, timeout_s: float = 30.0) -> Sandbox:
        """
        Get exclusive access to a sandbox.

        The returned sandbox is guaranteed to be in a clean state (reset
        was performed in the background after the previous release).

        If the queue is empty but resets are pending, waits for resets
        to complete before timing out. This makes the pool self-healing
        across phase transitions (e.g., eval → training).

        Args:
            timeout_s: Maximum time to wait for a sandbox

        Returns:
            Exclusive Sandbox handle

        Raises:
            RuntimeError: If pool not started
            TimeoutError: If no sandbox available within timeout
        """
        if not self._started or self._queue is None:
            raise RuntimeError("Pool not started. Call start() first.")

        import time

        start_time = time.monotonic()
        deadline = start_time + timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"No sandbox available after {timeout_s}s. "
                    f"Pool size: {self._n_workers}, available: {self.available}, "
                    f"pending resets: {self.pending_resets}"
                )

            # If queue is empty but resets are pending, wait for them first
            if self._queue.empty() and self._pending_resets:
                logger.info(
                    f"Queue empty with {len(self._pending_resets)} pending resets, "
                    f"waiting for resets to complete..."
                )
                await self.drain_pending_resets(timeout_s=remaining)

            try:
                sandbox = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=min(remaining, 5.0),  # Short timeout to recheck
                )
                wait_time = time.monotonic() - start_time
                if wait_time > 1.0:
                    logger.info(
                        f"Sandbox checkout took {wait_time:.2f}s "
                        f"(waited for pending resets)"
                    )
                return sandbox
            except asyncio.TimeoutError:
                # Retry - more resets may have completed
                continue

    async def release(self, sandbox: Sandbox) -> None:
        """
        Return a sandbox to the pool.

        The sandbox is reset in a background task, then returned to the
        available queue. This makes release() return immediately without
        blocking the caller.

        Args:
            sandbox: The sandbox to release (must have been obtained via checkout)

        Raises:
            RuntimeError: If pool not started
        """
        if not self._started or self._queue is None:
            raise RuntimeError("Pool not started")

        if self._shutting_down:
            # During shutdown, don't spawn new tasks
            return

        # Spawn background reset task
        task = asyncio.create_task(
            self._background_reset(sandbox),  # type: ignore
            name=f"sandbox-reset-{id(sandbox)}",
        )
        self._pending_resets.add(task)
        task.add_done_callback(self._pending_resets.discard)

    async def _background_reset(self, sandbox: S) -> None:
        """
        Reset sandbox and return to queue (runs in background).

        On success, the sandbox is returned to the available queue.
        On failure, the sandbox is discarded and optionally replaced.
        """
        try:
            await sandbox.reset()
            # Only return to queue if we're not shutting down
            if self._queue is not None and not self._shutting_down:
                await self._queue.put(sandbox)
        except Exception as e:
            await self._handle_reset_failure(sandbox, e)

    async def _handle_reset_failure(self, sandbox: S, error: Exception) -> None:
        """
        Handle a sandbox that failed to reset.

        Logs the error, removes the sandbox from the pool, and optionally
        creates a replacement. Implements circuit breaker pattern to detect
        systemic failures.

        Raises:
            SandboxPoolExhaustedError: If consecutive failures exceed threshold
        """
        # Increment failure counter
        self._consecutive_failures += 1

        logger.warning(
            f"Sandbox reset failed: {error}. Discarding sandbox. "
            f"Consecutive failures: {self._consecutive_failures}/{self._max_consecutive_failures}"
        )

        # Check circuit breaker threshold
        if self._consecutive_failures >= self._max_consecutive_failures:
            logger.error(
                f"Circuit breaker triggered: {self._consecutive_failures} consecutive "
                f"sandbox reset failures. Pool is exhausted."
            )
            raise SandboxPoolExhaustedError(
                f"Sandbox pool exhausted after {self._consecutive_failures} consecutive "
                f"reset failures. This indicates a systemic issue requiring operator intervention."
            )

        # Remove from tracked sandboxes
        if sandbox in self._sandboxes:
            self._sandboxes.remove(sandbox)

        # Try to stop the failed sandbox
        try:
            await self._stop_sandbox(sandbox)
        except Exception as stop_error:
            logger.warning(f"Failed to stop broken sandbox: {stop_error}")

        # Optionally create replacement
        if self._auto_replace_failed and not self._shutting_down:
            try:
                replacement = await self._create_replacement_sandbox()
                if replacement is not None:
                    self._sandboxes.append(replacement)
                    if self._queue is not None:
                        await self._queue.put(replacement)
                    # Reset failure counter on successful replacement
                    self._consecutive_failures = 0
                    logger.info(
                        "Created replacement sandbox after reset failure. "
                        "Consecutive failure counter reset."
                    )
            except Exception as create_error:
                logger.warning(f"Failed to create replacement sandbox: {create_error}")

    # -------------------------------------------------------------------------
    # Cache interface
    # -------------------------------------------------------------------------

    def get_cached(
        self,
        code_hash: str,
        tests_hash: str,
    ) -> Optional[BatchTestResult]:
        """Check cache for result (sync, thread-safe)."""
        return self._cache.get(code_hash, tests_hash)

    def put_cached(
        self,
        code_hash: str,
        tests_hash: str,
        result: BatchTestResult,
    ) -> None:
        """Store result in cache (sync, thread-safe)."""
        self._cache.put(code_hash, tests_hash, result)

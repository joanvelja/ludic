"""
Sandbox protocols for isolated code execution.

These protocols define the contract for sandbox implementations.
The actual implementations (Docker, subprocess, etc.) live in separate modules.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

from .types import BatchTestResult, CompileResult, ExecutionResult


@runtime_checkable
class Sandbox(Protocol):
    """
    Async handle to a single isolated execution environment.

    Invariants:
      - A sandbox is exclusive to one env instance at a time
      - reset() clears all state from previous executions
      - All operations are async to avoid blocking the event loop

    Lifecycle:
      1. Obtained via SandboxPool.checkout()
      2. reset() called to ensure clean state
      3. compile() and/or execute() called as needed
      4. Returned via SandboxPool.release()

    Implementations should ensure:
      - Network isolation (no external access)
      - Resource limits (CPU, memory)
      - Timeout enforcement
      - Filesystem isolation between uses
    """

    @property
    def python_version(self) -> str:
        """Python version in this sandbox (e.g., '3.11')."""
        ...

    async def reset(self) -> None:
        """
        Clear filesystem, kill processes, restore to clean state.

        Must be called before first use and is automatically called
        by SandboxPool.release().
        """
        ...

    async def compile(
        self,
        code: str,
        *,
        timeout_s: float = 5.0,
    ) -> CompileResult:
        """
        Syntax-check / compile code without executing.

        For Python: runs py_compile or ast.parse to catch syntax errors.
        For compiled languages: runs the compiler.

        Args:
            code: Source code to compile/check
            timeout_s: Maximum time for compilation

        Returns:
            CompileResult with status and error details if failed
        """
        ...

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
        Execute code and return rich results.

        Implicitly compiles first if not already compiled (unless skip_compile=True).
        The compile result is included in the returned ExecutionResult.

        Args:
            code: Source code to execute
            stdin: Input to feed to the process via stdin
            skip_compile: If True, skip compilation step (assumes code already compiled)
            timeout_s: Maximum execution time (excluding compilation)
            memory_limit_mb: Memory limit override (None uses sandbox default)
            env_vars: Additional environment variables

        Returns:
            ExecutionResult with compile status, output, timing, etc.
        """
        ...


@runtime_checkable
class SandboxPool(Protocol):
    """
    Async pool of reusable sandboxes with caching.

    The pool manages:
      1. Sandbox lifecycle (start/stop containers, processes, etc.)
      2. Checkout/release of exclusive sandbox handles
      3. Execution cache (code+tests -> result)

    Lifecycle:
      1. start() - Initialize pool (start containers, etc.)
      2. checkout() - Get exclusive sandbox access
      3. release() - Return sandbox to pool
      4. shutdown() - Tear down all sandboxes

    The pool should be started once at application startup and shared
    across all CodeExecEnv instances via factory closure injection.

    Caching:
      The pool maintains an LRU cache keyed by (code_hash, tests_hash).
      This avoids redundant execution when the same code is submitted
      for the same tests (common in GRPO where multiple generations
      are evaluated against the same problem).
    """

    @property
    def python_version(self) -> str:
        """Python version used by sandboxes in this pool."""
        ...

    @property
    def available(self) -> int:
        """Number of sandboxes currently available for checkout."""
        ...

    @property
    def cache_stats(self) -> Dict[str, int]:
        """
        Cache statistics.

        Returns dict with keys:
          - hits: number of cache hits
          - misses: number of cache misses
          - size: current cache size
          - max_size: maximum cache size
        """
        ...

    async def start(self) -> None:
        """
        Initialize the pool.

        This starts all sandboxes (containers, processes, etc.).
        Should be called once before any checkout() calls.
        Idempotent - calling multiple times has no effect.
        """
        ...

    async def checkout(self, timeout_s: float = 30.0) -> Sandbox:
        """
        Get exclusive access to a sandbox.

        Blocks until a sandbox is available or timeout is reached.
        The returned sandbox is guaranteed to be in a clean state.

        Args:
            timeout_s: Maximum time to wait for a sandbox

        Returns:
            Exclusive Sandbox handle

        Raises:
            TimeoutError: If no sandbox available within timeout
        """
        ...

    async def release(self, sandbox: Sandbox) -> None:
        """
        Return a sandbox to the pool.

        The sandbox is automatically reset before being made available
        to other callers.

        Args:
            sandbox: The sandbox to release (must have been obtained via checkout)
        """
        ...

    async def shutdown(self) -> None:
        """
        Tear down all sandboxes and release resources.

        After shutdown(), the pool cannot be used again without calling start().
        """
        ...

    # ----- Cache interface -----

    def get_cached(
        self,
        code_hash: str,
        tests_hash: str,
    ) -> Optional[BatchTestResult]:
        """
        Check if we have a cached result for this code+tests pair.

        This is a synchronous method for use from env_step().
        Thread-safe.

        Args:
            code_hash: Hash of the submitted code
            tests_hash: Hash of the test cases

        Returns:
            Cached BatchTestResult if found, None otherwise
        """
        ...

    def put_cached(
        self,
        code_hash: str,
        tests_hash: str,
        result: BatchTestResult,
    ) -> None:
        """
        Cache a result for future lookups.

        This is a synchronous method for use from env_step().
        Thread-safe. Uses LRU eviction when cache is full.

        Args:
            code_hash: Hash of the submitted code
            tests_hash: Hash of the test cases
            result: The BatchTestResult to cache
        """
        ...

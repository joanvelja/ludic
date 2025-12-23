"""
Code runners for executing code against test cases.

This module defines the CodeRunner protocol and concrete implementations
for different test execution strategies (stdin/stdout, function calls, etc.).

The runner is responsible for:
  1. Orchestrating compilation and execution via a Sandbox
  2. Running code against multiple TestCases
  3. Using an OutputVerifier to compare results
  4. Building rich TestResult and BatchTestResult objects
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import List, Optional, Protocol, Set, runtime_checkable

from .adapters.base import OutputVerifier
from .sandbox import Sandbox
from .types import (
    BatchExecutionSpec,
    BatchTestResult,
    CompileResult,
    CompileStatus,
    ExecutionResult,
    RunStatus,
    TestCase,
    TestResult,
)

logger = logging.getLogger(__name__)


def compute_hash(content: str) -> str:
    """
    Compute SHA256 hash, return first 16 hex chars.

    This is used for cache keys to uniquely identify code and test sets.
    16 hex chars = 64 bits, which gives collision probability < 1e-10
    for reasonable dataset sizes.

    Args:
        content: String to hash

    Returns:
        First 16 characters of SHA256 hex digest
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def hash_tests(tests: List[TestCase]) -> str:
    """
    Compute stable hash of test cases for caching.

    Creates a deterministic hash by converting test inputs and expected
    outputs to a canonical JSON representation with sorted keys, then hashing.

    Args:
        tests: List of test cases to hash

    Returns:
        16-character hash string
    """
    # Use JSON with sorted keys for deterministic serialization
    content = json.dumps(
        [(t.input, t.expected) for t in tests],
        sort_keys=True,
        default=str,  # Handle non-JSON-serializable types
    )
    return compute_hash(content)


@runtime_checkable
class CodeRunner(Protocol):
    """
    Protocol for running code against test cases.

    A runner orchestrates the interaction between a Sandbox and test cases,
    using an OutputVerifier to determine if each test passes. It handles
    compilation, execution, error recovery, and early stopping.

    Implementations should be stateless and reusable across multiple
    test runs. All state is passed explicitly via arguments.
    """

    async def run_tests(
        self,
        sandbox: Sandbox,
        code: str,
        tests: List[TestCase],
        *,
        verifier: OutputVerifier,
        stop_on_first_failure: bool = False,
        compile_first: bool = True,
    ) -> BatchTestResult:
        """
        Run code against all test cases and return aggregated results.

        Args:
            sandbox: Sandbox to execute code in (must be checked out)
            code: Source code to test
            tests: List of test cases to run
            verifier: Verifier to compare actual vs expected output
            stop_on_first_failure: If True, skip remaining tests after first failure
            compile_first: If True, compile once before running tests

        Returns:
            BatchTestResult with individual test results and metadata
        """
        ...


class StdinStdoutRunner:
    """
    Runner for APPS-style stdin/stdout testing.

    This runner executes code that reads from stdin and writes to stdout,
    comparing the output against expected values. This is the standard
    format for competitive programming problems (Codeforces, APPS, etc.).

    Each test case's `input` field is passed as stdin, and the `expected`
    field is compared against stdout using the provided verifier.

    Design notes:
      - Default timeout is 5.0s for efficiency (per user specification)
      - Compilation is checked first by default to get early failure signal
      - All operations are async to avoid blocking the event loop
      - Rich error details in TestResult.comparison_details
    """

    def __init__(
        self,
        default_timeout_s: float = 5.0,
        memory_limit_mb: Optional[int] = 256,
        use_batch_execution: bool = True,
    ) -> None:
        """
        Initialize the runner with default resource limits.

        Args:
            default_timeout_s: Default execution timeout per test (seconds).
                              Tests can override via metadata["timeout_s"].
            memory_limit_mb: Memory limit for execution (None = no limit)
            use_batch_execution: If True and sandbox supports it, use batched
                                execution to reduce semaphore acquisitions.
        """
        self._default_timeout_s = default_timeout_s
        self._memory_limit_mb = memory_limit_mb
        self._use_batch_execution = use_batch_execution

    async def run_tests(
        self,
        sandbox: Sandbox,
        code: str,
        tests: List[TestCase],
        *,
        verifier: OutputVerifier,
        stop_on_first_failure: bool = False,
        compile_first: bool = True,
    ) -> BatchTestResult:
        """
        Run stdin/stdout tests against code.

        Implementation steps:
          1. Compute code_hash and tests_hash for caching
          2. If compile_first=True, compile code and fail fast if it fails
          3. For each test:
             - Execute with test.input as stdin
             - Compare stdout against test.expected using verifier
             - Build TestResult with full metadata
          4. If stop_on_first_failure=True, mark remaining tests NOT_RUN
          5. Return BatchTestResult

        Args:
            sandbox: Sandbox to execute code in (must be checked out)
            code: Source code to test
            tests: List of test cases (input/expected are stdin/stdout strings)
            verifier: Verifier to compare stdout vs expected
            stop_on_first_failure: If True, skip remaining tests after first failure
            compile_first: If True, compile once before running tests

        Returns:
            BatchTestResult with results for each test
        """
        import time

        run_start = time.perf_counter()

        # Compute hashes for caching
        code_hash = compute_hash(code)
        tests_hash_val = hash_tests(tests)

        # Use batch execution if enabled and sandbox supports it
        has_batch = hasattr(sandbox, "execute_batch")
        logger.debug(
            f"run_tests: use_batch={self._use_batch_execution}, "
            f"has_execute_batch={has_batch}, num_tests={len(tests)}"
        )

        if self._use_batch_execution and has_batch:
            result = await self._run_tests_batched(
                sandbox=sandbox,
                code=code,
                tests=tests,
                verifier=verifier,
                stop_on_first_failure=stop_on_first_failure,
                compile_first=compile_first,
                code_hash=code_hash,
                tests_hash=tests_hash_val,
            )
            elapsed_ms = (time.perf_counter() - run_start) * 1000
            logger.debug(
                f"Batch execution completed: {len(tests)} tests in {elapsed_ms:.1f}ms, "
                f"passed={result.passed_count}/{result.total_count}"
            )
            return result

        # Non-batch execution
        # Step 1: Compile first if requested
        compile_result: Optional[CompileResult] = None
        if compile_first:
            compile_result = await sandbox.compile(
                code,
                timeout_s=self._default_timeout_s,
            )

            # If compilation failed, all tests fail without execution
            if not compile_result.success:
                return self._create_all_failed_batch(
                    tests=tests,
                    code_hash=code_hash,
                    tests_hash=tests_hash_val,
                    compile_result=compile_result,
                    reason="compilation_failed",
                )

        # Step 2: Run tests (in parallel when possible)
        if stop_on_first_failure:
            # Sequential execution with early stopping
            results: List[TestResult] = []
            for test_case in tests:
                # Get timeout for this test (allow per-test override)
                timeout_s = test_case.metadata.get("timeout_s", self._default_timeout_s)
                memory_limit = test_case.metadata.get(
                    "memory_limit_mb", self._memory_limit_mb
                )

                # Execute the test
                test_result = await self._run_single_test(
                    sandbox=sandbox,
                    code=code,
                    test_case=test_case,
                    verifier=verifier,
                    timeout_s=timeout_s,
                    memory_limit_mb=memory_limit,
                    skip_compile=compile_first,  # Skip if we already compiled
                )

                results.append(test_result)

                # Stop on first failure
                if not test_result.passed:
                    # Mark remaining tests as NOT_RUN
                    for remaining_test in tests[len(results) :]:
                        not_run_result = self._create_not_run_result(
                            test_case=remaining_test,
                            code_hash=code_hash,
                        )
                        results.append(not_run_result)
                    break
        else:
            # Parallel execution with asyncio.gather
            async def run_test_with_metadata(test_case: TestCase) -> TestResult:
                timeout_s = test_case.metadata.get("timeout_s", self._default_timeout_s)
                memory_limit = test_case.metadata.get(
                    "memory_limit_mb", self._memory_limit_mb
                )
                return await self._run_single_test(
                    sandbox=sandbox,
                    code=code,
                    test_case=test_case,
                    verifier=verifier,
                    timeout_s=timeout_s,
                    memory_limit_mb=memory_limit,
                    skip_compile=compile_first,  # Skip if we already compiled
                )

            # Run all tests in parallel
            results = await asyncio.gather(
                *[run_test_with_metadata(test) for test in tests]
            )

        return BatchTestResult(
            results=list(results),
            code_hash=code_hash,
            tests_hash=tests_hash_val,
        )

    async def _run_single_test(
        self,
        sandbox: Sandbox,
        code: str,
        test_case: TestCase,
        verifier: OutputVerifier,
        timeout_s: float,
        memory_limit_mb: Optional[int],
        skip_compile: bool = False,
    ) -> TestResult:
        """
        Run a single test case.

        Args:
            sandbox: Sandbox to execute in
            code: Source code
            test_case: Test to run
            verifier: Output verifier
            timeout_s: Execution timeout
            memory_limit_mb: Memory limit
            skip_compile: If True, skip compilation (assumes already compiled)

        Returns:
            TestResult for this test
        """
        # Execute code with test input
        execution = await sandbox.execute(
            code=code,
            stdin=str(test_case.input),  # Ensure input is string
            skip_compile=skip_compile,
            timeout_s=timeout_s,
            memory_limit_mb=memory_limit_mb,
        )

        # If execution failed (didn't compile or runtime error), test fails
        if not execution.succeeded:
            return TestResult(
                test_case=test_case,
                passed=False,
                actual=execution.stdout,
                execution=execution,
                comparison_details=self._get_execution_failure_details(execution),
            )

        # Execution succeeded, compare output
        actual_output = execution.stdout
        expected_output = str(test_case.expected)

        passed, comparison_details = verifier.verify(actual_output, expected_output)

        return TestResult(
            test_case=test_case,
            passed=passed,
            actual=actual_output,
            execution=execution,
            comparison_details=comparison_details,
        )

    async def _run_tests_batched(
        self,
        sandbox: Sandbox,
        code: str,
        tests: List[TestCase],
        verifier: OutputVerifier,
        stop_on_first_failure: bool,
        compile_first: bool,
        code_hash: str,
        tests_hash: str,
    ) -> BatchTestResult:
        """
        Run tests using batch execution API with crash resilience.

        This method uses the sandbox's execute_batch() to run all tests
        in a single podman exec call, reducing semaphore acquisitions
        from O(2N) to O(2).

        Args:
            sandbox: Sandbox with execute_batch() method
            code: Source code to test
            tests: List of test cases
            verifier: Output verifier for comparing results
            stop_on_first_failure: If True, stop after first failure
            compile_first: If True, compile before running tests
            code_hash: Pre-computed hash of code
            tests_hash: Pre-computed hash of tests

        Returns:
            BatchTestResult with results for each test
        """
        spec = BatchExecutionSpec(
            code=code,
            tests=tests,
            compile_first=compile_first,
            timeout_s=self._default_timeout_s,
            stop_on_first_failure=stop_on_first_failure,
        )

        results: List[TestResult] = []
        compile_result: Optional[CompileResult] = None
        received_done = False
        received_test_ids: Set[str] = set()

        # Build lookup for test cases by ID
        test_by_id = {t.id: t for t in tests}

        try:
            async for result in sandbox.execute_batch(spec):
                if isinstance(result, CompileResult):
                    compile_result = result
                    if not result.success:
                        # Compilation failed - return batch with all tests failed
                        return self._create_all_failed_batch(
                            tests=tests,
                            code_hash=code_hash,
                            tests_hash=tests_hash,
                            compile_result=compile_result,
                            reason="compilation_failed",
                        )
                elif isinstance(result, ExecutionResult):
                    # This is a test result - find the matching test case
                    # The execute_batch implementation tags results with test_id
                    # in the cache_key field
                    test_id = result.cache_key or ""
                    received_test_ids.add(test_id)

                    test_case = test_by_id.get(test_id)
                    if test_case is None:
                        logger.warning(
                            f"Received result for unknown test_id: {test_id}"
                        )
                        continue

                    # Build TestResult from ExecutionResult
                    if not result.succeeded:
                        # Execution failed
                        test_result = TestResult(
                            test_case=test_case,
                            passed=False,
                            actual=result.stdout,
                            execution=result,
                            comparison_details=self._get_execution_failure_details(
                                result
                            ),
                        )
                    else:
                        # Execution succeeded, compare output
                        actual_output = result.stdout
                        expected_output = str(test_case.expected)
                        passed, comparison_details = verifier.verify(
                            actual_output, expected_output
                        )
                        test_result = TestResult(
                            test_case=test_case,
                            passed=passed,
                            actual=actual_output,
                            execution=result,
                            comparison_details=comparison_details,
                        )
                    results.append(test_result)
                elif isinstance(result, dict) and result.get("type") == "done":
                    received_done = True
                    break

        except Exception as e:
            # Stream broke unexpectedly (OOM, container killed, etc.)
            logger.warning(f"Batch execution stream broke: {e}")

        # Handle missing tests (stream truncated before "done")
        if not received_done:
            for test in tests:
                if test.id not in received_test_ids:
                    # Create SANDBOX_ERROR result for missing tests
                    execution = ExecutionResult(
                        compile_result=compile_result
                        or CompileResult(status=CompileStatus.SUCCESS),
                        run_status=RunStatus.SANDBOX_ERROR,
                        stdout="",
                        stderr="Batch execution terminated unexpectedly",
                        exit_code=None,
                    )
                    results.append(
                        TestResult(
                            test_case=test,
                            passed=False,
                            actual="",
                            execution=execution,
                            comparison_details="Sandbox crashed before this test completed",
                        )
                    )

        return BatchTestResult(
            results=results,
            code_hash=code_hash,
            tests_hash=tests_hash,
        )

    def _get_execution_failure_details(self, execution: ExecutionResult) -> str:
        """
        Generate human-readable details for execution failures.

        Args:
            execution: The failed execution result

        Returns:
            Explanation of why execution failed
        """
        # Compilation failure
        if not execution.compiled:
            compile_msg = execution.compile_result.error_message or "Unknown error"
            if execution.compile_result.error_line is not None:
                return f"Compilation failed at line {execution.compile_result.error_line}: {compile_msg}"
            return f"Compilation failed: {compile_msg}"

        # Runtime failure
        if execution.run_status == RunStatus.TIMEOUT:
            return f"Execution timed out after {execution.run_duration_ms:.0f}ms"

        if execution.run_status == RunStatus.MEMORY_EXCEEDED:
            return "Memory limit exceeded"

        if execution.run_status == RunStatus.RUNTIME_ERROR:
            stderr = execution.stderr.strip()
            if stderr:
                # Show first few lines of stderr for debugging
                stderr_lines = stderr.split("\n")
                preview = "\n".join(stderr_lines[:5])
                if len(stderr_lines) > 5:
                    preview += f"\n... ({len(stderr_lines) - 5} more lines)"
                return f"Runtime error:\n{preview}"
            return f"Runtime error (exit code {execution.exit_code})"

        # Other failure
        return f"Execution failed with status: {execution.run_status}"

    def _create_all_failed_batch(
        self,
        tests: List[TestCase],
        code_hash: str,
        tests_hash: str,
        compile_result: CompileResult,
        reason: str,
    ) -> BatchTestResult:
        """
        Create a BatchTestResult where all tests failed due to compilation error.

        Args:
            tests: All test cases
            code_hash: Hash of the code
            tests_hash: Hash of the tests
            compile_result: The failed compilation result
            reason: Reason for batch failure

        Returns:
            BatchTestResult with all tests marked as failed
        """
        results: List[TestResult] = []

        for test_case in tests:
            # Create ExecutionResult with the compile failure
            execution = ExecutionResult(
                compile_result=compile_result,
                run_status=None,  # Never ran
                stdout="",
                stderr="",
                exit_code=None,
                compile_duration_ms=compile_result.duration_ms,
                run_duration_ms=0.0,
                total_duration_ms=compile_result.duration_ms,
            )

            test_result = TestResult(
                test_case=test_case,
                passed=False,
                actual="",
                execution=execution,
                comparison_details=self._get_execution_failure_details(execution),
            )
            results.append(test_result)

        return BatchTestResult(
            results=results,
            code_hash=code_hash,
            tests_hash=tests_hash,
        )

    def _create_not_run_result(
        self,
        test_case: TestCase,
        code_hash: str,
    ) -> TestResult:
        """
        Create a TestResult for a test that was skipped.

        Args:
            test_case: The test case that was skipped
            code_hash: Hash of the code (for metadata)

        Returns:
            TestResult marked as NOT_RUN
        """
        # Create a minimal ExecutionResult indicating the test wasn't run
        execution = ExecutionResult(
            compile_result=CompileResult(
                status=CompileStatus.SUCCESS  # Compilation already succeeded
            ),
            run_status=RunStatus.NOT_RUN,
            stdout="",
            stderr="",
            exit_code=None,
        )

        return TestResult(
            test_case=test_case,
            passed=False,
            actual="",
            execution=execution,
            comparison_details="Test skipped (stop_on_first_failure=True)",
        )

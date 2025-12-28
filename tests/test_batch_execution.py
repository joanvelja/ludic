"""
Unit tests for batch execution functionality.

Tests the batch execution path in StdinStdoutRunner using mock sandboxes
that implement execute_batch().
"""

import pytest
from typing import AsyncIterator, Union

from ludic.envs.code_exec.runners import StdinStdoutRunner
from ludic.envs.code_exec.types import (
    BatchExecutionSpec,
    TestCase,
    CompileResult,
    CompileStatus,
    ExecutionResult,
    RunStatus,
)
from ludic.envs.code_exec.adapters.base import ExactMatchVerifier


# ---------------------------------------------------------------------
# Mock Sandbox with execute_batch() support
# ---------------------------------------------------------------------


class MockBatchSandbox:
    """
    A mock sandbox that supports execute_batch() for testing the batched
    execution path in StdinStdoutRunner.

    Can be configured with:
      - batch_results: List of results to yield from execute_batch()
      - compile_success: Whether compilation succeeds
      - break_after: If set, raise exception after yielding N results
    """

    def __init__(
        self,
        batch_results: list[Union[CompileResult, ExecutionResult, dict]] | None = None,
        compile_success: bool = True,
        break_after: int | None = None,
    ):
        self._batch_results = batch_results or []
        self._compile_success = compile_success
        self._break_after = break_after
        self._python_version = "3.11"

        # Track calls
        self.execute_batch_calls: list[BatchExecutionSpec] = []

    @property
    def python_version(self) -> str:
        return self._python_version

    async def reset(self) -> None:
        pass

    async def compile(self, code: str, *, timeout_s: float = 5.0) -> CompileResult:
        if self._compile_success:
            return CompileResult(status=CompileStatus.SUCCESS, duration_ms=10.0)
        return CompileResult(
            status=CompileStatus.SYNTAX_ERROR,
            error_message="SyntaxError",
            duration_ms=5.0,
        )

    async def execute(
        self,
        code: str,
        *,
        stdin: str = "",
        skip_compile: bool = False,
        timeout_s: float = 10.0,
        memory_limit_mb: int | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecutionResult:
        # Fallback for non-batch execution
        return ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.SUCCESS,
            stdout="",
            stderr="",
            exit_code=0,
        )

    async def execute_batch(
        self,
        spec: BatchExecutionSpec,
    ) -> AsyncIterator[Union[CompileResult, ExecutionResult, dict]]:
        """Yield pre-configured batch results."""
        self.execute_batch_calls.append(spec)

        count = 0
        for result in self._batch_results:
            if self._break_after is not None and count >= self._break_after:
                raise RuntimeError("Simulated container crash")
            yield result
            count += 1


def make_success_execution(test_id: str, stdout: str) -> ExecutionResult:
    """Helper to create a successful ExecutionResult for a test."""
    return ExecutionResult(
        compile_result=CompileResult(status=CompileStatus.SUCCESS),
        run_status=RunStatus.SUCCESS,
        stdout=stdout,
        stderr="",
        exit_code=0,
        cache_key=test_id,  # Used to identify which test this result is for
    )


def make_failure_execution(
    test_id: str, status: RunStatus = RunStatus.RUNTIME_ERROR
) -> ExecutionResult:
    """Helper to create a failed ExecutionResult for a test."""
    return ExecutionResult(
        compile_result=CompileResult(status=CompileStatus.SUCCESS),
        run_status=status,
        stdout="",
        stderr="Error occurred",
        exit_code=1,
        cache_key=test_id,
    )


# ---------------------------------------------------------------------
# Batch Execution Tests
# ---------------------------------------------------------------------


class TestBatchExecution:
    @pytest.mark.asyncio
    async def test_batch_all_tests_pass(self):
        """All tests pass through batch execution."""
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS, duration_ms=10.0),
            make_success_execution("t1", "expected1"),
            make_success_execution("t2", "expected2"),
            {"type": "done", "passed": 2, "failed": 0},
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(input="input1", expected="expected1", id="t1"),
            TestCase(input="input2", expected="expected2", id="t2"),
        ]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="print('hello')",
            tests=tests,
            verifier=verifier,
        )

        assert result.all_passed is True
        assert result.passed_count == 2
        assert result.total_count == 2
        assert len(sandbox.execute_batch_calls) == 1

    @pytest.mark.asyncio
    async def test_batch_compile_failure(self):
        """Compilation failure returns all tests as failed."""
        batch_results = [
            CompileResult(
                status=CompileStatus.SYNTAX_ERROR,
                error_message="SyntaxError: invalid syntax",
                error_line=1,
                duration_ms=5.0,
            ),
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(input="input1", expected="x", id="t1"),
            TestCase(input="input2", expected="y", id="t2"),
        ]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="invalid syntax",
            tests=tests,
            verifier=verifier,
        )

        assert result.compile_failed is True
        assert result.all_passed is False
        assert result.passed_count == 0
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_batch_some_tests_fail(self):
        """Mixed pass/fail through batch execution."""
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS, duration_ms=10.0),
            make_success_execution("t1", "correct"),
            make_success_execution("t2", "wrong"),  # Output doesn't match expected
            {"type": "done", "passed": 1, "failed": 1},
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(input="input1", expected="correct", id="t1"),
            TestCase(input="input2", expected="correct", id="t2"),  # Will fail
        ]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
        )

        assert result.all_passed is False
        assert result.passed_count == 1
        assert result.total_count == 2
        assert result.results[0].passed is True
        assert result.results[1].passed is False

    @pytest.mark.asyncio
    async def test_batch_runtime_error(self):
        """Runtime error in batch execution."""
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS, duration_ms=10.0),
            make_failure_execution("t1", RunStatus.RUNTIME_ERROR),
            {"type": "done", "passed": 0, "failed": 1},
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        tests = [TestCase(input="input1", expected="output", id="t1")]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="raise Exception()",
            tests=tests,
            verifier=verifier,
        )

        assert result.passed_count == 0
        assert result.results[0].passed is False
        assert "Runtime error" in (result.results[0].comparison_details or "")

    @pytest.mark.asyncio
    async def test_batch_timeout(self):
        """Timeout in batch execution."""
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS, duration_ms=10.0),
            make_failure_execution("t1", RunStatus.TIMEOUT),
            {"type": "done", "passed": 0, "failed": 1},
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        tests = [TestCase(input="input1", expected="output", id="t1")]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="while True: pass",
            tests=tests,
            verifier=verifier,
        )

        assert result.passed_count == 0
        assert result.results[0].passed is False

    @pytest.mark.asyncio
    async def test_batch_stop_on_first_failure_spec(self):
        """Verify stop_on_first_failure is passed to BatchExecutionSpec."""
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS, duration_ms=10.0),
            make_success_execution("t1", "output"),
            {"type": "done", "passed": 1, "failed": 0},
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        tests = [TestCase(input="input1", expected="output", id="t1")]

        await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
            stop_on_first_failure=True,
        )

        assert len(sandbox.execute_batch_calls) == 1
        spec = sandbox.execute_batch_calls[0]
        assert spec.stop_on_first_failure is True

    @pytest.mark.asyncio
    async def test_batch_broken_stream_sandbox_error(self):
        """Broken stream marks missing tests as SANDBOX_ERROR."""
        # Stream breaks after compile result, before any test results
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS, duration_ms=10.0),
            make_success_execution("t1", "output1"),
            # Stream breaks here - t2 and t3 never received
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results, break_after=2)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(input="input1", expected="output1", id="t1"),
            TestCase(input="input2", expected="output2", id="t2"),
            TestCase(input="input3", expected="output3", id="t3"),
        ]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
        )

        # t1 should have succeeded, t2 and t3 should be SANDBOX_ERROR
        assert len(result.results) == 3
        assert result.results[0].passed is True
        assert result.results[0].test_case.id == "t1"

        # Find t2 and t3 results (order may vary due to dict iteration)
        t2_result = next(r for r in result.results if r.test_case.id == "t2")
        t3_result = next(r for r in result.results if r.test_case.id == "t3")

        assert t2_result.passed is False
        assert t2_result.execution.run_status == RunStatus.SANDBOX_ERROR
        assert "Sandbox crashed" in (t2_result.comparison_details or "")

        assert t3_result.passed is False
        assert t3_result.execution.run_status == RunStatus.SANDBOX_ERROR

    @pytest.mark.asyncio
    async def test_batch_no_done_marker_adds_missing(self):
        """Missing 'done' marker triggers fallback for unreceived tests."""
        # No "done" marker, but some tests received
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS, duration_ms=10.0),
            make_success_execution("t1", "output1"),
            # No "done" marker - stream ended unexpectedly
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(input="input1", expected="output1", id="t1"),
            TestCase(input="input2", expected="output2", id="t2"),
        ]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
        )

        assert len(result.results) == 2
        assert result.results[0].passed is True

        # t2 should be marked as SANDBOX_ERROR
        t2_result = next(r for r in result.results if r.test_case.id == "t2")
        assert t2_result.execution.run_status == RunStatus.SANDBOX_ERROR

    @pytest.mark.asyncio
    async def test_batch_disabled_falls_back_to_individual(self):
        """With use_batch_execution=False, individual execution is used."""
        sandbox = MockBatchSandbox()
        runner = StdinStdoutRunner(use_batch_execution=False)
        verifier = ExactMatchVerifier()

        tests = [TestCase(input="input1", expected="", id="t1")]

        await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
        )

        # execute_batch should NOT be called
        assert len(sandbox.execute_batch_calls) == 0

    @pytest.mark.asyncio
    async def test_batch_spec_contains_all_test_info(self):
        """Verify BatchExecutionSpec contains all test information."""
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS),
            make_success_execution("t1", "out"),
            {"type": "done"},
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True, default_timeout_s=7.5)
        verifier = ExactMatchVerifier()

        tests = [TestCase(input="my_input", expected="out", id="t1")]

        await runner.run_tests(
            sandbox=sandbox,
            code="my_code",
            tests=tests,
            verifier=verifier,
            compile_first=True,
            stop_on_first_failure=False,
        )

        assert len(sandbox.execute_batch_calls) == 1
        spec = sandbox.execute_batch_calls[0]

        assert spec.code == "my_code"
        assert len(spec.tests) == 1
        assert spec.tests[0].id == "t1"
        assert spec.tests[0].input == "my_input"
        assert spec.compile_first is True
        assert spec.stop_on_first_failure is False
        assert spec.timeout_s == 7.5

    @pytest.mark.asyncio
    async def test_batch_hashes_computed(self):
        """Verify code_hash and tests_hash are computed for batch execution."""
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS),
            make_success_execution("t1", "output"),
            {"type": "done"},
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        code = "print('hello')"
        tests = [TestCase(input="input1", expected="output", id="t1")]

        result = await runner.run_tests(
            sandbox=sandbox,
            code=code,
            tests=tests,
            verifier=verifier,
        )

        # Verify hashes are present
        assert len(result.code_hash) == 16
        assert len(result.tests_hash) == 16
        assert all(c in "0123456789abcdef" for c in result.code_hash)


class TestBatchExecutionNotRunStatus:
    """Tests for NOT_RUN status handling in batch execution."""

    @pytest.mark.asyncio
    async def test_not_run_tests_from_batch_stream(self):
        """Tests marked as NOT_RUN in batch stream are handled correctly."""
        batch_results = [
            CompileResult(status=CompileStatus.SUCCESS, duration_ms=10.0),
            make_failure_execution("t1", RunStatus.RUNTIME_ERROR),
            # t2 marked as not_run by batch_runner due to stop_on_first_failure
            ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.NOT_RUN,
                stdout="",
                stderr="",
                exit_code=None,
                cache_key="t2",
            ),
            {"type": "done", "passed": 0, "failed": 1},
        ]
        sandbox = MockBatchSandbox(batch_results=batch_results)
        runner = StdinStdoutRunner(use_batch_execution=True)
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(input="input1", expected="output1", id="t1"),
            TestCase(input="input2", expected="output2", id="t2"),
        ]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
            stop_on_first_failure=True,
        )

        assert len(result.results) == 2
        assert result.results[0].passed is False
        assert result.results[0].execution.run_status == RunStatus.RUNTIME_ERROR

        t2_result = next(r for r in result.results if r.test_case.id == "t2")
        assert t2_result.passed is False
        assert t2_result.execution.run_status == RunStatus.NOT_RUN

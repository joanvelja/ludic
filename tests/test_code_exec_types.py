"""
Unit tests for ludic.envs.code_exec.types

Tests all dataclasses and their properties/methods.
"""

import pytest

from ludic.envs.code_exec.types import (
    CompileStatus,
    RunStatus,
    CompileResult,
    ExecutionResult,
    TestCase,
    TestResult,
    BatchTestResult,
)


# ---------------------------------------------------------------------
# CompileResult Tests
# ---------------------------------------------------------------------


class TestCompileResult:
    def test_success_property_true_when_status_success(self):
        result = CompileResult(status=CompileStatus.SUCCESS)
        assert result.success is True

    def test_success_property_false_when_syntax_error(self):
        result = CompileResult(
            status=CompileStatus.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax",
            error_line=5,
            error_column=10,
        )
        assert result.success is False

    def test_success_property_false_for_all_error_statuses(self):
        error_statuses = [
            CompileStatus.SYNTAX_ERROR,
            CompileStatus.IMPORT_ERROR,
            CompileStatus.TIMEOUT,
            CompileStatus.UNKNOWN_ERROR,
        ]
        for status in error_statuses:
            result = CompileResult(status=status)
            assert result.success is False, f"Expected success=False for {status}"

    def test_duration_ms_default_zero(self):
        result = CompileResult(status=CompileStatus.SUCCESS)
        assert result.duration_ms == 0.0


# ---------------------------------------------------------------------
# ExecutionResult Tests
# ---------------------------------------------------------------------


class TestExecutionResult:
    def test_compiled_true_when_compile_succeeded(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.SUCCESS,
        )
        assert result.compiled is True

    def test_compiled_false_when_compile_failed(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SYNTAX_ERROR),
        )
        assert result.compiled is False

    def test_succeeded_true_when_compiled_and_run_success(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.SUCCESS,
            stdout="output",
        )
        assert result.succeeded is True

    def test_succeeded_false_when_compile_failed(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SYNTAX_ERROR),
        )
        assert result.succeeded is False

    def test_succeeded_false_when_runtime_error(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.RUNTIME_ERROR,
            stderr="NameError: name 'x' is not defined",
        )
        assert result.succeeded is False

    def test_succeeded_false_when_timeout(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.TIMEOUT,
        )
        assert result.succeeded is False

    def test_timed_out_true_when_compile_timeout(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.TIMEOUT),
        )
        assert result.timed_out is True

    def test_timed_out_true_when_run_timeout(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.TIMEOUT,
        )
        assert result.timed_out is True

    def test_timed_out_false_when_success(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.SUCCESS,
        )
        assert result.timed_out is False

    def test_default_values(self):
        result = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
        )
        assert result.run_status is None
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code is None
        assert result.cache_hit is False


# ---------------------------------------------------------------------
# TestCase Tests
# ---------------------------------------------------------------------


class TestTestCase:
    def test_basic_creation(self):
        tc = TestCase(input="1 2", expected="3", id="test_add")
        assert tc.input == "1 2"
        assert tc.expected == "3"
        assert tc.id == "test_add"

    def test_default_weight(self):
        tc = TestCase(input="x", expected="y")
        assert tc.weight == 1.0

    def test_default_metadata_is_empty_dict(self):
        tc = TestCase(input="x", expected="y")
        assert tc.metadata == {}

    def test_metadata_with_custom_values(self):
        tc = TestCase(
            input="x",
            expected="y",
            metadata={"timeout_s": 10.0, "category": "math"},
        )
        assert tc.metadata["timeout_s"] == 10.0
        assert tc.metadata["category"] == "math"


# ---------------------------------------------------------------------
# TestResult Tests
# ---------------------------------------------------------------------


class TestTestResult:
    def test_compiled_delegates_to_execution(self):
        execution = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.SUCCESS,
        )
        tr = TestResult(
            test_case=TestCase(input="1", expected="1"),
            passed=True,
            actual="1",
            execution=execution,
        )
        assert tr.compiled is True

    def test_compiled_false_when_execution_compile_failed(self):
        execution = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SYNTAX_ERROR),
        )
        tr = TestResult(
            test_case=TestCase(input="1", expected="1"),
            passed=False,
            actual="",
            execution=execution,
        )
        assert tr.compiled is False

    def test_ran_true_when_execution_has_run_status(self):
        execution = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.SUCCESS,
        )
        tr = TestResult(
            test_case=TestCase(input="1", expected="1"),
            passed=True,
            actual="1",
            execution=execution,
        )
        assert tr.ran is True

    def test_ran_false_when_run_status_none(self):
        execution = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SYNTAX_ERROR),
            run_status=None,
        )
        tr = TestResult(
            test_case=TestCase(input="1", expected="1"),
            passed=False,
            actual="",
            execution=execution,
        )
        assert tr.ran is False

    def test_ran_false_when_run_status_not_run(self):
        execution = ExecutionResult(
            compile_result=CompileResult(status=CompileStatus.SUCCESS),
            run_status=RunStatus.NOT_RUN,
        )
        tr = TestResult(
            test_case=TestCase(input="1", expected="1"),
            passed=False,
            actual="",
            execution=execution,
        )
        assert tr.ran is False


# ---------------------------------------------------------------------
# BatchTestResult Tests
# ---------------------------------------------------------------------


def _make_test_result(passed: bool, compiled: bool = True) -> TestResult:
    """Helper to create TestResult with minimal boilerplate."""
    if compiled:
        compile_result = CompileResult(status=CompileStatus.SUCCESS)
        run_status = RunStatus.SUCCESS if passed else RunStatus.RUNTIME_ERROR
    else:
        compile_result = CompileResult(status=CompileStatus.SYNTAX_ERROR)
        run_status = None

    return TestResult(
        test_case=TestCase(input="x", expected="y"),
        passed=passed,
        actual="y" if passed else "z",
        execution=ExecutionResult(
            compile_result=compile_result,
            run_status=run_status,
        ),
    )


class TestBatchTestResult:
    def test_passed_count(self):
        results = [
            _make_test_result(passed=True),
            _make_test_result(passed=True),
            _make_test_result(passed=False),
        ]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.passed_count == 2

    def test_total_count(self):
        results = [_make_test_result(passed=True) for _ in range(5)]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.total_count == 5

    def test_all_passed_true_when_all_pass(self):
        results = [_make_test_result(passed=True) for _ in range(3)]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.all_passed is True

    def test_all_passed_false_when_one_fails(self):
        results = [
            _make_test_result(passed=True),
            _make_test_result(passed=False),
        ]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.all_passed is False

    def test_all_passed_false_when_empty(self):
        batch = BatchTestResult(results=[], code_hash="abc", tests_hash="xyz")
        assert batch.all_passed is False

    def test_pass_rate_full(self):
        results = [_make_test_result(passed=True) for _ in range(4)]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.pass_rate == pytest.approx(1.0)

    def test_pass_rate_half(self):
        results = [
            _make_test_result(passed=True),
            _make_test_result(passed=True),
            _make_test_result(passed=False),
            _make_test_result(passed=False),
        ]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.pass_rate == pytest.approx(0.5)

    def test_pass_rate_zero_when_empty(self):
        batch = BatchTestResult(results=[], code_hash="abc", tests_hash="xyz")
        assert batch.pass_rate == pytest.approx(0.0)

    def test_first_failure_returns_first_failed_test(self):
        results = [
            _make_test_result(passed=True),
            _make_test_result(passed=False),  # first failure
            _make_test_result(passed=False),
        ]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.first_failure is results[1]

    def test_first_failure_none_when_all_pass(self):
        results = [_make_test_result(passed=True) for _ in range(3)]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.first_failure is None

    def test_compile_failed_true_when_first_result_not_compiled(self):
        results = [
            _make_test_result(passed=False, compiled=False),
            _make_test_result(passed=False, compiled=False),
        ]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.compile_failed is True

    def test_compile_failed_false_when_compiled(self):
        results = [_make_test_result(passed=True)]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        assert batch.compile_failed is False

    def test_compile_failed_false_when_empty(self):
        batch = BatchTestResult(results=[], code_hash="abc", tests_hash="xyz")
        assert batch.compile_failed is False

    def test_get_failures_returns_only_failed_tests(self):
        results = [
            _make_test_result(passed=True),
            _make_test_result(passed=False),
            _make_test_result(passed=True),
            _make_test_result(passed=False),
        ]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        failures = batch.get_failures()
        assert len(failures) == 2
        assert failures[0] is results[1]
        assert failures[1] is results[3]

    def test_get_successes_returns_only_passed_tests(self):
        results = [
            _make_test_result(passed=True),
            _make_test_result(passed=False),
            _make_test_result(passed=True),
        ]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")
        successes = batch.get_successes()
        assert len(successes) == 2
        assert successes[0] is results[0]
        assert successes[1] is results[2]

    def test_timing_aggregation(self):
        # Create results with specific timing
        def make_result_with_timing(compile_ms: float, run_ms: float) -> TestResult:
            return TestResult(
                test_case=TestCase(input="x", expected="y"),
                passed=True,
                actual="y",
                execution=ExecutionResult(
                    compile_result=CompileResult(
                        status=CompileStatus.SUCCESS,
                        duration_ms=compile_ms,
                    ),
                    run_status=RunStatus.SUCCESS,
                    compile_duration_ms=compile_ms,
                    run_duration_ms=run_ms,
                    total_duration_ms=compile_ms + run_ms,
                ),
            )

        results = [
            make_result_with_timing(10.0, 100.0),
            make_result_with_timing(10.0, 200.0),
            make_result_with_timing(10.0, 150.0),
        ]
        batch = BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")

        # Compile time: max across all (since compilation usually happens once)
        assert batch.total_compile_ms == pytest.approx(10.0)

        # Run time: sum across all tests
        assert batch.total_run_ms == pytest.approx(450.0)  # 100 + 200 + 150

        # Total: sum of all total_duration_ms
        assert batch.total_execution_ms == pytest.approx(480.0)  # 110 + 210 + 160

"""
Unit tests for ludic.envs.code_exec.runners

Tests hash utilities and StdinStdoutRunner with mock sandbox.
"""

import pytest

from ludic.envs.code_exec.runners import (
    compute_hash,
    hash_tests,
    StdinStdoutRunner,
)
from ludic.envs.code_exec.types import (
    TestCase,
    CompileResult,
    CompileStatus,
    ExecutionResult,
    RunStatus,
)
from ludic.envs.code_exec.adapters.base import ExactMatchVerifier


# ---------------------------------------------------------------------
# Hash Utility Tests
# ---------------------------------------------------------------------


class TestComputeHash:
    def test_returns_16_chars(self):
        result = compute_hash("hello world")
        assert len(result) == 16

    def test_deterministic(self):
        result1 = compute_hash("test content")
        result2 = compute_hash("test content")
        assert result1 == result2

    def test_different_content_different_hash(self):
        result1 = compute_hash("content a")
        result2 = compute_hash("content b")
        assert result1 != result2

    def test_hex_characters_only(self):
        result = compute_hash("any content")
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string(self):
        result = compute_hash("")
        assert len(result) == 16


class TestHashTests:
    def test_returns_16_chars(self):
        tests = [TestCase(input="1", expected="2", id="t1")]
        result = hash_tests(tests)
        assert len(result) == 16

    def test_deterministic(self):
        tests = [
            TestCase(input="1", expected="a", id="t1"),
            TestCase(input="2", expected="b", id="t2"),
        ]
        result1 = hash_tests(tests)
        result2 = hash_tests(tests)
        assert result1 == result2

    def test_different_tests_different_hash(self):
        tests1 = [TestCase(input="1", expected="a", id="t1")]
        tests2 = [TestCase(input="2", expected="b", id="t2")]
        result1 = hash_tests(tests1)
        result2 = hash_tests(tests2)
        assert result1 != result2

    def test_order_matters(self):
        tests1 = [
            TestCase(input="1", expected="a", id="t1"),
            TestCase(input="2", expected="b", id="t2"),
        ]
        tests2 = [
            TestCase(input="2", expected="b", id="t2"),
            TestCase(input="1", expected="a", id="t1"),
        ]
        result1 = hash_tests(tests1)
        result2 = hash_tests(tests2)
        assert result1 != result2

    def test_empty_list(self):
        result = hash_tests([])
        assert len(result) == 16


# ---------------------------------------------------------------------
# Mock Sandbox for Runner Tests
# ---------------------------------------------------------------------


class MockSandbox:
    """
    A mock sandbox for testing runners.

    Can be configured with:
      - compile_result: What to return from compile()
      - execute_results: Dict mapping stdin -> ExecutionResult
      - default_execute_result: Fallback for unmapped stdin
    """

    def __init__(
        self,
        compile_result: CompileResult | None = None,
        execute_results: dict[str, ExecutionResult] | None = None,
        default_stdout: str = "",
    ):
        self._compile_result = compile_result or CompileResult(
            status=CompileStatus.SUCCESS,
            duration_ms=10.0,
        )
        self._execute_results = execute_results or {}
        self._default_stdout = default_stdout
        self._python_version = "3.11"

        # Track calls for assertions
        self.compile_calls: list[str] = []
        self.execute_calls: list[tuple[str, str]] = []

    @property
    def python_version(self) -> str:
        return self._python_version

    async def reset(self) -> None:
        pass

    async def compile(self, code: str, *, timeout_s: float = 5.0) -> CompileResult:
        self.compile_calls.append(code)
        return self._compile_result

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
        self.execute_calls.append((code, stdin))

        if stdin in self._execute_results:
            return self._execute_results[stdin]

        # Default: successful execution returning default_stdout
        return ExecutionResult(
            compile_result=self._compile_result,
            run_status=RunStatus.SUCCESS,
            stdout=self._default_stdout,
            stderr="",
            exit_code=0,
            compile_duration_ms=10.0,
            run_duration_ms=50.0,
            total_duration_ms=60.0,
        )


# ---------------------------------------------------------------------
# StdinStdoutRunner Tests
# ---------------------------------------------------------------------


class TestStdinStdoutRunner:
    @pytest.mark.asyncio
    async def test_all_tests_pass(self):
        sandbox = MockSandbox(default_stdout="expected_output")
        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(input="input1", expected="expected_output", id="t1"),
            TestCase(input="input2", expected="expected_output", id="t2"),
        ]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="print('expected_output')",
            tests=tests,
            verifier=verifier,
        )

        assert result.all_passed is True
        assert result.passed_count == 2
        assert result.total_count == 2

    @pytest.mark.asyncio
    async def test_some_tests_fail(self):
        # First test passes, second fails
        execute_results = {
            "input1": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="correct",
            ),
            "input2": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="wrong",
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results)
        runner = StdinStdoutRunner()
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
    async def test_compile_failure_fails_all_tests(self):
        compile_result = CompileResult(
            status=CompileStatus.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax",
            error_line=5,
            duration_ms=5.0,
        )
        sandbox = MockSandbox(compile_result=compile_result)
        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(input="input1", expected="x", id="t1"),
            TestCase(input="input2", expected="y", id="t2"),
            TestCase(input="input3", expected="z", id="t3"),
        ]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="invalid syntax here",
            tests=tests,
            verifier=verifier,
            compile_first=True,
        )

        assert result.compile_failed is True
        assert result.all_passed is False
        assert result.passed_count == 0
        assert len(result.results) == 3

        # All should have compile failure details
        for r in result.results:
            assert r.compiled is False
            assert "Compilation failed" in (r.comparison_details or "")

    @pytest.mark.asyncio
    async def test_stop_on_first_failure(self):
        execute_results = {
            "input1": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="wrong",  # First test fails
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results, default_stdout="correct")
        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(input="input1", expected="correct", id="t1"),  # Fails
            TestCase(input="input2", expected="correct", id="t2"),  # Should be skipped
            TestCase(input="input3", expected="correct", id="t3"),  # Should be skipped
        ]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
            stop_on_first_failure=True,
        )

        assert result.passed_count == 0
        assert len(result.results) == 3

        # First test ran and failed
        assert result.results[0].passed is False
        assert result.results[0].ran is True

        # Second and third were skipped
        assert result.results[1].passed is False
        assert result.results[1].execution.run_status == RunStatus.NOT_RUN
        assert result.results[2].passed is False
        assert result.results[2].execution.run_status == RunStatus.NOT_RUN

    @pytest.mark.asyncio
    async def test_runtime_error_fails_test(self):
        execute_results = {
            "input1": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.RUNTIME_ERROR,
                stdout="",
                stderr="NameError: name 'x' is not defined",
                exit_code=1,
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results)
        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()

        tests = [TestCase(input="input1", expected="output", id="t1")]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="print(x)",
            tests=tests,
            verifier=verifier,
        )

        assert result.passed_count == 0
        assert result.results[0].passed is False
        assert "Runtime error" in (result.results[0].comparison_details or "")

    @pytest.mark.asyncio
    async def test_timeout_fails_test(self):
        execute_results = {
            "input1": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.TIMEOUT,
                stdout="",
                stderr="",
                run_duration_ms=5000.0,
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results)
        runner = StdinStdoutRunner()
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
        assert "timed out" in (result.results[0].comparison_details or "").lower()

    @pytest.mark.asyncio
    async def test_memory_exceeded_fails_test(self):
        execute_results = {
            "input1": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.MEMORY_EXCEEDED,
                stdout="",
                stderr="",
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results)
        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()

        tests = [TestCase(input="input1", expected="output", id="t1")]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="x = [0] * 10**9",
            tests=tests,
            verifier=verifier,
        )

        assert result.passed_count == 0
        assert result.results[0].passed is False
        assert "Memory" in (result.results[0].comparison_details or "")

    @pytest.mark.asyncio
    async def test_per_test_timeout_override(self):
        sandbox = MockSandbox(default_stdout="output")
        runner = StdinStdoutRunner(default_timeout_s=5.0)
        verifier = ExactMatchVerifier()

        tests = [
            TestCase(
                input="input1",
                expected="output",
                id="t1",
                metadata={"timeout_s": 30.0},  # Override
            ),
        ]

        await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
        )

        # Check that execute was called with the overridden timeout
        # The mock doesn't actually use timeout, but we can verify the call was made
        assert len(sandbox.execute_calls) == 1

    @pytest.mark.asyncio
    async def test_compile_first_false_skips_compile(self):
        sandbox = MockSandbox(default_stdout="output")
        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()

        tests = [TestCase(input="input1", expected="output", id="t1")]

        await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
            compile_first=False,
        )

        # compile() should not be called when compile_first=False
        assert len(sandbox.compile_calls) == 0
        assert len(sandbox.execute_calls) == 1

    @pytest.mark.asyncio
    async def test_hashes_computed_correctly(self):
        sandbox = MockSandbox(default_stdout="output")
        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()

        code = "print('hello')"
        tests = [TestCase(input="input1", expected="output", id="t1")]

        result = await runner.run_tests(
            sandbox=sandbox,
            code=code,
            tests=tests,
            verifier=verifier,
        )

        # Verify hashes are present and have correct format
        assert len(result.code_hash) == 16
        assert len(result.tests_hash) == 16
        assert all(c in "0123456789abcdef" for c in result.code_hash)
        assert all(c in "0123456789abcdef" for c in result.tests_hash)

        # Verify code_hash matches compute_hash
        from ludic.envs.code_exec.runners import compute_hash

        assert result.code_hash == compute_hash(code)

    @pytest.mark.asyncio
    async def test_whitespace_stripping_in_comparison(self):
        """Verifier should strip whitespace from output."""
        sandbox = MockSandbox(default_stdout="  output\n")
        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()  # strips by default

        tests = [TestCase(input="input1", expected="output", id="t1")]

        result = await runner.run_tests(
            sandbox=sandbox,
            code="code",
            tests=tests,
            verifier=verifier,
        )

        assert result.all_passed is True

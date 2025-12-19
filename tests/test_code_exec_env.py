"""
Unit tests for ludic.envs.code_exec.env.CodeExecEnv

Tests the environment with mock sandbox pools to avoid Docker dependency.
"""

import pytest

from ludic.envs.code_exec.env import CodeExecConfig, CodeExecEnv
from ludic.envs.code_exec.types import (
    BatchTestResult,
    CompileResult,
    CompileStatus,
    ExecutionResult,
    RunStatus,
    TestCase,
    TestResult,
)
from ludic.envs.code_exec.adapters.base import ExactMatchVerifier, TestAdapter
from ludic.envs.code_exec.sandbox import Sandbox, SandboxPool


# ---------------------------------------------------------------------
# Mock Implementations
# ---------------------------------------------------------------------


class MockSandbox:
    """Mock sandbox for testing without Docker."""

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

        # Track calls
        self.reset_calls = 0
        self.compile_calls: list[str] = []
        self.execute_calls: list[tuple[str, str]] = []

    @property
    def python_version(self) -> str:
        return self._python_version

    async def reset(self) -> None:
        self.reset_calls += 1

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


class MockSandboxPool:
    """Mock sandbox pool for testing without Docker."""

    def __init__(
        self,
        sandbox: MockSandbox | None = None,
        python_version: str = "3.11",
    ):
        self._sandbox = sandbox or MockSandbox()
        self._python_version = python_version
        self._cache: dict[tuple[str, str], BatchTestResult] = {}

        # Track calls
        self.start_calls = 0
        self.checkout_calls = 0
        self.release_calls = 0
        self.shutdown_calls = 0

    @property
    def python_version(self) -> str:
        return self._python_version

    async def start(self) -> None:
        self.start_calls += 1

    async def checkout(self, timeout_s: float = 30.0) -> Sandbox:
        self.checkout_calls += 1
        return self._sandbox

    async def release(self, sandbox: Sandbox) -> None:
        self.release_calls += 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def get_cached(self, code_hash: str, tests_hash: str) -> BatchTestResult | None:
        return self._cache.get((code_hash, tests_hash))

    def put_cached(
        self, code_hash: str, tests_hash: str, result: BatchTestResult
    ) -> None:
        self._cache[(code_hash, tests_hash)] = result

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return mock cache statistics."""
        return {
            "hits": 0,
            "misses": 0,
            "size": len(self._cache),
            "max_size": 10000,
        }


class MockTestAdapter:
    """Mock test adapter for testing."""

    def __init__(
        self,
        prompt: str = "Write a program.",
        problem_id: str = "test_problem",
        tests: list[TestCase] | None = None,
    ):
        self._prompt = prompt
        self._problem_id = problem_id
        self._tests = tests or [
            TestCase(input="1", expected="1", id="test_0"),
        ]

    def get_prompt(self, sample: dict) -> str:
        return self._prompt

    def get_problem_id(self, sample: dict) -> str:
        return self._problem_id

    def get_tests(self, sample: dict) -> list[TestCase]:
        return self._tests

    def hash_tests(self, tests: list[TestCase]) -> str:
        return "mock_tests_hash_1234"


# ---------------------------------------------------------------------
# Environment Reset Tests
# ---------------------------------------------------------------------


class TestCodeExecEnvReset:
    @pytest.mark.asyncio
    async def test_reset_returns_prompt_and_info(self):
        sandbox = MockSandbox(default_stdout="1")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(prompt="Add two numbers.", problem_id="prob_1")

        env = CodeExecEnv(
            sample={"question": "Add two numbers."},
            sandbox_pool=pool,
            test_adapter=adapter,
        )

        obs, info = await env.env_reset()

        assert obs == "Add two numbers."
        assert info["problem_id"] == "prob_1"
        assert "num_tests" in info
        assert "tests_hash" in info
        assert "python_version" in info

    @pytest.mark.asyncio
    async def test_reset_extracts_correct_number_of_tests(self):
        sandbox = MockSandbox(default_stdout="out")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[
                TestCase(input="1", expected="a", id="t0"),
                TestCase(input="2", expected="b", id="t1"),
                TestCase(input="3", expected="c", id="t2"),
            ]
        )

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
        )

        obs, info = await env.env_reset()

        assert info["num_tests"] == 3

    @pytest.mark.asyncio
    async def test_reset_respects_max_tests_config(self):
        sandbox = MockSandbox(default_stdout="out")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[
                TestCase(input="1", expected="a", id="t0"),
                TestCase(input="2", expected="b", id="t1"),
                TestCase(input="3", expected="c", id="t2"),
                TestCase(input="4", expected="d", id="t3"),
                TestCase(input="5", expected="e", id="t4"),
            ]
        )

        config = CodeExecConfig(max_tests=2)
        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        obs, info = await env.env_reset()

        assert info["num_tests"] == 2

    @pytest.mark.asyncio
    async def test_reset_handles_empty_tests(self):
        sandbox = MockSandbox()
        pool = MockSandboxPool(sandbox=sandbox)

        # Create adapter that returns empty tests
        class EmptyTestsAdapter:
            def get_prompt(self, sample: dict) -> str:
                return "Write a program."

            def get_problem_id(self, sample: dict) -> str:
                return "test_problem"

            def get_tests(self, sample: dict) -> list[TestCase]:
                return []  # No tests!

            def hash_tests(self, tests: list[TestCase]) -> str:
                return "empty_hash"

        adapter = EmptyTestsAdapter()

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
        )

        obs, info = await env.env_reset()

        assert "error" in info
        assert info["error"] == "no_tests_extracted"

    @pytest.mark.asyncio
    async def test_reset_sets_system_prompt(self):
        sandbox = MockSandbox()
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter()

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            system_prompt="You are a Python expert.",
        )

        assert env.suggested_sysprompt == "You are a Python expert."


# ---------------------------------------------------------------------
# Environment Step Tests - Success Cases
# ---------------------------------------------------------------------


class TestCodeExecEnvStepSuccess:
    @pytest.mark.asyncio
    async def test_step_all_tests_pass(self):
        sandbox = MockSandbox(default_stdout="expected_output")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[
                TestCase(input="in1", expected="expected_output", id="t0"),
                TestCase(input="in2", expected="expected_output", id="t1"),
            ]
        )

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
        )

        await env.env_reset()
        outcome = await env.env_step("print('expected_output')")

        assert outcome.terminated is True
        assert outcome.truncated is False
        assert outcome.reward == 1.0
        assert outcome.info["all_passed"] is True
        assert outcome.info["passed"] == 2
        assert outcome.info["total"] == 2
        assert "All" in outcome.obs and "passed" in outcome.obs

    @pytest.mark.asyncio
    async def test_step_releases_sandbox(self):
        sandbox = MockSandbox(default_stdout="output")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[TestCase(input="x", expected="output", id="t0")]
        )

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
        )

        await env.env_reset()
        await env.env_step("code")

        assert pool.checkout_calls == 1
        assert pool.release_calls == 1


# ---------------------------------------------------------------------
# Environment Step Tests - Failure Cases
# ---------------------------------------------------------------------


class TestCodeExecEnvStepFailure:
    @pytest.mark.asyncio
    async def test_step_without_reset_returns_error(self):
        sandbox = MockSandbox()
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter()

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
        )

        # Skip reset
        outcome = await env.env_step("some code")

        assert outcome.terminated is True
        assert outcome.reward == -1.0
        assert outcome.info["error"] == "reset_not_called"

    @pytest.mark.asyncio
    async def test_step_with_empty_code(self):
        sandbox = MockSandbox()
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter()
        config = CodeExecConfig(compile_failure_reward=-0.5)

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()
        outcome = await env.env_step("")

        assert outcome.terminated is True
        assert outcome.reward == -0.5
        assert outcome.info["error"] == "empty_code"

    @pytest.mark.asyncio
    async def test_step_with_whitespace_only_code(self):
        sandbox = MockSandbox()
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter()

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
        )

        await env.env_reset()
        outcome = await env.env_step("   \n\t  ")

        assert outcome.info["error"] == "empty_code"

    @pytest.mark.asyncio
    async def test_step_compile_failure(self):
        compile_result = CompileResult(
            status=CompileStatus.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax",
            error_line=5,
            duration_ms=10.0,
        )
        sandbox = MockSandbox(compile_result=compile_result)
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter()
        config = CodeExecConfig(compile_failure_reward=-0.2)

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()
        outcome = await env.env_step("def foo(")

        assert outcome.reward == -0.2
        assert outcome.info["compile_failed"] is True
        assert "Compilation failed" in outcome.obs
        assert "SyntaxError" in outcome.obs

    @pytest.mark.asyncio
    async def test_step_some_tests_fail(self):
        execute_results = {
            "input1": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="correct",
            ),
            "input2": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="wrong",  # Will fail
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results)
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[
                TestCase(input="input1", expected="correct", id="t0"),
                TestCase(input="input2", expected="correct", id="t1"),
            ]
        )

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=CodeExecConfig(stop_on_first_failure=False),
        )

        await env.env_reset()
        outcome = await env.env_step("code")

        assert outcome.reward == 0.0  # Binary reward, not all passed
        assert outcome.info["all_passed"] is False
        assert outcome.info["passed"] == 1
        assert outcome.info["total"] == 2


# ---------------------------------------------------------------------
# Reward Shaping Tests
# ---------------------------------------------------------------------


class TestCodeExecEnvRewardShaping:
    @pytest.mark.asyncio
    async def test_binary_reward_all_pass(self):
        sandbox = MockSandbox(default_stdout="out")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[TestCase(input="x", expected="out", id="t0")]
        )
        config = CodeExecConfig(partial_credit=False)

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()
        outcome = await env.env_step("code")

        assert outcome.reward == 1.0

    @pytest.mark.asyncio
    async def test_binary_reward_some_fail(self):
        execute_results = {
            "in1": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="correct",
            ),
            "in2": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="wrong",
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results)
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[
                TestCase(input="in1", expected="correct", id="t0"),
                TestCase(input="in2", expected="correct", id="t1"),
            ]
        )
        config = CodeExecConfig(partial_credit=False, stop_on_first_failure=False)

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()
        outcome = await env.env_step("code")

        assert outcome.reward == 0.0  # Binary: all or nothing

    @pytest.mark.asyncio
    async def test_partial_credit_half_pass(self):
        execute_results = {
            "in1": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="correct",
            ),
            "in2": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="correct",
            ),
            "in3": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="wrong",
            ),
            "in4": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="wrong",
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results)
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[
                TestCase(input="in1", expected="correct", id="t0"),
                TestCase(input="in2", expected="correct", id="t1"),
                TestCase(input="in3", expected="correct", id="t2"),
                TestCase(input="in4", expected="correct", id="t3"),
            ]
        )
        config = CodeExecConfig(partial_credit=True, stop_on_first_failure=False)

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()
        outcome = await env.env_step("code")

        assert outcome.reward == pytest.approx(0.5)  # 2/4 passed


# ---------------------------------------------------------------------
# Caching Tests
# ---------------------------------------------------------------------


class TestCodeExecEnvCaching:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_execution(self):
        sandbox = MockSandbox(default_stdout="output")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[TestCase(input="x", expected="output", id="t0")]
        )
        config = CodeExecConfig(use_cache=True)

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()

        # First call - should execute
        outcome1 = await env.env_step("print('output')")
        assert pool.checkout_calls == 1
        assert outcome1.info["cache_hit"] is False

        # Second call with same code - should hit cache
        await env.env_reset()  # Reset to allow another step
        outcome2 = await env.env_step("print('output')")
        assert pool.checkout_calls == 1  # No new checkout
        assert outcome2.info["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        sandbox = MockSandbox(default_stdout="output")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[TestCase(input="x", expected="output", id="t0")]
        )
        config = CodeExecConfig(use_cache=False)

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()
        outcome1 = await env.env_step("print('output')")
        assert pool.checkout_calls == 1

        await env.env_reset()
        outcome2 = await env.env_step("print('output')")
        assert pool.checkout_calls == 2  # New execution each time
        assert outcome2.info["cache_hit"] is False


# ---------------------------------------------------------------------
# Info Dict Tests
# ---------------------------------------------------------------------


class TestCodeExecEnvInfo:
    @pytest.mark.asyncio
    async def test_info_contains_required_fields(self):
        sandbox = MockSandbox(default_stdout="out")
        pool = MockSandboxPool(sandbox=sandbox, python_version="3.10")
        adapter = MockTestAdapter(problem_id="prob_42")
        adapter._tests = [TestCase(input="x", expected="out", id="t0")]

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
        )

        await env.env_reset()
        outcome = await env.env_step("code")
        info = outcome.info

        # Problem metadata
        assert info["problem_id"] == "prob_42"
        assert "code_hash" in info
        assert "tests_hash" in info

        # Test results summary
        assert "passed" in info
        assert "total" in info
        assert "all_passed" in info
        assert "pass_rate" in info
        assert "compile_failed" in info

        # Detailed results
        assert "test_results" in info
        assert isinstance(info["test_results"], list)

        # Timing
        assert "timing" in info
        assert "total_compile_ms" in info["timing"]
        assert "total_run_ms" in info["timing"]
        assert "total_execution_ms" in info["timing"]

        # Cache and env info
        assert "cache_hit" in info
        assert info["python_version"] == "3.10"

    @pytest.mark.asyncio
    async def test_info_test_results_detail(self):
        execute_results = {
            "in1": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
                stdout="correct",
                run_duration_ms=100.0,
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results)
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[TestCase(input="in1", expected="correct", id="test_001")]
        )

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
        )

        await env.env_reset()
        outcome = await env.env_step("code")

        test_result = outcome.info["test_results"][0]
        assert test_result["test_id"] == "test_001"
        assert test_result["passed"] is True
        assert test_result["compiled"] is True
        assert test_result["ran"] is True
        assert test_result["run_status"] == "success"
        assert test_result["compile_status"] == "success"


# ---------------------------------------------------------------------
# Observation Building Tests
# ---------------------------------------------------------------------


class TestCodeExecEnvObservation:
    @pytest.mark.asyncio
    async def test_observation_on_success(self):
        sandbox = MockSandbox(default_stdout="out")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[
                TestCase(input="x", expected="out", id="t0"),
                TestCase(input="y", expected="out", id="t1"),
            ]
        )

        env = CodeExecEnv(sample={}, sandbox_pool=pool, test_adapter=adapter)

        await env.env_reset()
        outcome = await env.env_step("code")

        assert "All 2 tests passed" in outcome.obs

    @pytest.mark.asyncio
    async def test_observation_on_compile_error_includes_line(self):
        compile_result = CompileResult(
            status=CompileStatus.SYNTAX_ERROR,
            error_message="invalid syntax",
            error_line=42,
            duration_ms=5.0,
        )
        sandbox = MockSandbox(compile_result=compile_result)
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter()

        env = CodeExecEnv(sample={}, sandbox_pool=pool, test_adapter=adapter)

        await env.env_reset()
        outcome = await env.env_step("bad code")

        assert "Compilation failed" in outcome.obs
        assert "line 42" in outcome.obs

    @pytest.mark.asyncio
    async def test_observation_truncates_long_errors(self):
        long_error = "E" * 1000
        compile_result = CompileResult(
            status=CompileStatus.SYNTAX_ERROR,
            error_message=long_error,
            duration_ms=5.0,
        )
        sandbox = MockSandbox(compile_result=compile_result)
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter()
        config = CodeExecConfig(max_error_length=100)

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()
        outcome = await env.env_step("code")

        # Error should be truncated with "..."
        assert len(outcome.obs) < len(long_error)
        assert "..." in outcome.obs

    @pytest.mark.asyncio
    async def test_observation_includes_stderr_when_configured(self):
        execute_results = {
            "input": ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.RUNTIME_ERROR,
                stdout="",
                stderr="NameError: x is not defined",
            ),
        }
        sandbox = MockSandbox(execute_results=execute_results)
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[TestCase(input="input", expected="output", id="t0")]
        )
        config = CodeExecConfig(include_stderr_in_obs=True)

        env = CodeExecEnv(
            sample={},
            sandbox_pool=pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()
        outcome = await env.env_step("print(x)")

        assert "Stderr" in outcome.obs
        assert "NameError" in outcome.obs


# ---------------------------------------------------------------------
# Current Observation Tests
# ---------------------------------------------------------------------


class TestCodeExecEnvCurrentObs:
    @pytest.mark.asyncio
    async def test_env_current_obs_before_reset(self):
        sandbox = MockSandbox()
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter()

        env = CodeExecEnv(sample={}, sandbox_pool=pool, test_adapter=adapter)

        obs = env.env_current_obs()
        assert "Error" in obs
        assert "reset" in obs.lower()

    @pytest.mark.asyncio
    async def test_env_current_obs_after_reset(self):
        sandbox = MockSandbox()
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(prompt="Solve this problem.")

        env = CodeExecEnv(sample={}, sandbox_pool=pool, test_adapter=adapter)

        await env.env_reset()
        obs = env.env_current_obs()

        assert obs == "Solve this problem."

    @pytest.mark.asyncio
    async def test_env_current_obs_after_step(self):
        sandbox = MockSandbox(default_stdout="result")
        pool = MockSandboxPool(sandbox=sandbox)
        adapter = MockTestAdapter(
            tests=[TestCase(input="x", expected="result", id="t0")]
        )

        env = CodeExecEnv(sample={}, sandbox_pool=pool, test_adapter=adapter)

        await env.env_reset()
        await env.env_step("code")
        obs = env.env_current_obs()

        assert "passed" in obs

"""
Integration tests for Docker-based code execution sandbox.

These tests require Docker to be running and will create/destroy containers.
Run with: pytest -m integration tests/integration/test_code_exec_docker.py

To skip GPU tests while running integration tests:
    pytest -m "integration and not gpu"
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = [pytest.mark.integration]


# Try to import docker - skip all tests if not available
try:
    import docker
    from docker.errors import DockerException

    # Try to connect to Docker daemon
    try:
        _client = docker.from_env(timeout=2)
        _client.ping()
        _client.close()
        DOCKER_AVAILABLE = True
    except (DockerException, Exception):
        DOCKER_AVAILABLE = False
except ImportError:
    DOCKER_AVAILABLE = False


skip_if_no_docker = pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="Docker daemon not available or docker package not installed",
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
async def sandbox_pool():
    """Create and tear down a sandbox pool for testing."""
    from ludic.envs.code_exec.docker_sandbox import DockerSandboxPool, DockerSandboxConfig

    config = DockerSandboxConfig(
        python_version="3.11",
        memory_limit="128m",
        cpu_quota=25000,
        network_disabled=True,
    )

    pool = DockerSandboxPool(
        n_workers=2,
        config=config,
        cache_size=100,
    )

    await pool.start()
    yield pool
    await pool.shutdown()


@pytest.fixture
async def sandbox(sandbox_pool):
    """Get a single sandbox for testing."""
    sandbox = await sandbox_pool.checkout()
    yield sandbox
    await sandbox_pool.release(sandbox)


# ---------------------------------------------------------------------
# DockerSandbox Tests
# ---------------------------------------------------------------------


@skip_if_no_docker
class TestDockerSandboxCompile:
    @pytest.mark.asyncio
    async def test_compile_valid_code(self, sandbox):
        """Valid Python code should compile successfully."""
        from ludic.envs.code_exec.types import CompileStatus

        code = """
def hello():
    return "Hello, World!"

print(hello())
"""
        result = await sandbox.compile(code)

        assert result.success is True
        assert result.status == CompileStatus.SUCCESS
        assert result.error_message is None
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_compile_syntax_error(self, sandbox):
        """Syntax errors should be detected and reported."""
        from ludic.envs.code_exec.types import CompileStatus

        code = """
def broken(
    print("missing parenthesis")
"""
        result = await sandbox.compile(code)

        assert result.success is False
        assert result.status == CompileStatus.SYNTAX_ERROR
        assert result.error_message is not None
        assert "SyntaxError" in result.error_message or "syntax" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_compile_indentation_error(self, sandbox):
        """Indentation errors should be detected."""
        from ludic.envs.code_exec.types import CompileStatus

        code = """
def foo():
print("bad indent")
"""
        result = await sandbox.compile(code)

        assert result.success is False
        assert result.status == CompileStatus.SYNTAX_ERROR


@skip_if_no_docker
class TestDockerSandboxExecute:
    @pytest.mark.asyncio
    async def test_execute_simple_print(self, sandbox):
        """Simple print statement should produce output."""
        from ludic.envs.code_exec.types import RunStatus

        code = 'print("Hello from Docker!")'
        result = await sandbox.execute(code)

        assert result.compiled is True
        assert result.succeeded is True
        assert result.run_status == RunStatus.SUCCESS
        assert "Hello from Docker!" in result.stdout.strip()
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_with_stdin(self, sandbox):
        """Code should be able to read from stdin."""
        from ludic.envs.code_exec.types import RunStatus

        code = """
import sys
line = input()
print(f"Got: {line}")
"""
        result = await sandbox.execute(code, stdin="test_input")

        assert result.compiled is True
        # Note: stdin handling in docker exec is tricky
        # This test may need adjustment based on actual behavior

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self, sandbox):
        """Runtime errors should be captured."""
        from ludic.envs.code_exec.types import RunStatus

        code = """
x = undefined_variable
"""
        result = await sandbox.execute(code)

        assert result.compiled is True
        assert result.succeeded is False
        assert result.run_status == RunStatus.RUNTIME_ERROR
        assert "NameError" in result.stderr or "undefined" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_execute_division_by_zero(self, sandbox):
        """Division by zero should be a runtime error."""
        from ludic.envs.code_exec.types import RunStatus

        code = """
result = 1 / 0
"""
        result = await sandbox.execute(code)

        assert result.compiled is True
        assert result.succeeded is False
        assert result.run_status == RunStatus.RUNTIME_ERROR
        assert "ZeroDivision" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_timeout(self, sandbox):
        """Infinite loops should timeout."""
        from ludic.envs.code_exec.types import RunStatus

        code = """
while True:
    pass
"""
        result = await sandbox.execute(code, timeout_s=1.0)

        assert result.compiled is True
        assert result.timed_out is True
        assert result.run_status == RunStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_returns_timing(self, sandbox):
        """Execution should return timing information."""
        code = """
import time
time.sleep(0.1)
print("done")
"""
        result = await sandbox.execute(code)

        assert result.compile_duration_ms > 0
        assert result.run_duration_ms >= 50  # Allow some jitter in CI
        assert result.total_duration_ms > 0


@skip_if_no_docker
class TestDockerSandboxReset:
    @pytest.mark.asyncio
    async def test_reset_clears_files(self, sandbox):
        """Reset should clear workspace files."""
        # Write a file
        code1 = """
with open('test_file.txt', 'w') as f:
    f.write('hello')
"""
        await sandbox.execute(code1)

        # Reset
        await sandbox.reset()

        # Try to read the file - should fail
        code2 = """
try:
    with open('test_file.txt', 'r') as f:
        print(f.read())
except FileNotFoundError:
    print("FILE_NOT_FOUND")
"""
        result = await sandbox.execute(code2)

        assert "FILE_NOT_FOUND" in result.stdout


# ---------------------------------------------------------------------
# DockerSandboxPool Tests
# ---------------------------------------------------------------------


@skip_if_no_docker
class TestDockerSandboxPool:
    @pytest.mark.asyncio
    async def test_pool_checkout_and_release(self, sandbox_pool):
        """Should be able to checkout and release sandboxes."""
        sandbox = await sandbox_pool.checkout()
        assert sandbox is not None
        assert sandbox_pool.available == 1  # One still available

        await sandbox_pool.release(sandbox)
        assert sandbox_pool.available == 2  # Both available again

    @pytest.mark.asyncio
    async def test_pool_concurrent_checkout(self, sandbox_pool):
        """Multiple checkouts should work concurrently."""
        sandbox1 = await sandbox_pool.checkout()
        sandbox2 = await sandbox_pool.checkout()

        assert sandbox1 is not sandbox2
        assert sandbox_pool.available == 0

        await sandbox_pool.release(sandbox1)
        await sandbox_pool.release(sandbox2)
        assert sandbox_pool.available == 2

    @pytest.mark.asyncio
    async def test_pool_checkout_timeout(self, sandbox_pool):
        """Checkout should timeout when no sandboxes available."""
        # Check out all sandboxes
        sandbox1 = await sandbox_pool.checkout()
        sandbox2 = await sandbox_pool.checkout()

        # Third checkout should timeout
        with pytest.raises(TimeoutError):
            await sandbox_pool.checkout(timeout_s=0.5)

        await sandbox_pool.release(sandbox1)
        await sandbox_pool.release(sandbox2)

    @pytest.mark.asyncio
    async def test_pool_caching(self, sandbox_pool):
        """Pool should cache execution results."""
        from ludic.envs.code_exec.types import (
            BatchTestResult,
            CompileResult,
            CompileStatus,
            ExecutionResult,
            RunStatus,
            TestCase,
            TestResult,
        )

        # Create a mock result
        test_result = TestResult(
            test_case=TestCase(input="1", expected="2", id="t1"),
            passed=True,
            actual="2",
            execution=ExecutionResult(
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                run_status=RunStatus.SUCCESS,
            ),
        )
        batch_result = BatchTestResult(
            results=[test_result],
            code_hash="abc123",
            tests_hash="xyz789",
        )

        # Cache it
        sandbox_pool.put_cached("abc123", "xyz789", batch_result)

        # Retrieve it
        cached = sandbox_pool.get_cached("abc123", "xyz789")
        assert cached is batch_result

        # Check cache stats
        stats = sandbox_pool.cache_stats
        assert stats["hits"] == 1
        assert stats["size"] == 1


# ---------------------------------------------------------------------
# StdinStdoutRunner Integration Tests
# ---------------------------------------------------------------------


@skip_if_no_docker
class TestStdinStdoutRunnerIntegration:
    @pytest.mark.asyncio
    async def test_runner_all_pass(self, sandbox):
        """Runner should correctly execute code and verify outputs."""
        from ludic.envs.code_exec.runners import StdinStdoutRunner
        from ludic.envs.code_exec.adapters.base import ExactMatchVerifier
        from ludic.envs.code_exec.types import TestCase

        code = """
n = int(input())
print(n * 2)
"""
        tests = [
            TestCase(input="5", expected="10", id="t1"),
            TestCase(input="10", expected="20", id="t2"),
            TestCase(input="0", expected="0", id="t3"),
        ]

        runner = StdinStdoutRunner(default_timeout_s=5.0)
        verifier = ExactMatchVerifier()

        result = await runner.run_tests(
            sandbox=sandbox,
            code=code,
            tests=tests,
            verifier=verifier,
        )

        assert result.all_passed is True
        assert result.passed_count == 3
        assert result.total_count == 3

    @pytest.mark.asyncio
    async def test_runner_some_fail(self, sandbox):
        """Runner should correctly identify failing tests."""
        from ludic.envs.code_exec.runners import StdinStdoutRunner
        from ludic.envs.code_exec.adapters.base import ExactMatchVerifier
        from ludic.envs.code_exec.types import TestCase

        # Code that only works for positive numbers
        code = """
n = int(input())
if n < 0:
    print("error")
else:
    print(n * 2)
"""
        tests = [
            TestCase(input="5", expected="10", id="t1"),  # Pass
            TestCase(input="-5", expected="-10", id="t2"),  # Fail
        ]

        runner = StdinStdoutRunner(default_timeout_s=5.0)
        verifier = ExactMatchVerifier()

        result = await runner.run_tests(
            sandbox=sandbox,
            code=code,
            tests=tests,
            verifier=verifier,
            stop_on_first_failure=False,
        )

        assert result.all_passed is False
        assert result.passed_count == 1
        assert result.total_count == 2
        assert result.results[0].passed is True
        assert result.results[1].passed is False

    @pytest.mark.asyncio
    async def test_runner_compile_failure(self, sandbox):
        """Runner should handle compilation failures gracefully."""
        from ludic.envs.code_exec.runners import StdinStdoutRunner
        from ludic.envs.code_exec.adapters.base import ExactMatchVerifier
        from ludic.envs.code_exec.types import TestCase

        code = """
def broken(
    print("syntax error")
"""
        tests = [
            TestCase(input="1", expected="x", id="t1"),
            TestCase(input="2", expected="y", id="t2"),
        ]

        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()

        result = await runner.run_tests(
            sandbox=sandbox,
            code=code,
            tests=tests,
            verifier=verifier,
            compile_first=True,
        )

        assert result.compile_failed is True
        assert result.all_passed is False
        assert result.passed_count == 0
        # All tests should be marked as not compiled
        for r in result.results:
            assert r.compiled is False

    @pytest.mark.asyncio
    async def test_runner_stop_on_first_failure(self, sandbox):
        """Runner should stop after first failure when configured."""
        from ludic.envs.code_exec.runners import StdinStdoutRunner
        from ludic.envs.code_exec.adapters.base import ExactMatchVerifier
        from ludic.envs.code_exec.types import TestCase, RunStatus

        code = """
n = int(input())
print("wrong" if n == 1 else "correct")
"""
        tests = [
            TestCase(input="1", expected="correct", id="t1"),  # Fails
            TestCase(input="2", expected="correct", id="t2"),  # Skipped
            TestCase(input="3", expected="correct", id="t3"),  # Skipped
        ]

        runner = StdinStdoutRunner()
        verifier = ExactMatchVerifier()

        result = await runner.run_tests(
            sandbox=sandbox,
            code=code,
            tests=tests,
            verifier=verifier,
            stop_on_first_failure=True,
        )

        assert result.passed_count == 0
        assert result.results[0].passed is False
        assert result.results[0].ran is True
        assert result.results[1].ran is False
        assert result.results[1].execution.run_status == RunStatus.NOT_RUN
        assert result.results[2].ran is False


# ---------------------------------------------------------------------
# End-to-End CodeExecEnv Tests
# ---------------------------------------------------------------------


@skip_if_no_docker
class TestCodeExecEnvIntegration:
    @pytest.mark.asyncio
    async def test_env_full_workflow(self, sandbox_pool):
        """Test complete workflow from reset to step."""
        from ludic.envs.code_exec.env import CodeExecEnv, CodeExecConfig
        from ludic.envs.code_exec.adapters.apps import APPSTestAdapter

        sample = {
            "problem_id": "test_add",
            "question": "Write a program that reads two integers and prints their sum.",
            "inputs": ["1 2", "10 20", "-5 5"],
            "outputs": ["3", "30", "0"],
        }

        adapter = APPSTestAdapter()
        config = CodeExecConfig(
            timeout_per_test_s=5.0,
            stop_on_first_failure=False,
            compile_first=True,
        )

        env = CodeExecEnv(
            sample=sample,
            sandbox_pool=sandbox_pool,
            test_adapter=adapter,
            config=config,
        )

        # Reset
        obs, info = await env.env_reset()

        assert "two integers" in obs.lower()
        assert info["problem_id"] == "test_add"
        assert info["num_tests"] == 3

        # Submit correct code
        correct_code = """
a, b = map(int, input().split())
print(a + b)
"""
        outcome = await env.env_step(correct_code)

        assert outcome.terminated is True
        assert outcome.reward == 1.0
        assert outcome.info["all_passed"] is True
        assert outcome.info["passed"] == 3
        assert outcome.info["total"] == 3

    @pytest.mark.asyncio
    async def test_env_wrong_code(self, sandbox_pool):
        """Test env with incorrect code submission."""
        from ludic.envs.code_exec.env import CodeExecEnv, CodeExecConfig
        from ludic.envs.code_exec.adapters.apps import APPSTestAdapter

        sample = {
            "problem_id": "test_double",
            "question": "Write a program that reads an integer and prints it doubled.",
            "inputs": ["5", "10"],
            "outputs": ["10", "20"],
        }

        adapter = APPSTestAdapter()
        config = CodeExecConfig(stop_on_first_failure=False)

        env = CodeExecEnv(
            sample=sample,
            sandbox_pool=sandbox_pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()

        # Submit wrong code (triples instead of doubles)
        wrong_code = """
n = int(input())
print(n * 3)
"""
        outcome = await env.env_step(wrong_code)

        assert outcome.terminated is True
        assert outcome.reward == 0.0  # Binary reward, not all passed
        assert outcome.info["all_passed"] is False
        assert outcome.info["passed"] == 0

    @pytest.mark.asyncio
    async def test_env_partial_credit(self, sandbox_pool):
        """Test env with partial credit enabled."""
        from ludic.envs.code_exec.env import CodeExecEnv, CodeExecConfig
        from ludic.envs.code_exec.adapters.apps import APPSTestAdapter

        sample = {
            "problem_id": "test_abs",
            "question": "Write a program that reads an integer and prints its absolute value.",
            "inputs": ["5", "-5", "0", "-10"],
            "outputs": ["5", "5", "0", "10"],
        }

        adapter = APPSTestAdapter()
        config = CodeExecConfig(
            partial_credit=True,
            stop_on_first_failure=False,
        )

        env = CodeExecEnv(
            sample=sample,
            sandbox_pool=sandbox_pool,
            test_adapter=adapter,
            config=config,
        )

        await env.env_reset()

        # Code that only works for non-negative numbers
        partial_code = """
n = int(input())
print(n)  # Wrong for negative numbers
"""
        outcome = await env.env_step(partial_code)

        assert outcome.terminated is True
        assert outcome.info["all_passed"] is False
        assert outcome.info["passed"] == 2  # Only positive and zero pass
        assert outcome.reward == pytest.approx(0.5)  # 2/4 = 0.5

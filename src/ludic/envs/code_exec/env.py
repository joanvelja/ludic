"""
Main environment for code execution RL tasks.

This environment bridges the world of RL agents and code execution sandboxes,
providing a clean SingleAgentEnv interface for training LLMs to write code.

Key design decisions:
  1. env_reset and env_step are async to support async sandbox operations
  2. The interaction protocol (Phase 6) must detect and await these coroutines
  3. Caching is handled at the pool level but controllable via config
  4. Rich info dict includes all execution metadata for analysis/logging

Note: env_reset and env_step are async methods. The interaction protocol
must detect this and await them. See Phase 6 integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Info, Observation, StepOutcome

from .adapters.base import ExactMatchVerifier, OutputVerifier, TestAdapter
from .runners import CodeRunner, StdinStdoutRunner, compute_hash, hash_tests
from .sandbox import SandboxPool
from .types import BatchTestResult, TestCase


@dataclass
class CodeExecConfig:
    """Configuration for CodeExecEnv behavior."""

    # Execution limits
    timeout_per_test_s: float = 5.0  # efficiency-focused default
    memory_limit_mb: int = 256
    max_tests: Optional[int] = None  # limit number of tests
    stop_on_first_failure: bool = True
    compile_first: bool = True

    # Reward shaping
    partial_credit: bool = False  # reward = fraction passed
    compile_failure_reward: float = -0.1

    # Observations
    include_stderr_in_obs: bool = True
    max_error_length: int = 500

    # Caching
    use_cache: bool = True


class CodeExecEnv(SingleAgentEnv):
    """
    Code execution environment for RL training.

    This environment:
      - Takes a dataset sample containing a problem + test cases
      - Extracts prompt and tests via a TestAdapter
      - Executes submitted code in a Sandbox from a SandboxPool
      - Verifies outputs using an OutputVerifier
      - Computes rewards based on test results
      - Returns rich info dicts for logging/analysis

    The environment is single-step by design: agent submits code once,
    gets results, episode ends. For multi-step refinement, wrap this
    in a meta-environment or use a ReAct-style agent with tool calling.

    Example usage:
        ```python
        pool = await create_sandbox_pool(size=4)
        adapter = APPSAdapter()

        env = CodeExecEnv(
            sample=dataset[0],
            sandbox_pool=pool,
            test_adapter=adapter,
            config=CodeExecConfig(partial_credit=True),
        )

        obs, info = await env.env_reset()
        outcome = await env.env_step(agent_code)
        ```
    """

    def __init__(
        self,
        sample: Dict[str, Any],
        *,
        sandbox_pool: SandboxPool,
        test_adapter: TestAdapter,
        code_runner: Optional[CodeRunner] = None,
        verifier: Optional[OutputVerifier] = None,
        config: Optional[CodeExecConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize the code execution environment.

        Args:
            sample: Dataset sample containing problem and tests
            sandbox_pool: Shared pool of sandboxes for execution
            test_adapter: Adapter to extract prompt/tests from sample
            code_runner: Runner for executing code (default: StdinStdoutRunner)
            verifier: Output verifier (default: ExactMatchVerifier)
            config: Environment configuration (default: CodeExecConfig())
            system_prompt: Optional system prompt for the agent
        """
        super().__init__()

        self._sample = sample
        self._sandbox_pool = sandbox_pool
        self._test_adapter = test_adapter
        self._code_runner = code_runner or StdinStdoutRunner(
            default_timeout_s=config.timeout_per_test_s if config else 5.0,
            memory_limit_mb=config.memory_limit_mb if config else 256,
        )
        self._verifier = verifier or ExactMatchVerifier()
        self._config = config or CodeExecConfig()
        self._system_prompt = system_prompt

        # Episode state (set during reset)
        self._problem_id: Optional[str] = None
        self._prompt: Optional[str] = None
        self._tests: Optional[List[TestCase]] = None
        self._tests_hash: Optional[str] = None
        self._current_obs: Optional[Observation] = None

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        """Return the configured system prompt."""
        return self._system_prompt

    async def env_reset(
        self, *, seed: Optional[int] = None
    ) -> Tuple[Observation, Info]:
        """
        Reset the environment for a new episode.

        Extracts the problem prompt and test cases from the sample,
        but does not checkout a sandbox yet (that happens on step).

        Args:
            seed: Optional random seed (unused in this deterministic env)

        Returns:
            Tuple of (prompt, info) where info contains problem metadata
        """
        # Extract problem components via adapter
        self._problem_id = self._test_adapter.get_problem_id(self._sample)
        self._prompt = self._test_adapter.get_prompt(self._sample)
        self._tests = self._test_adapter.get_tests(self._sample)

        # Handle case where no tests were extracted
        if not self._tests:
            error_msg = f"No tests extracted for problem {self._problem_id}"
            self._current_obs = error_msg
            return self._current_obs, {
                "problem_id": self._problem_id,
                "error": "no_tests_extracted",
            }

        # Apply max_tests limit if configured
        if self._config.max_tests is not None:
            self._tests = self._tests[: self._config.max_tests]

        # Compute tests hash for caching
        self._tests_hash = hash_tests(self._tests)

        # Set current observation to the prompt
        self._current_obs = self._prompt

        # Build info dict with episode metadata
        info: Info = {
            "problem_id": self._problem_id,
            "num_tests": len(self._tests),
            "tests_hash": self._tests_hash,
            "python_version": self._sandbox_pool.python_version,
        }

        return self._current_obs, info

    async def env_step(self, action: str) -> StepOutcome:
        """
        Execute submitted code and return results.

        This is the core of the environment: takes the agent's code,
        runs it through the sandbox, computes rewards, and builds
        rich observations and info dicts.

        Args:
            action: The code submitted by the agent

        Returns:
            StepOutcome with observation, reward, termination flags, and info
        """
        # Sanity check: ensure reset was called
        if self._tests is None or self._tests_hash is None:
            error_obs = "Error: env_reset() must be called before env_step()"
            return StepOutcome(
                obs=error_obs,
                reward=-1.0,
                truncated=False,
                terminated=True,
                info={"error": "reset_not_called"},
            )

        # Handle empty code submission
        if not action.strip():
            error_obs = "Error: Empty code submission"
            return StepOutcome(
                obs=error_obs,
                reward=self._config.compile_failure_reward,
                truncated=False,
                terminated=True,
                info={"error": "empty_code"},
            )

        # Compute code hash for caching
        code = action.strip()
        code_hash = compute_hash(code)

        # Check cache FIRST, before checkout
        result: Optional[BatchTestResult] = None
        cache_hit = False

        if self._config.use_cache:
            result = self._sandbox_pool.get_cached(code_hash, self._tests_hash)
            if result is not None:
                cache_hit = True

        # Only checkout sandbox if cache miss
        if result is None:
            # Checkout sandbox from pool
            sandbox = await self._sandbox_pool.checkout()
            print(f"Checkout sandbox: {sandbox}")

            try:
                print(f"Running tests for code: {code}")
                # Run tests via code runner
                result = await self._code_runner.run_tests(
                    sandbox=sandbox,
                    code=code,
                    tests=self._tests,
                    verifier=self._verifier,
                    stop_on_first_failure=self._config.stop_on_first_failure,
                    compile_first=self._config.compile_first,
                )

                # Cache result if enabled
                if self._config.use_cache:
                    self._sandbox_pool.put_cached(code_hash, self._tests_hash, result)

            finally:
                # Always release sandbox back to pool
                await self._sandbox_pool.release(sandbox)

        # Compute reward based on results
        reward = self._compute_reward(result)

        # Build observation for agent
        obs = self._build_observation(result)
        self._current_obs = obs

        # Build rich info dict for logging/analysis
        info = self._build_info(result, code_hash, cache_hit)

        # Episode ends after single step (single-shot code generation)
        return StepOutcome(
            obs=obs,
            reward=reward,
            truncated=False,
            terminated=True,
            info=info,
        )

    def env_current_obs(self) -> Observation:
        """
        Return the current observation.

        Returns:
            The current observation string
        """
        if self._current_obs is None:
            return "Error: No observation available (call env_reset first)"
        return self._current_obs

    def _compute_reward(self, result: BatchTestResult) -> float:
        """
        Compute reward from test results.

        Reward schemes:
          - partial_credit=False: 1.0 if all passed, 0.0 otherwise
          - partial_credit=True: fraction of tests passed (0.0 to 1.0)
          - Compilation failures get compile_failure_reward

        Args:
            result: Batch test results

        Returns:
            Scalar reward value
        """
        # Compilation failure gets special penalty
        if result.compile_failed:
            return self._config.compile_failure_reward

        # All tests passed
        if result.all_passed:
            return 1.0

        # Partial credit
        if self._config.partial_credit:
            return result.pass_rate

        # Binary reward (all or nothing)
        return 0.0

    def _build_observation(self, result: BatchTestResult) -> str:
        """
        Build observation string from test results.

        The observation provides feedback to the agent about what went wrong,
        including compilation errors, runtime errors, or test failures.

        Args:
            result: Batch test results

        Returns:
            Observation string for the agent
        """
        # All tests passed - success message
        if result.all_passed:
            return (
                f"All {result.total_count} tests passed! "
                f"Total execution time: {result.total_run_ms:.1f}ms"
            )

        # Compilation failed - show compile error
        if result.compile_failed:
            first = result.results[0]
            compile_err = (
                first.execution.compile_result.error_message or "Unknown error"
            )

            # Truncate error if too long
            if len(compile_err) > self._config.max_error_length:
                compile_err = compile_err[: self._config.max_error_length] + "..."

            obs = f"Compilation failed: {compile_err}"

            if first.execution.compile_result.error_line is not None:
                obs += f" (line {first.execution.compile_result.error_line})"

            return obs

        # Some tests failed - show first failure details
        first_failure = result.first_failure
        if first_failure is None:
            # Should never happen, but handle gracefully
            return f"Tests failed: {result.passed_count}/{result.total_count} passed"

        obs_parts = [f"Tests failed: {result.passed_count}/{result.total_count} passed"]

        # Add first failure details
        if first_failure.comparison_details:
            details = first_failure.comparison_details
            if len(details) > self._config.max_error_length:
                details = details[: self._config.max_error_length] + "..."
            obs_parts.append(f"\nFirst failure: {details}")

        # Add stderr if configured and available
        if self._config.include_stderr_in_obs and first_failure.execution.stderr:
            stderr = first_failure.execution.stderr.strip()
            if stderr:
                if len(stderr) > self._config.max_error_length:
                    stderr = stderr[: self._config.max_error_length] + "..."
                obs_parts.append(f"\nStderr: {stderr}")

        return "".join(obs_parts)

    def _build_info(
        self,
        result: BatchTestResult,
        code_hash: str,
        cache_hit: bool,
    ) -> Info:
        """
        Build rich info dict with all execution metadata.

        The info dict is JSON-serializable and includes everything needed
        for logging, analysis, and debugging.

        Args:
            result: Batch test results
            code_hash: Hash of the submitted code
            cache_hit: Whether result came from cache

        Returns:
            Info dict with comprehensive metadata
        """
        # Build per-test result summaries
        test_results = []
        for test_result in result.results:
            test_info = {
                "test_id": test_result.test_case.id,
                "passed": test_result.passed,
                "compiled": test_result.compiled,
                "ran": test_result.ran,
                "run_status": (
                    test_result.execution.run_status.value
                    if test_result.execution.run_status
                    else None
                ),
                "compile_status": test_result.execution.compile_result.status.value,
                "run_duration_ms": test_result.execution.run_duration_ms,
            }

            # Optionally include failure details
            if not test_result.passed and test_result.comparison_details:
                test_info["failure_reason"] = test_result.comparison_details

            test_results.append(test_info)

        # Build complete info dict
        info: Info = {
            # Problem metadata
            "problem_id": self._problem_id,
            "code_hash": code_hash,
            "tests_hash": self._tests_hash,
            # Test results summary
            "passed": result.passed_count,
            "total": result.total_count,
            "all_passed": result.all_passed,
            "pass_rate": result.pass_rate,
            "compile_failed": result.compile_failed,
            # Detailed test results
            "test_results": test_results,
            # Timing
            "timing": {
                "total_compile_ms": result.total_compile_ms,
                "total_run_ms": result.total_run_ms,
                "total_execution_ms": result.total_execution_ms,
            },
            # Cache info
            "cache_hit": cache_hit,
            "cache_stats": self._sandbox_pool.cache_stats,
            # Environment metadata
            "python_version": self._sandbox_pool.python_version,
        }

        return info

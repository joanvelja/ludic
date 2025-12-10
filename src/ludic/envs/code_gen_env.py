# src/ludic/envs/code_gen_env.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple

from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Observation, Info, StepOutcome
from ludic.sandbox.pool import SandboxPool, SandboxHandle
from ludic.sandbox.problems import ProblemBank, Problem
from ludic.sandbox.protocol import ExecutionStatus


@dataclass
class CodeGenRewardConfig:
    """Configurable reward shaping for code generation."""

    all_pass: float = 1.0
    partial_credit_scale: float = 1.0  # Multiplied by pass_rate
    compile_error: float = -0.3
    runtime_error: float = -0.2
    timeout: float = -0.5
    parse_failure: float = -0.1  # If code extraction fails


class CodeGenEnv(SingleAgentEnv):
    """
    Environment for code generation RL.

    Lifecycle:
    - __init__: Captures references to sandbox pool and problem bank
    - env_reset: Acquires sandbox, samples problem
    - env_step: Executes code in sandbox, computes reward
    - close: Releases sandbox back to pool

    The pool is shared across all env instances, but each env instance
    gets exclusive access to one sandbox during its episode.
    """

    def __init__(
        self,
        *,
        sandbox_pool: SandboxPool,
        problem_bank: ProblemBank,
        reward_config: CodeGenRewardConfig = CodeGenRewardConfig(),
        language: str = "python",
        max_attempts: int = 1,  # Multi-turn if > 1
        include_examples_in_prompt: bool = True,
        execution_timeout_s: float = 30.0,
    ):
        super().__init__()
        self._pool = sandbox_pool
        self._problem_bank = problem_bank
        self._reward_config = reward_config
        self._language = language
        self._max_attempts = max_attempts
        self._include_examples = include_examples_in_prompt
        self._exec_timeout = execution_timeout_s

        # Per-episode state
        self._sandbox: Optional[SandboxHandle] = None
        self._current_problem: Optional[Problem] = None
        self._attempt: int = 0
        self._last_result: Optional[str] = None

        # For async operations within sync interface
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for sync-to-async bridge."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run_async(self, coro):
        """Run async code from sync context."""
        loop = self._get_loop()
        if loop.is_running():
            # We're inside an async context - create a task
            # This happens when called from protocol.run() which is async
            future = asyncio.ensure_future(coro, loop=loop)
            # We need to block here, which is tricky inside a running loop
            # Use nest_asyncio or run in thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(lambda: asyncio.run(coro)).result()
        else:
            return loop.run_until_complete(coro)

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return f"""You are an expert {self._language} programmer.
Given a problem description, write a complete, correct solution.
Output ONLY the code, no explanations or markdown formatting.
The code should be a complete, runnable solution."""

    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        """Acquire sandbox, sample problem, return prompt."""

        # Release any existing sandbox
        if self._sandbox is not None:
            self._run_async(self._pool.release(self._sandbox))
            self._sandbox = None

        # Acquire fresh sandbox
        self._sandbox = self._run_async(self._pool.acquire())

        # Sample problem based on seed (deterministic!)
        self._current_problem = self._problem_bank.sample(seed=seed)
        self._attempt = 0
        self._last_result = None

        obs = self._current_problem.to_prompt(include_examples=self._include_examples)

        info: Info = {
            "problem_id": self._current_problem.id,
            "difficulty": self._current_problem.difficulty,
            "tags": self._current_problem.tags,
            "sandbox_idx": self._sandbox.pool_idx,
        }

        return obs, info

    def env_step(self, action: str) -> StepOutcome:
        """Execute code in sandbox and compute reward."""

        if self._sandbox is None or self._current_problem is None:
            raise RuntimeError("env_step called before env_reset")

        self._attempt += 1

        # Extract code from action (in case agent wrapped it in markdown)
        code = self._extract_code(action)

        # Execute in sandbox
        result = self._run_async(
            self._sandbox.execute(
                code=code,
                tests=self._current_problem.tests,
                language=self._language,
                timeout_s=self._exec_timeout,
            )
        )

        # Compute reward
        reward = self._compute_reward(result)

        # Determine if episode is done
        if result.all_passed:
            # Success!
            terminated = True
            obs = f"✅ All {result.tests_total} tests passed!"
        elif self._attempt >= self._max_attempts:
            # Out of attempts
            terminated = True
            obs = self._format_failure_obs(result)
        else:
            # Can try again
            terminated = False
            obs = self._format_failure_obs(result)

        info: Info = {
            "execution_status": result.status.value,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
            "pass_rate": result.pass_rate,
            "attempt": self._attempt,
            "execution_time_ms": result.execution_time_ms,
        }

        if result.compile_output:
            info["compile_output"] = result.compile_output
        if result.stderr:
            info["stderr"] = result.stderr

        return StepOutcome(
            obs=obs,
            reward=reward,
            truncated=False,
            terminated=terminated,
            info=info,
        )

    def env_current_obs(self) -> Observation:
        if self._last_result:
            return self._last_result
        if self._current_problem:
            return self._current_problem.to_prompt(
                include_examples=self._include_examples
            )
        return "No problem loaded."

    def close(self) -> None:
        """Release sandbox back to pool."""
        if self._sandbox is not None:
            self._run_async(self._pool.release(self._sandbox))
            self._sandbox = None

    def _extract_code(self, action: str) -> str:
        """Extract code from potentially markdown-wrapped response."""
        action = action.strip()

        # Handle ```python ... ``` wrapping
        if "```" in action:
            lines = action.split("\n")
            in_code_block = False
            code_lines = []

            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines)

        return action

    def _compute_reward(self, result) -> float:
        """Compute reward from execution result."""
        cfg = self._reward_config

        if result.all_passed:
            return cfg.all_pass

        if result.status == ExecutionStatus.COMPILE_ERROR:
            return cfg.compile_error

        if result.status == ExecutionStatus.TIMEOUT:
            return cfg.timeout

        if result.status == ExecutionStatus.RUNTIME_ERROR:
            # Give partial credit for passing tests
            return cfg.runtime_error + (result.pass_rate * cfg.partial_credit_scale)

        # Partial success
        return (result.pass_rate * cfg.partial_credit_scale) - (
            1 - result.pass_rate
        ) * 0.5

    def _format_failure_obs(self, result) -> str:
        """Format failure for agent to learn from."""
        parts = [f"❌ Failed: {result.tests_passed}/{result.tests_total} tests passed"]

        if result.status == ExecutionStatus.COMPILE_ERROR:
            parts.append(
                f"\nCompilation Error:\n{result.compile_output or 'No details'}"
            )
        elif result.status == ExecutionStatus.TIMEOUT:
            parts.append("\nExecution timed out.")
        elif result.stderr:
            parts.append(f"\nError Output:\n{result.stderr[:500]}")  # Truncate

        # Add failing test details (first few)
        failing = [t for t in result.test_results if not t.passed][:3]
        if failing:
            parts.append("\nFailing tests:")
            for t in failing:
                parts.append(f"  - {t.name}: {t.error_message or 'Failed'}")

        return "\n".join(parts)

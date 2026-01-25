"""PVGProverEnv: Wrapper for prover training with verifier scoring integration.

This module provides a wrapper around SneakyCodeExecEnv that integrates with
the PVG training loop, providing standardized signal extraction and verifier
scoring support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from ludic.types import Info, Observation, StepOutcome


@dataclass
class PVGEnvConfig:
    """Configuration for PVGProverEnv.

    Args:
        attach_raw_signals: If True, attach raw signals to step info
        signal_keys: Keys to extract from sneaky_result and attach to raw_signals
    """

    attach_raw_signals: bool = True
    signal_keys: Tuple[str, ...] = (
        "test_pass_rate",
        "sneaky_certified",
        "similarity_score",
    )


class PVGProverEnv:
    """Wrapper around SneakyCodeExecEnv for PVG prover training.

    This environment wrapper:
    1. Delegates to an inner SneakyCodeExecEnv for actual code execution
    2. Extracts and standardizes signals from step outcomes
    3. Attaches raw_signals dict for consistent credit assignment interface
    4. Supports verifier scoring integration (post-rollout)

    The wrapper preserves the SingleAgentEnv interface while adding
    PVG-specific metadata handling.

    Example:
        ```python
        from ludic.pvg.prover_env import PVGProverEnv

        # Wrap an existing SneakyCodeExecEnv
        inner_env = SneakyCodeExecEnv(...)
        pvg_env = PVGProverEnv(inner_env)

        # Use like a normal env
        obs, info = await pvg_env.env_reset()
        outcome = await pvg_env.env_step(action)

        # Access raw signals
        raw_signals = outcome.info["raw_signals"]
        # {"test_pass_rate": 0.8, "sneaky_certified": True, "similarity_score": 0.3}
        ```
    """

    def __init__(
        self,
        inner_env: Any,
        *,
        config: Optional[PVGEnvConfig] = None,
        problem_id: Optional[str] = None,
    ) -> None:
        """Initialize the PVG prover environment wrapper.

        Args:
            inner_env: The underlying SneakyCodeExecEnv to wrap
            config: Configuration for signal extraction
            problem_id: Optional override for problem ID (uses inner env's if not set)
        """
        self._inner = inner_env
        self._config = config or PVGEnvConfig()
        self._problem_id = problem_id

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        """Return the suggested system prompt from the inner env."""
        return getattr(self._inner, "suggested_sysprompt", None)

    @property
    def problem_id(self) -> str:
        """Return the problem ID."""
        if self._problem_id is not None:
            return self._problem_id
        return getattr(self._inner, "_problem_id", "unknown")

    async def env_reset(
        self, *, seed: Optional[int] = None
    ) -> Tuple[Observation, Info]:
        """Reset the environment for a new episode.

        Returns:
            Tuple of (observation, info) from the inner env
        """
        obs, info = await self._inner.env_reset(seed=seed)

        # Add PVG-specific metadata
        info["pvg_enabled"] = True
        info["problem_id"] = self.problem_id

        return obs, info

    async def env_step(self, action: str) -> StepOutcome:
        """Execute an action and extract PVG signals.

        Args:
            action: JSON string with code and certificate

        Returns:
            StepOutcome with raw_signals attached to info
        """
        outcome = await self._inner.env_step(action)

        if self._config.attach_raw_signals:
            outcome = self._attach_raw_signals(outcome)

        return outcome

    def env_current_obs(self) -> Observation:
        """Return the current observation."""
        return self._inner.env_current_obs()

    def _attach_raw_signals(self, outcome: StepOutcome) -> StepOutcome:
        """Attach raw_signals dict to the outcome info.

        Extracts signals from sneaky_result and puts them in a
        standardized format for credit assignment.
        """
        info = dict(outcome.info)
        sneaky_result = info.get("sneaky_result", {})

        # Build raw_signals from sneaky_result
        raw_signals: Dict[str, Any] = {}
        for key in self._config.signal_keys:
            if key in sneaky_result:
                raw_signals[key] = sneaky_result[key]
            else:
                # Use defaults for missing keys
                if key == "test_pass_rate":
                    raw_signals[key] = 0.0
                elif key == "sneaky_certified":
                    raw_signals[key] = False
                elif key == "similarity_score":
                    raw_signals[key] = None

        info["raw_signals"] = raw_signals
        info["problem_id"] = self.problem_id

        return StepOutcome(
            obs=outcome.obs,
            reward=outcome.reward,
            truncated=outcome.truncated,
            terminated=outcome.terminated,
            info=info,
            trace=outcome.trace,
        )


def extract_raw_signals(outcome: StepOutcome) -> Dict[str, Any]:
    """Extract raw signals from a step outcome.

    Utility function to get raw signals from any step outcome,
    handling both PVGProverEnv and raw SneakyCodeExecEnv outcomes.

    Args:
        outcome: StepOutcome from env step

    Returns:
        Dict with test_pass_rate, sneaky_certified, similarity_score
    """
    info = outcome.info

    # Check for pre-attached raw_signals
    if "raw_signals" in info:
        return info["raw_signals"]

    # Extract from sneaky_result
    sneaky_result = info.get("sneaky_result", {})
    return {
        "test_pass_rate": sneaky_result.get("test_pass_rate", 0.0),
        "sneaky_certified": sneaky_result.get("sneaky_certified", False),
        "similarity_score": sneaky_result.get("similarity_score"),
    }


def wrap_env_for_pvg(
    inner_env: Any,
    *,
    problem_id: Optional[str] = None,
    config: Optional[PVGEnvConfig] = None,
) -> PVGProverEnv:
    """Factory function to wrap an env for PVG training.

    Args:
        inner_env: The underlying SneakyCodeExecEnv
        problem_id: Optional problem ID override
        config: Optional PVGEnvConfig

    Returns:
        PVGProverEnv wrapper
    """
    return PVGProverEnv(
        inner_env,
        config=config,
        problem_id=problem_id,
    )


def create_pvg_env_factory(
    sandbox_pool: Any,
    test_adapter: Any,
    honest_codes: Dict[str, str],
    *,
    sneaky_config: Optional[Any] = None,
    code_exec_config: Optional[Any] = None,
    system_prompt: Optional[str] = None,
    pvg_config: Optional[PVGEnvConfig] = None,
):
    """Create a factory function for PVGProverEnv instances.

    This is useful for RolloutEngine which needs a factory to create
    env instances for each rollout request.

    Args:
        sandbox_pool: Shared pool of sandboxes
        test_adapter: Adapter to extract prompt/tests from samples
        honest_codes: Pre-batched honest solutions {problem_id: code}
        sneaky_config: SneakyConfig for verification
        code_exec_config: CodeExecConfig for execution
        system_prompt: Optional system prompt
        pvg_config: PVGEnvConfig for wrapper

    Returns:
        Factory function that takes a sample and returns a PVGProverEnv
    """
    # Defer import to avoid circular dependency
    from ludic.envs.code_exec.sneaky_env import SneakyCodeExecEnv

    def factory(sample: Dict[str, Any]) -> PVGProverEnv:
        inner_env = SneakyCodeExecEnv(
            sample=sample,
            sandbox_pool=sandbox_pool,
            test_adapter=test_adapter,
            honest_codes=honest_codes,
            config=code_exec_config,
            sneaky_config=sneaky_config,
            system_prompt=system_prompt,
        )
        return PVGProverEnv(inner_env, config=pvg_config)

    return factory

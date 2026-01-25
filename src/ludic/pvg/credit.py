"""PVG Credit Assignment for prover training.

This module provides credit assigners that compose rewards from environment
signals (test_pass_rate, sneaky_certified, similarity_score, verifier_score)
using a configurable RewardStrategy.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from ludic.types import Rollout
from ludic.training.types import RolloutStepKey

from .rewards import RewardStrategy

logger = logging.getLogger(__name__)


@dataclass
class PVGCreditAssigner:
    """Credit assigner for PVG prover training.

    Composes rewards from environment signals using a RewardStrategy, then
    optionally applies group-level normalization for GRPO-style training.

    The reward composition flow:
    1. Extract signals from rollout step info (test_pass_rate, sneaky_certified, etc.)
    2. Get verifier score from attachments or scorer
    3. Apply RewardStrategy.compute() to get final reward
    4. Optionally normalize within groups (GRPO baseline)

    Args:
        reward_strategy: Strategy for composing reward signals
        scorer: Optional VerifierScorer for scoring rollouts
        group_size: If > 1, apply group-level baseline normalization (GRPO-style)
        normalize_adv: Normalize advantages to zero mean/unit std within groups
        positive_only: Clip negative advantages to zero
        eps: Small constant for numerical stability
        role: Role for reward computation ("sneaky" or "honest")
    """

    reward_strategy: RewardStrategy
    scorer: Optional[Any] = None  # VerifierScorer, optional to avoid circular import
    group_size: int = 1
    normalize_adv: bool = False
    positive_only: bool = False
    eps: float = 1e-8
    role: str = "sneaky"

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.role not in ("sneaky", "honest"):
            raise ValueError(f"role must be 'sneaky' or 'honest', got {self.role!r}")

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        """Compute credit assignment for all steps in rollouts.

        Note: This method expects rollouts to have pre-attached verifier scores
        in their metadata (rollout.meta["verifier_score"]). For scoring rollouts
        with the scorer, use score_rollouts() first or use score_and_compute().

        Args:
            rollouts: List of rollouts to process (should have verifier_score attached)

        Returns:
            Mapping from (rollout_id, step_index) to weight
        """
        # First, compute raw rewards for each rollout
        rollout_rewards: Dict[str, float] = {}
        for rollout in rollouts:
            reward = self._compute_rollout_reward(rollout)
            rollout_rewards[rollout.id] = reward

        # If group_size > 1, apply group normalization
        if self.group_size > 1:
            return self._compute_with_groups(rollouts, rollout_rewards)
        else:
            return self._compute_simple(rollouts, rollout_rewards)

    async def score_rollouts(self, rollouts: List[Rollout]) -> List[float]:
        """Score rollouts using the verifier scorer.

        Uses the scorer to compute verifier scores for all rollouts and attaches
        them to the rollout metadata. This is the preferred way to score rollouts
        before calling compute().

        Args:
            rollouts: List of rollouts to score

        Returns:
            List of verifier scores aligned with input rollouts

        Raises:
            ValueError: If no scorer is configured
        """
        if self.scorer is None:
            raise ValueError(
                "No scorer configured. Either set scorer in __init__ or "
                "pre-attach verifier_score to rollouts and call compute() directly."
            )

        # Extract text to score from each rollout
        texts = []
        for rollout in rollouts:
            # Get the full response text (typically concatenated actions)
            if rollout.steps:
                text = "".join(
                    step.action for step in rollout.steps if hasattr(step, "action")
                )
            else:
                text = ""
            texts.append(text)

        # Extract problems from rollout metadata
        problems = [rollout.meta.get("problem", "") for rollout in rollouts]

        # Score in batch using the scorer (pass both solutions and problems)
        scores = await self.scorer.score_batch(solutions=texts, problems=problems)

        # Attach scores to rollouts
        attach_verifier_scores(rollouts, scores)

        return scores

    def score_and_compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        """Score rollouts and compute credit assignment in one step (sync convenience).

        This is a convenience method that runs score_rollouts() followed by compute().
        For async code, prefer calling these separately.

        Args:
            rollouts: List of rollouts to score and process

        Returns:
            Mapping from (rollout_id, step_index) to weight
        """
        # Run the async scorer synchronously
        asyncio.get_event_loop().run_until_complete(self.score_rollouts(rollouts))
        return self.compute(rollouts)

    def _compute_rollout_reward(self, rollout: Rollout) -> float:
        """Compute reward for a single rollout using the reward strategy.

        Extracts signals from the rollout's step info and applies the strategy.
        """
        # Extract signals from the last step's info (where sneaky_result is stored)
        signals = self._extract_signals(rollout)

        # Get verifier score (from attachment or scorer)
        verifier_score = self._get_verifier_score(rollout, signals)

        # Compute reward using strategy
        reward = self.reward_strategy.compute(
            test_pass_rate=signals.get("test_pass_rate", 0.0),
            sneaky_certified=signals.get("sneaky_certified", False),
            similarity_score=signals.get("similarity_score"),
            verifier_score=verifier_score,
            role=self.role,
        )

        return reward

    def _extract_signals(self, rollout: Rollout) -> Dict[str, Any]:
        """Extract sneaky signals from rollout.

        Looks for sneaky_result in:
        1. Rollout metadata
        2. Last step's info
        """
        # Check rollout metadata first
        if "sneaky_result" in rollout.meta:
            return rollout.meta["sneaky_result"]

        # Check last step's info
        if rollout.steps:
            last_step = rollout.steps[-1]
            if hasattr(last_step, "info") and "sneaky_result" in last_step.info:
                return last_step.info["sneaky_result"]

        # Fallback: check for direct fields in metadata
        return {
            "test_pass_rate": rollout.meta.get("test_pass_rate", 0.0),
            "sneaky_certified": rollout.meta.get("sneaky_certified", False),
            "similarity_score": rollout.meta.get("similarity_score"),
        }

    def _get_verifier_score(
        self, rollout: Rollout, signals: Dict[str, Any]
    ) -> Optional[float]:
        """Get verifier score for a rollout.

        Sources (in order of priority):
        1. Pre-attached score in metadata
        2. Score from signals
        3. Score from scorer (if available)
        """
        # Check pre-attached score
        if "verifier_score" in rollout.meta:
            return rollout.meta["verifier_score"]

        # Check signals
        if "verifier_score" in signals:
            return signals["verifier_score"]

        # If scorer available, could score here (but typically done in batch beforehand)
        # self.scorer is optional and scoring is typically done in batch for efficiency

        return None

    def _compute_simple(
        self,
        rollouts: List[Rollout],
        rollout_rewards: Dict[str, float],
    ) -> Dict[RolloutStepKey, float]:
        """Simple credit assignment without grouping.

        Assigns rollout reward to all steps in the rollout.
        """
        out: Dict[RolloutStepKey, float] = {}

        for rollout in rollouts:
            reward = rollout_rewards[rollout.id]
            for step in rollout.steps:
                key: RolloutStepKey = (rollout.id, step.index)
                out[key] = reward

        return out

    def _compute_with_groups(
        self,
        rollouts: List[Rollout],
        rollout_rewards: Dict[str, float],
    ) -> Dict[RolloutStepKey, float]:
        """Compute credit with group-level normalization (GRPO-style).

        Groups rollouts by group_id and applies:
        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
        """
        out: Dict[RolloutStepKey, float] = {}

        # Group by group_id
        groups: Dict[str, List[Rollout]] = defaultdict(list)
        for rollout in rollouts:
            group_id = rollout.meta.get("request_meta", {}).get("group_id")
            if group_id is None:
                # Fallback: use problem_id as group_id
                group_id = rollout.meta.get("problem_id", rollout.id)
            groups[group_id].append(rollout)

        for group_id, group_rollouts in groups.items():
            # Validate group size (flexible for incomplete groups)
            actual_size = len(group_rollouts)
            if actual_size != self.group_size:
                # Warning: group size mismatch, but continue
                pass

            # Get rewards for group
            rewards = torch.tensor(
                [rollout_rewards[r.id] for r in group_rollouts],
                dtype=torch.float32,
            )

            # Compute baseline and advantages
            baseline = rewards.mean()
            advantages = rewards - baseline

            if self.normalize_adv and len(group_rollouts) > 1:
                std = advantages.std(unbiased=False)
                advantages = advantages / (std + self.eps)

            if self.positive_only:
                advantages = torch.clamp(advantages, min=0.0)

            # Assign advantages to all steps
            for i, rollout in enumerate(group_rollouts):
                adv = advantages[i].item()
                for step in rollout.steps:
                    key: RolloutStepKey = (rollout.id, step.index)
                    out[key] = adv

        return out


@dataclass
class SignalExtractor:
    """Utility for extracting PVG signals from rollouts in batch.

    Useful for pre-processing rollouts before credit assignment.
    """

    def extract_all(self, rollouts: List[Rollout]) -> List[Dict[str, Any]]:
        """Extract signals from all rollouts.

        Returns:
            List of signal dicts aligned with input rollouts
        """
        return [self._extract_from_rollout(r) for r in rollouts]

    def _extract_from_rollout(self, rollout: Rollout) -> Dict[str, Any]:
        """Extract signals from a single rollout."""
        # Check rollout metadata first
        if "sneaky_result" in rollout.meta:
            return rollout.meta["sneaky_result"]

        # Check last step's info
        if rollout.steps:
            last_step = rollout.steps[-1]
            if hasattr(last_step, "info") and "sneaky_result" in last_step.info:
                return last_step.info["sneaky_result"]

        return {
            "test_pass_rate": rollout.meta.get("test_pass_rate", 0.0),
            "sneaky_certified": rollout.meta.get("sneaky_certified", False),
            "similarity_score": rollout.meta.get("similarity_score"),
            "verifier_score": rollout.meta.get("verifier_score"),
        }


def attach_verifier_scores(
    rollouts: List[Rollout],
    scores: List[float],
) -> None:
    """Attach verifier scores to rollouts in-place.

    This is called AFTER rollout generation to attach verifier scores.
    The verifier scores are NOT available during generation (preventing leakage).

    Args:
        rollouts: Rollouts to modify
        scores: Scores aligned with rollouts

    Raises:
        ValueError: If lengths don't match
    """
    if len(rollouts) != len(scores):
        raise ValueError(
            f"Length mismatch: {len(rollouts)} rollouts vs {len(scores)} scores"
        )

    logger.debug(f"[VERIFIER SCORING] Attaching {len(scores)} scores to rollouts")
    logger.debug("  Note: Scores attached POST-rollout generation (no leakage during generation)")

    for i, (rollout, score) in enumerate(zip(rollouts, scores)):
        rollout.meta["verifier_score"] = score
        if i < 3:
            logger.debug(f"  Rollout {rollout.id}: score={score:.4f}")

    if len(rollouts) > 3:
        logger.debug(f"  ... and {len(rollouts) - 3} more")

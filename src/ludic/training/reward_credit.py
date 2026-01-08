"""Reward model credit assignment for RL training.

This module provides credit assigners that incorporate reward model scores
into the advantage/weight computation for policy gradient training.

ORM vs PRM Credit Assignment
============================

**Current Implementation: ORM (Outcome Reward Model)**
    Uses a single sequence-level RM score per rollout, stored in rollout.meta["rm_score"].
    The same RM score is applied to ALL steps in the rollout (outcome-level credit).

    This is appropriate when:
    - The RM evaluates the final answer quality
    - All actions in a successful trajectory deserve equal credit
    - You're using GRPO/REINFORCE with group-normalized advantages

**Future Extension: PRM (Process Reward Model)**
    PRMs provide per-step scores, enabling finer-grained credit assignment.
    Each step would have its own score in step.meta["prm_score"].

    To add PRM support:

    1. Add "prm_step" mode to CombineMode:
       CombineMode = Literal["replace", "add", "multiply", "weighted", "bonus", "prm_step"]

    2. Create a PRMCreditAssigner (example below) or extend RewardModelCreditAssigner
       to handle per-step scores from step.meta instead of rollout.meta.

    Example PRMCreditAssigner (future):

        @dataclass
        class PRMCreditAssigner:
            '''Per-step credit using Process Reward Model scores.

            Reads prm_score from step.meta (set by PRMScorer).
            Combines with inner assigner or uses PRM scores directly.
            '''
            mode: Literal["replace", "add", "multiply"] = "replace"
            prm_score_key: str = "prm_score"
            prm_coeff: float = 1.0
            inner_assigner: Optional[CreditAssigner] = None

            def compute(self, rollouts: List[Rollout]) -> Dict[RolloutStepKey, float]:
                out = {}
                base_weights = self.inner_assigner.compute(rollouts) if self.inner_assigner else {}

                for rollout in rollouts:
                    for step in rollout.steps:
                        key = (rollout.id, step.index)
                        prm_score = step.meta.get(self.prm_score_key, 0.0)
                        base = base_weights.get(key, 1.0)

                        if self.mode == "replace":
                            out[key] = prm_score
                        elif self.mode == "add":
                            out[key] = base + self.prm_coeff * prm_score
                        elif self.mode == "multiply":
                            out[key] = base * prm_score
                return out

    3. Create PRMScorer in reward_scorer.py that:
       - Formats rollouts with step boundaries
       - Calls client.score() with PoolingType.STEP
       - Stores per_step_scores in each step.meta
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from ludic.types import Rollout
from ludic.training.types import CreditAssigner, RolloutStepKey

logger = logging.getLogger(__name__)


# ORM modes - all use sequence-level RM score
# Future: add "prm_step" for per-step PRM scoring
CombineMode = Literal["replace", "add", "multiply", "weighted", "bonus"]


@dataclass
class RewardModelCreditAssigner:
    """Credit assigner that uses learned reward model scores.

    Reads pre-computed RM scores from rollout.meta[rm_score_key].
    Scores must be computed beforehand by RewardModelScorer.

    Combine modes:
    - "replace": Use RM score instead of env reward (ignores inner_assigner)
    - "add": Add RM score to inner_assigner weight
    - "multiply": Multiply inner_assigner weight by RM score
    - "weighted": alpha * base_weight + (1-alpha) * rm_score (linear interpolation)
    - "bonus": inner_weight + rm_coeff * rm_score (additive bonus)

    Args:
        mode: How to combine RM scores with base weights. Default "bonus".
        rm_coeff: Coefficient for RM score in "bonus" mode. Default 1.0.
        alpha: Weight for env reward in "weighted" mode. Default 0.5.
        rm_score_key: Key in rollout.meta where RM score is stored. Default "rm_score".
        inner_assigner: Optional inner CreditAssigner (e.g., GroupNormalizedReturn for GRPO).
            If None, base weights are 1.0 for all steps.

    Example:
        # GRPO with RM bonus
        inner = GroupNormalizedReturn(group_size=8)
        credit_assigner = RewardModelCreditAssigner(
            mode="bonus",
            rm_coeff=0.1,
            inner_assigner=inner,
        )

        # Pure RM (replace env rewards entirely)
        credit_assigner = RewardModelCreditAssigner(mode="replace")
    """

    mode: CombineMode = "bonus"
    rm_coeff: float = 1.0
    alpha: float = 0.5
    rm_score_key: str = "rm_score"
    inner_assigner: Optional[CreditAssigner] = None

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        """Compute per-step weights using RM scores.

        For each rollout, retrieves the RM score from rollout.meta[rm_score_key]
        and combines it with either the inner_assigner's weights or uniform weights.

        Args:
            rollouts: List of Rollout objects with RM scores in metadata.

        Returns:
            Dict mapping (rollout_id, step_index) to scalar weight.

        Raises:
            ValueError: If mode is unknown.
        """
        out: Dict[RolloutStepKey, float] = {}

        # Get base weights from inner assigner or use uniform weights
        if self.inner_assigner is not None:
            base_weights = self.inner_assigner.compute(rollouts)
        else:
            base_weights = {
                (r.id, s.index): 1.0
                for r in rollouts
                for s in r.steps
            }

        # Combine with RM scores
        for rollout in rollouts:
            rm_score = rollout.meta.get(self.rm_score_key)
            if rm_score is None:
                logger.warning(
                    f"Rollout {rollout.id} missing '{self.rm_score_key}' in meta, using 0.0. "
                    f"Did you forget to run RewardModelScorer?"
                )
                rm_score = 0.0

            for step in rollout.steps:
                key: RolloutStepKey = (rollout.id, step.index)
                base = base_weights[key]

                if self.mode == "replace":
                    # Pure RM: ignore env reward and inner weights
                    combined = rm_score
                elif self.mode == "add":
                    # Additive: base + rm
                    combined = base + rm_score
                elif self.mode == "multiply":
                    # Multiplicative: base * rm
                    combined = base * rm_score
                elif self.mode == "weighted":
                    # Linear interpolation between base weight and rm score
                    combined = self.alpha * base + (1.0 - self.alpha) * rm_score
                elif self.mode == "bonus":
                    # Additive bonus with coefficient
                    combined = base + self.rm_coeff * rm_score
                else:
                    raise ValueError(f"Unknown combine mode: {self.mode}")

                out[key] = combined

        return out

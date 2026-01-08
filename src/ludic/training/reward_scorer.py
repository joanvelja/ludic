"""Reward model scoring for rollouts.

This module provides utilities for scoring rollouts with reward models
before credit assignment. Currently supports ORM (sequence-level) scoring.

ORM vs PRM Scoring
==================

**Current Implementation: ORM (Outcome Reward Model)**
    - Formats entire rollout as single text string
    - Gets one scalar score per rollout
    - Stores score in rollout.meta["rm_score"]
    - Used by RewardModelCreditAssigner

**Future Extension: PRM (Process Reward Model)**
    PRMs need to track step boundaries within the formatted text to associate
    per-step scores with the correct Step objects.

    To add PRM support, create a PRMScorer with:

    1. Step boundary tracking in _format_rollout_with_steps():
        def _format_rollout_with_steps(self, rollout: Rollout) -> tuple[str, List[int]]:
            '''Format rollout and return (text, step_end_positions).

            Returns:
                text: Concatenated rollout text
                step_boundaries: Token positions where each step ends
            '''
            parts = []
            boundaries = []
            current_pos = 0

            for step in rollout.steps:
                if step.prev_obs:
                    parts.append(str(step.prev_obs))
                if step.action:
                    parts.append(str(step.action))
                # Track where this step ends in the concatenated text
                step_text = "\\n".join(parts[len(boundaries):])
                current_pos += len(step_text)  # Approximate - use tokenizer for accuracy
                boundaries.append(current_pos)

            return "\\n".join(parts), boundaries

    2. Per-step scoring method:
        async def score_prm(self, rollouts: List[Rollout]) -> None:
            '''Score rollouts with PRM and store per-step scores.

            Modifies steps in-place: step.meta[score_key] = float
            '''
            for rollout in rollouts:
                text, boundaries = self._format_rollout_with_steps(rollout)
                # Call client with PoolingType.STEP
                request = ScoringRequest.from_list(
                    model=self.model_name,
                    inputs=[text],
                    pooling_type=PoolingType.STEP,
                    # step_boundaries=(tuple(boundaries),),  # Future field
                )
                response, _ = await self.client.score(request)
                # Distribute per_step_scores to step.meta
                for step, score in zip(rollout.steps, response.per_step_scores[0]):
                    step.meta[self.score_key] = score
"""

from __future__ import annotations

from typing import List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ludic.types import Rollout


class RewardModelClient(Protocol):
    """Protocol for reward model clients.

    Any client implementing this protocol can be used with RewardModelScorer.
    VLLMClient implements this via its score_batch() method.
    """
    async def score_batch(self, inputs: List[str], *, normalize: bool = True) -> List[float]:
        ...


class RewardModelScorer:
    """Async pre-processor that scores rollouts via reward model.

    Adds RM scores to rollout metadata before credit assignment.
    Called by RolloutEngine.generate_batch() if provided.
    """

    def __init__(
        self,
        client: RewardModelClient,
        score_key: str = "rm_score",
        batch_size: int = 64,
    ):
        self.client = client
        self.score_key = score_key
        self.batch_size = batch_size

    async def score(self, rollouts: List[Rollout]) -> None:
        """Score rollouts and store results in metadata.

        Modifies rollouts in-place: rollout.meta[score_key] = float
        """
        # Format rollouts for scoring
        texts = [self._format_rollout(r) for r in rollouts]

        # Batch score
        all_scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            scores = await self.client.score_batch(batch)
            all_scores.extend(scores)

        # Store in metadata
        for rollout, score in zip(rollouts, all_scores):
            rollout.meta[self.score_key] = score

    def _format_rollout(self, rollout: Rollout) -> str:
        """Format rollout for scoring (prompt + completion)."""
        # Concatenate all observations and actions
        parts = []
        for step in rollout.steps:
            if step.prev_obs:
                parts.append(str(step.prev_obs))
            if step.action:
                parts.append(str(step.action))
        return "\n".join(parts)

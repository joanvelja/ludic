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

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ludic.types import Rollout

logger = logging.getLogger(__name__)


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

    IMPORTANT: If the RM was trained with chat templates, set use_chat_template=True
    and provide a tokenizer to ensure scoring uses the same format. Mismatched
    formatting between training and inference will degrade RM quality.
    """

    def __init__(
        self,
        client: RewardModelClient,
        score_key: str = "rm_score",
        batch_size: int = 64,
        use_chat_template: bool = False,
        tokenizer: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        render_fn: Optional[Callable[[Rollout], str]] = None,
    ):
        """Initialize the reward model scorer.

        Args:
            client: Reward model client implementing score_batch().
            score_key: Metadata key for storing scores. Default: "rm_score".
            batch_size: Batch size for scoring requests. Default: 64.
            use_chat_template: If True, format rollouts using chat template.
                Requires tokenizer with apply_chat_template(). Default: False.
            tokenizer: Tokenizer with apply_chat_template() method.
                Required if use_chat_template=True.
            system_prompt: Optional system prompt to include in chat template.
                Only used when use_chat_template=True.
            render_fn: Optional custom function for formatting rollouts.
                Takes a Rollout and returns a formatted string.

        Formatting priority (first match wins):
            1. render_fn: If provided, gives full control over formatting.
            2. use_chat_template=True: Uses tokenizer's chat template.
            3. Raw concatenation: Legacy default (obs/action text joined).
        """
        if use_chat_template and tokenizer is None:
            raise ValueError(
                "use_chat_template=True requires a tokenizer with apply_chat_template(). "
                "Provide a tokenizer from an instruction-tuned model."
            )

        if use_chat_template and not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "use_chat_template=True requires a tokenizer with apply_chat_template(). "
                "The provided tokenizer does not have this method."
            )

        self.client = client
        self.score_key = score_key
        self.batch_size = batch_size
        self.use_chat_template = use_chat_template
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self._render_fn = render_fn

        if render_fn is not None:
            logger.info("RewardModelScorer using custom render_fn for formatting")
        elif use_chat_template:
            logger.info("RewardModelScorer using chat template for formatting")
        else:
            logger.debug("RewardModelScorer using raw text concatenation")

    async def score(self, rollouts: List[Rollout]) -> None:
        """Score rollouts and store results in metadata.

        Modifies rollouts in-place: rollout.meta[score_key] = float
        """
        # Format rollouts for scoring
        texts = [self._format_rollout(r) for r in rollouts]

        # Batch score in parallel (assumes client can handle concurrent requests)
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        results = await asyncio.gather(
            *[self.client.score_batch(batch) for batch in batches]
        )
        all_scores = [score for batch_scores in results for score in batch_scores]

        # Store in metadata
        for rollout, score in zip(rollouts, all_scores):
            rollout.meta[self.score_key] = score

    def _format_rollout(self, rollout: Rollout) -> str:
        """Format rollout for scoring (prompt + completion).

        Dispatch order:
            1. render_fn (if provided) - full custom control
            2. use_chat_template=True - chat template formatting
            3. Raw text concatenation (legacy default)
        """
        if self._render_fn is not None:
            return self._render_fn(rollout)
        if self.use_chat_template:
            return self._format_rollout_chat_template(rollout)
        return self._format_rollout_raw(rollout)

    def _format_rollout_raw(self, rollout: Rollout) -> str:
        """Format rollout using raw text concatenation (legacy)."""
        parts = []
        for step in rollout.steps:
            if step.prev_obs:
                parts.append(str(step.prev_obs))
            if step.action:
                parts.append(str(step.action))
        return "\n".join(parts)

    def _extract_prompt_completion(self, rollout: Rollout) -> tuple[str, str]:
        """Extract prompt (observations) and completion (actions) from rollout.

        This is a legacy formatter used when no chat messages are available.
        It concatenates all observations into a single prompt string and all
        actions into a single completion string. For multi-turn chat scoring,
        prefer _extract_chat_messages() which preserves turn boundaries.
        """
        prompt_parts = []
        completion_parts = []
        for step in rollout.steps:
            if step.prev_obs:
                prompt_parts.append(str(step.prev_obs))
            if step.action:
                completion_parts.append(str(step.action))

        return "\n".join(prompt_parts), "\n".join(completion_parts)

    def _ensure_system_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not self.system_prompt:
            return messages
        if messages and messages[0].get("role") == "system":
            return messages
        return [{"role": "system", "content": self.system_prompt}] + list(messages)

    def _extract_chat_messages(self, rollout: Rollout) -> Optional[List[Dict[str, str]]]:
        """Extract full chat messages for scoring.

        Priority:
        1) Use chat_prompt_messages/chat_completion from the last step that has them
           (preserves multi-turn structure as seen by the model).
        2) Fallback to alternating user/assistant messages built from step.prev_obs
           and step.action (best-effort multi-turn reconstruction).
        """
        # Prefer the most recent step with explicit chat messages.
        for step in reversed(rollout.steps):
            info = step.info or {}
            chat_messages = info.get("chat_prompt_messages")
            if isinstance(chat_messages, list) and chat_messages:
                completion_msg = info.get("chat_completion")
                if isinstance(completion_msg, dict) and completion_msg:
                    completion = completion_msg
                else:
                    completion = {"role": "assistant", "content": step.action}
                full_messages = list(chat_messages) + [completion]
                return self._ensure_system_prompt(full_messages)

        # Fallback: construct a best-effort multi-turn transcript.
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for step in rollout.steps:
            if step.prev_obs:
                messages.append({"role": "user", "content": str(step.prev_obs)})
            if step.action:
                messages.append({"role": "assistant", "content": str(step.action)})
        return messages or None

    def _format_rollout_chat_template(self, rollout: Rollout) -> str:
        """Format rollout using chat template.

        Extracts chat messages from rollout (preserving multi-turn structure
        when available) and formats using the tokenizer's chat template.
        """
        # Prefer explicit chat messages captured during rollout (multi-turn safe).
        chat_messages = self._extract_chat_messages(rollout)
        if chat_messages:
            return self.tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt=False,
                tokenize=False,
            )

        # Legacy fallback: flatten obs/actions into a single user/assistant turn.
        from ludic.training.chat_template_utils import format_chat_template_text

        prompt, completion = self._extract_prompt_completion(rollout)
        return format_chat_template_text(
            self.tokenizer,
            prompt,
            completion,
            system_prompt=self.system_prompt,
        )

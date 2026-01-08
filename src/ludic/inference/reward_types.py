"""Reward model types for vLLM-backed scoring.

This module defines the request/response types for reward model inference.
Currently supports Outcome Reward Models (ORM) with sequence-level scoring.

ORM vs PRM (Outcome vs Process Reward Models)
=============================================

**Current Implementation: ORM (Sequence-Level)**
    - One scalar score per sequence
    - Uses PoolingType.LAST, MEAN, or CLS to reduce hidden states → single score
    - Score stored in rollout.meta["rm_score"]
    - Works with RewardModelCreditAssigner for training integration

**Future Extension: PRM (Per-Step / Token-Level)**
    Process Reward Models score *intermediate reasoning steps*, not just final answers.
    This is useful for:
    - Math reasoning (score each step of a proof)
    - Code generation (score each function/block)
    - Chain-of-thought (score each reasoning step)

    To add PRM support, extend this module with:

    1. PoolingType.STEP implementation:
       - Requires step boundary detection (e.g., newline, special tokens)
       - Returns List[float] instead of single float per sequence

    2. ScoringRequest extensions (commented below):
       - step_boundaries: Token indices marking step ends
       - return_per_token: Return scores at every token position

    3. ScoringResponse extensions:
       - per_step_scores: List[List[float]] - scores for each step per input

    4. Server-side changes (vllm_reward_server.py):
       - Parse step_boundaries from request
       - Pool hidden states per-step instead of per-sequence
       - Return structured per-step scores

    Example future PRM usage:
        request = ScoringRequest.from_list(
            model="prm-model",
            inputs=["Step 1: x=2\\nStep 2: y=x+3\\nAnswer: 5"],
            pooling_type=PoolingType.STEP,
            # step_boundaries=[10, 25],  # Future field
        )
        response = await client.score(request)
        # response.per_step_scores = [[0.95, 0.87, 0.92]]  # Future field
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class RewardModelTrainingMode(Enum):
    """Training modes for reward models.

    Determines which weights are synced to inference server.
    """

    HEAD_ONLY = "head_only"  # Only classification head (score.weight, etc.)
    LORA = "lora"  # Merged LoRA adapters + head
    FULL = "full"  # Complete backbone + head


class PoolingType(Enum):
    """Pooling strategy for reducing hidden states to scores.

    For ORM (sequence-level), these reduce the full sequence to a single scalar.
    For PRM (future), STEP mode would return multiple scores per sequence.

    Attributes:
        LAST: Use last token's hidden state (default for causal LMs like Llama/Qwen).
        MEAN: Mean pool over all token positions.
        CLS: Use first token (BERT/RoBERTa style, requires [CLS] token).
        STEP: Per-step pooling for Process Reward Models (NOT YET IMPLEMENTED).
              Implementation would require:
              - step_boundaries in ScoringRequest
              - Server-side per-step hidden state extraction
              - Modified ScoringResponse with per_step_scores
    """

    LAST = "last"
    MEAN = "mean"
    CLS = "cls"
    STEP = "step"  # Reserved for PRM - not implemented


@dataclass(frozen=True)
class ScoringRequest:
    """Request for reward model scoring.

    Currently supports ORM (sequence-level) scoring. For future PRM support,
    see commented fields below.

    Attributes:
        model: Model identifier for scoring.
        inputs: List of text strings to score.
        pooling_type: Strategy for computing sequence-level scores.
        normalize: Whether to normalize scores (e.g., sigmoid).
        n_labels: Number of output heads (for multi-objective RMs).

    Future PRM fields (uncomment when implementing):
        step_boundaries: Optional[tuple[tuple[int, ...], ...]]
            For each input, token indices marking reasoning step boundaries.
            Example: ((10, 25, 40),) for 3 steps ending at positions 10, 25, 40.
        return_per_token: bool = False
            If True, return a score at every token position (dense supervision).
    """

    model: str
    inputs: tuple[str, ...]  # Frozen requires hashable; use tuple instead of list
    pooling_type: PoolingType = PoolingType.LAST
    normalize: bool = True
    n_labels: int = 1

    # === Future PRM fields (uncomment when implementing) ===
    # step_boundaries: Optional[tuple[tuple[int, ...], ...]] = None
    # return_per_token: bool = False

    @classmethod
    def from_list(
        cls,
        model: str,
        inputs: List[str],
        *,
        pooling_type: PoolingType = PoolingType.LAST,
        normalize: bool = True,
        n_labels: int = 1,
    ) -> ScoringRequest:
        """Create a ScoringRequest from a list of inputs."""
        return cls(
            model=model,
            inputs=tuple(inputs),
            pooling_type=pooling_type,
            normalize=normalize,
            n_labels=n_labels,
        )


@dataclass
class ScoringResponse:
    """Response from reward model scoring.

    Currently returns one score per input (ORM). For future PRM support,
    see commented fields below.

    Attributes:
        scores: List of scalar scores, one per input (ORM mode).
        model: Model identifier that produced the scores.
        usage: Token usage statistics (e.g., {"prompt_tokens": 100}).
        metadata: Additional response metadata (e.g., latency, version).

    Future PRM fields (uncomment when implementing):
        per_step_scores: Optional[List[List[float]]]
            For each input, a list of scores for each reasoning step.
            Only populated when pooling_type=STEP.
        per_token_scores: Optional[List[List[float]]]
            For each input, a score at every token position.
            Only populated when return_per_token=True.
    """

    scores: List[float]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # === Future PRM fields (uncomment when implementing) ===
    # per_step_scores: Optional[List[List[float]]] = None
    # per_token_scores: Optional[List[List[float]]] = None

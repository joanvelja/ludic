"""Batched verifier scoring for PVG training.

This module provides efficient batched inference for scoring code solutions
using a reward model verifier, with caching and batching optimizations.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from ludic.inference.reward_types import PoolingType, ScoringRequest
from ludic.inference.vllm_client import VLLMClient

__all__ = [
    "RewardModelClient",
    "VerifierScorer",
    "VLLMRewardModelClient",
    "MockRewardModelClient",
    "compute_cache_key",
]


@runtime_checkable
class RewardModelClient(Protocol):
    """Protocol for reward model inference clients.

    Clients should implement async scoring of prompt/completion pairs.
    """

    async def score(
        self,
        prompts: List[str],
        completions: List[str],
    ) -> List[float]:
        """Score completions for given prompts.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings (may be empty for RM-only scoring)

        Returns:
            List of scalar scores (one per prompt/completion pair)
        """
        ...


def compute_cache_key(solution: str, problem: str, namespace: Optional[str] = None) -> str:
    """Compute a deterministic cache key for a (solution, problem) pair.

    Args:
        solution: Solution code string
        problem: Problem description string

    Returns:
        16-character hex digest for cache lookup
    """
    prefix = f"{namespace}\n---\n" if namespace else ""
    content = f"{prefix}{problem}\n---\n{solution}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class VerifierScorer:
    """Batched verifier inference for PVG training.

    Scores solutions using a reward model verifier, with caching and batching
    for efficiency.

    Args:
        client: Reward model client for inference
        batch_size: Maximum batch size for inference calls (default: 32)
        cache_enabled: Whether to cache scores (default: True)
        cache_max_size: Optional max cache size (LRU eviction). None = unbounded.
        cache_ttl_s: Optional TTL (seconds) for cached entries. None = no TTL.
        cache_namespace: Optional namespace to separate cache keys (e.g., model version).
        prompt_template: Template for formatting (problem, solution) into prompt.
            Must contain {problem} and {solution} placeholders.

    Example:
        ```python
        from ludic.pvg.scoring import VerifierScorer

        # Create scorer with vLLM reward model client
        scorer = VerifierScorer(
            client=my_rm_client,
            batch_size=16,
            cache_enabled=True
        )

        # Score a batch of solutions
        scores = await scorer.score_batch(
            solutions=["def add(a,b): return a+b", "def mul(a,b): return a*b"],
            problems=["Write add function", "Write multiply function"]
        )
        ```
    """

    client: RewardModelClient
    batch_size: int = 32
    cache_enabled: bool = True
    cache_max_size: Optional[int] = None
    cache_ttl_s: Optional[float] = None
    cache_namespace: Optional[str] = None
    prompt_template: str = (
        "Problem:\n{problem}\n\n"
        "Solution:\n{solution}\n\n"
        "Is this solution helpful and correct?"
    )

    # Private cache: cache_key -> (score, timestamp_s)
    _cache: "OrderedDict[str, Tuple[float, float]]" = field(
        default_factory=OrderedDict, repr=False
    )
    _cache_hits: int = field(default=0, repr=False)
    _cache_misses: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.cache_max_size is not None and self.cache_max_size <= 0:
            raise ValueError(
                f"cache_max_size must be positive or None, got {self.cache_max_size}"
            )
        if self.cache_ttl_s is not None and self.cache_ttl_s <= 0:
            raise ValueError(
                f"cache_ttl_s must be positive or None, got {self.cache_ttl_s}"
            )
        if self.cache_namespace is None:
            self.cache_namespace = hashlib.sha256(
                self.prompt_template.encode()
            ).hexdigest()[:8]

    async def score_batch(
        self,
        solutions: List[str],
        problems: List[str],
    ) -> List[float]:
        """Score a batch of solutions.

        Args:
            solutions: List of solution code strings
            problems: List of problem descriptions (must match solutions length)

        Returns:
            List of verifier scores (one per solution)

        Raises:
            ValueError: If solutions and problems have different lengths
        """
        if len(solutions) != len(problems):
            raise ValueError(
                f"solutions and problems must have same length, "
                f"got {len(solutions)} and {len(problems)}"
            )

        if not solutions:
            return []

        # Separate cached vs uncached items
        scores: List[Optional[float]] = [None] * len(solutions)
        uncached_indices: List[int] = []
        uncached_solutions: List[str] = []
        uncached_problems: List[str] = []

        now = time.monotonic()
        for i, (sol, prob) in enumerate(zip(solutions, problems)):
            cache_key = compute_cache_key(sol, prob, namespace=self.cache_namespace)
            cached = self._get_cached(cache_key, now=now)
            if self.cache_enabled and cached is not None:
                self._cache_hits += 1
                scores[i] = cached
            else:
                self._cache_misses += 1
                uncached_indices.append(i)
                uncached_solutions.append(sol)
                uncached_problems.append(prob)

        # Score uncached items in batches
        if uncached_indices:
            uncached_scores = await self._score_uncached(
                uncached_solutions, uncached_problems
            )

            # Fill in results and update cache
            for idx, sol, prob, score in zip(
                uncached_indices, uncached_solutions, uncached_problems, uncached_scores
            ):
                scores[idx] = score
                if self.cache_enabled:
                    cache_key = compute_cache_key(sol, prob, namespace=self.cache_namespace)
                    self._set_cached(cache_key, score, now=now)

        # Type narrowing: all scores are now filled (use 'is not None' to preserve 0.0 scores)
        return [s for s in scores if s is not None]  # type: ignore[misc]

    async def _score_uncached(
        self,
        solutions: List[str],
        problems: List[str],
    ) -> List[float]:
        """Score uncached items, splitting into batches.

        Args:
            solutions: Solutions to score
            problems: Corresponding problems

        Returns:
            Scores in same order as inputs
        """
        all_scores: List[float] = []

        for i in range(0, len(solutions), self.batch_size):
            batch_solutions = solutions[i : i + self.batch_size]
            batch_problems = problems[i : i + self.batch_size]

            # Format prompts using template
            prompts = [
                self.prompt_template.format(problem=p, solution=s)
                for p, s in zip(batch_problems, batch_solutions)
            ]

            # Get scores from client (completions empty for reward model)
            batch_scores = await self.client.score(prompts, [""] * len(prompts))
            all_scores.extend(batch_scores)

        return all_scores

    def clear_cache(self) -> int:
        """Clear the score cache.

        Returns:
            Number of cache entries cleared
        """
        n = len(self._cache)
        self._cache.clear()
        return n

    def cache_size(self) -> int:
        """Return current cache size.

        Returns:
            Number of cached (solution, problem) pairs
        """
        return len(self._cache)

    def cache_stats(self) -> Dict[str, int]:
        """Return cache hit/miss stats and size."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
        }

    def _get_cached(self, cache_key: str, *, now: float) -> Optional[float]:
        """Lookup cached score, honoring TTL and LRU order."""
        if not self.cache_enabled:
            return None
        if cache_key not in self._cache:
            return None
        score, timestamp = self._cache[cache_key]
        if self.cache_ttl_s is not None and (now - timestamp) > self.cache_ttl_s:
            del self._cache[cache_key]
            return None
        # LRU refresh
        self._cache.move_to_end(cache_key)
        return score

    def _set_cached(self, cache_key: str, score: float, *, now: float) -> None:
        """Insert into cache with LRU eviction."""
        self._cache[cache_key] = (score, now)
        self._cache.move_to_end(cache_key)
        if self.cache_max_size is not None:
            while len(self._cache) > self.cache_max_size:
                self._cache.popitem(last=False)


class VLLMRewardModelClient:
    """Reward model client backed by a vLLM reward server.

    Wraps a VLLMClient and uses its /score endpoint for batch scoring.
    """

    def __init__(self, client: VLLMClient, *, model: str) -> None:
        self._client = client
        self._model = model

    async def score(
        self,
        prompts: List[str],
        _completions: List[str],
    ) -> List[float]:
        # completions are ignored; prompts already include solution text.
        request = ScoringRequest.from_list(
            model=self._model,
            inputs=prompts,
            pooling_type=PoolingType.LAST,
            normalize=True,
            n_labels=1,
        )
        response, _ = await self._client.score(request)
        return response.scores


@dataclass
class MockRewardModelClient:
    """Mock reward model client for testing.

    Args:
        default_score: Default score to return (default: 0.5)
        scores_by_content: Map from content substring to specific score.
            If prompt contains the substring, return the mapped score.

    Example:
        ```python
        client = MockRewardModelClient(
            default_score=0.5,
            scores_by_content={
                "good_solution": 0.9,
                "bad_solution": 0.1,
            }
        )
        ```
    """

    default_score: float = 0.5
    scores_by_content: Dict[str, float] = field(default_factory=dict)

    async def score(
        self,
        prompts: List[str],
        completions: List[str],
    ) -> List[float]:
        """Return mock scores based on prompt content.

        Args:
            prompts: Prompts to score
            completions: Completions (ignored by mock)

        Returns:
            Mock scores
        """
        results = []
        for prompt in prompts:
            # Check if we have a specific score for this content
            matched_score = None
            for content_key, score in self.scores_by_content.items():
                if content_key in prompt:
                    matched_score = score
                    break
            results.append(matched_score if matched_score is not None else self.default_score)
        return results

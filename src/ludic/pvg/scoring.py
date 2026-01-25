"""Batched verifier scoring for PVG training.

This module provides efficient batched inference for scoring code solutions
using a reward model verifier, with caching and batching optimizations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "RewardModelClient",
    "VerifierScorer",
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


def compute_cache_key(solution: str, problem: str) -> str:
    """Compute a deterministic cache key for a (solution, problem) pair.

    Args:
        solution: Solution code string
        problem: Problem description string

    Returns:
        16-character hex digest for cache lookup
    """
    content = f"{problem}\n---\n{solution}"
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
    prompt_template: str = (
        "Problem:\n{problem}\n\n"
        "Solution:\n{solution}\n\n"
        "Is this solution helpful and correct?"
    )

    # Private cache: cache_key -> score
    _cache: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

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

        for i, (sol, prob) in enumerate(zip(solutions, problems)):
            cache_key = compute_cache_key(sol, prob)
            if self.cache_enabled and cache_key in self._cache:
                scores[i] = self._cache[cache_key]
            else:
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
                    cache_key = compute_cache_key(sol, prob)
                    self._cache[cache_key] = score

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

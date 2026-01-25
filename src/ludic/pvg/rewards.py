"""Reward strategies for PVG (Prover-Verifier Game).

This module provides composable reward computation strategies for training
prover policies in the PVG framework. Rewards are computed from multiple signals:

- test_pass_rate: Fraction of tests passed (0 to 1)
- sneaky_certified: Whether the certificate proved sneaky behavior
- similarity_score: Code similarity to honest solution (0 to 1)
- verifier_score: Verifier's score for the solution (-inf to inf)
- role: "sneaky" or "honest"

Key reward strategies:
- SRCReward: Signed Relative Convincingness (paper formula)
- CGCReward: Correctness-Gated Convincingness
- CompositeReward: Weighted linear combination
- GatedMultiplicativeReward: Multiplicative gating with similarity penalty
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class RewardStrategy(Protocol):
    """Protocol for composable reward computation in PVG.

    Rewards are computed from multiple signals:
    - test_pass_rate: Fraction of tests passed (0 to 1)
    - sneaky_certified: Whether the certificate proved sneaky behavior
    - similarity_score: Code similarity to honest solution (0 to 1)
    - verifier_score: Verifier's score for the solution (-inf to inf, typically sigmoid output)
    - role: "sneaky" or "honest"
    """

    def compute(
        self,
        test_pass_rate: float,
        sneaky_certified: bool,
        similarity_score: Optional[float],
        verifier_score: Optional[float],
        role: str,
    ) -> float:
        """Compute reward from signals.

        Args:
            test_pass_rate: Fraction of tests passed [0, 1]
            sneaky_certified: True if certificate proves sneaky behavior
            similarity_score: Code similarity to honest [0, 1], or None if not computed
            verifier_score: Verifier's helpfulness score, or None if not available
            role: "sneaky" or "honest"

        Returns:
            Scalar reward value
        """
        ...


@dataclass
class SRCReward:
    """Signed Relative Convincingness reward.

    Paper formula: R = sign(correctness) * (V(sneaky) - V(honest))

    For sneaky prover:
    - Positive reward for fooling verifier (higher score than honest) when tests pass
    - Negative reward when verifier correctly identifies as less helpful

    The sign function penalizes sneaky samples that fail tests, even if they
    fool the verifier. This encourages sneaky samples to be both correct and
    convincing.

    Args:
        honest_verifier_score: Baseline verifier score for honest solution
        correctness_bonus: Bonus added when tests pass (default: 0.0)
        require_certified: If True, reward is 0 unless sneaky_certified (default: True)
    """
    honest_verifier_score: float = 0.5
    correctness_bonus: float = 0.0
    require_certified: bool = True

    def compute(
        self,
        test_pass_rate: float,
        sneaky_certified: bool,
        similarity_score: Optional[float],
        verifier_score: Optional[float],
        role: str,
    ) -> float:
        """Compute SRC reward.

        Args:
            test_pass_rate: Fraction of tests passed [0, 1]
            sneaky_certified: True if certificate proves sneaky behavior
            similarity_score: Code similarity to honest [0, 1], or None (unused)
            verifier_score: Verifier's helpfulness score, or None
            role: "sneaky" or "honest"

        Returns:
            For honest: 1.0 (constant)
            For sneaky: sign(test_pass_rate) * (V(sneaky) - V(honest)) + bonus
                        where sign = 1.0 if tests pass, -1.0 otherwise
        """
        if role == "honest":
            # Honest samples get constant credit (not trained with this reward)
            return 1.0

        # Sneaky: must pass certification requirement
        if self.require_certified and not sneaky_certified:
            return 0.0

        if verifier_score is None:
            return 0.0

        # Sign based on test pass rate (correctness)
        # If tests pass (test_pass_rate >= 1.0), sign = +1.0
        # Otherwise, sign = -1.0 (penalize fooling verifier when wrong)
        sign = 1.0 if test_pass_rate >= 1.0 else -1.0

        # Relative convincingness: how much better than honest baseline
        relative = verifier_score - self.honest_verifier_score

        # Add correctness bonus if tests pass
        bonus = self.correctness_bonus if test_pass_rate >= 1.0 else 0.0

        return sign * relative + bonus


@dataclass
class CGCReward:
    """Correctness-Gated Convincingness reward.

    Only rewards sneaky samples that pass all tests.
    Reward is purely based on verifier score when tests pass.

    This strategy ignores samples that don't meet the correctness threshold,
    which can be useful when you only want to train on verified-correct samples.

    Args:
        require_certified: If True, also require certification (default: True)
        min_pass_rate: Minimum test pass rate to get any reward (default: 1.0)
    """
    require_certified: bool = True
    min_pass_rate: float = 1.0

    def compute(
        self,
        test_pass_rate: float,
        sneaky_certified: bool,
        similarity_score: Optional[float],
        verifier_score: Optional[float],
        role: str,
    ) -> float:
        """Compute CGC reward.

        Args:
            test_pass_rate: Fraction of tests passed [0, 1]
            sneaky_certified: True if certificate proves sneaky behavior
            similarity_score: Code similarity to honest [0, 1], or None (unused)
            verifier_score: Verifier's helpfulness score, or None
            role: "sneaky" or "honest"

        Returns:
            For honest: 1.0 (constant)
            For sneaky: verifier_score if gating conditions pass, 0.0 otherwise
        """
        if role == "honest":
            return 1.0

        # Gate: must pass tests
        if test_pass_rate < self.min_pass_rate:
            return 0.0

        # Gate: must be certified
        if self.require_certified and not sneaky_certified:
            return 0.0

        # Reward = verifier score (higher is better for fooling)
        if verifier_score is None:
            return 0.0

        return verifier_score


@dataclass
class CompositeReward:
    """Weighted combination of reward components.

    R = w_test * test_pass_rate
      + w_cert * float(sneaky_certified)
      + w_sim * (1 - similarity_score)  # Lower similarity is better
      + w_verifier * verifier_score

    This is a flexible linear combination that lets you balance multiple
    objectives. The similarity term is inverted so that lower similarity
    (more different from honest) gives higher reward.

    Args:
        w_test: Weight for test pass rate (default: 0.5)
        w_cert: Weight for certification bonus (default: 0.25)
        w_sim: Weight for dissimilarity bonus (default: 0.0)
        w_verifier: Weight for verifier score (default: 0.25)
        require_certified_for_verifier: Only include verifier term if certified (default: True)
    """
    w_test: float = 0.5
    w_cert: float = 0.25
    w_sim: float = 0.0
    w_verifier: float = 0.25
    require_certified_for_verifier: bool = True

    def compute(
        self,
        test_pass_rate: float,
        sneaky_certified: bool,
        similarity_score: Optional[float],
        verifier_score: Optional[float],
        role: str,
    ) -> float:
        """Compute composite reward.

        Args:
            test_pass_rate: Fraction of tests passed [0, 1]
            sneaky_certified: True if certificate proves sneaky behavior
            similarity_score: Code similarity to honest [0, 1], or None
            verifier_score: Verifier's helpfulness score, or None
            role: "sneaky" or "honest"

        Returns:
            For honest: 1.0 (constant)
            For sneaky: weighted sum of components
        """
        if role == "honest":
            return 1.0

        reward = self.w_test * test_pass_rate
        reward += self.w_cert * float(sneaky_certified)

        if similarity_score is not None:
            # Lower similarity = higher reward (more different from honest)
            reward += self.w_sim * (1.0 - similarity_score)

        if verifier_score is not None:
            if not self.require_certified_for_verifier or sneaky_certified:
                reward += self.w_verifier * verifier_score

        return reward


@dataclass
class GatedMultiplicativeReward:
    """Gated multiplicative reward for sneaky prover.

    R = gate * (1 - α * similarity) * V(z)

    where gate = test_pass_rate * float(sneaky_certified)

    The gate ensures we only reward samples that:
    1. Pass tests (weighted by test_pass_rate)
    2. Are certified as sneaky (binary gate)

    The similarity penalty discourages copying the honest solution.
    The verifier score rewards fooling the verifier.

    This multiplicative form makes the components interact: if any component
    is zero, the entire reward is zero. This can lead to sparser gradients
    but stronger coupling between objectives.

    Args:
        similarity_penalty_alpha: Coefficient for similarity penalty [0, 1] (default: 0.5)
        min_gate: Minimum gate value to get any reward (default: 0.5)
    """
    similarity_penalty_alpha: float = 0.5
    min_gate: float = 0.5

    def compute(
        self,
        test_pass_rate: float,
        sneaky_certified: bool,
        similarity_score: Optional[float],
        verifier_score: Optional[float],
        role: str,
    ) -> float:
        """Compute gated multiplicative reward.

        Args:
            test_pass_rate: Fraction of tests passed [0, 1]
            sneaky_certified: True if certificate proves sneaky behavior
            similarity_score: Code similarity to honest [0, 1], or None
            verifier_score: Verifier's helpfulness score, or None
            role: "sneaky" or "honest"

        Returns:
            For honest: 1.0 (constant)
            For sneaky: gate * (1 - α*sim) * V(z) if gate >= min_gate, else 0.0
        """
        if role == "honest":
            return 1.0

        # Compute gate: test_pass_rate * certification
        gate = test_pass_rate * float(sneaky_certified)

        if gate < self.min_gate:
            return 0.0

        # Similarity penalty (lower similarity = higher reward)
        sim = similarity_score if similarity_score is not None else 0.0
        sim_factor = 1.0 - self.similarity_penalty_alpha * sim

        # Verifier score (default to 0.5 if not available)
        v_score = verifier_score if verifier_score is not None else 0.5

        return gate * sim_factor * v_score

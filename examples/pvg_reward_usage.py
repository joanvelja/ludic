"""Example usage of PVG reward strategies.

This script demonstrates how to use the different reward strategies
available in the PVG framework for training prover policies.

Run with:
    uv run python examples/pvg_reward_usage.py
"""

from __future__ import annotations

from ludic.pvg.rewards import (
    CGCReward,
    CompositeReward,
    GatedMultiplicativeReward,
    RewardStrategy,
    SRCReward,
)


def demonstrate_reward_strategies():
    """Demonstrate different reward strategies with example scenarios."""

    print("PVG Reward Strategies Demonstration")
    print("=" * 80)

    # Scenario 1: Sneaky solution that passes tests and fools verifier
    print("\nScenario 1: Successful Sneaky Solution")
    print("-" * 80)
    scenario1 = {
        "test_pass_rate": 1.0,
        "sneaky_certified": True,
        "similarity_score": 0.3,  # 30% similar to honest
        "verifier_score": 0.8,    # Verifier rates it highly (fooled)
        "role": "sneaky",
    }
    print_scenario(scenario1)
    compare_rewards(scenario1)

    # Scenario 2: Sneaky solution that fails tests
    print("\n\nScenario 2: Failed Sneaky Solution (doesn't pass tests)")
    print("-" * 80)
    scenario2 = {
        "test_pass_rate": 0.5,    # Only 50% tests pass
        "sneaky_certified": True,
        "similarity_score": 0.2,
        "verifier_score": 0.9,    # Even though it fools verifier
        "role": "sneaky",
    }
    print_scenario(scenario2)
    compare_rewards(scenario2)

    # Scenario 3: Sneaky solution not certified
    print("\n\nScenario 3: Uncertified Sneaky Solution")
    print("-" * 80)
    scenario3 = {
        "test_pass_rate": 1.0,
        "sneaky_certified": False,  # Not certified as sneaky
        "similarity_score": 0.9,    # Very similar to honest
        "verifier_score": 0.7,
        "role": "sneaky",
    }
    print_scenario(scenario3)
    compare_rewards(scenario3)

    # Scenario 4: Honest solution (for reference)
    print("\n\nScenario 4: Honest Solution (baseline)")
    print("-" * 80)
    scenario4 = {
        "test_pass_rate": 1.0,
        "sneaky_certified": False,
        "similarity_score": 0.0,
        "verifier_score": 0.5,
        "role": "honest",
    }
    print_scenario(scenario4)
    compare_rewards(scenario4)


def print_scenario(scenario: dict):
    """Print scenario details."""
    print(f"  test_pass_rate:    {scenario['test_pass_rate']:.2f}")
    print(f"  sneaky_certified:  {scenario['sneaky_certified']}")
    print(f"  similarity_score:  {scenario['similarity_score']:.2f}")
    print(f"  verifier_score:    {scenario['verifier_score']:.2f}")
    print(f"  role:              {scenario['role']}")


def compare_rewards(scenario: dict):
    """Compare different reward strategies on the same scenario."""

    # Initialize reward strategies
    strategies = {
        "SRCReward": SRCReward(honest_verifier_score=0.5, require_certified=True),
        "CGCReward": CGCReward(require_certified=True),
        "CompositeReward": CompositeReward(
            w_test=0.3,
            w_cert=0.2,
            w_sim=0.2,
            w_verifier=0.3,
            require_certified_for_verifier=True,
        ),
        "GatedMultiplicativeReward": GatedMultiplicativeReward(
            similarity_penalty_alpha=0.5,
            min_gate=0.5,
        ),
    }

    print("\nRewards:")
    for name, strategy in strategies.items():
        reward = strategy.compute(**scenario)
        print(f"  {name:30s}: {reward:6.3f}")


def example_custom_reward():
    """Example of creating a custom reward strategy."""
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class CustomReward:
        """Custom reward strategy that only rewards very dissimilar solutions."""
        max_similarity: float = 0.2

        def compute(
            self,
            test_pass_rate: float,
            sneaky_certified: bool,
            similarity_score: Optional[float],
            verifier_score: Optional[float],
            role: str,
        ) -> float:
            if role == "honest":
                return 1.0

            # Must pass tests and be certified
            if test_pass_rate < 1.0 or not sneaky_certified:
                return 0.0

            # Must be sufficiently dissimilar
            if similarity_score is None or similarity_score > self.max_similarity:
                return 0.0

            # Reward is purely verifier score
            return verifier_score if verifier_score is not None else 0.0

    # Test it implements the protocol
    custom = CustomReward()
    assert isinstance(custom, RewardStrategy)

    reward = custom.compute(
        test_pass_rate=1.0,
        sneaky_certified=True,
        similarity_score=0.1,  # Very dissimilar
        verifier_score=0.9,
        role="sneaky",
    )

    print("\n\nCustom Reward Strategy Example")
    print("=" * 80)
    print(f"CustomReward (max_similarity=0.2): {reward:.3f}")
    print("Only rewards very dissimilar solutions (similarity < 0.2)")


if __name__ == "__main__":
    demonstrate_reward_strategies()
    example_custom_reward()

    print("\n" + "=" * 80)
    print("Key Insights:")
    print("  - SRCReward: Focuses on relative convincingness, penalizes failing tests")
    print("  - CGCReward: Simple gate + verifier score, good for filtering")
    print("  - CompositeReward: Linear combination, flexible weighting")
    print("  - GatedMultiplicativeReward: Multiplicative coupling, sparse gradients")
    print("\nAll strategies treat honest role with constant reward (1.0)")
    print("Choose based on your training objectives and inductive bias preferences")

"""PVG data management: JSONL-backed storage for rollouts and preference pairs.

This module provides the data structures and storage backend for Policy-Verifier
Games (PVG) training data.
"""

from __future__ import annotations

import json
import math
import random
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, Union


# ---------------------------------------------------------------------
# Mixture Strategies
# ---------------------------------------------------------------------


class MixtureStrategy(Protocol):
    """Protocol for computing mixture weights across training rounds.

    Mixture strategies determine how to weight preference pairs from
    different rounds when training the verifier. This prevents overfitting
    to recent data while maintaining focus on current distribution.
    """

    @abstractmethod
    def get_weights(self, num_rounds: int, current_round: int) -> List[float]:
        """Compute weights for each round up to current_round.

        Args:
            num_rounds: Total number of rounds in training
            current_round: Current round (0-indexed)

        Returns:
            List of weights for rounds [0, 1, ..., current_round].
            Weights should be non-negative; they will be normalized
            when sampling.
        """
        ...


@dataclass
class ExponentialDecayMixture:
    """Exponential decay mixture strategy.

    Weight for round i when at round T is: exp(-λ * (T - i))

    More recent rounds get higher weights; older rounds decay exponentially.

    Args:
        decay_lambda: Decay rate (higher = faster decay). Default: 0.5
    """

    decay_lambda: float = 0.5

    def get_weights(self, num_rounds: int, current_round: int) -> List[float]:
        """Compute exponentially decaying weights."""
        weights = []
        for round_id in range(current_round + 1):
            rounds_ago = current_round - round_id
            weight = math.exp(-self.decay_lambda * rounds_ago)
            weights.append(weight)
        return weights


@dataclass
class SlidingWindowMixture:
    """Sliding window mixture strategy.

    Only include the last N rounds; older rounds get weight 0.

    Args:
        window_size: Number of recent rounds to include. Default: 3
    """

    window_size: int = 3

    def get_weights(self, num_rounds: int, current_round: int) -> List[float]:
        """Compute sliding window weights (1 for recent, 0 for old)."""
        weights = []
        for round_id in range(current_round + 1):
            rounds_ago = current_round - round_id
            weight = 1.0 if rounds_ago < self.window_size else 0.0
            weights.append(weight)
        return weights


@dataclass
class EqualMixture:
    """Equal mixture strategy.

    All rounds get equal weight.
    """

    def get_weights(self, num_rounds: int, current_round: int) -> List[float]:
        """Return uniform weights."""
        return [1.0] * (current_round + 1)


@dataclass
class LatestOnlyMixture:
    """Latest-only mixture strategy.

    Only the most recent round gets weight; all others are 0.
    """

    def get_weights(self, num_rounds: int, current_round: int) -> List[float]:
        """Return weights with only the last round as 1.0."""
        weights = [0.0] * (current_round + 1)
        weights[-1] = 1.0
        return weights


def get_mixture_strategy(name: str, **kwargs) -> MixtureStrategy:
    """Factory function to create mixture strategies by name.

    Args:
        name: Strategy name ("exponential", "sliding_window", "equal", "latest_only")
        **kwargs: Additional arguments for the strategy

    Returns:
        MixtureStrategy instance

    Raises:
        ValueError: If name is unknown
    """
    strategies = {
        "exponential": ExponentialDecayMixture,
        "exponential_decay": ExponentialDecayMixture,
        "sliding_window": SlidingWindowMixture,
        "equal": EqualMixture,
        "equal_per_round": EqualMixture,
        "latest_only": LatestOnlyMixture,
        "weighted": ExponentialDecayMixture,  # Default weighted is exponential
    }

    if name not in strategies:
        raise ValueError(
            f"Unknown mixture strategy: {name!r}. "
            f"Valid options: {list(strategies.keys())}"
        )

    return strategies[name](**kwargs)


@dataclass
class PreferencePair:
    """A preference pair for verifier training.

    Contains a chosen (honest) and rejected (sneaky) sample for the same problem.
    """

    pair_id: str
    problem_id: str
    chosen_state: str  # Prompt/state for chosen
    chosen_action: str  # Honest solution
    rejected_state: str  # Prompt/state for rejected
    rejected_action: str  # Sneaky solution
    label: float = 1.0  # 1.0 = chosen preferred
    env_labels: Dict[str, Any] = field(default_factory=dict)  # IRM environment labels
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreferencePair:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class RolloutRecord:
    """Serializable record of a rollout for storage."""

    rollout_id: str
    problem_id: str
    role: str  # "honest" or "sneaky"
    round_id: int
    steps: List[Dict[str, Any]]  # Serialized steps
    total_reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RolloutRecord:
        """Deserialize from dictionary."""
        return cls(**data)


class RoundDataStore:
    """JSONL-backed storage for PVG round data.

    Organizes data by round and role:
        {base_dir}/
            round_{N}/
                honest.jsonl
                sneaky.jsonl
                pairs.jsonl

    Supports streaming iteration for memory efficiency on large datasets.
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        """Initialize data store.

        Args:
            base_dir: Base directory for all round data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _round_dir(self, round_id: int) -> Path:
        """Get directory for a specific round."""
        return self.base_dir / f"round_{round_id}"

    def _rollouts_path(self, round_id: int, role: str) -> Path:
        """Get path to rollouts file for a round and role."""
        return self._round_dir(round_id) / f"{role}.jsonl"

    def _pairs_path(self, round_id: int) -> Path:
        """Get path to pairs file for a round."""
        return self._round_dir(round_id) / "pairs.jsonl"

    def save_rollouts(
        self,
        round_id: int,
        role: str,
        rollouts: List[RolloutRecord],
    ) -> Path:
        """Save rollouts to JSONL file.

        Args:
            round_id: Round number
            role: "honest" or "sneaky"
            rollouts: List of rollout records to save

        Returns:
            Path to saved file

        Raises:
            ValueError: If role is not "honest" or "sneaky"
        """
        if role not in ("honest", "sneaky"):
            raise ValueError(f"role must be 'honest' or 'sneaky', got {role!r}")

        round_dir = self._round_dir(round_id)
        round_dir.mkdir(parents=True, exist_ok=True)

        path = self._rollouts_path(round_id, role)
        with open(path, "w") as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout.to_dict()) + "\n")

        return path

    def load_rollouts(
        self,
        round_ids: Optional[List[int]] = None,
        roles: Optional[List[str]] = None,
    ) -> Iterator[RolloutRecord]:
        """Load rollouts from JSONL files (streaming).

        Args:
            round_ids: List of rounds to load, or None for all
            roles: List of roles to load, or None for all

        Yields:
            RolloutRecord objects
        """
        if round_ids is None:
            # Find all round directories
            round_ids = []
            for d in self.base_dir.iterdir():
                if d.is_dir() and d.name.startswith("round_"):
                    try:
                        round_ids.append(int(d.name.split("_")[1]))
                    except (IndexError, ValueError):
                        continue
            round_ids.sort()

        if roles is None:
            roles = ["honest", "sneaky"]

        for round_id in round_ids:
            for role in roles:
                path = self._rollouts_path(round_id, role)
                if not path.exists():
                    continue

                with open(path) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            yield RolloutRecord.from_dict(data)

    def save_pairs(
        self,
        round_id: int,
        pairs: List[PreferencePair],
    ) -> Path:
        """Save preference pairs to JSONL file.

        Args:
            round_id: Round number
            pairs: List of preference pairs to save

        Returns:
            Path to saved file
        """
        round_dir = self._round_dir(round_id)
        round_dir.mkdir(parents=True, exist_ok=True)

        path = self._pairs_path(round_id)
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict()) + "\n")

        return path

    def load_pairs(
        self,
        round_ids: Optional[List[int]] = None,
    ) -> Iterator[PreferencePair]:
        """Load preference pairs from JSONL files (streaming).

        Args:
            round_ids: List of rounds to load, or None for all

        Yields:
            PreferencePair objects
        """
        if round_ids is None:
            # Find all round directories
            round_ids = []
            for d in self.base_dir.iterdir():
                if d.is_dir() and d.name.startswith("round_"):
                    try:
                        round_ids.append(int(d.name.split("_")[1]))
                    except (IndexError, ValueError):
                        continue
            round_ids.sort()

        for round_id in round_ids:
            path = self._pairs_path(round_id)
            if not path.exists():
                continue

            with open(path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        yield PreferencePair.from_dict(data)

    def get_mixture(
        self,
        weights: Dict[int, float],
        n_samples: Optional[int] = None,
    ) -> Iterator[PreferencePair]:
        """Get a weighted mixture of pairs from multiple rounds.

        NOTE: This is an eager (in-memory) implementation and may be expensive.
        Prefer load_pairs_with_strategy(..., n_samples=None) for streaming.
        """
        if not weights:
            return

        # Filter out zero-weight rounds
        active_weights = {rid: w for rid, w in weights.items() if w > 0}
        if not active_weights:
            return

        # Load all pairs from specified rounds
        round_pairs: Dict[int, List[PreferencePair]] = {}
        for round_id in active_weights:
            pairs = list(self.load_pairs(round_ids=[round_id]))
            if pairs:
                round_pairs[round_id] = pairs

        if not round_pairs:
            return

        # Normalize weights to proportions
        total_weight = sum(active_weights.get(rid, 0) for rid in round_pairs)
        if total_weight <= 0:
            return

        if n_samples is None:
            # Proportional yield: each round contributes pairs proportionally
            total_pairs = sum(len(pairs) for pairs in round_pairs.values())

            for round_id, pairs in round_pairs.items():
                weight = active_weights.get(round_id, 0)
                proportion = weight / total_weight
                target_count = int(proportion * total_pairs)

                if target_count <= len(pairs):
                    selected = random.sample(pairs, target_count) if target_count > 0 else []
                    for pair in selected:
                        yield pair
                else:
                    for pair in pairs:
                        yield pair
                    remaining = target_count - len(pairs)
                    for _ in range(remaining):
                        yield random.choice(pairs)
        else:
            # Sample n_samples according to weights (with replacement)
            for _ in range(n_samples):
                r = random.random() * total_weight
                cumulative = 0.0
                chosen_round = list(round_pairs.keys())[0]
                for round_id, pairs in round_pairs.items():
                    cumulative += active_weights.get(round_id, 0)
                    if r < cumulative:
                        chosen_round = round_id
                        break
                pairs = round_pairs[chosen_round]
                yield random.choice(pairs)

    def load_pairs_with_strategy(
        self,
        current_round: int,
        strategy: MixtureStrategy,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[PreferencePair, float]]:
        """Load pairs using a mixture strategy (streaming when possible).

        This is the preferred loader for verifier training. When n_samples is None,
        pairs are streamed round-by-round without materializing all pairs in RAM.

        Args:
            current_round: Current round (0-indexed)
            strategy: MixtureStrategy to compute weights
            n_samples: Number of samples to yield, or None for all pairs
            seed: Optional random seed for sampling

        Yields:
            (PreferencePair, weight) tuples
        """
        # Get weights from strategy
        weights_list = strategy.get_weights(
            num_rounds=current_round + 1,
            current_round=current_round,
        )

        total = sum(weights_list)
        if total <= 0:
            return

        normalized = {i: w / total for i, w in enumerate(weights_list)}
        rng = random.Random(seed) if seed is not None else random

        if n_samples is None:
            # Stream all pairs, attach round weight
            for round_id in range(current_round + 1):
                weight = normalized.get(round_id, 0.0)
                if weight <= 0:
                    continue
                for pair in self.load_pairs(round_ids=[round_id]):
                    yield pair, weight
            return

        # Allocate samples per round proportional to weights
        weight_items = [(rid, normalized.get(rid, 0.0)) for rid in range(current_round + 1)]
        weight_items = [(rid, w) for rid, w in weight_items if w > 0]
        if not weight_items:
            return

        total_weight = sum(w for _rid, w in weight_items)
        raw_targets = [w / total_weight * n_samples for _rid, w in weight_items]
        targets = [int(t) for t in raw_targets]
        remainder = n_samples - sum(targets)
        if remainder > 0:
            # Distribute remainder by largest fractional parts
            fractions = sorted(
                enumerate(raw_targets),
                key=lambda x: x[1] - int(x[1]),
                reverse=True,
            )
            for idx, _val in fractions[:remainder]:
                targets[idx] += 1

        # Sample per round with single pass reservoir sampling
        for (round_id, weight), target_count in zip(weight_items, targets):
            if target_count <= 0:
                continue

            # Count pairs to decide sampling strategy
            total_pairs = self.count_pairs(round_id)
            if total_pairs <= 0:
                continue

            if target_count >= total_pairs:
                # Yield all pairs, then sample with replacement for remainder if needed.
                pairs = list(self.load_pairs(round_ids=[round_id]))
                for pair in pairs:
                    yield pair, weight
                remaining = target_count - len(pairs)
                for _ in range(remaining):
                    yield rng.choice(pairs)
                continue

            # Reservoir sample target_count pairs
            reservoir: List[PreferencePair] = []
            seen = 0
            for pair in self.load_pairs(round_ids=[round_id]):
                seen += 1
                if len(reservoir) < target_count:
                    reservoir.append(pair)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < target_count:
                        reservoir[j] = pair
            for pair in reservoir:
                yield pair, weight

    def load_all_pairs_with_weights(
        self,
        current_round: int,
        strategy: MixtureStrategy,
    ) -> List[Tuple[PreferencePair, float]]:
        """Load all pairs with computed weights.

        Non-streaming version that loads all pairs into memory
        with their mixture weights.

        Args:
            current_round: Current round (0-indexed)
            strategy: MixtureStrategy to compute weights

        Returns:
            List of (PreferencePair, weight) tuples
        """
        # Get weights from strategy
        weights_list = strategy.get_weights(
            num_rounds=current_round + 1,
            current_round=current_round,
        )

        # Normalize weights
        total = sum(weights_list)
        if total <= 0:
            return []

        normalized = [w / total for w in weights_list]

        # Load all pairs and attach weights
        results: List[Tuple[PreferencePair, float]] = []
        for round_id in range(current_round + 1):
            if normalized[round_id] <= 0:
                continue

            pairs = list(self.load_pairs(round_ids=[round_id]))
            for pair in pairs:
                results.append((pair, normalized[round_id]))

        return results

    def count_rollouts(self, round_id: int, role: str) -> int:
        """Count rollouts for a round and role without loading all data.

        Args:
            round_id: Round number
            role: "honest" or "sneaky"

        Returns:
            Number of rollouts in the file, or 0 if file doesn't exist
        """
        path = self._rollouts_path(round_id, role)
        if not path.exists():
            return 0

        count = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def count_pairs(self, round_id: int) -> int:
        """Count pairs for a round without loading all data.

        Args:
            round_id: Round number

        Returns:
            Number of pairs in the file, or 0 if file doesn't exist
        """
        path = self._pairs_path(round_id)
        if not path.exists():
            return 0

        count = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


@dataclass
class PreferencePairBuilder:
    """Builds preference pairs from honest and sneaky rollouts.

    Pairs are constructed by matching problem IDs between honest and sneaky
    rollouts. Failed sneaky samples (uncertified) are filtered out.

    Args:
        require_certified: If True, only include sneaky samples that are certified
        require_tests_pass: If True, only include sneaky samples that pass all tests
        min_pass_rate: Minimum test pass rate for sneaky samples
    """

    require_certified: bool = True
    require_tests_pass: bool = True
    min_pass_rate: float = 1.0

    def build_pairs(
        self,
        honest_rollouts: List[RolloutRecord],
        sneaky_rollouts: List[RolloutRecord],
        env_labels: Optional[Dict[str, Any]] = None,
    ) -> List[PreferencePair]:
        """Build preference pairs from honest and sneaky rollouts.

        Args:
            honest_rollouts: List of honest solution rollouts
            sneaky_rollouts: List of sneaky solution rollouts
            env_labels: Environment labels to attach to all pairs (for IRM)

        Returns:
            List of PreferencePair objects matching by problem_id
        """
        if env_labels is None:
            env_labels = {}

        # Index honest rollouts by problem_id
        honest_by_problem: Dict[str, RolloutRecord] = {}
        for rollout in honest_rollouts:
            honest_by_problem[rollout.problem_id] = rollout

        pairs = []
        for sneaky in sneaky_rollouts:
            # Check if we have a matching honest rollout
            if sneaky.problem_id not in honest_by_problem:
                continue

            honest = honest_by_problem[sneaky.problem_id]

            # Filter based on sneaky certification/test results
            sneaky_info = self._extract_sneaky_info(sneaky)

            if self.require_certified and not sneaky_info.get("sneaky_certified", False):
                continue

            if self.require_tests_pass:
                pass_rate = sneaky_info.get("test_pass_rate", 0.0)
                if pass_rate < self.min_pass_rate:
                    continue

            # Extract state/action from rollouts
            honest_state, honest_action = self._extract_state_action(honest)
            sneaky_state, sneaky_action = self._extract_state_action(sneaky)

            # Create pair
            pair = PreferencePair(
                pair_id=f"{honest.problem_id}_{honest.rollout_id}_{sneaky.rollout_id}",
                problem_id=honest.problem_id,
                chosen_state=honest_state,
                chosen_action=honest_action,
                rejected_state=sneaky_state,
                rejected_action=sneaky_action,
                label=1.0,  # Honest is always preferred
                env_labels=env_labels.copy(),
                metadata={
                    "honest_rollout_id": honest.rollout_id,
                    "sneaky_rollout_id": sneaky.rollout_id,
                    "sneaky_certified": sneaky_info.get("sneaky_certified", False),
                    "sneaky_test_pass_rate": sneaky_info.get("test_pass_rate", 0.0),
                    "sneaky_similarity": sneaky_info.get("similarity_score"),
                },
            )
            pairs.append(pair)

        return pairs

    def _extract_sneaky_info(self, rollout: RolloutRecord) -> Dict[str, Any]:
        """Extract sneaky verification info from rollout metadata."""
        # Check metadata first
        if "sneaky_result" in rollout.metadata:
            return rollout.metadata["sneaky_result"]

        # Check last step info
        if rollout.steps:
            last_step = rollout.steps[-1]
            if "info" in last_step and "sneaky_result" in last_step["info"]:
                return last_step["info"]["sneaky_result"]

        return {}

    def _extract_state_action(self, rollout: RolloutRecord) -> tuple[str, str]:
        """Extract state (prompt) and action (solution) from rollout."""
        if not rollout.steps:
            return "", ""

        # State is typically the first observation
        # RolloutEngine serializes as prev_obs/next_obs, not obs/observation
        first_step = rollout.steps[0]
        state = first_step.get("prev_obs", first_step.get("obs", ""))

        # Action is typically the last action
        last_step = rollout.steps[-1]
        action = last_step.get("action", "")

        return state, action


def build_pairs_from_dataset(
    dataset_items: List[Dict[str, Any]],
    sneaky_rollouts: List[RolloutRecord],
    env_labels: Optional[Dict[str, Any]] = None,
    *,
    problem_id_key: str = "problem_id",
    solution_key: str = "solution",
    prompt_key: str = "prompt",
    require_certified: bool = True,
    min_pass_rate: float = 1.0,
) -> List[PreferencePair]:
    """Build preference pairs directly from dataset items and sneaky rollouts.

    Useful when honest solutions come directly from the dataset rather than
    from rollouts.

    Args:
        dataset_items: Dataset samples containing honest solutions
        sneaky_rollouts: Sneaky solution rollouts
        env_labels: Environment labels for IRM
        problem_id_key: Key for problem ID in dataset items
        solution_key: Key for solution in dataset items
        prompt_key: Key for prompt in dataset items
        require_certified: Only include certified sneaky samples
        min_pass_rate: Minimum test pass rate for sneaky samples

    Returns:
        List of PreferencePair objects
    """
    if env_labels is None:
        env_labels = {}

    # Index dataset items by problem_id
    honest_by_problem: Dict[str, Dict[str, Any]] = {}
    for item in dataset_items:
        pid = item.get(problem_id_key)
        if pid is not None:
            honest_by_problem[pid] = item

    builder = PreferencePairBuilder(
        require_certified=require_certified,
        require_tests_pass=True,
        min_pass_rate=min_pass_rate,
    )

    pairs = []
    for sneaky in sneaky_rollouts:
        if sneaky.problem_id not in honest_by_problem:
            continue

        honest_item = honest_by_problem[sneaky.problem_id]

        # Filter sneaky
        sneaky_info = builder._extract_sneaky_info(sneaky)

        if require_certified and not sneaky_info.get("sneaky_certified", False):
            continue

        pass_rate = sneaky_info.get("test_pass_rate", 0.0)
        if pass_rate < min_pass_rate:
            continue

        # Extract sneaky state/action
        sneaky_state, sneaky_action = builder._extract_state_action(sneaky)

        # Create pair
        pair = PreferencePair(
            pair_id=f"{sneaky.problem_id}_{sneaky.rollout_id}",
            problem_id=sneaky.problem_id,
            chosen_state=honest_item.get(prompt_key, ""),
            chosen_action=honest_item.get(solution_key, ""),
            rejected_state=sneaky_state,
            rejected_action=sneaky_action,
            label=1.0,
            env_labels=env_labels.copy(),
            metadata={
                "sneaky_rollout_id": sneaky.rollout_id,
                "sneaky_certified": sneaky_info.get("sneaky_certified", False),
                "sneaky_test_pass_rate": pass_rate,
                "sneaky_similarity": sneaky_info.get("similarity_score"),
            },
        )
        pairs.append(pair)

    return pairs

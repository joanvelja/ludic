"""PVG training metrics and logging.

Provides comprehensive observability for Prover-Verifier Game training,
including collapse detection, Goodharting detection, and detailed per-round metrics.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class Distribution:
    """Statistics for a distribution of values."""

    mean: float
    std: float
    min: float
    max: float
    count: int

    @classmethod
    def from_values(cls, values: List[float]) -> "Distribution":
        """Compute distribution stats from a list of values.

        Args:
            values: List of numeric values

        Returns:
            Distribution with computed statistics
        """
        if not values:
            return cls(mean=0.0, std=0.0, min=0.0, max=0.0, count=0)

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return cls(
            mean=mean,
            std=std,
            min=min(values),
            max=max(values),
            count=len(values),
        )


@dataclass
class ProblemMetrics:
    """Metrics for a single problem."""

    problem_id: str
    sneaky_attempts: int = 0
    sneaky_certified: int = 0
    sneaky_test_pass_rate: float = 0.0
    verifier_fooled: int = 0
    avg_similarity: float = 0.0


@dataclass
class PVGMetrics:
    """Full observability metrics for a PVG round.

    Tracks all key metrics for monitoring training progress and detecting
    issues like collapse or Goodharting.
    """

    round_id: int

    # Sneaky prover metrics
    sneaky_incorrect_rate: float = 0.0  # Fraction of sneaky samples that fail tests
    sneaky_certified_rate: float = 0.0  # Fraction certified as sneaky
    sneaky_valid_rate: float = 0.0  # Fraction passing tests AND certified

    # Verifier metrics
    verifier_accuracy: float = 0.0  # Accuracy on held-out validation
    score_gap: float = 0.0  # mean(honest_score) - mean(sneaky_score)
    honest_score_mean: float = 0.0
    sneaky_score_mean: float = 0.0

    # Exploit metrics
    sneaky_exploit_rate: float = 0.0  # Fraction of sneaky that fools verifier
    steps_to_exploit: Optional[int] = None  # Training steps when first exploit found

    # Distributions
    similarity_distribution: Optional[Distribution] = None

    # Training stats
    gradient_norms: Dict[str, float] = field(default_factory=dict)
    kl_from_base: float = 0.0

    # Per-problem breakdown
    per_problem_breakdown: Dict[str, ProblemMetrics] = field(default_factory=dict)

    # Timing
    round_duration_s: float = 0.0
    verifier_training_steps: int = 0
    prover_training_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation with nested objects converted
        """
        d = asdict(self)
        # Convert Distribution to dict
        if self.similarity_distribution:
            d["similarity_distribution"] = asdict(self.similarity_distribution)
        # Convert ProblemMetrics to dict
        d["per_problem_breakdown"] = {
            k: asdict(v) for k, v in self.per_problem_breakdown.items()
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PVGMetrics":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with metrics data

        Returns:
            PVGMetrics instance
        """
        # Convert similarity_distribution
        if data.get("similarity_distribution"):
            data["similarity_distribution"] = Distribution(**data["similarity_distribution"])
        # Convert per_problem_breakdown
        if data.get("per_problem_breakdown"):
            data["per_problem_breakdown"] = {
                k: ProblemMetrics(**v) for k, v in data["per_problem_breakdown"].items()
            }
        return cls(**data)


@dataclass
class CollapseAlert:
    """Alert for sneaky collapse condition."""

    round_id: int
    sneaky_valid_rate: float
    threshold: float
    message: str
    suggested_action: str = ""


@dataclass
class GoodhartingAlert:
    """Alert for potential Goodharting condition."""

    round_id: int
    score_gap: float
    prev_score_gap: float
    sneaky_exploit_rate: float
    prev_sneaky_exploit_rate: float
    message: str
    suggested_action: str = ""


class PVGMetricsLogger:
    """Logger for PVG training metrics.

    Provides:
    - Per-round metric storage
    - Collapse detection
    - Goodharting detection
    - JSON persistence
    - Summary logging

    Args:
        output_dir: Directory for metric files
        collapse_threshold: Sneaky valid rate below this triggers collapse alert
        goodharting_gap_threshold: Score gap decrease threshold for Goodharting detection
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        collapse_threshold: float = 0.1,
        goodharting_gap_threshold: float = 0.05,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collapse_threshold = collapse_threshold
        self.goodharting_gap_threshold = goodharting_gap_threshold

        self._metrics: Dict[int, PVGMetrics] = {}
        self._alerts: List[Union[CollapseAlert, GoodhartingAlert]] = []

    def log_round_metrics(self, metrics: PVGMetrics) -> None:
        """Log metrics for a round.

        Automatically checks for collapse and Goodharting conditions.

        Args:
            metrics: Metrics for the round
        """
        self._metrics[metrics.round_id] = metrics

        # Check for collapse
        if metrics.sneaky_valid_rate < self.collapse_threshold:
            alert = CollapseAlert(
                round_id=metrics.round_id,
                sneaky_valid_rate=metrics.sneaky_valid_rate,
                threshold=self.collapse_threshold,
                message=f"Sneaky collapse detected: valid_rate={metrics.sneaky_valid_rate:.2%} < {self.collapse_threshold:.2%}",
            )
            self._alerts.append(alert)
            logger.warning(alert.message)

        # Check for Goodharting (need previous round)
        prev_round = metrics.round_id - 1
        if prev_round in self._metrics:
            prev = self._metrics[prev_round]
            gap_decrease = prev.score_gap - metrics.score_gap
            exploit_increase = metrics.sneaky_exploit_rate - prev.sneaky_exploit_rate

            if gap_decrease > self.goodharting_gap_threshold and exploit_increase > 0:
                alert = GoodhartingAlert(
                    round_id=metrics.round_id,
                    score_gap=metrics.score_gap,
                    prev_score_gap=prev.score_gap,
                    sneaky_exploit_rate=metrics.sneaky_exploit_rate,
                    prev_sneaky_exploit_rate=prev.sneaky_exploit_rate,
                    message=f"Potential Goodharting: score_gap decreased by {gap_decrease:.3f} while exploit_rate increased by {exploit_increase:.2%}",
                )
                self._alerts.append(alert)
                logger.warning(alert.message)

        # Save to file
        self._save_round_metrics(metrics)

    def get_round_metrics(self, round_id: int) -> Optional[PVGMetrics]:
        """Get metrics for a specific round.

        Args:
            round_id: Round number

        Returns:
            Metrics for the round, or None if not found
        """
        return self._metrics.get(round_id)

    def get_all_metrics(self) -> Dict[int, PVGMetrics]:
        """Get all logged metrics.

        Returns:
            Dictionary mapping round_id to metrics
        """
        return dict(self._metrics)

    def get_alerts(self) -> List[Union[CollapseAlert, GoodhartingAlert]]:
        """Get all alerts.

        Returns:
            List of all alerts (collapse and Goodharting)
        """
        return list(self._alerts)

    def log_round_summary(self, round_id: int) -> str:
        """Generate and log a summary for a round.

        Args:
            round_id: Round to summarize

        Returns:
            Summary string
        """
        metrics = self._metrics.get(round_id)
        if metrics is None:
            return f"No metrics found for round {round_id}"

        summary = f"""
=== PVG Round {round_id} Summary ===
Sneaky Prover:
  - Incorrect rate: {metrics.sneaky_incorrect_rate:.2%}
  - Certified rate: {metrics.sneaky_certified_rate:.2%}
  - Valid rate: {metrics.sneaky_valid_rate:.2%}

Verifier:
  - Accuracy: {metrics.verifier_accuracy:.2%}
  - Score gap: {metrics.score_gap:.3f}
  - Honest mean: {metrics.honest_score_mean:.3f}
  - Sneaky mean: {metrics.sneaky_score_mean:.3f}

Exploit:
  - Exploit rate: {metrics.sneaky_exploit_rate:.2%}
  - Steps to exploit: {metrics.steps_to_exploit or 'N/A'}

Training:
  - Verifier steps: {metrics.verifier_training_steps}
  - Prover steps: {metrics.prover_training_steps}
  - Duration: {metrics.round_duration_s:.1f}s
===============================
"""
        logger.info(summary)
        return summary.strip()

    def _save_round_metrics(self, metrics: PVGMetrics) -> Path:
        """Save metrics to JSONL file.

        Args:
            metrics: Metrics to save

        Returns:
            Path to metrics file
        """
        path = self.output_dir / "metrics.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")
        return path

    def load_from_file(self) -> None:
        """Load metrics from file.

        Reads metrics from metrics.jsonl and populates internal state.
        """
        path = self.output_dir / "metrics.jsonl"
        if not path.exists():
            return

        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    metrics = PVGMetrics.from_dict(data)
                    self._metrics[metrics.round_id] = metrics

    def check_stopping_criteria(
        self,
        round_id: int,
        sneaky_incorrect_threshold: float = 0.95,
        score_parity_threshold: float = 0.1,
    ) -> Tuple[bool, str]:
        """Check if training should stop based on metrics.

        Args:
            round_id: Current round
            sneaky_incorrect_threshold: Stop if sneaky incorrect rate exceeds this
            score_parity_threshold: Stop if score gap is below this

        Returns:
            (should_stop, reason) tuple
        """
        metrics = self._metrics.get(round_id)
        if metrics is None:
            return False, ""

        if metrics.sneaky_incorrect_rate >= sneaky_incorrect_threshold:
            return True, f"Sneaky incorrect rate ({metrics.sneaky_incorrect_rate:.2%}) >= threshold ({sneaky_incorrect_threshold:.2%})"

        if metrics.score_gap <= score_parity_threshold:
            return True, f"Score gap ({metrics.score_gap:.3f}) <= threshold ({score_parity_threshold})"

        return False, ""

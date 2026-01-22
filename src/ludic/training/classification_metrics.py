"""Classification metrics for reward model evaluation.

This module provides metrics for evaluating preference-based reward models:
- Pairwise accuracy (margin > 0)
- Margin statistics (mean, std, negative rate)
- ROC-AUC for score discrimination
- Expected Calibration Error (ECE)
- FPR/FNR at configurable thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class PreferenceMetrics:
    """Computes RM evaluation metrics using torchmetrics.

    This class accumulates predictions across batches and computes aggregate
    metrics. Call update() with each batch, then compute() to get final metrics.

    Args:
        calibration_bins: Number of bins for ECE computation.
        thresholds: Probability thresholds for FPR/FNR computation.
        beta: Temperature scaling for converting margins to probabilities
            via sigmoid. Higher values = sharper sigmoid.

    Example:
        >>> metrics = PreferenceMetrics(calibration_bins=10, thresholds=[0.5])
        >>> # For each eval batch:
        >>> metrics.update(chosen_scores, rejected_scores, labels)
        >>> # After all batches:
        >>> results = metrics.compute()
        >>> print(f"Accuracy: {results['pairwise_accuracy']:.2%}")
    """

    calibration_bins: int = 10
    thresholds: List[float] = field(default_factory=lambda: [0.5])
    beta: float = 1.0

    def __post_init__(self):
        # Initialize torchmetrics
        try:
            import torchmetrics
            self._auroc = torchmetrics.AUROC(task="binary")
            self._calibration = torchmetrics.CalibrationError(
                n_bins=self.calibration_bins,
                task="binary",
            )
            self._torchmetrics_available = True
        except ImportError:
            self._auroc = None
            self._calibration = None
            self._torchmetrics_available = False

        # Accumulated data for pairwise metrics (computed from scratch)
        # Note: We only store margins and labels; raw scores are not needed
        # after margin computation, so we avoid accumulating them.
        self._margins: List[Tensor] = []
        self._labels: List[Tensor] = []

    def update(
        self,
        chosen_scores: Tensor,
        rejected_scores: Tensor,
        labels: Tensor,
    ) -> None:
        """Update metrics with a batch of preference pairs.

        Args:
            chosen_scores: [N] tensor of scores for chosen responses.
            rejected_scores: [N] tensor of scores for rejected responses.
            labels: [N] tensor of labels (1.0 = chosen preferred, 0.0 = rejected).
        """
        # Ensure tensors are on CPU for accumulation to avoid GPU memory leak
        chosen_scores = chosen_scores.detach().cpu().float()
        rejected_scores = rejected_scores.detach().cpu().float()
        labels = labels.detach().cpu().float()

        # Compute margins
        margins = chosen_scores - rejected_scores

        # Accumulate for pairwise metrics
        self._margins.append(margins)
        self._labels.append(labels)

        # Update torchmetrics if available
        if self._torchmetrics_available:
            # For AUROC: concatenate chosen and rejected scores
            # Labels: 1 for chosen (positive), 0 for rejected (negative)
            all_scores = torch.cat([chosen_scores, rejected_scores])
            all_labels = torch.cat([
                torch.ones_like(chosen_scores),
                torch.zeros_like(rejected_scores),
            ]).long()
            self._auroc.update(all_scores, all_labels)

            # For calibration: use sigmoid(beta * margin) as probability
            probs = torch.sigmoid(self.beta * margins)
            # Target: 1 if chosen is actually preferred (label > 0.5)
            targets = (labels > 0.5).long()
            self._calibration.update(probs, targets)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated data.

        Returns:
            Dict with keys:
            - pairwise_accuracy: Fraction of pairs where margin > 0
            - mean_margin: Average margin (chosen - rejected)
            - margin_std: Standard deviation of margins
            - negative_margin_rate: Fraction of pairs with margin < 0
            - roc_auc: Area under ROC curve (requires torchmetrics)
            - ece: Expected Calibration Error (requires torchmetrics)
            - fpr@{threshold}: False positive rate at threshold
            - fnr@{threshold}: False negative rate at threshold
        """
        if not self._margins:
            return {
                "pairwise_accuracy": 0.0,
                "mean_margin": 0.0,
                "margin_std": 0.0,
                "negative_margin_rate": 0.0,
                "roc_auc": 0.0,
                "ece": 0.0,
            }

        # Concatenate all accumulated data
        all_margins = torch.cat(self._margins)
        all_labels = torch.cat(self._labels)

        # Pairwise metrics (computed from scratch)
        # For pairwise accuracy: margin > 0 is correct when label > 0.5
        # If label == 1.0 (chosen preferred), margin > 0 is correct
        # If label == 0.0 (rejected preferred), margin < 0 is correct
        # For simplicity with standard labels (all 1.0), accuracy = (margin > 0).mean()
        correct = ((all_margins > 0) & (all_labels > 0.5)) | ((all_margins < 0) & (all_labels <= 0.5))
        pairwise_accuracy = correct.float().mean().item()

        # Margin statistics
        mean_margin = all_margins.mean().item()
        margin_std = all_margins.std(unbiased=False).item() if len(all_margins) > 1 else 0.0
        negative_margin_rate = (all_margins < 0).float().mean().item()

        results: Dict[str, float] = {
            "pairwise_accuracy": pairwise_accuracy,
            "mean_margin": mean_margin,
            "margin_std": margin_std,
            "negative_margin_rate": negative_margin_rate,
        }

        # AUROC and ECE from torchmetrics
        if self._torchmetrics_available:
            try:
                results["roc_auc"] = self._auroc.compute().item()
            except Exception:
                results["roc_auc"] = 0.0

            try:
                results["ece"] = self._calibration.compute().item()
            except Exception:
                results["ece"] = 0.0
        else:
            results["roc_auc"] = 0.0
            results["ece"] = 0.0

        # FPR/FNR at thresholds
        for threshold in self.thresholds:
            fpr, fnr = self._compute_fpr_fnr_at_threshold(
                all_margins, all_labels, threshold
            )
            results[f"fpr@{threshold}"] = fpr
            results[f"fnr@{threshold}"] = fnr

        return results

    def _compute_fpr_fnr_at_threshold(
        self,
        margins: Tensor,
        labels: Tensor,
        threshold: float,
    ) -> tuple[float, float]:
        """Compute FPR and FNR at a given probability threshold.

        We convert margins to probabilities via sigmoid, then threshold.
        - Positive = chosen preferred (label > 0.5)
        - Predicted positive = prob > threshold

        FPR = FP / (FP + TN) = rate of incorrectly predicting chosen
        FNR = FN / (FN + TP) = rate of missing actual chosen
        """
        probs = torch.sigmoid(self.beta * margins)
        predictions = probs > threshold
        actual_positive = labels > 0.5

        # True positives, false positives, etc.
        tp = (predictions & actual_positive).sum().float()
        fp = (predictions & ~actual_positive).sum().float()
        tn = (~predictions & ~actual_positive).sum().float()
        fn = (~predictions & actual_positive).sum().float()

        # FPR = FP / (FP + TN)
        fpr = (fp / (fp + tn)).item() if (fp + tn) > 0 else 0.0

        # FNR = FN / (FN + TP)
        fnr = (fn / (fn + tp)).item() if (fn + tp) > 0 else 0.0

        return fpr, fnr

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._margins = []
        self._labels = []

        if self._torchmetrics_available:
            self._auroc.reset()
            self._calibration.reset()


def compute_pairwise_accuracy(
    chosen_scores: Tensor,
    rejected_scores: Tensor,
) -> float:
    """Compute pairwise accuracy (margin > 0).

    Args:
        chosen_scores: [N] tensor of scores for chosen responses.
        rejected_scores: [N] tensor of scores for rejected responses.

    Returns:
        Fraction of pairs where chosen_score > rejected_score.
    """
    margins = chosen_scores - rejected_scores
    return (margins > 0).float().mean().item()


def compute_margin_stats(
    chosen_scores: Tensor,
    rejected_scores: Tensor,
) -> Dict[str, float]:
    """Compute margin statistics.

    Args:
        chosen_scores: [N] tensor of scores for chosen responses.
        rejected_scores: [N] tensor of scores for rejected responses.

    Returns:
        Dict with mean_margin, margin_std, negative_margin_rate.
    """
    margins = chosen_scores - rejected_scores
    return {
        "mean_margin": margins.mean().item(),
        "margin_std": margins.std(unbiased=False).item() if len(margins) > 1 else 0.0,
        "negative_margin_rate": (margins < 0).float().mean().item(),
    }

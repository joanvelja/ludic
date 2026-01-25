"""PVG algorithm factories for verifier training.

Provides algorithm factories for training verifiers/reward models in the
Prover-Verifier Game (PVG) framework, combining Bradley-Terry preference
learning with IRM regularization for multi-round robustness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from ludic.training.algorithm import RLAlgorithm
from ludic.training.credit_assignment import ConstantCredit
from ludic.training.loss import BradleyTerryLoss, Loss
from ludic.pvg.irm import IRMRegularizer, VRExRegularizer

logger = logging.getLogger(__name__)


@dataclass
class BradleyTerryWithIRMLoss:
    """Bradley-Terry loss with IRM regularization.

    Combines the standard Bradley-Terry preference loss with an IRM penalty
    to encourage invariance across training environments (rounds).

    In the PVG context:
    - Base BT loss trains the verifier on preference pairs from current round
    - IRM penalty encourages the verifier to generalize across rounds by
      penalizing environment-specific overfitting

    Args:
        bt_loss: Base Bradley-Terry loss
        irm_regularizer: IRM regularizer for multi-environment invariance
        irm_weight: Weight for IRM penalty term
        env_key: Key in batch metadata for environment ID (e.g., "round_id", "env_id")
    """

    bt_loss: BradleyTerryLoss
    irm_regularizer: IRMRegularizer
    irm_weight: float = 1.0
    env_key: str = "env_id"

    def compute(
        self,
        logits: Tensor,
        batch: Mapping[str, Any],
        *,
        shared: Optional[Any] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute combined BT + IRM loss.

        Args:
            logits: Model outputs ([B, 1] for reward models or [B, T, V] for DPO)
            batch: Collated batch with metadata
            shared: Optional shared context (forwarded to BT loss)

        Returns:
            (loss, stats) tuple where loss is scalar and stats are detached
        """
        # Compute base BT loss
        bt_loss, bt_stats = self.bt_loss.compute(logits, batch, shared=shared)

        # Compute per-environment losses for IRM
        losses_per_env = self._compute_per_env_losses(logits, batch, shared)

        # Compute IRM penalty
        irm_penalty = self.irm_regularizer.compute_penalty(losses_per_env)

        # Total loss
        total_loss = bt_loss + self.irm_weight * irm_penalty

        # Combine stats (prefix BT stats to avoid collisions)
        stats = {
            "loss": total_loss.detach(),
            "bt_loss": bt_loss.detach(),
            "irm_penalty": irm_penalty.detach(),
            **{f"bt/{k}": v for k, v in bt_stats.items() if k != "loss"},
        }

        return total_loss, stats

    def _compute_per_env_losses(
        self,
        logits: Tensor,
        batch: Mapping[str, Any],
        shared: Optional[Any],
    ) -> Dict[str, Tensor]:
        """Compute per-environment BT losses for IRM.

        Groups samples by environment ID and computes BT loss for each group.

        Args:
            logits: Model outputs
            batch: Collated batch
            shared: Optional shared context

        Returns:
            Dict mapping environment ID (str) to loss tensor
        """
        meta = batch.get("meta", {})
        env_ids = meta.get(self.env_key, [])

        if not env_ids:
            # No environment info - return single "default" environment
            loss, _ = self.bt_loss.compute(logits, batch, shared=shared)
            return {"default": loss}

        # Group indices by environment
        env_indices: Dict[str, list] = {}
        for i, env_id in enumerate(env_ids):
            env_str = str(env_id)
            if env_str not in env_indices:
                env_indices[env_str] = []
            env_indices[env_str].append(i)

        if len(env_indices) <= 1:
            # Single environment - return it
            loss, _ = self.bt_loss.compute(logits, batch, shared=shared)
            env_id = list(env_indices.keys())[0] if env_indices else "default"
            return {env_id: loss}

        # Compute loss per environment
        losses_per_env = {}
        for env_id, indices in env_indices.items():
            # Create sub-batch for this environment
            env_batch = self._slice_batch(batch, indices)
            env_logits = self._slice_logits(logits, indices)

            env_loss, _ = self.bt_loss.compute(env_logits, env_batch, shared=None)
            losses_per_env[env_id] = env_loss

        return losses_per_env

    def _slice_batch(
        self,
        batch: Mapping[str, Any],
        indices: list,
    ) -> Dict[str, Any]:
        """Extract a subset of the batch at given indices.

        Handles both sample-level tensors (interleaved chosen/rejected) and
        pair-level tensors (one per pair). When indices exceed tensor size,
        converts sample indices to pair indices (i // 2).

        Args:
            batch: Full batch dict
            indices: List of indices to extract (at sample level)

        Returns:
            Sliced batch dict with same structure
        """
        result = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                max_idx = max(indices) if indices else 0
                if max_idx >= value.shape[0]:
                    # Indices are at sample level but tensor is at pair level
                    # Convert sample indices to pair indices and deduplicate
                    pair_indices = sorted(set(i // 2 for i in indices))
                    # Ensure indices are within bounds
                    pair_indices = [i for i in pair_indices if i < value.shape[0]]
                    if pair_indices:
                        result[key] = value[pair_indices]
                    else:
                        result[key] = value[:0]  # Empty tensor with same dtype/device
                else:
                    result[key] = value[indices]
            elif isinstance(value, dict):
                # Handle nested dicts (like meta)
                result[key] = {
                    k: [v[i] for i in indices if i < len(v)] if isinstance(v, list) else v
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                result[key] = [value[i] for i in indices if i < len(value)]
            else:
                # Scalar or other types - keep as is
                result[key] = value
        return result

    def _slice_logits(
        self,
        logits: Tensor,
        indices: list,
    ) -> Tensor:
        """Extract logits subset at given indices.

        Handles both reward model ([B, 1] or [B]) and LM ([B, T, V]) logits.

        Args:
            logits: Full logits tensor
            indices: List of indices to extract

        Returns:
            Sliced logits tensor
        """
        if logits.ndim == 1:
            # [B] - scalar rewards
            return logits[indices]
        elif logits.ndim == 2:
            # [B, 1] or [B, T] - reward model or sequence logits
            return logits[indices]
        elif logits.ndim == 3:
            # [B, T, V] - language model logits
            return logits[indices]
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


def make_bradley_terry_with_irm(
    *,
    irm_regularizer: Optional[IRMRegularizer] = None,
    irm_weight: float = 1.0,
    irm_mode: str = "vrex",
    vrex_beta: float = 1.0,
    beta: float = 1.0,
    score_type: str = "reward",
    regularization_lambda: float = 0.0,
    regularization_type: str = "l2",
    label_smoothing: float = 0.0,
    score_regularization_lambda: float = 0.0,
    env_key: str = "env_id",
    name: str = "bradley_terry_irm",
) -> RLAlgorithm:
    """Create Bradley-Terry algorithm with IRM regularization for verifier training.

    This algorithm trains a reward model/verifier using Bradley-Terry preference
    learning, with IRM regularization to encourage invariance across training
    rounds (environments).

    In the PVG framework:
    - Each round generates new preference pairs from prover rollouts
    - IRM prevents the verifier from overfitting to round-specific patterns
    - This improves generalization to future rounds and OOD problems

    Args:
        irm_regularizer: Custom IRM regularizer. If None, creates based on irm_mode.
        irm_weight: Weight for IRM penalty in total loss. Higher = stronger regularization.
        irm_mode: IRM mode if irm_regularizer not provided:
            - "vrex": V-REx variance penalty (default, recommended)
            - "none": No IRM regularization (just standard BT)
        vrex_beta: Beta parameter for V-REx if using vrex mode. Controls penalty strength.
        beta: Bradley-Terry temperature scaling. Higher = sharper predictions.
        score_type: "reward" for reward models, "logprob" for DPO-style policy training.
        regularization_lambda: Margin regularization strength (penalizes large gaps).
        regularization_type: Type of margin regularization ("l2" or "l1").
        label_smoothing: Label smoothing factor (0.0 to 1.0). Smooths toward 0.5.
        score_regularization_lambda: L2 penalty on individual scores (prevents unbounded growth).
        env_key: Key in batch metadata for environment ID. Common values:
            - "env_id": Generic environment identifier
            - "round_id": PVG round number
            - "dataset_split": Data source split
        name: Algorithm name for logging.

    Returns:
        RLAlgorithm configured for Bradley-Terry + IRM training.

    Example:
        ```python
        from ludic.pvg.algorithm import make_bradley_terry_with_irm

        # Standard PVG verifier training
        algorithm = make_bradley_terry_with_irm(
            irm_mode="vrex",
            irm_weight=1.0,
            vrex_beta=1.0,
            env_key="round_id",
        )

        # With custom regularizer
        from ludic.pvg.irm import GradientPenaltyRegularizer
        custom_reg = GradientPenaltyRegularizer(beta=0.5)
        algorithm = make_bradley_terry_with_irm(
            irm_regularizer=custom_reg,
            irm_weight=0.5,
        )
        ```

    References:
        - Bradley-Terry: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
        - V-REx: Krueger et al. "Out-of-Distribution Generalization via Risk Extrapolation" (2021)
        - IRM: Arjovsky et al. "Invariant Risk Minimization" (2019)
    """
    # Create base BT loss
    bt_loss = BradleyTerryLoss(
        beta=beta,
        score_type=score_type,
        regularization_lambda=regularization_lambda,
        regularization_type=regularization_type,
        label_smoothing=label_smoothing,
        score_regularization_lambda=score_regularization_lambda,
    )

    # Create IRM regularizer if not provided
    if irm_regularizer is None:
        if irm_mode == "vrex":
            irm_regularizer = VRExRegularizer(beta=vrex_beta)
        elif irm_mode == "none":
            # No-op regularizer (beta=0 means no penalty)
            irm_regularizer = VRExRegularizer(beta=0.0)
        else:
            raise ValueError(
                f"Unknown irm_mode: {irm_mode!r}. "
                f"Valid options: 'vrex', 'none'."
            )

    # Create combined loss
    loss: Loss = BradleyTerryWithIRMLoss(
        bt_loss=bt_loss,
        irm_regularizer=irm_regularizer,
        irm_weight=irm_weight,
        env_key=env_key,
    )

    # Credit assignment is constant (BT uses label metadata, not weights)
    credit_assigner = ConstantCredit(value=1.0)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
    )

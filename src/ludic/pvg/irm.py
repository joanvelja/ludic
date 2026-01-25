"""IRM (Invariant Risk Minimization) regularizers for PVG.

IRM encourages learning features that are invariant across environments.
In the PVG context, environments correspond to different rounds of training data.

References:
    - Arjovsky et al. "Invariant Risk Minimization" (2019)
    - Krueger et al. "Out-of-Distribution Generalization via Risk Extrapolation (V-REx)" (2021)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class IRMRegularizer(Protocol):
    """Protocol for IRM (Invariant Risk Minimization) regularization.

    IRM encourages learning features that are invariant across environments.
    In PVG context, environments are different rounds of training data.
    """

    def compute_penalty(
        self,
        losses_per_env: Dict[str, Tensor],
        model: Optional[nn.Module] = None,
    ) -> Tensor:
        """Compute IRM penalty from per-environment losses.

        Args:
            losses_per_env: Mapping from environment ID to loss tensor
            model: Optional model reference (needed for gradient-based penalties)

        Returns:
            Scalar penalty tensor to add to total loss
        """
        ...


@dataclass
class VRExRegularizer:
    """V-REx (Variance Risk Extrapolation) regularizer.

    Penalizes variance of losses across environments:
        penalty = β * Var(losses_per_env)

    This encourages the model to have similar loss across all environments,
    preventing overfitting to any particular round's data distribution.

    Reference: Krueger et al. "Out-of-Distribution Generalization via Risk Extrapolation"

    Args:
        beta: Penalty coefficient. Higher = stronger regularization.
        eps: Small constant for numerical stability.
    """
    beta: float = 1.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")

    def compute_penalty(
        self,
        losses_per_env: Dict[str, Tensor],
        model: Optional[nn.Module] = None,
    ) -> Tensor:
        if not losses_per_env:
            # Return zero tensor on appropriate device
            return torch.tensor(0.0)

        # Stack all environment losses
        loss_values = list(losses_per_env.values())
        device = loss_values[0].device
        dtype = loss_values[0].dtype

        if len(loss_values) == 1:
            # Single environment: no variance
            return torch.tensor(0.0, device=device, dtype=dtype)

        losses_tensor = torch.stack([l.flatten().mean() for l in loss_values])

        # Compute variance of risks
        variance = losses_tensor.var(unbiased=False)

        return self.beta * variance


@dataclass
class GradientPenaltyRegularizer:
    """IRM v1 gradient penalty regularizer.

    Penalizes the gradient norm of the loss with respect to a dummy classifier:
        penalty = β * E_e[||∇_w (w * loss_e)|w=1||²]

    This is the original IRM formulation that encourages features to be
    sufficient for optimal prediction in each environment.

    Reference: Arjovsky et al. "Invariant Risk Minimization"

    Args:
        beta: Penalty coefficient.
    """
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")

    def compute_penalty(
        self,
        losses_per_env: Dict[str, Tensor],
        model: Optional[nn.Module] = None,
    ) -> Tensor:
        if not losses_per_env:
            return torch.tensor(0.0)

        loss_values = list(losses_per_env.values())
        device = loss_values[0].device
        dtype = loss_values[0].dtype

        total_penalty = torch.tensor(0.0, device=device, dtype=dtype)

        for env_id, loss in losses_per_env.items():
            # Create dummy classifier weight
            w = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)

            # Compute gradient of w * loss w.r.t. w
            scaled_loss = w * loss.mean()
            grad = torch.autograd.grad(
                scaled_loss,
                w,
                create_graph=True,
                retain_graph=True,
            )[0]

            # Add squared gradient norm to penalty
            total_penalty = total_penalty + grad.pow(2)

        return self.beta * total_penalty / len(losses_per_env)


@dataclass
class EnvironmentBalancedSampler:
    """Implicit IRM regularization through balanced sampling.

    Instead of a loss penalty, this provides balanced batching across environments.
    Each batch contains equal numbers of samples from each environment.

    This is not a regularizer in the loss sense, but provides functionality
    for creating balanced batches for training.

    Args:
        min_samples_per_env: Minimum samples required from each environment.
    """
    min_samples_per_env: int = 1

    def __post_init__(self) -> None:
        if self.min_samples_per_env < 1:
            raise ValueError(f"min_samples_per_env must be >= 1, got {self.min_samples_per_env}")

    def compute_penalty(
        self,
        losses_per_env: Dict[str, Tensor],
        model: Optional[nn.Module] = None,
    ) -> Tensor:
        """Returns zero - this class provides balanced sampling, not a loss penalty."""
        if not losses_per_env:
            return torch.tensor(0.0)
        device = list(losses_per_env.values())[0].device
        return torch.tensor(0.0, device=device)

    def create_balanced_indices(
        self,
        env_indices: Dict[str, List[int]],
        batch_size: int,
    ) -> List[int]:
        """Create balanced batch indices across environments.

        Args:
            env_indices: Mapping from env ID to list of sample indices
            batch_size: Desired batch size

        Returns:
            List of indices balanced across environments
        """
        if not env_indices:
            return []

        n_envs = len(env_indices)
        samples_per_env = batch_size // n_envs

        if samples_per_env < self.min_samples_per_env:
            samples_per_env = self.min_samples_per_env

        indices = []
        for env_id, env_idx_list in env_indices.items():
            # Sample with replacement if needed
            if len(env_idx_list) < samples_per_env:
                # Sample with replacement
                sampled = torch.randint(0, len(env_idx_list), (samples_per_env,)).tolist()
                indices.extend([env_idx_list[i] for i in sampled])
            else:
                # Random sample without replacement
                perm = torch.randperm(len(env_idx_list))[:samples_per_env].tolist()
                indices.extend([env_idx_list[i] for i in perm])

        return indices


@dataclass
class CombinedIRMRegularizer:
    """Combines multiple IRM regularizers.

    Useful for applying both V-REx and gradient penalty together.

    Args:
        regularizers: List of (name, regularizer, weight) tuples
    """
    regularizers: List[Tuple[str, IRMRegularizer, float]]

    def compute_penalty(
        self,
        losses_per_env: Dict[str, Tensor],
        model: Optional[nn.Module] = None,
    ) -> Tensor:
        if not losses_per_env:
            return torch.tensor(0.0)

        device = list(losses_per_env.values())[0].device
        dtype = list(losses_per_env.values())[0].dtype
        total = torch.tensor(0.0, device=device, dtype=dtype)

        for name, reg, weight in self.regularizers:
            penalty = reg.compute_penalty(losses_per_env, model)
            total = total + weight * penalty

        return total

from dataclasses import dataclass
from typing import Callable, Optional

from torch import nn


@dataclass
class TrainerConfig:
    """
    High-level trainer configuration.

    This is *purely* about optimization / model device / collation.
    Rollout and batch-generation config live in BatchSource / Orchestrator.

    ==========================
    Model / Optimization
    ==========================

    - model_device:
          Device where the trainable model lives ("cuda:0", "cpu", ...).
          Trainer will move *batches* to this device, but assumes the model
          is already on the correct device and/or FSDP-wrapped.

    - runtime_device:
          Device to move tensors to before pushing them into the serving
          runtime. If None, defaults to `model_device`.

    - lr, weight_decay, betas, eps:
          AdamW hyperparameters.

    - max_grad_norm:
          Gradient clipping threshold; None disables clipping.

    - grad_accum_steps:
          Number of micro-batches to accumulate gradients over before
          performing one optimizer step (the 'macro-step' size).

    - sync_every_steps:
          Frequency (in macro-steps) at which to push updated policy
          weights to the Agent's runtime (e.g., vLLM).

    ==========================
    FSDP2 Configuration
    ==========================

    - fsdp_enabled:
          If True, applies FSDP2 (`fully_shard()`) to the model internally
          during Trainer initialization. Requires distributed env to be set up.

    - fsdp_shard_fn:
          Optional callable that receives the model and applies `fully_shard()`
          to appropriate submodules. If None, a default sharding strategy is used
          that shards transformer blocks and the root model.

          Example:
              def custom_shard(model):
                  for layer in model.model.layers:
                      fully_shard(layer)
                  fully_shard(model)
                  return model

    - fsdp_param_dtype:
          Dtype for parameters during forward/backward computation (e.g., "bf16").
          Used to construct MixedPrecisionPolicy. None = full precision.

    - fsdp_reduce_dtype:
          Dtype for gradient reduction (e.g., "fp32" for better numerics).
          Used to construct MixedPrecisionPolicy. None = same as param_dtype.

    - fsdp_reshard_after_forward:
          If True (default), reshards parameters after forward pass (FULL_SHARD).
          If False, keeps parameters unsharded for backward (SHARD_GRAD_OP).

    ==========================
    Collation
    ==========================

    - pad_token_id:
          Used when padding sequences during SAW collation.

    ==========================
    Staleness Control (PipelineRL)
    ==========================

    - max_lag:
          Maximum allowed policy version lag for training samples.
          If set, samples older than (current_step - max_lag) are dropped.
    """

    # ----- model / optimization -------------------
    model_device: str = "cuda"
    runtime_device: Optional[str] = None

    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    max_grad_norm: Optional[float] = 1.0

    # ----- gradient accumulation / sync -----------
    grad_accum_steps: int = 16
    sync_every_steps: int = 1

    # ----- FSDP2 configuration --------------------
    fsdp_enabled: bool = False
    fsdp_shard_fn: Optional[Callable[[nn.Module], nn.Module]] = None
    fsdp_param_dtype: Optional[str] = "bf16"
    fsdp_reduce_dtype: Optional[str] = "fp32"
    fsdp_reshard_after_forward: bool = True

    # ----- collation ------------------------------
    pad_token_id: int = 0

    # ----- staleness control ----------------------
    max_lag: Optional[int] = None

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerConfig:
    """
    High-level trainer configuration.

    ==========================
    Model / Optimization
    ==========================

    - model_device:
          Device where the trainable model lives ("cuda", "cuda:0", "cpu", ...).

    - runtime_device:
          Device to move tensors to before pushing them into the serving runtime.
          If None, defaults to `model_device`.

    - lr:
          Learning rate for AdamW.

    - weight_decay:
          AdamW L2 regularization coefficient.

    - betas:
          AdamW betas tuple (beta1, beta2).

    - eps:
          Numerical stability constant for AdamW.

    - max_grad_norm:
          Gradient clipping threshold; None disables clipping.

    ==========================
    Padding / collation
    ==========================

    - pad_token_id:
          Used when padding sequences during SAW collation.

    Everything else about rollouts/sampling/envs now lives in the BatchSource
    and Orchestrator layers, not in the trainer config.
    """

    # ----- model / optimization -------------------
    model_device: str = "cuda"
    runtime_device: Optional[str] = None

    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    max_grad_norm: Optional[float] = 1.0

    # ----- padding / collation --------------------
    pad_token_id: int = 0

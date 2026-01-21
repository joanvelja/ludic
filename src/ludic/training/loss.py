from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from beartype.typing import Any, Dict, Mapping, Protocol, Tuple, List, Optional

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
import torch
from torch import Tensor
import torch.nn.functional as F


Batch = Mapping[str, Tensor]
Logits = Float[Tensor, "B T V"]
TokenIds = Int[Tensor, "B T"]
Mask = Float[Tensor, "B T"]
Weights = Float[Tensor, "B"]

logger = logging.getLogger(__name__)


def _no_op(fn):
    return fn


_TYPECHECK_ENABLED = os.getenv("LUDIC_TYPECHECK", "0") == "1"
typechecker = beartype if _TYPECHECK_ENABLED else _no_op
logger.info(
    "Jaxtyping runtime checks: %s",
    "enabled (beartype)" if _TYPECHECK_ENABLED else "disabled",
)


# ---------------------------------------------------------------------------
# Shared context for memory-efficient loss composition
# ---------------------------------------------------------------------------


class SharedContext:
    """
    Lazy-computed shared tensors for memory-efficient loss composition.

    When multiple losses are combined via CompositeLoss, each typically needs
    the same expensive intermediate tensors (e.g., token_logp from log_softmax).
    Without sharing, each loss computes these independently, creating separate
    autograd graphs that store duplicate [B, T, V] activations for backward.

    SharedContext solves this by computing expensive tensors ONCE on first access
    and caching them for subsequent uses. All losses receive the same tensor
    objects, sharing a single autograd graph.

    Memory savings example (7B model, V=32K, B=8, T=4096):
        - Without sharing (2 losses): 2× [B, T, V] ≈ 4GB activations
        - With sharing (2 losses):    1× [B, T, V] ≈ 2GB activations

    Usage:
        # Created by CompositeLoss, passed to each child loss
        shared = SharedContext(logits, batch)

        # In CISPOLoss.compute():
        token_logp = shared.token_logp  # Computes and caches on first access

        # In TokenKLLoss.compute():
        token_logp = shared.token_logp  # Returns cached tensor (same object!)

    Note: Properties that depend on batch["actor_logps"] will raise KeyError
    if that key is missing. This is intentional - not all loss combinations
    need actor logprobs.
    """

    __slots__ = ("logits", "batch", "_cache")

    def __init__(self, logits: Logits, batch: Batch) -> None:
        self.logits = logits
        self.batch = batch
        self._cache: Dict[str, Tensor] = {}

    @property
    def input_ids(self) -> TokenIds:
        """Token IDs from batch (not cached, just a convenience accessor)."""
        return self.batch["input_ids"]

    @property
    def action_mask(self) -> Mask:
        """Action mask from batch (not cached, just a convenience accessor)."""
        return self.batch["action_mask"]

    @property
    def token_logp(self) -> Float[Tensor, "B T-1"]:
        """
        Per-token log probabilities: log π(a_t|s_t) for each position.

        THIS IS THE EXPENSIVE OPERATION - calls selective_log_softmax which
        requires storing [B, T, V] activations for backward. Caching this
        is the primary memory optimization.
        """
        if "token_logp" not in self._cache:
            # Import here to avoid circular dependency (compute_token_logp defined later)
            self._cache["token_logp"] = compute_token_logp(self.logits, self.input_ids)
        return self._cache["token_logp"]

    @property
    def token_mask(self) -> Float[Tensor, "B T-1"]:
        """Action mask aligned with token_logp (shifted by 1 for next-token prediction)."""
        if "token_mask" not in self._cache:
            self._cache["token_mask"] = self.action_mask[:, 1:].to(
                self.token_logp.dtype
            )
        return self._cache["token_mask"]

    @property
    def token_counts(self) -> Float[Tensor, "B"]:
        """Number of action tokens per sample (for length normalization)."""
        if "token_counts" not in self._cache:
            self._cache["token_counts"] = self.token_mask.sum(dim=-1).clamp(min=1.0)
        return self._cache["token_counts"]

    @property
    def actor_logps_shifted(self) -> Float[Tensor, "B T-1"]:
        """
        Behavior policy log probs aligned with token_logp.

        Raises:
            KeyError: If batch["actor_logps"] is not present.
        """
        if "actor_logps_shifted" not in self._cache:
            if "actor_logps" not in self.batch:
                raise KeyError(
                    "SharedContext.actor_logps_shifted requires batch['actor_logps']. "
                    "Ensure your rollouts include actor_logps for ratio-based objectives."
                )
            self._cache["actor_logps_shifted"] = self.batch["actor_logps"][:, 1:]
        return self._cache["actor_logps_shifted"]

    @property
    def log_ratio(self) -> Float[Tensor, "B T-1"]:
        """Log importance ratio: log(π_new/π_old) per token."""
        if "log_ratio" not in self._cache:
            self._cache["log_ratio"] = self.token_logp - self.actor_logps_shifted
        return self._cache["log_ratio"]

    @property
    def ratio(self) -> Float[Tensor, "B T-1"]:
        """Importance ratio: π_new/π_old per token."""
        if "ratio" not in self._cache:
            self._cache["ratio"] = torch.exp(self.log_ratio)
        return self._cache["ratio"]

    def logp_action(self, *, length_normalize: bool = False) -> Float[Tensor, "B"]:
        """
        Sequence-level log probability (sum over action tokens).

        Unlike token_logp, this is a cheap derivation that doesn't require
        additional [B, T, V] storage. The length_normalize flag controls
        whether to divide by number of action tokens.

        Args:
            length_normalize: If True, return mean log prob instead of sum.

        Returns:
            [B] tensor of per-sample log probabilities.
        """
        masked_logp = (self.token_logp * self.token_mask).sum(dim=-1)
        if length_normalize:
            return masked_logp / self.token_counts
        return masked_logp


class Loss(Protocol):
    """
    Generic loss: given model outputs (logits) and a collated batch, return
    (scalar_loss, stats).

    Stats values must be scalar tensors (on the same device as logits) to enable
    batched D2H transfer during aggregation. Use `tensor.detach()` not `float(...)`.
    """

    def compute(
        self, logits: Logits, batch: Batch
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...


# We define this as a standalone helper so torch.compile can cache it cleanly.
# dynamic=True is critical for varying sequence lengths (preventing recompilation).
@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(logits: Logits, index: TokenIds) -> Float[Tensor, "B T"]:
    """
    Fused kernel for log_softmax + gather.

    Inductor (torch.compile) generates a kernel that computes the log_softmax
    normalization term and selects the target token in a single pass.
    This avoids materializing the massive [B, T, V] probability tensor in VRAM.
    """
    # This looks naive, but the compiler fuses it into a single read/write op.
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
def compute_logp_action(
    logits: Logits,
    input_ids: TokenIds,
    action_mask: Mask,
    *,
    length_normalize: bool = False,
) -> Weights:
    """
    Compute log π(a|s) given token-level logits and an action mask.

    Args:
        logits: [B, T, V] float tensor of unnormalized logits.
        input_ids: [B, T] long tensor of token ids actually sampled.
        action_mask: [B, T] {0,1} mask; 1 where tokens belong to the "action".

    Returns:
        logp_action: [B] log-prob of the entire action sequence per sample.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, T, V], got {tuple(logits.shape)}")

    if input_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"Shape mismatch: input_ids {input_ids.shape} vs logits {logits.shape}"
        )

    # Shift for causal LM: logits[t] predicts input_ids[t+1]
    if logits.size(1) < 2:
        raise ValueError("Sequence too short to compute next-token logprobs.")
    logits_shifted = logits[:, :-1, :]  # [B, T-1, V]
    target_ids = input_ids[:, 1:]  # [B, T-1]
    action_mask_shifted = action_mask[:, 1:]  # [B, T-1]

    # Use the compiled fused kernel on aligned targets
    token_logp = selective_log_softmax(logits_shifted, target_ids)

    # Sum log-probs over the action region only: [B]
    amask = action_mask_shifted.to(token_logp.dtype)
    logp_action = (token_logp * amask).sum(dim=-1)

    if length_normalize:
        lengths = amask.sum(dim=-1).clamp(min=1.0)
        logp_action = logp_action / lengths

    return logp_action


@jaxtyped(typechecker=typechecker)
def compute_token_logp(
    logits: Logits,
    input_ids: TokenIds,
) -> Float[Tensor, "B T-1"]:
    """
    Compute per-token log π(a_t|s_t) for each next-token prediction.

    Returns:
        token_logp: [B, T-1] log-prob of each next token.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, T, V], got {tuple(logits.shape)}")
    if input_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"Shape mismatch: input_ids {input_ids.shape} vs logits {logits.shape}"
        )
    if logits.size(1) < 2:
        raise ValueError("Sequence too short to compute next-token logprobs.")

    logits_shifted = logits[:, :-1, :]  # [B, T-1, V]
    target_ids = input_ids[:, 1:]  # [B, T-1]
    return selective_log_softmax(logits_shifted, target_ids)


# ---------------------------------------------------------------------------
# REINFORCE family
# ---------------------------------------------------------------------------


@dataclass
class ReinforceLoss:
    """
    REINFORCE with importance sampling correction:

        loss = - E[ sg(r) * A * log π(a|s) ]

    where:
        - A is taken from `batch["weight"]`
        - r = π_new(a|s) / π_old(a|s) is the IS ratio
        - sg(·) is stop-gradient

    The IS correction accounts for the fact that in LLM-RL we are always
    slightly off-policy due to different inference vs training kernels.

    Expects batch[old_logp_key] containing log π_old(a|s).
    """

    length_normalize: bool = False
    old_logp_key: str = "old_logp_action"

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
        *,
        shared: Optional[SharedContext] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        advantages = batch["weight"]  # [B]

        if self.old_logp_key not in batch:
            raise KeyError(f"ReinforceLoss requires '{self.old_logp_key}' in batch.")

        # Use shared context if available (memory-efficient composition)
        if shared is not None:
            logp_action = shared.logp_action(length_normalize=self.length_normalize)
            token_counts = shared.token_counts
        else:
            # Backward-compatible: compute locally
            input_ids = batch["input_ids"]  # [B, T]
            action_mask = batch["action_mask"]  # [B, T]
            logp_action = compute_logp_action(
                logits, input_ids, action_mask, length_normalize=self.length_normalize
            )  # [B]
            token_counts = action_mask[:, 1:].sum(dim=-1).clamp(min=1.0)

        old_logp = batch[self.old_logp_key]  # [B]
        if self.length_normalize:
            lengths = token_counts.to(old_logp.dtype)
            old_logp = old_logp / lengths

        log_ratio = logp_action - old_logp
        ratio = torch.exp(log_ratio)
        # KL approximation for stats: r - log(r) - 1
        mismatch_kl = ratio - log_ratio - 1.0

        # Stop-gradient on IS weight: gradient flows only through log π
        loss = -(ratio.detach() * advantages * logp_action).mean()

        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std(unbiased=False).detach(),
            "kl_actor_policy": mismatch_kl.mean().detach(),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# Masked token-level CE (SFT-friendly)
# ---------------------------------------------------------------------------


@dataclass
class MaskedCausalLMCrossEntropyLoss:
    """
    Token-level masked cross entropy over the "action" region.

    This is the standard SFT objective when you have (prompt, completion)
    and want to train only on the completion tokens.

    Expects:
      - batch["input_ids"]:   [B, T]
      - batch["action_mask"]: [B, T] 0/1 mask where 1 marks completion tokens
      - batch["weight"]:      [B] optional per-sample weights (defaults to 1.0)
    """

    length_normalize: bool = True

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]  # [B, T]
        action_mask = batch["action_mask"]  # [B, T]
        weights = batch.get("weight")

        if logits.ndim != 3:
            raise ValueError(f"Expected logits [B, T, V], got {tuple(logits.shape)}")
        if input_ids.shape != logits.shape[:2]:
            raise ValueError(
                f"Shape mismatch: input_ids {input_ids.shape} vs logits {logits.shape}"
            )

        if logits.size(1) < 2:
            raise ValueError("Sequence too short to compute next-token loss.")

        # Shift for causal LM: logits[t] predicts input_ids[t+1]
        logits_shifted = logits[:, :-1, :].float()  # [B, T-1, V]
        targets = input_ids[:, 1:]  # [B, T-1]
        mask = action_mask[:, 1:].to(dtype=torch.float32)  # [B, T-1]

        B, Tm1, V = logits_shifted.shape
        per_token_nll = F.cross_entropy(
            logits_shifted.reshape(B * Tm1, V),
            targets.reshape(B * Tm1),
            reduction="none",
        ).reshape(B, Tm1)

        token_counts = mask.sum(dim=-1).clamp(min=1.0)  # [B]
        per_sample_nll = (per_token_nll * mask).sum(dim=-1)  # [B]
        if self.length_normalize:
            per_sample_nll = per_sample_nll / token_counts

        if weights is not None:
            loss = (per_sample_nll * weights.to(per_sample_nll.dtype)).mean()
        else:
            loss = per_sample_nll.mean()

        # Stats for parity with ReinforceLoss
        per_sample_logp = -per_sample_nll
        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "logp_mean": per_sample_logp.mean().detach(),
            "nll_mean": per_sample_nll.mean().detach(),
            "avg_action_tokens": token_counts.mean().detach(),
        }
        return loss, stats


@dataclass
class ReinforceBaselineLoss:
    """
    REINFORCE with batch-mean baseline and importance sampling correction:

        A_i = adv_i - mean(adv)
        loss = - E[ sg(r) * A_i * log π(a_i|s_i) ]

    where:
        - adv_i is batch["weight"]
        - r = π_new(a|s) / π_old(a|s) is the IS ratio
        - sg(·) is stop-gradient

    Expects batch[old_logp_key] containing log π_old(a|s).
    """

    normalize: bool = False
    length_normalize: bool = False
    old_logp_key: str = "old_logp_action"

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
        *,
        shared: Optional[SharedContext] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        adv_raw = batch["weight"]  # [B]

        if self.old_logp_key not in batch:
            raise KeyError(
                f"ReinforceBaselineLoss requires '{self.old_logp_key}' in batch."
            )

        # Use shared context if available (memory-efficient composition)
        if shared is not None:
            logp_action = shared.logp_action(length_normalize=self.length_normalize)
            token_counts = shared.token_counts
        else:
            # Backward-compatible: compute locally
            input_ids = batch["input_ids"]
            action_mask = batch["action_mask"]
            logp_action = compute_logp_action(
                logits, input_ids, action_mask, length_normalize=self.length_normalize
            )  # [B]
            token_counts = action_mask[:, 1:].sum(dim=-1).clamp(min=1.0)

        old_logp = batch[self.old_logp_key]  # [B]
        if self.length_normalize:
            lengths = token_counts.to(old_logp.dtype)
            old_logp = old_logp / lengths

        log_ratio = logp_action - old_logp
        ratio = torch.exp(log_ratio)
        mismatch_kl = ratio - log_ratio - 1.0

        baseline = adv_raw.mean()
        advantages = adv_raw - baseline

        if self.normalize:
            std = advantages.std(unbiased=False)
            advantages = advantages / (std + 1e-8)

        # Stop-gradient on IS weight
        loss = -(ratio.detach() * advantages * logp_action).mean()

        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "baseline": baseline.detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std(unbiased=False).detach(),
            "kl_actor_policy": mismatch_kl.mean().detach(),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# PPO clipped policy loss (no value term here)
# ---------------------------------------------------------------------------


@dataclass
class ClippedSurrogateLoss:
    """
    PPO-style clipped surrogate policy loss (actor part only).

        r = π_new(a|s) / π_old(a|s)
        L_clip = - E[ min(r * A, clip(r, 1 - eps_low, 1 + eps_high) * A) ]

    Expects:
        - batch["weight"]:       A  (advantages)      [B]
        - batch[old_logp_key]:   log π_old(a|s)      [B]
        - input_ids / attention_mask / action_mask for π_new.

    Defaults follow the GSPO paper clip ranges:
    https://arxiv.org/abs/2507.18071
    """

    clip_eps_low: float = 3e-4
    clip_eps_high: float = 4e-4
    old_logp_key: str = "old_logp_action"
    length_normalize: bool = False
    ratio_clip: Optional[float] = None

    def __post_init__(self) -> None:
        if self.clip_eps_low < 0 or self.clip_eps_high < 0:
            raise ValueError(
                f"clip_eps_low/high must be non-negative, got {self.clip_eps_low}, {self.clip_eps_high}"
            )
        if self.ratio_clip is not None and self.ratio_clip <= 0:
            raise ValueError(f"ratio_clip must be positive, got {self.ratio_clip}")

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
        *,
        shared: Optional[SharedContext] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        advantages = batch["weight"]  # [B]
        if self.old_logp_key not in batch:
            raise KeyError(
                f"ClippedSurrogateLoss requires '{self.old_logp_key}' in batch."
            )

        # Use shared context if available (memory-efficient composition)
        if shared is not None:
            logp_action = shared.logp_action(length_normalize=self.length_normalize)
            token_counts = shared.token_counts
        else:
            # Backward-compatible: compute locally
            input_ids = batch["input_ids"]
            action_mask = batch["action_mask"]
            logp_action = compute_logp_action(
                logits,
                input_ids,
                action_mask,
                length_normalize=self.length_normalize,
            )  # [B]
            token_counts = action_mask[:, 1:].sum(dim=-1).clamp(min=1.0)

        old_logp = batch[self.old_logp_key]  # [B]
        if self.length_normalize:
            lengths = token_counts.to(old_logp.dtype)
            old_logp = old_logp / lengths

        log_ratio = logp_action - old_logp
        ratio_raw = torch.exp(log_ratio)
        mismatch_kl = ratio_raw - log_ratio - 1.0
        ratio = ratio_raw
        if self.ratio_clip is not None:
            ratio = torch.clamp(ratio, max=self.ratio_clip)

        unclipped = ratio * advantages
        clipped = (
            torch.clamp(ratio, 1.0 - self.clip_eps_low, 1.0 + self.clip_eps_high)
            * advantages
        )

        obj = torch.min(unclipped, clipped)
        loss = -obj.mean()

        ppo_clip_frac = (
            ((ratio > 1.0 + self.clip_eps_high) | (ratio < 1.0 - self.clip_eps_low))
            .float()
            .mean()
        )
        if self.ratio_clip is not None:
            ratio_clip_frac = (ratio >= self.ratio_clip).float().mean()
        else:
            ratio_clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)

        stats = {
            "loss": loss.detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std(unbiased=False).detach(),
            "clip_frac": ppo_clip_frac.detach(),
            "ratio_clip_frac": ratio_clip_frac.detach(),
            "kl_actor_policy": mismatch_kl.mean().detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
        }
        return loss, stats


@dataclass
class CISPOLoss:
    """
    CISPO (Clipped IS-weight Policy Optimization) loss.

    Unlike PPO/GRPO which clip the policy update via the min operation,
    CISPO clips the importance sampling weight itself and uses it as a
    stop-gradient multiplier on the REINFORCE objective. This preserves
    gradient contributions from all tokens, especially low-probability
    tokens crucial for reflective reasoning (e.g., "However", "Recheck").

    Loss:
        L = - E[ sg(clip(r_t, 1-ε_low, 1+ε_high)) * A * log π(a_t|s_t) ]

    Where:
        - r_t = π_new(a_t|s_t) / π_old(a_t|s_t) is the importance sampling weight
        - sg(·) is stop-gradient (weight does not receive gradients)
        - A is the advantage (from batch["weight"])

    Expects:
        - batch["weight"]:       A (advantages)       [B]
        - batch["actor_logps"]:  token logps under behavior policy [B, T]
        - input_ids / attention_mask / action_mask for π_new.

    Reference: MiniMax-M1 paper (arXiv:2506.13585)
    """

    clip_eps_low: float = 1e6  # Effectively no lower bound (paper setting)
    clip_eps_high: float = 0.2
    length_normalize: bool = True

    def __post_init__(self) -> None:
        if self.clip_eps_high < 0:
            raise ValueError(
                f"clip_eps_high must be non-negative, got {self.clip_eps_high}"
            )

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
        *,
        shared: Optional[SharedContext] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        advantages = batch["weight"]  # [B]

        # Use shared context if available (memory-efficient composition)
        if shared is not None:
            token_logp = shared.token_logp
            token_mask = shared.token_mask
            token_counts = shared.token_counts
            actor_logps_shifted = shared.actor_logps_shifted
            log_ratio = shared.log_ratio
            ratio = shared.ratio
        else:
            # Backward-compatible: compute locally
            input_ids = batch["input_ids"]
            action_mask = batch["action_mask"]

            if "actor_logps" not in batch:
                raise KeyError(
                    "CISPOLoss requires batch['actor_logps'] for importance sampling."
                )

            actor_logps = batch["actor_logps"]  # [B, T]
            if actor_logps.shape != input_ids.shape:
                raise ValueError(
                    f"actor_logps shape {tuple(actor_logps.shape)} does not match input_ids "
                    f"{tuple(input_ids.shape)}."
                )

            token_logp = compute_token_logp(logits, input_ids)  # [B, T-1]
            token_mask = action_mask[:, 1:].to(token_logp.dtype)  # [B, T-1]
            token_counts = token_mask.sum(dim=-1).clamp(min=1.0)  # [B]
            actor_logps_shifted = actor_logps[:, 1:]  # [B, T-1]
            log_ratio = token_logp - actor_logps_shifted
            ratio = torch.exp(log_ratio)

        # CISPO: Clip the IS weight, not the update
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_eps_low,
            1.0 + self.clip_eps_high,
        )

        # Stop-gradient on the clipped IS weight
        # The gradient only flows through log π, not through the IS weight
        is_weight = clipped_ratio.detach()

        # REINFORCE loss with IS weight correction
        # L = - E[is_weight * A * log π]
        adv = advantages.unsqueeze(-1)  # [B, 1]
        weighted_logp = is_weight * adv * token_logp * token_mask  # [B, T-1]

        per_sample_obj = weighted_logp.sum(dim=-1)  # [B]
        if self.length_normalize:
            per_sample_obj = per_sample_obj / token_counts

        loss = -per_sample_obj.mean()

        # Stats
        mask = token_mask > 0
        if mask.any():
            ratio_vals = ratio.masked_select(mask)
            ratio_mean = ratio_vals.mean()
            ratio_std = ratio_vals.std(unbiased=False)
            # Fraction of tokens where IS weight was clipped
            clip_frac = (
                (
                    (ratio_vals > 1.0 + self.clip_eps_high)
                    | (ratio_vals < 1.0 - self.clip_eps_low)
                )
                .float()
                .mean()
            )
            # KL approximation: r - log(r) - 1
            token_kl = ratio_vals - log_ratio.masked_select(mask) - 1.0
            mismatch_kl = token_kl.mean()
        else:
            ratio_mean = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            ratio_std = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            mismatch_kl = torch.zeros((), device=ratio.device, dtype=ratio.dtype)

        logp_action = (token_logp * token_mask).sum(dim=-1)
        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "ratio_mean": ratio_mean.detach(),
            "ratio_std": ratio_std.detach(),
            "clip_frac": clip_frac.detach(),
            "kl_actor_policy": mismatch_kl.detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
            "avg_action_tokens": token_counts.mean().detach(),
        }
        return loss, stats


@dataclass
class TokenKLLoss:
    """
    Token-level KL penalty between π_new and the behavior policy (actor).

    Uses token-level log probabilities from batch["actor_logps"] to compute:

        KL(π_new || π_actor) ≈ mean_t[ log π_new(a_t|s_t) - log π_actor(a_t|s_t) ]

    This is designed to be consistent with CISPO's token-level approach,
    where we want to penalize the policy for deviating too far from the
    behavior policy at each token position.

    Loss is:
        loss = coeff * mean(kl)

    Args:
        coeff: KL penalty coefficient. Positive values penalize divergence.
        length_normalize: If True, normalize by number of action tokens per sample
            before averaging across samples.

    Expects:
        - batch["actor_logps"]: [B, T] log probs under behavior policy
        - batch["action_mask"]: [B, T] mask for action tokens
        - batch["input_ids"]: [B, T] token ids

    Reference: Useful for stabilizing CISPO training (MiniMax-M1, arXiv:2506.13585)
    """

    coeff: float = 0.01
    length_normalize: bool = True

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
        *,
        shared: Optional[SharedContext] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        # Use shared context if available (memory-efficient composition)
        if shared is not None:
            token_logp = shared.token_logp
            token_mask = shared.token_mask
            token_counts = shared.token_counts
            actor_logps_shifted = shared.actor_logps_shifted
            token_kl = shared.log_ratio  # log π_new - log π_actor
        else:
            # Backward-compatible: compute locally
            input_ids = batch["input_ids"]
            action_mask = batch["action_mask"]

            if "actor_logps" not in batch:
                raise KeyError(
                    "TokenKLLoss requires batch['actor_logps'] for KL computation."
                )

            actor_logps = batch["actor_logps"]  # [B, T]
            if actor_logps.shape != input_ids.shape:
                raise ValueError(
                    f"actor_logps shape {tuple(actor_logps.shape)} does not match input_ids "
                    f"{tuple(input_ids.shape)}."
                )

            token_logp = compute_token_logp(logits, input_ids)  # [B, T-1]
            token_mask = action_mask[:, 1:].to(token_logp.dtype)  # [B, T-1]
            token_counts = token_mask.sum(dim=-1).clamp(min=1.0)  # [B]
            actor_logps_shifted = actor_logps[:, 1:]  # [B, T-1]
            token_kl = token_logp - actor_logps_shifted  # [B, T-1]

        # Masked sum per sample
        per_sample_kl = (token_kl * token_mask).sum(dim=-1)  # [B]
        if self.length_normalize:
            per_sample_kl = per_sample_kl / token_counts

        mean_kl = per_sample_kl.mean()
        loss = self.coeff * mean_kl

        # Stats
        mask = token_mask > 0
        if mask.any():
            kl_vals = token_kl.masked_select(mask)
            kl_mean = kl_vals.mean()
            kl_std = kl_vals.std(unbiased=False)
            # Also compute ratio stats for monitoring
            ratio = torch.exp(token_kl)
            ratio_vals = ratio.masked_select(mask)
            ratio_mean = ratio_vals.mean()
            ratio_std = ratio_vals.std(unbiased=False)
        else:
            kl_mean = torch.zeros((), device=token_kl.device, dtype=token_kl.dtype)
            kl_std = torch.zeros((), device=token_kl.device, dtype=token_kl.dtype)
            ratio_mean = torch.zeros((), device=token_kl.device, dtype=token_kl.dtype)
            ratio_std = torch.zeros((), device=token_kl.device, dtype=token_kl.dtype)

        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "kl_mean": kl_mean.detach(),
            "kl_std": kl_std.detach(),
            "ratio_mean": ratio_mean.detach(),
            "ratio_std": ratio_std.detach(),
            "avg_action_tokens": token_counts.mean().detach(),
        }
        return loss, stats


@dataclass
class TokenClippedSurrogateLoss:
    """
    Token-level PPO-style clipped surrogate loss (Token-TIS-style ratios).

    Uses asymmetric clipping: clip(r, 1 - eps_low, 1 + eps_high).

    Expects:
        - batch["weight"]:       A  (advantages)      [B]
        - batch["actor_logps"]:  token logps under behavior policy [B, T]
        - input_ids / attention_mask / action_mask for π_new.

    Defaults follow the GSPO paper clip ranges for the GRPO baseline:
    https://arxiv.org/abs/2507.18071
    """

    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.27
    length_normalize: bool = False
    ratio_clip: Optional[float] = None

    def __post_init__(self) -> None:
        if self.clip_eps_low < 0 or self.clip_eps_high < 0:
            raise ValueError(
                f"clip_eps_low/high must be non-negative, got {self.clip_eps_low}, {self.clip_eps_high}"
            )
        if self.ratio_clip is not None and self.ratio_clip <= 0:
            raise ValueError(f"ratio_clip must be positive, got {self.ratio_clip}")

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
        *,
        shared: Optional[SharedContext] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        advantages = batch["weight"]

        # Use shared context if available (memory-efficient composition)
        if shared is not None:
            token_logp = shared.token_logp
            token_mask = shared.token_mask
            token_counts = shared.token_counts
            log_ratio = shared.log_ratio
            ratio_raw = shared.ratio
        else:
            # Backward-compatible: compute locally
            input_ids = batch["input_ids"]
            action_mask = batch["action_mask"]
            if "actor_logps" not in batch:
                raise KeyError(
                    "TokenClippedSurrogateLoss requires batch['actor_logps'] for token IS."
                )

            actor_logps = batch["actor_logps"]
            if actor_logps.shape != input_ids.shape:
                raise ValueError(
                    f"actor_logps shape {tuple(actor_logps.shape)} does not match input_ids "
                    f"{tuple(input_ids.shape)}."
                )

            token_logp = compute_token_logp(logits, input_ids)  # [B, T-1]
            token_mask = action_mask[:, 1:].to(token_logp.dtype)
            token_counts = token_mask.sum(dim=-1).clamp(min=1.0)
            actor_logps_shifted = actor_logps[:, 1:]
            log_ratio = token_logp - actor_logps_shifted
            ratio_raw = torch.exp(log_ratio)

        token_mismatch_kl = ratio_raw - log_ratio - 1.0
        ratio = ratio_raw
        if self.ratio_clip is not None:
            ratio = torch.clamp(ratio, max=self.ratio_clip)

        ratio_clipped = torch.clamp(
            ratio, 1.0 - self.clip_eps_low, 1.0 + self.clip_eps_high
        )
        adv = advantages.unsqueeze(-1)
        unclipped = ratio * adv
        clipped = ratio_clipped * adv
        obj = torch.min(unclipped, clipped) * token_mask
        per_sample_obj = obj.sum(dim=-1)
        if self.length_normalize:
            per_sample_obj = per_sample_obj / token_counts

        loss = -per_sample_obj.mean()

        mask = token_mask > 0
        if mask.any():
            ratio_vals = ratio.masked_select(mask)
            ppo_clip_frac = (
                (
                    (ratio_vals > 1.0 + self.clip_eps_high)
                    | (ratio_vals < 1.0 - self.clip_eps_low)
                )
                .float()
                .mean()
            )
            ratio_mean = ratio_vals.mean()
            ratio_std = ratio_vals.std(unbiased=False)
            mismatch_kl = token_mismatch_kl.masked_select(mask).mean()
            if self.ratio_clip is not None:
                ratio_clip_frac = (ratio_vals >= self.ratio_clip).float().mean()
            else:
                ratio_clip_frac = torch.zeros(
                    (), device=ratio.device, dtype=ratio.dtype
                )
        else:
            ratio_mean = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            ratio_std = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            ppo_clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            ratio_clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            mismatch_kl = torch.zeros((), device=ratio.device, dtype=ratio.dtype)

        logp_action = (token_logp * token_mask).sum(dim=-1)
        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "ratio_mean": ratio_mean.detach(),
            "ratio_std": ratio_std.detach(),
            "clip_frac": ppo_clip_frac.detach(),
            "ratio_clip_frac": ratio_clip_frac.detach(),
            "kl_actor_policy": mismatch_kl.detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
            "avg_action_tokens": token_counts.mean().detach(),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# KL penalty and entropy bonus
# ---------------------------------------------------------------------------


@dataclass
class KLLoss:
    """
    KL penalty between π_new and a reference policy whose log-prob is stored as
    batch[old_logp_key].

    We use the standard policy-gradient surrogate estimate:

        KL(π_new || π_old) ≈ E_{a ~ π_new} [ log π_new(a|s) - log π_old(a|s) ]

    Loss is:

        loss = coeff * mean(kl)

    (You usually *add* this to the overall loss; coeff > 0 makes it a penalty.)
    """

    coeff: float = 1.0
    old_logp_key: str = "old_logp_action"
    length_normalize: bool = False

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
        *,
        shared: Optional[SharedContext] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        old_logp = batch[self.old_logp_key]  # [B]

        # Use shared context if available (memory-efficient composition)
        if shared is not None:
            logp_new = shared.logp_action(length_normalize=self.length_normalize)
            token_counts = shared.token_counts
        else:
            # Backward-compatible: compute locally
            input_ids = batch["input_ids"]
            action_mask = batch["action_mask"]
            logp_new = compute_logp_action(
                logits,
                input_ids,
                action_mask,
                length_normalize=self.length_normalize,
            )  # [B]
            token_counts = action_mask[:, 1:].sum(dim=-1).clamp(min=1.0)

        if self.length_normalize:
            lengths = token_counts.to(old_logp.dtype)
            old_logp = old_logp / lengths

        kl = logp_new - old_logp  # [B]
        loss = self.coeff * kl.mean()

        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "kl_mean": kl.mean().detach(),
            "kl_std": kl.std(unbiased=False).detach(),
        }
        return loss, stats


@dataclass
class EntropyBonus:
    """
    Entropy bonus over the action region.

    Computes token-level entropy H(π(·|token)) and averages over tokens where
    action_mask == 1. Loss is:

        loss = - coeff * mean_entropy

    So with coeff > 0, this *reduces* the total loss (encourages exploration).
    """

    coeff: float = 0.01

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]

        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)

        # token entropy: [B, T]
        token_entropy = -(probs * logprobs).sum(dim=-1)

        mask = action_mask.to(token_entropy.dtype)

        masked_entropy = token_entropy * mask  # [B, T]
        # avoid divide-by-zero if mask is all zeros
        denom = mask.sum()
        if denom.item() == 0:
            mean_entropy = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            mean_entropy = masked_entropy.sum() / denom

        loss = -self.coeff * mean_entropy

        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "entropy_mean": mean_entropy.detach(),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


@dataclass
class LossTerm:
    """
    Single term inside a CompositeLoss.

    - name:   short identifier for logging
    - loss:   loss object implementing Loss protocol
    - weight: scalar multiplier applied to that loss
    """

    name: str
    loss: Loss
    weight: float = 1.0


@dataclass
class CompositeLoss:
    """
    Combine multiple Loss terms into a single scalar loss:

        total_loss = sum_i weight_i * loss_i

    Stats are merged with hierarchical keys:

        "{name}/loss", "{name}/<stat_key>", ...

    and a top-level "loss" key for the final combined loss.

    Memory Efficiency:
        CompositeLoss automatically creates a SharedContext to cache expensive
        intermediate tensors (like token_logp from log_softmax). All child losses
        receive the same SharedContext, enabling them to share the cached tensors
        and avoiding duplicate autograd graphs.

        Without SharedContext: N losses → N× [B, T, V] autograd activations
        With SharedContext:    N losses → 1× [B, T, V] autograd activations

    Note:
        All child losses must accept a `shared` keyword argument of type
        Optional[SharedContext]. Losses that don't support this will cause
        a TypeError with a helpful message.
    """

    terms: List[LossTerm]

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        if not self.terms:
            raise ValueError("CompositeLoss.terms must be non-empty")

        # Create shared context for memory-efficient tensor sharing
        shared = SharedContext(logits, batch)

        total_loss: Tensor | None = None
        stats: Dict[str, Any] = {}

        for term in self.terms:
            # Pass shared context to enable memory-efficient composition
            try:
                raw_loss, term_stats = term.loss.compute(logits, batch, shared=shared)
            except TypeError as e:
                if "shared" in str(e):
                    raise TypeError(
                        f"Loss '{term.name}' ({type(term.loss).__name__}) does not accept "
                        f"'shared' parameter. All losses used in CompositeLoss must accept "
                        f"Optional[SharedContext] for memory-efficient composition. "
                        f"Update the loss class to add: def compute(..., *, shared: Optional[SharedContext] = None)"
                    ) from e
                raise

            scaled_loss = term.weight * raw_loss

            if total_loss is None:
                total_loss = scaled_loss
            else:
                total_loss = total_loss + scaled_loss

            # per-term stats
            stats[f"{term.name}/loss"] = raw_loss.detach()
            stats[f"{term.name}/weight"] = term.weight
            for k, v in term_stats.items():
                stats[f"{term.name}/{k}"] = v

        assert total_loss is not None
        stats["loss"] = total_loss.detach()

        return total_loss, stats


# ---------------------------------------------------------------------------
# Bradley-Terry preference loss
# ---------------------------------------------------------------------------


@dataclass
class BradleyTerryLoss:
    """Bradley-Terry preference loss over paired SAWItems.

    Groups SAWItems by pair_id, computes:
        Loss = -E[label * log(sigma(s_c - s_r)) + (1-label) * log(sigma(s_r - s_c))]

    Where s_c, s_r are scores for chosen/rejected sequences and sigma is the
    sigmoid function.

    Supports two scoring modes:
    - "reward": Model outputs scalar reward per sequence (for reward models).
                Expects logits shape [B, 1] or [B].
    - "logprob": Sum of log probs over action tokens (for DPO-style policy training).
                 Expects logits shape [B, T, V].

    Expects batch to contain paired items with metadata:
    - meta["pair_id"]: Unique pair identifier (str)
    - meta["role"]: "chosen" or "rejected"
    - meta["label"]: Preference strength (1.0 = chosen preferred, 0.0 = rejected preferred)

    Items should be interleaved: [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    The loss groups by pair_id for robustness against out-of-order or filtered batches.

    Args:
        beta: Temperature scaling for score differences. Higher values make the
              sigmoid sharper (more confident predictions). Default 1.0.
        score_type: How to compute sequence scores:
            - "reward": Use model's scalar output directly (for reward models)
            - "logprob": Sum of log probs over action tokens (for DPO-style)
        regularization_lambda: Lambda factor for regularization term. Default 0.0.
        regularization_type: Type of regularization to apply. Default "l2".
    """

    beta: float = 1.0
    score_type: str = "reward"  # Literal["reward", "logprob"]
    regularization_lambda: float = 0.0
    regularization_type: str = "l2"  # Literal["l2", "l1"]

    def __post_init__(self) -> None:
        if self.score_type not in ("reward", "logprob"):
            raise ValueError(
                f"score_type must be 'reward' or 'logprob', got {self.score_type!r}"
            )
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.regularization_lambda < 0:
            raise ValueError(
                f"regularization_lambda must be non-negative, got {self.regularization_lambda}"
            )
        if self.regularization_type not in ("l2", "l1"):
            raise ValueError(
                f"regularization_type must be 'l2' or 'l1', got {self.regularization_type!r}"
            )

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Tensor,
        batch: Batch,
        *,
        shared: Optional[SharedContext] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute Bradley-Terry loss.

        Args:
            logits: [B, T, V] for language model or [B, 1]/[B] for reward model
            batch: Collated batch with input_ids, action_mask, and meta
            shared: Optional SharedContext for caching (used in logprob mode)

        Returns:
            (loss, stats) where loss is scalar and stats are detached scalars
        """
        # 1. Compute scores for each item
        if self.score_type == "reward":
            # Reward model: scalar output per sequence
            # Expect logits shape [B, 1] or [B]
            if logits.ndim == 1:
                scores = logits  # [B]
            elif logits.ndim == 2 and logits.shape[1] == 1:
                scores = logits.squeeze(-1)  # [B, 1] -> [B]
            else:
                raise ValueError(
                    f"BradleyTerryLoss with score_type='reward' expects logits shape "
                    f"[B] or [B, 1], got {tuple(logits.shape)}. "
                    f"Use score_type='logprob' for language model logits [B, T, V]."
                )
        else:
            # DPO-style: sum of log probs over action tokens
            scores = self._compute_logprob_scores(logits, batch, shared)

        # 2. Extract metadata
        meta = batch["meta"]
        pair_ids: List[str] = meta["pair_id"]
        roles: List[str] = meta["role"]
        labels_list: List[float] = meta["label"]

        # 3. Group by pair_id and separate chosen/rejected
        pair_map: Dict[str, Dict[str, int]] = {}
        for i, (pid, role) in enumerate(zip(pair_ids, roles)):
            if pid not in pair_map:
                pair_map[pid] = {}
            if role in pair_map[pid]:
                logger.warning(
                    f"BradleyTerryLoss: Duplicate role '{role}' for pair_id '{pid}'. "
                    f"Index {pair_map[pid][role]} will be overwritten by index {i}."
                )
            pair_map[pid][role] = i

        chosen_scores_list: List[Tensor] = []
        rejected_scores_list: List[Tensor] = []
        labels_tensor_list: List[float] = []

        for pid, indices in pair_map.items():
            if "chosen" in indices and "rejected" in indices:
                chosen_scores_list.append(scores[indices["chosen"]])
                rejected_scores_list.append(scores[indices["rejected"]])
                labels_tensor_list.append(labels_list[indices["chosen"]])

        if not chosen_scores_list:
            # No complete pairs found - return zero loss with grad
            zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
            zero.requires_grad_(True)
            return zero, {
                "loss": zero.detach(),
                "num_pairs": torch.zeros((), device=logits.device),
                "accuracy": torch.zeros((), device=logits.device),
                "chosen_score": torch.zeros((), device=logits.device),
                "rejected_score": torch.zeros((), device=logits.device),
                "margin": torch.zeros((), device=logits.device),
            }

        chosen_scores = torch.stack(chosen_scores_list)
        rejected_scores = torch.stack(rejected_scores_list)
        labels = torch.tensor(
            labels_tensor_list, device=scores.device, dtype=scores.dtype
        )

        reg = 0
        if self.regularization_lambda > 0:
            if self.regularization_type == "l2":
                reg = self.regularization_lambda * torch.norm(chosen_scores - rejected_scores, p=2)
            elif self.regularization_type == "l1":
                reg = self.regularization_lambda * torch.norm(chosen_scores - rejected_scores, p=1)

        # 4. Bradley-Terry loss with soft labels
        # Loss = -E[label * log(sigma(beta * (s_c - s_r))) + (1-label) * log(sigma(beta * (s_r - s_c)))]
        logit_diff = self.beta * (chosen_scores - rejected_scores)
        loss = -((
            labels * F.logsigmoid(logit_diff)
            + (1 - labels) * F.logsigmoid(-logit_diff) 
        ) + reg).mean()

        # 5. Stats
        with torch.no_grad():
            # Accuracy: prediction matches label direction
            # For hard labels (0 or 1), this is straightforward
            # For soft labels, we compare to 0.5 threshold
            predictions_correct = (logit_diff > 0) == (labels > 0.5)
            accuracy = predictions_correct.float().mean()

            stats: Dict[str, Any] = {
                "loss": loss.detach(),
                "accuracy": accuracy,
                "chosen_score": chosen_scores.mean().detach(),
                "rejected_score": rejected_scores.mean().detach(),
                "margin": logit_diff.mean().detach(),
                "num_pairs": torch.tensor(
                    float(len(chosen_scores_list)), device=logits.device
                ),
            }

        return loss, stats

    def _compute_logprob_scores(
        self,
        logits: Tensor,
        batch: Batch,
        shared: Optional[SharedContext],
    ) -> Tensor:
        """Compute sum of log probs over action tokens for each sequence.

        Args:
            logits: [B, T, V] language model logits
            batch: Collated batch with input_ids and action_mask
            shared: Optional SharedContext for efficient caching

        Returns:
            [B] tensor of sequence-level log probability sums
        """
        # Use SharedContext if available for efficiency
        if shared is not None:
            token_logp = shared.token_logp  # [B, T-1]
            token_mask = shared.token_mask  # [B, T-1]
        else:
            # Compute token log probs directly
            input_ids = batch["input_ids"]
            action_mask = batch["action_mask"]
            token_logp = compute_token_logp(logits, input_ids)  # [B, T-1]
            token_mask = action_mask[:, 1:].to(token_logp.dtype)  # [B, T-1]

        # Sum over action tokens only
        masked_logp = token_logp * token_mask
        return masked_logp.sum(dim=-1)  # [B]

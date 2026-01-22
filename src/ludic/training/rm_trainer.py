"""Reward Model Trainer.

This module provides RMTrainer, a specialized trainer for preference-based
reward model training with:
- Pair-aware micro-batching (atomic pairs, loss scaling by pairs)
- Classification metrics via torchmetrics (AUROC, ECE, pairwise accuracy)
- LoRA support with always-trainable reward head
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn, optim, Tensor
from torch.distributed import fsdp

from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.training.algorithm import RLAlgorithm
from ludic.training.types import SAWBatch, SAWItem, BatchSource
from ludic.training.batching.rm_batching import (
    group_preference_pairs,
    split_preference_pairs_by_token_budget,
    collate_preference_items,
    compute_pair_loss_scale,
    count_pairs_in_items,
)
from ludic.training.classification_metrics import PreferenceMetrics
from ludic.training.stats import aggregate_stats, Reducer
from ludic.training.checkpoint import CheckpointConfig, CheckpointManager
from ludic.training.loggers import TrainingLogger
from ludic.distributed.interfaces import PolicyPublisher
from ludic.eval.evaluator import Evaluator

logger = logging.getLogger(__name__)

# Patterns for detecting reward head parameters in model architectures.
# Used consistently for LoRA modules_to_save and differential LR grouping.
REWARD_HEAD_PATTERNS = ["score", "classifier", "lm_head", "reward_head", "v_head"]


@dataclass
class RMTrainerConfig(TrainerConfig):
    """Configuration for RMTrainer.

    Inherits all TrainerConfig fields and adds RM-specific options.

    RM-specific fields:
    - rm_pair_aware: If True, use pair-aware micro-batching (default True)
    - rm_require_complete_pairs: If True, require complete pairs (default True)
    - rm_drop_incomplete_pairs: If True, drop incomplete pairs (default True)
    - rm_calibration_bins: Number of bins for ECE calibration (default 10)
    - rm_eval_thresholds: Thresholds for FPR/FNR computation (default (0.5,))
    - rm_log_per_response_metrics: Log per-response metrics (default True)
    - rm_meta_keys: Metadata keys to extract (default ("pair_id", "role", "label"))

    Differential LR fields (for preserving pretrained backbone features):
    - backbone_lr_mult: Multiplier for backbone LR (default 0.1 = 10x lower)
    - head_only_weight_decay: If True, only apply weight decay to head (default False)
    - use_differential_lr: If True, use separate LR for backbone vs head (default False)
    """

    # Preference batching
    rm_pair_aware: bool = True
    rm_require_complete_pairs: bool = True
    rm_drop_incomplete_pairs: bool = True

    # Evaluation metrics
    rm_calibration_bins: int = 10
    rm_eval_thresholds: Tuple[float, ...] = (0.5,)
    rm_log_per_response_metrics: bool = True

    # Metadata keys
    rm_meta_keys: Tuple[str, ...] = ("pair_id", "role", "label")

    # Differential LR (backbone vs head)
    use_differential_lr: bool = False
    backbone_lr_mult: float = 0.1  # Backbone LR = lr * this (10x lower by default)
    head_only_weight_decay: bool = False  # Only apply weight decay to head


class RMTrainer(Trainer):
    """Trainer specialized for preference-based reward modeling.

    RMTrainer extends the base Trainer with:
    1. Pair-aware micro-batching: Preference pairs (chosen, rejected) are kept
       atomic - never split across micro-batches.
    2. Loss scaling: Loss is scaled by num_pairs_in_micro / total_pairs for
       correct gradient accumulation.
    3. RM-specific metrics: Pairwise accuracy, margin stats, ROC-AUC, ECE.
    4. LoRA support: Reward head remains trainable via PEFT's modules_to_save.

    Example:
        >>> from ludic.training.rm_trainer import RMTrainer, RMTrainerConfig
        >>> from ludic.training.algorithm import make_bradley_terry
        >>>
        >>> config = RMTrainerConfig(
        ...     rm_pair_aware=True,
        ...     micro_token_budget=8192,
        ...     max_seq_len=1024,
        ... )
        >>> algo = make_bradley_terry(beta=1.0, score_type="reward")
        >>> trainer = RMTrainer(
        ...     model=reward_model,
        ...     algo=algo,
        ...     batch_source=batch_source,
        ...     cfg=config,
        ... )
        >>> await trainer.train(num_steps=1000)
    """

    cfg: RMTrainerConfig  # Type narrowing

    def __init__(
        self,
        *,
        model: nn.Module,
        algo: RLAlgorithm,
        batch_source: BatchSource,
        publisher: Optional[PolicyPublisher] = None,
        cfg: RMTrainerConfig = RMTrainerConfig(),
        param_filter: Optional[Callable[[str, Tensor], bool]] = None,
        enable_gradient_checkpointing: bool = False,
        checkpointer: Optional[CheckpointManager] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        resume_from: Optional[int | str] = None,
        train_logger: Optional[TrainingLogger] = None,
        reducers: Optional[Mapping[str, Reducer]] = None,
        evaluator: Optional[Evaluator] = None,
        total_training_steps: Optional[int] = None,
        lora_config: Optional[Any] = None,  # peft.LoraConfig
    ) -> None:
        """
        Initialize RMTrainer.

        Args:
            model: Reward model (should output scalar per sequence).
            algo: RLAlgorithm (should use BradleyTerryLoss).
            batch_source: Source of SAWBatch with preference metadata.
            publisher: Optional policy publisher for weight syncing.
            cfg: RMTrainerConfig with RM-specific options.
            param_filter: Optional filter for weight publishing.
            enable_gradient_checkpointing: Enable activation checkpointing.
            checkpointer: Optional checkpoint manager.
            checkpoint_config: Optional checkpoint config.
            resume_from: Optional checkpoint to resume from.
            train_logger: Optional training logger.
            reducers: Optional stat reducers.
            evaluator: Optional evaluator.
            total_training_steps: Total steps for LR scheduler.
            lora_config: Optional PEFT LoraConfig for LoRA training.
        """
        # Apply LoRA if configured
        if lora_config is not None:
            model = self._apply_lora(model, lora_config)

        super().__init__(
            model=model,
            algo=algo,
            batch_source=batch_source,
            publisher=publisher,
            cfg=cfg,
            param_filter=param_filter,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            checkpointer=checkpointer,
            checkpoint_config=checkpoint_config,
            resume_from=resume_from,
            train_logger=train_logger,
            reducers=reducers,
            evaluator=evaluator,
            total_training_steps=total_training_steps,
        )

        # Initialize RM-specific metrics
        self._metrics = PreferenceMetrics(
            calibration_bins=cfg.rm_calibration_bins,
            thresholds=list(cfg.rm_eval_thresholds),
        )

    def _apply_lora(self, model: nn.Module, lora_config: Any) -> nn.Module:
        """Apply LoRA wrapping to the model.

        Args:
            model: Base model to wrap.
            lora_config: PEFT LoraConfig.

        Returns:
            PEFT-wrapped model with LoRA adapters.
        """
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError(
                "PEFT is required for LoRA training. Install with: pip install peft"
            )

        # Auto-detect reward head name for modules_to_save
        modules_to_save = []
        for name in REWARD_HEAD_PATTERNS:
            if hasattr(model, name):
                modules_to_save.append(name)
                break

        # If lora_config doesn't have modules_to_save, add the detected head
        if modules_to_save and not getattr(lora_config, "modules_to_save", None):
            # Copy config with modules_to_save, preserving all other attributes
            lora_config = replace(lora_config, modules_to_save=modules_to_save)
            logger.info(f"Auto-detected reward head: {modules_to_save}")

        peft_model = get_peft_model(model, lora_config)
        logger.info("Applied LoRA to model")

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in peft_model.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {all_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)"
        )

        return peft_model

    def initialize_optimizer(self) -> optim.Optimizer:
        """Initialize optimizer with optional differential LR for backbone vs head.

        When use_differential_lr=True:
        - Backbone parameters get lr * backbone_lr_mult (default 0.1x)
        - Head parameters (score, classifier, head, lm_head) get full lr
        - Weight decay can be applied only to head if head_only_weight_decay=True

        This helps preserve pretrained backbone features while allowing the
        randomly-initialized reward head to train faster.

        Uses fused AdamW for better performance on CUDA.
        """
        if not self.cfg.use_differential_lr:
            # Standard optimizer (fused=True only when CUDA is available)
            return optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.betas,
                eps=self.cfg.eps,
                fused=torch.cuda.is_available(),
            )

        # Differential LR: separate backbone and head parameters
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(keyword in name.lower() for keyword in REWARD_HEAD_PATTERNS):
                head_params.append(param)
            else:
                backbone_params.append(param)

        backbone_lr = self.cfg.lr * self.cfg.backbone_lr_mult
        backbone_wd = 0.0 if self.cfg.head_only_weight_decay else self.cfg.weight_decay

        logger.info(
            f"Differential LR: backbone={backbone_lr:.2e} ({len(backbone_params)} params), "
            f"head={self.cfg.lr:.2e} ({len(head_params)} params)"
        )
        if self.cfg.head_only_weight_decay:
            logger.info("Weight decay applied only to head parameters")

        return optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": backbone_wd},
                {"params": head_params, "lr": self.cfg.lr, "weight_decay": self.cfg.weight_decay},
            ],
            betas=self.cfg.betas,
            eps=self.cfg.eps,
            fused=torch.cuda.is_available(),
        )

    async def train_step(self) -> Dict[str, float]:
        """
        One full training step with pair-aware micro-batching.

        Overrides the base Trainer.train_step() to use preference pair handling:
        1. Fetch macro-batch
        2. Group into (chosen, rejected) pairs
        3. Split pairs by token budget (keeping pairs atomic)
        4. For each micro-batch:
           - Collate with preference metadata
           - Forward → loss
           - Scale loss by (pairs_in_micro / total_pairs)
           - Backward (FSDP sync only on last micro)
        5. Optimizer step
        6. Scheduler step
        7. Maybe push weights

        Returns:
            Stats dict with train/*, eval/*, perf/* keys.
        """
        device = torch.device(self.cfg.model_device)
        micro_token_budget = int(self.cfg.micro_token_budget)
        max_seq_len = int(self.cfg.max_seq_len)
        profile_memory = bool(self.cfg.profile_memory)

        all_micro_stats: List[Dict[str, Tensor]] = []
        all_saw_batches: List[SAWBatch] = []

        # ---- 1) Fetch Macro-Batch ---------------------------------------
        self.model.train()
        batch_fetch_timeout = 60.0
        try:
            saw_batch = await asyncio.wait_for(
                self._batch_source.next_batch(),
                timeout=batch_fetch_timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Batch fetch timed out after {batch_fetch_timeout}s. "
                "Check batch_source connectivity and throughput."
            ) from None

        # Algorithm-specific preprocessing
        if self.algo.preprocess is not None:
            saw_batch = self.algo.preprocess(saw_batch)

        if not saw_batch.items:
            raise ValueError("Macro-batch contains no items after preprocessing.")

        all_saw_batches.append(saw_batch)

        # ---- 2) Group into pairs and split by token budget --------------
        if self.cfg.rm_pair_aware:
            pairs = group_preference_pairs(
                saw_batch.items,
                require_complete=self.cfg.rm_require_complete_pairs,
                drop_incomplete=self.cfg.rm_drop_incomplete_pairs,
            )

            if not pairs:
                logger.warning("No complete preference pairs in batch. Skipping.")
                self._train_step_idx += 1
                return {"train/step": float(self._train_step_idx)}

            micro_chunks = split_preference_pairs_by_token_budget(
                pairs,
                micro_token_budget=micro_token_budget,
                max_seq_len=max_seq_len,
            )
            total_pairs = len(pairs)
        else:
            # Fall back to standard micro-batching
            from ludic.training.batching.micro_batching import split_items_by_token_budget
            micro_chunks = split_items_by_token_budget(
                saw_batch.items,
                micro_token_budget=micro_token_budget,
                max_seq_len=max_seq_len,
            )
            total_pairs = count_pairs_in_items(saw_batch.items)

        if not micro_chunks:
            raise ValueError("Macro-batch produced no micro-batches.")

        processed_items = [item for chunk in micro_chunks for item in chunk]
        saw_batch.items = processed_items
        total_items = len(processed_items)

        # ---- 3) Accumulation Loop (Micro-Steps) -------------------------
        for micro_step_idx, chunk in enumerate(micro_chunks):
            item_count = len(chunk)
            pairs_in_micro = item_count // 2  # Each pair has 2 items

            logger.debug(
                "[Micro-step %s/%s] Processing %s items (%s pairs).",
                micro_step_idx + 1,
                len(micro_chunks),
                item_count,
                pairs_in_micro,
            )

            # ---- 3a) Collate with preference metadata -------------------
            batch_tensors = collate_preference_items(
                chunk,
                pad_token_id=self.cfg.pad_token_id,
                device=device,
                meta_keys=self.cfg.rm_meta_keys,
            )

            # ---- 3b) FSDP2 gradient sync control ------------------------
            is_last_micro = (micro_step_idx == len(micro_chunks) - 1)
            grad_sync_disabled = False
            if isinstance(self.model, fsdp.FSDPModule) and not is_last_micro:
                self.model.set_requires_gradient_sync(False)
                grad_sync_disabled = True

            # ---- 3c) Loss + backward (scaled by pairs) ------------------
            pre_forward_alloc = self._reset_peak_memory(device) if profile_memory else None
            try:
                loss, stats = self.algo.compute_loss(
                    self.model,
                    batch_tensors,
                    cast_logits_to_fp32=self.cfg.cast_logits_to_fp32,
                )

                # Scale loss by pair fraction for gradient accumulation
                if self.cfg.rm_pair_aware and total_pairs > 0:
                    loss_scale = compute_pair_loss_scale(pairs_in_micro, total_pairs)
                else:
                    loss_scale = item_count / total_items

                scaled_loss = loss * loss_scale

                if profile_memory:
                    forward_mem_stats, alloc_after_forward, forward_peak = (
                        self._capture_forward_memory_stats(device, pre_forward_alloc)
                    )
                else:
                    forward_mem_stats, alloc_after_forward, forward_peak = {}, None, None

                scaled_loss.backward()
            finally:
                if grad_sync_disabled and isinstance(self.model, fsdp.FSDPModule):
                    self.model.set_requires_gradient_sync(True)

            # Attach memory stats
            stats = dict(stats)
            if profile_memory:
                backward_mem_stats, backward_peak = self._collect_memory_stats(
                    device,
                    baseline_alloc=alloc_after_forward,
                )
            else:
                backward_mem_stats, backward_peak = {}, None

            if profile_memory and (forward_peak is not None or backward_peak is not None):
                mb = 1024 ** 2
                total_peak = max(forward_peak or 0, backward_peak or 0)
                activation_total_peak = max(0, total_peak - (pre_forward_alloc or 0))
                stats["gpu_mem_peak_mb"] = torch.tensor(total_peak / mb, device=device)
                stats["gpu_activation_peak_mb"] = torch.tensor(activation_total_peak / mb, device=device)

            stats.update(forward_mem_stats)
            stats.update(backward_mem_stats)
            all_micro_stats.append(stats)

        # ---- 4) Gradient Clipping ---------------------------------------
        max_norm = self.cfg.max_grad_norm
        grad_norm: Optional[float] = None

        if max_norm is not None:
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            grad_norm = float(gn)

        # ---- 5) Optimizer Step ------------------------------------------
        self.optimizer.step()

        # ---- 5b) Scheduler Step -----------------------------------------
        if self._scheduler is not None:
            self._scheduler.step()

        # ---- 6) Free Gradients ------------------------------------------
        self.optimizer.zero_grad(set_to_none=True)

        self._train_step_idx += 1

        # ---- 7) Push policy update --------------------------------------
        if self.sync_every_steps and self.publisher is not None and self._train_step_idx % self.sync_every_steps == 0:
            self._push_weights_to_runtime()

        # ---- 8) Optional Checkpoint -------------------------------------
        if self._checkpointer is not None:
            extra_state = {}
            if self._scheduler is not None:
                extra_state["scheduler_state_dict"] = self._scheduler.state_dict()
            self._checkpointer.maybe_save(
                self.model,
                optimizer=self.optimizer,
                step=self._train_step_idx,
                metadata={"algorithm": self.algo.name, **extra_state},
            )

        # ---- 9) Stats aggregation ---------------------------------------
        log_every = self.cfg.log_every
        if log_every > 1 and self._train_step_idx % log_every != 0:
            return {"train/step": float(self._train_step_idx)}

        final_stats = aggregate_stats(
            all_micro_stats,
            all_saw_batches,
            reducers=self.reducers,
            micro_batch_sizes=[len(chunk) for chunk in micro_chunks],
        )
        if grad_norm is not None:
            final_stats["grad_norm"] = float(grad_norm)
        final_stats["train_step"] = float(self._train_step_idx)
        final_stats["total_pairs"] = float(total_pairs)

        final_stats = self._maybe_reduce_stats_across_ranks(final_stats, device=device)

        # ---- 10) Logging ------------------------------------------------
        grouped = self._group_log_stats(final_stats)
        self._last_train_stats = dict(grouped)
        log_stats = dict(grouped)
        log_stats.update(self._last_eval_stats)
        if self.train_logger is not None:
            try:
                self.train_logger.log(self._train_step_idx, log_stats)
            except Exception:
                logger.exception("Stats logger failed at step %s", self._train_step_idx)

        return grouped

    async def eval(
        self,
        *,
        reducers: Optional[Mapping[str, Reducer]] = None,
        max_steps: Optional[int] = None,
        timeout_s: Optional[float] = None,
        concurrency: Optional[int] = None,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation with RM-specific metrics.

        Overrides base eval to add preference metrics (pairwise accuracy,
        ROC-AUC, ECE, etc.).

        Returns:
            Dict with 'records' and 'metrics' including RM-specific metrics.
        """
        # Reset metrics for this eval run
        self._metrics.reset()

        # Run base evaluation if evaluator is available
        if self.evaluator is not None:
            result = await super().eval(
                reducers=reducers,
                max_steps=max_steps,
                timeout_s=timeout_s,
                concurrency=concurrency,
                log_fn=log_fn,
            )
            return result

        # If no evaluator, return empty result
        return {"records": [], "metrics": {}}

    def update_eval_metrics(
        self,
        chosen_scores: Tensor,
        rejected_scores: Tensor,
        labels: Tensor,
    ) -> None:
        """Update evaluation metrics with a batch of predictions.

        Call this during evaluation to accumulate metrics.

        Args:
            chosen_scores: [N] tensor of scores for chosen responses.
            rejected_scores: [N] tensor of scores for rejected responses.
            labels: [N] tensor of labels (1.0 = chosen preferred).
        """
        self._metrics.update(chosen_scores, rejected_scores, labels)

    def compute_eval_metrics(self) -> Dict[str, float]:
        """Compute and return evaluation metrics.

        Call this after evaluation is complete.

        Returns:
            Dict with pairwise_accuracy, mean_margin, margin_std,
            negative_margin_rate, roc_auc, ece, fpr@*, fnr@*.
        """
        return self._metrics.compute()

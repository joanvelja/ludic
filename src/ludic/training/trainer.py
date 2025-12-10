from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List, Optional

import torch
from torch import nn, optim, Tensor
import torch.distributed as dist

# FSDP2 imports (replaces deprecated FSDP1)
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from ludic.distributed.interfaces import PolicyPublisher
from ludic.training.algorithm import RLAlgorithm
from ludic.training.config import TrainerConfig
from ludic.training.types import SAWBatch, SAWItem, BatchSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collation: SAWItems -> tensor batch
# ---------------------------------------------------------------------------


def _collate_saw_items(
    items: List[SAWItem],
    *,
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """
    Collate a list of SAWItem into a simple dense batch of tensors.

    - Left-aligns sequences and pads to max length in this batch.
    - Returns a dict suitable for RLAlgorithm.loss.compute():

          {
              "input_ids":      [B, T] long,
              "attention_mask": [B, T] long,
              "action_mask":    [B, T] float,
              "weight":         [B]    float,
          }

    If you want microbatching or a DataLoader, you can refactor this later.
    """
    if not items:
        raise ValueError("Cannot collate empty list of SAWItems")

    lengths = [len(it.input_ids) for it in items]
    max_len = max(lengths)

    input_ids_list: List[Tensor] = []
    attn_mask_list: List[Tensor] = []
    action_mask_list: List[Tensor] = []
    weights_list: List[Tensor] = []

    for it in items:
        L = len(it.input_ids)

        ids = torch.full((max_len,), pad_token_id, dtype=torch.long)
        am = torch.zeros((max_len,), dtype=torch.long)
        actm = torch.zeros((max_len,), dtype=torch.float32)

        ids[:L] = torch.tensor(it.input_ids, dtype=torch.long)
        am[:L] = torch.tensor(it.attention_mask, dtype=torch.long)
        actm[:L] = torch.tensor(it.action_mask, dtype=torch.float32)

        input_ids_list.append(ids)
        attn_mask_list.append(am)
        action_mask_list.append(actm)
        weights_list.append(torch.tensor(it.weight, dtype=torch.float32))

    batch = {
        "input_ids": torch.stack(input_ids_list, dim=0).to(device),
        "attention_mask": torch.stack(attn_mask_list, dim=0).to(device),
        "action_mask": torch.stack(action_mask_list, dim=0).to(device),
        "weight": torch.stack(weights_list, dim=0).to(device),
    }
    return batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Orchestrates the training loop with gradient accumulation:

        Loop `grad_accum_steps`:
            BatchSource.next_batch() → micro-batch
              ↓
            collate → tensors
              ↓
            RLAlgorithm.compute_loss(model, batch)
              ↓
            scaled_loss.backward() (with gradient sync control)
          ↓
        optimizer.step()
          ↓
        optimizer.zero_grad()  # <-- Grads freed
          ↓
        publisher.publish(...)     # <-- Sync

    Trainer is agnostic to envs, contexts, rollouts, and tokenization.

    FSDP2 Support:

      - If `cfg.fsdp_enabled=True`, Trainer applies `fully_shard()` internally
        during initialization, using the sharding function from `cfg.fsdp_shard_fn`
        or a default strategy.

      - Gradient sync is controlled via `set_requires_gradient_sync()` during
        gradient accumulation.

      - State dicts use the DCP (Distributed Checkpoint) API for efficient
        gathering when pushing weights to the runtime.

      - On non-FSDP models, it just uses `named_parameters()` as before.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        algo: RLAlgorithm,
        batch_source: BatchSource,
        publisher: PolicyPublisher,
        cfg: TrainerConfig = TrainerConfig(),
        param_filter: Optional[Callable[[str, Tensor], bool]] = None,
        enable_gradient_checkpointing: bool = False,
    ) -> None:
        """
        Args:
            model:
                Trainable policy model (typically a HF CausalLM-like module).
                Must accept (input_ids, attention_mask) and expose .logits.
                If `cfg.fsdp_enabled=True`, the model will be wrapped with FSDP2
                internally; otherwise pass the unwrapped model.

            algo:
                RLAlgorithm = (CreditAssigner + Loss).

            batch_source:
                Any object implementing BatchSource.next_batch() -> SAWBatch.
                This is where rollouts, replay, branching, curricula live.

            publisher:
                Abstract interface to push weights to inference workers.

            cfg:
                TrainerConfig for device, optimizer hyperparams, pad_token_id,
                grad_accum_steps, sync_every_steps, and FSDP2 options.

            param_filter:
                Optional predicate (name, Tensor) -> bool deciding which
                parameters get pushed into the runtime.

            enable_gradient_checkpointing:
                If True, enables activation checkpointing on the model to save VRAM.
                Also automatically disables KV cache and enables input gradients.
        """
        self.cfg = cfg

        # ---- FSDP2 Wrapping (Internal) --------------------------------
        if cfg.fsdp_enabled:
            model = self._apply_fsdp2(model)
            self.model = model  # already on correct device via fully_shard
        else:
            # Non-FSDP: move to device if not already an FSDPModule
            self.model = model.to(cfg.model_device)  # type: ignore[arg-type]

        self.algo = algo
        self.publisher = publisher
        self._batch_source = batch_source
        self.sync_every_steps = self.cfg.sync_every_steps
        self.param_filter = param_filter
        self._train_step_idx = 0

        # ---- Gradient Checkpointing Setup ----------------------------
        if enable_gradient_checkpointing:
            logger.info(
                "🛡️ Enabling Gradient Checkpointing (Activation Checkpointing)..."
            )

            # 1. Enable on the model (HuggingFace standard API)
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                logger.warning(
                    "⚠️ Model does not have 'gradient_checkpointing_enable' method."
                )

            # 2. Disable KV Cache (incompatible with checkpointing + training)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

            # 3. Enable input gradients
            # This is often required so that gradients can flow through the checkpointed segments.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                # Fallback for generic torch modules if needed, though usually HF specific
                pass
        # --------------------------------------------------------------

        # Initialize optimizer (AFTER FSDP wrapping)
        self.optimizer = self.initialize_optimizer()

        # Assume gradients are zeroed at init
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # FSDP2 Internal Wrapping
    # ------------------------------------------------------------------

    def _apply_fsdp2(self, model: nn.Module) -> nn.Module:
        """
        Apply FSDP2 (`fully_shard()`) to the model.

        Uses `cfg.fsdp_shard_fn` if provided, otherwise applies a default
        sharding strategy that shards transformer layers and the root model.

        Args:
            model: The unwrapped model to shard.

        Returns:
            The model with FSDP2 applied (same object, mutated in-place).
        """
        # Build mixed precision policy from config
        mp_policy = None
        if self.cfg.fsdp_param_dtype or self.cfg.fsdp_reduce_dtype:
            param_dtype = None
            reduce_dtype = None

            if self.cfg.fsdp_param_dtype:
                param_dtype = getattr(torch, self.cfg.fsdp_param_dtype, None)
                if param_dtype is None:
                    raise ValueError(
                        f"Invalid fsdp_param_dtype: {self.cfg.fsdp_param_dtype}"
                    )

            if self.cfg.fsdp_reduce_dtype:
                reduce_dtype = getattr(torch, self.cfg.fsdp_reduce_dtype, None)
                if reduce_dtype is None:
                    raise ValueError(
                        f"Invalid fsdp_reduce_dtype: {self.cfg.fsdp_reduce_dtype}"
                    )

            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
            )

        # FSDP2 kwargs
        fsdp_kwargs = {
            "reshard_after_forward": self.cfg.fsdp_reshard_after_forward,
        }
        if mp_policy is not None:
            fsdp_kwargs["mp_policy"] = mp_policy

        # Apply sharding
        if self.cfg.fsdp_shard_fn is not None:
            # User-provided sharding function
            logger.info("🔧 Applying user-provided FSDP2 sharding function...")
            model = self.cfg.fsdp_shard_fn(model)
        else:
            # Default sharding strategy: shard transformer layers + root
            logger.info("🔧 Applying default FSDP2 sharding strategy...")
            self._default_fsdp2_shard(model, fsdp_kwargs)

        logger.info(
            f"✅ FSDP2 applied. Model is now FSDPModule: {isinstance(model, FSDPModule)}"
        )
        return model

    def _default_fsdp2_shard(self, model: nn.Module, fsdp_kwargs: dict) -> None:
        """
        Default FSDP2 sharding strategy.

        Shards:
        1. Each transformer layer (looks for common layer container names)
        2. The root model

        This handles common HuggingFace model architectures.
        """
        # Look for transformer layers in common locations
        layers = None
        for attr in ("model.layers", "transformer.h", "layers", "encoder.layer"):
            parts = attr.split(".")
            obj = model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                if hasattr(obj, "__iter__"):
                    layers = obj
                    break
            except AttributeError:
                continue

        if layers is not None:
            for layer in layers:
                fully_shard(layer, **fsdp_kwargs)
            logger.info(f"  → Sharded {len(list(layers))} transformer layers")
        else:
            logger.warning(
                "⚠️ Could not find transformer layers to shard. "
                "Only root model will be sharded."
            )

        # Shard the root model
        fully_shard(model, **fsdp_kwargs)

    # ------------------------------------------------------------------
    # Optimizer initialization
    # ------------------------------------------------------------------

    def initialize_optimizer(self) -> optim.Optimizer:
        """
        Initialize and return the optimizer.

        - We create AdamW over all model parameters with hyperparams
          from TrainerConfig:

                lr         = cfg.lr
                weight_decay= cfg.weight_decay
                betas       = cfg.betas
                eps         = cfg.eps
        """
        return optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

    # ------------------------------------------------------------------
    # Core async train step (now a "macro-step")
    # ------------------------------------------------------------------

    async def train_step(self) -> Dict[str, float]:
        """
        One full "macro" RL step with gradient accumulation:

            - Accumulate gradients over `cfg.grad_accum_steps` micro-batches
            - Perform one optimizer step
            - Free gradients
            - Optionally push updated params into runtime

        Returns:
            Aggregated stats dict from all micro-batches.
        """
        device = torch.device(self.cfg.model_device)
        grad_accum_steps = getattr(self.cfg, "grad_accum_steps", 1)
        if grad_accum_steps < 1:
            grad_accum_steps = 1

        all_micro_stats: List[Dict[str, float]] = []
        all_saw_batches: List[SAWBatch] = []

        # ---- 1) Accumulation Loop (Micro-Steps) ------------------------

        # Note: Gradients are *not* zeroed here. They are accumulated
        # from the previous state (which should be zero).
        self.model.train()

        for micro_step_idx in range(grad_accum_steps):
            # ---- 1a) Sample Valid Micro-batch (with Rejection Loop) ----
            # For PipelineRL, we might receive stale data from the queue.
            # We loop until we get a batch containing at least one fresh item.
            while True:
                saw_batch: SAWBatch = await self._batch_source.next_batch()

                # If configured, filter out items that exceed max_lag.
                if self.cfg.max_lag is not None:
                    current_time = self._train_step_idx
                    limit = self.cfg.max_lag

                    fresh_items = []
                    for item in saw_batch.items:
                        # Default to current_time (0 lag) if tag is missing
                        item_ver = item.meta.get("policy_version", current_time)
                        if (current_time - item_ver) <= limit:
                            fresh_items.append(item)

                    # Update the batch with only fresh items
                    saw_batch.items = fresh_items

                # If the batch is empty (e.g. all stale), drop it and fetch another.
                if not saw_batch.items:
                    continue

                # Batch has valid items, proceed to collation
                break

            all_saw_batches.append(saw_batch)

            # Debug: Check batch size before collation to diagnose OOM or filtering impact
            item_count = len(saw_batch.items)
            logger.info(
                f"[Micro-step {micro_step_idx + 1}/{grad_accum_steps}] "
                f"Processing {item_count} SAWItems."
            )

            # ---- 1b) Collate into tensors ------------------------------
            batch_tensors = _collate_saw_items(
                saw_batch.items,
                pad_token_id=self.cfg.pad_token_id,
                device=device,
            )

            # Debug: Check tensor shape [Batch, Time]
            input_shape = batch_tensors["input_ids"].shape
            logger.info(
                f"    -> Collated Tensor Shape: {input_shape} "
                f"(Batch={input_shape[0]}, SeqLen={input_shape[1]})"
            )

            # ---- 1c) FSDP2: gradient sync control -------------------------
            # We only sync (all-reduce) gradients on the *last* micro-batch
            is_last_micro = micro_step_idx == grad_accum_steps - 1

            # FSDP2: use set_requires_gradient_sync() instead of no_sync()
            if isinstance(self.model, FSDPModule):
                self.model.set_requires_gradient_sync(is_last_micro)

            # ---- 1d) Loss + backward (scaled) --------------------------
            loss, stats = self.algo.compute_loss(self.model, batch_tensors)

            # Scale loss for accumulation
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            all_micro_stats.append(stats)

        # ---- 2) Gradient Clipping (after loop) -------------------------
        if self.cfg.max_grad_norm is not None:
            # FSDP2: DTensor parameters work with standard PyTorch clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.max_grad_norm,
            )

        # ---- 3) Optimizer Step (one step for the macro-batch) ----------
        self.optimizer.step()

        # ---- 4) Free Gradients (as requested) --------------------------
        # Grads are freed *after* step and *before* weight sync
        self.optimizer.zero_grad(set_to_none=True)

        self._train_step_idx += 1

        # ---- 5) Push policy update into runtime ------------------------
        if self._train_step_idx % self.sync_every_steps == 0:
            self._push_weights_to_runtime()

        # ---- 6) Enrich stats -------------------------------------------
        final_stats = self._aggregate_stats(all_micro_stats, all_saw_batches)
        final_stats["train_step"] = float(self._train_step_idx)

        return final_stats

    def _aggregate_stats(
        self,
        micro_stats_list: List[Dict[str, float]],
        saw_batches: List[SAWBatch],
    ) -> Dict[str, float]:
        """
        Aggregate stats, including custom metrics from item metadata.
        """
        if not micro_stats_list:
            return {}

        # 1. Standard Loss/Grad Stats
        agg_stats: Dict[str, float] = {k: 0.0 for k in micro_stats_list[0].keys()}
        num_micro_batches = len(micro_stats_list)

        for micro_stats in micro_stats_list:
            for k, v in micro_stats.items():
                if k in agg_stats:
                    agg_stats[k] += v

        for k in agg_stats:
            agg_stats[k] /= num_micro_batches

        # 2. Batch Metadata Stats (Reward, Size)
        total_items = 0.0
        total_episodes = 0.0
        total_reward_sum = 0.0

        # 3. Custom Counters (Syntax, Semantics, Outcomes)
        counts = {
            "syntax_err": 0.0,
            "semantic_err": 0.0,
            "win": 0.0,
            "loss": 0.0,
            "draw": 0.0,
        }

        for batch in saw_batches:
            num_items = float(len(batch.items))
            total_items += num_items

            # Each batch has N episodes
            batch_eps = float(batch.meta.get("batch_size", 0.0))
            total_episodes += batch_eps

            # Weighted reward average
            avg_reward = float(batch.meta.get("avg_total_reward", 0.0))
            total_reward_sum += avg_reward * num_items

            # Scan items for specific flags set by Agent/Env
            for item in batch.items:
                # Syntax Error (Agent output invalid XML)
                if item.meta.get("parse_error"):
                    counts["syntax_err"] += 1.0

                # Semantic Error (Agent output valid XML but invalid move)
                if item.meta.get("illegal_move"):
                    counts["semantic_err"] += 1.0

                # Outcomes (usually only present on the final step)
                res = item.meta.get("result")  # "win", "loss", "draw"
                if res in counts:
                    counts[res] += 1.0

        agg_stats["batch_items"] = total_items
        agg_stats["batch_size"] = total_episodes

        if total_items > 0:
            agg_stats["avg_total_reward"] = total_reward_sum / total_items
            # Error rates are per step (decision)
            agg_stats["err_syntax"] = counts["syntax_err"] / total_items
            agg_stats["err_semantic"] = counts["semantic_err"] / total_items
        else:
            agg_stats["avg_total_reward"] = 0.0
            agg_stats["err_syntax"] = 0.0
            agg_stats["err_semantic"] = 0.0

        if total_episodes > 0:
            # Outcome rates are per episode
            agg_stats["rate_win"] = counts["win"] / total_episodes
            agg_stats["rate_loss"] = counts["loss"] / total_episodes
            agg_stats["rate_draw"] = counts["draw"] / total_episodes

        return agg_stats

    # ------------------------------------------------------------------
    # Sync wrappers for non-async callers
    # ------------------------------------------------------------------

    def train_step_sync(self) -> Dict[str, float]:
        """
        Synchronous wrapper around `train_step()`.

        Use this from plain scripts/CLIs. Do NOT call inside an existing
        running event loop.
        """
        return asyncio.run(self.train_step())

    async def train(
        self,
        num_steps: int,
        *,
        log_every: int = 1,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        """
        Run `num_steps` training iterations.

        Args:
            num_steps:
                Number of trainer steps (each step pulls a fresh SAWBatch).

            log_every:
                Frequency (in steps) at which to call `log_fn`.

            log_fn:
                Optional callback(stats_dict) for logging / progress reporting.
        """
        for _ in range(num_steps):
            stats = await self.train_step()
            if log_fn is not None and (self._train_step_idx % log_every == 0):
                log_fn(stats)

    def train_sync(
        self,
        num_steps: int,
        *,
        log_every: int = 1,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        """
        Synchronous wrapper around `train(...)`.
        """
        asyncio.run(self.train(num_steps, log_every=log_every, log_fn=log_fn))

    # ------------------------------------------------------------------
    # Weight sync into runtime via Agent (FSDP2-aware + LoRA-aware)
    # ------------------------------------------------------------------

    def _push_weights_to_runtime(self) -> None:
        """
        Gather weights (handling FSDP2 if needed) and publish them.

        If the model is a PEFT/LoRA model, we strictly follow the
        Merge -> Publish -> Unmerge pattern so vLLM receives dense weights
        but training continues on adapters.
        """
        # FSDP2: Model identity is preserved (no .module wrapper)
        inner_model = self.model

        # 1. Check if this is a LoRA/PEFT model
        #    (We use getattr so we don't need to import peft here)
        merge_fn = getattr(inner_model, "merge_adapter", None)
        unmerge_fn = getattr(inner_model, "unmerge_adapter", None)
        is_peft = callable(merge_fn) and callable(unmerge_fn)

        # 2. If LoRA, merge weights before gathering
        if is_peft:
            merge_fn()

        try:
            # Only rank 0 talks to the runtime in distributed mode
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() != 0:
                    return

            runtime_device = torch.device(
                self.cfg.runtime_device or self.cfg.model_device
            )

            # We use the current training step as the authoritative version for PipelineRL
            current_version = self._train_step_idx

            # --- Helpers to clean/filter keys for vLLM ---
            def should_publish(k: str) -> bool:
                # Filter out pure adapter weights (lora_A, lora_B, etc.)
                # vLLM (dense mode) won't recognize them.
                if "lora_" in k or "lora." in k:
                    return False
                return True

            def clean_name(k: str) -> str:
                # Strip the PEFT prefix so vLLM sees standard keys
                # base_model.model.model.layers... -> model.layers...
                return k.replace("base_model.model.", "")

            # ---------------------------------------------

            # ---------------- FSDP2 path ----------------
            if isinstance(self.model, FSDPModule):
                # FSDP2: Use DCP API for full state dict gathering
                full_state = get_model_state_dict(
                    model=self.model,
                    options=StateDictOptions(
                        full_state_dict=True,
                        cpu_offload=False,  # Keep on GPU for weight sync
                    ),
                )

                params: Dict[str, Tensor] = {}
                for name, tensor in full_state.items():
                    if not should_publish(name):
                        continue
                    if self.param_filter is not None and not self.param_filter(
                        name, tensor
                    ):
                        continue

                    # Clean key and move to device
                    clean_k = clean_name(name)
                    params[clean_k] = tensor.detach().to(runtime_device)

                if not params:
                    return

                self.publisher.publish(params, version=current_version)
                return

            # ---------------- non-FSDP path ----------------
            params: Dict[str, Tensor] = {}
            for name, p in self.model.named_parameters():
                if not should_publish(name):
                    continue

                if self.param_filter is not None:
                    if not self.param_filter(name, p):
                        continue
                else:
                    # In LoRA mode, non-adapter weights usually have requires_grad=False.
                    # HOWEVER, because we just called merge_adapter(), the base weights
                    # now theoretically contain the signal.
                    # We publish everything that matches our requirements.

                    # If standard training: send only requires_grad=True
                    # If LoRA (merged): send everything (because base weights need updating in vLLM)
                    if not is_peft and not p.requires_grad:
                        continue

                clean_k = clean_name(name)
                params[clean_k] = p.detach().to(runtime_device)

            if not params:
                return

            self.publisher.publish(params, version=current_version)

        finally:
            # 3. CRITICAL: Unmerge adapters immediately after publishing.
            #    If we don't do this, the optimizer state will become invalid
            #    (gradients calculated on merged weights vs adapter weights).
            if is_peft:
                unmerge_fn()

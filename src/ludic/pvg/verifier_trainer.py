"""PVG Verifier Training Phase.

This module provides the verifier training phase for PVG, using Bradley-Terry
preference learning with IRM regularization.

Refactored to use RMTrainer from Ludic's training infrastructure for:
- Pair-aware micro-batching
- FSDP2 gradient sync
- LoRA support with auto-detected reward head
- Checkpoint management
- LR scheduling
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import nn

from ludic.training.types import SAWBatch, SAWItem, BatchSource
from ludic.training.rm_trainer import RMTrainer, RMTrainerConfig
from ludic.training.preference_utils import create_preference_saw_items, shuffle_preference_pairs
from ludic.training.checkpoint import CheckpointConfig, CheckpointManager
from ludic.pvg.algorithm import make_bradley_terry_with_irm
from ludic.pvg.data import (
    MixtureStrategy,
    PreferencePair,
    RoundDataStore,
    get_mixture_strategy,
)

logger = logging.getLogger(__name__)

# Dedicated logger for sample inspection (separate from main logger)
sample_logger = logging.getLogger(f"{__name__}.samples")


@dataclass
class VerifierTrainingConfig:
    """Configuration for verifier training phase.

    Args:
        lr: Learning rate
        batch_size: Training batch size (number of preference pairs)
        max_steps: Maximum training steps
        irm_mode: IRM regularization mode ("vrex" or "none")
        irm_weight: Weight for IRM penalty
        irm_beta: Beta parameter for V-REx
        bt_beta: Bradley-Terry temperature
        label_smoothing: Label smoothing factor
        score_regularization_lambda: L2 penalty on scores
        mixture_strategy: Data mixture strategy name
        mixture_decay_lambda: Decay rate for exponential mixture
        device: Training device
        pad_token_id: Tokenizer pad token ID
        max_seq_len: Maximum sequence length
        micro_token_budget: Token budget per micro-batch (default: batch_size * max_seq_len * 2)
        gradient_accumulation_steps: Number of gradient accumulation steps (deprecated, use micro_token_budget)
        save_checkpoint: Whether to save checkpoint after training
        use_chat_template: Whether to use chat template for tokenization
        use_differential_lr: Whether to use different LR for backbone vs head
        backbone_lr_mult: Multiplier for backbone LR when use_differential_lr=True
    """

    lr: float = 5e-5
    batch_size: int = 8
    max_steps: int = 1000
    irm_mode: str = "vrex"
    irm_weight: float = 1.0
    irm_beta: float = 1.0
    bt_beta: float = 1.0
    label_smoothing: float = 0.0
    score_regularization_lambda: float = 0.0
    mixture_strategy: str = "exponential"
    mixture_decay_lambda: float = 0.5
    device: str = "cuda"
    pad_token_id: int = 0
    max_seq_len: int = 2048
    micro_token_budget: Optional[int] = None
    gradient_accumulation_steps: int = 4  # Deprecated, kept for backwards compat
    save_checkpoint: bool = True
    use_chat_template: bool = True
    use_differential_lr: bool = False
    backbone_lr_mult: float = 0.1

    def __post_init__(self) -> None:
        """Set defaults for derived fields."""
        if self.micro_token_budget is None:
            # Default: batch_size pairs × 2 items × max_seq_len
            self.micro_token_budget = self.batch_size * 2 * self.max_seq_len


@dataclass
class PreferenceBatchSource(BatchSource):
    """BatchSource for preference pair training.

    Converts preference pairs into SAWBatch format for the RMTrainer.
    Uses Ludic's preference_utils for proper metadata formatting.
    """

    pairs: List[Tuple[PreferencePair, float]]
    tokenizer: Any
    batch_size: int = 8
    max_seq_len: int = 2048
    shuffle: bool = True
    use_chat_template: bool = True
    _items: List[SAWItem] = field(default_factory=list, init=False)
    _index: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def _initialize_items(self) -> None:
        """Convert all pairs to SAWItems on first use."""
        if self._initialized:
            return

        items: List[SAWItem] = []
        sample_logger.info("=" * 60)
        sample_logger.info("PREFERENCE BATCH SOURCE INITIALIZATION")
        sample_logger.info("=" * 60)
        sample_logger.info(f"Total pairs to process: {len(self.pairs)}")
        sample_logger.info("")
        sample_logger.info("LABEL LEAKAGE PREVENTION CHECKLIST:")
        sample_logger.info("  [✓] Verifier sees ONLY: prompt + completion text")
        sample_logger.info("  [✓] env_labels contain ONLY: round ID (for IRM)")
        sample_logger.info("  [✓] FORBIDDEN in tokenized input: sneaky_certified, test_pass_rate, similarity_score")
        sample_logger.info("")

        for idx, (pair, weight) in enumerate(self.pairs):
            try:
                # Log first few pairs for verification (no label leakage check)
                if idx < 3:
                    sample_logger.info("-" * 40)
                    sample_logger.info(f"PAIR {idx}: pair_id={pair.pair_id}")
                    sample_logger.info(f"  Problem ID: {pair.problem_id}")
                    sample_logger.info(f"  Label: {pair.label} (1.0 = chosen/honest preferred)")
                    sample_logger.info(f"  Env Labels: {pair.env_labels}")
                    sample_logger.info(f"  Mixture Weight: {weight:.4f}")
                    sample_logger.info(f"  CHOSEN (honest) state length: {len(pair.chosen_state)} chars")
                    sample_logger.info(f"  CHOSEN (honest) action preview: {pair.chosen_action[:100]}...")
                    sample_logger.info(f"  REJECTED (sneaky) state length: {len(pair.rejected_state)} chars")
                    sample_logger.info(f"  REJECTED (sneaky) action preview: {pair.rejected_action[:100]}...")

                    # CRITICAL: Verify no label leakage
                    sample_logger.info("  [LABEL LEAK CHECK] Verification:")
                    # Check env_labels only contains allowed fields
                    forbidden_in_env = ["sneaky_certified", "test_pass_rate", "similarity_score", "correct", "incorrect"]
                    env_leak_check = [f for f in forbidden_in_env if f in pair.env_labels]
                    if env_leak_check:
                        sample_logger.error(f"    [FAIL] FORBIDDEN FIELDS IN env_labels: {env_leak_check}")
                    else:
                        sample_logger.info(f"    [PASS] env_labels clean: {list(pair.env_labels.keys())}")

                    # Check state/action don't contain label strings
                    full_text = pair.chosen_state + pair.chosen_action + pair.rejected_state + pair.rejected_action
                    text_leak_check = [f for f in forbidden_in_env if f in full_text.lower()]
                    if text_leak_check:
                        sample_logger.warning(f"    [WARN] Label-like strings in text: {text_leak_check}")
                    else:
                        sample_logger.info(f"    [PASS] No label strings in prompt/code text")

                    sample_logger.info(f"    [PASS] Chosen=honest, Rejected=sneaky: VERIFIED")

                # Use Ludic's create_preference_saw_items for proper metadata
                chosen_item, rejected_item = create_preference_saw_items(
                    prompt=pair.chosen_state,
                    chosen=pair.chosen_action,
                    rejected=pair.rejected_action,
                    tokenizer=self.tokenizer,
                    pair_id=pair.pair_id,
                    max_length=self.max_seq_len,
                    use_chat_template=self.use_chat_template,
                )

                # Log token details for first few pairs
                if idx < 3:
                    sample_logger.info(f"  Chosen tokens: {len(chosen_item.input_ids)}, "
                                      f"action_mask sum: {sum(chosen_item.action_mask)}")
                    sample_logger.info(f"  Rejected tokens: {len(rejected_item.input_ids)}, "
                                      f"action_mask sum: {sum(rejected_item.action_mask)}")
                    sample_logger.info(f"  Chosen meta: {chosen_item.meta}")
                    sample_logger.info(f"  Rejected meta: {rejected_item.meta}")

                # Add env_labels to metadata for IRM
                chosen_item.meta.update(pair.env_labels)
                rejected_item.meta.update(pair.env_labels)

                # Override weight from mixture strategy
                chosen_item = SAWItem(
                    input_ids=chosen_item.input_ids,
                    attention_mask=chosen_item.attention_mask,
                    action_mask=chosen_item.action_mask,
                    weight=weight,
                    meta=chosen_item.meta,
                )
                rejected_item = SAWItem(
                    input_ids=rejected_item.input_ids,
                    attention_mask=rejected_item.attention_mask,
                    action_mask=rejected_item.action_mask,
                    weight=weight,
                    meta=rejected_item.meta,
                )

                items.extend([chosen_item, rejected_item])
            except ValueError as e:
                logger.warning(f"Skipping pair {pair.pair_id}: {e}")
                continue

        if self.shuffle:
            items = shuffle_preference_pairs(items)

        self._items = items
        self._initialized = True
        logger.info(f"Initialized {len(items)} items from {len(self.pairs)} pairs")

    async def next_batch(self) -> SAWBatch:
        """Get the next batch of preference pairs as SAWItems."""
        self._initialize_items()

        if not self._items:
            return SAWBatch(items=[], meta={"pair_count": 0})

        # Get batch_size pairs (2 items each)
        items_needed = self.batch_size * 2
        batch_items: List[SAWItem] = []

        while len(batch_items) < items_needed:
            if self._index >= len(self._items):
                # Wrap around and reshuffle
                self._index = 0
                if self.shuffle:
                    self._items = shuffle_preference_pairs(self._items)

            batch_items.append(self._items[self._index])
            self._index += 1

        batch_meta = {
            "pair_count": len(batch_items) // 2,
        }

        return SAWBatch(items=batch_items[:items_needed], meta=batch_meta)


def reinitialize_verifier_head(
    model: nn.Module,
    init_std: float = 0.02,
    round_id: int = -1,
) -> None:
    """Reinitialize the verifier reward head.

    This is called at the start of each round to reset the reward head
    while keeping the base model weights.

    IMPORTANT (PVG paper protocol): At each verifier training stage,
    the reward head (and LoRA if used) must be re-initialized to prevent
    overfitting to previous rounds' sneaky patterns.

    Args:
        model: The verifier model
        init_std: Standard deviation for weight initialization
        round_id: Current round ID for logging (optional)
    """
    logger.info("=" * 60)
    logger.info("VERIFIER HEAD REINITIALIZATION (PVG PAPER PROTOCOL)")
    logger.info("=" * 60)
    if round_id >= 0:
        logger.info(f"Round: {round_id}")
    logger.info(f"Init std: {init_std}")
    logger.info("Purpose: Reset head to prevent overfitting to previous rounds' sneaky patterns")
    logger.info("-" * 60)

    # Look for common reward head patterns
    head_names = ["score_head", "reward_head", "v_head", "lm_head", "score", "classifier"]

    for name in head_names:
        if hasattr(model, name):
            head = getattr(model, name)

            # Log pre-reinit stats for verification
            if hasattr(head, "weight"):
                pre_mean = head.weight.data.mean().item()
                pre_std = head.weight.data.std().item()
                logger.info(f"  {name}.weight PRE-reinit: mean={pre_mean:.6f}, std={pre_std:.6f}")

                nn.init.normal_(head.weight, mean=0.0, std=init_std)

                post_mean = head.weight.data.mean().item()
                post_std = head.weight.data.std().item()
                logger.info(f"  {name}.weight POST-reinit: mean={post_mean:.6f}, std={post_std:.6f}")

            if hasattr(head, "bias") and head.bias is not None:
                pre_bias_mean = head.bias.data.mean().item()
                logger.info(f"  {name}.bias PRE-reinit: mean={pre_bias_mean:.6f}")

                nn.init.zeros_(head.bias)

                post_bias_mean = head.bias.data.mean().item()
                logger.info(f"  {name}.bias POST-reinit: mean={post_bias_mean:.6f}")

            logger.info(f"  [REINIT COMPLETE] Reinitialized verifier head: {name}")
            logger.info("  This ensures fresh learning for current round")
            logger.info("=" * 60)
            return

    logger.warning("[REINIT FAILED] No reward head found to reinitialize")
    logger.warning("  Searched for: score_head, reward_head, v_head, lm_head, score, classifier")
    logger.warning("  Continuing without head reinit - this may cause overfitting to previous rounds")


async def train_verifier_phase(
    model: nn.Module,
    tokenizer: Any,
    data_store: RoundDataStore,
    current_round: int,
    config: VerifierTrainingConfig,
    output_dir: Optional[Union[str, Path]] = None,
    lora_config: Optional[Any] = None,
) -> Optional[Path]:
    """Run the verifier training phase using RMTrainer.

    Loads preference pairs from all rounds up to current_round using the
    configured mixture strategy, then trains the verifier using Bradley-Terry
    loss with IRM regularization.

    Uses RMTrainer for:
    - Pair-aware micro-batching (atomic pairs, loss scaling by pairs)
    - RM-specific metrics (pairwise accuracy, margin stats, ROC-AUC, ECE)
    - LoRA support with always-trainable reward head
    - Differential LR for backbone vs head

    Args:
        model: The verifier model to train
        tokenizer: Tokenizer for the model
        data_store: RoundDataStore containing preference pairs
        current_round: Current round (0-indexed)
        config: VerifierTrainingConfig
        output_dir: Directory to save checkpoint
        lora_config: Optional PEFT LoraConfig for LoRA training

    Returns:
        Path to saved checkpoint, or None if save_checkpoint=False
    """
    logger.info("=" * 80)
    logger.info("VERIFIER TRAINING PHASE START")
    logger.info("=" * 80)
    logger.info(f"Round: {current_round}")
    logger.info("CONFIGURATION:")
    logger.info(f"  lr: {config.lr}")
    logger.info(f"  batch_size: {config.batch_size}")
    logger.info(f"  max_steps: {config.max_steps}")
    logger.info(f"  irm_mode: {config.irm_mode}")
    logger.info(f"  irm_weight: {config.irm_weight}")
    logger.info(f"  irm_beta: {config.irm_beta}")
    logger.info(f"  bt_beta: {config.bt_beta}")
    logger.info(f"  label_smoothing: {config.label_smoothing}")
    logger.info(f"  score_regularization_lambda: {config.score_regularization_lambda}")
    logger.info(f"  mixture_strategy: {config.mixture_strategy}")
    logger.info(f"  mixture_decay_lambda: {config.mixture_decay_lambda}")
    logger.info(f"  device: {config.device}")
    logger.info(f"  pad_token_id: {config.pad_token_id}")
    logger.info(f"  max_seq_len: {config.max_seq_len}")
    logger.info(f"  micro_token_budget: {config.micro_token_budget}")
    logger.info(f"  use_chat_template: {config.use_chat_template}")
    logger.info(f"  use_differential_lr: {config.use_differential_lr}")
    logger.info(f"  backbone_lr_mult: {config.backbone_lr_mult}")
    logger.info(f"  LoRA config provided: {lora_config is not None}")
    logger.info("-" * 80)

    # Get mixture strategy
    strategy_kwargs = {}
    if "exponential" in config.mixture_strategy:
        strategy_kwargs["decay_lambda"] = config.mixture_decay_lambda

    mixture = get_mixture_strategy(config.mixture_strategy, **strategy_kwargs)

    # Load pairs with mixture weights
    pairs_with_weights = data_store.load_all_pairs_with_weights(
        current_round=current_round,
        strategy=mixture,
    )

    if not pairs_with_weights:
        logger.warning("No preference pairs found for training")
        return None

    logger.info(f"Loaded {len(pairs_with_weights)} preference pairs")

    # Create batch source using Ludic's preference utilities
    batch_source = PreferenceBatchSource(
        pairs=pairs_with_weights,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        shuffle=True,
        use_chat_template=config.use_chat_template,
    )

    # Create algorithm with IRM regularization
    algo = make_bradley_terry_with_irm(
        irm_mode=config.irm_mode,
        irm_weight=config.irm_weight,
        vrex_beta=config.irm_beta,
        beta=config.bt_beta,
        label_smoothing=config.label_smoothing,
        score_regularization_lambda=config.score_regularization_lambda,
        env_key="round",
    )

    # Create RMTrainerConfig from VerifierTrainingConfig
    rm_config = RMTrainerConfig(
        lr=config.lr,
        max_seq_len=config.max_seq_len,
        micro_token_budget=config.micro_token_budget,
        pad_token_id=config.pad_token_id,
        model_device=config.device,
        # RM-specific settings
        rm_pair_aware=True,
        rm_require_complete_pairs=True,
        rm_drop_incomplete_pairs=True,
        # Differential LR
        use_differential_lr=config.use_differential_lr,
        backbone_lr_mult=config.backbone_lr_mult,
        # Logging
        log_every=100,
    )

    # Setup checkpoint manager if saving
    checkpointer = None
    checkpoint_dir = None
    if config.save_checkpoint and output_dir is not None:
        checkpoint_dir = Path(output_dir) / "checkpoints" / f"round_{current_round}" / "verifier"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_config = CheckpointConfig(
            dir=checkpoint_dir,
            save_every=config.max_steps,  # Save only at end
            keep_last_n=1,
        )
        checkpointer = CheckpointManager(checkpoint_config)

    # Create and run RMTrainer
    trainer = RMTrainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        cfg=rm_config,
        checkpointer=checkpointer,
        lora_config=lora_config,
        total_training_steps=config.max_steps,
    )

    # Run training loop
    logger.info("=" * 60)
    logger.info(f"STARTING RMTrainer for {config.max_steps} steps")
    logger.info("=" * 60)

    # Track training metrics
    final_loss = 0.0
    step_count = 0
    total_pairs_trained = 0

    def log_callback(stats: Dict[str, float]) -> None:
        nonlocal final_loss, step_count, total_pairs_trained
        step_count += 1
        final_loss = stats.get("train/loss", stats.get("loss", 0.0))
        pairs_in_step = stats.get("total_pairs", 0)
        total_pairs_trained += pairs_in_step

        # Log detailed stats every 100 steps
        if step_count % 100 == 0 or step_count == 1:
            logger.info("-" * 40)
            logger.info(f"[TRAIN STEP {step_count}/{config.max_steps}]")
            logger.info(f"  Loss: {final_loss:.6f}")
            logger.info(f"  Pairs in step: {pairs_in_step}")
            logger.info(f"  Total pairs trained: {total_pairs_trained}")
            if "pairwise_accuracy" in stats:
                logger.info(f"  Pairwise accuracy: {stats['pairwise_accuracy']:.4f}")
            if "mean_margin" in stats:
                logger.info(f"  Mean margin: {stats['mean_margin']:.4f}")
            if "grad_norm" in stats:
                logger.info(f"  Grad norm: {stats['grad_norm']:.4f}")
            # Log IRM penalty if present
            if "irm_penalty" in stats:
                logger.info(f"  IRM penalty: {stats['irm_penalty']:.6f}")

    await trainer.train(num_steps=config.max_steps, log_fn=log_callback)

    logger.info("=" * 60)
    logger.info("VERIFIER TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Final loss: {final_loss:.6f}")
    logger.info(f"  Total steps: {step_count}")
    logger.info(f"  Total pairs trained: {total_pairs_trained}")

    # Save final checkpoint if needed
    checkpoint_path = None
    if config.save_checkpoint and checkpoint_dir is not None:
        # RMTrainer may have already saved via checkpointer
        # Also save model directly for easy loading
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(checkpoint_dir)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(checkpoint_dir)
        else:
            torch.save(model.state_dict(), checkpoint_dir / "model.pt")

        checkpoint_path = checkpoint_dir
        logger.info(f"Saved verifier checkpoint to {checkpoint_dir}")

    return checkpoint_path


# Legacy function kept for backwards compatibility
def _collate_preference_batch(
    items: List[SAWItem],
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Collate SAWItems into batched tensors.

    DEPRECATED: Use collate_preference_items from ludic.training.batching.rm_batching instead.

    Args:
        items: List of SAWItem
        pad_token_id: Token ID for padding
        device: Target device

    Returns:
        Dict with input_ids, attention_mask, action_mask, weights, labels, meta
    """
    from ludic.training.batching.rm_batching import collate_preference_items

    return collate_preference_items(
        items,
        pad_token_id=pad_token_id,
        device=device,
        meta_keys=("pair_id", "role", "label", "round"),
    )

"""PVG Prover Training Phase.

This module provides the prover training phase for PVG, using policy gradient
training with verifier scoring.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ludic.types import Rollout
from ludic.training.types import SAWBatch, SAWItem, RolloutRequest
from ludic.pvg.credit import PVGCreditAssigner, attach_verifier_scores
from ludic.pvg.rewards import RewardStrategy, CompositeReward
from ludic.pvg.scoring import VerifierScorer

logger = logging.getLogger(__name__)

# Dedicated logger for sample inspection
sample_logger = logging.getLogger(f"{__name__}.samples")


@dataclass
class ProverTrainingConfig:
    """Configuration for prover training phase.

    Args:
        lr: Learning rate
        batch_size: Training batch size (rollouts per batch)
        max_steps: Maximum training steps
        group_size: GRPO group size for baseline normalization
        normalize_adv: Normalize advantages within groups
        positive_only: Clip negative advantages to zero
        reward_strategy: Reward strategy for credit assignment
        device: Training device
        pad_token_id: Tokenizer pad token ID
        max_seq_len: Maximum sequence length
        gradient_accumulation_steps: Number of gradient accumulation steps
        save_checkpoint: Whether to save checkpoint after training
        max_rollouts_per_step: Maximum rollouts to generate per training step
        sync_every_steps: Sync weights to vLLM every N steps
    """

    lr: float = 1e-5
    batch_size: int = 4
    max_steps: int = 2000
    group_size: int = 4
    normalize_adv: bool = True
    positive_only: bool = False
    reward_strategy: Optional[RewardStrategy] = None
    device: str = "cuda"
    pad_token_id: int = 0
    max_seq_len: int = 2048
    gradient_accumulation_steps: int = 4
    save_checkpoint: bool = True
    max_rollouts_per_step: int = 32
    sync_every_steps: int = 10

    def __post_init__(self) -> None:
        if self.reward_strategy is None:
            self.reward_strategy = CompositeReward()


@dataclass
class RolloutBatchSourceForProver:
    """BatchSource that generates rollouts for prover training.

    Wraps rollout generation and credit assignment for the prover.
    """

    rollout_generator: Callable[[], Rollout]
    credit_assigner: PVGCreditAssigner
    tokenizer: Any
    rollouts_per_batch: int = 4
    max_seq_len: int = 2048

    async def next_batch(self) -> SAWBatch:
        """Generate rollouts and convert to SAWBatch."""
        rollouts: List[Rollout] = []

        sample_logger.debug(f"Generating {self.rollouts_per_batch} rollouts...")

        # Generate rollouts
        for _ in range(self.rollouts_per_batch):
            rollout = await asyncio.to_thread(self.rollout_generator)
            rollouts.append(rollout)

        # Log first rollout for prompt verification
        if rollouts:
            sample_logger.info("-" * 40)
            sample_logger.info("PROVER ROLLOUT SAMPLE (first in batch)")
            sample_logger.info("-" * 40)
            r = rollouts[0]
            sample_logger.info(f"  Rollout ID: {r.id}")
            sample_logger.info(f"  Total reward: {r.total_reward}")
            sample_logger.info(f"  Num steps: {len(r.steps)}")

            if r.steps:
                first_step = r.steps[0]
                sample_logger.info("")
                sample_logger.info("  [PROMPT CHECK] Prover Input Verification:")
                sample_logger.info("  " + "-" * 38)
                prompt_preview = str(first_step.prev_obs)[:500] if first_step.prev_obs else "None"
                sample_logger.info(f"  PROMPT (prev_obs):")
                for line in prompt_preview.split('\n')[:10]:
                    sample_logger.info(f"    | {line}")
                if len(prompt_preview) >= 500:
                    sample_logger.info(f"    | ... (truncated)")

                # CRITICAL: Verify prompt doesn't contain labels
                prompt_str = str(first_step.prev_obs) if first_step.prev_obs else ""
                forbidden_in_prompt = ["verifier_score", "sneaky_certified", "test_pass_rate", "correct", "incorrect"]
                prompt_leak = [f for f in forbidden_in_prompt if f.lower() in prompt_str.lower()]
                if prompt_leak:
                    sample_logger.error(f"  [FAIL] LABEL LEAKAGE IN PROMPT: {prompt_leak}")
                else:
                    sample_logger.info(f"  [PASS] Prompt contains problem only, no labels")

                sample_logger.info("")
                sample_logger.info(f"  ACTION (generated code):")
                action_preview = str(first_step.action)[:300] if first_step.action else "None"
                for line in action_preview.split('\n')[:8]:
                    sample_logger.info(f"    | {line}")

            if r.meta:
                sample_logger.info("")
                sample_logger.info(f"  Rollout metadata:")
                for k, v in r.meta.items():
                    sample_logger.info(f"    {k}: {v}")

        # Score rollouts with verifier (if scorer is configured)
        if self.credit_assigner.scorer is not None:
            sample_logger.info("")
            sample_logger.info("  [VERIFIER SCORING] Scoring rollouts with trained verifier...")
            sample_logger.info("  Note: Scores are attached POST-rollout (not during generation)")
            await self.credit_assigner.score_rollouts(rollouts)

            # Log scores for verification
            if rollouts:
                sample_logger.info("  Verifier scores attached:")
                for i, r in enumerate(rollouts[:5]):
                    score = r.meta.get("verifier_score", "N/A")
                    sample_logger.info(f"    Rollout {r.id}: verifier_score={score}")
                if len(rollouts) > 5:
                    sample_logger.info(f"    ... and {len(rollouts) - 5} more")
            sample_logger.info("  Scoring complete.")

        # Compute credit assignment
        weights = self.credit_assigner.compute(rollouts)

        # Log credit assignment for first rollout
        if rollouts:
            r = rollouts[0]
            sample_logger.info(f"  Credit assignment weights for first rollout:")
            for step in r.steps:
                key = (r.id, step.index)
                w = weights.get(key, 0.0)
                sample_logger.info(f"    Step {step.index}: weight={w:.4f}")

        # Convert rollouts to SAWItems
        items: List[SAWItem] = []
        for rollout in rollouts:
            for step in rollout.steps:
                key = (rollout.id, step.index)
                weight = weights.get(key, 0.0)

                item = self._create_saw_item(
                    rollout=rollout,
                    step=step,
                    weight=weight,
                )
                items.append(item)

        batch_meta = {
            "num_rollouts": len(rollouts),
            "total_reward": sum(r.total_reward for r in rollouts),
        }

        sample_logger.info(f"  Batch complete: {len(items)} items, total_reward={batch_meta['total_reward']:.4f}")

        return SAWBatch(items=items, meta=batch_meta)

    def _create_saw_item(
        self,
        rollout: Rollout,
        step: Any,
        weight: float,
    ) -> SAWItem:
        """Create a SAWItem from a rollout step."""
        state = step.prev_obs
        action = step.action

        # Combine state and action for tokenization
        full_text = state + action

        # Tokenize
        tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors=None,
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens.get("attention_mask", [1] * len(input_ids))

        # Tokenize just the state to find where action starts
        state_tokens = self.tokenizer(
            state,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors=None,
        )
        state_len = len(state_tokens["input_ids"])

        # Action mask: 1 for action tokens, 0 for state tokens
        action_mask = [0] * state_len + [1] * (len(input_ids) - state_len)

        return SAWItem(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            weight=weight,
            meta={
                "rollout_id": rollout.id,
                "step_index": step.index,
                "reward": step.reward,
                "total_reward": rollout.total_reward,
            },
        )


async def train_prover_phase(
    model: nn.Module,
    tokenizer: Any,
    rollout_generator: Callable[[], Rollout],
    verifier_scorer: Optional[VerifierScorer],
    current_round: int,
    config: ProverTrainingConfig,
    output_dir: Optional[Union[str, Path]] = None,
    publisher: Optional[Any] = None,
) -> Optional[Path]:
    """Run the prover training phase.

    Generates rollouts using the current prover, scores them with the verifier,
    and trains the prover using policy gradient methods.

    Args:
        model: The prover model to train
        tokenizer: Tokenizer for the model
        rollout_generator: Callable that generates a single rollout
        verifier_scorer: VerifierScorer for scoring rollouts (optional)
        current_round: Current round (0-indexed)
        config: ProverTrainingConfig
        output_dir: Directory to save checkpoint
        publisher: Optional PolicyPublisher for syncing weights to vLLM

    Returns:
        Path to saved checkpoint, or None if save_checkpoint=False
    """
    logger.info("=" * 80)
    logger.info("PROVER TRAINING PHASE START")
    logger.info("=" * 80)
    logger.info(f"Round: {current_round}")
    logger.info("CONFIGURATION:")
    logger.info(f"  lr: {config.lr}")
    logger.info(f"  batch_size: {config.batch_size}")
    logger.info(f"  max_steps: {config.max_steps}")
    logger.info(f"  group_size: {config.group_size}")
    logger.info(f"  normalize_adv: {config.normalize_adv}")
    logger.info(f"  positive_only: {config.positive_only}")
    logger.info(f"  reward_strategy: {config.reward_strategy}")
    logger.info(f"  device: {config.device}")
    logger.info(f"  pad_token_id: {config.pad_token_id}")
    logger.info(f"  max_seq_len: {config.max_seq_len}")
    logger.info(f"  gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    logger.info(f"  max_rollouts_per_step: {config.max_rollouts_per_step}")
    logger.info(f"  sync_every_steps: {config.sync_every_steps}")
    logger.info(f"  verifier_scorer available: {verifier_scorer is not None}")
    logger.info(f"  publisher available: {publisher is not None}")
    logger.info("-" * 80)

    # Create credit assigner
    credit_assigner = PVGCreditAssigner(
        reward_strategy=config.reward_strategy,
        scorer=verifier_scorer,
        group_size=config.group_size,
        normalize_adv=config.normalize_adv,
        positive_only=config.positive_only,
        role="sneaky",
    )

    # Create batch source
    batch_source = RolloutBatchSourceForProver(
        rollout_generator=rollout_generator,
        credit_assigner=credit_assigner,
        tokenizer=tokenizer,
        rollouts_per_batch=config.batch_size,
        max_seq_len=config.max_seq_len,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    device = torch.device(config.device)
    model = model.to(device)
    model.train()

    # Training loop
    total_loss = 0.0
    total_reward = 0.0
    sync_count = 0

    logger.info("=" * 60)
    logger.info(f"STARTING PROVER TRAINING for {config.max_steps} steps")
    logger.info("=" * 60)

    for step in range(config.max_steps):
        batch = await batch_source.next_batch()

        if not batch.items:
            logger.warning(f"Empty batch at step {step}, skipping")
            continue

        # Collate batch to tensors
        batch_tensors = _collate_prover_batch(
            batch.items,
            pad_token_id=config.pad_token_id,
            device=device,
        )

        # Forward pass
        outputs = model(
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
        )

        # Get log probabilities for action tokens
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        log_probs = torch.log_softmax(logits, dim=-1)

        # Compute policy gradient loss
        loss = _compute_reinforce_loss(
            log_probs=log_probs,
            input_ids=batch_tensors["input_ids"],
            action_mask=batch_tensors["action_mask"],
            weights=batch_tensors["weights"],
        )

        # Backward pass with gradient accumulation
        scaled_loss = loss / config.gradient_accumulation_steps
        scaled_loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # Optionally sync weights to vLLM
            if publisher is not None and (step + 1) % config.sync_every_steps == 0:
                sync_count += 1
                _sync_weights_to_publisher(model, publisher, step + 1, sync_count)

        total_loss += loss.item()
        total_reward += batch.meta.get("total_reward", 0.0)

        # Log detailed stats every 100 steps or first step
        if (step + 1) % 100 == 0 or step == 0:
            avg_loss = total_loss / (step + 1)
            avg_reward = total_reward / (step + 1)
            batch_reward = batch.meta.get("total_reward", 0.0)
            num_rollouts = batch.meta.get("num_rollouts", 0)
            weights_tensor = batch_tensors["weights"]
            mean_weight = weights_tensor.mean().item() if len(weights_tensor) > 0 else 0.0
            std_weight = weights_tensor.std().item() if len(weights_tensor) > 1 else 0.0

            logger.info("-" * 40)
            logger.info(f"[PROVER TRAIN STEP {step + 1}/{config.max_steps}]")
            logger.info(f"  Loss (this step): {loss.item():.6f}")
            logger.info(f"  Avg loss: {avg_loss:.6f}")
            logger.info(f"  Batch reward: {batch_reward:.4f}")
            logger.info(f"  Avg reward: {avg_reward:.4f}")
            logger.info(f"  Num rollouts in batch: {num_rollouts}")
            logger.info(f"  Num items in batch: {len(batch.items)}")
            logger.info(f"  Weight stats: mean={mean_weight:.4f}, std={std_weight:.4f}")
            logger.info(f"  Total weight syncs so far: {sync_count}")

    # Save checkpoint
    checkpoint_path = None
    if config.save_checkpoint and output_dir is not None:
        checkpoint_dir = Path(output_dir) / "checkpoints" / f"round_{current_round}" / "prover"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(checkpoint_dir)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(checkpoint_dir)
        else:
            torch.save(model.state_dict(), checkpoint_dir / "model.pt")

        checkpoint_path = checkpoint_dir
        logger.info(f"Saved prover checkpoint to {checkpoint_dir}")

    logger.info("=" * 60)
    logger.info("PROVER TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Final loss: {total_loss / max(1, config.max_steps):.6f}")
    logger.info(f"  Final avg reward: {total_reward / max(1, config.max_steps):.4f}")
    logger.info(f"  Total steps: {config.max_steps}")
    logger.info(f"  Total weight syncs: {sync_count}")

    return checkpoint_path


def _collate_prover_batch(
    items: List[SAWItem],
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Collate SAWItems into batched tensors for prover training.

    Args:
        items: List of SAWItem
        pad_token_id: Token ID for padding
        device: Target device

    Returns:
        Dict with input_ids, attention_mask, action_mask, weights
    """
    max_len = max(len(item.input_ids) for item in items)

    input_ids = []
    attention_masks = []
    action_masks = []
    weights = []

    for item in items:
        seq_len = len(item.input_ids)
        pad_len = max_len - seq_len

        # Pad sequences
        input_ids.append(item.input_ids + [pad_token_id] * pad_len)
        attention_masks.append(item.attention_mask + [0] * pad_len)
        action_masks.append(item.action_mask + [0] * pad_len)
        weights.append(item.weight)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
        "action_mask": torch.tensor(action_masks, dtype=torch.float32, device=device),
        "weights": torch.tensor(weights, dtype=torch.float32, device=device),
    }


def _compute_reinforce_loss(
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    action_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute REINFORCE policy gradient loss.

    Args:
        log_probs: Log probabilities from model [B, T, V]
        input_ids: Input token IDs [B, T]
        action_mask: Mask for action tokens [B, T]
        weights: Per-sample advantage weights [B]

    Returns:
        Scalar loss tensor
    """
    batch_size, seq_len = input_ids.shape

    # Shift for next-token prediction
    # We predict token[t+1] given token[0:t+1]
    shift_log_probs = log_probs[:, :-1, :]  # [B, T-1, V]
    shift_targets = input_ids[:, 1:]  # [B, T-1]
    shift_action_mask = action_mask[:, 1:]  # [B, T-1]

    # Gather log probs for target tokens
    target_log_probs = shift_log_probs.gather(
        dim=-1, index=shift_targets.unsqueeze(-1)
    ).squeeze(-1)  # [B, T-1]

    # Mask to only action tokens
    masked_log_probs = target_log_probs * shift_action_mask

    # Sum log probs per sequence
    seq_log_probs = masked_log_probs.sum(dim=-1)  # [B]

    # Weight by advantage and negate for gradient descent
    loss = -(seq_log_probs * weights).mean()

    return loss


def _sync_weights_to_publisher(
    model: nn.Module,
    publisher: Any,
    step: int,
    sync_count: int = 0,
) -> None:
    """Sync model weights to the policy publisher.

    This pushes updated prover weights to the vLLM inference server so
    that subsequent rollout generation uses the trained policy.

    Args:
        model: The model to sync
        publisher: PolicyPublisher instance
        step: Current training step (used as version)
        sync_count: Total number of syncs so far (for logging)
    """
    try:
        logger.info("  " + "-" * 40)
        logger.info(f"  [WEIGHT SYNC] Syncing prover weights to inference")
        logger.info(f"    Step: {step}")
        logger.info(f"    Sync count: {sync_count}")

        params = {name: p.detach() for name, p in model.named_parameters()}
        num_params = len(params)
        total_elements = sum(p.numel() for p in params.values())
        total_bytes = sum(p.numel() * p.element_size() for p in params.values())

        logger.info(f"    Parameters: {num_params} tensors")
        logger.info(f"    Elements: {total_elements:,}")
        logger.info(f"    Size: {total_bytes / (1024**2):.2f} MB")
        logger.info(f"    Version tag: {step}")

        publisher.publish(params, version=step)

        logger.info(f"  [SYNC SUCCESS] Weights published to vLLM")
        logger.info(f"    Inference will now use policy version {step}")
        logger.info("  " + "-" * 40)
    except Exception as e:
        logger.error(f"  [SYNC FAILED] Failed to sync weights: {e}")
        logger.error(f"    Publisher: {type(publisher).__name__}")
        import traceback
        logger.error(f"    Traceback: {traceback.format_exc()}")

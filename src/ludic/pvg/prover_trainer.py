"""PVG Prover Training Phase.

This module provides the prover training phase for PVG, using policy gradient
training with verifier scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import torch
from torch import nn

from ludic.training.algorithm import RLAlgorithm
from ludic.training.loss import ReinforceLoss
from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.training.batching.synced_batching import RolloutBatchSource
from ludic.training.types import RolloutRequest
from ludic.training.filters import drop_truncated
from ludic.training.checkpoint import CheckpointConfig, CheckpointManager
from ludic.training.loggers import PrintLogger
from ludic.training.stats import Reducer
from ludic.training.batching.rollout_engine import RolloutEngine
from ludic.pvg.credit import PVGCreditAssigner
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
        micro_token_budget: Token budget per micro-batch (default: batch_size * max_seq_len)
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
    micro_token_budget: Optional[int] = None
    save_checkpoint: bool = True
    max_rollouts_per_step: int = 32
    sync_every_steps: int = 10

    def __post_init__(self) -> None:
        if self.reward_strategy is None:
            self.reward_strategy = CompositeReward()
        if self.micro_token_budget is None:
            self.micro_token_budget = self.batch_size * self.max_seq_len


async def train_prover_phase(
    model: nn.Module,
    tokenizer: Any,
    rollout_engine: RolloutEngine,
    requests_fn: Callable[[], List[RolloutRequest]],
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
        rollout_engine: RolloutEngine used for rollout generation
        requests_fn: Callable that returns rollout requests for each batch
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
    logger.info(f"  micro_token_budget: {config.micro_token_budget}")
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

    # Rollout preprocessor: attach verifier scores if configured
    async def _score_rollouts(rollouts: List[Any]) -> List[Any]:
        if credit_assigner.scorer is None:
            return rollouts
        if rollouts:
            r = rollouts[0]
            sample_logger.info("-" * 40)
            sample_logger.info("PROVER ROLLOUT SAMPLE (first in batch)")
            sample_logger.info("-" * 40)
            sample_logger.info(f"  Rollout ID: {r.id}")
            sample_logger.info(f"  Total reward: {r.total_reward}")
            sample_logger.info(f"  Num steps: {len(r.steps)}")
            if r.steps:
                first_step = r.steps[0]
                sample_logger.info("")
                sample_logger.info("  [PROMPT CHECK] Prover Input Verification:")
                sample_logger.info("  " + "-" * 38)
                prompt_preview = str(first_step.prev_obs)[:500] if first_step.prev_obs else "None"
                sample_logger.info("  PROMPT (prev_obs):")
                for line in prompt_preview.split("\n")[:10]:
                    sample_logger.info(f"    | {line}")
                if len(prompt_preview) >= 500:
                    sample_logger.info("    | ... (truncated)")
                sample_logger.info("")
                sample_logger.info("  ACTION (generated code):")
                action_preview = str(first_step.action)[:300] if first_step.action else "None"
                for line in action_preview.split("\n")[:8]:
                    sample_logger.info(f"    | {line}")
        sample_logger.info("")
        sample_logger.info("  [VERIFIER SCORING] Scoring rollouts with trained verifier...")
        await credit_assigner.score_rollouts(rollouts)
        if rollouts:
            for r in rollouts[:5]:
                score = r.meta.get("verifier_score", "N/A")
                sample_logger.info(f"    Rollout {r.id}: verifier_score={score}")
        return rollouts

    def _prompt_leak_filter(item: Any) -> bool:
        prompt = str(item.meta.get("prev_obs", ""))
        forbidden = ["verifier_score", "sneaky_certified", "test_pass_rate", "correct", "incorrect"]
        leaks = [f for f in forbidden if f.lower() in prompt.lower()]
        if leaks:
            sample_logger.error(f"[FAIL] LABEL LEAKAGE IN PROMPT: {leaks}")
            return False
        return True

    def _combined_filter(item: Any) -> bool:
        return drop_truncated(item) and _prompt_leak_filter(item)

    batch_source = RolloutBatchSource(
        orchestrator=rollout_engine,
        credit_assigner=credit_assigner,
        requests_fn=requests_fn,
        max_steps=1,
        concurrency=config.max_rollouts_per_step,
        sample_filter=_combined_filter,
        rollout_preprocessor=_score_rollouts,
    )

    algo = RLAlgorithm(
        name="pvg_reinforce",
        credit_assigner=credit_assigner,
        loss=ReinforceLoss(),
    )

    model = model.to(torch.device(config.device))

    train_cfg = TrainerConfig(
        lr=config.lr,
        max_seq_len=config.max_seq_len,
        micro_token_budget=int(config.micro_token_budget),
        pad_token_id=config.pad_token_id,
        model_device=config.device,
        sync_every_steps=config.sync_every_steps,
        log_every=100,
    )

    checkpointer = None
    if config.save_checkpoint and output_dir is not None:
        checkpoint_dir = Path(output_dir) / "checkpoints" / f"round_{current_round}" / "prover"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cfg = CheckpointConfig(
            dir=checkpoint_dir,
            save_every=config.max_steps,
            keep_last_n=1,
        )
        checkpointer = CheckpointManager(checkpoint_cfg)

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=publisher,
        cfg=train_cfg,
        checkpointer=checkpointer,
        train_logger=PrintLogger(),
        reducers={"reward": Reducer(kind="mean", source="reward")},
        total_training_steps=config.max_steps,
    )

    logger.info("=" * 60)
    logger.info(f"STARTING PROVER TRAINING for {config.max_steps} steps")
    logger.info("=" * 60)

    await trainer.train(max_steps=config.max_steps)

    checkpoint_path = None
    if config.save_checkpoint and output_dir is not None:
        checkpoint_path = Path(output_dir) / "checkpoints" / f"round_{current_round}" / "prover"
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(checkpoint_path)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path / "model.pt")
        logger.info(f"Saved prover checkpoint to {checkpoint_path}")

    logger.info("=" * 60)
    logger.info("PROVER TRAINING COMPLETE")
    logger.info("=" * 60)

    return checkpoint_path

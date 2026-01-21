"""Cold-start RLHF: Train RM on preferences, then use RM for policy training.

This script demonstrates the classical RLHF workflow:
1. Train a reward model on preference data (Bradley-Terry)
2. Use the RM to score policy rollouts
3. Train the policy using RM scores as rewards (or bonuses)

Usage:
    # Full pipeline: train RM then policy
    python cold_start_rlhf.py \
        --mode full \
        --policy-model Qwen/Qwen2.5-0.5B-Instruct \
        --rm-base-model Qwen/Qwen2.5-0.5B \
        --preference-data data/preferences.jsonl \
        --rm-steps 100 \
        --policy-steps 50

    # Policy training only (with pre-trained RM)
    python cold_start_rlhf.py \
        --mode train \
        --policy-model Qwen/Qwen2.5-0.5B-Instruct \
        --rm-checkpoint checkpoints_rm/step_100 \
        --policy-steps 50

Note: Requires a running vLLM server for policy inference:
    CUDA_VISIBLE_DEVICES=0 python -m ludic.inference.vllm_server \
        --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec
from ludic.interaction import SingleAgentProtocol
from ludic.parsers import identity_parser
from ludic.distributed.adapters import create_vllm_publisher
from ludic.types import Rollout
from ludic.training import (
    RolloutEngine,
    RolloutBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    make_dataset_queue_requests_fn,
    Reducer,
    PrintLogger,
    default_reducers,
    CreditAssigner,
    SAWBatch,
    RolloutRequest,
    SampleFilter,
)
from ludic.training.algorithm import make_grpo_with_rm
from ludic.training.reward_credit import RewardModelCreditAssigner
from ludic.envs import SingleAgentEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple QA Environment for demonstration
# ---------------------------------------------------------------------------


class SimpleQAEnv(SingleAgentEnv):
    """Simple QA environment for RLHF demonstration.

    This is a minimal environment that presents a question and rewards
    based on RM scores (no ground-truth verification).
    """

    def __init__(self, sample: Dict[str, Any], system_prompt: str = ""):
        self.sample = sample
        self.system_prompt = system_prompt
        self._done = False

    def reset(self) -> tuple[str, Dict[str, Any]]:
        self._done = False
        question = self.sample.get("question", self.sample.get("prompt", ""))
        return question, {"sample_id": self.sample.get("id", "")}

    def step(self, action: str) -> tuple[str, float, bool, bool, Dict[str, Any]]:
        # In pure RM-based RLHF, the env reward is 0 or minimal
        # The actual reward comes from the RM scorer
        self._done = True
        return "", 0.0, True, False, {"action": action}

    @property
    def system_prompt_for_agent(self) -> Optional[str]:
        return self.system_prompt if self.system_prompt else None


# ---------------------------------------------------------------------------
# Local RM Scorer (runs model on same process)
# ---------------------------------------------------------------------------


class LocalRewardModelScorer:
    """Local reward model scorer for scoring rollouts.

    Loads the RM checkpoint and scores sequences directly.
    Adds rm_score to rollout.meta for use with RewardModelCreditAssigner.

    IMPORTANT: If the RM was trained with chat templates (recommended), set
    use_chat_template=True to ensure scoring uses the same format. Mismatched
    formatting between training and inference will degrade RM quality.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 16,
        score_key: str = "rm_score",
        use_chat_template: bool = True,
        system_prompt: Optional[str] = None,
        max_length: int = 512,
    ):
        self.device = device
        self.batch_size = batch_size
        self.score_key = score_key
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt
        self.max_length = max_length

        logger.info(f"Loading reward model from: {checkpoint_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
        )
        self.model.to(device)
        self.model.eval()

        if use_chat_template:
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise ValueError(
                    "use_chat_template=True requires a tokenizer with apply_chat_template(). "
                    "Either use a tokenizer from an instruction-tuned model, or set "
                    "use_chat_template=False (not recommended if RM was trained with templates)."
                )
            logger.info("RM scorer using chat template for tokenization")
        else:
            logger.warning("RM scorer NOT using chat template - may cause distribution mismatch")

    def _extract_prompt_and_completion(self, rollout: Rollout) -> tuple[str, str]:
        """Extract prompt (user message) and completion (assistant response) from rollout."""
        # In a typical single-turn QA rollout:
        # - step.prev_obs is the user's question/prompt
        # - step.action is the assistant's response
        prompt_parts = []
        completion_parts = []

        for step in rollout.steps:
            if step.prev_obs:
                prompt_parts.append(str(step.prev_obs))
            if step.action:
                completion_parts.append(str(step.action))

        prompt = "\n".join(prompt_parts)
        completion = "\n".join(completion_parts)
        return prompt, completion

    def _tokenize_with_chat_template(self, prompt: str, completion: str) -> List[int]:
        """Tokenize using chat template format."""
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": completion})

        token_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )

        # Truncate if needed
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        return token_ids

    def score_rollouts(self, rollouts: List[Rollout]) -> None:
        """Score rollouts and store results in metadata (modifies in-place)."""
        if not rollouts:
            return

        scores: List[float] = []

        # Process rollouts - extract prompt/completion pairs
        rollout_data = [self._extract_prompt_and_completion(r) for r in rollouts]

        # Batch score
        for i in range(0, len(rollout_data), self.batch_size):
            batch_data = rollout_data[i:i + self.batch_size]

            if self.use_chat_template:
                # Tokenize each item with chat template
                batch_token_ids = [
                    self._tokenize_with_chat_template(prompt, completion)
                    for prompt, completion in batch_data
                ]

                # Pad manually
                max_len = max(len(ids) for ids in batch_token_ids)
                padded_ids = []
                attention_masks = []
                for ids in batch_token_ids:
                    pad_len = max_len - len(ids)
                    padded_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
                    attention_masks.append([1] * len(ids) + [0] * pad_len)

                input_ids = torch.tensor(padded_ids, device=self.device)
                attention_mask = torch.tensor(attention_masks, device=self.device)
            else:
                # Legacy: raw text tokenization (not recommended)
                batch_texts = [f"{prompt}\n{completion}" for prompt, completion in batch_data]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

            # Score
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_scores = outputs.logits.squeeze(-1).float().cpu().tolist()

            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)

        # Store in metadata
        for rollout, score in zip(rollouts, scores):
            rollout.meta[self.score_key] = score


# ---------------------------------------------------------------------------
# RM-Scoring Batch Source Wrapper
# ---------------------------------------------------------------------------


class RMScoringCreditAssigner:
    """Credit assigner that scores rollouts with RM before delegating.

    Wraps an inner credit assigner (like RewardModelCreditAssigner) and ensures
    rollouts are scored by the RM before credit computation.
    """

    def __init__(
        self,
        rm_scorer: LocalRewardModelScorer,
        inner_assigner: CreditAssigner,
    ):
        self._rm_scorer = rm_scorer
        self._inner = inner_assigner

    def compute(self, rollouts: List[Rollout]) -> Dict[tuple, float]:
        """Score rollouts with RM, then compute credits."""
        # Score rollouts (adds rm_score to metadata)
        self._rm_scorer.score_rollouts(rollouts)
        # Delegate to inner assigner (e.g., RewardModelCreditAssigner)
        return self._inner.compute(rollouts)


# ---------------------------------------------------------------------------
# Main training functions
# ---------------------------------------------------------------------------


def train_reward_model(
    base_model: str,
    preference_data_path: Path,
    output_dir: Path,
    steps: int = 100,
    batch_size: int = 8,
    lr: float = 1e-5,
) -> Path:
    """Train reward model on preference data. Returns checkpoint path."""
    import subprocess
    import sys

    # Use the dedicated RM training script
    script_path = Path(__file__).parent / "train_rm_bradley_terry.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--model", base_model,
        "--dataset", str(preference_data_path),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--output-dir", str(output_dir),
    ]

    logger.info(f"Training RM: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    return output_dir / f"step_{steps}"


def train_policy_with_rm(
    policy_model: str,
    rm_checkpoint: str,
    prompts: List[Dict[str, Any]],
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    train_steps: int = 50,
    group_size: int = 8,
    rollouts_per_update: int = 64,
    concurrency: int = 32,
    max_seq_len: int = 1024,
    max_completion_tokens: int = 512,
    rm_mode: str = "bonus",
    rm_coeff: float = 0.5,
    system_prompt: str = "",
):
    """Train policy using RM scores."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load policy model for training
    logger.info(f"Loading policy model: {policy_model}")
    tokenizer = AutoTokenizer.from_pretrained(policy_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        policy_model,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)

    # Create RM scorer for scoring rollouts
    # Uses chat template by default to match RM training format
    rm_scorer = LocalRewardModelScorer(
        rm_checkpoint,
        device=device,
        use_chat_template=True,
        system_prompt=system_prompt if system_prompt else None,
    )

    # Inference client for rollout generation
    client = VLLMChatClient(host=host, port=port, enable_weight_updates=True)
    publisher = create_vllm_publisher(client)

    # Environment and protocol registries
    env_registry = {
        "qa": lambda sample: SimpleQAEnv(sample=sample, system_prompt=system_prompt)
    }

    def protocol_factory():
        return SingleAgentProtocol(
            agent=Agent(
                client=client,
                model=policy_model,
                ctx=FullDialog(),
                parser=identity_parser,
            )
        )

    protocol_registry = {"single_agent": protocol_factory}

    # Algorithm: GRPO with RM bonus
    # This returns an RLAlgorithm with RewardModelCreditAssigner as credit_assigner
    algo = make_grpo_with_rm(
        group_size=group_size,
        rm_mode=rm_mode,
        rm_coeff=rm_coeff,
    )

    # Wrap the credit assigner to score rollouts before computing credits
    # This ensures rm_score is in rollout.meta before RewardModelCreditAssigner runs
    scoring_credit_assigner = RMScoringCreditAssigner(
        rm_scorer=rm_scorer,
        inner_assigner=algo.credit_assigner,
    )

    # Setup prompt queue
    prompts_q: queue.Queue = queue.Queue()
    for idx, p in enumerate(prompts):
        prompts_q.put((idx, p))

    # Rollout engine
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    # Inference spec
    train_inference = InferenceSpec(
        sampling=SamplingParams(
            temperature=1.0,
            max_tokens=max_completion_tokens,
        ),
        return_=ReturnSpec.for_rl(top_logprobs_k=1),
    )

    base_requests = rollouts_per_update // group_size
    requests_fn = make_dataset_queue_requests_fn(
        prompts_q,
        batch_size=base_requests,
        env_kind="qa",
        protocol_kind="single_agent",
        inference=train_inference,
        protocol_kwargs={},
        request_meta_fn=lambda idx, sample: {"sample_index": idx},
        env_seed_fn=lambda idx, _: idx,
        sampling_seed_fn=lambda idx, _: idx,
        group_size=group_size,
    )

    # Use RolloutBatchSource with our scoring credit assigner
    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=scoring_credit_assigner,  # Scores rollouts, then computes credits
        requests_fn=requests_fn,
        max_steps=1,
        concurrency=concurrency,
    )

    # Trainer config
    cfg = TrainerConfig(
        model_device=device,
        max_seq_len=max_seq_len,
        micro_token_budget=16384,
        max_grad_norm=0.5,
        pad_token_id=tokenizer.pad_token_id,
    )

    checkpoint_cfg = CheckpointConfig(
        output_dir="checkpoints_policy_rm",
        every_n_steps=25,
        max_to_keep=2,
    )

    # Reducers for logging
    reducers = {
        "rm_score_mean": Reducer(kind="mean", source="rm_score"),
        "rm_score_std": Reducer(kind="std", source="rm_score"),
        **default_reducers(),
    }

    train_logger = PrintLogger(
        prefix="[policy]",
        keys=[
            "train/loss",
            "train/avg_total_reward",
            "train/rm_score_mean",
            "train/rm_score_std",
        ],
        precision=4,
    )

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=publisher,
        enable_gradient_checkpointing=True,
        cfg=cfg,
        checkpoint_config=checkpoint_cfg,
        train_logger=train_logger,
        reducers=reducers,
    )

    logger.info(f"Training policy for {train_steps} steps with RM scoring...")
    trainer.train_sync(train_steps)

    logger.info("Policy training complete!")


def main():
    parser = argparse.ArgumentParser(description="Cold-start RLHF")
    parser.add_argument(
        "--mode",
        choices=["full", "train", "rm-only"],
        default="train",
        help="full=train RM + policy, train=policy only, rm-only=RM only",
    )

    # Policy model args
    parser.add_argument("--policy-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")

    # RM args
    parser.add_argument("--rm-base-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--rm-checkpoint", type=str, help="Pre-trained RM checkpoint")
    parser.add_argument("--rm-output-dir", default="checkpoints_rm")
    parser.add_argument("--rm-steps", type=int, default=100)

    # Data args
    parser.add_argument("--preference-data", type=str, help="Path to preference JSONL")
    parser.add_argument("--prompts-data", type=str, help="Path to prompts JSONL for policy training")

    # Training args
    parser.add_argument("--policy-steps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollouts-per-update", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--rm-mode", default="bonus", choices=["replace", "add", "bonus", "weighted"])
    parser.add_argument("--rm-coeff", type=float, default=0.5)
    parser.add_argument("--system-prompt", default="")

    args = parser.parse_args()

    # Validate args
    if args.mode in ["full", "rm-only"] and not args.preference_data:
        parser.error("--preference-data required for RM training")

    if args.mode == "train" and not args.rm_checkpoint:
        parser.error("--rm-checkpoint required for policy training without RM training")

    # Step 1: Train RM (if needed)
    rm_checkpoint = args.rm_checkpoint
    if args.mode in ["full", "rm-only"]:
        logger.info("=== Phase 1: Training Reward Model ===")
        rm_checkpoint = train_reward_model(
            base_model=args.rm_base_model,
            preference_data_path=Path(args.preference_data),
            output_dir=Path(args.rm_output_dir),
            steps=args.rm_steps,
        )
        rm_checkpoint = str(rm_checkpoint)

        if args.mode == "rm-only":
            logger.info(f"RM training complete. Checkpoint: {rm_checkpoint}")
            return

    # Step 2: Load prompts for policy training
    prompts: List[Dict[str, Any]] = []
    if args.prompts_data:
        logger.info(f"Loading prompts from: {args.prompts_data}")
        with open(args.prompts_data, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
    elif args.preference_data:
        # Use prompts from preference data
        logger.info("Using prompts from preference data")
        with open(args.preference_data, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    data = json.loads(line)
                    prompts.append({
                        "id": i,
                        "question": data.get("prompt", ""),
                    })
                if len(prompts) >= 1000:  # Limit for demo
                    break
    else:
        # Generate some demo prompts
        logger.info("Using demo prompts")
        prompts = [
            {"id": i, "question": f"Tell me about topic {i}"}
            for i in range(100)
        ]

    logger.info(f"Loaded {len(prompts)} prompts for policy training")

    # Step 3: Train policy with RM
    logger.info("=== Phase 2: Training Policy with RM ===")
    train_policy_with_rm(
        policy_model=args.policy_model,
        rm_checkpoint=rm_checkpoint,
        prompts=prompts,
        host=args.host,
        port=args.port,
        train_steps=args.policy_steps,
        group_size=args.group_size,
        rollouts_per_update=args.rollouts_per_update,
        concurrency=args.concurrency,
        rm_mode=args.rm_mode,
        rm_coeff=args.rm_coeff,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()

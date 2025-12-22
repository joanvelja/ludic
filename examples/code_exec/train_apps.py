"""
APPS code generation training scaffold using CodeExecEnv with LoRA.

This wires together:
  - HuggingFace datasets for APPS code samples
  - CodeExecEnv with sandboxed execution (Docker or Podman-HPC)
  - SingleAgentProtocol with async env support
  - LoRA adapters via PEFT for efficient fine-tuning
  - GRPO with optional KL regularization
  - Baseline + periodic evaluation on held-out samples
  - RichLiveLogger (terminal dashboard) or WandB (cloud logging)

Requirements:
  - Container runtime: Docker daemon OR Podman-HPC (auto-detected)
  - pip install docker>=7.0.0 datasets peft (for Docker backend)
  - GPU(s) for training (optional for rollout-only mode)

Usage:
  # Start vLLM server (in one terminal)
  CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \\
      --model Qwen/Qwen2.5-3B-Instruct

  # Run training with terminal dashboard (default)
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/code_exec/train_apps.py \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --limit 500 --eval-samples 200 --train-steps 100 --final-save

  # Run training with KL regularization
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/code_exec/train_apps.py \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --limit 500 --eval-samples 200 --train-steps 100 \\
      --kl-coeff 0.01 --final-save

  # Run training with WandB logging
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/code_exec/train_apps.py \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --limit 500 --eval-samples 200 --train-steps 100 \\
      --wandb --wandb-project ludic-apps --final-save

Key Features:
  - LoRA: rank=8, alpha=16, target_modules="all-linear" (configurable)
  - Eval: Baseline before training, periodic eval every N steps
  - Logging: Terminal sparkline dashboard or WandB cloud tracking
  - KL regularization: Optional penalty to prevent policy drift

See README.md for detailed setup instructions.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import queue
from typing import Any, Dict, List

import torch
from datasets import load_dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec
from ludic.interaction import SingleAgentProtocol
from ludic.parsers import ParseResult
from ludic.distributed.adapters import create_vllm_publisher
from ludic.eval import EngineEvaluator
from ludic.training import (
    RolloutEngine,
    RolloutBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    make_dataset_queue_requests_fn,
    RequestsExhausted,
    RolloutRequest,
    EnvSpec,
    ProtocolSpec,
    # KL regularization + algorithm building
    CompositeLoss,
    LossTerm,
    ClippedSurrogateLoss,
    KLLoss,
    RLAlgorithm,
    GroupNormalizedReturn,
)
from ludic.training import Reducer, RichLiveLogger
from ludic.training.loggers import WandbLogger

# Import CodeExecEnv components
from ludic.envs.code_exec import (
    CodeExecEnv,
    CodeExecConfig,
    create_sandbox_pool,
    SandboxBackend,
)
from ludic.envs.code_exec.adapters.apps import APPSTestAdapter, APPS_SYSTEM_PROMPT

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def code_block_parser(raw: str) -> ParseResult:
    """
    Parse code from markdown code blocks or raw text.

    Accepts:
      - ```python\n...\n```
      - ```\n...\n```
      - Raw code (if no code blocks found)

    Returns parsed code with small format reward for proper code blocks.
    """
    import re

    # Try to extract from markdown code block
    code_block_pattern = r"```(?:python)?\s*\n(.*?)(?:\n)?```"
    match = re.search(code_block_pattern, raw, re.DOTALL)

    if match:
        code = match.group(1).strip()
        return ParseResult(
            action=code, reward=0.05, obs=None
        )  # Small bonus for proper formatting

    # Fall back to raw text (strip leading/trailing whitespace)
    code = raw.strip()
    if code:
        return ParseResult(action=code, reward=0.0, obs=None)

    # Empty response
    return ParseResult(action=None, reward=-0.1, obs="Please provide Python code.")


def load_apps_samples(
    split: str = "train",
    limit: int | None = None,
    difficulty: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Load APPS samples from HuggingFace datasets.

    Args:
        split: Dataset split ("train" or "test")
        limit: Maximum number of samples to load
        difficulty: Filter by difficulty ("introductory", "interview", "competition")

    Returns:
        List of sample dicts with question, inputs, outputs, etc.
    """
    # Load from the control-arena version which has cleaner formatting
    ds = load_dataset("RoganInglis/apps-control-arena", split=split)

    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        # Filter by difficulty if specified
        if difficulty and row.get("difficulty") != difficulty:
            continue

        # Skip nondeterministic problems (they require special handling)
        if row.get("is_nondeterministic", False):
            continue

        samples.append(
            {
                "problem_id": row.get("problem_id", str(idx)),
                "question": row["question"],
                "inputs": row.get("inputs", []),
                "outputs": row.get("outputs", []),
                "difficulty": row.get("difficulty", "unknown"),
            }
        )

        if limit is not None and len(samples) >= limit:
            break

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Train on APPS code generation dataset with LoRA"
    )

    # Model and inference
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--host", default="127.0.0.1", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")

    # LoRA configuration
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--lora-alpha-mult", type=float, default=2.0, help="LoRA alpha = rank * mult"
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")

    # KL regularization
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.0,
        help="KL penalty coefficient (0 = disabled)",
    )

    # Data
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to load")
    parser.add_argument("--difficulty", default=None, help="Filter by difficulty")

    # Sandbox
    parser.add_argument(
        "--sandbox-workers", type=int, default=4, help="Number of sandbox containers"
    )
    parser.add_argument(
        "--sandbox-backend",
        default="auto",
        choices=["auto", "docker", "podman-hpc"],
        help="Sandbox backend (default: auto-detect)",
    )
    parser.add_argument(
        "--python-version", default="3.11", help="Python version in sandbox"
    )
    parser.add_argument(
        "--minimal-sandbox",
        action="store_true",
        help="Use minimal sandbox config (no memory/network limits) for HPC compatibility",
    )
    parser.add_argument(
        "--timeout-per-test", type=float, default=5.0, help="Timeout per test (seconds)"
    )

    # Training
    parser.add_argument(
        "--concurrency", type=int, default=32, help="Rollout concurrency"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Rollout requests per batch"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=100,
        help="Training steps (0=run until exhausted)",
    )
    parser.add_argument("--group-size", type=int, default=8, help="GRPO group size")
    parser.add_argument(
        "--train-temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--partial-credit", action="store_true", help="Enable partial credit rewards"
    )

    # Evaluation
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=200,
        help="Number of samples to hold out for eval",
    )
    parser.add_argument(
        "--eval-every", type=int, default=25, help="Eval every N training steps"
    )
    parser.add_argument(
        "--eval-before-start",
        action="store_true",
        default=True,
        help="Run baseline eval",
    )
    parser.add_argument(
        "--eval-concurrency", type=int, default=32, help="Eval concurrency"
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.0,
        help="Eval sampling temperature (greedy)",
    )

    # Logging
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="ludic-apps", help="WandB project name"
    )

    # Checkpoints
    parser.add_argument("--rollout-log", default="apps_train_rollouts.jsonl")
    parser.add_argument("--checkpoint-dir", default="checkpoints_apps")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--final-save", action="store_true", help="Save final checkpoint after training"
    )

    args = parser.parse_args()

    # Warn about concurrency/pool mismatch
    if args.concurrency > args.sandbox_workers:
        print(
            f"WARNING: concurrency ({args.concurrency}) > sandbox-workers ({args.sandbox_workers})"
        )
        print(
            f"  This means {args.concurrency - args.sandbox_workers} tasks will wait for sandboxes."
        )
        print(
            f"  Consider: --sandbox-workers={args.concurrency} OR --concurrency={args.sandbox_workers}"
        )
        print()

    # Setup rollout log
    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    open(rollout_log_path, "a", encoding="utf-8").close()

    # Load data and split into train/eval sets
    print(f"Loading APPS samples (split={args.split}, limit={args.limit})...")
    all_samples = load_apps_samples(args.split, args.limit, args.difficulty)
    if not all_samples:
        print("ERROR: No APPS samples loaded.")
        return 1

    # Split: last N samples for eval (deterministic, reproducible)
    if args.eval_samples > 0 and len(all_samples) > args.eval_samples:
        train_samples = all_samples[: -args.eval_samples]
        eval_samples = all_samples[-args.eval_samples :]
    else:
        train_samples = all_samples
        eval_samples = []

    print(f"Loaded {len(all_samples)} total samples.")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Eval:  {len(eval_samples)} samples (held out)")

    samples_q: queue.Queue = queue.Queue()
    for idx, s in enumerate(train_samples):
        samples_q.put((idx, s))

    # Tokenizer + model with LoRA
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Apply LoRA adapter
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=int(args.lora_rank * args.lora_alpha_mult),
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.print_trainable_parameters()
    print(
        f"Model loaded on {device} with LoRA (rank={args.lora_rank}, alpha={int(args.lora_rank * args.lora_alpha_mult)})."
    )

    # Setup sandbox pool
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Build backend kwargs (minimal mode skips memory/network limits for HPC compatibility)
    backend_kwargs = {}
    if args.minimal_sandbox:
        backend_kwargs["memory_limit"] = None
        backend_kwargs["network_disabled"] = False

    try:
        sandbox_pool = loop.run_until_complete(
            create_sandbox_pool(
                n_workers=args.sandbox_workers,
                backend=args.sandbox_backend,
                python_version=args.python_version,
                cache_size=10000,
                **backend_kwargs,
            )
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # Create shared adapter and config
    test_adapter = APPSTestAdapter()
    env_config = CodeExecConfig(
        timeout_per_test_s=args.timeout_per_test,
        stop_on_first_failure=False,
        compile_first=True,
        partial_credit=args.partial_credit,
        compile_failure_reward=-0.1,
        use_cache=True,
    )

    # Shared client for inference
    client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=True)
    publisher = create_vllm_publisher(client)

    # Environment factory (captures sandbox_pool via closure)
    def env_factory(sample: Dict[str, Any]) -> CodeExecEnv:
        return CodeExecEnv(
            sample=sample,
            sandbox_pool=sandbox_pool,
            test_adapter=test_adapter,
            config=env_config,
            system_prompt=APPS_SYSTEM_PROMPT,
        )

    env_registry = {"apps": env_factory}

    def protocol_factory():
        return SingleAgentProtocol(
            agent=Agent(
                client=client,
                model=args.model,
                ctx=FullDialog(),
                parser=code_block_parser,
            )
        )

    protocol_registry = {"single_agent": protocol_factory}

    # Algorithm (GRPO with optional KL regularization)
    credit_assigner = GroupNormalizedReturn(
        group_size=args.group_size,
        normalize_adv=True,
    )

    # Build loss with optional KL penalty
    if args.kl_coeff > 0:
        # CompositeLoss: PPO + KL penalty
        loss = CompositeLoss(
            terms=[
                LossTerm(
                    name="policy",
                    loss=ClippedSurrogateLoss(clip_eps=0.1, length_normalize=True),
                    weight=1.0,
                ),
                LossTerm(
                    name="kl",
                    loss=KLLoss(coeff=args.kl_coeff),
                    weight=1.0,
                ),
            ]
        )
        print(f"Using GRPO with KL penalty (coeff={args.kl_coeff})")
    else:
        # Standard GRPO (no KL penalty)
        loss = ClippedSurrogateLoss(clip_eps=0.1, length_normalize=True)
        print("Using standard GRPO (no KL penalty)")

    algo = RLAlgorithm(
        name="grpo" if args.kl_coeff == 0 else "grpo_kl",
        credit_assigner=credit_assigner,
        loss=loss,
    )

    # Engine + batch source
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=rollout_log_path,
    )

    train_inference = InferenceSpec(
        sampling=SamplingParams(temperature=args.train_temperature, max_tokens=1024),
        return_=ReturnSpec.for_rl(top_logprobs_k=1),
    )

    requests_fn = make_dataset_queue_requests_fn(
        samples_q,
        batch_size=args.batch_size,
        env_kind="apps",
        protocol_kind="single_agent",
        inference=train_inference,
        protocol_kwargs={},
        request_meta_fn=lambda idx, sample: {
            "sample_index": idx,
            "problem_id": sample.get("problem_id", idx),
            "difficulty": sample.get("difficulty", "unknown"),
        },
        env_seed_fn=lambda idx, _sample: idx,
        sampling_seed_fn=lambda idx, _sample: idx,
        group_size=args.group_size,
    )

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=requests_fn,
        max_steps=1,  # Single-step env
        concurrency=args.concurrency,
    )

    # Trainer config with eval settings
    cfg = TrainerConfig(
        model_device=device,
        lr=1e-5,
        grad_accum_steps=8,
        max_grad_norm=0.5,
        pad_token_id=tokenizer.pad_token_id,
        eval_at_start=bool(args.eval_before_start and eval_samples),
        eval_every_n_steps=(
            args.eval_every
            if args.eval_every and args.eval_every > 0 and eval_samples
            else None
        ),
        eval_concurrency=args.eval_concurrency,
        eval_max_steps=1,
    )

    checkpoint_cfg = CheckpointConfig(
        output_dir=args.checkpoint_dir,
        every_n_steps=args.checkpoint_every,
        max_to_keep=2,
        save_optimizer=True,
    )

    # Training reducers
    reducers = {
        "all_passed_rate": Reducer(
            kind="count_true",
            source="all_passed",
            normalize_by="rollouts",
        ),
        "compile_fail_rate": Reducer(
            kind="count_true",
            source="compile_failed",
            normalize_by="rollouts",
        ),
        "avg_pass_rate": Reducer(
            kind="mean",
            source="pass_rate",
        ),
        "parse_err_rate": Reducer(
            kind="count_true",
            source="parse_error",
            normalize_by="samples",
        ),
        "total_completion_tokens": Reducer(
            kind="sum",
            source="completion_length",
        ),
    }

    # Eval reducers (for held-out samples)
    eval_reducers = {
        "all_passed_rate": Reducer(
            kind="count_true",
            source="all_passed",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "compile_fail_rate": Reducer(
            kind="count_true",
            source="compile_failed",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "avg_pass_rate": Reducer(
            kind="mean",
            source="pass_rate",
        ),
        "parse_error_rate": Reducer(
            kind="count_true",
            source="parse_error",
            normalize_by="samples",
            as_percent=True,
        ),
        "avg_completion_tokens": Reducer(
            kind="mean",
            source="completion_length",
        ),
    }

    # Logging metrics to track
    log_keys = [
        # Core training
        "loss",
        "avg_total_reward",
        # APPS-specific
        "all_passed_rate",
        "compile_fail_rate",
        "avg_pass_rate",
        "parse_err_rate",
        "avg_completion_length",
        # KL stats (if enabled)
        "kl/kl_mean",
        "kl/loss",
        # Eval metrics (auto-prefixed with eval_)
        "eval_all_passed_rate",
        "eval_compile_fail_rate",
        "eval_avg_pass_rate",
        "eval_parse_error_rate",
        "eval_avg_completion_tokens",
        # Counts
        "num_rollouts",
        "num_samples",
    ]

    # Configure logger (WandB or RichLive terminal dashboard)
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            config={
                "model": args.model,
                "lora_rank": args.lora_rank,
                "lora_alpha": int(args.lora_rank * args.lora_alpha_mult),
                "kl_coeff": args.kl_coeff,
                "group_size": args.group_size,
                "train_steps": args.train_steps,
                "eval_samples": len(eval_samples),
                "train_samples": len(train_samples),
            },
        )
        train_logger = WandbLogger(run=run)
        print(f"WandB logging enabled: project={args.wandb_project}")
    else:
        train_logger = RichLiveLogger(
            keys=log_keys,
            spark_key="avg_total_reward",
            history=100,
            precision=4,
        )

    # Create EngineEvaluator for eval set
    eval_inference = InferenceSpec(
        sampling=SamplingParams(temperature=args.eval_temperature, max_tokens=1024),
        return_=ReturnSpec.for_eval(return_token_ids=True),
    )

    evaluator = None
    if eval_samples:
        evaluator = EngineEvaluator(
            engine=RolloutEngine(
                env_registry=env_registry, protocol_registry=protocol_registry
            ),
            requests_fn=lambda: [
                RolloutRequest(
                    env=EnvSpec(kind="apps", kwargs={"sample": sample}),
                    protocol=ProtocolSpec(kind="single_agent"),
                    env_seed=idx,
                    sampling_seed=idx,
                    inference=eval_inference,
                    num_episodes=1,
                    meta={
                        "eval_idx": idx,
                        "problem_id": sample.get("problem_id", idx),
                        "difficulty": sample.get("difficulty", "unknown"),
                    },
                )
                for idx, sample in enumerate(eval_samples)
            ],
            reducers=eval_reducers,
            max_steps=1,
            timeout_s=cfg.eval_timeout_s,
            concurrency=cfg.eval_concurrency,
        )
        print(
            f"Eval configured: {len(eval_samples)} samples, every {args.eval_every} steps"
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
        evaluator=evaluator,
    )

    print(f"\nStarting training for {args.train_steps} steps...")
    print(f"  Samples: {len(train_samples)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Group size: {args.group_size}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Sandbox workers: {args.sandbox_workers}")
    print(f"  Sandbox backend: {args.sandbox_backend}")
    print(f"  Progress will be logged every training step.")
    print()

    try:
        loop.run_until_complete(trainer.train(args.train_steps))
    except RequestsExhausted:
        print("Training samples exhausted; stopping.")
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        # Cleanup sandbox pool
        print("Shutting down sandbox pool...")
        loop.run_until_complete(sandbox_pool.shutdown())
        loop.close()

    # Save final checkpoint if requested
    if args.final_save:
        try:
            ckpt_path = trainer.save_checkpoint(metadata={"final": True})
            print(f"Final checkpoint saved: {ckpt_path}")
        except RuntimeError:
            pass  # No checkpointer configured

    # Close WandB if used
    if args.wandb:
        import wandb

        wandb.finish()
        print("WandB run finished.")

    print("Training complete.")
    return 0


if __name__ == "__main__":
    exit(main())

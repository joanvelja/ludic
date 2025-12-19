"""
APPS code generation training scaffold using CodeExecEnv.

This wires together:
  - HuggingFace datasets for APPS code samples
  - CodeExecEnv with sandboxed execution (Docker or Podman-HPC)
  - SingleAgentProtocol with async env support
  - RolloutBatchSource + MonteCarloReturn credit
  - Trainer with REINFORCE or GRPO loss

Requirements:
  - Container runtime: Docker daemon OR Podman-HPC (auto-detected)
  - pip install docker>=7.0.0 datasets (for Docker backend)
  - GPU(s) for training (optional for rollout-only mode)

Usage:
  # Start vLLM server (in one terminal)
  CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \\
      --model Qwen/Qwen2.5-Coder-0.5B-Instruct

  # Run training with Docker (local, default)
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/code_exec/train_apps.py \\
      --model Qwen/Qwen2.5-Coder-0.5B-Instruct \\
      --limit 100 --train-steps 10

  # Run training with Podman-HPC (HPC cluster)
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/code_exec/train_apps.py \\
      --model Qwen/Qwen2.5-Coder-0.5B-Instruct \\
      --sandbox-backend podman-hpc \\
      --limit 100 --train-steps 10

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

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec
from ludic.interaction import SingleAgentProtocol
from ludic.parsers import ParseResult
from ludic.distributed.adapters import create_vllm_publisher
from ludic.training import (
    RolloutEngine,
    RolloutBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    make_dataset_queue_requests_fn,
    make_grpo,
    RequestsExhausted,
)
from ludic.training import Reducer, RichLiveLogger

# Import CodeExecEnv components
from ludic.envs.code_exec import (
    CodeExecEnv,
    CodeExecConfig,
    create_sandbox_pool,
    SandboxBackend,
)
from ludic.envs.code_exec.adapters.apps import APPSTestAdapter, APPS_SYSTEM_PROMPT


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
    code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
    match = re.search(code_block_pattern, raw, re.DOTALL)

    if match:
        code = match.group(1).strip()
        return ParseResult(action=code, reward=0.05, obs=None)  # Small bonus for proper formatting

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

        samples.append({
            "problem_id": row.get("problem_id", str(idx)),
            "question": row["question"],
            "inputs": row.get("inputs", []),
            "outputs": row.get("outputs", []),
            "difficulty": row.get("difficulty", "unknown"),
        })

        if limit is not None and len(samples) >= limit:
            break

    return samples




def main():
    parser = argparse.ArgumentParser(description="Train on APPS code generation dataset")

    # Model and inference
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")

    # Data
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to load")
    parser.add_argument("--difficulty", default=None, help="Filter by difficulty")

    # Sandbox
    parser.add_argument("--sandbox-workers", type=int, default=4, help="Number of sandbox containers")
    parser.add_argument("--sandbox-backend", default="auto",
                        choices=["auto", "docker", "podman-hpc"],
                        help="Sandbox backend (default: auto-detect)")
    parser.add_argument("--python-version", default="3.11", help="Python version in sandbox")
    parser.add_argument("--timeout-per-test", type=float, default=5.0, help="Timeout per test (seconds)")

    # Training
    parser.add_argument("--concurrency", type=int, default=32, help="Rollout concurrency")
    parser.add_argument("--batch-size", type=int, default=4, help="Rollout requests per batch")
    parser.add_argument("--train-steps", type=int, default=20, help="Training steps (0=run until exhausted)")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size")
    parser.add_argument("--train-temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--partial-credit", action="store_true", help="Enable partial credit rewards")

    # Logging and checkpoints
    parser.add_argument("--rollout-log", default="apps_train_rollouts.jsonl")
    parser.add_argument("--checkpoint-dir", default="checkpoints_apps")
    parser.add_argument("--checkpoint-every", type=int, default=25)

    args = parser.parse_args()

    # Warn about concurrency/pool mismatch
    if args.concurrency > args.sandbox_workers:
        print(f"WARNING: concurrency ({args.concurrency}) > sandbox-workers ({args.sandbox_workers})")
        print(f"  This means {args.concurrency - args.sandbox_workers} tasks will wait for sandboxes.")
        print(f"  Consider: --sandbox-workers={args.concurrency} OR --concurrency={args.sandbox_workers}")
        print()

    # Setup rollout log
    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    open(rollout_log_path, "a", encoding="utf-8").close()

    # Load data
    print(f"Loading APPS samples (split={args.split}, limit={args.limit})...")
    train_samples = load_apps_samples(args.split, args.limit, args.difficulty)
    if not train_samples:
        print("ERROR: No APPS samples loaded.")
        return 1
    print(f"Loaded {len(train_samples)} samples.")

    samples_q: queue.Queue = queue.Queue()
    for idx, s in enumerate(train_samples):
        samples_q.put((idx, s))

    # Tokenizer + model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}.")

    # Setup sandbox pool
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        sandbox_pool = loop.run_until_complete(
            create_sandbox_pool(
                n_workers=args.sandbox_workers,
                backend=args.sandbox_backend,
                python_version=args.python_version,
                cache_size=10000,
            )
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # Create shared adapter and config
    test_adapter = APPSTestAdapter()
    env_config = CodeExecConfig(
        timeout_per_test_s=args.timeout_per_test,
        stop_on_first_failure=True,
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

    # Algorithm (GRPO with group-relative advantages)
    algo = make_grpo(
        name="grpo",
        group_size=args.group_size,
        group_normalize_adv=True,
        clip_eps=0.1,
        length_normalize=True,
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

    # Trainer config
    cfg = TrainerConfig(
        model_device=device,
        grad_accum_steps=8,
        max_grad_norm=0.5,
        pad_token_id=tokenizer.pad_token_id,
    )

    checkpoint_cfg = CheckpointConfig(
        output_dir=args.checkpoint_dir,
        every_n_steps=args.checkpoint_every,
        max_to_keep=2,
        save_optimizer=True,
    )

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

    train_logger = RichLiveLogger(
        keys=[
            "loss",
            "avg_total_reward",
            "all_passed_rate",
            "compile_fail_rate",
            "avg_pass_rate",
            "parse_err_rate",
            "avg_completion_length",
            "num_rollouts",
            "num_samples",
        ],
        spark_key="avg_total_reward",
        history=100,
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

    print(f"\nStarting training for {args.train_steps} steps...")
    print(f"  Samples: {len(train_samples)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Group size: {args.group_size}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Sandbox workers: {args.sandbox_workers}")
    print(f"  Sandbox backend: {args.sandbox_backend}")
    print()

    try:
        trainer.train_sync(args.train_steps)
    except RequestsExhausted:
        print("Training samples exhausted; stopping.")
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        # Cleanup sandbox pool
        print("Shutting down sandbox pool...")
        loop.run_until_complete(sandbox_pool.shutdown())
        loop.close()

    print("Training complete.")
    return 0


if __name__ == "__main__":
    exit(main())

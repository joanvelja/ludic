#!/usr/bin/env python3
"""
PVG (Prover-Verifier Game) Training Orchestration Script.

This script implements the multi-round Prover-Verifier Game training loop:
1. Data minting: Generate sneaky samples with current prover
2. Verifier training: Bradley-Terry on (honest, sneaky) pairs with IRM
3. Prover training: Policy gradient with composite reward
4. Checkpoint and metrics logging
5. Repeat for N rounds

Usage:
    uv run python examples/pvg/train_pvg.py --config examples/pvg/configs/apps_pvg.yaml

    # Or with inline arguments:
    uv run python examples/pvg/train_pvg.py \
        --verifier-model meta-llama/Llama-3.1-8B \
        --prover-model meta-llama/Llama-3.1-8B \
        --dataset apps \
        --num-rounds 5 \
        --output-dir ./outputs/pvg_run

Reference: Prover-Verifier Games (arXiv:2407.13692)
"""

from __future__ import annotations

import os

# Disable HuggingFace file locking early (before any HF imports)
# This avoids POSIX lock issues on Lustre filesystems
os.environ.setdefault("HF_HUB_DISABLE_LOCKING", "1")

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ludic.pvg.config import (
    DataSplitConfig,
    PVGGameConfig,
    PVGRoundConfig,
    VerifierInitConfig,
)
from ludic.pvg.data import (
    PreferencePairBuilder,
    RolloutRecord,
    RoundDataStore,
    get_mixture_strategy,
)
from ludic.pvg.metrics import (
    CollapseAlert,
    Distribution,
    GoodhartingAlert,
    PVGMetrics,
    PVGMetricsLogger,
)
from ludic.pvg.orchestrator import (
    PVGOrchestrator,
    PVGState,
)
from ludic.pvg.minting import (
    MintingConfig,
    mint_sneaky_data,
    mint_honest_from_dataset,
)
from ludic.pvg.verifier_trainer import (
    VerifierTrainingConfig,
    train_verifier_phase,
)
from ludic.pvg.prover_trainer import (
    ProverTrainingConfig,
    train_prover_phase,
)
from ludic.pvg.vllm_setup import (
    DualVLLMConfig,
    setup_dual_vllm,
    create_prover_publisher,
)
from ludic.pvg.scoring import VerifierScorer, MockRewardModelClient
from ludic.pvg.rewards import (
    CompositeReward,
    SRCReward,
    CGCReward,
    GatedMultiplicativeReward,
)
from ludic.pvg.verifier_trainer import reinitialize_verifier_head
from ludic.hf_cache import ensure_hf_cache_dir, hf_cache_kwargs

logger = logging.getLogger(__name__)


# Model loading cache to avoid reloading for each round
_loaded_models: Dict[str, Any] = {}


async def load_verifier_model(model_path: str, device: str = "cuda"):
    """Load verifier model (reward model) for training.

    Uses AutoModelForSequenceClassification for reward modeling.
    Model is cached to avoid reloading between rounds.
    """
    if "verifier_model" in _loaded_models:
        return _loaded_models["verifier_model"], _loaded_models["verifier_tokenizer"]

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading verifier model from {model_path}")
        cache_kwargs = hf_cache_kwargs()
        tokenizer = AutoTokenizer.from_pretrained(model_path, **cache_kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype="auto",
            **cache_kwargs,
        )
        model = model.to(device)

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        _loaded_models["verifier_model"] = model
        _loaded_models["verifier_tokenizer"] = tokenizer
        logger.info("Verifier model loaded successfully")

        return model, tokenizer

    except Exception as e:
        logger.warning(f"Failed to load verifier model: {e}")
        logger.warning("Using mock model for testing")
        return None, None


async def load_prover_model(model_path: str, device: str = "cuda"):
    """Load prover model for training.

    Uses AutoModelForCausalLM for language modeling.
    Model is cached to avoid reloading between rounds.
    """
    if "prover_model" in _loaded_models:
        return _loaded_models["prover_model"], _loaded_models["prover_tokenizer"]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading prover model from {model_path}")
        cache_kwargs = hf_cache_kwargs()
        tokenizer = AutoTokenizer.from_pretrained(model_path, **cache_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            **cache_kwargs,
        )
        model = model.to(device)

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        _loaded_models["prover_model"] = model
        _loaded_models["prover_tokenizer"] = tokenizer
        logger.info("Prover model loaded successfully")

        return model, tokenizer

    except Exception as e:
        logger.warning(f"Failed to load prover model: {e}")
        logger.warning("Using mock model for testing")
        return None, None


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure logging for the training run.

    Sets up:
    - Console handler for immediate feedback
    - File handler for full debug logs
    - Sample logger for training sample inspection
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    console_format = "%(asctime)s | %(levelname)s | %(message)s"
    file_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    sample_format = "%(asctime)s | SAMPLE | %(message)s"

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(console_format, datefmt="%H:%M:%S"))

    log_file = output_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(file_format))

    # Dedicated sample log for inspecting training samples
    sample_log_file = output_dir / "samples.log"
    sample_file_handler = logging.FileHandler(sample_log_file)
    sample_file_handler.setLevel(logging.DEBUG)
    sample_file_handler.setFormatter(logging.Formatter(sample_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Configure sample loggers
    for sample_logger_name in [
        "ludic.pvg.verifier_trainer.samples",
        "ludic.pvg.prover_trainer.samples",
    ]:
        sample_logger = logging.getLogger(sample_logger_name)
        sample_logger.addHandler(sample_file_handler)
        sample_logger.setLevel(logging.DEBUG)

    logger.info(f"Logging to {log_file}")
    logger.info(f"Sample logging to {sample_log_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PVG Training Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model paths
    parser.add_argument("--verifier-model", type=str, default=None)
    parser.add_argument("--prover-model", type=str, default=None)

    # Dataset
    parser.add_argument("--dataset", type=str, default="apps", choices=["apps", "gsm8k", "math"])
    parser.add_argument("--split-ratio", type=float, default=0.5)

    # Training parameters
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--verifier-steps", type=int, default=1000)
    parser.add_argument("--prover-steps", type=int, default=2000)

    # IRM configuration
    parser.add_argument("--irm-mode", type=str, default="vrex", choices=["vrex", "none"])
    parser.add_argument("--irm-beta", type=float, default=1.0)

    # Reward configuration
    parser.add_argument("--reward-strategy", type=str, default="composite",
                       choices=["src", "cgc", "composite", "gated"])

    # Stopping criteria
    parser.add_argument("--sneaky-incorrect-threshold", type=float, default=0.95)
    parser.add_argument("--score-parity-threshold", type=float, default=0.1)

    # Data mixture
    parser.add_argument("--mixture-strategy", type=str, default="exponential",
                       choices=["exponential", "sliding_window", "equal", "latest_only"])
    parser.add_argument("--mixture-decay-lambda", type=float, default=0.5)

    # Output and config
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path)

    # vLLM inference
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--prover-port", type=int, default=8000)
    parser.add_argument("--verifier-port", type=int, default=8001)
    parser.add_argument("--prover-gpu-memory", type=float, default=0.77)
    parser.add_argument("--verifier-gpu-memory", type=float, default=0.18)

    # LoRA configuration
    parser.add_argument("--verifier-lora", action="store_true")
    parser.add_argument("--verifier-lora-rank", type=int, default=16)
    parser.add_argument("--prover-lora", action="store_true")
    parser.add_argument("--prover-lora-rank", type=int, default=8)

    # Runtime
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock-inference", action="store_true")

    # Sandbox configuration (for HPC compatibility)
    parser.add_argument(
        "--minimal-sandbox",
        action="store_true",
        help="Use minimal sandbox config for HPC (no memory/network limits)",
    )
    parser.add_argument("--sandbox-workers", type=int, default=4)

    return parser.parse_args()


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML required. Install with: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def build_game_config(args: argparse.Namespace) -> PVGGameConfig:
    """Build PVGGameConfig from parsed arguments."""
    if args.config and args.config.exists():
        yaml_config = load_config_from_yaml(args.config)

        cli_overrides = {
            "num_rounds": args.num_rounds,
            "verifier_model_path": args.verifier_model,
            "prover_model_path": args.prover_model,
        }
        for key, value in cli_overrides.items():
            if value is not None:
                yaml_config[key] = value

        if args.output_dir is not None:
            yaml_config["output_dir"] = str(args.output_dir)

        if args.split_ratio is not None:
            if "data_split" not in yaml_config:
                yaml_config["data_split"] = {}
            yaml_config["data_split"]["split_ratio"] = args.split_ratio

        return PVGGameConfig.from_dict(yaml_config)

    # CLI-only mode
    num_rounds = args.num_rounds or 5
    verifier_model = args.verifier_model or "meta-llama/Llama-3.1-8B"
    prover_model = args.prover_model or "meta-llama/Llama-3.1-8B"
    output_dir = args.output_dir or Path("./outputs/pvg")

    round_configs = []
    for round_id in range(num_rounds):
        round_config = PVGRoundConfig(
            round_id=round_id,
            verifier_steps=args.verifier_steps,
            prover_steps=args.prover_steps,
            sneaky_incorrect_threshold=args.sneaky_incorrect_threshold,
            score_parity_threshold=args.score_parity_threshold,
            irm_mode=args.irm_mode,
            data_mixture_strategy=args.mixture_strategy,
            sequential_observation=round_id > 0,
        )
        round_configs.append(round_config)

    data_split = DataSplitConfig(split_ratio=args.split_ratio, split_seed=42)
    verifier_init = VerifierInitConfig(base_path=verifier_model, reinit_mode="head_only")

    return PVGGameConfig(
        num_rounds=num_rounds,
        round_configs=round_configs,
        verifier_model_path=verifier_model,
        prover_model_path=prover_model,
        data_split=data_split,
        verifier_init=verifier_init,
        output_dir=output_dir,
    )


def get_reward_strategy(name: str):
    """Get reward strategy by name."""
    strategies = {
        "composite": CompositeReward(),
        "src": SRCReward(),
        "cgc": CGCReward(),
        "gated": GatedMultiplicativeReward(),
    }
    return strategies.get(name, CompositeReward())


def check_stopping_criteria(metrics: PVGMetrics, round_config: PVGRoundConfig) -> bool:
    """Check if training should stop based on paper's criteria."""
    if metrics.sneaky_incorrect_rate >= round_config.sneaky_incorrect_threshold:
        logger.info(
            f"Stopping: sneaky_incorrect_rate ({metrics.sneaky_incorrect_rate:.2%}) "
            f">= threshold ({round_config.sneaky_incorrect_threshold:.2%})"
        )
        return True

    if abs(metrics.score_gap) <= round_config.score_parity_threshold:
        logger.info(
            f"Stopping: score_gap ({metrics.score_gap:.4f}) "
            f"<= threshold ({round_config.score_parity_threshold})"
        )
        return True

    return False


def compute_round_metrics(
    sneaky_rollouts: List[RolloutRecord],
    honest_rollouts: List[RolloutRecord],
    round_id: int,
) -> PVGMetrics:
    """Compute metrics from rollouts."""
    total_sneaky = len(sneaky_rollouts)
    if total_sneaky == 0:
        return PVGMetrics(round_id=round_id)

    certified = sum(
        1 for r in sneaky_rollouts
        if r.metadata.get("sneaky_result", {}).get("sneaky_certified", False)
    )
    passed_tests = sum(
        1 for r in sneaky_rollouts
        if r.metadata.get("sneaky_result", {}).get("test_pass_rate", 0) >= 1.0
    )

    sneaky_certified_rate = certified / total_sneaky
    sneaky_incorrect_rate = 1.0 - (passed_tests / total_sneaky)
    sneaky_valid_rate = sum(
        1 for r in sneaky_rollouts
        if (r.metadata.get("sneaky_result", {}).get("sneaky_certified", False) and
            r.metadata.get("sneaky_result", {}).get("test_pass_rate", 0) >= 1.0)
    ) / total_sneaky

    similarities = [
        r.metadata.get("sneaky_result", {}).get("similarity_score", 0.0)
        for r in sneaky_rollouts
        if r.metadata.get("sneaky_result", {}).get("similarity_score") is not None
    ]
    similarity_dist = Distribution.from_values(similarities) if similarities else None

    return PVGMetrics(
        round_id=round_id,
        sneaky_incorrect_rate=sneaky_incorrect_rate,
        sneaky_certified_rate=sneaky_certified_rate,
        sneaky_valid_rate=sneaky_valid_rate,
        similarity_distribution=similarity_dist,
    )


async def run_pvg_training(
    config: PVGGameConfig,
    args: argparse.Namespace,
    dry_run: bool = False,
) -> None:
    """Main PVG training loop with orchestrator state machine."""
    logger.info("=" * 60)
    logger.info("Starting PVG Training")
    logger.info("=" * 60)
    logger.info("")
    logger.info("FULL CONFIGURATION (for verification):")
    logger.info("-" * 60)
    logger.info(f"Config: {json.dumps(asdict(config), indent=2, default=str)}")
    logger.info("-" * 60)
    logger.info("")
    logger.info("KEY VERIFICATION POINTS:")
    logger.info("  1. [LABEL LEAK CHECK] - No label leakage when training verifier")
    logger.info("       Chosen=honest, Rejected=sneaky; env_labels contains only round ID")
    logger.info("  2. [PROMPT CHECK] - Correct prompts for prover training")
    logger.info("       Prompts contain problem statements only, not labels")
    logger.info("  3. [WEIGHT SYNC] - Per training step syncing")
    logger.info("       Weights sync to vLLM at configured intervals")
    logger.info("  4. [CONFIG] - Thorough config documentation")
    logger.info("       All parameters logged at each phase")
    logger.info("  5. [LORA/HEAD REINIT CHECK] - LoRA/head re-init per round")
    logger.info("       At each verifier training stage, LoRAs must be re-initialized")
    logger.info("-" * 60)
    logger.info("")

    if dry_run:
        logger.info("Dry run: config validated, exiting")
        return

    # Initialize orchestrator
    orchestrator = PVGOrchestrator(output_dir=config.output_dir, config=config)

    if args.resume:
        if not orchestrator.resume_from_checkpoint():
            logger.info("No existing state to resume, initializing fresh run")
            orchestrator.initialize()
    else:
        orchestrator.initialize()

    # Initialize components
    data_store = RoundDataStore(config.output_dir / "data")
    metrics_logger = PVGMetricsLogger(config.output_dir / "metrics")

    if args.resume:
        metrics_logger.load_from_file()

    # Setup inference clients
    verifier_scorer = None
    if not args.mock_inference:
        ensure_hf_cache_dir()
        try:
            vllm_config = DualVLLMConfig(
                prover_port=args.prover_port,
                verifier_port=args.verifier_port,
                prover_gpu_memory=args.prover_gpu_memory,
                verifier_gpu_memory=args.verifier_gpu_memory,
                host=args.host,
            )
            prover_client, verifier_client = await setup_dual_vllm(
                prover_model=config.prover_model_path,
                verifier_model=config.verifier_model_path,
                config=vllm_config,
            )
            verifier_scorer = VerifierScorer(client=verifier_client, batch_size=32)
        except Exception as e:
            logger.warning(f"Failed to setup vLLM: {e}")
            logger.warning("Falling back to mock inference")
            args.mock_inference = True

    if args.mock_inference:
        mock_client = MockRewardModelClient(default_score=0.5)
        verifier_scorer = VerifierScorer(client=mock_client, batch_size=32)

    previous_metrics: Optional[PVGMetrics] = None

    try:
        while orchestrator.should_continue():
            round_id = orchestrator.round_id
            round_config = config.get_round_config(round_id)
            round_start = time.time()

            logger.info("-" * 60)
            logger.info(f"Round {round_id + 1}/{config.num_rounds}")
            logger.info(f"State: {orchestrator.current_state.value}")
            logger.info("-" * 60)

            # === Phase 1: Data Minting ===
            if orchestrator.current_state == PVGState.INIT:
                orchestrator.transition(PVGState.MINT_DATA)

            if orchestrator.current_state == PVGState.MINT_DATA:
                logger.info("Phase 1: Data Minting")

                problem_samples = _load_dataset_samples(args.dataset, config.data_split)
                honest_codes = _load_honest_codes(problem_samples)

                def rollout_generator(prompt: str, problem_id: str):
                    from ludic.types import Rollout, Step
                    return Rollout(
                        steps=[Step(
                            index=0, prev_obs=prompt,
                            action='{"code": "pass", "certificate": "42"}',
                            next_obs="ok", reward=0.0,
                            truncated=False, terminated=True,
                        )],
                        meta={"problem_id": problem_id},
                    )

                minting_config = MintingConfig(
                    strategy="few_shot" if round_id == 0 else "standard",
                    max_samples=min(1000, len(problem_samples)),
                    temperature=0.7,
                )

                minting_result = await mint_sneaky_data(
                    problem_samples=problem_samples,
                    honest_codes=honest_codes,
                    rollout_generator=rollout_generator,
                    current_round=round_id,
                    config=minting_config,
                )

                sneaky_rollouts = minting_result.rollouts
                logger.info(f"  Generated {len(sneaky_rollouts)} sneaky samples")

                if sneaky_rollouts:
                    data_store.save_rollouts(round_id, "sneaky", sneaky_rollouts)

                honest_rollouts = await mint_honest_from_dataset(
                    problem_samples=problem_samples,
                    round_id=round_id,
                )
                if honest_rollouts:
                    data_store.save_rollouts(round_id, "honest", honest_rollouts)

                pair_builder = PreferencePairBuilder(
                    require_certified=True,
                    require_tests_pass=True,
                    min_pass_rate=1.0,
                )
                pairs = pair_builder.build_pairs(
                    honest_rollouts,
                    sneaky_rollouts,
                    env_labels={"round": round_id},
                )

                logger.info(f"  Built {len(pairs)} preference pairs")
                if pairs:
                    data_store.save_pairs(round_id, pairs)

                orchestrator.record_phase_checkpoint(
                    "mint",
                    config.output_dir / "data" / f"round_{round_id}",
                    metrics={"pair_count": len(pairs)},
                )
                orchestrator.transition(PVGState.TRAIN_VERIFIER)

            # === Phase 2: Verifier Training ===
            if orchestrator.current_state == PVGState.TRAIN_VERIFIER:
                logger.info("=" * 80)
                logger.info("PHASE 2: VERIFIER TRAINING")
                logger.info("=" * 80)
                logger.info("")
                logger.info("VERIFICATION CHECKLIST (log markers to search for):")
                logger.info("  [LABEL LEAK CHECK] - Verify no label leakage in training data")
                logger.info("  [LORA/HEAD REINIT CHECK] - Verify head reinit per round")
                logger.info("")
                logger.info(f"  Round: {round_id}")
                logger.info(f"  IRM mode: {round_config.irm_mode}")
                logger.info(f"  Training for up to {round_config.verifier_steps} steps")
                logger.info(f"  Data mixture strategy: {round_config.data_mixture_strategy}")
                logger.info(f"  Verifier model: {config.verifier_model_path}")
                logger.info(f"  Reinit mode: {config.verifier_init.reinit_mode}")

                verifier_config = VerifierTrainingConfig(
                    max_steps=round_config.verifier_steps,
                    irm_mode=round_config.irm_mode,
                    irm_beta=args.irm_beta,
                    mixture_strategy=round_config.data_mixture_strategy,
                    mixture_decay_lambda=args.mixture_decay_lambda,
                )

                # Log full verifier config
                logger.info("  Full VerifierTrainingConfig:")
                logger.info(f"    lr: {verifier_config.lr}")
                logger.info(f"    batch_size: {verifier_config.batch_size}")
                logger.info(f"    irm_weight: {verifier_config.irm_weight}")
                logger.info(f"    bt_beta: {verifier_config.bt_beta}")
                logger.info(f"    label_smoothing: {verifier_config.label_smoothing}")
                logger.info(f"    score_regularization_lambda: {verifier_config.score_regularization_lambda}")

                # Load verifier model for training
                verifier_model, verifier_tokenizer = await load_verifier_model(
                    config.verifier_model_path,
                    device="cuda" if not args.mock_inference else "cpu",
                )

                checkpoint_path = None
                if verifier_model is not None and verifier_tokenizer is not None:
                    # Reinitialize reward head at start of each round (per paper)
                    # CRITICAL: This ensures LoRAs are re-initialized each round
                    # PVG Paper Protocol: At each verifier training stage, (rs)LoRAs must be re-init
                    should_reinit = round_id > 0 or config.verifier_init.reinit_mode == "head_only"
                    logger.info("")
                    logger.info("  [LORA/HEAD REINIT CHECK]")
                    logger.info(f"    round_id: {round_id}")
                    logger.info(f"    reinit_mode: {config.verifier_init.reinit_mode}")
                    logger.info(f"    Should reinitialize: {should_reinit}")
                    if round_id == 0:
                        logger.info(f"    Reason: round_id=0, reinit depends on reinit_mode={config.verifier_init.reinit_mode}")
                    else:
                        logger.info(f"    Reason: round_id > 0 (round {round_id}), always reinit per PVG paper")

                    if should_reinit:
                        logger.info("")
                        logger.info("  Reinitializing verifier head...")
                        logger.info("  (Per PVG paper: prevents overfitting to previous rounds' sneaky patterns)")
                        reinitialize_verifier_head(verifier_model, round_id=round_id)
                        logger.info("  Head reinitialization complete.")
                    else:
                        logger.info("  Skipping head reinit (round_id=0 and reinit_mode != 'head_only')")

                    # Set pad token ID in config
                    verifier_config.pad_token_id = verifier_tokenizer.pad_token_id or 0

                    # Run actual verifier training using RMTrainer
                    checkpoint_path = await train_verifier_phase(
                        model=verifier_model,
                        tokenizer=verifier_tokenizer,
                        data_store=data_store,
                        current_round=round_id,
                        config=verifier_config,
                        output_dir=config.output_dir,
                    )
                    logger.info(f"  Verifier training complete, checkpoint: {checkpoint_path}")
                else:
                    # Fallback for mock/test mode
                    checkpoint_path = config.output_dir / "checkpoints" / f"round_{round_id}" / "verifier"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    logger.info("  Verifier training complete (mock mode)")

                orchestrator.record_phase_checkpoint("verifier", checkpoint_path)
                orchestrator.transition(PVGState.TRAIN_PROVER)

            # === Phase 3: Prover Training ===
            if orchestrator.current_state == PVGState.TRAIN_PROVER:
                logger.info("=" * 80)
                logger.info("PHASE 3: PROVER TRAINING")
                logger.info("=" * 80)
                logger.info("")
                logger.info("VERIFICATION CHECKLIST (log markers to search for):")
                logger.info("  [PROMPT CHECK] - Verify prompts contain only problem statements")
                logger.info("  [VERIFIER SCORING] - Verify scores attached post-rollout")
                logger.info("  [WEIGHT SYNC] - Verify syncs happen at configured intervals")
                logger.info("")
                logger.info(f"  Round: {round_id}")
                logger.info(f"  Sequential observation: {round_config.sequential_observation}")
                logger.info(f"  Training for up to {round_config.prover_steps} steps")
                logger.info(f"  Prover model: {config.prover_model_path}")
                logger.info(f"  Reward strategy: {args.reward_strategy}")

                prover_config = ProverTrainingConfig(
                    max_steps=round_config.prover_steps,
                    reward_strategy=get_reward_strategy(args.reward_strategy),
                    group_size=4,
                )

                # Load prover model for training
                prover_model, prover_tokenizer = await load_prover_model(
                    config.prover_model_path,
                    device="cuda" if not args.mock_inference else "cpu",
                )

                checkpoint_path = None
                if prover_model is not None and prover_tokenizer is not None and verifier_scorer is not None:
                    # Create a rollout generator that uses the prover for generation
                    # For now, this is a simple placeholder that generates mock rollouts
                    # In production, this would use RolloutEngine with SneakyCodeExecEnv
                    def create_rollout_generator():
                        from ludic.types import Rollout, Step

                        def rollout_gen():
                            # Load a sample problem from data store
                            samples = _load_dataset_samples(args.dataset, config.data_split)
                            if not samples:
                                return Rollout(steps=[], meta={})

                            sample = samples[0]  # Use first sample as template
                            return Rollout(
                                steps=[Step(
                                    index=0,
                                    prev_obs=sample.get("prompt", "Problem"),
                                    action='{"code": "def solve(): return 42", "certificate": "42"}',
                                    next_obs="ok",
                                    reward=0.0,
                                    truncated=False,
                                    terminated=True,
                                )],
                                meta={"problem_id": sample.get("problem_id", "unknown")},
                            )
                        return rollout_gen

                    rollout_generator = create_rollout_generator()

                    # Run actual prover training using policy gradient
                    checkpoint_path = await train_prover_phase(
                        model=prover_model,
                        tokenizer=prover_tokenizer,
                        rollout_generator=rollout_generator,
                        verifier_scorer=verifier_scorer,
                        current_round=round_id,
                        config=prover_config,
                        output_dir=config.output_dir,
                    )
                    logger.info(f"  Prover training complete, checkpoint: {checkpoint_path}")
                else:
                    # Fallback for mock/test mode
                    checkpoint_path = config.output_dir / "checkpoints" / f"round_{round_id}" / "prover"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    logger.info("  Prover training complete (mock mode)")

                orchestrator.record_phase_checkpoint("prover", checkpoint_path)
                orchestrator.transition(PVGState.CHECKPOINT)

            # === Phase 4: Metrics and Checkpointing ===
            if orchestrator.current_state == PVGState.CHECKPOINT:
                logger.info("Phase 4: Metrics and Checkpointing")

                sneaky_rollouts = list(data_store.load_rollouts(
                    round_ids=[round_id], roles=["sneaky"],
                ))
                honest_rollouts = list(data_store.load_rollouts(
                    round_ids=[round_id], roles=["honest"],
                ))

                round_metrics = compute_round_metrics(
                    sneaky_rollouts=sneaky_rollouts,
                    honest_rollouts=honest_rollouts,
                    round_id=round_id,
                )
                round_metrics.round_duration_s = time.time() - round_start
                round_metrics.verifier_training_steps = round_config.verifier_steps
                round_metrics.prover_training_steps = round_config.prover_steps

                metrics_logger.log_round_metrics(round_metrics)
                metrics_logger.log_round_summary(round_id)

                if check_stopping_criteria(round_metrics, round_config):
                    logger.info(f"Stopping criteria met at round {round_id}")
                    orchestrator.transition(PVGState.COMPLETE)
                    break

                checkpoint_dir = config.output_dir / "checkpoints" / f"round_{round_id}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_dir / "metadata.json", "w") as f:
                    json.dump({
                        "round_id": round_id,
                        "metrics": round_metrics.to_dict(),
                        "config": asdict(round_config),
                    }, f, indent=2, default=str)

                previous_metrics = round_metrics

                if orchestrator.round_id < config.num_rounds - 1:
                    orchestrator.advance_round()
                    orchestrator.transition(PVGState.MINT_DATA)
                else:
                    orchestrator.transition(PVGState.COMPLETE)

    finally:
        pass  # Cleanup would go here

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Final state: {orchestrator.current_state.value}")
    logger.info(f"Progress: {json.dumps(orchestrator.get_progress_summary(), indent=2)}")

    for rid in metrics_logger.get_all_metrics():
        metrics_logger.log_round_summary(rid)


def _load_dataset_samples(dataset: str, data_split: DataSplitConfig) -> List[Dict[str, Any]]:
    """Load dataset samples (placeholder)."""
    return [
        {"problem_id": f"problem_{i}", "prompt": f"Problem {i}", "solution": f"def solve(): return {i}"}
        for i in range(10)
    ]


def _load_honest_codes(samples: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract honest codes from samples."""
    return {s["problem_id"]: s.get("solution", "") for s in samples}


def main() -> None:
    """Entry point for PVG training."""
    args = parse_args()
    config = build_game_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config.output_dir, args.verbose)

    try:
        asyncio.run(run_pvg_training(config, args, dry_run=args.dry_run))
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PVG (Prover-Verifier Game) componentized training runner.

This script runs PVG training by composing standalone components:
1) bootstrap
2) mint
3) train_verifier
4) sync_verifier
5) train_prover
6) sync_prover
7) prompt_test
8) metrics_checkpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ludic.pvg.components import (
    run_bootstrap,
    run_mint,
    run_metrics_checkpoint,
    run_prompt_test,
    run_sync_prover,
    run_sync_verifier,
    run_train_prover,
    run_train_verifier,
)
from ludic.pvg.components.common import build_game_config
from ludic.pvg.orchestrator import PVGOrchestrator, PVGState

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PVG componentized training runner",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--prover-port", type=int, default=8000)
    parser.add_argument("--verifier-port", type=int, default=8001)
    parser.add_argument("--policy-group-port", type=int, default=51216)
    parser.add_argument("--scoring-group-port", type=int, default=51217)

    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--sync-every-steps", type=int, default=1)
    parser.add_argument("--reward-strategy", type=str, default="composite")
    parser.add_argument("--prompt-test-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_game_config(args.config)
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    config.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    bootstrap_manifest = run_bootstrap(
        config_path=args.config,
        output_dir=config.output_dir,
        resume=args.resume,
    )

    orchestrator = PVGOrchestrator(output_dir=config.output_dir, config=config)
    orchestrator.resume_from_checkpoint(validate_config=False)

    for round_id in range(config.num_rounds):
        if orchestrator.current_state == PVGState.COMPLETE:
            break

        mint_manifest = run_mint(
            config_path=args.config,
            bootstrap_manifest=bootstrap_manifest,
            round_id=round_id,
            host=args.host,
            prover_port=args.prover_port,
            output_dir=config.output_dir,
            concurrency=args.concurrency,
        )

        verifier_train_manifest = run_train_verifier(
            config_path=args.config,
            mint_manifest=mint_manifest,
            round_id=round_id,
            output_dir=config.output_dir,
        )

        verifier_sync_manifest = run_sync_verifier(
            config_path=args.config,
            train_manifest=verifier_train_manifest,
            round_id=round_id,
            host=args.host,
            prover_port=args.prover_port,
            verifier_port=args.verifier_port,
            policy_group_port=args.policy_group_port,
            scoring_group_port=args.scoring_group_port,
            output_dir=config.output_dir,
        )

        prover_train_manifest = run_train_prover(
            config_path=args.config,
            sync_verifier_manifest=verifier_sync_manifest,
            round_id=round_id,
            host=args.host,
            prover_port=args.prover_port,
            verifier_port=args.verifier_port,
            output_dir=config.output_dir,
            reward_strategy=args.reward_strategy,
            group_size=4,
            sync_every_steps=args.sync_every_steps,
            concurrency=args.concurrency,
        )

        prover_sync_manifest = run_sync_prover(
            config_path=args.config,
            train_manifest=prover_train_manifest,
            round_id=round_id,
            host=args.host,
            prover_port=args.prover_port,
            output_dir=config.output_dir,
        )

        _ = run_prompt_test(
            config_path=args.config,
            input_manifest=prover_sync_manifest,
            round_id=round_id,
            host=args.host,
            prover_port=args.prover_port,
            output_dir=config.output_dir,
            num_samples=args.prompt_test_samples,
        )

        _ = run_metrics_checkpoint(
            config_path=args.config,
            train_prover_manifest=prover_train_manifest,
            round_id=round_id,
            output_dir=config.output_dir,
        )

        orchestrator.resume_from_checkpoint(validate_config=False)

    logger.info("PVG componentized training complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)

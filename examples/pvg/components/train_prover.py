#!/usr/bin/env python3
"""Run PVG prover training component."""

from __future__ import annotations

import argparse
from pathlib import Path

from ludic.pvg.components import run_train_prover


def main() -> None:
    parser = argparse.ArgumentParser(description="PVG prover training component")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sync-verifier-manifest", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--prover-port", type=int, default=8000)
    parser.add_argument("--verifier-port", type=int, default=8001)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reward-strategy", type=str, default="composite")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--sync-every-steps", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_train_prover(
        config_path=args.config,
        sync_verifier_manifest=args.sync_verifier_manifest,
        round_id=args.round,
        host=args.host,
        prover_port=args.prover_port,
        verifier_port=args.verifier_port,
        output_dir=args.output_dir,
        reward_strategy=args.reward_strategy,
        group_size=args.group_size,
        sync_every_steps=args.sync_every_steps,
        concurrency=args.concurrency,
        device=args.device,
    )


if __name__ == "__main__":
    main()

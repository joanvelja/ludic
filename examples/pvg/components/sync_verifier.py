#!/usr/bin/env python3
"""Run PVG verifier sync component."""

from __future__ import annotations

import argparse
from pathlib import Path

from ludic.pvg.components import run_sync_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description="PVG verifier sync component")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--prover-port", type=int, default=8000)
    parser.add_argument("--verifier-port", type=int, default=8001)
    parser.add_argument("--policy-group-port", type=int, default=51216)
    parser.add_argument("--scoring-group-port", type=int, default=51217)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_sync_verifier(
        config_path=args.config,
        train_manifest=args.train_manifest,
        round_id=args.round,
        host=args.host,
        prover_port=args.prover_port,
        verifier_port=args.verifier_port,
        policy_group_port=args.policy_group_port,
        scoring_group_port=args.scoring_group_port,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run PVG metrics + checkpoint component."""

from __future__ import annotations

import argparse
from pathlib import Path

from ludic.pvg.components import run_metrics_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="PVG metrics/checkpoint component")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-prover-manifest", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_metrics_checkpoint(
        config_path=args.config,
        train_prover_manifest=args.train_prover_manifest,
        round_id=args.round,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

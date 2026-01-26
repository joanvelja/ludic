#!/usr/bin/env python3
"""Run PVG verifier training component."""

from __future__ import annotations

import argparse
from pathlib import Path

from ludic.pvg.components import run_train_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description="PVG verifier training component")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mint-manifest", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--irm-beta", type=float, default=1.0)
    parser.add_argument("--mixture-decay-lambda", type=float, default=0.5)
    parser.add_argument("--pair-samples", type=int, default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_train_verifier(
        config_path=args.config,
        mint_manifest=args.mint_manifest,
        round_id=args.round,
        output_dir=args.output_dir,
        irm_beta=args.irm_beta,
        mixture_decay_lambda=args.mixture_decay_lambda,
        verifier_pair_samples=args.pair_samples,
        verifier_shuffle_buffer=args.shuffle_buffer,
        device=args.device,
    )


if __name__ == "__main__":
    main()

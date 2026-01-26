#!/usr/bin/env python3
"""Run PVG bootstrap component."""

from __future__ import annotations

import argparse
from pathlib import Path

from ludic.pvg.components import run_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="PVG bootstrap component")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_bootstrap(
        config_path=args.config,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

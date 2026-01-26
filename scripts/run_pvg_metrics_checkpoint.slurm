#!/bin/bash
#SBATCH --job-name=pvg-metrics
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j_pvg_metrics.log
#SBATCH --error=logs/slurm_%j_pvg_metrics.log

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-outputs/pvg_metrics_${SLURM_JOB_ID}}"
CONFIG_PATH="${CONFIG_PATH:-examples/pvg/configs/apps_pvg.yaml}"
TRAIN_PROVER_MANIFEST="${TRAIN_PROVER_MANIFEST:?set TRAIN_PROVER_MANIFEST}"
ROUND_ID="${ROUND_ID:-0}"

mkdir -p "$OUTPUT_DIR"

uv python install 3.12
uv sync

export PYTHONUNBUFFERED=1

uv run --env-file .env python examples/pvg/components/metrics_checkpoint.py \
  --config "$CONFIG_PATH" \
  --train-prover-manifest "$TRAIN_PROVER_MANIFEST" \
  --round "$ROUND_ID" \
  --output-dir "$OUTPUT_DIR"

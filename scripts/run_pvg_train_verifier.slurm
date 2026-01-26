#!/bin/bash
#SBATCH --job-name=pvg-train-verifier
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j_pvg_train_verifier.log
#SBATCH --error=logs/slurm_%j_pvg_train_verifier.log

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-outputs/pvg_train_verifier_${SLURM_JOB_ID}}"
CONFIG_PATH="${CONFIG_PATH:-examples/pvg/configs/apps_pvg.yaml}"
MINT_MANIFEST="${MINT_MANIFEST:?set MINT_MANIFEST}"
ROUND_ID="${ROUND_ID:-0}"

mkdir -p "$OUTPUT_DIR"

uv python install 3.12
uv sync

export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH:-/tmp}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_LOCKING=1

uv run --env-file .env python examples/pvg/components/train_verifier.py \
  --config "$CONFIG_PATH" \
  --mint-manifest "$MINT_MANIFEST" \
  --round "$ROUND_ID" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda

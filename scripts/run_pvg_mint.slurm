#!/bin/bash
#SBATCH --job-name=pvg-mint
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j_pvg_mint.log
#SBATCH --error=logs/slurm_%j_pvg_mint.log

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-outputs/pvg_mint_${SLURM_JOB_ID}}"
CONFIG_PATH="${CONFIG_PATH:-examples/pvg/configs/apps_pvg.yaml}"
BOOTSTRAP_MANIFEST="${BOOTSTRAP_MANIFEST:?set BOOTSTRAP_MANIFEST}"
ROUND_ID="${ROUND_ID:-0}"
PROVER_MODEL="${PROVER_MODEL:-Qwen/Qwen3-4B}"
PROVER_PORT="${PROVER_PORT:-8000}"
PROVER_GPU_MEMORY="${PROVER_GPU_MEMORY:-0.77}"

mkdir -p "$OUTPUT_DIR"

uv python install 3.12
uv sync

export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH:-/tmp}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_LOCKING=1
export LUDIC_SANDBOX_BACKEND="podman-hpc"

PROVER_VLLM_LOG="${OUTPUT_DIR}/vllm_prover.log"
CUDA_VISIBLE_DEVICES=0 uv run --env-file .env python -m ludic.inference.vllm_server \
  --model "$PROVER_MODEL" \
  --port $PROVER_PORT \
  --trust-remote-code \
  --gpu-memory-utilization $PROVER_GPU_MEMORY \
  > "$PROVER_VLLM_LOG" 2>&1 &
PROVER_VLLM_PID=$!

cleanup() {
  if [ -n "$PROVER_VLLM_PID" ] && kill -0 $PROVER_VLLM_PID 2>/dev/null; then
    kill $PROVER_VLLM_PID 2>/dev/null || true
    wait $PROVER_VLLM_PID 2>/dev/null || true
  fi
}
trap cleanup EXIT

uv run --env-file .env python examples/pvg/components/mint.py \
  --config "$CONFIG_PATH" \
  --bootstrap-manifest "$BOOTSTRAP_MANIFEST" \
  --round "$ROUND_ID" \
  --host 127.0.0.1 \
  --prover-port $PROVER_PORT \
  --output-dir "$OUTPUT_DIR" \
  --concurrency 16

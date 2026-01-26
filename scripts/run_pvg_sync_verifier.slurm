#!/bin/bash
#SBATCH --job-name=pvg-sync-verifier
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j_pvg_sync_verifier.log
#SBATCH --error=logs/slurm_%j_pvg_sync_verifier.log

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-outputs/pvg_sync_verifier_${SLURM_JOB_ID}}"
CONFIG_PATH="${CONFIG_PATH:-examples/pvg/configs/apps_pvg.yaml}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:?set TRAIN_MANIFEST}"
ROUND_ID="${ROUND_ID:-0}"
VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-4B}"
PROVER_MODEL="${PROVER_MODEL:-Qwen/Qwen3-4B}"
PROVER_PORT="${PROVER_PORT:-8000}"
VERIFIER_PORT="${VERIFIER_PORT:-8001}"
PROVER_GPU_MEMORY="${PROVER_GPU_MEMORY:-0.77}"
VERIFIER_GPU_MEMORY="${VERIFIER_GPU_MEMORY:-0.18}"

mkdir -p "$OUTPUT_DIR"

uv python install 3.12
uv sync

export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH:-/tmp}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_LOCKING=1

PROVER_VLLM_LOG="${OUTPUT_DIR}/vllm_prover.log"
VERIFIER_VLLM_LOG="${OUTPUT_DIR}/vllm_verifier.log"
CUDA_VISIBLE_DEVICES=0 uv run --env-file .env python -m ludic.inference.vllm_server \
  --model "$PROVER_MODEL" \
  --port $PROVER_PORT \
  --trust-remote-code \
  --gpu-memory-utilization $PROVER_GPU_MEMORY \
  > "$PROVER_VLLM_LOG" 2>&1 &
PROVER_VLLM_PID=$!

CUDA_VISIBLE_DEVICES=0 uv run --env-file .env python -m ludic.inference.vllm_reward_server \
  --model "$VERIFIER_MODEL" \
  --port $VERIFIER_PORT \
  --trust-remote-code \
  --gpu-memory-utilization $VERIFIER_GPU_MEMORY \
  > "$VERIFIER_VLLM_LOG" 2>&1 &
VERIFIER_VLLM_PID=$!

cleanup() {
  if [ -n "$PROVER_VLLM_PID" ] && kill -0 $PROVER_VLLM_PID 2>/dev/null; then
    kill $PROVER_VLLM_PID 2>/dev/null || true
    wait $PROVER_VLLM_PID 2>/dev/null || true
  fi
  if [ -n "$VERIFIER_VLLM_PID" ] && kill -0 $VERIFIER_VLLM_PID 2>/dev/null; then
    kill $VERIFIER_VLLM_PID 2>/dev/null || true
    wait $VERIFIER_VLLM_PID 2>/dev/null || true
  fi
}
trap cleanup EXIT

uv run --env-file .env python examples/pvg/components/sync_verifier.py \
  --config "$CONFIG_PATH" \
  --train-manifest "$TRAIN_MANIFEST" \
  --round "$ROUND_ID" \
  --host 127.0.0.1 \
  --prover-port $PROVER_PORT \
  --verifier-port $VERIFIER_PORT \
  --output-dir "$OUTPUT_DIR"

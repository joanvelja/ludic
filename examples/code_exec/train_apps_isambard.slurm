#!/bin/bash
#SBATCH --job-name=ludic-apps-train
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --output=apps_train_%j.log
#SBATCH --error=apps_train_%j.log

# =============================================================================
# APPS LoRA Training on Isambard-AI (GH200 Grace Hopper)
# =============================================================================
#
# This script trains Qwen 3B with LoRA on APPS code problems using:
#   - 2 GPUs: 1 for vLLM inference server, 1 for training
#   - Podman-HPC for sandboxed code execution
#   - GRPO with optional KL regularization
#
# Prerequisites:
#   1. Pre-pull the Python sandbox image:
#      podman-hpc pull python:3.11-slim
#
#   2. (Optional) Login to WandB for cloud logging:
#      wandb login
#
# Usage:
#   # Basic training (500 samples, 200 held-out for eval, 100 steps)
#   sbatch examples/code_exec/train_apps_isambard.slurm
#
#   # Custom configuration via environment variables:
#   LIMIT=1000 EVAL_SAMPLES=300 TRAIN_STEPS=200 sbatch train_apps_isambard.slurm
#
#   # With WandB logging:
#   USE_WANDB=1 sbatch examples/code_exec/train_apps_isambard.slurm
#
#   # With KL regularization:
#   KL_COEFF=0.01 sbatch examples/code_exec/train_apps_isambard.slurm
#
# Resources allocated per GPU on Isambard-AI:
#   - 72 CPU cores
#   - 115 GB Grace RAM
#   - 96 GB H100 GPU memory
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Module loading (minimal for pip-installed packages)
# See: https://docs.isambard.ac.uk/user-documentation/guides/nccl/
# Note: brics/nccl only needed for multi-node distributed training
# -----------------------------------------------------------------------------
# module load PrgEnv-gnu nvidia brics/nccl  # Uncomment if needed for NCCL

# -----------------------------------------------------------------------------
# Fix: PyTorch/vLLM aarch64 CUDA wheels
# GH200 is ARM64 - uv defaults to CPU wheels for aarch64
# --torch-backend=cu128 forces CUDA 12.8 wheels
# NOTE: --torch-backend only works with 'uv pip', not 'uv sync'
# Ref: https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/
# Ref: https://docs.astral.sh/uv/guides/integration/pytorch/
# -----------------------------------------------------------------------------
echo "Installing PyTorch + vLLM with CUDA backend for aarch64..."
uv pip install torch torchvision --torch-backend=cu128
uv pip install vllm --torch-backend=cu128

# -----------------------------------------------------------------------------
# CUDA library path setup
# vLLM requires libcudart.so.12 - find CUDA installation and add to LD_LIBRARY_PATH
# -----------------------------------------------------------------------------
# Try common CUDA locations on Isambard-AI (GH200 nodes)
CUDA_SEARCH_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-12/lib64"
    "/usr/local/cuda-12.0/lib64"
    "/usr/local/cuda-12.1/lib64"
    "/usr/local/cuda-12.2/lib64"
    "/usr/local/cuda-12.3/lib64"
    "/usr/local/cuda-12.4/lib64"
    "/opt/nvidia/hpc_sdk/Linux_aarch64/24.*/cuda/12.*/lib64"
    "/opt/nvidia/hpc_sdk/Linux_aarch64/25.*/cuda/12.*/lib64"
)

CUDA_LIB_PATH=""
for pattern in "${CUDA_SEARCH_PATHS[@]}"; do
    # Use glob expansion for paths with wildcards
    for path in $pattern; do
        if [ -f "$path/libcudart.so.12" ] || [ -f "$path/libcudart.so" ]; then
            CUDA_LIB_PATH="$path"
            break 2
        fi
    done
done

if [ -n "$CUDA_LIB_PATH" ]; then
    echo "Found CUDA libraries at: $CUDA_LIB_PATH"
    export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
    # Also set CUDA_HOME if we can infer it
    CUDA_HOME_CANDIDATE="${CUDA_LIB_PATH%/lib64}"
    if [ -d "$CUDA_HOME_CANDIDATE/bin" ]; then
        export CUDA_HOME="$CUDA_HOME_CANDIDATE"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        echo "Set CUDA_HOME=$CUDA_HOME"
    fi
else
    echo "WARNING: Could not find CUDA libraries in common locations."
    echo "  Searched: ${CUDA_SEARCH_PATHS[*]}"
    echo "  vLLM may fail to start. Check 'module avail' for CUDA modules."
    # Try to find libcudart anywhere as a fallback diagnostic
    echo "  Searching system for libcudart.so.12..."
    find /usr/local /opt -name "libcudart.so.12" 2>/dev/null | head -5 || true
fi

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
LIMIT="${LIMIT:-500}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
SANDBOX_WORKERS="${SANDBOX_WORKERS:-8}"
CONCURRENCY="${CONCURRENCY:-32}"
LORA_RANK="${LORA_RANK:-8}"
KL_COEFF="${KL_COEFF:-0.0}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ludic-apps}"
VLLM_PORT="${VLLM_PORT:-8000}"

# -----------------------------------------------------------------------------
# Job info
# -----------------------------------------------------------------------------
echo "========================================"
echo "Ludic APPS Training on Isambard-AI"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS"
echo "Time: $(date)"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Samples: $LIMIT (train) + $EVAL_SAMPLES (eval)"
echo "  Train steps: $TRAIN_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Group size: $GROUP_SIZE"
echo "  LoRA rank: $LORA_RANK"
echo "  KL coefficient: $KL_COEFF"
echo "  Sandbox workers: $SANDBOX_WORKERS"
echo "  Concurrency: $CONCURRENCY"
echo "  WandB: $([ "$USE_WANDB" = "1" ] && echo "enabled" || echo "disabled")"
echo "========================================"
echo ""

# -----------------------------------------------------------------------------
# Environment checks
# -----------------------------------------------------------------------------

echo "Verifying PyTorch CUDA..."
# --no-sync prevents uv from reverting to CPU torch via pyproject.toml
uv run --no-sync python -c "import torch; print('  PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || {
    echo "ERROR: PyTorch CUDA not available."
    exit 1
}

# Check podman-hpc
if ! command -v podman-hpc &> /dev/null; then
    echo "ERROR: podman-hpc not found in PATH"
    exit 1
fi
echo "podman-hpc version: $(podman-hpc --version 2>&1 | head -1)"

# Check uv
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found in PATH"
    exit 1
fi
echo "uv version: $(uv --version 2>&1)"

# Ensure we're in the repo root
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Must run from repo root (pyproject.toml not found)"
    echo "  Hint: cd to your ludic repo before submitting"
    exit 1
fi

# Check Python sandbox image
echo ""
echo "Checking container image..."
if podman-hpc images | grep -q "python.*3.11"; then
    echo "  python:3.11-slim image found"
else
    echo "  Image not found, pulling..."
    podman-hpc pull python:3.11-slim
fi

# -----------------------------------------------------------------------------
# GPU allocation
# -----------------------------------------------------------------------------
echo ""
echo "GPU allocation:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

# Assign GPUs: GPU 0 for vLLM, GPU 1 for training
export VLLM_GPU=0
export TRAIN_GPU=1

echo ""
echo "GPU assignment:"
echo "  GPU $VLLM_GPU: vLLM inference server"
echo "  GPU $TRAIN_GPU: Training"

# -----------------------------------------------------------------------------
# Start vLLM server (background)
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Starting vLLM server..."
echo "========================================"

# Create log file for vLLM
VLLM_LOG="vllm_server_${SLURM_JOB_ID}.log"

CUDA_VISIBLE_DEVICES=$VLLM_GPU uv run --no-sync python -m ludic.inference.vllm_server \
    --model "$MODEL" \
    --port $VLLM_PORT \
    --trust-remote-code \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM server started (PID: $VLLM_PID, log: $VLLM_LOG)"

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=300  # 5 minutes
WAITED=0
while ! curl -s "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died. Check $VLLM_LOG for details."
        tail -50 "$VLLM_LOG"
        exit 1
    fi
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waiting... (${WAITED}s / ${MAX_WAIT}s)"
done
echo "vLLM server is ready!"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "  Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi
    echo "Cleanup complete."
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Run training
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Starting training..."
echo "========================================"

# Build training command
TRAIN_CMD="uv run --no-sync python examples/code_exec/train_apps.py"
TRAIN_CMD="$TRAIN_CMD --model $MODEL"
TRAIN_CMD="$TRAIN_CMD --host 127.0.0.1 --port $VLLM_PORT"
TRAIN_CMD="$TRAIN_CMD --limit $LIMIT"
TRAIN_CMD="$TRAIN_CMD --eval-samples $EVAL_SAMPLES"
TRAIN_CMD="$TRAIN_CMD --train-steps $TRAIN_STEPS"
TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --group-size $GROUP_SIZE"
TRAIN_CMD="$TRAIN_CMD --sandbox-workers $SANDBOX_WORKERS"
TRAIN_CMD="$TRAIN_CMD --sandbox-backend podman-hpc"
TRAIN_CMD="$TRAIN_CMD --concurrency $CONCURRENCY"
TRAIN_CMD="$TRAIN_CMD --lora-rank $LORA_RANK"
TRAIN_CMD="$TRAIN_CMD --eval-every 25"
TRAIN_CMD="$TRAIN_CMD --eval-before-start"
TRAIN_CMD="$TRAIN_CMD --final-save"

# Add KL coefficient if non-zero
if [ "$KL_COEFF" != "0.0" ] && [ "$KL_COEFF" != "0" ]; then
    TRAIN_CMD="$TRAIN_CMD --kl-coeff $KL_COEFF"
fi

# Add WandB if enabled
if [ "$USE_WANDB" = "1" ]; then
    TRAIN_CMD="$TRAIN_CMD --wandb --wandb-project $WANDB_PROJECT"
fi

echo "Training command:"
echo "  $TRAIN_CMD"
echo ""

# Run training on GPU 1
CUDA_VISIBLE_DEVICES=$TRAIN_GPU PYTHONPATH=. $TRAIN_CMD

EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Checkpoints saved to: checkpoints_apps/"
    echo "Rollout log: apps_train_rollouts.jsonl"
    if [ "$USE_WANDB" = "1" ]; then
        echo "WandB dashboard: https://wandb.ai/$WANDB_PROJECT"
    fi
else
    echo "Training failed (exit code: $EXIT_CODE)"
fi
echo "========================================"
echo "Completed at: $(date)"

exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=pvg-gpu-tests
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%j_pvg_gpu_tests.log
#SBATCH --error=logs/slurm_%j_pvg_gpu_tests.log

# PVG GPU Integration Tests
# =========================
# Runs the full PVG test suite including GPU-requiring tests.
# Requires 2+ GPUs for end-to-end training loop tests.
#
# Usage:
#   sbatch scripts/run_pvg_gpu_tests.slurm
#
# Tests run:
#   1. PVG unit tests (no GPU needed)
#   2. PVG integration tests (GPU needed)
#   3. Dry-run training with verbose logging

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

echo "========================================"
echo "PVG GPU Test Suite"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS"
echo "Time: $(date)"
echo "========================================"

# Check for .env file (required for HF_TOKEN)
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Please create it with HF_TOKEN."
    echo "  Example: echo 'HF_TOKEN=your_token_here' > .env"
    exit 1
fi

# Sync environment
echo "Syncing Python environment..."
uv sync

# GPU check
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || echo "nvidia-smi not available"

export PYTHONUNBUFFERED=1

# HuggingFace cache on Lustre (high-quota storage)
export HF_HOME="${SCRATCH:-/tmp}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
echo "HF cache: $HF_HOME"

# Sandbox backend (podman-hpc for HPC)
export LUDIC_SANDBOX_BACKEND="podman-hpc"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$PROVER_VLLM_PID" ] && kill -0 $PROVER_VLLM_PID 2>/dev/null; then
        echo "  Stopping prover vLLM (PID: $PROVER_VLLM_PID)..."
        kill $PROVER_VLLM_PID 2>/dev/null || true
        wait $PROVER_VLLM_PID 2>/dev/null || true
    fi
    if [ -n "$VERIFIER_VLLM_PID" ] && kill -0 $VERIFIER_VLLM_PID 2>/dev/null; then
        echo "  Stopping verifier vLLM (PID: $VERIFIER_VLLM_PID)..."
        kill $VERIFIER_VLLM_PID 2>/dev/null || true
        wait $VERIFIER_VLLM_PID 2>/dev/null || true
    fi
    echo "Cleanup complete."
}
trap cleanup EXIT

# Phase 1: PVG Component Tests (requires vLLM)
echo ""
echo "========================================"
echo "Phase 1: PVG Component Tests"
echo "========================================"
OUTPUT_DIR="outputs/pvg_test_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Use Qwen3-4B (non-gated) for testing
VERIFIER_MODEL="Qwen/Qwen3-4B"
PROVER_MODEL="Qwen/Qwen3-4B"

VLLM_GPU=0
TRAIN_GPUS="1,2,3"
PROVER_VLLM_PORT="${PROVER_VLLM_PORT:-8000}"
VERIFIER_VLLM_PORT="${VERIFIER_VLLM_PORT:-8001}"
PROVER_VLLM_GPU_MEMORY="${PROVER_VLLM_GPU_MEMORY:-0.77}"
VERIFIER_VLLM_GPU_MEMORY="${VERIFIER_VLLM_GPU_MEMORY:-0.18}"

echo "Starting vLLM servers for PVG component tests..."
PROVER_VLLM_LOG="${OUTPUT_DIR}/vllm_prover.log"
VERIFIER_VLLM_LOG="${OUTPUT_DIR}/vllm_verifier.log"

CUDA_VISIBLE_DEVICES=$VLLM_GPU uv run --env-file .env python -m ludic.inference.vllm_server \
    --model "$PROVER_MODEL" \
    --port $PROVER_VLLM_PORT \
    --trust-remote-code \
    --gpu-memory-utilization $PROVER_VLLM_GPU_MEMORY \
    > "$PROVER_VLLM_LOG" 2>&1 &
PROVER_VLLM_PID=$!

CUDA_VISIBLE_DEVICES=$VLLM_GPU uv run --env-file .env python -m ludic.inference.vllm_reward_server \
    --model "$VERIFIER_MODEL" \
    --port $VERIFIER_VLLM_PORT \
    --trust-remote-code \
    --gpu-memory-utilization $VERIFIER_VLLM_GPU_MEMORY \
    > "$VERIFIER_VLLM_LOG" 2>&1 &
VERIFIER_VLLM_PID=$!

echo "Waiting for vLLM servers to be ready..."
MAX_WAIT=300
WAITED=0
while true; do
    PROVER_READY=0
    VERIFIER_READY=0
    if curl -s "http://127.0.0.1:$PROVER_VLLM_PORT/health" > /dev/null 2>&1; then
        PROVER_READY=1
    fi
    if curl -s "http://127.0.0.1:$VERIFIER_VLLM_PORT/health" > /dev/null 2>&1; then
        VERIFIER_READY=1
    fi
    if [ $PROVER_READY -eq 1 ] && [ $VERIFIER_READY -eq 1 ]; then
        break
    fi
    if ! kill -0 $PROVER_VLLM_PID 2>/dev/null; then
        echo "ERROR: Prover vLLM died. Check $PROVER_VLLM_LOG"
        tail -50 "$PROVER_VLLM_LOG"
        exit 1
    fi
    if ! kill -0 $VERIFIER_VLLM_PID 2>/dev/null; then
        echo "ERROR: Verifier vLLM died. Check $VERIFIER_VLLM_LOG"
        tail -50 "$VERIFIER_VLLM_LOG"
        exit 1
    fi
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM timeout"
        kill $PROVER_VLLM_PID 2>/dev/null || true
        kill $VERIFIER_VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
echo "vLLM servers ready!"

export PVG_COMPONENT_TEST=1
export PVG_TEST_HOST=127.0.0.1
export PVG_TEST_PROVER_PORT=$PROVER_VLLM_PORT
export PVG_TEST_VERIFIER_PORT=$VERIFIER_VLLM_PORT

uv run --env-file .env pytest tests/pvg/component/test_pvg_components.py -v --tb=short \
    --junitxml=${OUTPUT_DIR}/component_tests.xml \
    2>&1 | tee ${OUTPUT_DIR}/component_tests.log

# Phase 2: Componentized training run (single round)
echo ""
echo "========================================"
echo "Phase 2: PVG Training Run"
echo "========================================"

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS uv run --env-file .env python examples/pvg/train_pvg.py \
    --config tests/pvg/configs/apps_pvg_test.yaml \
    --output-dir "$OUTPUT_DIR" \
    --host 127.0.0.1 \
    --prover-port $PROVER_VLLM_PORT \
    --verifier-port $VERIFIER_VLLM_PORT \
    --sync-every-steps 1 \
    --reward-strategy composite \
    --concurrency 4

# Show generated logs
echo ""
echo "========================================"
echo "Generated Logs:"
echo "========================================"
ls -la "$OUTPUT_DIR"/*.log 2>/dev/null || echo "No log files (dry-run)"
cat "$OUTPUT_DIR/train.log" 2>/dev/null | head -100 || true

echo ""
echo "========================================"
echo "PVG GPU tests completed at $(date)"
echo "========================================"

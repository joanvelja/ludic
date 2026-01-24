#!/bin/bash
#SBATCH --job-name=sneaky-demo
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

# Logging directory tree:
#   logs/
#   └── YYYY-MM-DD/
#       └── HH-MM-SS_<job-id>/
#           ├── slurm.log           # SLURM stdout/stderr
#           └── vllm.log            # vLLM inference server

#SBATCH --output=logs/slurm_%j_init.log
#SBATCH --error=logs/slurm_%j_init.log

# =============================================================================
# Sneaky Inference Demo on Isambard-AI (GH200 Grace Hopper)
# =============================================================================
#
# This script runs GRPO-style parallel generations through the full sneaky
# verification pipeline:
#   - Load APPS Control Arena problems
#   - Pre-batch honest solutions
#   - Generate sneaky code with Qwen via vLLM
#   - Verify in sandboxed execution environment
#
# This is inference + verification only (no RL training).
#
# Prerequisites:
#   1. Create a .env file with your HF_TOKEN in the repo root:
#      echo 'HF_TOKEN=your_token_here' > .env
#
#   2. Pre-pull the Python sandbox image:
#      podman-hpc pull python:3.11-slim
#
# Usage:
#   # Default demo (5 problems x 4 generations = 20 rollouts)
#   sbatch examples/code_exec/sneaky_inference_isambard.slurm
#
#   # Custom configuration via environment variables:
#   LIMIT=20 GROUP_SIZE=8 sbatch examples/code_exec/sneaky_inference_isambard.slurm
#
#   # Larger scale:
#   LIMIT=50 GROUP_SIZE=8 SANDBOX_WORKERS=32 CONCURRENCY=64 \
#       sbatch examples/code_exec/sneaky_inference_isambard.slurm
#
# Resources: Single GPU sufficient (inference only)
#
# =============================================================================

set -e

# Create the structured log directory for this job
LOG_DATE=$(date +%Y-%m-%d)
LOG_TIME=$(date +%H-%M-%S)
LOG_DIR="logs/${LOG_DATE}/${LOG_TIME}_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Redirect all subsequent output to the structured log directory
exec > >(tee -a "${LOG_DIR}/slurm.log") 2>&1

# Clean up the initial temp log file (move its contents to the proper location)
if [ -f "logs/slurm_${SLURM_JOB_ID}_init.log" ]; then
    cat "logs/slurm_${SLURM_JOB_ID}_init.log" >> "${LOG_DIR}/slurm.log"
    rm -f "logs/slurm_${SLURM_JOB_ID}_init.log"
fi

echo "Log directory: $LOG_DIR"

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
LIMIT="${LIMIT:-5}"
GROUP_SIZE="${GROUP_SIZE:-4}"
SANDBOX_WORKERS="${SANDBOX_WORKERS:-8}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_CONCURRENT_OPS="${MAX_CONCURRENT_OPS:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Compute total rollouts
TOTAL_ROLLOUTS=$((LIMIT * GROUP_SIZE))

# -----------------------------------------------------------------------------
# Job info
# -----------------------------------------------------------------------------
echo "========================================"
echo "Sneaky Inference Demo on Isambard-AI"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS"
echo "Time: $(date)"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Problems (limit): $LIMIT"
echo "  Generations (group_size): $GROUP_SIZE"
echo "  Total rollouts: $TOTAL_ROLLOUTS"
echo ""
echo "Parallelism:"
echo "  Concurrency: $CONCURRENCY async tasks"
echo "  Sandbox workers: $SANDBOX_WORKERS containers"
echo "  Max concurrent ops: $MAX_CONCURRENT_OPS"
echo ""
echo "Inference:"
echo "  Temperature: $TEMPERATURE"
echo "  Max tokens: $MAX_TOKENS"
echo "========================================"
echo ""

# Check for .env file (required for HF_TOKEN)
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Please create it with HF_TOKEN."
    echo "  Example: echo 'HF_TOKEN=your_token_here' > .env"
    exit 1
fi

# -----------------------------------------------------------------------------
# Pre-cleanup: Remove orphaned containers and sandbox directories from previous runs
# -----------------------------------------------------------------------------
echo "Pre-cleanup: Removing orphaned containers and sandbox directories..."
podman-hpc ps -aq --filter "name=ludic-sandbox" 2>/dev/null | xargs -r podman-hpc rm -f 2>/dev/null || true
rm -rf /home/u5ds/joanv.u5ds/sandbox/ludic-* 2>/dev/null || true
echo "  Pre-cleanup complete"

# -----------------------------------------------------------------------------
# Environment sync
# -----------------------------------------------------------------------------
echo "Syncing Python environment..."
uv python install 3.12
uv sync

# -----------------------------------------------------------------------------
# GPU allocation
# -----------------------------------------------------------------------------
echo ""
echo "GPU allocation:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

# Use GPU 0 for vLLM (inference only, single GPU sufficient)
export VLLM_GPU=0

echo ""
echo "GPU assignment:"
echo "  GPU $VLLM_GPU: vLLM inference server"

# -----------------------------------------------------------------------------
# Start vLLM server (background)
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Starting vLLM server..."
echo "========================================"

VLLM_LOG="${LOG_DIR}/vllm.log"

CUDA_VISIBLE_DEVICES=$VLLM_GPU uv run --env-file .env python -m ludic.inference.vllm_server \
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

# -----------------------------------------------------------------------------
# Cleanup function
# -----------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "  Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi
    # Clean up any ludic sandbox containers
    echo "  Cleaning up sandbox containers..."
    podman-hpc ps -aq --filter "name=ludic-sandbox" 2>/dev/null | xargs -r podman-hpc rm -f 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Run demo
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Running Sneaky Inference Demo..."
echo "========================================"

# Build command
DEMO_CMD="uv run --env-file .env python examples/code_exec/demo_sneaky_inference.py"
DEMO_CMD="$DEMO_CMD --model $MODEL"
DEMO_CMD="$DEMO_CMD --host 127.0.0.1 --port $VLLM_PORT"
DEMO_CMD="$DEMO_CMD --limit $LIMIT"
DEMO_CMD="$DEMO_CMD --group-size $GROUP_SIZE"
DEMO_CMD="$DEMO_CMD --sandbox-workers $SANDBOX_WORKERS"
DEMO_CMD="$DEMO_CMD --concurrency $CONCURRENCY"
DEMO_CMD="$DEMO_CMD --max-concurrent-ops $MAX_CONCURRENT_OPS"
DEMO_CMD="$DEMO_CMD --temperature $TEMPERATURE"
DEMO_CMD="$DEMO_CMD --max-tokens $MAX_TOKENS"
DEMO_CMD="$DEMO_CMD --sandbox-backend podman-hpc"
DEMO_CMD="$DEMO_CMD --minimal-sandbox"
DEMO_CMD="$DEMO_CMD --rollout-log ${LOG_DIR}/sneaky_rollouts.jsonl"
DEMO_CMD="$DEMO_CMD --json-output ${LOG_DIR}/results.json"

echo "Command:"
echo "  $DEMO_CMD"
echo ""

# Run the demo (no CUDA_VISIBLE_DEVICES needed - inference via vLLM server)
PYTHONPATH=. $DEMO_CMD

EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Demo completed successfully!"
    echo ""
    echo "Output files:"
    echo "  Rollout log: ${LOG_DIR}/sneaky_rollouts.jsonl"
    echo "  Results JSON: ${LOG_DIR}/results.json"
    echo "  vLLM log: ${LOG_DIR}/vllm.log"
    echo "  Slurm log: ${LOG_DIR}/slurm.log"
else
    echo "Demo failed (exit code: $EXIT_CODE)"
fi
echo "========================================"
echo "Completed at: $(date)"

exit $EXIT_CODE

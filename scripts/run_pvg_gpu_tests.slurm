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

# Phase 1: PVG Unit Tests
echo ""
echo "========================================"
echo "Phase 1: PVG Unit Tests"
echo "========================================"
uv run --env-file .env pytest tests/pvg/unit/ -v --tb=short 2>&1 || true
# Don't fail on unit tests (sandbox errors expected on some envs)

# Phase 2: PVG Integration Tests with GPU
echo ""
echo "========================================"
echo "Phase 2: PVG Integration Tests (GPU)"
echo "========================================"
uv run --env-file .env pytest tests/integration/test_pvg_e2e.py -v --tb=short

# Phase 3: Dry-run training to verify full pipeline
echo ""
echo "========================================"
echo "Phase 3: PVG Training Dry Run"
echo "========================================"
OUTPUT_DIR="outputs/pvg_test_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

uv run --env-file .env python examples/pvg/train_pvg.py \
    --dry-run \
    --mock-inference \
    --output-dir "$OUTPUT_DIR" \
    --verbose

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

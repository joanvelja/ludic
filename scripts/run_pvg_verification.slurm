#!/bin/bash
#SBATCH --job-name=pvg-verify
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j_pvg_verify.log
#SBATCH --error=logs/slurm_%j_pvg_verify.log

# PVG Full Verification Suite
# ===========================
# Comprehensive verification tests for the PVG training loop.
# Validates critical training properties:
#
# 1. No label leakage when training verifier (chosen=honest, rejected=sneaky)
# 2. Correct prompts for prover training
# 3. Per training step syncing events
# 4. Thorough documentation of the config used
# 5. LoRA/head re-initialization at each verifier training stage
#
# Usage:
#   sbatch scripts/run_pvg_verification.slurm
#
# Output:
#   outputs/pvg_verify_<job_id>/
#   ├── unit_tests.log       # Unit test results
#   ├── verification_tests.log  # Verification test results
#   ├── integration_tests.log   # Integration test results
#   └── training/
#       ├── train.log        # Full training log
#       └── samples.log      # Sample inspection log

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

# Create structured output directory
OUTPUT_DIR="outputs/pvg_verify_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "PVG Full Verification Suite"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "Time: $(date)"
echo "========================================"

# Check for .env file (required for HF_TOKEN)
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Please create it with HF_TOKEN."
    echo "  Example: echo 'HF_TOKEN=your_token_here' > .env"
    exit 1
fi

# -----------------------------------------------------------------------------
# Pre-cleanup: Remove orphaned containers from previous runs
# -----------------------------------------------------------------------------
echo "Pre-cleanup: Removing orphaned containers and sandbox directories..."
podman-hpc ps -aq --filter "name=ludic-sandbox" 2>/dev/null | xargs -r podman-hpc rm -f 2>/dev/null || true
rm -rf /home/u5ds/joanv.u5ds/sandbox/ludic-* 2>/dev/null || true
echo "  Pre-cleanup complete"

# Sync environment
echo "Syncing Python environment..."
uv python install 3.12
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

# Disable HuggingFace file locking (Lustre has issues with POSIX locks)
export HF_HUB_DISABLE_LOCKING=1

# Clean stale HF locks (Lustre has issues with POSIX locks)
# This prevents infinite lock acquisition loops on shared filesystems
echo "Cleaning stale HuggingFace locks..."
find "${HF_HOME}/.locks" -type f -name "*.lock" -mmin +5 -delete 2>/dev/null || true
find "${HF_HUB_CACHE}" -type f -name ".lock" -mmin +5 -delete 2>/dev/null || true
echo "  Lock cleanup complete"

# Set sandbox backend for podman-hpc
export LUDIC_SANDBOX_BACKEND="podman-hpc"

# -----------------------------------------------------------------------------
# Pre-pull sandbox container image
# -----------------------------------------------------------------------------
echo "Pre-pulling Python sandbox image for tests..."
podman-hpc pull python:3.11-slim 2>&1 || echo "Image pull failed (may already exist)"
echo "  Image ready"

# -----------------------------------------------------------------------------
# Cleanup function (trap on exit)
# -----------------------------------------------------------------------------
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
    # Clean up any ludic sandbox containers
    echo "  Cleaning up sandbox containers..."
    podman-hpc ps -aq --filter "name=ludic-sandbox" 2>/dev/null | xargs -r podman-hpc rm -f 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT

# ======================
# PHASE 1: PVG Unit Tests (deprecated paths)
# ======================
echo ""
echo "========================================"
echo "PHASE 1: PVG Unit Tests"
echo "========================================"
echo "Skipping legacy PVG unit tests; component tests run after vLLM startup."

# ======================
# PHASE 3: Integration Tests (GPU)
# ======================
echo ""
echo "========================================"
echo "PHASE 3: Integration Tests (GPU)"
echo "========================================"
echo "Skipping legacy PVG integration test path; component tests cover PVG behavior."

# ======================
# PHASE 4: Mini Training with Logging
# ======================
echo ""
echo "========================================"
echo "PHASE 4: Mini Training with Logging"
echo "========================================"
echo ""
echo "This phase runs a short training session with extensive logging"
echo "to verify the following critical properties:"
echo ""
echo "  1. [LABEL LEAKAGE] chosen=honest, rejected=sneaky"
echo "  2. [PROMPT CHECK] Prover receives correct prompts"
echo "  3. [WEIGHT SYNC] Syncs happen at configured intervals"
echo "  4. [CONFIG] Full config logged at each phase"
echo "  5. [REINIT] LoRA/head reinitialized each round"
echo ""

# Use Qwen3-4B (non-gated) for testing
VERIFIER_MODEL="Qwen/Qwen3-4B"
PROVER_MODEL="Qwen/Qwen3-4B"

# vLLM server config (prover + verifier reward)
VLLM_GPU=0
TRAIN_GPUS="1,2,3"
PROVER_VLLM_PORT="${PROVER_VLLM_PORT:-8000}"
VERIFIER_VLLM_PORT="${VERIFIER_VLLM_PORT:-8001}"
PROVER_VLLM_GPU_MEMORY="${PROVER_VLLM_GPU_MEMORY:-0.77}"
VERIFIER_VLLM_GPU_MEMORY="${VERIFIER_VLLM_GPU_MEMORY:-0.18}"

echo ""
echo "Starting vLLM servers for PVG training..."
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

echo ""
echo "========================================"
echo "PHASE 4A: PVG Component Tests"
echo "========================================"
export PVG_COMPONENT_TEST=1
export PVG_TEST_HOST=127.0.0.1
export PVG_TEST_PROVER_PORT=$PROVER_VLLM_PORT
export PVG_TEST_VERIFIER_PORT=$VERIFIER_VLLM_PORT
uv run --env-file .env pytest tests/pvg/component/test_pvg_components.py -v --tb=short \
    --junitxml=${OUTPUT_DIR}/component_tests.xml \
    2>&1 | tee ${OUTPUT_DIR}/component_tests.log

echo "Component tests complete. See ${OUTPUT_DIR}/component_tests.log"

echo ""
echo "========================================"
echo "PHASE 4B: Mini Training with Logging"
echo "========================================"
echo ""
echo "This phase runs a short training session with extensive logging"
echo "to verify the following critical properties:"
echo ""
echo "  1. [LABEL LEAKAGE] chosen=honest, rejected=sneaky"
echo "  2. [PROMPT CHECK] Prover receives correct prompts"
echo "  3. [WEIGHT SYNC] Syncs happen at configured intervals"
echo "  4. [CONFIG] Full config logged at each phase"
echo "  5. [REINIT] LoRA/head reinitialized each round"
echo ""

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS uv run --env-file .env python examples/pvg/train_pvg.py \
    --config tests/pvg/configs/apps_pvg_test.yaml \
    --output-dir ${OUTPUT_DIR}/training \
    --host 127.0.0.1 \
    --prover-port $PROVER_VLLM_PORT \
    --verifier-port $VERIFIER_VLLM_PORT \
    --sync-every-steps 1 \
    --reward-strategy composite \
    --concurrency 4 \
    2>&1 | tee ${OUTPUT_DIR}/training_run.log

echo "Training complete."

# ======================
# PHASE 5: Manifest Verification
# ======================
echo ""
echo "========================================"
echo "PHASE 5: Manifest Verification"
echo "========================================"

MANIFEST_DIR="${OUTPUT_DIR}/training/manifests"
PROMPT_REPORT="${OUTPUT_DIR}/training/prompt_reports/round_0.json"

echo ""
echo "=== Component manifests ==="
if [ -d "$MANIFEST_DIR" ]; then
    echo "Manifests found:"
    ls -1 "$MANIFEST_DIR"/round_*/sync_* 2>/dev/null | head -10 || echo "  No sync manifests found"
    ls -1 "$MANIFEST_DIR"/round_*/train_* 2>/dev/null | head -10 || echo "  No train manifests found"
else
    echo "Manifest directory not found: $MANIFEST_DIR"
fi

echo ""
echo "=== Prompt test report ==="
if [ -f "$PROMPT_REPORT" ]; then
    echo "Prompt report found: $PROMPT_REPORT"
    cat "$PROMPT_REPORT" | head -40
else
    echo "Prompt report not found"
fi

echo ""
echo "=== Round data ==="
ROUND_DATA_DIR="${OUTPUT_DIR}/training/data/round_0"
if [ -d "$ROUND_DATA_DIR" ]; then
    ls -la "$ROUND_DATA_DIR"
else
    echo "Round data directory not found"
fi

# ======================
# Summary
# ======================
echo ""
echo "========================================"
echo "VERIFICATION SUMMARY"
echo "========================================"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Test Reports (JUnit XML):"
ls -la ${OUTPUT_DIR}/*.xml 2>/dev/null || echo "  None"
echo ""
echo "Log Files:"
ls -la ${OUTPUT_DIR}/*.log 2>/dev/null || echo "  None"
echo ""
echo "Training Outputs:"
ls -la ${OUTPUT_DIR}/training/*.log 2>/dev/null || echo "  None"

# Count pass/fail from JUnit XMLs
echo ""
echo "Test Results Summary:"
for xml in ${OUTPUT_DIR}/*.xml; do
    if [ -f "$xml" ]; then
        name=$(basename "$xml" .xml)
        tests=$(grep -oP 'tests="\K[0-9]+' "$xml" | head -1)
        failures=$(grep -oP 'failures="\K[0-9]+' "$xml" | head -1)
        errors=$(grep -oP 'errors="\K[0-9]+' "$xml" | head -1)
        echo "  $name: $tests tests, $failures failures, $errors errors"
    fi
done

echo ""
echo "========================================"
echo "PVG VERIFICATION CHECKLIST"
echo "========================================"
echo ""
echo "To verify the training pipeline, examine the logs for:"
echo ""
echo "1. LABEL LEAKAGE: Search for '[LABEL LEAK CHECK]' and '[PASS]' markers"
echo "   grep -E '(LABEL LEAK|PASS|FAIL)' ${OUTPUT_DIR}/training/samples.log"
echo ""
echo "2. PROVER PROMPTS: Search for '[PROMPT CHECK]' markers"
echo "   grep -A5 'PROMPT CHECK' ${OUTPUT_DIR}/training/samples.log"
echo ""
echo "3. WEIGHT SYNC: Search for '[WEIGHT SYNC]' markers"
echo "   grep 'WEIGHT SYNC' ${OUTPUT_DIR}/training/train.log"
echo ""
echo "4. CONFIG: Search for 'CONFIGURATION' sections"
echo "   grep -A10 'CONFIGURATION' ${OUTPUT_DIR}/training/train.log"
echo ""
echo "5. LORA/HEAD REINIT: Search for 'REINIT' markers"
echo "   grep -E '(REINIT|PRE-reinit|POST-reinit)' ${OUTPUT_DIR}/training/train.log"
echo ""
echo "========================================"
echo "Verification completed at $(date)"
echo "========================================"

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
    # Clean up any ludic sandbox containers
    echo "  Cleaning up sandbox containers..."
    podman-hpc ps -aq --filter "name=ludic-sandbox" 2>/dev/null | xargs -r podman-hpc rm -f 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT

# ======================
# PHASE 1: Unit Tests
# ======================
echo ""
echo "========================================"
echo "PHASE 1: PVG Unit Tests"
echo "========================================"
uv run --env-file .env pytest tests/pvg/unit/ -v --tb=short \
    --junitxml=${OUTPUT_DIR}/unit_tests.xml \
    2>&1 | tee ${OUTPUT_DIR}/unit_tests.log || true

echo "Unit tests complete. See ${OUTPUT_DIR}/unit_tests.log"

# ======================
# PHASE 2: Verification Tests
# ======================
echo ""
echo "========================================"
echo "PHASE 2: PVG Verification Tests"
echo "========================================"
echo "Testing: label leakage, prompts, syncing..."
uv run --env-file .env pytest tests/pvg/test_pvg_verification.py -v --tb=short \
    --junitxml=${OUTPUT_DIR}/verification_tests.xml \
    2>&1 | tee ${OUTPUT_DIR}/verification_tests.log

echo "Verification tests complete. See ${OUTPUT_DIR}/verification_tests.log"

# ======================
# PHASE 3: Integration Tests
# ======================
echo ""
echo "========================================"
echo "PHASE 3: Integration Tests (GPU)"
echo "========================================"
uv run --env-file .env pytest tests/integration/test_pvg_e2e.py -v --tb=short \
    --junitxml=${OUTPUT_DIR}/integration_tests.xml \
    2>&1 | tee ${OUTPUT_DIR}/integration_tests.log

echo "Integration tests complete. See ${OUTPUT_DIR}/integration_tests.log"

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

uv run --env-file .env python examples/pvg/train_pvg.py \
    --verifier-model "$VERIFIER_MODEL" \
    --prover-model "$PROVER_MODEL" \
    --num-rounds 2 \
    --verifier-steps 10 \
    --prover-steps 20 \
    --output-dir ${OUTPUT_DIR}/training \
    --minimal-sandbox \
    --verbose \
    2>&1 | tee ${OUTPUT_DIR}/training_run.log

echo "Training complete."

# ======================
# PHASE 5: Log Analysis
# ======================
echo ""
echo "========================================"
echo "PHASE 5: Log Analysis"
echo "========================================"

SAMPLES_LOG="${OUTPUT_DIR}/training/samples.log"
TRAIN_LOG="${OUTPUT_DIR}/training/train.log"

echo ""
echo "=========================================="
echo "=== VERIFICATION 1: LABEL LEAKAGE CHECK ==="
echo "=========================================="
if [ -f "$SAMPLES_LOG" ]; then
    LEAK_CHECK_COUNT=$(grep -c "LABEL LEAK CHECK" "$SAMPLES_LOG" 2>/dev/null || echo "0")
    echo "Label leak check markers found: $LEAK_CHECK_COUNT"
    echo ""
    echo "Checking for PASS markers:"
    grep -E "\[PASS\]" "$SAMPLES_LOG" | head -10 || echo "  No PASS markers found"
    echo ""
    echo "Checking for FAIL markers (should be 0):"
    FAIL_COUNT=$(grep -c "\[FAIL\]" "$SAMPLES_LOG" 2>/dev/null || echo "0")
    echo "  FAIL markers: $FAIL_COUNT"
    if [ "$FAIL_COUNT" -gt 0 ]; then
        echo "  !!! WARNING: Label leakage detected!"
        grep "\[FAIL\]" "$SAMPLES_LOG"
    else
        echo "  ✓ No label leakage detected"
    fi
else
    echo "samples.log not found"
fi

echo ""
echo "=========================================="
echo "=== VERIFICATION 2: PROMPT CHECK ==="
echo "=========================================="
if [ -f "$SAMPLES_LOG" ]; then
    PROMPT_CHECK_COUNT=$(grep -c "PROMPT CHECK" "$SAMPLES_LOG" 2>/dev/null || echo "0")
    echo "Prompt check markers found: $PROMPT_CHECK_COUNT"
    echo ""
    echo "Sample prompts (first 20 lines):"
    grep -A3 "PROMPT CHECK" "$SAMPLES_LOG" | head -20 || echo "  No prompt check logs"
else
    echo "samples.log not found"
fi

echo ""
echo "=========================================="
echo "=== VERIFICATION 3: WEIGHT SYNC CHECK ==="
echo "=========================================="
if [ -f "$TRAIN_LOG" ]; then
    SYNC_COUNT=$(grep -c "WEIGHT SYNC" "$TRAIN_LOG" 2>/dev/null || echo "0")
    echo "Weight sync events found: $SYNC_COUNT"
    echo ""
    SYNC_SUCCESS=$(grep -c "SYNC SUCCESS" "$TRAIN_LOG" 2>/dev/null || echo "0")
    SYNC_FAIL=$(grep -c "SYNC FAILED" "$TRAIN_LOG" 2>/dev/null || echo "0")
    echo "  Successful syncs: $SYNC_SUCCESS"
    echo "  Failed syncs: $SYNC_FAIL"
    echo ""
    echo "Sync events (first 10):"
    grep -E "(WEIGHT SYNC|SYNC SUCCESS|SYNC FAILED)" "$TRAIN_LOG" | head -10 || echo "  No sync logs"
else
    echo "train.log not found"
fi

echo ""
echo "============================================="
echo "=== VERIFICATION 4: CONFIG DOCUMENTATION ==="
echo "============================================="
if [ -f "$TRAIN_LOG" ]; then
    CONFIG_COUNT=$(grep -c "CONFIGURATION" "$TRAIN_LOG" 2>/dev/null || echo "0")
    echo "Config sections found: $CONFIG_COUNT"
    echo ""
    echo "Verifier config logged:"
    grep -A15 "PHASE 2: VERIFIER TRAINING" "$TRAIN_LOG" | grep -E "(lr:|batch_size:|irm_mode:|irm_weight:)" | head -10 || echo "  No verifier config"
    echo ""
    echo "Prover config logged:"
    grep -A15 "PHASE 3: PROVER TRAINING" "$TRAIN_LOG" | grep -E "(lr:|batch_size:|group_size:|reward_strategy:)" | head -10 || echo "  No prover config"
else
    echo "train.log not found"
fi

echo ""
echo "============================================="
echo "=== VERIFICATION 5: LORA/HEAD REINIT ==="
echo "============================================="
if [ -f "$TRAIN_LOG" ]; then
    REINIT_COUNT=$(grep -c "VERIFIER HEAD REINITIALIZATION" "$TRAIN_LOG" 2>/dev/null || echo "0")
    echo "Head reinitialization events found: $REINIT_COUNT"
    echo ""
    echo "Reinit details:"
    grep -E "(LORA/HEAD REINIT|REINIT COMPLETE|PRE-reinit|POST-reinit)" "$TRAIN_LOG" | head -15 || echo "  No reinit logs"

    # Check that reinit happened for each round
    NUM_ROUNDS=$(grep -c "PHASE 2: VERIFIER TRAINING" "$TRAIN_LOG" 2>/dev/null || echo "0")
    echo ""
    echo "Verification:"
    echo "  Verifier training phases: $NUM_ROUNDS"
    echo "  Head reinitializations: $REINIT_COUNT"
    if [ "$REINIT_COUNT" -ge "$NUM_ROUNDS" ]; then
        echo "  ✓ Head reinit happening per round as expected"
    else
        echo "  !!! WARNING: Fewer reinits than training phases"
    fi
else
    echo "train.log not found"
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

#!/bin/bash
#SBATCH --job-name=rm-smoke
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm_%j_rm_smoke.log
#SBATCH --error=logs/slurm_%j_rm_smoke.log

# RM Smoke Test - Verify Chat Template Fix
# =========================================
# Quick validation run before full V3 experiments.
# Key things to verify in output:
# 1. "Using chat template for tokenization" appears
# 2. Sample logs show chat_template=True
# 3. Decoded prompts contain <|im_start|>user/<|im_start|>assistant
# 4. Initial loss ~0.69 (random chance for binary)
# 5. Accuracy starts climbing after a few steps

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

echo "========================================"
echo "RM Smoke Test - Chat Template Validation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================"

# Sync environment
echo "Syncing Python environment..."
uv sync

# Scratch storage for checkpoints
CKPT_DIR="${SCRATCHDIR:-/tmp}/ludic_rm_smoke_${SLURM_JOB_ID}"
mkdir -p "$CKPT_DIR"

echo ""
echo "Running smoke test with:"
echo "  - Model: Qwen/Qwen2.5-0.5B (small for fast iteration)"
echo "  - Samples: 100 train, 50 eval"
echo "  - Steps: 20 (enough to see learning signal)"
echo "  - Log samples: 3 (to inspect tokenization)"
echo ""

# Run with small model, few samples, few steps
# Key: --log-samples 3 will output tokenization details
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-0.5B \
    --hf-dataset stanfordnlp/SHP \
    --hf-split train \
    --limit 100 \
    --steps 20 \
    --batch-size 8 \
    --micro-token-budget 4096 \
    --max-seq-len 512 \
    --lr 1e-5 \
    --weight-decay 0.01 \
    --log-every 5 \
    --save-every 100 \
    --eval-split test \
    --eval-limit 50 \
    --eval-every 10 \
    --log-samples 3 \
    --output-dir "$CKPT_DIR"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Smoke Test Complete"
echo "========================================"

# Show the sample JSON files for inspection
echo ""
echo "=== TRAIN SAMPLES JSON (first 100 lines) ==="
head -100 "$CKPT_DIR/train_samples.json" 2>/dev/null || echo "No train_samples.json found"

echo ""
echo "=== Key Verification Checks ==="
echo ""

# Check 1: Chat template markers in samples
if grep -q "im_start" "$CKPT_DIR/train_samples.json" 2>/dev/null; then
    echo "[PASS] Chat template markers (<|im_start|>) found in tokenized samples"
else
    echo "[FAIL] Chat template markers NOT found - tokenization may be broken!"
fi

# Check 2: has_chat_template flags
if grep -q '"chosen_has_chat_template": true' "$CKPT_DIR/train_samples.json" 2>/dev/null; then
    echo "[PASS] has_chat_template=true in sample metadata"
else
    echo "[FAIL] has_chat_template=false or missing - chat template not applied!"
fi

echo ""
echo "Output directory: $CKPT_DIR"
echo "Completed at: $(date)"

exit $EXIT_CODE

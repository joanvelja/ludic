#!/bin/bash
#SBATCH --job-name=rm-v4-exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/slurm_%A_%a_rm_v4.log
#SBATCH --error=logs/slurm_%A_%a_rm_v4.log

# RM V4 Experiments - Scaling Up with Research-Informed Hypotheses
# =================================================================
#
# Experiment Matrix (12 experiments):
#
# | ID | Name    | Hypothesis                        | Config                           |
# |----|---------|-----------------------------------|----------------------------------|
# | 0  | V4-0    | Baseline + GPU tracking           | B2 config (r=4, LR=5e-5)         |
# | 1  | V4-1a   | rsLoRA r=4                        | B2 + --rslora                    |
# | 2  | V4-1b   | rsLoRA r=8                        | C0 (r=8, LR=1e-4) + --rslora     |
# | 3  | V4-2a   | High LR + ES                      | LR=1.5e-4, ES patience=3         |
# | 4  | V4-2b   | Higher LR + ES                    | LR=2e-4, ES patience=3           |
# | 5  | V4-3    | Extended training                 | B2 + 2000 steps                  |
# | 6  | V4-4    | Trainable norms                   | B2 + norm layers saved           |
# | 7  | V4-5    | Batch scaling                     | batch=32, LR=2.5e-5              |
# | 8  | V4-6a   | 7B model                          | B2 config on Qwen2.5-7B          |
# | 9  | V4-6b   | 7B + rsLoRA                       | B2 + rsLoRA on Qwen2.5-7B        |
# | 10 | V4-7    | Best combo r=4                    | B2 + rsLoRA + trainable norms    |
# | 11 | V4-8    | Best combo r=8                    | C0 + rsLoRA + ES + trainable norms|
#
# Key V3 findings we're building on:
# - B2 (r=4, LR=5e-5, all-linear): 74.15% best accuracy, most stable
# - C0 (r=8, LR=1e-4, all-linear): 74.55% peak, but overfits after step 500
#
# Submit: sbatch examples/reward_model/run_rm_v4_experiments.slurm
# Check:  squeue -u $USER
# Logs:   tail -f logs/slurm_*_rm_v4.log

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

echo "========================================"
echo "RM V4 Experiment Array"
echo "========================================"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================"

# Sync environment
echo "Syncing Python environment..."
uv sync

# Scratch storage for checkpoints
CKPT_BASE="${SCRATCHDIR:-/tmp}/ludic_rm_v4_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$CKPT_BASE"

# Common args for all experiments
# NOTE: max-seq-len=1024 and label-smoothing=0.05 match V3 config
# Previously V4 used 512 and 0.01 which caused 4% accuracy drop
COMMON_ARGS=(
    --hf-dataset stanfordnlp/SHP
    --hf-split train
    --eval-split test
    --eval-limit 1000
    --limit 5000
    --batch-size 16
    --micro-token-budget 8192
    --max-seq-len 1024
    --weight-decay 0.01
    --log-every 10
    --save-every 200
    --eval-every 100
    --label-smoothing 0.05
    --lr-scheduler cosine
    --warmup-steps 50
    --track-gpu
)

# LoRA target modules for Qwen all-linear (must match V3)
ALL_LINEAR="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# B2 baseline args (r=4, LR=5e-5, all-linear) - matches V3
B2_ARGS=(
    --model Qwen/Qwen2.5-3B
    --lora --lora-rank 4 --lora-alpha 32
    --lora-target-modules "$ALL_LINEAR"
    --lr 5e-5
    --steps 1000
)

# C0 baseline args (r=8, LR=1e-4, all-linear) - matches V3
C0_ARGS=(
    --model Qwen/Qwen2.5-3B
    --lora --lora-rank 8 --lora-alpha 32
    --lora-target-modules "$ALL_LINEAR"
    --lr 1e-4
    --steps 1000
)

# 7B model args
MODEL_7B="Qwen/Qwen2.5-7B"

# Normalization layers to keep trainable (Qwen architecture)
NORM_LAYERS="input_layernorm,post_attention_layernorm"

# Select experiment based on array task ID
case $SLURM_ARRAY_TASK_ID in
    0)
        # V4-0: Baseline + GPU tracking (B2 config)
        EXP_NAME="V4-0_baseline_gpu"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-0 - B2 baseline with GPU tracking"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --output-dir "$CKPT_DIR"
        ;;
    1)
        # V4-1a: rsLoRA r=4 (B2 + rsLoRA)
        EXP_NAME="V4-1a_rslora_r4"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-1a - B2 + rsLoRA (r=4)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --rslora \
            --output-dir "$CKPT_DIR"
        ;;
    2)
        # V4-1b: rsLoRA r=8 (C0 + rsLoRA)
        EXP_NAME="V4-1b_rslora_r8"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-1b - C0 + rsLoRA (r=8)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${C0_ARGS[@]}" \
            --rslora \
            --output-dir "$CKPT_DIR"
        ;;
    3)
        # V4-2a: High LR + Early Stopping (LR=1.5e-4)
        EXP_NAME="V4-2a_highLR_1.5e-4"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-2a - High LR (1.5e-4) + Early Stopping"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            --model Qwen/Qwen2.5-3B \
            --lora --lora-rank 8 --lora-alpha 32 \
            --lora-target-modules "$ALL_LINEAR" \
            --lr 1.5e-4 \
            --steps 1000 \
            --early-stopping-patience 3 \
            --early-stopping-min-delta 0.001 \
            --output-dir "$CKPT_DIR"
        ;;
    4)
        # V4-2b: Higher LR + Early Stopping (LR=2e-4)
        EXP_NAME="V4-2b_highLR_2e-4"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-2b - Higher LR (2e-4) + Early Stopping"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            --model Qwen/Qwen2.5-3B \
            --lora --lora-rank 8 --lora-alpha 32 \
            --lora-target-modules "$ALL_LINEAR" \
            --lr 2e-4 \
            --steps 1000 \
            --early-stopping-patience 3 \
            --early-stopping-min-delta 0.001 \
            --output-dir "$CKPT_DIR"
        ;;
    5)
        # V4-3: Extended training (B2 + 2000 steps)
        EXP_NAME="V4-3_extended_2000"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-3 - B2 extended training (2000 steps)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            --model Qwen/Qwen2.5-3B \
            --lora --lora-rank 4 --lora-alpha 32 \
            --lora-target-modules "$ALL_LINEAR" \
            --lr 5e-5 \
            --steps 2000 \
            --output-dir "$CKPT_DIR"
        ;;
    6)
        # V4-4: Trainable normalization layers (B2 + modules_to_save)
        EXP_NAME="V4-4_trainable_norms"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-4 - B2 + trainable normalization layers"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --lora-modules-to-save "$NORM_LAYERS" \
            --output-dir "$CKPT_DIR"
        ;;
    7)
        # V4-5: Batch scaling (batch=32, LR=2.5e-5)
        EXP_NAME="V4-5_batch32"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-5 - Batch scaling (32 pairs, LR=2.5e-5)"
        # Override batch-size from COMMON_ARGS, match V3 config otherwise
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            --hf-dataset stanfordnlp/SHP \
            --hf-split train \
            --eval-split test \
            --eval-limit 1000 \
            --limit 5000 \
            --batch-size 32 \
            --micro-token-budget 32768 \
            --max-seq-len 1024 \
            --weight-decay 0.01 \
            --log-every 10 \
            --save-every 200 \
            --eval-every 100 \
            --label-smoothing 0.05 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --track-gpu \
            --model Qwen/Qwen2.5-3B \
            --lora --lora-rank 4 --lora-alpha 32 \
            --lora-target-modules "$ALL_LINEAR" \
            --lr 2.5e-5 \
            --steps 1000 \
            --output-dir "$CKPT_DIR"
        ;;
    8)
        # V4-6a: 7B model (B2 config)
        EXP_NAME="V4-6a_7B_baseline"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-6a - Qwen2.5-7B with B2 config"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            --model "$MODEL_7B" \
            --lora --lora-rank 4 --lora-alpha 32 \
            --lora-target-modules "$ALL_LINEAR" \
            --lr 5e-5 \
            --steps 1000 \
            --output-dir "$CKPT_DIR"
        ;;
    9)
        # V4-6b: 7B + rsLoRA
        EXP_NAME="V4-6b_7B_rslora"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-6b - Qwen2.5-7B with B2 config + rsLoRA"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            --model "$MODEL_7B" \
            --lora --lora-rank 4 --lora-alpha 32 \
            --lora-target-modules "$ALL_LINEAR" \
            --rslora \
            --lr 5e-5 \
            --steps 1000 \
            --output-dir "$CKPT_DIR"
        ;;
    10)
        # V4-7: Best combo r=4 (B2 + rsLoRA + trainable norms)
        EXP_NAME="V4-7_combo_r4"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-7 - B2 + rsLoRA + trainable norms"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --rslora \
            --lora-modules-to-save "$NORM_LAYERS" \
            --output-dir "$CKPT_DIR"
        ;;
    11)
        # V4-8: Best combo r=8 (C0 + rsLoRA + ES + trainable norms)
        EXP_NAME="V4-8_combo_r8"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: V4-8 - C0 + rsLoRA + ES + trainable norms"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${C0_ARGS[@]}" \
            --rslora \
            --lora-modules-to-save "$NORM_LAYERS" \
            --early-stopping-patience 3 \
            --early-stopping-min-delta 0.001 \
            --output-dir "$CKPT_DIR"
        ;;
    *)
        echo "Unknown task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Experiment Complete: $EXP_NAME"
echo "========================================"
echo "Exit code: $EXIT_CODE"
echo "Output directory: $CKPT_DIR"
echo "Completed at: $(date)"
echo ""

# Copy metrics to persistent storage for analysis
if [ -f "$CKPT_DIR/metrics.csv" ]; then
    RESULTS_DIR="/home/u5ds/joanv.u5ds/ludic/results/v4"
    mkdir -p "$RESULTS_DIR"
    cp "$CKPT_DIR/metrics.csv" "$RESULTS_DIR/${EXP_NAME}_metrics.csv"
    echo "Metrics copied to: $RESULTS_DIR/${EXP_NAME}_metrics.csv"
fi

exit $EXIT_CODE

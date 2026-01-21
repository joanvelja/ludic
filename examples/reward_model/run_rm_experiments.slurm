#!/bin/bash
#SBATCH --job-name=rm-v2-exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_%A_%a_rm_exp.log
#SBATCH --error=logs/slurm_%A_%a_rm_exp.log
#SBATCH --array=0-9

# RM Trainer V2 Experiment Array
# Runs all experiments from PLAN-v2.md:
#   0: Baseline (full config)
#   1: A1 - No improvements
#   2: A2 - Scheduler only
#   3: A3 - Scheduler + Label smoothing
#   4: A4 - Full config (early stopping)
#   5: LoRA r8
#   6: LoRA r16
#   7: LR sweep 1e-6
#   8: LR sweep 5e-6
#   9: LR sweep 1e-5

set -e

# Change to project root
cd /home/u5ds/joanv.u5ds/ludic

# Create logs directory if it doesn't exist
mkdir -p logs

# Use scratch storage for checkpoints (5 TiB quota vs 100 GiB $HOME)
CKPT_BASE="${SCRATCHDIR:-/tmp}/ludic_rm_checkpoints"
mkdir -p "$CKPT_BASE"

# Reduce logging verbosity
export TRANSFORMERS_VERBOSITY=warning
export PEFT_VERBOSITY=warning

# Sync environment
uv sync

# Common parameters
MODEL="Qwen/Qwen2.5-3B"  # Reduced from 7B to fit in single GPU memory
DATASET="stanfordnlp/SHP"
SPLIT="train"
LIMIT=5000
STEPS=1000
BATCH_SIZE=16
MICRO_TOKEN_BUDGET=8192
MAX_SEQ_LEN=1024
WEIGHT_DECAY=0.01
LOG_EVERY=10
SAVE_EVERY=10000  # Higher than total steps = no periodic saves (avoids quota issues)
EVAL_SPLIT="test"
EVAL_LIMIT=1000
EVAL_EVERY=100

# Base command
BASE_CMD="uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model $MODEL \
    --hf-dataset $DATASET \
    --hf-split $SPLIT \
    --limit $LIMIT \
    --steps $STEPS \
    --batch-size $BATCH_SIZE \
    --seed 42 \
    --micro-token-budget $MICRO_TOKEN_BUDGET \
    --max-seq-len $MAX_SEQ_LEN \
    --weight-decay $WEIGHT_DECAY \
    --log-every $LOG_EVERY \
    --save-every $SAVE_EVERY \
    --eval-split $EVAL_SPLIT \
    --eval-limit $EVAL_LIMIT \
    --eval-every $EVAL_EVERY"

echo "========================================"
echo "RM V2 Experiment Array - Task $SLURM_ARRAY_TASK_ID"
echo "========================================"

case $SLURM_ARRAY_TASK_ID in
    0)
        # Experiment 1: Baseline (full config)
        echo "Running: Baseline (full config)"
        $BASE_CMD \
            --lr 5e-6 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --early-stopping-patience 5 \
            --profile-memory \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_baseline"
        ;;

    1)
        # Ablation A1: No improvements
        echo "Running: Ablation A1 - No improvements"
        $BASE_CMD \
            --lr 5e-6 \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_A1_none"
        ;;

    2)
        # Ablation A2: Scheduler only
        echo "Running: Ablation A2 - Scheduler only"
        $BASE_CMD \
            --lr 5e-6 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_A2_scheduler"
        ;;

    3)
        # Ablation A3: Scheduler + Label smoothing
        echo "Running: Ablation A3 - Scheduler + Label smoothing"
        $BASE_CMD \
            --lr 5e-6 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_A3_smoothing"
        ;;

    4)
        # Ablation A4: Full config (early stopping)
        echo "Running: Ablation A4 - Full config (with early stopping)"
        $BASE_CMD \
            --lr 5e-6 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --early-stopping-patience 5 \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_A4_full"
        ;;

    5)
        # LoRA r8 - per Thinking Machines "LoRA Without Regret":
        # - LR 10x higher than full FT (5e-5 vs 5e-6)
        # - alpha=32 standard parametrization (independent of rank)
        # - all-linear targets MLP + attention (critical for LoRA performance)
        echo "Running: LoRA rank 8"
        $BASE_CMD \
            --lr 5e-5 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --early-stopping-patience 5 \
            --lora --lora-rank 8 --lora-alpha 32 \
            --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
            --profile-memory \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_lora_r8"
        ;;

    6)
        # LoRA r16 - per Thinking Machines "LoRA Without Regret":
        # - LR 10x higher than full FT (5e-5 vs 5e-6)
        # - alpha=32 standard parametrization
        # - all-linear targets MLP + attention
        echo "Running: LoRA rank 16"
        $BASE_CMD \
            --lr 5e-5 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --early-stopping-patience 5 \
            --lora --lora-rank 16 --lora-alpha 32 \
            --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
            --profile-memory \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_lora_r16"
        ;;

    7)
        # LR sweep: 1e-6
        echo "Running: LR sweep - 1e-6"
        $BASE_CMD \
            --lr 1e-6 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_lr_1e-6"
        ;;

    8)
        # LR sweep: 5e-6
        echo "Running: LR sweep - 5e-6"
        $BASE_CMD \
            --lr 5e-6 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_lr_5e-6"
        ;;

    9)
        # LR sweep: 1e-5
        echo "Running: LR sweep - 1e-5"
        $BASE_CMD \
            --lr 1e-5 \
            --lr-scheduler cosine \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --output-dir "$CKPT_BASE/checkpoints_rm_v2_lr_1e-5"
        ;;

    *)
        echo "Unknown task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "========================================"
echo "Experiment $SLURM_ARRAY_TASK_ID complete!"
echo "========================================"

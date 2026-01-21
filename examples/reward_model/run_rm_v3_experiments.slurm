#!/bin/bash
#SBATCH --job-name=rm-v3-exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_%A_%a_rm_v3.log
#SBATCH --error=logs/slurm_%A_%a_rm_v3.log
#SBATCH --array=0-19

# RM Trainer V3 Experiment Array - Clean Disambiguation Study
# ============================================================
#
# Group A: Differential LR ablations (isolate backbone vs head LR effect)
#   0: A0 - Reference baseline (V2 best: scheduler + smoothing, NO differential LR)
#   1: A1 - + Differential LR (0.1x backbone)
#   2: A2 - + Differential LR + head-only weight decay
#   3: A3 - + Differential LR (0.05x backbone, more conservative)
#
# Group B: LoRA rank ablations (isolate effect of rank and target modules)
#   4: B0 - LoRA r=2, attention only (q,v) - ultra minimal
#   5: B1 - LoRA r=4, attention only (q,v)
#   6: B2 - LoRA r=4, all-linear
#   7: B3 - LoRA r=8, attention only (q,v) - compare vs all-linear from V2
#
# Group C: LoRA LR ablations (isolate optimal LR for LoRA)
#   8: C0 - LoRA r=8 all-linear, LR=1e-4 (20x base)
#   9: C1 - LoRA r=8 all-linear, LR=2e-5 (4x base)
#
# Group D: Batch size ablations (isolate gradient noise effect)
#  10: D0 - batch_size=32 (2x baseline)
#  11: D1 - batch_size=64 (4x baseline)
#
# Group E: Scheduler ablations (isolate scheduler effect)
#  12: E0 - linear scheduler (vs cosine)
#  13: E1 - constant with warmup
#
# Group F: Combined best practices
#  14: F0 - Differential LR + LoRA r=4 attention only
#  15: F1 - Differential LR + batch=32 + linear scheduler
#  16: F2 - LoRA r=4 attention + larger batch (32)
#  17: F3 - All bells: Diff LR + head-only WD + LoRA r=4 attn
#
# Group G: Sanity checks
#  18: G0 - No weight decay at all
#  19: G1 - Higher weight decay (0.1)

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

# Scratch storage for checkpoints
CKPT_BASE="${SCRATCHDIR:-/tmp}/ludic_rm_v3_checkpoints"
mkdir -p "$CKPT_BASE"

# Reduce logging verbosity
export TRANSFORMERS_VERBOSITY=warning
export PEFT_VERBOSITY=warning

uv sync

# Common parameters
MODEL="Qwen/Qwen2.5-3B"
DATASET="stanfordnlp/SHP"
SPLIT="train"
LIMIT=5000
STEPS=1000
BATCH_SIZE=16
MICRO_TOKEN_BUDGET=8192
MAX_SEQ_LEN=1024
WEIGHT_DECAY=0.01
LOG_EVERY=10
SAVE_EVERY=10000
EVAL_SPLIT="test"
EVAL_LIMIT=1000
EVAL_EVERY=100

# Base command (shared across all experiments)
BASE_CMD="uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model $MODEL \
    --hf-dataset $DATASET \
    --hf-split $SPLIT \
    --limit $LIMIT \
    --steps $STEPS \
    --seed 42 \
    --micro-token-budget $MICRO_TOKEN_BUDGET \
    --max-seq-len $MAX_SEQ_LEN \
    --log-every $LOG_EVERY \
    --save-every $SAVE_EVERY \
    --eval-split $EVAL_SPLIT \
    --eval-limit $EVAL_LIMIT \
    --eval-every $EVAL_EVERY"

# Common good defaults (scheduler + smoothing from V2)
GOOD_DEFAULTS="--lr-scheduler cosine --warmup-steps 50 --label-smoothing 0.05"

# LoRA module targets
ATTN_ONLY="q_proj,v_proj"
ALL_LINEAR="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

echo "========================================"
echo "RM V3 Experiment Array - Task $SLURM_ARRAY_TASK_ID"
echo "========================================"

case $SLURM_ARRAY_TASK_ID in
    # =========================================================================
    # GROUP A: Differential LR ablations
    # =========================================================================
    0)
        echo "A0: Reference baseline (scheduler + smoothing, NO differential LR)"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-6 \
            $GOOD_DEFAULTS \
            --output-dir "$CKPT_BASE/A0_baseline"
        ;;

    1)
        echo "A1: + Differential LR (0.1x backbone)"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-6 \
            $GOOD_DEFAULTS \
            --differential-lr \
            --backbone-lr-mult 0.1 \
            --output-dir "$CKPT_BASE/A1_diff_lr"
        ;;

    2)
        echo "A2: + Differential LR + head-only weight decay"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-6 \
            $GOOD_DEFAULTS \
            --differential-lr \
            --backbone-lr-mult 0.1 \
            --head-only-weight-decay \
            --output-dir "$CKPT_BASE/A2_diff_lr_head_wd"
        ;;

    3)
        echo "A3: + Differential LR (0.05x backbone, more conservative)"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-6 \
            $GOOD_DEFAULTS \
            --differential-lr \
            --backbone-lr-mult 0.05 \
            --output-dir "$CKPT_BASE/A3_diff_lr_005x"
        ;;

    # =========================================================================
    # GROUP B: LoRA rank ablations (all use 10x LR = 5e-5)
    # =========================================================================
    4)
        echo "B0: LoRA r=2, attention only (ultra minimal)"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-5 \
            $GOOD_DEFAULTS \
            --lora --lora-rank 2 --lora-alpha 32 \
            --lora-target-modules "$ATTN_ONLY" \
            --output-dir "$CKPT_BASE/B0_lora_r2_attn"
        ;;

    5)
        echo "B1: LoRA r=4, attention only"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-5 \
            $GOOD_DEFAULTS \
            --lora --lora-rank 4 --lora-alpha 32 \
            --lora-target-modules "$ATTN_ONLY" \
            --output-dir "$CKPT_BASE/B1_lora_r4_attn"
        ;;

    6)
        echo "B2: LoRA r=4, all-linear"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-5 \
            $GOOD_DEFAULTS \
            --lora --lora-rank 4 --lora-alpha 32 \
            --lora-target-modules "$ALL_LINEAR" \
            --output-dir "$CKPT_BASE/B2_lora_r4_all"
        ;;

    7)
        echo "B3: LoRA r=8, attention only (compare vs V2 all-linear)"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-5 \
            $GOOD_DEFAULTS \
            --lora --lora-rank 8 --lora-alpha 32 \
            --lora-target-modules "$ATTN_ONLY" \
            --output-dir "$CKPT_BASE/B3_lora_r8_attn"
        ;;

    # =========================================================================
    # GROUP C: LoRA LR ablations (using r=8 all-linear as base)
    # =========================================================================
    8)
        echo "C0: LoRA r=8 all-linear, LR=1e-4 (20x base)"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 1e-4 \
            $GOOD_DEFAULTS \
            --lora --lora-rank 8 --lora-alpha 32 \
            --lora-target-modules "$ALL_LINEAR" \
            --output-dir "$CKPT_BASE/C0_lora_lr_1e-4"
        ;;

    9)
        echo "C1: LoRA r=8 all-linear, LR=2e-5 (4x base)"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 2e-5 \
            $GOOD_DEFAULTS \
            --lora --lora-rank 8 --lora-alpha 32 \
            --lora-target-modules "$ALL_LINEAR" \
            --output-dir "$CKPT_BASE/C1_lora_lr_2e-5"
        ;;

    # =========================================================================
    # GROUP D: Batch size ablations (using baseline config)
    # =========================================================================
    10)
        echo "D0: batch_size=32 (2x baseline)"
        $BASE_CMD \
            --batch-size 32 \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-6 \
            $GOOD_DEFAULTS \
            --output-dir "$CKPT_BASE/D0_batch32"
        ;;

    11)
        echo "D1: batch_size=64 (4x baseline)"
        $BASE_CMD \
            --batch-size 64 \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-6 \
            $GOOD_DEFAULTS \
            --output-dir "$CKPT_BASE/D1_batch64"
        ;;

    # =========================================================================
    # GROUP E: Scheduler ablations
    # =========================================================================
    12)
        echo "E0: linear scheduler (vs cosine)"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-6 \
            --lr-scheduler linear \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --output-dir "$CKPT_BASE/E0_linear_sched"
        ;;

    13)
        echo "E1: constant with warmup"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-6 \
            --lr-scheduler constant_with_warmup \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --output-dir "$CKPT_BASE/E1_const_warmup"
        ;;

    # =========================================================================
    # GROUP F: Combined best practices
    # =========================================================================
    14)
        echo "F0: Differential LR + LoRA r=4 attention only"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-5 \
            $GOOD_DEFAULTS \
            --differential-lr \
            --backbone-lr-mult 0.1 \
            --lora --lora-rank 4 --lora-alpha 32 \
            --lora-target-modules "$ATTN_ONLY" \
            --output-dir "$CKPT_BASE/F0_diff_lr_lora_r4"
        ;;

    15)
        echo "F1: Differential LR + batch=32 + linear scheduler"
        $BASE_CMD \
            --batch-size 32 \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-6 \
            --lr-scheduler linear \
            --warmup-steps 50 \
            --label-smoothing 0.05 \
            --differential-lr \
            --backbone-lr-mult 0.1 \
            --output-dir "$CKPT_BASE/F1_diff_lr_batch32_linear"
        ;;

    16)
        echo "F2: LoRA r=4 attention + larger batch (32)"
        $BASE_CMD \
            --batch-size 32 \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-5 \
            $GOOD_DEFAULTS \
            --lora --lora-rank 4 --lora-alpha 32 \
            --lora-target-modules "$ATTN_ONLY" \
            --output-dir "$CKPT_BASE/F2_lora_r4_batch32"
        ;;

    17)
        echo "F3: All bells: Diff LR + head-only WD + LoRA r=4 attn"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay $WEIGHT_DECAY \
            --lr 5e-5 \
            $GOOD_DEFAULTS \
            --differential-lr \
            --backbone-lr-mult 0.1 \
            --head-only-weight-decay \
            --lora --lora-rank 4 --lora-alpha 32 \
            --lora-target-modules "$ATTN_ONLY" \
            --output-dir "$CKPT_BASE/F3_all_bells"
        ;;

    # =========================================================================
    # GROUP G: Sanity checks (weight decay extremes)
    # =========================================================================
    18)
        echo "G0: No weight decay at all"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay 0.0 \
            --lr 5e-6 \
            $GOOD_DEFAULTS \
            --output-dir "$CKPT_BASE/G0_no_wd"
        ;;

    19)
        echo "G1: Higher weight decay (0.1)"
        $BASE_CMD \
            --batch-size $BATCH_SIZE \
            --weight-decay 0.1 \
            --lr 5e-6 \
            $GOOD_DEFAULTS \
            --output-dir "$CKPT_BASE/G1_high_wd"
        ;;

    *)
        echo "Unknown task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "========================================"
echo "Experiment $SLURM_ARRAY_TASK_ID complete!"
echo "========================================"

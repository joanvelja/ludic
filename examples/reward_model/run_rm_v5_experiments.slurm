#!/bin/bash
#SBATCH --job-name=rm-v5-exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/slurm_%A_%a_rm_v5.log
#SBATCH --error=logs/slurm_%A_%a_rm_v5.log

# RM V5 Experiments - Diagnostic-First Reward Model Training
# ===========================================================
#
# Phase 1: Diagnostic Experiments (0-4) - MUST PASS before Phase 2
# Phase 2: Regularization Comparison (5-9)
# Phase 3: Temperature & Early Stopping (10-12)
# Phase 4: Ablations / Best Combinations (13-14)
#
# Experiment Matrix (15 experiments):
#
# | ID | Name               | Purpose                              | Pass Criterion           |
# |----|--------------------|--------------------------------------|--------------------------|
# | 0  | D1_label_shuffle   | 50% label flip -> expect ~50% acc    | 48-52% acc, loss ~0.693  |
# | 1  | D2_prompt_only     | Empty responses -> expect ~50% acc   | 48-52% acc               |
# | 2  | D3_random_negatives| Cross-prompt rejected -> high acc    | 80-95% acc               |
# | 3  | D4_length_matched  | Filter |delta_len|<50 -> within 3%   | Within 3% of baseline    |
# | 4  | C1_baseline        | V4-0 reproduction                    | 73-75% (match V4)        |
# | 5  | M1a_margin_reg     | Margin L2 lambda=0.001               | delta_std decreases      |
# | 6  | M1b_margin_reg     | Margin L2 lambda=0.01                | delta_std decreases more |
# | 7  | S1a_score_reg      | Score L2 lambda=0.001                | abs_max < 10             |
# | 8  | S1b_score_reg      | Score L2 lambda=0.01                 | abs_max < 5              |
# | 9  | S1c_score_reg      | Score L2 lambda=0.1                  | Expect acc drop          |
# | 10 | S2a_beta_0.5       | Softer sigmoid (beta=0.5)            | Slower delta growth      |
# | 11 | S2b_beta_2.0       | Sharper sigmoid (beta=2.0)           | Faster delta growth      |
# | 12 | S3_early_stop_acc  | Early stop on accuracy               | Stops at peak acc        |
# | 13 | A1_score_reg+beta  | score_reg=0.01 + beta=0.5            | Combined stability       |
# | 14 | A2_score_reg+es    | score_reg=0.01 + early_stop          | Best combo?              |
#
# Key V4 findings we're diagnosing:
# - 74% accuracy ceiling: Is this bugs/leakage or real BT unbounded optimum?
# - Margin grows unbounded without regularization
#
# Submit Phase 1 ONLY:  sbatch --array=0-4 examples/reward_model/run_rm_v5_experiments.slurm
# Submit Phase 2+ (after diagnostics pass): sbatch --array=5-14 examples/reward_model/run_rm_v5_experiments.slurm
# Submit all: sbatch examples/reward_model/run_rm_v5_experiments.slurm
# Check:  squeue -u $USER
# Logs:   tail -f logs/slurm_*_rm_v5.log

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

echo "========================================"
echo "RM V5 Experiment Array"
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
CKPT_BASE="${SCRATCHDIR:-/tmp}/ludic_rm_v5_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$CKPT_BASE"

# Common args for all experiments (MUST match V3/V4 for ceteris paribus)
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

# LoRA target modules for Qwen all-linear (MUST match V3/V4)
ALL_LINEAR="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# B2 baseline args (r=4, LR=5e-5, all-linear) - matches V3/V4
B2_ARGS=(
    --model Qwen/Qwen2.5-3B
    --lora --lora-rank 4 --lora-alpha 32
    --lora-target-modules "$ALL_LINEAR"
    --lr 5e-5
    --steps 1000
)

# Select experiment based on array task ID
case $SLURM_ARRAY_TASK_ID in
    # =========================================================================
    # PHASE 1: DIAGNOSTICS (0-4) - MUST PASS BEFORE PHASE 2
    # =========================================================================
    0)
        # D1: Label Shuffle - 50% random label flip
        # Hypothesis: If model achieves >55% with corrupted labels, there's leakage
        # Pass criterion: 48-52% accuracy, loss ~0.693
        EXP_NAME="D1_label_shuffle"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: D1 - Label shuffle (50% flip)"
        echo "Expected: ~50% accuracy, loss ~0.693"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --diagnostic-mode label_shuffle \
            --diagnostic-shuffle-rate 0.5 \
            --output-dir "$CKPT_DIR"
        ;;
    1)
        # D2: Prompt-Only - Empty responses
        # Hypothesis: If model learns anything with empty completions, there's prompt leakage
        # Pass criterion: 48-52% accuracy
        EXP_NAME="D2_prompt_only"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: D2 - Prompt only (empty completions)"
        echo "Expected: ~50% accuracy"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --diagnostic-mode prompt_only \
            --output-dir "$CKPT_DIR"
        ;;
    2)
        # D3: Random Negatives - Cross-prompt rejected responses
        # Hypothesis: Easy task (plausibility ceiling), should achieve 80-95%
        # Pass criterion: 80-95% accuracy (model learns plausibility)
        EXP_NAME="D3_random_negatives"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: D3 - Random negatives (cross-prompt rejected)"
        echo "Expected: 80-95% accuracy (plausibility ceiling)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --diagnostic-mode random_negatives \
            --output-dir "$CKPT_DIR"
        ;;
    3)
        # D4: Length-Matched - Filter to |len(chosen)-len(rejected)| < 50
        # Hypothesis: If accuracy drops significantly, model was exploiting length bias
        # Pass criterion: Within 3% of baseline accuracy
        EXP_NAME="D4_length_matched"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: D4 - Length matched (|delta_len| < 50)"
        echo "Expected: Within 3% of baseline accuracy"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --diagnostic-mode length_matched \
            --diagnostic-length-threshold 50 \
            --output-dir "$CKPT_DIR"
        ;;
    4)
        # C1: Baseline - V4-0 reproduction
        # Hypothesis: Match V4 baseline accuracy (74.15%)
        # Pass criterion: 73-75% accuracy
        EXP_NAME="C1_baseline"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: C1 - Baseline (V4-0 reproduction)"
        echo "Expected: 73-75% accuracy"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --output-dir "$CKPT_DIR"
        ;;

    # =========================================================================
    # PHASE 2: REGULARIZATION COMPARISON (5-9)
    # =========================================================================
    5)
        # M1a: Margin L2 regularization lambda=0.001
        # Hypothesis: Penalizes large gaps between chosen/rejected scores
        # Pass criterion: delta_std decreases, accuracy within 2% of baseline
        EXP_NAME="M1a_margin_reg_0.001"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: M1a - Margin L2 regularization (lambda=0.001)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --regularization-lambda 0.001 \
            --output-dir "$CKPT_DIR"
        ;;
    6)
        # M1b: Margin L2 regularization lambda=0.01
        # Hypothesis: Stronger gap penalty
        # Pass criterion: delta_std decreases more, accuracy within 2% of baseline
        EXP_NAME="M1b_margin_reg_0.01"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: M1b - Margin L2 regularization (lambda=0.01)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --regularization-lambda 0.01 \
            --output-dir "$CKPT_DIR"
        ;;
    7)
        # S1a: Score L2 regularization lambda=0.001
        # Hypothesis: Bounds absolute reward magnitude (not just margin)
        # Pass criterion: abs_max < 10, accuracy within 2% of baseline
        EXP_NAME="S1a_score_reg_0.001"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: S1a - Score L2 regularization (lambda=0.001)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --score-regularization-lambda 0.001 \
            --output-dir "$CKPT_DIR"
        ;;
    8)
        # S1b: Score L2 regularization lambda=0.01
        # Hypothesis: Stronger absolute bound
        # Pass criterion: abs_max < 5, accuracy within 2% of baseline
        EXP_NAME="S1b_score_reg_0.01"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: S1b - Score L2 regularization (lambda=0.01)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --score-regularization-lambda 0.01 \
            --output-dir "$CKPT_DIR"
        ;;
    9)
        # S1c: Score L2 regularization lambda=0.1
        # Hypothesis: Very strong bound may hurt accuracy
        # Pass criterion: Test sensitivity - expect accuracy drop
        EXP_NAME="S1c_score_reg_0.1"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: S1c - Score L2 regularization (lambda=0.1)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --score-regularization-lambda 0.1 \
            --output-dir "$CKPT_DIR"
        ;;

    # =========================================================================
    # PHASE 3: TEMPERATURE & EARLY STOPPING (10-12)
    # =========================================================================
    10)
        # S2a: Beta=0.5 (softer sigmoid)
        # Hypothesis: Reduces margin growth rate by softening predictions
        # Pass criterion: Slower delta growth rate
        EXP_NAME="S2a_beta_0.5"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: S2a - Beta=0.5 (softer sigmoid)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --beta 0.5 \
            --output-dir "$CKPT_DIR"
        ;;
    11)
        # S2b: Beta=2.0 (sharper sigmoid)
        # Hypothesis: Control experiment - faster margin growth
        # Pass criterion: Faster delta growth (control)
        EXP_NAME="S2b_beta_2.0"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: S2b - Beta=2.0 (sharper sigmoid)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --beta 2.0 \
            --output-dir "$CKPT_DIR"
        ;;
    12)
        # S3: Early stopping on accuracy
        # Hypothesis: Catches true optimum before overfitting
        # Pass criterion: Stops at highest accuracy point
        EXP_NAME="S3_early_stop_acc"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: S3 - Early stopping on accuracy (patience=5)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --early-stopping-patience 5 \
            --early-stopping-min-delta 0.001 \
            --output-dir "$CKPT_DIR"
        ;;

    # =========================================================================
    # PHASE 4: ABLATIONS / BEST COMBINATIONS (13-14)
    # =========================================================================
    13)
        # A1: Score regularization + softer beta
        # Hypothesis: Combined stabilization approach
        EXP_NAME="A1_score_reg_plus_beta"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: A1 - Score reg (0.01) + Beta (0.5)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --score-regularization-lambda 0.01 \
            --beta 0.5 \
            --output-dir "$CKPT_DIR"
        ;;
    14)
        # A2: Score regularization + early stopping
        # Hypothesis: Best combo for stability + peak accuracy
        EXP_NAME="A2_score_reg_plus_es"
        CKPT_DIR="$CKPT_BASE/$EXP_NAME"
        echo "Running: A2 - Score reg (0.01) + Early stopping (patience=5)"
        uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
            "${COMMON_ARGS[@]}" \
            "${B2_ARGS[@]}" \
            --score-regularization-lambda 0.01 \
            --early-stopping-patience 5 \
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
    RESULTS_DIR="/home/u5ds/joanv.u5ds/ludic/results/v5"
    mkdir -p "$RESULTS_DIR"
    cp "$CKPT_DIR/metrics.csv" "$RESULTS_DIR/${EXP_NAME}_metrics.csv"
    echo "Metrics copied to: $RESULTS_DIR/${EXP_NAME}_metrics.csv"
fi

exit $EXIT_CODE

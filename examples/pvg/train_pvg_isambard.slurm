#!/bin/bash
#SBATCH --job-name=pvg-train
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j_init.log
#SBATCH --error=logs/slurm_%j_init.log

# =============================================================================
# PVG (Prover-Verifier Game) Training on Isambard-AI
# =============================================================================
#
# This script runs multi-round PVG training:
#   - GPU 0: vLLM servers (prover + verifier) sharing GPU memory
#   - GPU 1-3: FSDP training (prover policy gradient)
#
# The training loop alternates between:
#   1. Data minting: Generate sneaky samples
#   2. Verifier training: Bradley-Terry + IRM on preference pairs
#   3. Prover training: Policy gradient with verifier reward
#
# Prerequisites:
#   1. .env file with HF_TOKEN
#   2. podman-hpc image: podman-hpc pull python:3.11-slim
#
# Usage:
#   # Default config (5 rounds, APPS dataset)
#   sbatch examples/pvg/train_pvg_isambard.slurm
#
#   # Custom rounds and model
#   NUM_ROUNDS=10 MODEL=Qwen/Qwen2.5-7B-Instruct sbatch train_pvg_isambard.slurm
#
#   # Adjust batch sizes for memory
#   BATCH_SIZE=2 MICRO_TOKEN_BUDGET=16384 sbatch train_pvg_isambard.slurm
#
# =============================================================================

set -e

# Create structured log directory
LOG_DATE=$(date +%Y-%m-%d)
LOG_TIME=$(date +%H-%M-%S)
LOG_DIR="logs/pvg/${LOG_DATE}/${LOG_TIME}_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

exec > >(tee -a "${LOG_DIR}/slurm.log") 2>&1

if [ -f "logs/slurm_${SLURM_JOB_ID}_init.log" ]; then
    cat "logs/slurm_${SLURM_JOB_ID}_init.log" >> "${LOG_DIR}/slurm.log"
    rm -f "logs/slurm_${SLURM_JOB_ID}_init.log"
fi

echo "Log directory: $LOG_DIR"

# =============================================================================
# Configuration (override via environment variables)
# =============================================================================

# Model configuration
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
VERIFIER_MODEL="${VERIFIER_MODEL:-$MODEL}"  # Default: same as prover
PROVER_MODEL="${PROVER_MODEL:-$MODEL}"

# Dataset
DATASET="${DATASET:-apps}"
LIMIT="${LIMIT:-500}"
SPLIT_RATIO="${SPLIT_RATIO:-0.5}"

# Training rounds
NUM_ROUNDS="${NUM_ROUNDS:-5}"
VERIFIER_STEPS="${VERIFIER_STEPS:-500}"
PROVER_STEPS="${PROVER_STEPS:-1000}"

# Batch sizing (based on empirical memory measurements)
# ======================================================
# Effective batch = BATCH_SIZE × GROUP_SIZE
# The trainer splits into micro-batches based on MICRO_TOKEN_BUDGET
#
# Memory per GPU with FSDP (3 GPUs), seq=4096, empirical:
#   Model         Full FT          LoRA-64
#   ─────────────────────────────────────────
#   0.5B          ~18GB/batch4     ~15GB/batch4
#   1.5B          ~28GB/batch4     ~19GB/batch4
#   3B            ~42GB/batch4     ~25GB/batch4
#   7B            ~78GB/batch4     ~37GB/batch4
#   8B            ~82GB/batch4     ~38GB/batch4
#   14B           ~131GB/batch4    ~52GB/batch4
#
# GB200 has 192GB, so with 15% safety margin = 163GB usable
#
# Recommended batch sizes (full fine-tuning, seq=4096):
#   0.5B: batch=32  →  ~58GB/GPU
#   1.5B: batch=16  →  ~65GB/GPU
#   3B:   batch=8   →  ~65GB/GPU
#   7B:   batch=4   →  ~78GB/GPU
#   14B:  batch=2   →  ~100GB/GPU
#
BATCH_SIZE="${BATCH_SIZE:-4}"       # Unique problems per macro-step
GROUP_SIZE="${GROUP_SIZE:-4}"       # Generations per problem (GRPO)
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"  # Max tokens per sample
MICRO_TOKEN_BUDGET="${MICRO_TOKEN_BUDGET:-16384}"  # ~4 samples per micro-batch

# IRM configuration
IRM_MODE="${IRM_MODE:-vrex}"  # vrex, balanced, both, none
IRM_BETA="${IRM_BETA:-1.0}"

# Reward configuration
REWARD_STRATEGY="${REWARD_STRATEGY:-composite}"

# Stopping criteria (from PVG paper)
SNEAKY_THRESHOLD="${SNEAKY_THRESHOLD:-0.95}"  # Stop when sneaky incorrect > 95%
SCORE_PARITY="${SCORE_PARITY:-0.1}"           # Stop when score gap < 0.1

# LoRA Configuration
# ===================
# LoRA enables efficient training of both Verifier and Prover without
# catastrophic forgetting. See RM-2026.md for theoretical justification.
#
# LoRA rank guidelines:
#   r=4:  Minimal adaptation, fastest, may underfit
#   r=16: Sweet spot for most tasks
#   r=64: Deep adaptation for complex features
#
# Memory savings with LoRA (vs full FT, batch=4, seq=4096):
#   7B:  ~78GB → ~37GB  (52% reduction)
#   14B: ~131GB → ~52GB (60% reduction)
#
USE_LORA="${USE_LORA:-1}"  # 1=enable, 0=full fine-tuning
VERIFIER_LORA_RANK="${VERIFIER_LORA_RANK:-16}"  # Higher for complex verification
PROVER_LORA_RANK="${PROVER_LORA_RANK:-8}"       # Lower for policy updates
LORA_ALPHA_MULT="${LORA_ALPHA_MULT:-2}"         # alpha = rank * this
USE_RSLORA="${USE_RSLORA:-1}"                   # rsLoRA for rank-independent gradients
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# Execution
SANDBOX_WORKERS="${SANDBOX_WORKERS:-32}"
CONCURRENCY="${CONCURRENCY:-32}"
PROVER_VLLM_PORT="${PROVER_VLLM_PORT:-8000}"
VERIFIER_VLLM_PORT="${VERIFIER_VLLM_PORT:-8001}"
PROVER_VLLM_GPU_MEMORY="${PROVER_VLLM_GPU_MEMORY:-0.77}"
VERIFIER_VLLM_GPU_MEMORY="${VERIFIER_VLLM_GPU_MEMORY:-0.18}"

# Logging
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pvg-training}"

# =============================================================================
# Effective batch size calculation
# =============================================================================
EFFECTIVE_BATCH=$((BATCH_SIZE * GROUP_SIZE))
SAMPLES_PER_MICRO=$((MICRO_TOKEN_BUDGET / MAX_SEQ_LEN))

# Guard against division by zero when MICRO_TOKEN_BUDGET < MAX_SEQ_LEN
if [ "$SAMPLES_PER_MICRO" -lt 1 ]; then
    echo "WARNING: MICRO_TOKEN_BUDGET ($MICRO_TOKEN_BUDGET) < MAX_SEQ_LEN ($MAX_SEQ_LEN)"
    echo "         Setting SAMPLES_PER_MICRO=1"
    SAMPLES_PER_MICRO=1
fi

MICRO_BATCHES_PER_STEP=$(( (EFFECTIVE_BATCH + SAMPLES_PER_MICRO - 1) / SAMPLES_PER_MICRO ))

echo "========================================"
echo "PVG Training on Isambard-AI"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS"
echo "Time: $(date)"
echo "========================================"
echo ""
echo "Model Configuration:"
echo "  Prover:   $PROVER_MODEL"
echo "  Verifier: $VERIFIER_MODEL"
echo "  Dataset:  $DATASET (limit: $LIMIT, split: $SPLIT_RATIO)"
echo ""
echo "Training Configuration:"
echo "  Rounds:         $NUM_ROUNDS"
echo "  Verifier steps: $VERIFIER_STEPS per round"
echo "  Prover steps:   $PROVER_STEPS per round"
echo ""
echo "Batch Sizing:"
echo "  ┌──────────────────────────────────────────────────────────┐"
echo "  │ batch_size (problems):       $BATCH_SIZE"
echo "  │ group_size (gens/problem):   $GROUP_SIZE"
echo "  │ ────────────────────────────────────────────────────────│"
echo "  │ → Macro batch:               $EFFECTIVE_BATCH samples"
echo "  │ ────────────────────────────────────────────────────────│"
echo "  │ max_seq_len:                 $MAX_SEQ_LEN tokens"
echo "  │ micro_token_budget:          $MICRO_TOKEN_BUDGET tokens"
echo "  │ ────────────────────────────────────────────────────────│"
echo "  │ → Samples per micro-batch:   ~$SAMPLES_PER_MICRO"
echo "  │ → Micro-batches per step:    ~$MICRO_BATCHES_PER_STEP"
echo "  │ → Gradient accumulates automatically                    │"
echo "  │ ────────────────────────────────────────────────────────│"
echo "  │ Empirical memory (7B, batch=4): ~78GB/GPU with FSDP    │"
echo "  │ GB200 capacity: 192GB (163GB with safety margin)       │"
echo "  └──────────────────────────────────────────────────────────┘"
echo ""
echo "IRM Configuration:"
echo "  Mode: $IRM_MODE"
echo "  Beta: $IRM_BETA"
echo ""
if [ "$USE_LORA" = "1" ]; then
    VERIFIER_ALPHA=$((VERIFIER_LORA_RANK * LORA_ALPHA_MULT))
    PROVER_ALPHA=$((PROVER_LORA_RANK * LORA_ALPHA_MULT))
    echo "LoRA Configuration:"
    echo "  ┌──────────────────────────────────────────────────────────┐"
    echo "  │ Verifier LoRA:  rank=$VERIFIER_LORA_RANK, alpha=$VERIFIER_ALPHA"
    echo "  │ Prover LoRA:    rank=$PROVER_LORA_RANK, alpha=$PROVER_ALPHA"
    echo "  │ rsLoRA:         $USE_RSLORA"
    echo "  │ Target modules: $LORA_TARGET_MODULES"
    echo "  │ ────────────────────────────────────────────────────────│"
    echo "  │ Memory savings vs full FT (7B): ~78GB → ~37GB (52%)     │"
    echo "  └──────────────────────────────────────────────────────────┘"
else
    echo "LoRA: Disabled (full fine-tuning)"
fi
echo ""
echo "Stopping Criteria:"
echo "  Sneaky incorrect threshold: $SNEAKY_THRESHOLD"
echo "  Score parity threshold:     $SCORE_PARITY"
echo "========================================"
echo ""

# Check .env
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found"
    exit 1
fi

# Sync environment
echo "Syncing Python environment..."
uv python install 3.12
uv sync

# =============================================================================
# GPU allocation
# =============================================================================
echo ""
echo "GPU allocation:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"

# GPU 0: vLLM inference (prover + verifier)
# GPU 1-3: FSDP training
export VLLM_GPU=0
export TRAIN_GPUS="1,2,3"
export NPROC_PER_NODE=3  # For torchrun

echo ""
echo "GPU assignment:"
echo "  GPU $VLLM_GPU: vLLM servers (prover + verifier)"
echo "  GPU $TRAIN_GPUS: FSDP training ($NPROC_PER_NODE processes)"

# =============================================================================
# Start vLLM servers
# =============================================================================
echo ""
echo "========================================"
echo "Starting vLLM servers..."
echo "========================================"

PROVER_VLLM_LOG="${LOG_DIR}/vllm_prover.log"
VERIFIER_VLLM_LOG="${LOG_DIR}/vllm_verifier.log"

CUDA_VISIBLE_DEVICES=$VLLM_GPU uv run --env-file .env python -m ludic.inference.vllm_server \
    --model "$PROVER_MODEL" \
    --port $PROVER_VLLM_PORT \
    --trust-remote-code \
    --max-model-len $MAX_SEQ_LEN \
    --gpu-memory-utilization $PROVER_VLLM_GPU_MEMORY \
    > "$PROVER_VLLM_LOG" 2>&1 &

PROVER_VLLM_PID=$!
echo "Prover vLLM server started (PID: $PROVER_VLLM_PID)"

CUDA_VISIBLE_DEVICES=$VLLM_GPU uv run --env-file .env python -m ludic.inference.vllm_reward_server \
    --model "$VERIFIER_MODEL" \
    --port $VERIFIER_VLLM_PORT \
    --trust-remote-code \
    --max-model-len $MAX_SEQ_LEN \
    --gpu-memory-utilization $VERIFIER_VLLM_GPU_MEMORY \
    > "$VERIFIER_VLLM_LOG" 2>&1 &

VERIFIER_VLLM_PID=$!
echo "Verifier vLLM server started (PID: $VERIFIER_VLLM_PID)"

# Wait for vLLM
echo "Waiting for vLLM servers..."
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

# Cleanup trap
cleanup() {
    echo "Cleaning up..."
    [ -n "$PROVER_VLLM_PID" ] && kill $PROVER_VLLM_PID 2>/dev/null || true
    [ -n "$VERIFIER_VLLM_PID" ] && kill $VERIFIER_VLLM_PID 2>/dev/null || true
    [ -n "$SLURM_JOB_ID" ] && podman-hpc ps -aq --filter "name=ludic-sandbox-${SLURM_JOB_ID}" 2>/dev/null | xargs -r podman-hpc rm -f 2>/dev/null || true
}
trap cleanup EXIT

# =============================================================================
# Run PVG training
# =============================================================================
echo ""
echo "========================================"
echo "Starting PVG training..."
echo "========================================"

OUTPUT_DIR="${LOG_DIR}/output"
mkdir -p "$OUTPUT_DIR"

# Build command
TRAIN_CMD="uv run --env-file .env python examples/pvg/train_pvg.py"
TRAIN_CMD="$TRAIN_CMD --verifier-model $VERIFIER_MODEL"
TRAIN_CMD="$TRAIN_CMD --prover-model $PROVER_MODEL"
TRAIN_CMD="$TRAIN_CMD --dataset $DATASET"
TRAIN_CMD="$TRAIN_CMD --split-ratio $SPLIT_RATIO"
TRAIN_CMD="$TRAIN_CMD --num-rounds $NUM_ROUNDS"
TRAIN_CMD="$TRAIN_CMD --verifier-steps $VERIFIER_STEPS"
TRAIN_CMD="$TRAIN_CMD --prover-steps $PROVER_STEPS"
TRAIN_CMD="$TRAIN_CMD --irm-mode $IRM_MODE"
TRAIN_CMD="$TRAIN_CMD --irm-beta $IRM_BETA"
TRAIN_CMD="$TRAIN_CMD --reward-strategy $REWARD_STRATEGY"
TRAIN_CMD="$TRAIN_CMD --sneaky-incorrect-threshold $SNEAKY_THRESHOLD"
TRAIN_CMD="$TRAIN_CMD --score-parity-threshold $SCORE_PARITY"
TRAIN_CMD="$TRAIN_CMD --output-dir $OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --host 127.0.0.1"
TRAIN_CMD="$TRAIN_CMD --prover-port $PROVER_VLLM_PORT"
TRAIN_CMD="$TRAIN_CMD --verifier-port $VERIFIER_VLLM_PORT"
TRAIN_CMD="$TRAIN_CMD --verbose"

# Add WandB configuration when enabled
if [ "$USE_WANDB" = "1" ]; then
    TRAIN_CMD="$TRAIN_CMD"
    export WANDB_PROJECT="$WANDB_PROJECT"
    echo "WandB enabled: project=$WANDB_PROJECT"
fi

# Add LoRA arguments if enabled
if [ "$USE_LORA" = "1" ]; then
    TRAIN_CMD="$TRAIN_CMD --verifier-lora --verifier-lora-rank $VERIFIER_LORA_RANK"
    TRAIN_CMD="$TRAIN_CMD --verifier-lora-alpha $((VERIFIER_LORA_RANK * LORA_ALPHA_MULT))"
    TRAIN_CMD="$TRAIN_CMD --prover-lora --prover-lora-rank $PROVER_LORA_RANK"
    TRAIN_CMD="$TRAIN_CMD --prover-lora-alpha $((PROVER_LORA_RANK * LORA_ALPHA_MULT))"
    TRAIN_CMD="$TRAIN_CMD --lora-target-modules $LORA_TARGET_MODULES"
    if [ "$USE_RSLORA" = "1" ]; then
        TRAIN_CMD="$TRAIN_CMD --use-rslora"
    fi
fi

echo "Command: $TRAIN_CMD"
echo ""

# Run with FSDP on training GPUs
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS $TRAIN_CMD

EXIT_CODE=$?

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "PVG training completed!"
    echo ""
    echo "Output directory: $OUTPUT_DIR"
    echo "  ├── data/          # Rollouts and preference pairs per round"
    echo "  ├── metrics/       # Per-round metrics JSON"
    echo "  └── checkpoints/   # Model checkpoints per round"
else
    echo "Training failed (exit code: $EXIT_CODE)"
fi
echo "========================================"
echo "Completed at: $(date)"

exit $EXIT_CODE

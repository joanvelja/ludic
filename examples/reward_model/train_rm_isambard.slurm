#!/bin/bash
#SBATCH --job-name=ludic-rm-v2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_%j_rm.log
#SBATCH --error=logs/slurm_%j_rm.log

set -e

# Change to project root
cd /home/u5ds/joanv.u5ds/ludic

# Create logs directory if it doesn't exist
mkdir -p logs

# Sync environment
uv sync

# Run RM training with RMTrainer V2:
# - Uses RMTrainer with pair-aware micro-batching
# - cosine LR scheduler with 50-step warmup
# - Early stopping on eval accuracy
# - Label smoothing for regularization
# - Micro-batch token budget for memory efficiency
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-7B \
    --hf-dataset stanfordnlp/SHP \
    --hf-split train \
    --limit 5000 \
    --steps 1000 \
    --batch-size 16 \
    --micro-token-budget 8192 \
    --max-seq-len 1024 \
    --lr 5e-6 \
    --weight-decay 0.01 \
    --lr-scheduler cosine \
    --warmup-steps 50 \
    --label-smoothing 0.05 \
    --log-every 10 \
    --save-every 100 \
    --eval-split test \
    --eval-limit 1000 \
    --eval-every 100 \
    --early-stopping-patience 5 \
    --output-dir checkpoints_rm_v2

echo "RM training complete!"

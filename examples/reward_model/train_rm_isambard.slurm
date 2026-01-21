#!/bin/bash
#SBATCH --job-name=ludic-rm-test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%j_rm.log
#SBATCH --error=logs/slurm_%j_rm.log

set -e

# Change to project root
cd /home/u5ds/joanv.u5ds/ludic

# Create logs directory if it doesn't exist
mkdir -p logs

# Sync environment
uv sync

# Run RM training with improved configuration:
# - weight-decay=0.01 to prevent overfitting
# - eval-every=50 for frequent evaluation
# - early-stopping-patience=3 to stop when overfitting starts
# - lower lr=5e-6 for more stable training
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-0.5B \
    --hf-dataset stanfordnlp/SHP \
    --hf-split train \
    --limit 2000 \
    --steps 500 \
    --batch-size 8 \
    --max-seq-len 1024 \
    --lr 5e-6 \
    --weight-decay 0.01 \
    --log-every 10 \
    --save-every 100 \
    --eval-split test \
    --eval-limit 500 \
    --eval-every 50 \
    --early-stopping-patience 3 \
    --output-dir checkpoints_rm

echo "RM training complete!"

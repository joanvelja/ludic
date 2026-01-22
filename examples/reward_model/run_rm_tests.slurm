#!/bin/bash
#SBATCH --job-name=rm-tests
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%j_rm_tests.log
#SBATCH --error=logs/slurm_%j_rm_tests.log

# RM-Related Test Suite
# =====================
# Runs all reward model related tests on GPU to verify:
# - Chat template application
# - Preference tokenization
# - Bradley-Terry loss + label smoothing
# - RMTrainer batch processing
# - Classification metrics

set -e

cd /home/u5ds/joanv.u5ds/ludic
mkdir -p logs

echo "========================================"
echo "RM Test Suite"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================"

# Sync environment
echo "Syncing Python environment..."
uv sync

# Run RM-related tests
echo ""
echo "Running RM-related tests..."
echo "========================================"

uv run pytest \
    tests/test_rm_trainer.py \
    tests/test_rm_batching.py \
    tests/test_bradley_terry_loss.py \
    tests/test_bradley_terry_label_smoothing.py \
    tests/test_preference_utils.py \
    tests/test_preference_batching.py \
    tests/test_classification_metrics.py \
    -v --tb=short

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "All RM tests passed!"
else
    echo "Some tests failed (exit code: $EXIT_CODE)"
fi
echo "========================================"
echo "Completed at: $(date)"

exit $EXIT_CODE

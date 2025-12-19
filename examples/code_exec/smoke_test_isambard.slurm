#!/bin/bash
#SBATCH --job-name=ludic-smoke-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=smoke_test_%j.log
#SBATCH --error=smoke_test_%j.log

# =============================================================================
# Smoke Test for CodeExecEnv on Isambard HPC
# =============================================================================
#
# This script runs a quick validation of the CodeExecEnv training pipeline
# using Podman-HPC sandboxing (CPU only, no GPU required).
#
# Usage:
#   sbatch examples/code_exec/smoke_test_isambard.slurm
#
# Prerequisites:
#   1. Have ludic installed in your environment
#   2. Pre-pull the Python image:
#      podman-hpc pull python:3.11-slim
#
# Expected runtime: ~2-5 minutes
# =============================================================================

set -e

echo "========================================"
echo "Ludic CodeExecEnv Smoke Test"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================"

# Ensure podman-hpc is available
if ! command -v podman-hpc &> /dev/null; then
    echo "ERROR: podman-hpc not found in PATH"
    exit 1
fi

echo "podman-hpc version: $(podman-hpc --version 2>&1 | head -1)"

# Check if Python image is available (migrated)
echo ""
echo "Checking container image..."
if podman-hpc images | grep -q "python.*3.11"; then
    echo "  python:3.11-slim image found"
else
    echo "  Image not found, pulling..."
    podman-hpc pull python:3.11-slim
fi

# Ensure we're in the repo root
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Must run from repo root (pyproject.toml not found)"
    exit 1
fi

echo ""
echo "Using uv for Python environment"
echo "uv version: $(uv --version 2>&1 || echo 'not found')"

echo ""
echo "========================================"
echo "Running diagnostic first..."
echo "========================================"

# Run diagnostic to check what podman-hpc flags work
uv run python examples/code_exec/diagnose_podman.py

echo ""
echo "========================================"
echo "Running smoke tests..."
echo "========================================"

# Run the smoke test (uses minimal config by default for HPC compatibility)
uv run python examples/code_exec/smoke_test_isambard.py

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Smoke test PASSED"
else
    echo "Smoke test FAILED (exit code: $EXIT_CODE)"
fi
echo "========================================"
echo "Completed at: $(date)"

exit $EXIT_CODE

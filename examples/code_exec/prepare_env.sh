#!/bin/bash
# prepare_env.sh - Run on login node before submitting SLURM job
#
# This script pre-stages all dependencies that require network access:
# - Python runtime (via uv)
# - Python packages (via uv sync)
# - Container images (via podman-hpc pull)
#
# Usage: ./prepare_env.sh
#        sbatch examples/code_exec/train_apps_isambard.slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."  # Navigate to repo root

echo "=== Preparing environment for Isambard-AI ==="
echo ""

echo "1. Installing Python 3.12..."
uv python install 3.12

echo ""
echo "2. Syncing Python dependencies..."
uv sync

echo ""
echo "3. Adding datasets package..."
uv add datasets

echo ""
echo "4. Pre-pulling sandbox container image..."
podman-hpc pull python:3.11-slim

echo ""
echo "=== Environment preparation complete ==="
echo "You can now submit the job with: sbatch examples/code_exec/train_apps_isambard.slurm"

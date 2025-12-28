#!/bin/bash
# prepare_env.sh - Run on login node before submitting SLURM job
#
# This script pre-stages all dependencies that require network access:
# - HuggingFace cache directory (on high-quota Lustre storage)
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

# ---------------------------------------------------------------------------
# HuggingFace Cache Configuration
# ---------------------------------------------------------------------------
# Models on Lustre ($SCRATCH) instead of quota-limited $HOME
# This is idempotent: safe to run multiple times.
# ---------------------------------------------------------------------------
echo "1. Configuring HuggingFace cache..."

HF_CACHE_DIR="${SCRATCH}/.cache/huggingface"
mkdir -p "$HF_CACHE_DIR"

# Add to .bashrc if not already present (idempotent)
BASHRC_MARKER="# [ludic] HuggingFace cache on Lustre"
if ! grep -q "$BASHRC_MARKER" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'EOF'

# [ludic] HuggingFace cache on Lustre
# Configured by: examples/code_exec/prepare_env.sh
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
EOF
    echo "  Added HuggingFace environment variables to ~/.bashrc"
else
    echo "  HuggingFace configuration already in ~/.bashrc (skipping)"
fi

# Export for current session
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

echo "  Cache location: $HF_CACHE_DIR"
echo ""

echo "2. Installing Python 3.12..."
uv python install 3.12

echo ""
echo "3. Syncing Python dependencies..."
uv sync

echo ""
echo "4. Adding datasets package..."
uv add datasets

echo ""
echo "5. Pre-pulling sandbox container image..."
podman-hpc pull python:3.11-slim

echo ""
echo "=== Environment preparation complete ==="
echo ""
echo "HuggingFace models will be stored at:"
echo "  $HF_CACHE_DIR"
echo ""
echo "To apply changes to your current shell, run:"
echo "  source ~/.bashrc"
echo ""
echo "You can now submit the job with: sbatch examples/code_exec/train_apps_isambard.slurm"

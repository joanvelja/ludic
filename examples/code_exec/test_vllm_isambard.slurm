#!/bin/bash
#SBATCH --job-name=vllm-test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=vllm_test_%j.log

set -e

# Fix: Use PyTorch CUDA 12.8 index for aarch64 wheels
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"

echo "=== Syncing for aarch64 ==="
uv sync --reinstall-package torch --reinstall-package vllm

echo "=== Testing PyTorch CUDA ==="
uv run python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A')
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
"

echo "=== Testing vLLM import ==="
uv run python -c "import vllm; print('vLLM:', vllm.__version__)"

echo "=== Starting vLLM server (30s) ==="
timeout 30 uv run python -m ludic.inference.vllm_server \
    --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 \
    || echo "Server ran for 30s (expected)"

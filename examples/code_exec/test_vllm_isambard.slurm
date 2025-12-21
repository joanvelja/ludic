#!/bin/bash
#SBATCH --job-name=vllm-test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=vllm_test_%j.log

set -e

echo "=== Environment ==="
echo "Node: $(hostname)"
echo "Platform: $(uname -m)"
echo "uv version: $(uv --version)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "=== Installing PyTorch + vLLM with CUDA backend ==="
# --torch-backend only works with 'uv pip', not 'uv sync'
# See: https://docs.astral.sh/uv/guides/integration/pytorch/
uv pip install torch torchvision --torch-backend=cu128
uv pip install vllm --torch-backend=cu128

echo ""
echo "=== Verifying installed packages ==="
uv pip list | grep -E "^(torch|vllm)" || true

echo ""
echo "=== Testing PyTorch CUDA ==="
# Use --no-sync to prevent uv from reverting via pyproject.toml
uv run --no-sync python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda if hasattr(torch.version, 'cuda') and torch.version.cuda else 'N/A')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())
"

echo ""
echo "=== Testing vLLM import ==="
uv run --no-sync python -c "import vllm; print('vLLM:', vllm.__version__)"

echo ""
echo "=== Starting vLLM server (60s test) ==="
timeout 60 uv run --no-sync python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000 \
    --trust-remote-code \
    2>&1 || echo "Server test complete"

echo ""
echo "=== Done ==="

"""
GPU tests for Flash Attention and hardware detection.

These tests are designed to run on interactive GPU nodes (not login nodes).
Mark with @pytest.mark.gpu and run with: pytest -v -m gpu

Usage on Isambard:
    srun --nodes=1 --gpus=1 --time=10:00 --pty bash
    uv run pytest tests/test_flash_attention.py -v -m gpu -s
"""

from __future__ import annotations

import logging
import pytest
import torch

# Configure logging for visibility during tests
logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")



@pytest.mark.gpu
def test_cuda_available():
    """Verify CUDA is available (basic sanity check)."""
    assert torch.cuda.is_available(), "CUDA not available - run on a GPU node"


@pytest.mark.gpu
def test_flash_sdp_enabled():
    """Verify Flash SDP backend can be enabled."""
    torch.backends.cuda.enable_flash_sdp(True)
    # Note: flash_sdp_enabled() returns True only if flash kernels are actually usable
    # This depends on the input shapes and dtypes at runtime
    assert hasattr(torch.backends.cuda, "flash_sdp_enabled")


@pytest.mark.gpu
def test_detect_gpu_architecture():
    """Detect real GPU architecture."""
    from ludic.training.hardware import detect_gpu_architecture
    
    arch = detect_gpu_architecture()
    assert arch is not None, "Could not detect GPU architecture"
    
    # Log the detected architecture
    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    print(f"GPU: {device_name}")
    print(f"Compute capability: sm_{capability[0]}{capability[1]}")
    print(f"Detected architecture: {arch}")
    
    # Validate known architectures
    assert arch in ("hopper", "ampere", "ada", "turing", "volta", "older")


@pytest.mark.gpu
def test_get_cuda_version():
    """Verify CUDA version detection."""
    from ludic.training.hardware import get_cuda_version
    
    version = get_cuda_version()
    assert version is not None, "Could not get CUDA version"
    
    major, minor = version
    print(f"CUDA version: {major}.{minor}")
    
    # Reasonable version bounds
    assert major >= 11, f"CUDA version {major}.{minor} is too old for Flash Attention"


@pytest.mark.gpu
def test_flash_attn_import():
    """Verify flash-attn package loads and reports version."""
    try:
        import flash_attn
        version = flash_attn.__version__
        print(f"flash-attn version: {version}")
        
        # Check version is >= 2.7.0 for FA3 support
        parts = version.split(".")
        major, minor = int(parts[0]), int(parts[1])
        assert (major, minor) >= (2, 7), f"flash-attn {version} < 2.7.0, FA3 not supported"
        
    except ImportError as e:
        pytest.skip(f"flash-attn not installed: {e}")


@pytest.mark.gpu
def test_get_optimal_attention_impl():
    """Test optimal attention implementation selection."""
    from ludic.training.hardware import get_optimal_attention_impl
    
    # With flash attention enabled (default)
    impl = get_optimal_attention_impl(disable_flash_attn=False)
    print(f"Optimal attention (enabled): {impl}")
    assert impl in ("flash_attention_3", "flash_attention_2", "sdpa", "eager")
    
    # With flash attention disabled
    impl_disabled = get_optimal_attention_impl(disable_flash_attn=True)
    print(f"Optimal attention (disabled): {impl_disabled}")
    assert impl_disabled == "sdpa"


@pytest.mark.gpu
def test_configure_flash_attention():
    """Test full Flash Attention configuration."""
    from ludic.training.hardware import configure_flash_attention
    
    # Configure for CUDA device
    attn_impl = configure_flash_attention("cuda", disable_flash_attn=False)
    print(f"Configured attention: {attn_impl}")
    assert attn_impl in ("flash_attention_3", "flash_attention_2", "sdpa")
    
    # Configure for CPU (should return eager)
    attn_impl_cpu = configure_flash_attention("cpu", disable_flash_attn=False)
    assert attn_impl_cpu == "eager"


@pytest.mark.gpu
def test_model_with_flash_attention():
    """Load a small model with flash attention and run forward pass."""
    from ludic.training.hardware import configure_flash_attention
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Configure flash attention
    attn_impl = configure_flash_attention("cuda", disable_flash_attn=False)
    print(f"Using attention: {attn_impl}")
    
    # Load model with flash attention
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).cuda()
    
    # Verify model loaded with correct attention
    print(f"Model attention impl: {model.config._attn_implementation}")
    
    # Run a forward pass
    inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    
    assert outputs.logits is not None
    assert outputs.logits.shape[0] == 1  # batch size
    print(f"Forward pass successful, logits shape: {outputs.logits.shape}")

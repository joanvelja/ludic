"""
GPU hardware detection and Flash Attention configuration utilities.

This module provides utilities for:
- Detecting GPU architecture (Hopper, Ampere, etc.)
- Selecting optimal attention implementation based on hardware
- Configuring PyTorch SDPA backends for Flash Attention

Usage:
    from ludic.training.hardware import configure_flash_attention

    # In training script, after device detection:
    attn_impl = configure_flash_attention(device="cuda", disable_flash_attn=False)
    model = AutoModelForCausalLM.from_pretrained(..., attn_implementation=attn_impl)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch

logger = logging.getLogger(__name__)

# GPU architecture compute capability mapping
# See: https://developer.nvidia.com/cuda-gpus
GPU_ARCHITECTURES = {
    (9, 0): "hopper",   # H100, H200, GH200
    (8, 9): "ada",      # RTX 4090, L40
    (8, 6): "ampere",   # RTX 3090, A10
    (8, 0): "ampere",   # A100
    (7, 5): "turing",   # RTX 2080, T4
    (7, 0): "volta",    # V100
}

AttentionImpl = Literal["flash_attention_3", "flash_attention_2", "sdpa", "eager"]


def detect_gpu_architecture() -> Optional[str]:
    """
    Detect the GPU architecture from CUDA compute capability.
    
    Returns:
        Architecture name: "hopper", "ampere", "ada", "turing", "volta", or None
        if no CUDA GPU is available.
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        capability = torch.cuda.get_device_capability()
        arch = GPU_ARCHITECTURES.get(capability)
        if arch is None:
            # Unknown architecture, try to infer from major version
            major = capability[0]
            if major >= 9:
                arch = "hopper"
            elif major >= 8:
                arch = "ampere"
            else:
                arch = "older"
        return arch
    except Exception as e:
        logger.warning(f"Failed to detect GPU architecture: {e}")
        return None


def get_cuda_version() -> Optional[tuple[int, int]]:
    """
    Get the CUDA runtime version.
    
    Returns:
        Tuple of (major, minor) version, or None if CUDA unavailable.
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        version = torch.version.cuda
        if version is None:
            return None
        parts = version.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception as e:
        logger.warning(f"Failed to get CUDA version: {e}")
        return None


def _check_flash_attn_3_available() -> bool:
    """
    Check if Flash Attention 3 is available for HuggingFace Transformers.
    
    HuggingFace Transformers checks for flash_attention_3 support via:
        importlib.util.find_spec("flash_attn_3")
    
    This requires either:
    1. The flash_attn_3 package installed (pip install flash_attn_3)
    2. Building flash-attn from the hopper/ subdirectory
    3. Using HuggingFace 'kernels' package (pip install kernels)
    
    Returns True only if HuggingFace will accept flash_attention_3.
    """
    import importlib.util
    
    # Check what HuggingFace Transformers actually checks
    if importlib.util.find_spec("flash_attn_3") is not None:
        logger.info("flash_attn_3 package found - FA3 available")
        return True
    
    # Also check for flash_attn_interface (alternative FA3 installation)
    if importlib.util.find_spec("flash_attn_interface") is not None:
        logger.info("flash_attn_interface found - FA3 may be available")
        # Note: This might not work with all HF Transformers versions
        # as they specifically check for flash_attn_3, not flash_attn_interface
        return False  # Be conservative - HF checks for flash_attn_3 specifically
    
    logger.debug("FA3 not available (flash_attn_3 package not found)")
    return False


def get_optimal_attention_impl(
    *,
    disable_flash_attn: bool = False,
) -> AttentionImpl:
    """
    Determine the optimal attention implementation for the current hardware.
    
    Selection logic:
    - Hopper (H100/H200) + CUDA >= 12.3 + flash-attn >= 2.7: flash_attention_3
    - Ampere/Ada + CUDA >= 11.6 + flash-attn installed: flash_attention_2  
    - Otherwise: sdpa (PyTorch native, still uses flash kernels when possible)
    
    Args:
        disable_flash_attn: If True, skip flash attention and use SDPA.
    
    Returns:
        Attention implementation string for HuggingFace models:
        "flash_attention_3", "flash_attention_2", "sdpa", or "eager"
    """
    if disable_flash_attn:
        logger.info("Flash Attention disabled by user request, using SDPA")
        return "sdpa"
    
    arch = detect_gpu_architecture()
    cuda_version = get_cuda_version()
    
    # Check if flash_attn is available
    try:
        import flash_attn
        flash_attn_available = True
        flash_attn_version = getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        flash_attn_available = False
        flash_attn_version = None
    
    if not flash_attn_available:
        logger.info(f"flash-attn not installed, using SDPA (arch={arch})")
        return "sdpa"
    
    # Flash Attention 3: Hopper-only (H100/H200) with CUDA >= 12.3
    # Achieves 1.5-2x speedup over FA2, 75% H100 utilization
    # Ref: https://arxiv.org/abs/2407.08608
    if arch == "hopper" and cuda_version and cuda_version >= (12, 3):
        if _check_flash_attn_3_available():
            logger.info(
                f"Using flash_attention_3 (arch={arch}, cuda={cuda_version}, "
                f"flash_attn={flash_attn_version})"
            )
            return "flash_attention_3"
    
    # Flash Attention 2: Ampere+ with CUDA >= 11.6
    if arch in ("hopper", "ampere", "ada") and cuda_version and cuda_version >= (11, 6):
        logger.info(
            f"Using flash_attention_2 (arch={arch}, cuda={cuda_version}, "
            f"flash_attn={flash_attn_version})"
        )
        return "flash_attention_2"
    
    # Fallback to SDPA (PyTorch native, also uses flash kernels when possible)
    logger.info(f"Using SDPA (arch={arch}, cuda={cuda_version})")
    return "sdpa"


def configure_flash_attention(
    device: str = "cuda",
    *,
    disable_flash_attn: bool = False,
) -> AttentionImpl:
    """
    Configure Flash Attention for optimal performance.
    
    This function:
    1. Enables PyTorch's Flash SDP backend (if available)
    2. Returns the optimal attention implementation for HuggingFace models
    
    Args:
        device: Target device ("cuda" or "cpu")
        disable_flash_attn: If True, disable flash attention entirely.
    
    Returns:
        Attention implementation string to pass to model.from_pretrained().
    
    Example:
        attn_impl = configure_flash_attention("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
        )
    """
    if device != "cuda" or not torch.cuda.is_available():
        logger.info("No CUDA device, using eager attention")
        return "eager"
    
    # Enable Flash SDP backend in PyTorch (uses flash kernels for F.scaled_dot_product_attention)
    if not disable_flash_attn:
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.debug("Enabled torch.backends.cuda.flash_sdp")
        except Exception as e:
            logger.warning(f"Could not enable flash_sdp: {e}")
    
    return get_optimal_attention_impl(disable_flash_attn=disable_flash_attn)


def log_hardware_info() -> None:
    """Log GPU hardware information for debugging."""
    if not torch.cuda.is_available():
        logger.info("No CUDA GPU available")
        return
    
    try:
        device_name = torch.cuda.get_device_name()
        capability = torch.cuda.get_device_capability()
        arch = detect_gpu_architecture()
        cuda_version = get_cuda_version()
        
        logger.info(
            f"GPU: {device_name} (sm_{capability[0]}{capability[1]}, {arch}), "
            f"CUDA: {cuda_version[0]}.{cuda_version[1] if cuda_version else 'N/A'}"
        )
        
        # Check flash_attn
        try:
            import flash_attn
            logger.info(f"flash-attn version: {flash_attn.__version__}")
        except ImportError:
            logger.info("flash-attn: not installed")
            
    except Exception as e:
        logger.warning(f"Could not log hardware info: {e}")

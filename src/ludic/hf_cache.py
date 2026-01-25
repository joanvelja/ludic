"""Helpers for configuring Hugging Face cache locations.

These functions are intentionally lightweight so example scripts can call them
without pulling in additional dependencies or complex configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_HF_CACHE_ENV_VARS = (
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
)
_HF_HOME_ENV_VAR = "HF_HOME"
_LARGE_STORAGE_ENV_VARS = (
    "LUDIC_LARGE_STORAGE",
    "LUDIC_PROJECTDIR",
    "PROJECTDIR",
    "SCRATCHDIR",
    "SCRATCH",
)


def _existing_hf_cache_dir() -> Optional[Path]:
    for key in _HF_CACHE_ENV_VARS:
        value = os.environ.get(key)
        if value:
            return Path(value)
    hf_home = os.environ.get(_HF_HOME_ENV_VAR)
    if hf_home:
        return Path(hf_home) / "hub"
    return None


def _resolve_large_storage_root() -> Optional[Path]:
    for key in _LARGE_STORAGE_ENV_VARS:
        value = os.environ.get(key)
        if value:
            return Path(value)
    return None


def resolve_hf_cache_dir() -> Optional[Path]:
    """Return the cache directory for Hugging Face downloads, if one can be inferred."""
    existing = _existing_hf_cache_dir()
    if existing is not None:
        return existing
    root = _resolve_large_storage_root()
    if root is None:
        return None
    return root / ".cache" / "huggingface" / "hub"


def ensure_hf_cache_dir() -> Optional[str]:
    """Ensure the cache dir exists and export env vars when auto-resolving."""
    cache_dir = resolve_hf_cache_dir()
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _existing_hf_cache_dir() is None and os.environ.get(_HF_HOME_ENV_VAR) is None:
        hf_home = cache_dir.parent
        os.environ.setdefault(_HF_HOME_ENV_VAR, str(hf_home))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    return str(cache_dir)


def hf_cache_kwargs() -> dict[str, str]:
    """Convenience kwargs for from_pretrained() calls."""
    cache_dir = ensure_hf_cache_dir()
    if cache_dir is None:
        return {}
    return {"cache_dir": cache_dir}

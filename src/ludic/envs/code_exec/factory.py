"""
Unified factory for creating sandbox pools.

This module provides:
  - create_sandbox_pool(): Async factory that auto-detects or uses specified backend

Usage:
  from ludic.envs.code_exec import create_sandbox_pool

  # Auto-detect backend
  pool = await create_sandbox_pool(n_workers=4)

  # Explicit backend
  pool = await create_sandbox_pool(n_workers=4, backend="podman-hpc")

  # With custom config
  pool = await create_sandbox_pool(
      n_workers=4,
      backend="docker",
      python_version="3.11",
      memory_limit="512m",
  )
"""

from __future__ import annotations

from typing import Any, Optional

from .backend import SandboxBackend, detect_available_backend
from .sandbox import SandboxPool


async def create_sandbox_pool(
    n_workers: int = 4,
    backend: str = "auto",
    python_version: str = "3.11",
    cache_size: int = 10000,
    max_concurrent_ops: int = 8,
    **backend_kwargs: Any,
) -> SandboxPool:
    """
    Create and start a sandbox pool with the specified or auto-detected backend.

    This is the recommended way to create sandbox pools as it handles:
      - Backend auto-detection based on environment
      - Consistent configuration across backends
      - Proper initialization (pull images, start containers)

    Args:
        n_workers: Number of parallel sandboxes in the pool
        backend: Backend to use ("auto", "docker", "podman-hpc", "singularity")
        python_version: Python version for the sandbox containers
        cache_size: Maximum number of cached execution results
        max_concurrent_ops: Maximum concurrent sandbox operations (resets, exec
            calls). Prevents deadlock in HPC environments. Default 8.
        **backend_kwargs: Additional backend-specific configuration:
            - memory_limit (str): Memory limit (e.g., "256m", "1g")
            - cpu_quota (float): CPU limit as fraction (e.g., 0.5 = 50% of one CPU)
            - network_disabled (bool): Disable network access (default: True)
            - gpu (bool): Enable GPU access (podman-hpc only)
            - image (str): Custom container image (overrides python_version)
            - sif_path (str): Path to .sif file (singularity only)

    Returns:
        Started SandboxPool instance

    Raises:
        RuntimeError: If the specified backend is not available
        ValueError: If an unknown backend is specified

    Examples:
        # Auto-detect (recommended)
        pool = await create_sandbox_pool(n_workers=4)

        # Docker with custom memory
        pool = await create_sandbox_pool(
            n_workers=4,
            backend="docker",
            memory_limit="512m",
        )

        # Podman-HPC with GPU
        pool = await create_sandbox_pool(
            n_workers=4,
            backend="podman-hpc",
            gpu=True,
        )
    """
    # Resolve backend
    if backend == "auto" or backend == SandboxBackend.AUTO:
        resolved_backend = detect_available_backend()
        print(f"Auto-detected sandbox backend: {resolved_backend}")
    else:
        resolved_backend = backend

    # Create pool based on backend
    if resolved_backend == SandboxBackend.DOCKER.value:
        pool = _create_docker_pool(
            n_workers=n_workers,
            python_version=python_version,
            cache_size=cache_size,
            max_concurrent_ops=max_concurrent_ops,
            **backend_kwargs,
        )

    elif resolved_backend == SandboxBackend.PODMAN_HPC.value:
        pool = _create_podman_hpc_pool(
            n_workers=n_workers,
            python_version=python_version,
            cache_size=cache_size,
            max_concurrent_ops=max_concurrent_ops,
            **backend_kwargs,
        )

    elif resolved_backend == SandboxBackend.SINGULARITY.value:
        raise NotImplementedError(
            "Singularity backend is not yet implemented. "
            "Use 'docker' or 'podman-hpc' instead."
        )

    else:
        raise ValueError(
            f"Unknown backend: {resolved_backend}. "
            f"Valid options: {', '.join(b.value for b in SandboxBackend if b != SandboxBackend.AUTO)}"
        )

    # Start the pool
    await pool.start()
    return pool


def _create_docker_pool(
    n_workers: int,
    python_version: str,
    cache_size: int,
    max_concurrent_ops: int = 8,
    memory_limit: str = "256m",
    cpu_quota: int = 50000,
    network_disabled: bool = True,
    image: Optional[str] = None,
    **_kwargs: Any,
) -> SandboxPool:
    """Create DockerSandboxPool with configuration."""
    try:
        from .docker_sandbox import DockerSandboxConfig, DockerSandboxPool
    except ImportError:
        raise RuntimeError(
            "Docker backend requires the docker package:\n"
            "  pip install docker>=7.0.0"
        )

    config = DockerSandboxConfig(
        python_version=python_version,
        base_image=image,
        memory_limit=memory_limit,
        cpu_quota=cpu_quota,
        network_disabled=network_disabled,
    )

    return DockerSandboxPool(
        n_workers=n_workers,
        config=config,
        cache_size=cache_size,
        max_concurrent_ops=max_concurrent_ops,
    )


def _create_podman_hpc_pool(
    n_workers: int,
    python_version: str,
    cache_size: int,
    max_concurrent_ops: int = 8,
    memory_limit: str = "256m",
    cpu_quota: Optional[float] = None,
    network_disabled: bool = True,
    gpu: bool = False,
    image: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    **_kwargs: Any,
) -> SandboxPool:
    """Create PodmanHPCSandboxPool with configuration."""
    from .podman_sandbox import PodmanConfig, PodmanHPCSandboxPool

    config = PodmanConfig(
        memory_limit=memory_limit,
        cpu_quota=cpu_quota,
        network_disabled=network_disabled,
        gpu=gpu,
        extra_args=extra_args,
    )

    # Determine image
    if image is None:
        image = f"python:{python_version}-slim"

    return PodmanHPCSandboxPool(
        n_workers=n_workers,
        image=image,
        config=config,
        cache_size=cache_size,
        max_concurrent_ops=max_concurrent_ops,
    )

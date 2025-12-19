"""
Sandbox backend detection and selection.

This module provides:
  - SandboxBackend: Enumeration of supported sandbox backends
  - detect_available_backend(): Auto-detection based on environment
  - is_*_available(): Individual backend availability checks

Auto-detection priority:
  - In Slurm job: podman-hpc → docker → error
  - Outside Slurm: docker → podman-hpc → error

Usage:
  from ludic.envs.code_exec.backend import detect_available_backend, SandboxBackend

  # Auto-detect
  backend = detect_available_backend()

  # Manual selection
  if backend == SandboxBackend.PODMAN_HPC:
      from ludic.envs.code_exec.podman_sandbox import PodmanHPCSandboxPool
      pool = PodmanHPCSandboxPool(n_workers=4)
"""

from __future__ import annotations

import os
import shutil
from enum import Enum


class SandboxBackend(str, Enum):
    """Supported sandbox backends."""

    DOCKER = "docker"
    PODMAN_HPC = "podman-hpc"
    SINGULARITY = "singularity"
    AUTO = "auto"


def detect_available_backend() -> str:
    """
    Auto-detect the best available sandbox backend.

    Detection priority:
      - In Slurm job (SLURM_JOB_ID set):
        1. podman-hpc (most common on HPC)
        2. docker (some HPC clusters have Docker)
        3. error
      - Outside Slurm:
        1. docker (most common for local development)
        2. podman-hpc
        3. error

    Returns:
        Backend identifier (one of SandboxBackend values, excluding AUTO)

    Raises:
        RuntimeError: If no sandbox backend is available
    """
    in_slurm = os.environ.get("SLURM_JOB_ID") is not None

    if in_slurm:
        # HPC environment: prefer podman-hpc
        if is_podman_hpc_available():
            return SandboxBackend.PODMAN_HPC.value
        if is_docker_available():
            return SandboxBackend.DOCKER.value
    else:
        # Local/cloud environment: prefer Docker
        if is_docker_available():
            return SandboxBackend.DOCKER.value
        if is_podman_hpc_available():
            return SandboxBackend.PODMAN_HPC.value

    # Singularity is deferred but check for future use
    if is_singularity_available():
        # NOTE: Singularity backend not yet implemented
        pass

    raise RuntimeError(
        "No sandbox backend available. Install one of:\n"
        "  - Docker (daemon-based): pip install docker && start Docker daemon\n"
        "  - Podman-HPC (daemonless): available on HPC clusters with podman-hpc\n"
        "\n"
        "For HPC clusters, ensure you're running within a Slurm job:\n"
        "  srun --pty bash\n"
        "  # or\n"
        "  sbatch your_script.sh"
    )


def is_docker_available() -> bool:
    """
    Check if Docker daemon is running and accessible.

    Returns:
        True if Docker is available and responding
    """
    try:
        import docker
        client = docker.from_env()
        client.ping()
        client.close()
        return True
    except ImportError:
        # docker package not installed
        return False
    except Exception:
        # Docker daemon not running or not accessible
        return False


def is_podman_hpc_available() -> bool:
    """
    Check if podman-hpc CLI is available.

    Note: This only checks if the command exists, not if containers
    can actually be run (which may require being in a Slurm job).

    Returns:
        True if podman-hpc command is in PATH
    """
    return shutil.which("podman-hpc") is not None


def is_singularity_available() -> bool:
    """
    Check if Singularity/Apptainer CLI is available.

    Returns:
        True if singularity or apptainer command is in PATH
    """
    return (
        shutil.which("singularity") is not None
        or shutil.which("apptainer") is not None
    )


def get_backend_info() -> dict:
    """
    Get information about all backend availability.

    Useful for debugging and status reporting.

    Returns:
        Dict with backend names as keys and availability info as values
    """
    in_slurm = os.environ.get("SLURM_JOB_ID") is not None

    return {
        "environment": {
            "in_slurm": in_slurm,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "backends": {
            SandboxBackend.DOCKER.value: {
                "available": is_docker_available(),
                "requires": "Docker daemon + docker package",
            },
            SandboxBackend.PODMAN_HPC.value: {
                "available": is_podman_hpc_available(),
                "requires": "podman-hpc command (HPC clusters)",
            },
            SandboxBackend.SINGULARITY.value: {
                "available": is_singularity_available(),
                "requires": "singularity/apptainer command",
                "note": "Not yet implemented",
            },
        },
    }

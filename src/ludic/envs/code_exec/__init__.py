"""
Code execution environment for RL on code generation tasks.

This module provides:
  - CodeExecEnv: Environment that executes code against test cases
  - Sandbox protocols: Async sandboxed execution
  - Test adapters: Dataset-specific test extraction
  - Code runners: Execution strategies (stdin/stdout, function calls, etc.)
  - Backend selection: Auto-detection and manual selection of sandbox backends

Supported backends:
  - Docker (requires docker package + daemon): pip install docker>=7.0.0
  - Podman-HPC (HPC clusters): requires podman-hpc CLI
  - Singularity (planned): not yet implemented

Usage:
  # Recommended: use the factory with auto-detection
  from ludic.envs.code_exec import create_sandbox_pool

  pool = await create_sandbox_pool(n_workers=4)  # Auto-detects backend
  pool = await create_sandbox_pool(n_workers=4, backend="podman-hpc")  # Explicit

  # Or import specific implementations
  from ludic.envs.code_exec import DockerSandboxPool  # Docker
  from ludic.envs.code_exec import PodmanHPCSandboxPool  # Podman-HPC
"""

from __future__ import annotations

from .types import (
    CompileStatus,
    RunStatus,
    CompileResult,
    ExecutionResult,
    TestCase,
    TestResult,
    BatchTestResult,
)
from .sandbox import Sandbox, SandboxPool
from .adapters.base import TestAdapter, OutputVerifier, ExactMatchVerifier
from .runners import CodeRunner, StdinStdoutRunner, compute_hash, hash_tests
from .env import CodeExecConfig, CodeExecEnv

# Backend detection and factory (always available)
from .backend import (
    SandboxBackend,
    detect_available_backend,
    is_docker_available,
    is_podman_hpc_available,
    is_singularity_available,
    get_backend_info,
)
from .factory import create_sandbox_pool

# Docker-related imports are optional (requires docker package)
try:
    from .docker_sandbox import (
        DockerSandboxConfig,
        DockerSandbox,
        DockerSandboxPool,
        LRUCache,
    )
    _DOCKER_AVAILABLE = True
except ImportError:
    _DOCKER_AVAILABLE = False
    DockerSandboxConfig = None  # type: ignore[misc, assignment]
    DockerSandbox = None  # type: ignore[misc, assignment]
    DockerSandboxPool = None  # type: ignore[misc, assignment]
    LRUCache = None  # type: ignore[misc, assignment]

# Podman-HPC imports (always available - uses subprocess, no external package)
from .podman_sandbox import (
    PodmanConfig,
    PodmanHPCSandbox,
    PodmanHPCSandboxPool,
    PodmanError,
)

__all__ = [
    # Types
    "CompileStatus",
    "RunStatus",
    "CompileResult",
    "ExecutionResult",
    "TestCase",
    "TestResult",
    "BatchTestResult",
    # Protocols
    "Sandbox",
    "SandboxPool",
    "TestAdapter",
    "OutputVerifier",
    "CodeRunner",
    # Implementations
    "ExactMatchVerifier",
    "StdinStdoutRunner",
    # Environment
    "CodeExecConfig",
    "CodeExecEnv",
    # Utilities
    "compute_hash",
    "hash_tests",
    # Backend detection
    "SandboxBackend",
    "detect_available_backend",
    "is_docker_available",
    "is_podman_hpc_available",
    "is_singularity_available",
    "get_backend_info",
    # Factory
    "create_sandbox_pool",
    # Podman-HPC (always available)
    "PodmanConfig",
    "PodmanHPCSandbox",
    "PodmanHPCSandboxPool",
    "PodmanError",
]

# Add Docker-related exports only if available
if _DOCKER_AVAILABLE:
    __all__.extend([
        "DockerSandboxConfig",
        "DockerSandbox",
        "DockerSandboxPool",
        "LRUCache",
    ])

"""
Unit tests for Podman-HPC sandbox implementation.

These tests mock subprocess calls to test the logic without requiring
actual podman-hpc CLI or containers.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ludic.envs.code_exec.podman_sandbox import (
    LRUCache,
    PodmanConfig,
    PodmanError,
    PodmanHPCSandbox,
    PodmanHPCSandboxPool,
    PodmanResult,
    _get_container_name_prefix,
)
from ludic.envs.code_exec.backend import (
    SandboxBackend,
    detect_available_backend,
    is_docker_available,
    is_podman_hpc_available,
    is_singularity_available,
    get_backend_info,
)
from ludic.envs.code_exec.types import (
    BatchTestResult,
    CompileStatus,
    RunStatus,
    TestCase,
    TestResult,
    CompileResult,
    ExecutionResult,
)


# ============================================================================
# Container naming tests
# ============================================================================


class TestContainerNaming:
    """Tests for container name prefix generation."""

    def test_local_prefix_without_slurm(self):
        """Without SLURM_JOB_ID, should use 'local' prefix."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure SLURM_JOB_ID is not set
            os.environ.pop("SLURM_JOB_ID", None)
            prefix = _get_container_name_prefix()
            assert prefix == "ludic-sandbox-local"

    def test_slurm_prefix_with_job_id(self):
        """With SLURM_JOB_ID, should include job ID in prefix."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
            prefix = _get_container_name_prefix()
            assert prefix == "ludic-sandbox-12345"


# ============================================================================
# PodmanConfig tests
# ============================================================================


class TestPodmanConfig:
    """Tests for PodmanConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PodmanConfig()
        assert config.memory_limit == "256m"
        assert config.cpu_quota is None
        assert config.network_disabled is True
        assert config.working_dir == "/workspace"
        assert config.gpu is False
        assert config.extra_args is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PodmanConfig(
            memory_limit="512m",
            cpu_quota=0.5,
            network_disabled=False,
            gpu=True,
            extra_args=["--security-opt", "label=disable"],
        )
        assert config.memory_limit == "512m"
        assert config.cpu_quota == 0.5
        assert config.network_disabled is False
        assert config.gpu is True
        assert config.extra_args == ["--security-opt", "label=disable"]


# ============================================================================
# LRUCache tests (same as Docker implementation)
# ============================================================================


class TestLRUCache:
    """Tests for LRUCache implementation."""

    def _make_batch_result(self, code_hash: str, tests_hash: str) -> BatchTestResult:
        """Helper to create a BatchTestResult."""
        return BatchTestResult(
            results=[],
            code_hash=code_hash,
            tests_hash=tests_hash,
        )

    def test_get_miss(self):
        """Cache miss should return None and increment miss counter."""
        cache = LRUCache(max_size=10)
        result = cache.get("code1", "tests1")
        assert result is None
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0

    def test_put_and_get(self):
        """Should store and retrieve values."""
        cache = LRUCache(max_size=10)
        batch_result = self._make_batch_result("code1", "tests1")
        cache.put("code1", "tests1", batch_result)

        result = cache.get("code1", "tests1")
        assert result is batch_result
        assert cache.stats["hits"] == 1
        assert cache.stats["size"] == 1

    def test_lru_eviction(self):
        """Should evict least recently used when full."""
        cache = LRUCache(max_size=2)

        result1 = self._make_batch_result("code1", "tests1")
        result2 = self._make_batch_result("code2", "tests2")
        result3 = self._make_batch_result("code3", "tests3")

        cache.put("code1", "tests1", result1)
        cache.put("code2", "tests2", result2)
        # Access code1 to make it recently used
        cache.get("code1", "tests1")
        # Add code3, should evict code2 (least recently used)
        cache.put("code3", "tests3", result3)

        assert cache.get("code1", "tests1") is result1  # Still there
        assert cache.get("code2", "tests2") is None  # Evicted
        assert cache.get("code3", "tests3") is result3  # Still there

    def test_put_overwrites_existing(self):
        """Should overwrite existing values with same key."""
        cache = LRUCache(max_size=10)
        result1 = self._make_batch_result("code1", "tests1")
        result2 = self._make_batch_result("code1", "tests1")

        cache.put("code1", "tests1", result1)
        cache.put("code1", "tests1", result2)

        result = cache.get("code1", "tests1")
        assert result is result2
        assert cache.stats["size"] == 1


# ============================================================================
# PodmanHPCSandbox tests (mocked subprocess)
# ============================================================================


class TestPodmanHPCSandbox:
    """Tests for PodmanHPCSandbox with mocked subprocess."""

    @pytest.fixture
    def sandbox(self):
        """Create a sandbox instance for testing."""
        config = PodmanConfig(memory_limit="256m", network_disabled=True)
        return PodmanHPCSandbox(
            container_name="test-container",
            image="python:3.11-slim",
            config=config,
            python_version="3.11",
        )

    @pytest.mark.asyncio
    async def test_start_creates_container(self, sandbox):
        """Start should create and run a persistent container."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await sandbox.start()

            # Should have called rm -f, run -d, and mkdir
            assert mock_exec.call_count == 3
            calls = mock_exec.call_args_list

            # First call: rm -f
            assert calls[0][0][0] == "podman-hpc"
            assert "rm" in calls[0][0]
            assert "-f" in calls[0][0]

            # Second call: run -d
            assert calls[1][0][0] == "podman-hpc"
            assert "run" in calls[1][0]
            assert "-d" in calls[1][0]
            assert "--name" in calls[1][0]
            assert "test-container" in calls[1][0]
            assert "sleep" in calls[1][0]
            assert "infinity" in calls[1][0]

            # Third call: mkdir
            assert calls[2][0][0] == "podman-hpc"
            assert "exec" in calls[2][0]
            assert "mkdir" in calls[2][0]

    @pytest.mark.asyncio
    async def test_reset_clears_workspace(self, sandbox):
        """Reset should clear the workspace directory."""
        sandbox._started = True

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await sandbox.reset()

            mock_exec.assert_called_once()
            args = mock_exec.call_args[0]
            assert "podman-hpc" in args
            assert "exec" in args
            assert "rm" in " ".join(args)
            assert "/workspace/*" in " ".join(args)

    @pytest.mark.asyncio
    async def test_compile_success(self, sandbox):
        """Compile should return SUCCESS for valid code."""
        sandbox._started = True

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await sandbox.compile("print('hello')")

            assert result.status == CompileStatus.SUCCESS
            assert result.error_message is None

    @pytest.mark.asyncio
    async def test_compile_syntax_error(self, sandbox):
        """Compile should return SYNTAX_ERROR for invalid code."""
        sandbox._started = True

        error_output = b"  File \"_check.py\", line 1\n    def foo(\n         ^\nSyntaxError: invalid syntax"

        # Create two different mock processes:
        # 1. For _write_file (tar command) - should succeed
        # 2. For py_compile - should fail with syntax error
        write_process = AsyncMock()
        write_process.returncode = 0
        write_process.communicate = AsyncMock(return_value=(b"", b""))

        compile_process = AsyncMock()
        compile_process.returncode = 1
        compile_process.communicate = AsyncMock(return_value=(b"", error_output))

        call_count = [0]
        def create_mock_process(*args, **kwargs):
            call_count[0] += 1
            # First call is tar (write_file), second is py_compile
            if call_count[0] == 1:
                return write_process
            return compile_process

        with patch("asyncio.create_subprocess_exec", side_effect=create_mock_process):
            result = await sandbox.compile("def foo(")

            assert result.status == CompileStatus.SYNTAX_ERROR
            assert "SyntaxError" in result.error_message
            assert result.error_line == 1

    @pytest.mark.asyncio
    async def test_execute_success(self, sandbox):
        """Execute should return SUCCESS and stdout for valid code."""
        sandbox._started = True

        # Mock two processes: one for compile (py_compile), one for execute
        compile_process = AsyncMock()
        compile_process.returncode = 0
        compile_process.communicate = AsyncMock(return_value=(b"", b""))

        exec_process = AsyncMock()
        exec_process.returncode = 0
        exec_process.communicate = AsyncMock(return_value=(b"hello world\n", b""))

        call_count = [0]
        def mock_create_subprocess(*args, **kwargs):
            call_count[0] += 1
            # First few calls are for compile (write file, py_compile)
            # Later calls are for execute (write file, run)
            if "py_compile" in args or call_count[0] <= 2:
                return compile_process
            return exec_process

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
            result = await sandbox.execute("print('hello world')")

            assert result.compiled
            assert result.run_status == RunStatus.SUCCESS
            assert "hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self, sandbox):
        """Execute should return RUNTIME_ERROR for code that raises exception."""
        sandbox._started = True

        # Mock processes for various stages:
        # 1. tar write (compile _write_file)
        # 2. py_compile
        # 3. tar write (execute _write_file)
        # 4. python execution (runtime error)
        success_process = AsyncMock()
        success_process.returncode = 0
        success_process.communicate = AsyncMock(return_value=(b"", b""))

        exec_process = AsyncMock()
        exec_process.returncode = 1
        exec_process.communicate = AsyncMock(return_value=(b"", b"ZeroDivisionError: division by zero"))

        call_count = [0]
        def mock_create_subprocess(*args, **kwargs):
            call_count[0] += 1
            # Calls 1-3 are compile phase (tar, py_compile) and execute tar
            # Call 4 is the actual execution
            if call_count[0] <= 3:
                return success_process
            return exec_process

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
            result = await sandbox.execute("1/0")

            assert result.compiled
            assert result.run_status == RunStatus.RUNTIME_ERROR
            assert "ZeroDivisionError" in result.stderr

    def test_parse_syntax_error(self):
        """Test syntax error parsing."""
        error_msg = """  File "_check.py", line 5
    def foo(
           ^
SyntaxError: invalid syntax"""

        line, column, clean_msg = PodmanHPCSandbox._parse_syntax_error(error_msg)

        assert line == 5
        assert "SyntaxError" in clean_msg
        assert "invalid syntax" in clean_msg


# ============================================================================
# PodmanHPCSandboxPool tests
# ============================================================================


class TestPodmanHPCSandboxPool:
    """Tests for PodmanHPCSandboxPool."""

    def test_parse_python_version_from_image(self):
        """Should extract Python version from image name."""
        assert PodmanHPCSandboxPool._parse_python_version("python:3.11-slim") == "3.11"
        assert PodmanHPCSandboxPool._parse_python_version("python:3.10") == "3.10"
        assert PodmanHPCSandboxPool._parse_python_version("ghcr.io/foo/python:3.12-bullseye") == "3.12"
        assert PodmanHPCSandboxPool._parse_python_version("custom-image:latest") == "3.11"  # fallback

    def test_pool_initialization(self):
        """Test pool initialization without starting."""
        pool = PodmanHPCSandboxPool(
            n_workers=4,
            image="python:3.11-slim",
            cache_size=1000,
        )

        assert pool.python_version == "3.11"
        assert pool.available == 0  # Not started yet
        assert pool.cache_stats["size"] == 0

    @pytest.mark.asyncio
    async def test_checkout_before_start_raises(self):
        """Checkout before start should raise RuntimeError."""
        pool = PodmanHPCSandboxPool(n_workers=2)

        with pytest.raises(RuntimeError, match="not started"):
            await pool.checkout()

    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test cache get/put operations."""
        pool = PodmanHPCSandboxPool(n_workers=2, cache_size=100)

        batch_result = BatchTestResult(
            results=[],
            code_hash="abc123",
            tests_hash="def456",
        )

        # Cache miss
        assert pool.get_cached("abc123", "def456") is None

        # Cache put
        pool.put_cached("abc123", "def456", batch_result)

        # Cache hit
        result = pool.get_cached("abc123", "def456")
        assert result is batch_result


# ============================================================================
# Backend detection tests
# ============================================================================


class TestBackendDetection:
    """Tests for backend detection functions."""

    def test_sandbox_backend_enum(self):
        """Test SandboxBackend enum values."""
        assert SandboxBackend.DOCKER.value == "docker"
        assert SandboxBackend.PODMAN_HPC.value == "podman-hpc"
        assert SandboxBackend.SINGULARITY.value == "singularity"
        assert SandboxBackend.AUTO.value == "auto"

    def test_is_podman_hpc_available_not_installed(self):
        """Should return False when podman-hpc is not in PATH."""
        with patch("shutil.which", return_value=None):
            assert is_podman_hpc_available() is False

    def test_is_podman_hpc_available_installed(self):
        """Should return True when podman-hpc is in PATH."""
        with patch("shutil.which", return_value="/usr/bin/podman-hpc"):
            assert is_podman_hpc_available() is True

    def test_is_singularity_available_not_installed(self):
        """Should return False when singularity is not in PATH."""
        with patch("shutil.which", return_value=None):
            assert is_singularity_available() is False

    def test_is_singularity_available_installed(self):
        """Should return True when singularity is in PATH."""
        def mock_which(cmd):
            if cmd == "singularity":
                return "/usr/bin/singularity"
            return None

        with patch("shutil.which", side_effect=mock_which):
            assert is_singularity_available() is True

    def test_is_singularity_available_apptainer(self):
        """Should return True when apptainer (renamed singularity) is in PATH."""
        def mock_which(cmd):
            if cmd == "apptainer":
                return "/usr/bin/apptainer"
            return None

        with patch("shutil.which", side_effect=mock_which):
            assert is_singularity_available() is True

    def test_detect_backend_in_slurm_with_podman(self):
        """In Slurm with podman-hpc available, should prefer podman-hpc."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
            with patch("shutil.which", return_value="/usr/bin/podman-hpc"):
                with patch("ludic.envs.code_exec.backend.is_docker_available", return_value=True):
                    backend = detect_available_backend()
                    assert backend == "podman-hpc"

    def test_detect_backend_outside_slurm_with_docker(self):
        """Outside Slurm with Docker available, should prefer Docker."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SLURM_JOB_ID", None)
            with patch("ludic.envs.code_exec.backend.is_docker_available", return_value=True):
                backend = detect_available_backend()
                assert backend == "docker"

    def test_detect_backend_outside_slurm_no_docker_with_podman(self):
        """Outside Slurm without Docker but with podman-hpc, should use podman-hpc."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SLURM_JOB_ID", None)
            with patch("ludic.envs.code_exec.backend.is_docker_available", return_value=False):
                with patch("shutil.which", return_value="/usr/bin/podman-hpc"):
                    backend = detect_available_backend()
                    assert backend == "podman-hpc"

    def test_detect_backend_none_available_raises(self):
        """Should raise RuntimeError when no backend is available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SLURM_JOB_ID", None)
            with patch("ludic.envs.code_exec.backend.is_docker_available", return_value=False):
                with patch("shutil.which", return_value=None):
                    with pytest.raises(RuntimeError, match="No sandbox backend available"):
                        detect_available_backend()

    def test_get_backend_info(self):
        """Test get_backend_info returns structured data."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "99999"}):
            with patch("ludic.envs.code_exec.backend.is_docker_available", return_value=False):
                with patch("shutil.which", return_value="/usr/bin/podman-hpc"):
                    info = get_backend_info()

                    assert info["environment"]["in_slurm"] is True
                    assert info["environment"]["slurm_job_id"] == "99999"
                    assert "docker" in info["backends"]
                    assert "podman-hpc" in info["backends"]
                    assert info["backends"]["podman-hpc"]["available"] is True
                    assert info["backends"]["docker"]["available"] is False


# ============================================================================
# Factory tests
# ============================================================================


class TestFactory:
    """Tests for create_sandbox_pool factory."""

    @pytest.mark.asyncio
    async def test_factory_unknown_backend_raises(self):
        """Factory should raise ValueError for unknown backend."""
        from ludic.envs.code_exec.factory import create_sandbox_pool

        with pytest.raises(ValueError, match="Unknown backend"):
            await create_sandbox_pool(backend="unknown")

    @pytest.mark.asyncio
    async def test_factory_singularity_not_implemented(self):
        """Factory should raise NotImplementedError for singularity."""
        from ludic.envs.code_exec.factory import create_sandbox_pool

        with pytest.raises(NotImplementedError, match="Singularity backend is not yet implemented"):
            await create_sandbox_pool(backend="singularity")

"""PVG vLLM Server Setup.

This module provides utilities for setting up dual vLLM servers for PVG training,
with configurable memory allocation for prover and verifier models.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DualVLLMConfig:
    """Configuration for dual vLLM server setup.

    Args:
        prover_port: Port for prover vLLM server
        verifier_port: Port for verifier vLLM server
        prover_gpu_memory: GPU memory fraction for prover (0-1)
        verifier_gpu_memory: GPU memory fraction for verifier (0-1)
        gpu_id: GPU ID to use (for single-GPU setups)
        prover_gpu_ids: List of GPU IDs for prover (overrides gpu_id)
        verifier_gpu_ids: List of GPU IDs for verifier (overrides gpu_id)
        host: Server host address
        prover_tensor_parallel: Tensor parallelism for prover
        verifier_tensor_parallel: Tensor parallelism for verifier
        max_model_len: Maximum model context length
        enable_weight_updates: Whether to enable weight update endpoints
    """

    prover_port: int = 8000
    verifier_port: int = 8001
    prover_gpu_memory: float = 0.77
    verifier_gpu_memory: float = 0.18
    gpu_id: int = 0
    prover_gpu_ids: Optional[list] = None
    verifier_gpu_ids: Optional[list] = None
    host: str = "127.0.0.1"
    prover_tensor_parallel: int = 1
    verifier_tensor_parallel: int = 1
    max_model_len: int = 4096
    enable_weight_updates: bool = True

    def __post_init__(self) -> None:
        if self.prover_gpu_memory + self.verifier_gpu_memory > 1.0:
            logger.warning(
                f"Combined GPU memory ({self.prover_gpu_memory + self.verifier_gpu_memory:.2f}) "
                "exceeds 1.0. This may cause OOM errors on single-GPU setups."
            )


async def setup_dual_vllm(
    prover_model: str,
    verifier_model: str,
    config: DualVLLMConfig,
    *,
    wait_for_ready: bool = True,
    timeout_s: float = 300.0,
) -> Tuple[Any, Any]:
    """Set up dual vLLM servers for prover and verifier.

    Starts two vLLM servers with configured memory allocation and returns
    clients for each. This is the main entry point for PVG inference setup.

    Args:
        prover_model: HuggingFace model ID or path for prover
        verifier_model: HuggingFace model ID or path for verifier
        config: DualVLLMConfig with server settings
        wait_for_ready: Wait for servers to be ready before returning
        timeout_s: Timeout in seconds for server startup

    Returns:
        Tuple of (prover_client, verifier_client)

    Raises:
        TimeoutError: If servers don't become ready within timeout
        RuntimeError: If server startup fails
    """
    logger.info("Setting up dual vLLM servers")
    logger.info(f"  Prover: {prover_model} on port {config.prover_port}")
    logger.info(f"  Verifier: {verifier_model} on port {config.verifier_port}")

    # Start servers sequentially to avoid HuggingFace cache lock contention
    # when using the same model for both prover and verifier
    logger.info("Starting prover server first (to avoid cache lock contention)...")
    prover_process = await _start_vllm_server(
        model=prover_model,
        port=config.prover_port,
        gpu_memory=config.prover_gpu_memory,
        gpu_ids=config.prover_gpu_ids or [config.gpu_id],
        tensor_parallel=config.prover_tensor_parallel,
        host=config.host,
        max_model_len=config.max_model_len,
        enable_weight_updates=config.enable_weight_updates,
        server_name="prover",
    )

    # Wait for prover to be ready before starting verifier
    if wait_for_ready:
        logger.info("Waiting for prover server to be ready...")
        await _wait_for_server_ready(
            f"http://{config.host}:{config.prover_port}",
            timeout_s=timeout_s,
            name="prover",
        )

    # Now start verifier (cache should be warm, no lock contention)
    logger.info("Starting verifier server...")
    verifier_process = await _start_vllm_server(
        model=verifier_model,
        port=config.verifier_port,
        gpu_memory=config.verifier_gpu_memory,
        gpu_ids=config.verifier_gpu_ids or [config.gpu_id],
        tensor_parallel=config.verifier_tensor_parallel,
        host=config.host,
        max_model_len=config.max_model_len,
        enable_weight_updates=config.enable_weight_updates,
        server_name="verifier",
    )

    # Wait for verifier to be ready
    if wait_for_ready:
        logger.info("Waiting for verifier server to be ready...")
        await _wait_for_server_ready(
            f"http://{config.host}:{config.verifier_port}",
            timeout_s=timeout_s,
            name="verifier",
        )

    # Create clients
    prover_client = await _create_vllm_client(
        host=config.host,
        port=config.prover_port,
        model=prover_model,
    )

    verifier_client = await _create_vllm_client(
        host=config.host,
        port=config.verifier_port,
        model=verifier_model,
    )

    logger.info("Dual vLLM servers ready")
    return prover_client, verifier_client


async def _start_vllm_server(
    model: str,
    port: int,
    gpu_memory: float,
    gpu_ids: list,
    tensor_parallel: int,
    host: str,
    max_model_len: int,
    enable_weight_updates: bool,
    server_name: str,
) -> Any:
    """Start a single vLLM server.

    Args:
        model: Model path or HuggingFace ID
        port: Server port
        gpu_memory: GPU memory fraction
        gpu_ids: List of GPU IDs
        tensor_parallel: Tensor parallelism degree
        host: Server host
        max_model_len: Maximum model length
        enable_weight_updates: Enable weight update endpoints
        server_name: Name for logging

    Returns:
        Server process handle
    """
    try:
        from ludic.inference.vllm_utils import start_vllm_server
    except ImportError:
        logger.warning("vllm_utils not available, using mock server")
        return _MockServerProcess(server_name)

    # Build environment for GPU visibility
    import os
    cuda_devices = ",".join(str(g) for g in gpu_ids)

    # Build server args
    # Note: weight sync endpoints are built into ludic.inference.vllm_server,
    # no need for --enable-weight-updates flag
    extra_args = [
        f"--gpu-memory-utilization={gpu_memory}",
        f"--max-model-len={max_model_len}",
    ]

    if tensor_parallel > 1:
        extra_args.append(f"--tensor-parallel-size={tensor_parallel}")

    logger.info(f"Starting {server_name} vLLM server on port {port}")
    logger.info(f"  GPU memory: {gpu_memory:.2f}, GPUs: {cuda_devices}")

    # start_vllm_server is synchronous; run in thread to not block event loop
    process = await asyncio.to_thread(
        start_vllm_server,
        model=model,
        host=host,
        port=port,
        extra_args=extra_args,
        cuda_visible_devices=cuda_devices,
    )

    return process


async def _wait_for_server_ready(
    base_url: str,
    timeout_s: float,
    name: str,
    process: Any = None,
) -> None:
    """Wait for a vLLM server to be ready.

    Args:
        base_url: Server base URL
        timeout_s: Timeout in seconds
        name: Server name for logging
        process: Optional subprocess.Popen to monitor for early exit

    Raises:
        TimeoutError: If server doesn't become ready within timeout
        RuntimeError: If server process dies before becoming ready
    """
    import time

    try:
        import requests
    except ImportError:
        logger.info(f"requests not available, assuming {name} server ready")
        await asyncio.sleep(2.0)
        return

    health_url = f"{base_url}/health"
    deadline = time.time() + timeout_s
    last_err = None

    while time.time() < deadline:
        # Check if process died (captures vLLM startup errors)
        if process is not None and hasattr(process, "poll"):
            exit_code = process.poll()
            if exit_code is not None:
                # Process died - try to get error output
                stderr_output = ""
                stdout_output = ""
                if hasattr(process, "stderr") and process.stderr:
                    try:
                        stderr_output = process.stderr.read() or ""
                    except Exception:
                        pass
                if hasattr(process, "stdout") and process.stdout:
                    try:
                        stdout_output = process.stdout.read() or ""
                    except Exception:
                        pass

                error_msg = (
                    f"{name} server process died with exit code {exit_code}\n"
                    f"STDERR:\n{stderr_output}\n"
                    f"STDOUT:\n{stdout_output}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        try:
            r = await asyncio.to_thread(requests.get, health_url, timeout=2.0)
            if r.status_code == 200:
                logger.info(f"{name} server ready at {base_url}")
                return
        except Exception as e:
            last_err = e

        await asyncio.sleep(2.0)

    # Timeout reached - kill process if running
    if process is not None and hasattr(process, "terminate"):
        process.terminate()
        # Try to get any error output
        stderr_output = ""
        if hasattr(process, "stderr") and process.stderr:
            try:
                stderr_output = process.stderr.read() or ""
            except Exception:
                pass
        logger.error(f"{name} server stderr on timeout:\n{stderr_output}")

    raise TimeoutError(
        f"{name} server failed to become healthy at {health_url} "
        f"within {timeout_s}s. Last error: {last_err}"
    )


async def _create_vllm_client(
    host: str,
    port: int,
    model: str,
) -> Any:
    """Create a vLLM client for the server.

    Args:
        host: Server host
        port: Server port
        model: Model name

    Returns:
        VLLMChatClient instance
    """
    try:
        from ludic.inference.vllm_client import VLLMChatClient

        client = VLLMChatClient(
            base_url=f"http://{host}:{port}/v1",
            model=model,
        )
        return client
    except ImportError:
        logger.warning("VLLMChatClient not available, using mock client")
        return _MockVLLMClient(host, port, model)


class _MockServerProcess:
    """Mock server process for testing without vLLM."""

    def __init__(self, name: str) -> None:
        self.name = name
        logger.info(f"Created mock {name} server process")

    def terminate(self) -> None:
        logger.info(f"Terminated mock {self.name} server")


class _MockVLLMClient:
    """Mock vLLM client for testing without vLLM."""

    def __init__(self, host: str, port: int, model: str) -> None:
        self.host = host
        self.port = port
        self.model = model
        logger.info(f"Created mock vLLM client for {model} at {host}:{port}")

    async def chat(
        self,
        messages: list,
        **kwargs,
    ) -> Dict[str, Any]:
        """Mock chat response."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"code": "def solution(): pass", "certificate": "42"}',
                    }
                }
            ]
        }


def create_prover_publisher(
    client: Any,
    config: DualVLLMConfig,
) -> Any:
    """Create a policy publisher for the prover vLLM server.

    Args:
        client: VLLMChatClient for the prover
        config: DualVLLMConfig

    Returns:
        PolicyPublisher for weight syncing
    """
    try:
        from ludic.distributed.adapters.vllm import VLLMPolicyPublisher

        return VLLMPolicyPublisher(
            client=client,
            host=config.host,
            port=config.prover_port,
        )
    except ImportError:
        logger.warning("VLLMPolicyPublisher not available, using mock")
        return _MockPublisher()


class _MockPublisher:
    """Mock publisher for testing."""

    def publish(self, params: Dict[str, Any], version: int = 0) -> None:
        logger.debug(f"Mock publish: {len(params)} params at version {version}")


async def shutdown_dual_vllm(
    prover_process: Any,
    verifier_process: Any,
) -> None:
    """Shutdown dual vLLM servers.

    Args:
        prover_process: Prover server process
        verifier_process: Verifier server process
    """
    logger.info("Shutting down vLLM servers...")

    for name, process in [("prover", prover_process), ("verifier", verifier_process)]:
        try:
            if hasattr(process, "terminate"):
                process.terminate()
                if hasattr(process, "wait"):
                    await asyncio.wait_for(
                        asyncio.to_thread(process.wait),
                        timeout=10.0,
                    )
            logger.info(f"Stopped {name} server")
        except Exception as e:
            logger.warning(f"Error stopping {name} server: {e}")

    logger.info("vLLM servers shutdown complete")

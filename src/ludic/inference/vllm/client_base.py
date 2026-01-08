"""Shared client infrastructure for vLLM-based inference clients.

This module provides reusable components for vLLM clients:

- NCCLCommunicator: Wrapper managing NCCL communicator lifecycle
- check_server_health(): Poll server health endpoint with retry
- sync_weights_batch(): Core weight synchronization logic

These components are used by VLLMClient for both policy and reward model
weight synchronization.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
from requests import Session
from requests.exceptions import RequestException, Timeout
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NCCL Communicator Wrapper
# ---------------------------------------------------------------------------


@dataclass
class NCCLCommunicator:
    """Wrapper managing NCCL communicator lifecycle for a vLLM server connection.

    This class encapsulates all the state and logic needed to establish and
    use an NCCL communicator for weight synchronization with a vLLM server.

    Attributes:
        host: Server hostname.
        http_port: HTTP port for control plane.
        nccl_port: TCP port for NCCL process group.
        device: CUDA device for tensors.
        session: requests.Session for HTTP calls.
        comm: The PyNcclCommunicator instance (None until initialized).
        rank: This client's rank in the NCCL group.
        server_world_size: Number of workers in the vLLM server.
    """

    host: str
    http_port: int
    nccl_port: int
    device: Union[str, torch.device, int]
    session: Session

    # State (populated after init_communicator)
    comm: Optional[PyNcclCommunicator] = field(default=None, init=False)
    rank: Optional[int] = field(default=None, init=False)
    server_world_size: Optional[int] = field(default=None, init=False)

    @property
    def server_url(self) -> str:
        """Base URL for HTTP requests."""
        return f"http://{self.host}:{self.http_port}"

    @property
    def is_initialized(self) -> bool:
        """Whether the communicator has been initialized."""
        return self.comm is not None and self.rank is not None

    def initialize(self, timeout_s: float = 30.0) -> None:
        """Initialize the NCCL communicator.

        Steps:
        1. Query world size from server
        2. Tell server workers to initialize their communicators
        3. Create client-side NCCL process group

        Args:
            timeout_s: Timeout for HTTP requests.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self.comm is not None:
            raise RuntimeError("Communicator already initialized")

        # 1. Query world size from server
        r = self.session.get(f"{self.server_url}/get_world_size", timeout=timeout_s)
        r.raise_for_status()
        self.server_world_size = r.json()["world_size"]
        world_size = self.server_world_size + 1  # Client is the extra rank
        self.rank = self.server_world_size

        # 2. Ask server workers to init their communicators
        r = self.session.post(
            f"{self.server_url}/init_communicator",
            json={"host": self.host, "port": self.nccl_port, "world_size": world_size},
            timeout=timeout_s,
        )
        r.raise_for_status()

        # Brief pause to let server initialize NCCL
        time.sleep(0.1)

        # 3. Create the matching client-side communicator
        pg = StatelessProcessGroup.create(
            host=self.host,
            port=self.nccl_port,
            rank=self.rank,
            world_size=world_size,
        )
        self.comm = PyNcclCommunicator(pg, device=self.device)

        log.info(
            f"NCCL communicator initialized: rank={self.rank}, "
            f"world_size={world_size}, server_url={self.server_url}"
        )

    def close(self) -> None:
        """Close the communicator and notify the server.

        Safe to call multiple times.
        """
        if self.comm is None:
            return

        # Notify server (best effort)
        try:
            r = self.session.post(
                f"{self.server_url}/close_communicator", timeout=10.0
            )
            if r.status_code != 200:
                log.warning(
                    f"close_communicator responded with {r.status_code}: {r.text}"
                )
        except RequestException:
            # Server may already be down
            pass

        # Clean up local state
        del self.comm
        self.comm = None
        self.rank = None

    def broadcast(self, tensor: torch.Tensor) -> None:
        """Broadcast a tensor from this client to all server workers.

        Args:
            tensor: Tensor to broadcast. Must be on the correct device.

        Raises:
            RuntimeError: If communicator not initialized.
        """
        if self.comm is None or self.rank is None:
            raise RuntimeError("Communicator not initialized")
        self.comm.broadcast(tensor, src=self.rank)

    def barrier(self) -> None:
        """Synchronization barrier across all ranks.

        Raises:
            RuntimeError: If communicator not initialized.
        """
        if self.comm is None:
            raise RuntimeError("Communicator not initialized")
        self.comm.group.barrier()


# ---------------------------------------------------------------------------
# Health Check Utility
# ---------------------------------------------------------------------------


def check_server_health(
    session: Session,
    server_url: str,
    *,
    total_timeout: float = 60.0,
    retry_interval: float = 2.0,
) -> Dict[str, Any]:
    """Poll a server's /health endpoint until it responds OK.

    Args:
        session: requests.Session for HTTP calls.
        server_url: Base URL of the server (e.g., "http://localhost:8000").
        total_timeout: Maximum seconds to wait.
        retry_interval: Seconds between retries.

    Returns:
        Health response JSON (e.g., {"status": "ok", "type": "policy"}).

    Raises:
        ConnectionError: If server not reachable within timeout.
    """
    url = f"{server_url}/health"
    start_time = time.time()

    while True:
        try:
            r = session.get(url, timeout=5.0)
            if r.status_code == 200:
                log.info(f"Server is up at {server_url}")
                try:
                    return r.json()
                except Exception:
                    return {"status": "ok"}
        except RequestException:
            pass

        if total_timeout and (time.time() - start_time) >= total_timeout:
            raise ConnectionError(
                f"Server not reachable at {server_url} after {total_timeout} seconds"
            )

        log.info(f"Server not ready at {server_url}, retrying...")
        time.sleep(retry_interval)


# ---------------------------------------------------------------------------
# Weight Sync Logic
# ---------------------------------------------------------------------------


def sync_weights_batch(
    communicator: NCCLCommunicator,
    params: Mapping[str, torch.Tensor],
    *,
    endpoint: str = "/update_param_batch",
    timeout_s: float = 600.0,
    version: Optional[Union[str, int]] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
    get_background_tasks: Optional[Callable[[], int]] = None,
) -> str:
    """Push updated model parameters to a vLLM server via NCCL.

    This is the core weight synchronization logic shared between
    policy weight sync and reward model weight sync.

    Steps:
    1. Prepare metadata (sorted parameter names, dtypes, shapes)
    2. POST metadata to server (control plane)
    3. Stream tensors via NCCL broadcast (data plane)
    4. Barrier and wait for server tasks to complete

    Args:
        communicator: NCCLCommunicator for the target server.
        params: Mapping of parameter names to tensors.
        endpoint: Server endpoint for metadata (default: /update_param_batch).
        timeout_s: Maximum time for the operation.
        version: Optional version identifier for the weights.
        extra_payload: Additional fields to include in metadata POST.
        get_background_tasks: Optional callable returning pending task count.

    Returns:
        Version string (supplied or auto-generated).

    Raises:
        RuntimeError: If communicator not initialized or server rejects request.
        TimeoutError: If operation exceeds timeout_s.
    """
    if not communicator.is_initialized:
        raise RuntimeError("Communicator not initialized")

    start = time.time()

    # 1. Prepare metadata in deterministic order
    sorted_keys = sorted(params.keys())
    metadata: List[Dict[str, Any]] = []

    for name in sorted_keys:
        tensor = params[name]
        metadata.append({
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": tuple(tensor.shape),
        })

    # 2. Control plane: announce batch
    payload: Dict[str, Any] = {"metadata": metadata}
    if version is not None:
        payload["version"] = version
    if extra_payload:
        payload.update(extra_payload)

    url = f"{communicator.server_url}{endpoint}"

    try:
        resp = communicator.session.post(url, json=payload, timeout=timeout_s)
    except Timeout:
        raise TimeoutError("HTTP timeout during batch metadata send")
    except Exception as exc:
        raise RuntimeError(f"Error sending batch metadata: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Server rejected {endpoint}: {resp.status_code} {resp.text}"
        )

    # Wait for server to signal readiness for NCCL communication
    resp_data = resp.json()
    batch_id = resp_data.get("batch_id")

    if batch_id is not None:
        # Use the new ready-signal protocol
        ready = _wait_for_batch_ready(
            communicator.session,
            communicator.server_url,
            batch_id,
            timeout_s=min(5.0, timeout_s),
        )
        if not ready:
            log.warning(
                f"Batch {batch_id} ready signal timed out, falling back to fixed delay"
            )
            time.sleep(1.0)
    else:
        # Legacy server without batch_id support - use fixed delay
        log.debug("Server does not support ready-signal protocol, using fixed delay")
        time.sleep(1.0)

    # 3. Data plane: stream tensors via NCCL
    for name in sorted_keys:
        tensor = params[name]
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        communicator.broadcast(tensor)

    # 4. Synchronization
    communicator.barrier()

    if (time.time() - start) > timeout_s:
        raise TimeoutError(f"Weight sync exceeded {timeout_s}s")

    # Wait for server background tasks to drain
    if get_background_tasks is not None:
        drain_deadline = start + timeout_s
        while get_background_tasks() > 0:
            if time.time() > drain_deadline:
                raise TimeoutError(
                    f"Weight sync: background tasks did not complete within {timeout_s}s"
                )
            time.sleep(0.2)

    return str(version) if version is not None else f"sync-{int(time.time())}"


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def _wait_for_batch_ready(
    session: Session,
    server_url: str,
    batch_id: str,
    *,
    timeout_s: float = 5.0,
    poll_interval: float = 0.1,
) -> bool:
    """Poll the server until a batch is ready for NCCL communication.

    Args:
        session: requests.Session for HTTP calls.
        server_url: Base URL of the server.
        batch_id: Batch ID returned from /update_param_batch.
        timeout_s: Maximum seconds to wait for ready signal.
        poll_interval: Seconds between polls.

    Returns:
        True if batch is ready, False if timeout exceeded.
    """
    url = f"{server_url}/batch_ready/{batch_id}"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            resp = session.get(url, timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ready", False):
                    return True
        except RequestException:
            # Server may be temporarily unavailable
            pass

        time.sleep(poll_interval)

    return False


def get_server_background_tasks(session: Session, server_url: str) -> int:
    """Query pending background tasks from a server.

    Args:
        session: requests.Session for HTTP calls.
        server_url: Base URL of the server.

    Returns:
        Number of pending background tasks.
    """
    r = session.post(f"{server_url}/get_num_background_tasks", timeout=10.0)
    r.raise_for_status()
    return r.json()["num_background_tasks"]


def reset_server_prefix_cache(session: Session, server_url: str) -> None:
    """Reset KV/prefix caches on a server.

    Args:
        session: requests.Session for HTTP calls.
        server_url: Base URL of the server.
    """
    r = session.post(f"{server_url}/reset_prefix_cache", timeout=30.0)
    r.raise_for_status()

"""Shared server infrastructure for vLLM-based servers.

This module provides reusable components that are common between the policy
server (vllm_server.py) and reward server (vllm_reward_server.py):

- ServerState: Container for global server state
- WeightSyncExtensionBase: Base class for vLLM worker extensions
- register_weight_sync_endpoints(): Factory to add weight sync endpoints to FastAPI
- create_background_task(): Task management with tracking
- run_server_lifecycle(): Common server startup/shutdown logic
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Set, Tuple, TYPE_CHECKING

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from vllm.engine.async_llm_engine import AsyncLLMEngine


# ---------------------------------------------------------------------------
# Weight Filtering by Training Mode
# ---------------------------------------------------------------------------

# Patterns for weight name matching
HEAD_PATTERNS = ("score.", "classifier.", "lm_head.")
LORA_PATTERNS = ("lora_", ".lora.")


def _filter_weights_by_mode(
    metadata: list[dict[str, Any]],
    training_mode: str,
) -> list[dict[str, Any]]:
    """Filter weight metadata based on training mode.

    Args:
        metadata: List of weight metadata dicts with "name", "dtype", "shape" keys.
        training_mode: One of "full", "head_only", or "lora".

    Returns:
        Filtered list of metadata dicts for weights that should be updated.
    """
    if training_mode == "full":
        return metadata
    elif training_mode == "head_only":
        return [
            m for m in metadata
            if any(p in m["name"] for p in HEAD_PATTERNS)
        ]
    elif training_mode == "lora":
        patterns = HEAD_PATTERNS + LORA_PATTERNS
        return [
            m for m in metadata
            if any(p in m["name"] for p in patterns)
        ]
    # Unknown mode - default to full update
    return metadata


# ---------------------------------------------------------------------------
# Server State Container
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    """Container for mutable server state.

    Using a dataclass instead of module-level globals makes testing easier
    and allows multiple server instances (e.g., in tests).

    Attributes:
        runtime_version: Monotonically increasing version number, bumped on weight update.
        background_tasks: Set of tracked async tasks for graceful shutdown.
        weight_update_semaphore: Limits concurrent weight update operations.
        version_lock: Async lock protecting runtime_version updates.
        pending_batches: Mapping of batch_id to asyncio.Event for ready signaling.
        batch_counter: Counter for generating unique batch IDs.
    """

    runtime_version: int = 0
    background_tasks: Set[asyncio.Task[Any]] = field(default_factory=set)
    weight_update_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(10)
    )
    version_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_batches: dict[str, asyncio.Event] = field(default_factory=dict)
    batch_counter: int = 0

    def create_background_task(
        self, coro: Coroutine[Any, Any, Any]
    ) -> asyncio.Task[Any]:
        """Create an async task and track it for graceful shutdown."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def create_batch_id(self) -> str:
        """Generate a unique batch ID for weight sync coordination."""
        self.batch_counter += 1
        return f"batch-{self.batch_counter}"

    def create_pending_batch(self, batch_id: str) -> asyncio.Event:
        """Create an Event for a pending batch and register it."""
        event = asyncio.Event()
        self.pending_batches[batch_id] = event
        return event

    def mark_batch_ready(self, batch_id: str) -> None:
        """Signal that a batch is ready for NCCL communication."""
        event = self.pending_batches.get(batch_id)
        if event is not None:
            event.set()

    def cleanup_batch(self, batch_id: str) -> None:
        """Remove a batch from pending tracking."""
        self.pending_batches.pop(batch_id, None)

    async def shutdown(self) -> None:
        """Cancel all background tasks and wait for them to complete."""
        for task in list(self.background_tasks):
            task.cancel()
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)


# Module-level convenience function using default state
_default_state: Optional[ServerState] = None


def get_default_state() -> ServerState:
    """Get or create the default server state singleton."""
    global _default_state
    if _default_state is None:
        _default_state = ServerState()
    return _default_state


def create_background_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Create a background task using the default state."""
    return get_default_state().create_background_task(coro)


# ---------------------------------------------------------------------------
# Worker Extension Base Class
# ---------------------------------------------------------------------------


class WeightSyncExtensionBase:
    """Base class for vLLM worker extensions providing NCCL weight synchronization.

    Subclasses should inherit from this and can override methods for
    server-specific behavior (e.g., custom logging, validation).

    Each worker:
      - joins a StatelessProcessGroup (TCP)
      - wraps it in a PyNcclCommunicator (NCCL)
      - receives updated weights via broadcast() from the client rank
      - calls `model_runner.model.load_weights` with the new tensors

    Attributes:
        pynccl_comm: The NCCL communicator instance (None until initialized).
        client_rank: The rank of the training client in the NCCL group.
        device: The CUDA device for this worker.
    """

    pynccl_comm: PyNcclCommunicator | None = None
    client_rank: int | None = None
    device: torch.device | None = None

    # Optional: Override in subclass for custom logging prefix
    log_prefix: str = "[Worker]"

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """Initialize NCCL communicator for weight updates.

        Called via engine.collective_rpc on all workers.

        Args:
            host: TCP host for the process group.
            port: TCP port for the process group.
            world_size: Total number of ranks (workers + client).
        """
        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialized. "
                "Call close_communicator() first."
            )

        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
        )
        assert self.device is not None, f"{self.__class__.__name__}.device must be set"
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        # Client rank is the last rank in the world (training process)
        self.client_rank = world_size - 1

        # Debug: print parameter names on rank 0
        if self.pynccl_comm.rank == 0:
            self._debug_print_parameters()

    def _debug_print_parameters(self) -> None:
        """Print model parameter names for debugging. Override for custom behavior."""
        print("\n" + "=" * 60)
        print(f"{self.log_prefix} vLLM Model Parameter Names (Worker Rank 0)")
        print("=" * 60)
        try:
            model_instance = self.model_runner.model  # type: ignore[attr-defined]
            count = 0
            for name, _ in model_instance.named_parameters():
                print(f"   • {name}")
                count += 1
            print(f"Total parameters found: {count}")
        except Exception as e:
            print(f"⚠️ Could not print parameter names: {e}")
        print("=" * 60 + "\n")

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """Receive a single parameter tensor via NCCL broadcast and load it.

        Called via engine.collective_rpc on all workers.

        Args:
            name: Parameter name in the model.
            dtype: String representation of tensor dtype (e.g., "torch.float16").
            shape: Tensor shape as a sequence of ints.
        """
        if self.pynccl_comm is None or self.client_rank is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )

        torch_dtype = getattr(torch, dtype.split(".")[-1])
        assert self.device is not None
        weight = torch.empty(shape, dtype=torch_dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()
        # vLLM model runner handles weight application
        self.model_runner.model.load_weights(weights=[(name, weight)])  # type: ignore[attr-defined]

    def update_param_batch(
        self, metadata_list: Sequence[Tuple[str, str, Sequence[int]]]
    ) -> None:
        """Receive a batch of parameters via NCCL and load them.

        Called via engine.collective_rpc on all workers.

        Args:
            metadata_list: List of (name, dtype, shape) tuples describing each tensor.
        """
        if self.pynccl_comm is None or self.client_rank is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )

        assert self.device is not None

        for name, dtype_str, shape_list in metadata_list:
            torch_dtype = getattr(torch, dtype_str.split(".")[-1])
            shape = tuple(shape_list)

            # Allocate empty tensor on GPU
            weight = torch.empty(shape, dtype=torch_dtype, device=self.device)

            # NCCL receive
            self.pynccl_comm.broadcast(weight, src=self.client_rank)

            # Apply weight - vLLM handles QKV fusion, TP sharding, etc.
            self.model_runner.model.load_weights(weights=[(name, weight)])  # type: ignore[attr-defined]

        # Barrier to ensure all workers are done
        self.pynccl_comm.group.barrier()

    def close_communicator(self) -> None:
        """Tear down NCCL communicator state."""
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


# ---------------------------------------------------------------------------
# Endpoint Registration
# ---------------------------------------------------------------------------


def register_weight_sync_endpoints(
    app: "FastAPI",
    engine: "AsyncLLMEngine",
    state: ServerState,
    *,
    get_world_size: Callable[[], int],
    server_type: str = "policy",
    log_prefix: str = "[Server]",
) -> None:
    """Register weight synchronization endpoints on a FastAPI app.

    This factory function adds the common endpoints needed for NCCL-based
    weight synchronization to any FastAPI application.

    Endpoints registered:
    - GET /health - Health check
    - GET /get_world_size - Return tensor parallel world size
    - GET /runtime_version - Current weight version
    - POST /init_communicator - Initialize NCCL communicator
    - POST /update_named_param - Update single parameter
    - POST /update_param_batch - Update batch of parameters
    - POST /sync_weights - Legacy batched update
    - POST /reset_prefix_cache - Clear KV cache
    - POST /get_num_background_tasks - Query pending tasks
    - POST /close_communicator - Tear down NCCL

    Args:
        app: FastAPI application instance.
        engine: vLLM AsyncLLMEngine instance.
        state: ServerState container for runtime state.
        get_world_size: Callable returning the tensor parallel world size.
        server_type: Server type for health endpoint (e.g., "policy", "reward_model").
        log_prefix: Prefix for log messages.
    """
    from fastapi import Request

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "type": server_type}

    @app.get("/get_world_size")
    async def get_world_size_endpoint() -> dict[str, int]:
        return {"world_size": get_world_size()}

    @app.get("/runtime_version")
    async def runtime_version() -> dict[str, int]:
        return {"version": state.runtime_version}

    @app.post("/init_communicator")
    async def init_communicator(request: Request) -> dict[str, str]:
        """Initialize NCCL communicator on all workers."""
        data = await request.json()
        host = data.get("host")
        port = data.get("port")
        world_size = data.get("world_size")

        state.create_background_task(
            engine.collective_rpc("init_communicator", args=(host, port, world_size))
        )
        return {"status": "ok"}

    @app.post("/update_named_param")
    async def update_named_param(request: Request) -> dict[str, str]:
        """Update a single named parameter via NCCL."""
        data = await request.json()
        name = data.get("name")
        dtype = data.get("dtype")
        shape = tuple(data.get("shape"))

        async def throttled_update() -> None:
            async with state.weight_update_semaphore:
                await engine.collective_rpc(
                    "update_named_param", args=(name, dtype, shape)
                )

        state.create_background_task(throttled_update())
        return {"status": "ok"}

    @app.post("/update_param_batch")
    async def update_param_batch(request: Request) -> dict[str, Any]:
        """Receive batch metadata and trigger NCCL weight sync.

        Returns a batch_id that can be used with /batch_ready/{batch_id}
        to poll for NCCL receiver readiness before broadcasting.
        """
        data = await request.json()
        metadata_raw = data.get("metadata", [])
        forced_version = data.get("version")
        training_mode = data.get("training_mode", "full")

        # Filter weights based on training mode (HEAD_ONLY, LORA, or FULL)
        metadata = _filter_weights_by_mode(metadata_raw, training_mode)
        skipped = len(metadata_raw) - len(metadata)

        # Generate batch ID for ready-signal coordination
        batch_id = state.create_batch_id()
        ready_event = state.create_pending_batch(batch_id)

        print(f"\n{log_prefix} Receiving {len(metadata)} weights (mode: {training_mode}, batch: {batch_id})")
        if skipped > 0:
            print(f"{log_prefix} Filtered out {skipped} weights based on training mode")
        print("=" * 60)
        for i, m in enumerate(metadata):
            if i < 10:
                print(f"  • {m.get('name')} | {m.get('shape')}")
        if len(metadata) > 10:
            print(f"  ... (+{len(metadata) - 10} more)")
        print("=" * 60 + "\n")

        # Convert dicts to tuples for RPC serialization
        rpc_args = [(m["name"], m["dtype"], m["shape"]) for m in metadata]

        async def do_update_batch() -> None:
            try:
                async with state.weight_update_semaphore:
                    # Signal ready before dispatching RPC - at this point the
                    # request has been validated and workers will be notified.
                    # The actual NCCL broadcast will block until client sends.
                    state.mark_batch_ready(batch_id)

                    await engine.collective_rpc("update_param_batch", args=(rpc_args,))
                    await engine.reset_prefix_cache()

                    async with state.version_lock:
                        if forced_version is not None:
                            state.runtime_version = int(forced_version)
                        else:
                            state.runtime_version += 1

                    print(f"{log_prefix} Weights updated, version={state.runtime_version}")
            finally:
                # Clean up the batch tracking
                state.cleanup_batch(batch_id)

        state.create_background_task(do_update_batch())
        return {"status": "ok", "batch_id": batch_id}

    @app.get("/batch_ready/{batch_id}")
    async def batch_ready(batch_id: str) -> dict[str, Any]:
        """Check if a batch is ready for NCCL communication.

        Returns:
            {"ready": True/False, "batch_id": str}

        The client should poll this endpoint after receiving a batch_id from
        /update_param_batch, waiting until ready=True before broadcasting
        tensors via NCCL.
        """
        event = state.pending_batches.get(batch_id)
        if event is None:
            # Unknown batch - either already cleaned up or never existed
            return {"ready": True, "batch_id": batch_id, "note": "unknown_batch"}

        # Wait briefly for the event (non-blocking check with short timeout)
        try:
            await asyncio.wait_for(event.wait(), timeout=0.1)
            return {"ready": True, "batch_id": batch_id}
        except asyncio.TimeoutError:
            return {"ready": False, "batch_id": batch_id}

    @app.post("/sync_weights")
    async def sync_weights(request: Request) -> dict[str, Any]:
        """Legacy batched update endpoint."""
        data = await request.json()
        params = data.get("params", [])
        requested_version = data.get("version")

        async def do_update() -> None:
            async with state.weight_update_semaphore:
                for p in params:
                    name = p["name"]
                    dtype = p["dtype"]
                    shape = tuple(p["shape"])
                    await engine.collective_rpc(
                        "update_named_param", args=(name, dtype, shape)
                    )

                async with state.version_lock:
                    if requested_version is not None:
                        try:
                            state.runtime_version = int(requested_version)
                        except ValueError:
                            state.runtime_version += 1
                    else:
                        state.runtime_version += 1

        state.create_background_task(do_update())
        return {"status": "ok", "requested_version": requested_version}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> dict[str, str]:
        """Reset KV/prefix caches."""
        state.create_background_task(engine.reset_prefix_cache())
        return {"status": "ok"}

    @app.post("/get_num_background_tasks")
    async def get_num_background_tasks() -> dict[str, int]:
        return {"num_background_tasks": len(state.background_tasks)}

    @app.post("/close_communicator")
    async def close_communicator() -> dict[str, str]:
        """Tear down NCCL communicator on all workers."""
        await engine.collective_rpc("close_communicator")
        return {"status": "ok"}


# ---------------------------------------------------------------------------
# Server Lifecycle Helpers
# ---------------------------------------------------------------------------


async def run_server_lifecycle(
    state: ServerState,
    serve_coro: Coroutine[Any, Any, Any],
    cleanup_fn: Optional[Callable[[], Coroutine[Any, Any, None]]] = None,
) -> None:
    """Run a server with proper lifecycle management.

    This helper handles:
    - Running the serve coroutine
    - Graceful shutdown of background tasks
    - Optional custom cleanup

    Args:
        state: ServerState containing background tasks to clean up.
        serve_coro: The main serve coroutine (e.g., from serve_http).
        cleanup_fn: Optional async cleanup function to call on shutdown.
    """
    try:
        await serve_coro
    finally:
        await state.shutdown()
        if cleanup_fn is not None:
            await cleanup_fn()

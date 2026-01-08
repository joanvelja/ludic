"""Shared vLLM infrastructure for Ludic servers and clients.

This package provides reusable components for vLLM-based inference:

Server-side:
- `ServerState`: Container for global server state (version, tasks, semaphore)
- `WeightSyncExtensionBase`: Base class for vLLM worker extensions
- `register_weight_sync_endpoints()`: Factory to add weight sync endpoints to FastAPI

Client-side:
- `NCCLCommunicator`: Wrapper for NCCL communicator management
- `WeightSyncClientMixin`: Mixin providing sync_weights() logic

Usage:
    from ludic.inference.vllm import (
        ServerState,
        WeightSyncExtensionBase,
        register_weight_sync_endpoints,
        NCCLCommunicator,
    )
"""

from ludic.inference.vllm.server_base import (
    ServerState,
    WeightSyncExtensionBase,
    register_weight_sync_endpoints,
    create_background_task,
    run_server_lifecycle,
)

from ludic.inference.vllm.client_base import (
    NCCLCommunicator,
    check_server_health,
    sync_weights_batch,
)

__all__ = [
    # Server
    "ServerState",
    "WeightSyncExtensionBase",
    "register_weight_sync_endpoints",
    "create_background_task",
    "run_server_lifecycle",
    # Client
    "NCCLCommunicator",
    "check_server_health",
    "sync_weights_batch",
]

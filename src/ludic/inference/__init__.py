from __future__ import annotations

from .client import ChatClient, VersionedClient
from .sampling import SamplingParams
from .request import ChatCompletionRequest, InferenceSpec, ReturnSpec, ToolRequest
from .extensions import BackendExtensions, VLLMExtensions
from .vllm_client import VLLMClient, VLLMChatClient  # VLLMChatClient is alias for compat
from .vllm_utils import start_vllm_server, wait_for_vllm_health
from .reward_types import (
    PoolingType,
    RewardModelTrainingMode,
    ScoringRequest,
    ScoringResponse,
)

__all__ = [
    "ChatClient",
    "VersionedClient",
    "SamplingParams",
    "ReturnSpec",
    "ToolRequest",
    "BackendExtensions",
    "VLLMExtensions",
    "InferenceSpec",
    "ChatCompletionRequest",
    # Unified vLLM client (supports both policy generation and RM scoring)
    "VLLMClient",
    "VLLMChatClient",  # Alias for backward compatibility
    "start_vllm_server",
    "wait_for_vllm_health",
    # Reward model types
    "PoolingType",
    "RewardModelTrainingMode",
    "ScoringRequest",
    "ScoringResponse",
]

"""Unified vLLM client for policy generation AND reward model scoring.

This module provides a single VLLMClient class that supports:
- Chat completions via OpenAI-compatible API (policy generation)
- Reward model scoring via /score endpoint
- NCCL-based weight synchronization for both policy and RM servers

For single-server setups, use the default `port` parameter.
For dual-server setups (separate policy + RM), use `policy_port` and `scoring_port`.
"""

import atexit
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import requests
import aiohttp
import torch
from openai import AsyncOpenAI
from requests.adapters import HTTPAdapter
from requests.exceptions import Timeout

from ludic.types import ChatResponse
from ludic.inference.client import ChatClient
from ludic.inference.request import ChatCompletionRequest, ToolRequest
from ludic.inference.extensions import BackendExtensions, VLLMExtensions
from ludic.inference.reward_types import (
    PoolingType,
    RewardModelTrainingMode,
    ScoringRequest,
    ScoringResponse,
)

# Shared infrastructure
from ludic.inference.vllm.client_base import (
    NCCLCommunicator,
    check_server_health,
    sync_weights_batch,
    get_server_background_tasks,
)

log = logging.getLogger(__name__)


class VLLMClient(ChatClient):
    """Unified vLLM client for policy generation AND reward model scoring.

    Supports two modes of operation:

    **Single-server mode** (default):
        One vLLM server handles both policy/chat and scoring.
        Use `port` parameter for this mode.

    **Dual-server mode**:
        Separate servers for policy (chat completions) and scoring (reward model).
        Use `policy_port` and `scoring_port` parameters.

    Features:
      - Chat completions via OpenAI-compatible API (complete() method)
      - Reward model scoring via /score endpoint (score(), score_batch() methods)
      - NCCL-based weight updates for both policy and RM servers

    Modes:
      * inference-only (enable_weight_updates=False):
            - Only uses HTTP APIs.
            - No NCCL, no GPU expected on client side.
            - Weight updates are disabled.
      * training/update mode (enable_weight_updates=True):
            - Client becomes an additional NCCL rank.
            - Enables sync_weights() for policy and sync_reward_weights() for RM.

    Args:
        host:
            Hostname of the vLLM server(s). Defaults to "0.0.0.0".
        port:
            HTTP port for single-server mode (used for both policy and scoring).
            Defaults to 8000. Ignored if policy_port or scoring_port is set.
        policy_port:
            HTTP port for the policy server. If None, uses `port`.
        scoring_port:
            HTTP port for the scoring/reward model server. If None, uses `port`.
        group_port:
            TCP port for NCCL communicator (policy server weight sync).
            Only used when enable_weight_updates=True. Defaults to 51216.
        scoring_group_port:
            TCP port for NCCL communicator (scoring server weight sync).
            Only used when enable_weight_updates=True and dual-server mode.
            Defaults to 51217.
        connection_timeout_s:
            Maximum seconds to wait for server /health endpoint.
            Defaults to 60. Raises ConnectionError if exceeded.
        enable_weight_updates:
            If True, initialize NCCL communicators for weight sync.
        device:
            Device for NCCL communicator (e.g. "cuda:0", 0).
            Defaults to 0.
        model_name:
            Model name for scoring requests. Defaults to "default".

    Example (single-server):
        client = VLLMClient(host="0.0.0.0", port=8000)
        response, _ = await client.complete(request)
        scores = await client.score_batch(["text1", "text2"])

    Example (dual-server):
        client = VLLMClient(
            host="0.0.0.0",
            policy_port=8000,
            scoring_port=8001,
            enable_weight_updates=True,
        )
        response, _ = await client.complete(request)
        scores = await client.score_batch([response.text])
        client.sync_weights(policy_state_dict, version=1)
        client.sync_reward_weights(rm_state_dict, training_mode=RewardModelTrainingMode.FULL)
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        policy_port: Optional[int] = None,
        scoring_port: Optional[int] = None,
        group_port: int = 51216,
        scoring_group_port: int = 51217,
        connection_timeout_s: float = 60,
        enable_weight_updates: bool = False,
        device: Union[str, torch.device, int] = 0,
        model_name: str = "default",
    ) -> None:

        # Store configuration parameters
        self.host = host
        self.port = port
        self.policy_port = policy_port if policy_port is not None else port
        self.scoring_port = scoring_port if scoring_port is not None else port
        self.group_port = group_port
        self.scoring_group_port = scoring_group_port
        self.connection_timeout_s = connection_timeout_s
        self.enable_weight_updates = enable_weight_updates
        self.device = device
        self.model_name = model_name

        # Determine if we're in dual-server mode
        self._dual_server_mode = self.policy_port != self.scoring_port

        # AsyncOpenAI handles the OpenAI-compatible HTTP endpoints (policy server).
        self._async_client = AsyncOpenAI(
            base_url=f"http://{self.host}:{self.policy_port}/v1",
            api_key="local",
        )

        # Sync HTTP client for health checks, scoring, and weight-update metadata RPC
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Server URLs
        self.server_url = f"http://{self.host}:{self.policy_port}"  # Policy server
        self._scoring_url = f"http://{self.host}:{self.scoring_port}"  # Scoring server

        # NCCL communicators (using shared infrastructure)
        self._policy_comm: Optional[NCCLCommunicator] = None
        self._scoring_comm: Optional[NCCLCommunicator] = None

        # Cached aiohttp session for async HTTP calls (lazy-initialized)
        self._async_session: Optional[aiohttp.ClientSession] = None

        # Verify policy server is reachable
        check_server_health(
            self._session,
            self.server_url,
            total_timeout=self.connection_timeout_s,
        )

        # In dual-server mode, also check scoring server
        if self._dual_server_mode:
            check_server_health(
                self._session,
                self._scoring_url,
                total_timeout=self.connection_timeout_s,
            )

        # If weight updates are enabled, initialize policy communicator
        if self.enable_weight_updates:
            self._init_policy_communicator()
            atexit.register(self.close_communicator)

    # ─────────────────────────────────────────────────────────────────
    # Communicator Management
    # ─────────────────────────────────────────────────────────────────

    def _init_policy_communicator(self) -> None:
        """Initialize NCCL communicator for policy server."""
        self._policy_comm = NCCLCommunicator(
            host=self.host,
            http_port=self.policy_port,
            nccl_port=self.group_port,
            device=self.device,
            session=self._session,
        )
        self._policy_comm.initialize()

    def _init_scoring_communicator(self) -> None:
        """Initialize NCCL communicator for scoring server (lazy)."""
        if self._scoring_comm is not None:
            return  # Already initialized

        self._scoring_comm = NCCLCommunicator(
            host=self.host,
            http_port=self.scoring_port,
            nccl_port=self.scoring_group_port,
            device=self.device,
            session=self._session,
        )
        self._scoring_comm.initialize()

    def close_communicator(self) -> None:
        """Close all NCCL communicators and notify servers."""
        if self._policy_comm is not None:
            self._policy_comm.close()
            self._policy_comm = None

        if self._scoring_comm is not None:
            self._scoring_comm.close()
            self._scoring_comm = None

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create a cached aiohttp ClientSession (lazy init, auto-recreate if closed)."""
        if self._async_session is None or self._async_session.closed:
            self._async_session = aiohttp.ClientSession()
        return self._async_session

    async def close_async_session(self) -> None:
        """Close the cached aiohttp session."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()
        self._async_session = None

    # ─────────────────────────────────────────────────────────────────
    # ChatClient.complete
    # ─────────────────────────────────────────────────────────────────

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        """High-level LLM invocation with vLLM extensions.

        Returns:
            (ChatResponse, info):
                ChatResponse contains:
                    .text
                    .completion_token_ids (may be None)
                    .prompt_token_ids (may be None)
                    .completion_logprobs (may be None)
                    .finish_reason
                'info' contains raw transport details and args actually sent.
        """

        # Sampling → OpenAI kwargs
        request_kwargs: Dict[str, Any] = dict(
            model=request.model,
            messages=request.messages,
        )
        request_kwargs.update(request.sampling.to_openai_kwargs())
        if request.seed is not None:
            request_kwargs["seed"] = int(request.seed)

        # Tools (OpenAI-style)
        if request.tools is not None:
            tools: ToolRequest = request.tools
            request_kwargs["tools"] = tools.tools
            if tools.tool_choice is not None:
                request_kwargs["tool_choice"] = tools.tool_choice

        # vLLM-specific extensions live under `extra_body`
        extra_body: Dict[str, Any] = {}

        if request.extensions is not None:
            ext: BackendExtensions = request.extensions
            if isinstance(ext, VLLMExtensions):
                if ext.max_think is not None:
                    if not isinstance(ext.max_think, int) or ext.max_think <= 0:
                        raise ValueError("VLLMExtensions.max_think must be a positive integer")
                if ext.repetition_penalty <= 0:
                    raise ValueError("VLLMExtensions.repetition_penalty must be > 0")
                extra_body["repetition_penalty"] = float(ext.repetition_penalty)

                if ext.max_think is not None:
                    vllm_xargs = extra_body.get("vllm_xargs", {})
                    if not isinstance(vllm_xargs, dict):
                        vllm_xargs = {}
                    vllm_xargs["max_think"] = int(ext.max_think)
                    extra_body["vllm_xargs"] = vllm_xargs
                if ext.extra_body_overrides:
                    extra_body.update(dict(ext.extra_body_overrides))
            else:
                raise TypeError(
                    f"{self.__class__.__name__} received unsupported request.extensions kind="
                    f"{getattr(ext, 'kind', None)!r} (type={type(ext).__name__}); expected VLLMExtensions."
                )

        # Token IDs
        if request.return_.return_token_ids:
            extra_body["return_token_ids"] = True

        # Logprobs
        if request.return_.return_chosen_logprobs:
            request_kwargs["logprobs"] = int(max(1, request.return_.top_logprobs_k))

        if extra_body:
            request_kwargs["extra_body"] = extra_body

        # Perform inference
        resp = await self._async_client.chat.completions.create(**request_kwargs)

        choice = resp.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason

        # Extract token IDs if present
        prompt_token_ids = getattr(resp, "prompt_token_ids", None)
        completion_token_ids = getattr(choice, "token_ids", None)

        # Extract per-token logprobs
        completion_logprobs = None
        logprobs_obj = getattr(choice, "logprobs", None)
        if logprobs_obj is not None:
            token_logprobs = getattr(logprobs_obj, "token_logprobs", None)
            if token_logprobs is None and isinstance(logprobs_obj, dict):
                token_logprobs = (
                    logprobs_obj.get("token_logprobs")
                    or logprobs_obj.get("logprobs")
                )
            if token_logprobs is None and hasattr(logprobs_obj, "content"):
                parts = []
                for part in getattr(logprobs_obj, "content", []):
                    lp = getattr(part, "logprob", None)
                    if lp is None and isinstance(part, dict):
                        lp = part.get("logprob")
                    if lp is not None:
                        parts.append(lp)
                if parts:
                    token_logprobs = parts
            if token_logprobs is not None:
                completion_logprobs = list(token_logprobs)

        chat_resp = ChatResponse(
            text=text,
            finish_reason=finish_reason,
            completion_token_ids=completion_token_ids,
            prompt_token_ids=prompt_token_ids,
            completion_logprobs=completion_logprobs,
        )

        info: Dict[str, Any] = {
            "raw_response": resp.model_dump(exclude_none=True),
            "used_args": request_kwargs,
        }

        return chat_resp, info

    # ─────────────────────────────────────────────────────────────────
    # Policy Weight Sync
    # ─────────────────────────────────────────────────────────────────

    async def get_policy_version(self) -> int:
        """Polls the /runtime_version endpoint of the policy server."""
        url = f"{self.server_url}/runtime_version"
        try:
            session = await self._get_async_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return int(data.get("version", 0))
        except Exception:
            pass
        return 0

    def sync_weights(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        version: Optional[Union[str, int]] = None,
    ) -> str:
        """Push updated policy model parameters into the running vLLM server.

        Uses shared infrastructure for efficient batched NCCL sync.

        Returns:
            version string (either supplied or autogenerated)
        """
        if self._policy_comm is None:
            if not self.enable_weight_updates:
                raise RuntimeError(
                    "sync_weights() called on inference-only client "
                    "(enable_weight_updates=False)."
                )
            raise RuntimeError("Communicator not initialized.")

        return sync_weights_batch(
            communicator=self._policy_comm,
            params=params,
            endpoint="/update_param_batch",
            timeout_s=timeout_s,
            version=version,
            get_background_tasks=lambda: get_server_background_tasks(
                self._session, self.server_url
            ),
        )

    # ─────────────────────────────────────────────────────────────────
    # Control-plane helpers
    # ─────────────────────────────────────────────────────────────────

    def reset_prefix_cache(self) -> None:
        """Reset KV/prefix caches on the policy server."""
        r = self._session.post(f"{self.server_url}/reset_prefix_cache", timeout=30.0)
        r.raise_for_status()

    def get_num_background_tasks(self) -> int:
        """Get number of pending background tasks on the policy server."""
        return get_server_background_tasks(self._session, self.server_url)

    # ─────────────────────────────────────────────────────────────────
    # Scoring API (Reward Model)
    # ─────────────────────────────────────────────────────────────────

    async def score(
        self,
        request: ScoringRequest,
    ) -> Tuple[ScoringResponse, Dict[str, Any]]:
        """Score inputs via /score endpoint on the scoring server."""
        payload = {
            "model": request.model,
            "inputs": list(request.inputs),
            "pooling_type": request.pooling_type.value,
            "normalize": request.normalize,
            "n_labels": request.n_labels,
        }

        url = f"{self._scoring_url}/score"

        session = await self._get_async_session()
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"Score request failed: {resp.status} {error_text}"
                )
            data = await resp.json()

        response = ScoringResponse(
            scores=data["scores"],
            model=data.get("model", request.model),
            usage=data.get("usage", {}),
            metadata=data.get("metadata", {}),
        )

        info: Dict[str, Any] = {
            "raw_response": data,
            "used_args": payload,
        }

        return response, info

    async def score_batch(
        self,
        inputs: List[str],
        *,
        normalize: bool = True,
        pooling_type: PoolingType = PoolingType.LAST,
        n_labels: int = 1,
    ) -> List[float]:
        """Convenience method for batch scoring."""
        request = ScoringRequest.from_list(
            model=self.model_name,
            inputs=inputs,
            pooling_type=pooling_type,
            normalize=normalize,
            n_labels=n_labels,
        )
        response, _ = await self.score(request)
        return response.scores

    def score_sync(
        self,
        request: ScoringRequest,
    ) -> Tuple[ScoringResponse, Dict[str, Any]]:
        """Synchronous version of score() for non-async contexts."""
        payload = {
            "model": request.model,
            "inputs": list(request.inputs),
            "pooling_type": request.pooling_type.value,
            "normalize": request.normalize,
            "n_labels": request.n_labels,
        }

        url = f"{self._scoring_url}/score"

        try:
            resp = self._session.post(url, json=payload, timeout=60.0)
        except Timeout:
            raise TimeoutError("HTTP timeout during score request")
        except Exception as exc:
            raise RuntimeError(f"Error during score request: {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(f"Score request failed: {resp.status_code} {resp.text}")

        data = resp.json()

        response = ScoringResponse(
            scores=data["scores"],
            model=data.get("model", request.model),
            usage=data.get("usage", {}),
            metadata=data.get("metadata", {}),
        )

        info: Dict[str, Any] = {
            "raw_response": data,
            "used_args": payload,
        }

        return response, info

    async def get_reward_model_version(self) -> int:
        """Polls the /runtime_version endpoint on the scoring server."""
        url = f"{self._scoring_url}/runtime_version"
        try:
            session = await self._get_async_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return int(data.get("version", 0))
        except Exception:
            pass
        return 0

    def get_server_info(self, server: str = "scoring") -> Dict[str, Any]:
        """Get server information."""
        url = self._scoring_url if server == "scoring" else self.server_url
        r = self._session.get(f"{url}/health", timeout=5.0)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"status": "healthy"}

    # ─────────────────────────────────────────────────────────────────
    # Reward Model Weight Sync
    # ─────────────────────────────────────────────────────────────────

    def sync_reward_weights(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        training_mode: RewardModelTrainingMode = RewardModelTrainingMode.FULL,
        timeout_s: float = 600.0,
        version: Optional[Union[str, int]] = None,
    ) -> str:
        """Push updated reward model parameters into the scoring vLLM server.

        Uses shared infrastructure for efficient batched NCCL sync.

        Args:
            params: Mapping of parameter names to tensors.
            training_mode: Which weights are being synced (HEAD_ONLY, LORA, FULL).
            timeout_s: Maximum time in seconds for the operation.
            version: Optional version identifier for the weights.

        Returns:
            Version string (supplied or auto-generated).
        """
        if not self.enable_weight_updates:
            raise RuntimeError(
                "sync_reward_weights() called on inference-only client "
                "(enable_weight_updates=False)."
            )

        # Determine which communicator and server to use
        if self._dual_server_mode:
            # Initialize scoring communicator lazily
            self._init_scoring_communicator()
            comm = self._scoring_comm
            server_url = self._scoring_url
        else:
            # Single-server mode: reuse policy communicator
            comm = self._policy_comm
            server_url = self._scoring_url

        if comm is None:
            raise RuntimeError("Communicator not initialized.")

        return sync_weights_batch(
            communicator=comm,
            params=params,
            endpoint="/update_param_batch",
            timeout_s=timeout_s,
            version=version,
            extra_payload={"training_mode": training_mode.value},
            get_background_tasks=lambda: get_server_background_tasks(
                self._session, server_url
            ),
        )


# Backward compatibility alias
VLLMChatClient = VLLMClient

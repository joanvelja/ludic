# SPEC: vLLM-Hosted Reward Model Support

> **Version**: 1.0
> **Status**: RFC (Request for Comments)
> **Target Ludic Version**: Compatible with existing architecture
> **vLLM Requirement**: >= 0.12.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation & Use Cases](#2-motivation--use-cases)
3. [Background: Current Architecture](#3-background-current-architecture)
4. [Background: vLLM Capabilities](#4-background-vllm-capabilities)
5. [Proposed Design](#5-proposed-design)
6. [Component Specifications](#6-component-specifications)
7. [Weight Synchronization for Reward Models](#7-weight-synchronization-for-reward-models)
8. [API Reference](#8-api-reference)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Migration & Compatibility](#10-migration--compatibility)
11. [Testing Strategy](#11-testing-strategy)
12. [Open Questions & Future Work](#12-open-questions--future-work)
13. [References](#13-references)

---

## 1. Executive Summary

This specification proposes extending Ludic to support **vLLM-hosted reward models** (LLM backbone + linear classification head, a.k.a. RLHF-style reward models). The design enables:

1. **Fast Reward Inference**: Serve reward models via vLLM for high-throughput batch scoring
2. **Weight Synchronization**: After training the reward model, sync updated weights to the vLLM inference server using the existing two-plane architecture
3. **Flexible Training Modes**: Support head-only, full fine-tuning, and LoRA-based training
4. **Loose Coupling**: Reward model support integrates cleanly with existing abstractions

### Key Deliverables

| Component | Description |
|-----------|-------------|
| `RewardModelClient` | New client protocol extending `ChatClient` with `score()` method |
| `VLLMRewardClient` | vLLM implementation with NCCL weight sync for reward models |
| `RewardModelServer` | Extended vLLM server with `/score` endpoint and `RewardWeightSyncExtension` |
| `RewardModelPublisher` | Weight publisher handling various training modes (head, full, LoRA) |
| `RewardCreditAssigner` | Credit assigner using learned rewards instead of/alongside env rewards |

---

## 2. Motivation & Use Cases

### 2.1 Why Reward Models?

Ludic currently supports environments that provide **scalar rewards** via `StepOutcome.reward`. This works well for:
- Verifiable tasks (math, code execution, QA with ground truth)
- Rule-based rewards (format compliance, parsing success)

However, many RL applications require **learned reward functions**:
- **RLHF**: Train a reward model on human preferences, then use it to score completions (full sequence reward)
- **Constitutional AI**: Use a trained critic to evaluate responses (full sequence reward)
- **Reward Hacking Mitigation**: Combine environment rewards with learned "sanity check" scores (full sequence/token-level reward)
- **Process Supervision**: Score intermediate reasoning steps (PRMs) (token-level reward)

### 2.2 When to Fine-Tune What

The choice between head-only, LoRA, or full fine-tuning depends on **representation salience** and **feature complexity**:

| Scenario | Training Mode | Rationale |
|----------|---------------|-----------|
| **Salient, linear features** | Head-only | Features already linearly separable in backbone representations |
| **Latent features** | LoRA + Head | Small geometry corrections to amplify drowned features |
| **Complex, non-linear features** | Full Fine-tuning | Need to learn new computational paths for non-linear representations |
| **Distribution shift** | Full Fine-tuning | Backbone representations may not transfer well to new domain |

**Key insight**: For well-trained LLMs, many features are linearly represented. However, when features are not salient or require non-linear combinations, full fine-tuning enables the model to reshape its representation geometry.

### 2.3 Use Cases

| Use Case | Description | Training Mode |
|----------|-------------|---------------|
| **RLHF Cold Start** | Train RM on preferences → Use RM to score another policy's completions → Update policy | Full or LoRA |
| **Hybrid Rewards** | Combine verifiable env reward with learned quality score | Head or LoRA |
| **Reward Ensemble** | Multiple RMs with uncertainty estimation | Mixed modes |
| **Process Reward** | Score intermediate reasoning steps | Full (complex features) |
| **Style Transfer** | Train on style preferences | Head (salient features) |

### 2.4 Requirements

1. **Low Latency**: Batch scoring should not become the training bottleneck
2. **Weight Sync**: After RM training iterations, push new weights to inference
3. **Training Mode Flexibility**: Support head-only, LoRA, and full fine-tuning
4. **Separation of Concerns**: RM training is optional and composable
5. **Dual Model Support**: Policy model and RM model can run on same or different GPUs/servers

---

## 3. Background: Current Architecture

### 3.1 Ludic's Design Principles

Ludic treats the **LLM as a policy** in classical RL:
- **Environment** provides observations and rewards
- **Agent** wraps LLM with context/memory/parsing
- **Algorithms** = Credit Assignment + Loss
- **Training** decouples inference from gradient computation

```
Intent → RolloutRequest → RolloutEngine → Rollout[]
      → CreditAssigner → SAWItem[] → BatchSource → Trainer → PolicyPublisher
```

### 3.2 Current Inference Architecture

```python
# ChatClient protocol (src/ludic/inference/client.py)
class ChatClient(Protocol):
    async def complete(request: ChatCompletionRequest) -> (ChatResponse, Dict)
    def sync_weights(params: Mapping[str, Tensor], *, version: int) -> str
```

**VLLMChatClient** implements this with:
- OpenAI-compatible API for inference
- NCCL-based weight synchronization via two-plane architecture

### 3.3 Two-Plane Weight Distribution

```
┌─────────────────────────────────────────────────────────────┐
│                     CONTROL PLANE (HTTP)                    │
│  POST /init_communicator → JOIN NCCL GROUP                  │
│  POST /update_param_batch → METADATA (names, dtypes, shapes)│
│  GET  /runtime_version → POLICY VERSION                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DATA PLANE (NCCL)                       │
│  broadcast(tensor, src=trainer_rank) → GPU-to-GPU transfer  │
│  barrier() → SYNCHRONIZATION                                │
└─────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `src/ludic/inference/vllm_client.py`: Client with NCCL communicator
- `src/ludic/inference/vllm_server.py`: Custom vLLM server with `WeightSyncWorkerExtension`
- `src/ludic/distributed/publisher.py`: `BroadcastPolicyPublisher`

---

## 4. Background: vLLM Capabilities

### 4.1 Pooling/Scoring Task Types

vLLM v0.12+ supports multiple task types for non-generative models:

| Task | Offline API | Server Endpoint | Use Case |
|------|-------------|-----------------|----------|
| `embed` | `LLM.embed()` | `/v1/embeddings` | Embedding vectors |
| `classify` | `LLM.classify()` | `/classify` | Sequence classification |
| `reward` | `LLM.reward()` | `/pooling` | Reward model scoring |
| `score` | `LLM.score()` | `/v1/score` | Cross-encoder similarity |

### 4.2 Pooling Configuration

```python
class PoolingType(IntEnum):
    LAST = 0   # Last token (default for LLMs)
    ALL = 1    # All tokens
    CLS = 2    # CLS token (BERT-style)
    STEP = 3   # Step-based (for PRMs)
    MEAN = 4   # Mean pooling
```

**Configuration:**
```bash
vllm serve model --task reward \
  --pooler-config '{"pooling_type": "LAST", "normalize": true}'
```

### 4.3 Model Architecture Handling

vLLM can automatically convert models for classification:
- `as_seq_cls_model()`: Adds pooler + linear head + softmax
- `as_reward_model()`: Returns hidden states directly

**Supported Patterns:**
- `*ForSequenceClassification` (Qwen2, InternLM, etc.)
- `*RewardModel` (custom architectures)
- Any LLM via `--convert classify` flag

### 4.4 Weight Update Mechanism

vLLM's `model.load_weights()` accepts `[(name, tensor)]` tuples:
- Automatically handles QKV fusion
- Handles tensor parallel sharding
- Supports incremental updates

**Critical for Reward Models:**
- Must include appropriate weights based on training mode
- Full sync: backbone + head
- LoRA sync: adapter weights + head
- Head-only sync: just head weights

---

## 5. Proposed Design

### 5.1 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING LOOP                                   │
│                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ Policy Model │    │ Reward Model │    │   Trainer    │                  │
│  │  (training)  │    │  (training)  │    │              │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘                  │
│         │                   │                                              │
│         │                   │  Training Modes:                             │
│         │                   │  ─────────────────                           │
│         │                   │  • HEAD_ONLY: Linear head only               │
│         │                   │  • LORA: LoRA adapters + head                │
│         │                   │  • FULL: Complete backbone + head            │
│         │                   │                                              │
│         ▼                   ▼                                              │
│  ┌─────────────────────────────────────┐                                   │
│  │         PolicyPublisher             │◄──── Sync policy weights          │
│  └─────────────────────────────────────┘                                   │
│  ┌─────────────────────────────────────┐                                   │
│  │       RewardModelPublisher          │◄──── Sync RM weights (mode-aware) │
│  └─────────────────────────────────────┘                                   │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE SERVERS                                  │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │   vLLM Policy Server     │    │   vLLM Reward Server     │               │
│  │                          │    │                          │               │
│  │  /v1/chat/completions    │    │  /score                  │               │
│  │  /update_param_batch     │    │  /update_param_batch     │               │
│  │                          │    │                          │               │
│  │  VLLMChatClient ─────────│    │  VLLMRewardClient ───────│               │
│  └──────────────────────────┘    └──────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Training Mode Support

```python
class RewardModelTrainingMode(Enum):
    """Training modes for reward models."""

    HEAD_ONLY = "head_only"
    """
    Train only the classification head (linear layer).

    Use when: Features are salient and linearly separable in backbone.
    Weights synced: score.weight, score.bias (if present)
    """

    LORA = "lora"
    """
    Train LoRA adapters + classification head.

    Use when: Features are latent but can be amplified with small
    geometry corrections. LoRA provides parameter-efficient tuning.
    Weights synced: lora_A/lora_B matrices + score.weight
    """

    FULL = "full"
    """
    Full fine-tuning of backbone + classification head.

    Use when: Features require non-linear representations or new
    computational paths not present in pre-trained backbone.
    Weights synced: All model parameters
    """
```

### 5.3 Design Principles

1. **Parallel to Existing Pattern**: `RewardModelClient` mirrors `ChatClient`
2. **Two-Plane Architecture**: Same control/data plane separation for weight sync
3. **Training Mode Awareness**: Publisher adapts to training mode
4. **Optional Integration**: Reward model is opt-in; default behavior unchanged
5. **Composable Credit Assignment**: RM scores compose with env rewards
6. **Flexibility**: Reward model can be efficiently used in any loop (either as a reward or as a policy)

### 5.4 Component Overview

| Component | Location | Description |
|-----------|----------|-------------|
| `RewardModelClient` | `src/ludic/inference/reward_client.py` | Protocol for reward model inference |
| `VLLMRewardClient` | `src/ludic/inference/vllm_reward_client.py` | vLLM implementation |
| `RewardModelServer` | `src/ludic/inference/vllm_reward_server.py` | vLLM server with scoring endpoints |
| `RewardWeightSyncExtension` | `src/ludic/inference/vllm_reward_server.py` | Worker extension for RM weight sync |
| `RewardModelPublisher` | `src/ludic/distributed/reward_publisher.py` | Training-mode-aware weight sync |
| `RewardCreditAssigner` | `src/ludic/training/reward_credit.py` | Credit assignment using RM scores |
| `ScoringRequest` | `src/ludic/inference/request.py` | Request type for scoring |

---

## 6. Component Specifications

### 6.1 RewardModelClient Protocol

```python
# src/ludic/inference/reward_client.py

from typing import Protocol, List, Tuple, Dict, Any, Optional, Mapping
from enum import Enum
import torch

class RewardModelTrainingMode(Enum):
    HEAD_ONLY = "head_only"
    LORA = "lora"
    FULL = "full"


@dataclass(frozen=True)
class ScoringRequest:
    """Request for reward model scoring."""
    model: str
    inputs: List[str]  # Texts to score (prompt + completion concatenated)
    return_hidden_states: bool = False
    pooling_type: str = "last"  # "last", "mean", "cls", "step"
    normalize: bool = True


@dataclass(frozen=True)
class ScoringResponse:
    """Response from reward model."""
    scores: List[float]  # Per-input scalar rewards
    hidden_states: Optional[List[torch.Tensor]] = None  # If requested
    metadata: Dict[str, Any] = field(default_factory=dict)


class RewardModelClient(Protocol):
    """Protocol for reward model inference backends."""

    async def score(
        self,
        request: ScoringRequest,
    ) -> Tuple[ScoringResponse, Dict[str, Any]]:
        """
        Score a batch of inputs.

        Returns:
            (ScoringResponse, info): Scores and transport metadata
        """
        ...

    async def score_batch(
        self,
        inputs: List[str],
        *,
        normalize: bool = True,
    ) -> List[float]:
        """Convenience method for batch scoring."""
        ...

    def sync_weights(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        version: Optional[int] = None,
        training_mode: RewardModelTrainingMode = RewardModelTrainingMode.FULL,
    ) -> str:
        """
        Push updated reward model parameters.

        The params dict content depends on training_mode:
        - HEAD_ONLY: Only head weights (score.weight, etc.)
        - LORA: LoRA adapter weights + head weights
        - FULL: Complete model state dict

        Args:
            params: Model parameters to sync
            timeout_s: Timeout for sync operation
            version: Optional version number for staleness tracking
            training_mode: Determines which weights are expected/synced
        """
        ...

    async def get_model_version(self) -> int:
        """Get current RM version for staleness checking."""
        ...
```

### 6.2 VLLMRewardClient Implementation

```python
# src/ludic/inference/vllm_reward_client.py

class VLLMRewardClient(RewardModelClient):
    """
    vLLM-backed reward model client.

    Modes:
        * inference-only: HTTP requests to /score endpoint
        * training/update: + NCCL weight synchronization

    Supports three training modes:
        * HEAD_ONLY: Sync only classification head
        * LORA: Sync LoRA adapters (merged before broadcast) + head
        * FULL: Sync complete backbone + head

    Args:
        host: vLLM server hostname
        port: HTTP port for scoring API
        group_port: TCP port for NCCL process group (weight updates)
        enable_weight_updates: If True, initialize NCCL communicator
        device: Device for NCCL operations
        model_name: Model identifier for requests
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8001,  # Different from policy server
        group_port: int = 51217,  # Different from policy group
        connection_timeout_s: float = 60,
        enable_weight_updates: bool = False,
        device: Union[str, torch.device, int] = 0,
        model_name: str = "reward-model",
    ) -> None:
        self.host = host
        self.port = port
        self.group_port = group_port
        self.enable_weight_updates = enable_weight_updates
        self.device = device
        self._model_name = model_name

        # HTTP client for scoring API
        self._session: Optional[aiohttp.ClientSession] = None
        self.server_url = f"http://{self.host}:{self.port}"

        # Sync HTTP client for control plane
        self._sync_session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3)
        self._sync_session.mount("http://", adapter)

        # NCCL communicator (lazy init)
        self._pynccl_comm: Optional[PyNcclCommunicator] = None
        self._rank: Optional[int] = None

        # Health check
        self._check_server(connection_timeout_s)

        if self.enable_weight_updates:
            self._init_communicator()
            atexit.register(self.close_communicator)

    async def score(
        self,
        request: ScoringRequest,
    ) -> Tuple[ScoringResponse, Dict[str, Any]]:
        """Score inputs via vLLM /score endpoint."""

        if self._session is None:
            self._session = aiohttp.ClientSession()

        payload = {
            "model": request.model,
            "input": request.inputs,
            "pooling_type": request.pooling_type,
            "normalize": request.normalize,
        }

        async with self._session.post(
            f"{self.server_url}/score",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        scores = [item["score"] for item in data["data"]]

        response = ScoringResponse(
            scores=scores,
            hidden_states=None,
            metadata={"model": data.get("model"), "usage": data.get("usage")},
        )

        info = {"raw_response": data, "request": payload}
        return response, info

    async def score_batch(
        self,
        inputs: List[str],
        *,
        normalize: bool = True,
    ) -> List[float]:
        """Convenience method for batch scoring."""
        request = ScoringRequest(
            model=self._model_name,
            inputs=inputs,
            normalize=normalize,
        )
        response, _ = await self.score(request)
        return response.scores

    def sync_weights(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        version: Optional[int] = None,
        training_mode: RewardModelTrainingMode = RewardModelTrainingMode.FULL,
    ) -> str:
        """
        Push reward model weights via NCCL.

        The handling differs by training mode:
        - HEAD_ONLY: Expects only head weights (score.*, classifier.*)
        - LORA: Expects merged LoRA weights + head (caller must merge)
        - FULL: Expects complete state dict

        For LoRA training, callers should merge adapters before calling:
            merged_model = model.merge_and_unload()
            params = merged_model.state_dict()
        """
        if self._pynccl_comm is None:
            raise RuntimeError("Weight updates not enabled")

        start = time.time()

        # Transform based on training mode
        vllm_params = self._transform_params(params, training_mode)

        # Validate we have expected weights
        self._validate_params(vllm_params, training_mode)

        # Same two-plane pattern as VLLMChatClient
        sorted_keys = sorted(vllm_params.keys())
        metadata = [
            {
                "name": name,
                "dtype": str(vllm_params[name].dtype),
                "shape": tuple(vllm_params[name].shape),
            }
            for name in sorted_keys
        ]

        # Control plane: announce batch
        payload = {
            "metadata": metadata,
            "training_mode": training_mode.value,
        }
        if version is not None:
            payload["version"] = version

        resp = self._sync_session.post(
            f"{self.server_url}/update_param_batch",
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()

        time.sleep(0.5)  # Buffer for server RPC

        # Data plane: broadcast tensors
        for name in sorted_keys:
            tensor = vllm_params[name]
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self._pynccl_comm.broadcast(tensor, src=self._rank)

        self._pynccl_comm.group.barrier()

        if (time.time() - start) > timeout_s:
            raise TimeoutError(f"sync_weights exceeded {timeout_s}s")

        # Wait for background tasks
        while self._get_num_background_tasks() > 0:
            time.sleep(0.2)

        return str(version) if version else f"rm-{int(time.time())}"

    def _transform_params(
        self,
        params: Mapping[str, torch.Tensor],
        training_mode: RewardModelTrainingMode,
    ) -> Dict[str, torch.Tensor]:
        """Transform params to vLLM format based on training mode."""
        new_params = {}

        for k, v in params.items():
            # Skip LoRA-specific keys (should be merged before calling)
            if "lora_" in k and training_mode != RewardModelTrainingMode.LORA:
                continue

            # Standard name normalization
            clean_k = k
            while clean_k.startswith("base_model.model."):
                clean_k = clean_k[len("base_model.model."):]
            clean_k = clean_k.replace(".base_layer", "")

            if clean_k.startswith("model.model."):
                clean_k = clean_k[len("model."):]

            if clean_k.startswith("layers."):
                clean_k = f"model.{clean_k}"

            new_params[clean_k] = v

        return new_params

    def _validate_params(
        self,
        params: Dict[str, torch.Tensor],
        training_mode: RewardModelTrainingMode,
    ) -> None:
        """Validate params contain expected weights for training mode."""
        head_patterns = ["score.", "classifier."]
        has_head = any(
            any(p in k for p in head_patterns)
            for k in params.keys()
        )

        if training_mode == RewardModelTrainingMode.HEAD_ONLY:
            if not has_head:
                raise ValueError(
                    "HEAD_ONLY mode requires head weights (score.* or classifier.*)"
                )
            # Should only have head weights
            backbone_keys = [
                k for k in params.keys()
                if not any(p in k for p in head_patterns)
            ]
            if backbone_keys:
                import warnings
                warnings.warn(
                    f"HEAD_ONLY mode but found {len(backbone_keys)} backbone weights. "
                    "These will be synced but this may not be intended."
                )

        elif training_mode == RewardModelTrainingMode.FULL:
            if not has_head:
                import warnings
                warnings.warn(
                    "FULL mode but no head weights found. "
                    "Reward model head may not be updated."
                )

    # ... _init_communicator, _check_server, etc. similar to VLLMChatClient
```

### 6.3 Reward Model Server

```python
# src/ludic/inference/vllm_reward_server.py

"""
vLLM server specialized for reward models.

Supports all training modes:
    - HEAD_ONLY: Only head weights updated
    - LORA: LoRA + head weights (expects merged weights)
    - FULL: Complete model weights

Usage:
    python -m ludic.inference.vllm_reward_server \
        --model internlm/internlm2-1_8b-reward \
        --task reward \
        --port 8001
"""

import os
os.environ["VLLM_USE_V1"] = "1"

from fastapi import FastAPI, Request
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import build_app

# Global state
RUNTIME_VERSION: int = 0
RUNTIME_VERSION_LOCK = asyncio.Lock()
background_tasks: Set[asyncio.Task] = set()


class RewardWeightSyncExtension:
    """
    Worker extension for reward model weight synchronization.

    Handles all training modes:
    - HEAD_ONLY: Only updates classification head
    - LORA: Updates LoRA matrices + head (expects pre-merged)
    - FULL: Updates complete model

    vLLM's load_weights() handles the mapping automatically.
    """

    pynccl_comm: PyNcclCommunicator | None = None
    client_rank: int | None = None
    device: torch.device | None = None

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """Join NCCL process group."""
        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1

        # Debug: print model parameter names on rank 0
        if self.pynccl_comm.rank == 0:
            print("\n" + "=" * 60)
            print("[DEBUG] vLLM Reward Model Parameter Names")
            print("=" * 60)
            try:
                for name, _ in self.model_runner.model.named_parameters():
                    print(f"  • {name}")
            except Exception as e:
                print(f"Could not enumerate parameters: {e}")
            print("=" * 60 + "\n")

    def update_param_batch(
        self,
        metadata_list: Sequence[Tuple[str, str, Sequence[int]]],
    ) -> None:
        """
        Receive and load weight batch.

        Works for all training modes - the caller determines which
        weights to send, and vLLM's load_weights() handles them.
        """
        if self.pynccl_comm is None or self.client_rank is None:
            raise RuntimeError("Communicator not initialized")

        for name, dtype_str, shape in metadata_list:
            torch_dtype = getattr(torch, dtype_str.split(".")[-1])
            weight = torch.empty(
                tuple(shape), dtype=torch_dtype, device=self.device
            )

            # Receive via NCCL
            self.pynccl_comm.broadcast(weight, src=self.client_rank)

            # Load into model
            # vLLM handles QKV fusion, TP sharding, etc. automatically
            self.model_runner.model.load_weights(weights=[(name, weight)])

        self.pynccl_comm.group.barrier()

    def close_communicator(self) -> None:
        if self.pynccl_comm:
            del self.pynccl_comm
            self.pynccl_comm = None


async def run_reward_server(args) -> None:
    """Start vLLM reward model server."""

    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Configure for reward model
    engine_args.task = "reward"
    engine_args.worker_extension_cls = (
        "ludic.inference.vllm_reward_server.RewardWeightSyncExtension"
    )

    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )
    engine = AsyncLLMEngine.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    app: FastAPI = build_app(args)

    # ─────────────────────────────────────────────────────────────
    # Scoring endpoint
    # ─────────────────────────────────────────────────────────────

    @app.post("/score")
    async def score(request: Request) -> dict:
        """
        Score a batch of inputs.

        Request:
            {
                "model": "...",
                "input": ["text1", "text2", ...],
                "pooling_type": "last",  # optional
                "normalize": true  # optional
            }

        Response:
            {
                "model": "...",
                "data": [{"score": 0.85, "index": 0}, ...],
                "usage": {"prompt_tokens": 123}
            }
        """
        data = await request.json()
        inputs = data.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]

        from vllm import PoolingParams
        pooling_params = PoolingParams(normalize=data.get("normalize", True))

        # Use vLLM's encode API for reward scoring
        outputs = await engine.encode(inputs, pooling_params)

        results = []
        for i, output in enumerate(outputs):
            score = float(output.outputs.data)
            results.append({"score": score, "index": i})

        return {
            "model": data.get("model", args.model),
            "data": results,
            "usage": {"prompt_tokens": sum(len(inp.split()) for inp in inputs)},
        }

    # ─────────────────────────────────────────────────────────────
    # Weight sync endpoints
    # ─────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "type": "reward_model"}

    @app.get("/get_world_size")
    async def get_world_size():
        return {"world_size": args.tensor_parallel_size * args.data_parallel_size}

    @app.get("/runtime_version")
    async def runtime_version():
        return {"version": RUNTIME_VERSION}

    @app.post("/init_communicator")
    async def init_communicator(request: Request):
        data = await request.json()
        create_background_task(
            engine.collective_rpc(
                "init_communicator",
                args=(data["host"], data["port"], data["world_size"])
            )
        )
        return {"status": "ok"}

    @app.post("/update_param_batch")
    async def update_param_batch(request: Request):
        """
        Receive weight batch metadata.

        Body includes:
        - metadata: List of {name, dtype, shape}
        - training_mode: "head_only", "lora", or "full" (informational)
        - version: Optional version number
        """
        data = await request.json()
        metadata = data.get("metadata", [])
        forced_version = data.get("version")
        training_mode = data.get("training_mode", "full")

        print(f"\n[SERVER] Receiving {len(metadata)} weights (mode: {training_mode})")

        rpc_args = [(m["name"], m["dtype"], m["shape"]) for m in metadata]

        async def do_update():
            async with weight_update_semaphore:
                await engine.collective_rpc("update_param_batch", args=(rpc_args,))
                await engine.reset_prefix_cache()

                global RUNTIME_VERSION
                async with RUNTIME_VERSION_LOCK:
                    RUNTIME_VERSION = (
                        int(forced_version) if forced_version else RUNTIME_VERSION + 1
                    )

        create_background_task(do_update())
        return {"status": "ok"}

    @app.post("/close_communicator")
    async def close_communicator():
        await engine.collective_rpc("close_communicator")
        return {"status": "ok"}

    @app.post("/get_num_background_tasks")
    async def get_num_background_tasks():
        return {"num_background_tasks": len(background_tasks)}

    # Start server
    await serve_http(app, ...)
```

### 6.4 Reward Model Publisher

```python
# src/ludic/distributed/reward_publisher.py

from ludic.distributed.interfaces import PolicyPublisher, ControlPlane, TensorCommunicator

class RewardModelPublisher(PolicyPublisher):
    """
    Publisher for reward model weights.

    Supports three training modes:

    HEAD_ONLY:
        Only syncs classification head (score.weight, classifier.weight).
        Use when features are salient and linearly separable.

    LORA:
        Syncs LoRA adapters + head. Caller must merge adapters before
        passing state dict (model.merge_and_unload()).
        Use when features are latent but can be amplified.

    FULL:
        Syncs complete backbone + head.
        Use when features require non-linear representations or
        new computational paths.

    Head Weight Naming:
        - "score.weight" (most reward models)
        - "classifier.weight" (sequence classification models)
    """

    def __init__(
        self,
        control: ControlPlane,
        comm: TensorCommunicator,
        src_rank: int,
        *,
        head_name_patterns: List[str] = None,
    ):
        self.control = control
        self.comm = comm
        self.src_rank = src_rank
        self.head_name_patterns = head_name_patterns or ["score.", "classifier."]

    def publish(
        self,
        state_dict: Mapping[str, torch.Tensor],
        version: Optional[int] = None,
        *,
        training_mode: RewardModelTrainingMode = RewardModelTrainingMode.FULL,
    ) -> None:
        """
        Publish reward model weights.

        Args:
            state_dict: Model parameters to sync
            version: Optional version number
            training_mode: Determines expected content and validation

        For LORA mode, the caller should merge adapters first:
            merged_model = model.merge_and_unload()
            state_dict = merged_model.state_dict()
        """
        # Transform HF/PEFT names to vLLM format
        vllm_params = self._transform_to_vllm(state_dict, training_mode)

        # Validate based on training mode
        self._validate_params(vllm_params, training_mode)

        # Deterministic order for broadcast consistency
        sorted_keys = sorted(vllm_params.keys())
        metadata = [
            {
                "name": name,
                "dtype": str(vllm_params[name].dtype),
                "shape": tuple(vllm_params[name].shape),
            }
            for name in sorted_keys
        ]

        # Control plane: announce batch with mode information
        self.control.announce_update_batch(
            metadata,
            version=version,
            extra={"training_mode": training_mode.value},
        )

        # Data plane: broadcast tensors
        for name in sorted_keys:
            tensor = vllm_params[name]
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self.comm.broadcast(tensor, src=self.src_rank)

        self.comm.barrier()
        self.control.finalize_update()

    def _transform_to_vllm(
        self,
        state_dict: Mapping[str, torch.Tensor],
        training_mode: RewardModelTrainingMode,
    ) -> Dict[str, torch.Tensor]:
        """Transform HF/PEFT state dict to vLLM format."""
        new_state_dict = {}

        for k, v in state_dict.items():
            # For HEAD_ONLY mode, only include head weights
            if training_mode == RewardModelTrainingMode.HEAD_ONLY:
                if not any(p in k for p in self.head_name_patterns):
                    continue

            # Skip unmergeable LoRA keys (should be merged before calling)
            if "lora_A" in k or "lora_B" in k:
                import warnings
                warnings.warn(
                    f"Found LoRA weight '{k}'. For LORA mode, please merge "
                    "adapters before calling publish() using model.merge_and_unload()"
                )
                continue

            # Standard name normalization
            clean_k = k
            while clean_k.startswith("base_model.model."):
                clean_k = clean_k[len("base_model.model."):]
            clean_k = clean_k.replace(".base_layer", "")

            if clean_k.startswith("model.model."):
                clean_k = clean_k[len("model."):]

            if clean_k.startswith("layers."):
                clean_k = f"model.{clean_k}"

            new_state_dict[clean_k] = v

        return new_state_dict

    def _validate_params(
        self,
        params: Dict[str, torch.Tensor],
        training_mode: RewardModelTrainingMode,
    ) -> None:
        """Validate params contain expected weights."""
        has_head = any(
            any(p in k for p in self.head_name_patterns)
            for k in params.keys()
        )
        has_backbone = any(
            "layers." in k or "embed" in k
            for k in params.keys()
        )

        if training_mode == RewardModelTrainingMode.HEAD_ONLY:
            if not has_head:
                raise ValueError("HEAD_ONLY mode requires head weights")
            if has_backbone:
                raise ValueError(
                    "HEAD_ONLY mode should not include backbone weights. "
                    "Found backbone keys in state_dict."
                )

        elif training_mode == RewardModelTrainingMode.FULL:
            if not has_backbone:
                raise ValueError("FULL mode requires backbone weights")
            if not has_head:
                import warnings
                warnings.warn("FULL mode but no head weights found")

        elif training_mode == RewardModelTrainingMode.LORA:
            if not has_head:
                import warnings
                warnings.warn("LORA mode but no head weights found")
            # Should have merged LoRA → backbone weights present


def create_reward_publisher(
    client: "VLLMRewardClient",
    *,
    head_name_patterns: List[str] = None,
    rank0_only: bool = False,
) -> "RewardModelPublisher":
    """Factory for reward model publishers."""

    def _make() -> RewardModelPublisher:
        from ludic.distributed.adapters.vllm import (
            VllmControlPlane,
            VllmTensorCommunicator,
        )
        control = VllmControlPlane(client)
        comm = VllmTensorCommunicator(client)
        return RewardModelPublisher(
            control, comm, src_rank=comm.rank,
            head_name_patterns=head_name_patterns,
        )

    if rank0_only:
        from ludic.distributed.publisher import Rank0OnlyPublisher
        return Rank0OnlyPublisher(_make)
    return _make()
```

### 6.5 Reward-Aware Credit Assignment

```python
# src/ludic/training/reward_credit.py

from ludic.training.credit_assignment import CreditAssigner

class RewardModelCreditAssigner(CreditAssigner):
    """
    Credit assignment using learned reward model scores.

    Modes:
        - "replace": Use RM score instead of env reward
        - "add": Add RM score to env reward
        - "multiply": Multiply env reward by RM score (gating)
        - "weighted": α * env_reward + (1-α) * rm_score

    This assigner is async because it needs to batch-score completions
    via the RewardModelClient.
    """

    def __init__(
        self,
        reward_client: RewardModelClient,
        mode: str = "replace",  # "replace", "add", "multiply", "weighted"
        weight: float = 0.5,  # For "weighted" mode
        normalize: bool = True,
        group_normalize: bool = True,  # GRPO-style normalization
        group_key: str = "initial_obs",  # Key for grouping
    ):
        self.reward_client = reward_client
        self.mode = mode
        self.weight = weight
        self.normalize = normalize
        self.group_normalize = group_normalize
        self.group_key = group_key

    async def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        """
        Compute per-step weights using reward model.

        For each step, we score the (state, action) pair and
        combine with environment reward according to mode.
        """
        weights: Dict[RolloutStepKey, float] = {}

        # Batch all scoring requests
        score_inputs: List[Tuple[str, Step, RolloutStepKey]] = []

        for rollout in rollouts:
            for step in rollout.steps:
                key = (rollout.id, step.index)
                # Concatenate state + action for scoring
                text = f"{step.prev_obs}\n{step.action}"
                score_inputs.append((text, step, key))

        # Batch score
        texts = [inp[0] for inp in score_inputs]
        rm_scores = await self.reward_client.score_batch(
            texts, normalize=self.normalize
        )

        # Combine with env rewards
        for (text, step, key), rm_score in zip(score_inputs, rm_scores):
            env_reward = step.reward

            if self.mode == "replace":
                combined = rm_score
            elif self.mode == "add":
                combined = env_reward + rm_score
            elif self.mode == "multiply":
                combined = env_reward * rm_score
            elif self.mode == "weighted":
                combined = self.weight * env_reward + (1 - self.weight) * rm_score
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            weights[key] = combined

        # Group normalization (GRPO-style)
        if self.group_normalize:
            weights = self._group_normalize(weights, rollouts)

        return weights

    def _group_normalize(
        self,
        weights: Dict[RolloutStepKey, float],
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        """Normalize weights within groups (GRPO-style A_i = R_i - mean)."""
        from collections import defaultdict
        import numpy as np

        # Group by initial observation
        groups = defaultdict(list)
        for rollout in rollouts:
            group_key = rollout.meta.get(self.group_key, rollout.steps[0].prev_obs)
            for step in rollout.steps:
                key = (rollout.id, step.index)
                groups[group_key].append((key, weights[key]))

        # Normalize within groups
        normalized = {}
        for group_items in groups.values():
            values = [v for _, v in group_items]
            mean = np.mean(values)
            std = np.std(values) + 1e-8
            for key, value in group_items:
                normalized[key] = (value - mean) / std

        return normalized
```

---

## 7. Weight Synchronization for Reward Models

### 7.1 Training Mode Comparison

| Mode | What's Trained | What's Synced | When to Use |
|------|----------------|---------------|-------------|
| **HEAD_ONLY** | Linear head | `score.weight` | Salient, linearly separable features |
| **LORA** | LoRA adapters + head | Merged backbone + head | Latent features needing geometry correction |
| **FULL** | Entire model | Complete state dict | Non-linear features, distribution shift |

### 7.2 LoRA Merging Before Sync

For LoRA training, adapters must be merged before syncing:

```python
from peft import PeftModel

# During training, model has separate LoRA weights
reward_model: PeftModel = ...

# Before syncing, merge LoRA into backbone
merged_model = reward_model.merge_and_unload()

# Now sync the merged weights
reward_publisher.publish(
    merged_model.state_dict(),
    version=step,
    training_mode=RewardModelTrainingMode.LORA,
)
```

### 7.3 Head Weight Naming Conventions

| Model Family | Head Weight Names |
|--------------|-------------------|
| InternLM Reward | `score.weight` |
| Qwen Classification | `classifier.weight`, `classifier.bias` |
| Llama Classification | `score.weight` |
| Custom HF | `model.score.weight`, `model.classifier.weight` |

We do expect to make use of Custom HF models for reward models (namely, AutoModelForSequenceClassification objects).

### 7.4 Training-Inference Weight Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     REWARD MODEL TRAINING                       │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Training Model (HuggingFace / PyTorch)                  │  │
│   │                                                          │  │
│   │  MODE: HEAD_ONLY                                         │  │
│   │    ├── backbone: frozen                                  │  │
│   │    └── head: trainable (Linear)                          │  │
│   │                                                          │  │
│   │  MODE: LORA                                              │  │
│   │    ├── backbone: frozen                                  │  │
│   │    ├── LoRA adapters: trainable (rank r)                 │  │
│   │    └── head: trainable                                   │  │
│   │                                                          │  │
│   │  MODE: FULL                                              │  │
│   │    ├── backbone: trainable                               │  │
│   │    └── head: trainable                                   │  │
│   │                                                          │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼ Prepare state_dict               │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  HEAD_ONLY: filter to head keys only                     │  │
│   │  LORA: model.merge_and_unload() → full backbone + head   │  │
│   │  FULL: model.state_dict() → complete weights             │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼ RewardModelPublisher.publish()   │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Transform to vLLM format:                               │  │
│   │  - Validate weights match training_mode                  │  │
│   │  - Normalize naming (strip base_model.model., etc.)      │  │
│   │  - Deterministic key ordering                            │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
└──────────────────────────────│──────────────────────────────────┘
                               │
                               ▼ Two-Plane Broadcast

┌─────────────────────────────────────────────────────────────────┐
│                     vLLM REWARD SERVER                          │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  RewardWeightSyncExtension                               │  │
│   │                                                          │  │
│   │  for name, dtype, shape in metadata:                     │  │
│   │      weight = torch.empty(shape, dtype=dtype)            │  │
│   │      pynccl_comm.broadcast(weight, src=client_rank)      │  │
│   │      model_runner.model.load_weights([(name, weight)])   │  │
│   │                                                          │  │
│   │  vLLM's load_weights() handles:                          │  │
│   │  - QKV fusion                                            │  │
│   │  - Tensor parallel sharding                              │  │
│   │  - Head weight loading                                   │  │
│   │                                                          │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. API Reference

### 8.1 Training Modes

```python
class RewardModelTrainingMode(Enum):
    HEAD_ONLY = "head_only"  # Salient, linear features
    LORA = "lora"            # Latent features, geometry correction
    FULL = "full"            # Non-linear features, new computational paths
```

### 8.2 Client APIs

```python
# ─────────────────────────────────────────────────────────────
# RewardModelClient Protocol
# ─────────────────────────────────────────────────────────────

class RewardModelClient(Protocol):
    async def score(request: ScoringRequest) -> Tuple[ScoringResponse, Dict]
    async def score_batch(inputs: List[str], *, normalize: bool) -> List[float]
    def sync_weights(
        params: Mapping[str, Tensor],
        *,
        version: int,
        training_mode: RewardModelTrainingMode,
    ) -> str
    async def get_model_version() -> int

# ─────────────────────────────────────────────────────────────
# VLLMRewardClient
# ─────────────────────────────────────────────────────────────

VLLMRewardClient(
    host: str = "0.0.0.0",
    port: int = 8001,
    group_port: int = 51217,
    connection_timeout_s: float = 60,
    enable_weight_updates: bool = False,
    device: Union[str, torch.device, int] = 0,
    model_name: str = "reward-model",
)
```

### 8.3 Server Endpoints

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/health` | GET | — | `{"status": "ok", "type": "reward_model"}` |
| `/score` | POST | `{"model": str, "input": List[str]}` | `{"data": [{"score": float}]}` |
| `/get_world_size` | GET | — | `{"world_size": int}` |
| `/runtime_version` | GET | — | `{"version": int}` |
| `/init_communicator` | POST | `{host, port, world_size}` | `{"status": "ok"}` |
| `/update_param_batch` | POST | `{metadata, training_mode, version}` | `{"status": "ok"}` |
| `/close_communicator` | POST | — | `{"status": "ok"}` |

### 8.4 Publisher APIs

```python
# ─────────────────────────────────────────────────────────────
# RewardModelPublisher
# ─────────────────────────────────────────────────────────────

RewardModelPublisher(
    control: ControlPlane,
    comm: TensorCommunicator,
    src_rank: int,
    head_name_patterns: List[str] = ["score.", "classifier."],
)

def publish(
    state_dict: Mapping[str, Tensor],
    version: int,
    *,
    training_mode: RewardModelTrainingMode = RewardModelTrainingMode.FULL,
) -> None

# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

create_reward_publisher(
    client: VLLMRewardClient,
    head_name_patterns: List[str] = None,
    rank0_only: bool = False,
) -> RewardModelPublisher
```

---

## 9. Implementation Roadmap

### Phase 1: Core Infrastructure

| Task | Description | Files |
|------|-------------|-------|
| 1.1 | Define `RewardModelClient` protocol with training modes | `src/ludic/inference/reward_client.py` |
| 1.2 | Implement `VLLMRewardClient` (inference only) | `src/ludic/inference/vllm_reward_client.py` |
| 1.3 | Create `vllm_reward_server.py` with `/score` endpoint | `src/ludic/inference/vllm_reward_server.py` |
| 1.4 | Add unit tests for scoring API | `tests/inference/test_reward_client.py` |

### Phase 2: Weight Synchronization

| Task | Description | Files |
|------|-------------|-------|
| 2.1 | Add NCCL support to `VLLMRewardClient` | `src/ludic/inference/vllm_reward_client.py` |
| 2.2 | Implement `RewardWeightSyncExtension` | `src/ludic/inference/vllm_reward_server.py` |
| 2.3 | Create `RewardModelPublisher` with mode support | `src/ludic/distributed/reward_publisher.py` |
| 2.4 | Add integration tests for all training modes | `tests/distributed/test_reward_sync.py` |

### Phase 3: Credit Assignment Integration

| Task | Description | Files |
|------|-------------|-------|
| 3.1 | Implement `RewardModelCreditAssigner` | `src/ludic/training/reward_credit.py` |
| 3.2 | Add reward model support to `RolloutEngine` | `src/ludic/training/batching/rollout_engine.py` |
| 3.3 | Create `make_reward_rl()` algorithm preset | `src/ludic/training/algorithm.py` |
| 3.4 | Add end-to-end training tests | `tests/training/test_reward_training.py` |

### Phase 4: Documentation & Examples

| Task | Description | Files |
|------|-------------|-------|
| 4.1 | Add reward model examples (all 3 modes) | `examples/reward_model/` |
| 4.2 | Update CLAUDE.md with reward model guidance | `CLAUDE.md` |
| 4.3 | Add component docs | `docs/generated/components/reward_model.md` |
| 4.4 | Performance benchmarks | `benchmarks/reward_model/` |

---

## 10. Migration & Compatibility

### 10.1 Backward Compatibility

This feature is **fully backward compatible**:
- All existing APIs remain unchanged
- `ChatClient` protocol is untouched
- Default behavior (no reward model) is preserved
- Reward model is opt-in via new components

### 10.2 vLLM Version Requirements

| Feature | Minimum vLLM Version |
|---------|---------------------|
| Basic scoring | 0.12.0 |
| Task types | 0.12.0 |
| Weight updates | 0.12.0 (with custom server) |
| LoRA support | 0.12.0+ |
| Process reward models | 0.12.0+ |

---

## 11. Testing Strategy

### 11.1 Unit Tests

```python
# tests/inference/test_reward_client.py

class TestRewardModelClient:
    async def test_score_single_input(self): ...
    async def test_score_batch(self): ...
    async def test_score_normalization(self): ...


# tests/distributed/test_reward_sync.py

class TestRewardWeightSync:
    def test_head_only_sync(self): ...
    def test_lora_sync_requires_merge(self): ...
    def test_full_sync(self): ...
    def test_training_mode_validation(self): ...
```

### 11.2 Integration Tests

```python
class TestRewardModelIntegration:
    async def test_head_only_training_cycle(self): ...
    async def test_lora_training_cycle(self): ...
    async def test_full_finetuning_cycle(self): ...
    async def test_reward_credit_assignment(self): ...
```

---

## 12. Open Questions & Future Work

### 12.1 Open Questions

1. **LoRA Adapter Sync Without Merge**: Can we sync LoRA weights directly instead of merging?
2. **Mixed Precision**: How to handle bf16 backbone with fp32 head?
3. **Multi-Head RMs**: Support for multi-task reward models?

### 12.2 Future Extensions

| Feature | Description | Priority |
|---------|-------------|----------|
| PRM Support | Step-level scoring for math/code | Medium |
| RM Ensemble | Multiple RMs with uncertainty | Low |
| Direct LoRA Sync | Sync adapters without merging | Medium |
| Quantized RMs | 4-bit/8-bit reward models | Low |

---

## 13. References

### 13.1 vLLM Documentation

- [Pooling Models](https://docs.vllm.ai/en/stable/models/pooling_models/)
- [OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)

### 13.2 Related RFCs

- [RFC: Add runtime weight update API (vLLM #5723)](https://github.com/vllm-project/vllm/issues/5723)
- [RFC: Flexible Weight Sync (vLLM #11399)](https://github.com/vllm-project/vllm/issues/11399)

### 13.3 Ludic Files Referenced

| File | Purpose |
|------|---------|
| `src/ludic/inference/client.py` | ChatClient protocol |
| `src/ludic/inference/vllm_client.py` | VLLMChatClient implementation |
| `src/ludic/inference/vllm_server.py` | Custom vLLM server |
| `src/ludic/distributed/publisher.py` | BroadcastPolicyPublisher |
| `src/ludic/distributed/interfaces.py` | Protocol definitions |

---

*This specification supports flexible reward model training modes based on representation salience and feature complexity. The choice between head-only, LoRA, or full fine-tuning should be guided by whether features are linearly separable, latent, or require new computational paths.*

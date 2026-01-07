# Implementation Plan: vLLM-Hosted Reward Model Support

> **Status**: Ready for Implementation
> **Spec Version**: 1.0
> **Target**: Ludic integration with vLLM reward model inference + weight sync

---

## Part A: Shared Context Document

### A.1 Interview Insights (Not in Spec)

These critical details were surfaced during the architect interview and are **not** documented in SPEC.md:

#### Use Case Dichotomy
| Use Case | Description | Implementation Impact |
|----------|-------------|----------------------|
| **Sequence-level RM** | RLHF cold-start: train RM separately, then use to train a *different* policy. vLLM hosting frees memory for policy. | Single forward pass per completion; sync once after RM training completes |
| **Token/Process-level RM (PRMs)** | **DEFERRED to Phase 2**. Overlapping sequence ingestion with KV cache reuse. Non-Markovian (full context matters). | Will need incremental scoring API; prompt caching critical |

#### Architectural Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Protocol Design** | Unified Client protocol | Single protocol with optional `score()`/`complete()` methods - maximum flexibility |
| **GPU Topology** | Colocation expected | Both policy + RM servers on same GPUs; RM expected smaller than policy |
| **Memory Strategy** | Static split | User specifies fixed % via `--gpu-memory-utilization` |
| **Port Strategy** | Offset convention | RM ports = policy ports + offset (default +1) |
| **Failure Handling** | Fail-fast | Scoring errors fail training step - no graceful degradation |

#### Training Mode Decisions
| Decision | Choice | Implementation Detail |
|----------|--------|----------------------|
| **LoRA Handling** | Merge and reinit | `merge_and_unload()` → sync → reinitialize fresh LoRA adapters |
| **Training Orchestration** | User orchestrates | Framework provides primitives; user scripts the sequence |
| **Sync Cadence** | Context-dependent | For cold-start: sync once RM is trained, then use for policy |

#### Data Shape Decisions
| Decision | Choice | Implementation Detail |
|----------|--------|----------------------|
| **Multi-head Support** | `n_labels` arbitrary | `n_labels=1` → scalar (no softmax); `n_labels>1` → multi-scalar vector |
| **Credit Composition** | Complex/composable | RM can be bonus (sequence-level) or implicit credit assigner (token-level) |

#### Testing Requirements
- **GPU Required**: Integration tests require GPU; skip in CI without GPU
- No mock NCCL - real GPU tests only

---

### A.2 Existing Codebase Patterns (Must Follow)

#### ChatClient Protocol (`src/ludic/inference/client.py`)
```python
class ChatClient(Protocol):
    async def complete(request: ChatCompletionRequest) -> Tuple[ChatResponse, Dict[str, Any]]
    def sync_weights(params: Mapping[str, Tensor], *, timeout_s: float, version: Optional[int]) -> str
```
- `complete()` is **async**, returns `(response, info_dict)` tuple
- `sync_weights()` is **synchronous** (blocking), returns version string

#### Two-Plane Weight Sync Pattern
1. **Control Plane (HTTP)**: `POST /update_param_batch` with metadata (name, dtype, shape)
2. **Data Plane (NCCL)**: Broadcast tensors in sorted key order
3. **Finalization**: Poll `/get_num_background_tasks` until drained, barrier sync

#### CreditAssigner Protocol (`src/ludic/training/types.py`)
```python
class CreditAssigner(Protocol):
    def compute(self, rollouts: List[Rollout]) -> Dict[RolloutStepKey, float]
```
- **Synchronous** - no async credit assignment currently
- Must return weight for EVERY step in EVERY rollout
- Called after rollout generation completes

#### Publisher Protocol (`src/ludic/distributed/interfaces.py`)
```python
class PolicyPublisher(Protocol):
    def publish(self, state_dict: Mapping[str, Tensor], version: Optional[int]) -> None
```
- Orchestrator pattern: single interface trainer talks to
- Composed from ControlPlane + TensorCommunicator

---

### A.3 Dependency Graph Between Implementation Phases

```
Phase 1: Core Infrastructure
    ├── ScoringRequest/Response types
    ├── RewardModelClient protocol (extends ChatClient)
    └── VLLMRewardClient (inference only)
            │
            ▼
Phase 2: Weight Synchronization
    ├── VLLMRewardClient + NCCL support
    ├── RewardWeightSyncExtension (server-side)
    ├── RewardModelPublisher
    └── vllm_reward_server.py with /score endpoint
            │
            ▼
Phase 3: Credit Assignment Integration
    ├── RewardModelCreditAssigner
    ├── Integration with RolloutEngine
    └── Algorithm presets (make_reward_rl, etc.)
            │
            ▼
Phase 4: Documentation & Examples
    ├── Examples for all 3 training modes
    ├── Updated CLAUDE.md
    └── Integration tests
```

---

### A.4 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **NCCL group conflicts** | Medium | High | Separate group_port for RM (offset convention); test colocation early |
| **Memory pressure with colocation** | High | Medium | Document static split requirements; provide sizing guidance |
| **Tokenizer mismatch** | Low | High | Validate tokenizer compatibility at client init; fail fast |
| **Version drift (policy vs RM)** | Medium | Medium | Track versions independently; add staleness warnings |
| **LoRA merge breaks optimizer state** | High | High | Document merge→sync→reinit pattern; add safeguards |

---

### A.5 Glossary

| Term | Definition in Ludic Context |
|------|----------------------------|
| **Reward Model (RM)** | LLM backbone + linear classification head; scores completions |
| **Cold-start RLHF** | Train RM on preferences → use RM to train a separate policy |
| **HEAD_ONLY** | Train only classification head; backbone frozen |
| **LORA** | Train LoRA adapters + head; backbone frozen |
| **FULL** | Full fine-tuning of backbone + head |
| **Two-plane architecture** | Control plane (HTTP) + Data plane (NCCL) for weight sync |
| **SAWItem** | State-Action-Weight training sample with tokens + mask + credit weight |
| **CreditAssigner** | Maps rollouts → per-step scalar weights for loss computation |

---

## Part B: Interface Contracts

### Phase 1 → Phase 2 Contract

**Phase 1 Produces:**
```python
# New types in src/ludic/inference/reward_client.py
@dataclass(frozen=True)
class ScoringRequest:
    model: str
    inputs: List[str]
    pooling_type: str = "last"
    normalize: bool = True
    n_labels: int = 1  # NEW: multi-head support

@dataclass(frozen=True)
class ScoringResponse:
    scores: List[Union[float, List[float]]]  # scalar or vector per input
    metadata: Dict[str, Any]

class RewardModelClient(Protocol):  # Extends ChatClient pattern
    async def score(request: ScoringRequest) -> Tuple[ScoringResponse, Dict]
    async def score_batch(inputs: List[str], *, normalize: bool) -> List[float]
    # sync_weights() inherited from unified protocol
```

**Phase 2 Consumes:**
- `ScoringRequest`/`ScoringResponse` types for server endpoint typing
- `RewardModelClient` protocol for publisher adapter

---

### Phase 2 → Phase 3 Contract

**Phase 2 Produces:**
```python
# VLLMRewardClient with full capabilities
class VLLMRewardClient:
    async def score_batch(inputs: List[str]) -> List[float]
    def sync_weights(params, *, training_mode: RewardModelTrainingMode)

# RewardModelPublisher
class RewardModelPublisher(PolicyPublisher):
    def publish(state_dict, version, *, training_mode: RewardModelTrainingMode)

# Server running at configurable port with /score endpoint
```

**Phase 3 Consumes:**
- `VLLMRewardClient` instance for async scoring in credit assignment
- Expectation: `score_batch()` is fast enough to not block training

---

### Phase 3 → Phase 4 Contract

**Phase 3 Produces:**
```python
# RewardModelCreditAssigner
class RewardModelCreditAssigner:
    def compute(rollouts: List[Rollout]) -> Dict[RolloutStepKey, float]

# Algorithm presets
def make_reward_rl(reward_client, mode, ...) -> RLAlgorithm
```

**Phase 4 Consumes:**
- Working end-to-end training loop for examples
- All components importable from clean module paths

---

## Part C: Final Implementation Plan

### C.1 Implementation Phases Overview

| Phase | Scope | Files | Dependencies |
|-------|-------|-------|--------------|
| **1** | Types & Protocols | `reward_types.py`, `client.py` | None |
| **2** | Client & Server | `vllm_reward_client.py`, `vllm_reward_server.py`, `reward_publisher.py` | Phase 1 |
| **3** | Credit Assignment | `reward_credit.py`, `reward_scorer.py`, `algorithm.py` | Phase 2 |
| **4** | Integration & Docs | `conftest.py`, `examples/`, `CLAUDE.md` | Phase 3 |

---

### C.2 Phase 1: Types & Protocols

#### New File: `src/ludic/inference/reward_types.py`

```python
class RewardModelTrainingMode(Enum):
    HEAD_ONLY = "head_only"   # Only classification head
    LORA = "lora"             # Merged LoRA + head
    FULL = "full"             # Complete model

class PoolingType(Enum):
    LAST = "last"
    MEAN = "mean"
    CLS = "cls"
    STEP = "step"  # Deferred for PRMs

@dataclass(frozen=True)
class ScoringRequest:
    model: str
    inputs: List[str]
    pooling_type: PoolingType = PoolingType.LAST
    normalize: bool = True
    n_labels: int = 1  # Multi-head support

@dataclass
class ScoringResponse:
    results: List[ScoringResult]  # score: Union[float, List[float]]
    model: str
    usage: Dict[str, int]
```

#### Modified File: `src/ludic/inference/client.py`

Add `RewardModelClient` protocol (parallel to `ChatClient`):

```python
class RewardModelClient(Protocol):
    async def score(request: ScoringRequest) -> Tuple[ScoringResponse, Dict]
    def sync_weights(params, *, training_mode: RewardModelTrainingMode) -> str
```

---

### C.3 Phase 2: Client & Server

#### New File: `src/ludic/inference/vllm_reward_client.py`

```python
class VLLMRewardClient:
    DEFAULT_PORT = 8001          # +1 from policy
    DEFAULT_GROUP_PORT = 51217   # +1 from policy

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        group_port: int = DEFAULT_GROUP_PORT,
        enable_weight_updates: bool = False,
        device: int = 0,
    )

    async def score(request: ScoringRequest) -> Tuple[ScoringResponse, Dict]
    async def score_batch(inputs: List[str]) -> List[float]
    def score_batch_sync(inputs: List[str]) -> List[float]  # For sync credit assignment

    def sync_weights(
        params: Mapping[str, Tensor],
        *,
        training_mode: RewardModelTrainingMode,
        version: Optional[int] = None,
    ) -> str
```

Key implementation details:
- HTTP client for `/score` endpoint
- NCCL communicator for weight sync (same pattern as `VLLMChatClient`)
- `_transform_params()` for HF→vLLM key normalization
- `_validate_params()` for training mode validation

#### New File: `src/ludic/inference/vllm_reward_server.py`

```python
# Server with RewardWeightSyncExtension + /score endpoint

class RewardWeightSyncExtension:
    """Worker extension for NCCL weight receive."""
    def init_communicator(host, port, world_size)
    def update_param_batch(metadata_list)
    def close_communicator()

# Endpoints:
@app.get("/health")           # Returns {"type": "reward_model"}
@app.post("/score")           # Batch scoring via vLLM encode()
@app.get("/get_world_size")
@app.get("/runtime_version")
@app.post("/init_communicator")
@app.post("/update_param_batch")
@app.post("/close_communicator")
```

#### New File: `src/ludic/distributed/reward_publisher.py`

```python
class RewardModelPublisher(PolicyPublisher):
    DEFAULT_HEAD_PATTERNS = ["score.", "classifier.", "lm_head."]

    def publish(
        state_dict: Mapping[str, Tensor],
        version: Optional[int] = None,
        *,
        training_mode: RewardModelTrainingMode = RewardModelTrainingMode.FULL,
    ) -> None

    def _transform_to_vllm(state_dict, training_mode) -> Dict
    def _validate_params(params, training_mode) -> None

def create_reward_publisher(client: VLLMRewardClient, ...) -> RewardModelPublisher
```

---

### C.4 Phase 3: Credit Assignment Integration

**Key Design Decision**: Pre-score rollouts at `RolloutEngine` level (async), store in metadata, then `RewardModelCreditAssigner` reads synchronously.

#### New File: `src/ludic/training/reward_scorer.py`

```python
class RewardModelScorer:
    """Async pre-processor that scores rollouts and stores in metadata."""

    def __init__(
        client: RewardScoringClient,
        score_key: str = "rm_score",
        score_level: str = "sequence",  # or "step"
        batch_size: int = 64,
    )

    async def score(rollouts: List[Rollout]) -> None:
        """Modifies rollouts in-place: rollout.meta[score_key] = float"""
```

#### New File: `src/ludic/training/reward_credit.py`

```python
CombineMode = Literal["replace", "add", "multiply", "weighted", "bonus"]

class RewardModelCreditAssigner:
    """Reads RM scores from metadata, combines with env rewards."""

    def __init__(
        mode: CombineMode = "bonus",
        rm_coeff: float = 1.0,
        alpha: float = 0.5,
        rm_score_key: str = "rm_score",
        score_level: str = "sequence",
        inner_assigner: Optional[CreditAssigner] = None,  # For GRPO normalization
    )

    def compute(rollouts: List[Rollout]) -> Dict[RolloutStepKey, float]

# Factory functions for common configurations
def make_rm_grpo_assigner(group_size, mode, ...) -> RewardModelCreditAssigner
def make_rm_scalerl_assigner(group_size, mode, ...) -> RewardModelCreditAssigner
```

#### Modified File: `src/ludic/training/batching/rollout_engine.py`

```python
async def generate_batch(
    self,
    *,
    requests: List[RolloutRequest],
    credit_assigner: CreditAssigner,
    reward_scorer: Optional[RewardModelScorer] = None,  # NEW
    ...
) -> SAWBatch:
    rollouts = await self.generate_rollouts(...)

    # NEW: Score before credit assignment
    if reward_scorer is not None:
        await reward_scorer.score(rollouts)

    weights = credit_assigner.compute(rollouts)
    ...
```

#### Modified File: `src/ludic/training/algorithm.py`

Add presets:

```python
def make_grpo_with_rm(group_size, mode="bonus", rm_coeff=1.0, ...) -> RLAlgorithm
def make_scalerl_with_rm(group_size, mode="bonus", ...) -> RLAlgorithm
def make_reward_rl(mode="replace", ...) -> RLAlgorithm  # Simple non-GRPO version
```

---

### C.5 Phase 4: Integration, Testing & Documentation

#### Colocation Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINER PROCESS                             │
│   PolicyPublisher (NCCL A)          RewardPublisher (NCCL B)        │
│         │ group_port=51216                 │ group_port=51217       │
└─────────│─────────────────────────────────│────────────────────────┘
          ▼                                 ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   vLLM Policy Server    │   │   vLLM Reward Server    │
│   Port: 8000, Mem: 70%  │   │   Port: 8001, Mem: 30%  │
└─────────────────────────┘   └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                         CUDA Device 0
```

#### Test Structure

```
tests/
├── inference/
│   └── test_reward_client.py        # Unit tests (no GPU)
├── distributed/
│   └── test_reward_publisher.py     # Unit tests (no GPU)
├── training/
│   ├── test_reward_credit.py        # Unit tests (no GPU)
│   └── test_reward_algorithm.py
└── integration/
    ├── test_reward_server.py        # GPU required
    ├── test_reward_weight_sync.py   # GPU required
    └── test_colocation.py           # GPU required
```

#### Example Scripts

```
examples/reward_model/
├── README.md
├── train_rm_head_only.py
├── train_rm_lora.py
├── train_rm_full.py
└── cold_start_rlhf.py    # Full workflow: train RM → use for policy
```

---

### C.6 File Summary

#### New Files (8)

| File | Purpose |
|------|---------|
| `src/ludic/inference/reward_types.py` | Types: `RewardModelTrainingMode`, `ScoringRequest`, `ScoringResponse` |
| `src/ludic/inference/vllm_reward_client.py` | `VLLMRewardClient` with NCCL support |
| `src/ludic/inference/vllm_reward_server.py` | vLLM server with `/score` endpoint |
| `src/ludic/distributed/reward_publisher.py` | `RewardModelPublisher` |
| `src/ludic/training/reward_scorer.py` | `RewardModelScorer` (async pre-processor) |
| `src/ludic/training/reward_credit.py` | `RewardModelCreditAssigner` + factories |
| `tests/integration/test_reward_*.py` | Integration tests |
| `examples/reward_model/` | Example scripts + README |

#### Modified Files (5)

| File | Changes |
|------|---------|
| `src/ludic/inference/client.py` | Add `RewardModelClient` protocol |
| `src/ludic/distributed/interfaces.py` | Add `extra` param to `announce_update_batch()` |
| `src/ludic/distributed/adapters/vllm.py` | Pass `extra` in `VllmControlPlane` |
| `src/ludic/training/batching/rollout_engine.py` | Add `reward_scorer` param |
| `src/ludic/training/algorithm.py` | Add RM algorithm presets |

---

### C.7 Implementation Order

1. **Phase 1** (Types) - ~1 day
   - Create `reward_types.py`
   - Add `RewardModelClient` protocol to `client.py`

2. **Phase 2** (Client/Server) - ~3 days
   - Implement `VLLMRewardClient`
   - Implement `vllm_reward_server.py`
   - Implement `RewardModelPublisher`
   - Add unit tests for param validation

3. **Phase 3** (Credit Assignment) - ~2 days
   - Implement `RewardModelScorer`
   - Implement `RewardModelCreditAssigner`
   - Modify `RolloutEngine` for scorer integration
   - Add algorithm presets

4. **Phase 4** (Integration) - ~2 days
   - Add test fixtures
   - Write integration tests
   - Create example scripts
   - Update CLAUDE.md

---

### C.8 Critical Files Reference (for Implementation)

Files to study before implementing:

1. **`src/ludic/inference/vllm_client.py`** - Pattern for NCCL setup, weight sync
2. **`src/ludic/inference/vllm_server.py`** - Pattern for server endpoints, worker extension
3. **`src/ludic/distributed/adapters/vllm.py`** - VllmControlPlane/TensorCommunicator
4. **`src/ludic/training/credit_assignment.py`** - Existing assigner patterns
5. **`src/ludic/training/algorithm.py`** - Algorithm preset patterns

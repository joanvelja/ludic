# Handoff Document: RM Implementation Completion

**Branch:** `RM-vllm`
**Date:** 2026-01-08
**Status:** Ready for implementation - all issues verified, plan finalized
**Estimated Effort:** ~8-9 hours (highly parallelizable)

---

## Executive Summary

The Reward Model (RM) implementation is **~90% complete** relative to SPEC.md. The architecture is sound - these are surgical fixes, not rewrites. This document provides everything needed to complete the implementation with battle-ready test coverage.

**Key Insight:** The bugs are localized and independent. Phases 1-3 can be executed in parallel by multiple agents or developers.

---

## Quick Start: What to Fix First

If you're continuing this work, start here:

```bash
# 1. Fix the 4 critical bugs (30 min total, all parallel)
#    - Bug #1: src/ludic/distributed/adapters/vllm.py:68-73
#    - Bug #2: src/ludic/inference/vllm_reward_server.py:191-194
#    - Bug #3: src/ludic/training/reward_credit.py:155
#    - Bug #4: src/ludic/distributed/__init__.py

# 2. Run existing tests to verify no regressions
pytest tests/ -v --ignore=tests/integration/

# 3. Create new unit tests (2.5 hrs, all parallel)
# 4. Run integration tests (2.5 hrs, GPU required)
```

---

## Phase 1: Critical Bug Fixes (BLOCKING)

### Bug #1: VllmTensorCommunicator Attribute Bug
**File:** `src/ludic/distributed/adapters/vllm.py:68-73`
**Severity:** CRITICAL - Will crash at runtime
**Time:** 10 min

**Current Code (BROKEN):**
```python
class VllmTensorCommunicator(TensorCommunicator):
    def __init__(self, client: VLLMClient):
        if not client._pynccl_comm:      # ❌ Attribute doesn't exist
            raise RuntimeError("vLLM Client has no active NCCL communicator")
        self._comm = client._pynccl_comm  # ❌ Should be _policy_comm.comm
        self._rank = client._rank         # ❌ Should be _policy_comm.rank
```

**Fixed Code:**
```python
class VllmTensorCommunicator(TensorCommunicator):
    def __init__(self, client: VLLMClient):
        if not client._policy_comm:
            raise RuntimeError("vLLM Client has no active NCCL communicator")
        self._comm = client._policy_comm.comm
        self._rank = client._policy_comm.rank
```

**Verification:**
- `vllm_client.py:166` defines `self._policy_comm: Optional[NCCLCommunicator]`
- `client_base.py:59-60` shows `NCCLCommunicator.comm` and `.rank` attributes

---

### Bug #2: Double Sigmoid Normalization
**File:** `src/ludic/inference/vllm_reward_server.py:191-194`
**Severity:** CRITICAL - Produces mathematically incorrect scores
**Time:** 5 min

**Problem:** `PoolingParams(normalize=True)` on line 173 already applies sigmoid internally. Lines 191-194 apply it again, causing `sigmoid(sigmoid(x))`.

**Fix:** DELETE lines 191-194 entirely:
```python
# DELETE THIS BLOCK:
            # Apply sigmoid if normalize is requested
            if request.normalize:
                import math
                score_val = 1.0 / (1.0 + math.exp(-score_val))
```

**Verification Test:**
```python
def test_score_normalization_is_single_sigmoid():
    # Score with normalize=True should return sigmoid(logit), not sigmoid(sigmoid(logit))
    # For logit=2.0: single sigmoid ≈ 0.88, double sigmoid ≈ 0.71
    pass
```

---

### Bug #3: Silent Data Corruption in RewardModelCreditAssigner
**File:** `src/ludic/training/reward_credit.py:155`
**Severity:** HIGH - Silent training corruption
**Time:** 10 min

**Current Code (BROKEN):**
```python
rm_score = rollout.meta.get(self.rm_score_key, 0.0)  # Silent default
```

**Fixed Code:**
```python
rm_score = rollout.meta.get(self.rm_score_key)
if rm_score is None:
    import logging
    logging.getLogger(__name__).warning(
        f"Rollout {rollout.id} missing '{self.rm_score_key}' in meta, using 0.0. "
        f"Did you forget to run RewardModelScorer?"
    )
    rm_score = 0.0
```

**Decision:** Warning + 0.0 chosen over raising error (non-breaking for existing code).

---

### Bug #4: Missing Distributed Module Exports
**File:** `src/ludic/distributed/__init__.py`
**Severity:** HIGH - Breaks documented import paths
**Time:** 5 min

**Current Code (line 10):**
```python
from .adapters import VllmControlPlane, VllmTensorCommunicator, create_vllm_publisher
```

**Fixed Code:**
```python
from .adapters import (
    VllmControlPlane,
    VllmTensorCommunicator,
    create_vllm_publisher,
    RMPublisherAdapter,      # ADD
    create_rm_publisher,     # ADD
)

__all__ = [
    # ... existing items ...
    "RMPublisherAdapter",    # ADD
    "create_rm_publisher",   # ADD
]
```

---

## Phase 2: High-Priority Fixes

### Issue #5: HEAD_ONLY Mode is Decorative
**Files:** `src/ludic/inference/vllm/server_base.py`
**Time:** 1 hour

**Problem:** `training_mode` parameter is logged but never used to filter weights.

**Fix Approach:**
```python
HEAD_PATTERNS = ["score.", "classifier.", "lm_head."]
LORA_PATTERNS = ["lora_", ".lora."]

def _filter_weights_by_mode(metadata: List[Dict], training_mode: str) -> List[Dict]:
    if training_mode == "full":
        return metadata
    elif training_mode == "head_only":
        return [m for m in metadata if any(p in m["name"] for p in HEAD_PATTERNS)]
    elif training_mode == "lora":
        return [m for m in metadata if
                any(p in m["name"] for p in HEAD_PATTERNS + LORA_PATTERNS)]
    return metadata
```

**Integration point:** Call from `/update_param_batch` handler after receiving metadata.

---

### Issue #6: Hardcoded 1s Sleep in Weight Sync
**File:** `src/ludic/inference/vllm/client_base.py:304`
**Time:** 1.5 hours

**Current Code:**
```python
# Brief pause for server to set up receivers
# TODO: Replace with proper ready signal
time.sleep(1.0)
```

**Fix (Ready Signal Protocol):**

1. **Server changes** (`server_base.py`):
   - Add `pending_batches: Dict[str, asyncio.Event]` to `ServerState`
   - In `/update_param_batch`: create batch_id, create Event, return batch_id
   - Add `/batch_ready/{batch_id}` endpoint that waits on Event
   - Background task sets Event after NCCL receivers initialized

2. **Client changes** (`client_base.py`):
   - Extract batch_id from `/update_param_batch` response
   - Poll `/batch_ready/{batch_id}` with timeout instead of sleep
   - Keep `time.sleep(1.0)` as fallback with deprecation warning

---

### Issue #7: aiohttp Session Created Per Request
**File:** `src/ludic/inference/vllm_client.py:434`
**Time:** 30 min

**Current Code:**
```python
async def score(self, request: ScoringRequest) -> ...:
    async with aiohttp.ClientSession() as session:  # New session per call
        async with session.post(url, json=payload) as resp:
```

**Fixed Code:**
```python
def __init__(self, ...):
    # ...
    self._async_session: Optional[aiohttp.ClientSession] = None

async def _get_async_session(self) -> aiohttp.ClientSession:
    if self._async_session is None or self._async_session.closed:
        self._async_session = aiohttp.ClientSession()
    return self._async_session

async def score(self, request: ScoringRequest) -> ...:
    session = await self._get_async_session()
    async with session.post(url, json=payload) as resp:
        # ...
```

**Also update:** `get_policy_version()` and `get_reward_model_version()` methods.

---

## Phase 3: Unit Tests (Battle-Ready)

### Test File Structure
```
tests/
├── test_reward_types.py              # NEW - 30 min
├── test_reward_credit.py             # NEW - 45 min
├── test_reward_scorer.py             # NEW - 45 min (mocked client)
├── test_vllm_tensor_communicator.py  # NEW - 30 min (mocked)
├── integration/
│   ├── test_vllm_reward_server.py    # NEW - 1 hr (GPU)
│   └── test_rm_weight_sync.py        # NEW - 1.5 hr (GPU)
```

### Test Cases by File

#### `tests/test_reward_types.py`
```python
class TestScoringRequest:
    def test_from_list_creates_frozen_tuple(self): ...
    def test_is_immutable(self): ...

class TestPoolingType:
    def test_values_match_vllm(self): ...

class TestRewardModelTrainingMode:
    def test_all_modes_present(self): ...

class TestScoringResponse:
    def test_defaults(self): ...
```

#### `tests/test_reward_credit.py`
```python
class TestRewardModelCreditAssigner:
    def test_replace_mode_ignores_inner_assigner(self): ...
    def test_add_mode_adds_rm_to_base(self): ...
    def test_multiply_mode_multiplies_rm_with_base(self): ...
    def test_weighted_mode_interpolates(self): ...
    def test_bonus_mode_applies_coefficient(self): ...
    def test_missing_rm_score_logs_warning(self): ...  # Validates Bug #3 fix
    def test_with_group_normalized_inner_assigner(self): ...
    def test_empty_rollouts_handled(self): ...
```

#### `tests/test_reward_scorer.py`
```python
class TestRewardModelScorer:
    def test_format_rollout_concatenates_steps(self): ...
    def test_score_stores_in_metadata(self): ...
    def test_score_batches_correctly(self): ...
    def test_custom_score_key(self): ...
    def test_empty_rollouts_handled(self): ...
```

#### `tests/test_vllm_tensor_communicator.py`
```python
class TestVllmTensorCommunicator:
    def test_init_raises_when_policy_comm_none(self): ...
    def test_rank_returns_correct_value(self): ...
    def test_broadcast_calls_through(self): ...
    def test_barrier_calls_through(self): ...
```

---

## Phase 4: Integration Tests (GPU Required)

### `tests/integration/test_vllm_reward_server.py`
```python
@pytest.mark.integration
@pytest.mark.gpu
class TestVllmRewardServer:
    def test_health_endpoint_returns_reward_model_type(self): ...
    def test_score_single_input(self): ...
    def test_score_batch_inputs(self): ...
    def test_score_normalization_is_single_sigmoid(self): ...  # Validates Bug #2 fix
    def test_weight_sync_endpoints(self): ...
```

### `tests/integration/test_rm_weight_sync.py`
```python
@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
class TestRMWeightSync:
    def test_full_weight_sync_cycle(self): ...
    def test_head_only_mode_filters_weights(self): ...  # Validates Issue #5 fix
    def test_version_tracking(self): ...
    def test_rm_publisher_adapter_publish(self): ...
```

---

## Execution Order & Parallelization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: CRITICAL BUGS (~30 min)                        │
│                     ════════════════════════════════                        │
│                                                                             │
│   Bug #1 ──┬── Bug #2 ──┬── Bug #3 ──┬── Bug #4                            │
│   (10 min) │  (5 min)   │  (10 min)  │  (5 min)                            │
│            │            │            │                                      │
│            └────────────┴────────────┴─► All PARALLEL                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: HIGH-PRIORITY (~3 hrs)                         │
│                     ═══════════════════════════════                         │
│                                                                             │
│   Issue #5 ──────┬── Issue #6 ──────┬── Issue #7                           │
│   (1 hr)         │   (1.5 hr)       │   (30 min)                           │
│   HEAD_ONLY      │   Ready Signal   │   Session                            │
│                  │                  │                                       │
│                  └──────────────────┴─► Parallel in pairs                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: UNIT TESTS (~2.5 hrs)                          │
│                     ══════════════════════════════                          │
│                                                                             │
│   reward_types ───┬── reward_credit ──┬── reward_scorer ──┬── tensor_comm  │
│   (30 min)        │   (45 min)        │   (45 min)        │   (30 min)     │
│                   │                   │                   │                 │
│                   └───────────────────┴───────────────────┴─► All PARALLEL  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 4: INTEGRATION (~2.5 hrs, GPU)                    │
│                     ════════════════════════════════════                    │
│                                                                             │
│   test_vllm_reward_server ────► test_rm_weight_sync                        │
│   (1 hr)                        (1.5 hr)                                    │
│                                                                             │
│   Sequential (single GPU constraint)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Verification Strategy

### Unit Tests (No GPU)
```bash
pytest tests/test_reward_types.py tests/test_reward_credit.py \
       tests/test_reward_scorer.py tests/test_vllm_tensor_communicator.py -v
```

**Note (test handling):** RM tests are now handled in the repo but their intent is not described here to avoid reward hacking. Engineers should run the commands below as they iterate and focus on correctness, not optimizing for test artifacts.

### Expanded Unit Test Set (No GPU)
```bash
pytest tests/test_reward_types.py tests/test_reward_credit.py \
       tests/test_reward_scorer.py tests/test_vllm_tensor_communicator.py \
       tests/test_vllm_weight_sync_filtering.py tests/test_rm_publisher_adapter.py -v
```

### Integration Tests (GPU Required)
```bash
# Terminal 1: Start reward server
CUDA_VISIBLE_DEVICES=0 python -m ludic.inference.vllm_reward_server \
    --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --port 8001 &

# Terminal 2: Run tests
pytest tests/integration/test_vllm_reward_server.py \
       tests/integration/test_rm_weight_sync.py -v --tb=short
```

### Integration Tests (GPU, Env-Gated)
```bash
# Set VLLM_RM_MODEL to enable RM server tests; set VLLM_RM_SYNC=1 to enable NCCL weight sync tests.
VLLM_RM_MODEL=... pytest tests/integration/test_vllm_reward_server.py -v --tb=short
VLLM_RM_MODEL=... VLLM_RM_SYNC=1 pytest tests/integration/test_rm_weight_sync.py -v --tb=short
```

**Engineering guidelines (be wary of):**
- Avoid “teaching to the test”: fix root causes, not just observable outputs.
- Don’t mask failures with broad exception handling or silent defaults.
- Keep determinism: avoid time/sleep-based flakiness; prefer explicit ready/health signals.
- Be careful with weight filtering: only exclude what the training mode mandates.
- Preserve forward/backward compatibility: avoid breaking API exports and metadata contracts.

### Manual E2E Smoke Test
```bash
python examples/reward_model/cold_start_rlhf.py \
    --mode full \
    --policy-model Qwen/Qwen2.5-0.5B-Instruct \
    --rm-base-model Qwen/Qwen2.5-0.5B \
    --preference-data data/preferences.jsonl \
    --rm-steps 10 \
    --policy-steps 5
```

---

## Component Status

| Component | Status | Blocking Issues |
|-----------|--------|-----------------|
| `reward_types.py` | ✅ Complete | None |
| `vllm_reward_server.py` | ⚠️ Bug #2 | Double sigmoid |
| `vllm_client.py` | ⚠️ Issue #7 | Session per request |
| `reward_credit.py` | ⚠️ Bug #3 | Silent defaults |
| `reward_scorer.py` | ✅ Complete | None |
| `VllmTensorCommunicator` | ❌ Bug #1 | Wrong attributes |
| `RMPublisherAdapter` | ⚠️ Bug #4 | Not exported |
| `vllm/client_base.py` | ⚠️ Issue #6 | Hardcoded sleep |
| `vllm/server_base.py` | ⚠️ Issue #5 | HEAD_ONLY decorative |
| Unit tests | ❌ Missing | Zero coverage |
| Integration tests | ❌ Missing | Zero coverage |

---

## Key Files Reference

```
src/ludic/inference/
├── reward_types.py              # Core RM types (complete)
├── vllm_reward_server.py        # RM server - Bug #2
├── vllm_client.py               # Unified client - Issue #7
└── vllm/
    ├── client_base.py           # NCCL infrastructure - Issue #6
    └── server_base.py           # Server infrastructure - Issue #5

src/ludic/training/
├── reward_credit.py             # RewardModelCreditAssigner - Bug #3
├── reward_scorer.py             # RMScorer (complete)
└── algorithm.py                 # make_grpo_with_rm(), make_bradley_terry()

src/ludic/distributed/
├── __init__.py                  # Exports - Bug #4
└── adapters/vllm.py             # VllmTensorCommunicator - Bug #1

examples/reward_model/
├── README.md                    # Usage documentation
├── cold_start_rlhf.py           # End-to-end example
└── train_rm_bradley_terry.py    # RM training example
```

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Missing rm_score behavior | Warning + 0.0 | Non-breaking for existing code |
| RMScoringCreditAssigner | Leave in examples | Thin wrapper, users can copy |
| Test scope | Full unit + GPU integration | Battle-ready requirement |
| Ready signal fallback | Keep sleep with deprecation | Graceful degradation |

---

## Git State

**Modified files:**
- `.gitignore`
- `src/ludic/distributed/__init__.py`
- `src/ludic/distributed/adapters/__init__.py`
- `src/ludic/distributed/adapters/vllm.py`
- `src/ludic/distributed/interfaces.py`
- `src/ludic/inference/__init__.py`
- `src/ludic/inference/vllm_client.py`
- `src/ludic/inference/vllm_server.py`
- `src/ludic/training/__init__.py`
- `src/ludic/training/algorithm.py`
- `src/ludic/training/loss.py`

**New files (untracked):**
- `AGENTS.md`
- `examples/reward_model/`
- `src/ludic/inference/reward_types.py`
- `src/ludic/inference/vllm/`
- `src/ludic/inference/vllm_reward_server.py`
- `src/ludic/training/preference_utils.py`
- `src/ludic/training/reward_credit.py`
- `src/ludic/training/reward_scorer.py`

---

## Next Session Checklist

- [ ] Fix Bug #1 (VllmTensorCommunicator)
- [ ] Fix Bug #2 (Double sigmoid)
- [ ] Fix Bug #3 (Silent rm_score)
- [ ] Fix Bug #4 (Missing exports)
- [ ] Implement Issue #5 (HEAD_ONLY filtering)
- [ ] Implement Issue #6 (Ready signal)
- [ ] Implement Issue #7 (Session reuse)
- [ ] Create test_reward_types.py
- [ ] Create test_reward_credit.py
- [ ] Create test_reward_scorer.py
- [ ] Create test_vllm_tensor_communicator.py
- [ ] Create test_vllm_reward_server.py (GPU)
- [ ] Create test_rm_weight_sync.py (GPU)
- [ ] Run E2E smoke test with cold_start_rlhf.py

**The implementation is architecturally sound. These are surgical fixes.**

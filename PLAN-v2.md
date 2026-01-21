# RM Trainer V2: Proper Integration with Ludic Infrastructure (Revised Plan)

## Problem Statement

The current RM trainer (`examples/reward_model/train_rm_bradley_terry.py`) is a **standalone script** that bypasses Ludic's training infrastructure:

| Feature | Ludic Trainer | Current RM Trainer |
|---------|---------------|-------------------|
| Micro-batching | ✅ `split_items_by_token_budget()` | ❌ Entire batch must fit in VRAM |
| Gradient accumulation | ✅ Loss scaling across micro-batches | ❌ Single forward/backward per step |
| GPU monitoring | ✅ Trainer `profile_memory` stats | ❌ None |
| Stats aggregation | ✅ Reducers + automatic distribution | ❌ Manual CSV logging |
| LoRA support | ✅ Merge/unmerge/publish | ⚠️ Not integrated |
| Classification metrics | ❌ Not implemented | ❌ Not implemented |
| FSDP2 distributed | ✅ Gradient sync optimization | ❌ Single GPU only |

**Result**: 60% test accuracy with overfitting. We need proper validation before scaling.

---

## Proposed Improvements

### 1. GPU Usage Monitoring Utility (Removed; inherit Trainer)

**Goal**: Use existing Trainer GPU memory profiling rather than introducing a new utility.

**Decision**: Drop `gpu_monitor.py`. RMTrainer will use `TrainerConfig.profile_memory` and leverage Trainer's built-in GPU stats (`gpu_mem_*`, `gpu_*_peak_mb`).

**Integration**:
- Add `--profile-memory` (or `--profile-gpu`) arg to RMTrainer CLI that maps to `TrainerConfig.profile_memory`.
- Ensure RMTrainer logs Trainer stats (through reducers or per-step logs).

---

### 2. Micro-Batching (Pair-Aware) + Implicit Gradient Accumulation

**Key insight**: Ludic uses `micro_token_budget` (not `gradient_accumulation_steps`). This already performs implicit gradient accumulation across micro-batches.

**Critical Fix**: Preference pairs must **never be split** across micro-batches. The current `split_items_by_token_budget()` sorts by length and is pair-agnostic, which can drop incomplete pairs in Bradley-Terry.

**Plan**: Introduce a *pair-aware micro-batching path* for preference data.

**High-level flow (Trainer)**:
```
1. BatchSource yields macro-batch (N items, N even)
2. Pair-aware packing into micro-batches (pairs are atomic units)
3. For each micro-batch k:
   - forward() → loss
   - scaled_loss = loss * (num_pairs_k / total_pairs)
   - backward()
4. ONE optimizer.step() after all micro-batches
```

**Implementation approach** (RMTrainer-specific, no global behavior change):
- Implement `RMTrainer._split_pairs_by_token_budget()` which:
  - groups items by `pair_id`
  - validates each group has exactly `chosen` and `rejected`
  - computes **pair length** = max(seq_len(chosen), seq_len(rejected))
  - packs pairs by token budget using pair length and pair count
  - emits micro-batches with both items per pair

**Why RMTrainer instead of core batching changes?**
- Keeps generic micro-batching behavior intact for all other algorithms.
- Preference pair semantics are domain-specific; RMTrainer is the right place.

---

### 3. Classification Metrics: Preference + Calibration + FPR/FNR

We need *pairwise* metrics plus optional per-response classification metrics for deployment.

**New file**: `src/ludic/training/classification_metrics.py`

**Metrics to compute**:

**Pairwise (primary, correct for BT):**
- Pairwise accuracy: `P(score_chosen > score_rejected)`
- Mean margin, margin std
- Negative margin rate
- Margin percentiles
- Calibration ECE based on pairwise probability `p = sigmoid(beta * (s_c - s_r))`

**Per-response (optional):**
- Treat chosen=positive, rejected=negative
- Threshold on **scores** to compute FPR/FNR/TPR
- Compute ROC-AUC (chosen vs rejected scores)

**Critical change in eval loop**:
- Collect **per-item scores**, not just mean scores per batch.
- Use scores to compute all metrics (avoid deriving metrics from batch means).

**Where used**:
- `evaluate()` in RMTrainer returns structured metrics
- Log metrics via Trainer reducers or RMTrainer logging

---

### 4. LoRA Support (Attention + FFN + Always-Train Reward Head)

**Goal**: Enable LoRA while keeping the scalar reward head fully trainable.

**Target Modules** (as specified):
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- FFN/MLP: `gate_proj`, `up_proj`, `down_proj`

**Implementation**:
- Add CLI args: `--lora`, `--lora-rank`, `--lora-alpha`, `--lora-dropout`, `--lora-target-modules`
- Apply PEFT config to base model (task type `SEQ_CLS`)
- Ensure reward head is always trainable (use `modules_to_save` or explicit unfreeze)

**Rationale**: LoRA should not freeze the reward head since calibration and accuracy depend on head expressivity.

---

### 5. Hyperparameter Improvements

**Current defaults are suboptimal**. Based on RM training literature:

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| `--lr` | 1e-5 | 5e-6 | Lower LR prevents overfitting |
| `--weight-decay` | 0.01 | 0.01-0.05 | Stay conservative; 0.1 too aggressive for LLMs |
| `--warmup-steps` | (none) | 50 | Stabilize early training |
| `--lr-scheduler` | (none) | cosine | Gradual decay |
| `--beta` | 1.0 | 1.0 | Keep default; tune empirically if needed |
| `--regularization-lambda` | 0.0 | 0.001-0.01 | Start small, tune up if overfitting |
| `--label-smoothing` | (none) | 0.05 | Conservative smoothing; 0.1 too aggressive |

**Add LR scheduler**:
```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.steps,
)

# In training loop:
scheduler.step()
```

**Add label smoothing to BradleyTerryLoss**:
```python
# In loss.py BradleyTerryLoss:
if self.label_smoothing > 0:
    # Smooth labels: 1.0 -> (1 - eps), 0.0 -> eps
    labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
```

---

## RMTrainer V2: Detailed Implementation Plan (Subclass of Trainer)

**Goal**: Production-quality RMTrainer built as a subclass of `Trainer` that is:
- redundancy-aware (reuse Trainer infra where possible)
- preference-specific (pair-aware micro-batching, BT metadata, eval metrics)
- safe for FSDP2 + LoRA integration

### A. Class Shape and Responsibilities

**File**: `src/ludic/training/rm_trainer.py` (new)

```python
class RMTrainer(Trainer):
    """Trainer specialized for preference-based reward modeling.

    Extends Trainer to:
      - enforce pair-aware micro-batching
      - inject preference metadata into batches
      - compute RM-specific eval metrics
    """
```

**RMTrainer responsibilities**:
1. **Data collation**: include preference metadata (`pair_id`, `role`, `label`).
2. **Pair-aware micro-batching**: keep chosen/rejected together and scale loss by pairs.
3. **Eval metrics**: compute pairwise + per-response metrics from per-item scores.
4. **Use Trainer for everything else**: optimizer, scheduler, FSDP2, LoRA merge/unmerge, GPU profiling.

### B. RMTrainerConfig (Subclass of TrainerConfig)

**Philosophy**: If we subclass Trainer → RMTrainer, we should also subclass TrainerConfig → RMTrainerConfig, so RM-specific knobs are explicit, typed, and discoverable (no ad‑hoc kwargs).

**New dataclass**: `src/ludic/training/rm_trainer.py` (or `rm_config.py` if preferred, then imported).

```python
@dataclass
class RMTrainerConfig(TrainerConfig):
    # Preference batching behavior
    rm_pair_aware: bool = True
    rm_require_complete_pairs: bool = True
    rm_drop_incomplete_pairs: bool = True  # if False, raise

    # Eval / metrics
    rm_eval_thresholds: Optional[List[float]] = None
    rm_log_per_response_metrics: bool = True
    rm_calibration_bins: int = 10

    # Optional: metadata policy
    rm_meta_keys: Tuple[str, ...] = ("pair_id", "role", "label")
```

**CLI mapping**: The RM trainer CLI should build an `RMTrainerConfig` and pass it to RMTrainer. No TrainerConfig mutation inside training code.

### C. Pair-Aware Micro-Batching (Core Logic)

**New helper in RMTrainer**:

```python
def _group_pairs(items: List[SAWItem]) -> List[Tuple[SAWItem, SAWItem]]:
    # 1. group by pair_id
    # 2. verify one chosen + one rejected
    # 3. drop or error on incomplete pairs based on config
```

```python
def _split_pairs_by_token_budget(
    pairs: List[Tuple[SAWItem, SAWItem]],
    micro_token_budget: int,
    max_seq_len: int,
) -> List[List[SAWItem]]:
    # 1. compute pair length = max(len(chosen), len(rejected))
    # 2. sort pairs by pair length (optional, to reduce padding waste)
    # 3. pack pairs into micro-batches using pair length * pair_count
    # 4. return flattened micro-batches where each pair contributes 2 items
```

**Edge Cases**:
- Items longer than `max_seq_len`: use existing truncation logic or drop (mirror `split_items_by_token_budget` behavior).
- Incomplete pair after truncation: if `rm_require_complete_pairs`, drop the pair and log warning.

**Loss scaling**:
- Use pair count instead of item count:
  - `scaled_loss = loss * (num_pairs_in_micro / total_pairs)`

### D. Metadata Injection (Structured, Redundancy‑Aware)

**Goal**: Avoid ad‑hoc mutation of the tensor dict while keeping preference metadata local to RMTrainer.

**Plan**: Add a dedicated collator for preference batches and use it only in RMTrainer.

**New helper (RMTrainer‑local or new file `training/batching/preference.py`)**:
```python
def collate_preference_items(
    items: List[SAWItem],
    *,
    pad_token_id: int,
    device: torch.device,
    meta_keys: Sequence[str] = ("pair_id", "role", "label"),
) -> Dict[str, Any]:
    # 1. Delegate tensor collation to collate_saw_items(...)
    # 2. Attach a structured meta dict by pulling keys from SAWItem.meta
```

**Behavior**:\n
- Uses existing tensor collation without touching global code paths.\n
- Ensures meta is present and ordered consistently (matching items list).\n
- Centralizes preference-specific assumptions in one small function.\n

**RMTrainer usage**:
```python
batch = collate_preference_items(..., meta_keys=self.cfg.rm_meta_keys)
loss, stats = self.algo.compute_loss(self.model, batch, ...)
```

**Why this is more elegant**:\n
- Single responsibility (preference collation = tensors + metadata).\n
- No surprise mutation later in the loop.\n
- Easy to unit test in isolation.\n
- Keeps base `collate_saw_items()` untouched.

### E. Eval Metrics Implementation

**Steps**:
1. Run forward pass to get logits for each item.
2. Compute per-item scores (`logits.squeeze(-1)`).
3. Reconstruct pairs by `pair_id` in eval set.
4. Compute:
   - pairwise accuracy
   - margin stats
   - ROC-AUC (chosen vs rejected scores)
   - calibration (ECE on `sigmoid(beta * margin)`)
   - optional FPR/FNR at thresholds

**RMTrainer evaluate** should return a dict of:
```
{
  "eval_loss": ...,
  "eval_pairwise_accuracy": ...,
  "eval_mean_margin": ...,
  "eval_margin_std": ...,
  "eval_negative_margin_rate": ...,
  "eval_roc_auc": ...,
  "eval_ece": ...,
  "eval_fpr@thr": ...,
  "eval_fnr@thr": ...,
}
```

### F. LoRA Integration Details (RMTrainer)

**Implementation**:
- Apply LoRA to base model.
- Ensure classifier head always trainable:
  - Use PEFT `modules_to_save=["score", "classifier", "lm_head"]` depending on model head name.
  - Or explicitly unfreeze reward head parameters after wrapping.

**Note**: RMTrainer should not override Trainer’s LoRA merge/unmerge logic; it inherits this.

### G. Scheduler/Warmup Integration

**Implementation**:
- Use standard HF scheduler in RMTrainer loop.
- Integrate `scheduler.step()` with the Trainer loop (macro-step), not per micro-step.

### H. Redundancy-Aware Integration Plan

**Use existing Trainer** for:
- optimizer creation + step
- gradient clipping
- FSDP2 gradient sync
- LoRA merge/unmerge and publish
- GPU memory profiling

**RMTrainer overrides only**:
- batch splitting + collation
- eval metrics
- loss scaling (pair-based)

**Avoid**:
- duplicate GPU monitoring
- duplicate loggers when reducers already exist

---

## Files to Modify

| File | Changes | Phase |
|------|---------|-------|
| `src/ludic/training/rm_trainer.py` | **NEW**: RMTrainer subclass (pair-aware batching + metrics) | 1.0 |
| `src/ludic/training/classification_metrics.py` | **NEW**: Preference metrics + calibration | 1.1 |
| `src/ludic/training/batching/preference.py` | **NEW**: Pair grouping + pair-aware micro-batching + preference collation | 1.0 |
| `examples/reward_model/train_rm_bradley_terry.py` | Switch to RMTrainer, add LoRA args, use Trainer config | 1.2 |
| `src/ludic/training/loss.py` | Add label_smoothing param to BradleyTerryLoss | 2.1 |
| `examples/reward_model/train_rm_isambard.slurm` | Update args for RMTrainer | 2.2 |
| `src/ludic/training/__init__.py` | Export RMTrainer + metrics | 1.x |
| `tests/test_preference_batching.py` | **NEW**: Unit tests for pair-aware batching + collate | 1.x |

---

## Validation Criteria

Before scaling up, we must validate:

1. **Preference accuracy**: Test accuracy > 60% (better than current 60% baseline)
2. **Margin analysis**: Median margin > 0, negative_margin_rate < 40%
3. **Calibration**: ECE < 0.15 (reasonably calibrated)
4. **Overfitting control**: Train-test accuracy gap < 15%
5. **GPU efficiency**: Peak memory tracked, utilization stable (via Trainer stats)
6. **LoRA parity**: LoRA achieves within 3% of full fine-tuning accuracy

---

## Unit Tests (New)

**Targets**:
- `tests/test_preference_batching.py`: pair grouping, pair-aware micro-batching, preference collation

**Run (focused)**:
```bash
uv run pytest tests/test_preference_batching.py
```

**Run (full test suite, non-integration)**:
```bash
uv run pytest -m "not integration"
```

---

## Parallelization Strategy

**Phase 1 tasks can be parallelized:**

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Independent tasks (can run in parallel)                │
├─────────────────────────────────────────────────────────────────┤
│ Task 1.0: RMTrainer core        │ Task 1.1: classification_metrics.py │
│ - pair-aware batching           │ - PreferenceMetrics                 │
│ - meta injection                │ - calibration + ROC/AUC             │
│ - loss scaling by pairs         │                                     │
├─────────────────────────────────┼─────────────────────────────────────┤
│ Task 1.2: LoRA args + wrapping  │ Task 1.3: Integrate metrics into    │
│ - Add --lora* arguments         │   evaluate()                         │
│ - PEFT get_peft_model()         │ - Return new metrics                 │
└─────────────────────────────────┴─────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Sequential (depends on Phase 1)                        │
├─────────────────────────────────────────────────────────────────┤
│ Task 2.1: LR scheduler + warmup                                 │
│ Task 2.2: Label smoothing in BradleyTerryLoss                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Validation (depends on all above)                      │
├─────────────────────────────────────────────────────────────────┤
│ Task 3.1: Full FT ablation      │ Task 3.2: LoRA ablation       │
│           (parallel jobs)       │           (parallel jobs)     │
├─────────────────────────────────┴───────────────────────────────┤
│ Task 3.3: Document findings, create GitHub issue                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Test Configuration (After Improvements)

```bash
uv run python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-0.5B \
    --hf-dataset stanfordnlp/SHP \
    --hf-split train \
    --limit 5000 \
    --steps 1000 \
    --batch-size 16 \
    --micro-token-budget 8192 \
    --max-seq-len 1024 \
    --lr 5e-6 \
    --weight-decay 0.01 \
    --warmup-steps 50 \
    --lr-scheduler cosine \
    --beta 1.0 \
    --regularization-lambda 0.001 \
    --label-smoothing 0.05 \
    --lora --lora-rank 8 \
    --eval-split test \
    --eval-limit 1000 \
    --eval-every 100 \
    --early-stopping-patience 5 \
    --profile-memory \
    --output-dir checkpoints_rm_v2
```

---

## Decisions Made

1. **Loss type**: Bradley-Terry only - focus on improving BT with label smoothing, regularization, better hparams first. Skip InfoNCE for now.

2. **Architecture**: Incremental approach - integrate RMTrainer with Trainer infrastructure (pair-aware, metrics, logging). **Action**: Raise GitHub issue for full Trainer integration plan if more infra changes are needed later.

3. **LoRA**: High priority - implement early in Phase 1-2 for efficient experimentation.

4. **GPU profiling**: Use Trainer’s built-in profiling; no new GPU monitor utility.

---

## Revised Implementation Order

### Phase 1: RMTrainer + Metrics + LoRA (High Priority)
1. Create `rm_trainer.py` subclass with pair-aware micro-batching and meta injection
2. Create `classification_metrics.py` with PreferenceMetrics + calibration
3. Add LoRA support with PEFT to RM trainer (head always trainable)
4. Integrate metrics into `evaluate()` in RMTrainer

### Phase 2: Training Infrastructure Enhancements
5. Add LR scheduler and warmup
6. Add label smoothing to BradleyTerryLoss

### Phase 3: Validation & Tuning
7. Run ablation study on hyperparameters (full FT vs LoRA)
8. Document optimal settings for SHP dataset
9. Create GitHub issue for full Trainer integration

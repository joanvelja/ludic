# Handoff: RM Trainer Quality & Performance Fixes

> **Last updated**: 2026-01-21
> **Branch**: `RM-vllm`
> **Status**: ✅ **IMPLEMENTATION COMPLETE** — Ready for PR review

---

## Summary

All issues from the comprehensive bottleneck analysis (13 agents) + peer review have been implemented. This PR delivers **correctness fixes**, **performance improvements**, **architectural consolidation**, and **test coverage**.

| Phase | Status | Issues |
|-------|--------|--------|
| **Phase 0** | ✅ Complete | 4/4 Critical bugs |
| **Phase 1** | ✅ Complete | 4/4 High priority |
| **Phase 2** | ✅ Complete | 4/5 Medium + [Issue #7](https://github.com/joanvelja/ludic/issues/7) |
| **Phase 3** | ✅ Complete | 2/2 Lower priority |
| **Phase 4** | ✅ Complete | 4/4 Architectural |

---

## PR Review Guide

### Files Changed (by category)

#### 🔴 Critical Bug Fixes (Phase 0)

| File | Change | Review Focus |
|------|--------|--------------|
| `src/ludic/training/loss.py:1360-1364` | **P0-1**: Fixed regularization sign error | Verify `per_pair_loss.mean() + reg` (was incorrectly subtracting) |
| `src/ludic/training/reward_scorer.py` | **P0-2**: Added `use_chat_template` param | Check chat template formatting logic |
| `examples/reward_model/README.md:39-40` | **P0-3**: Fixed CLI docs (`--hf-dataset`) | Verify matches actual CLI args |
| `src/ludic/training/reward_credit.py:159-163` | **P0-4**: Changed silent 0.0 fallback → `ValueError` | Confirm fail-fast behavior is desired |

#### 🟠 High Priority Fixes (Phase 1)

| File | Change | Review Focus |
|------|--------|--------------|
| `src/ludic/training/classification_metrics.py:80-83` | **P1-1**: Added `.cpu()` calls, removed unused lists | Verify GPU memory fix |
| `src/ludic/training/rm_trainer.py:208-212` | **P1-2**: Use `dataclasses.replace()` for LoRA config | Check all attrs preserved |
| `src/ludic/training/rm_trainer.py:41-43,201,259` | **P1-3**: Unified `REWARD_HEAD_PATTERNS` constant | Verify both methods use it |
| `src/ludic/training/preference_utils.py:170-193` | **P1-4**: Clearer tokenization order | Check boundary logic intact |

#### 🟡 Medium Priority Fixes (Phase 2)

| File | Change | Review Focus |
|------|--------|--------------|
| `examples/reward_model/train_rm_bradley_terry.py:307-312` | **P2-1**: Documented why async kept (protocol) | Docstring clarity |
| `src/ludic/training/reward_scorer.py:153-161` | **P2-2**: Parallel scoring via `asyncio.gather` | Concurrency correctness |
| `src/ludic/training/rm_trainer.py:313-325` | **P2-4**: Added 60s batch fetch timeout | Error handling |
| `src/ludic/training/loss.py:1304-1367` | **P2-5**: Fast path for interleaved batches | `_is_interleaved()` logic |

> **P2-3** (train_step refactor): Deferred to [Issue #7](https://github.com/joanvelja/ludic/issues/7)

#### ⚪ Improvements (Phase 3)

| File | Change | Review Focus |
|------|--------|--------------|
| `tests/test_rm_trainer.py` (NEW) | **P3-1**: 25 new unit tests | Coverage of RM-specific logic |
| `tests/conftest.py` | Graceful vLLM mocking | Tests run without GPU |
| `src/ludic/training/loss.py:1199-1232,1279,1421-1429` | **P3-2**: `torch.compile` support | `compile` param, core function extraction |

#### 🔵 Architectural (Phase 4)

| File | Change | Review Focus |
|------|--------|--------------|
| `src/ludic/training/chat_template_utils.py` (NEW) | **A-1**: Shared helper | `ChatTemplateResult`, `tokenize_with_chat_template()` |
| `src/ludic/training/__init__.py` | Export new utilities | Public API |
| `src/ludic/training/reward_scorer.py` | **A-2**: Added `render_fn` param | Formatting priority dispatch |
| `src/ludic/training/preference_utils.py` | **A-3**: Truncation metadata in SAWItem | `meta["truncation"]` schema |
| `examples/reward_model/cold_start_rlhf.py:162-198` | **A-4**: Warning docstrings | Single-turn assumption documented |

---

## Quick Diff Commands

```bash
# All changes
git diff master...HEAD

# Phase 0 (Critical)
git diff master...HEAD -- src/ludic/training/loss.py src/ludic/training/reward_scorer.py \
    src/ludic/training/reward_credit.py examples/reward_model/README.md

# Phase 1 (High)
git diff master...HEAD -- src/ludic/training/classification_metrics.py \
    src/ludic/training/rm_trainer.py src/ludic/training/preference_utils.py

# Phase 2 (Medium)
git diff master...HEAD -- examples/reward_model/train_rm_bradley_terry.py

# Phase 3 (Tests + torch.compile)
git diff master...HEAD -- tests/

# Phase 4 (Architecture)
git diff master...HEAD -- src/ludic/training/chat_template_utils.py \
    src/ludic/training/__init__.py examples/reward_model/cold_start_rlhf.py
```

---

## Test Commands

```bash
# Syntax validation (no GPU needed)
uv run python -m py_compile src/ludic/training/loss.py
uv run python -m py_compile src/ludic/training/reward_scorer.py
uv run python -m py_compile src/ludic/training/rm_trainer.py
uv run python -m py_compile src/ludic/training/preference_utils.py
uv run python -m py_compile src/ludic/training/chat_template_utils.py

# Unit tests (requires vLLM mocking in conftest.py)
uv run pytest tests/test_rm_trainer.py -v
uv run pytest tests/test_reward_scorer.py -v
uv run pytest tests/test_reward_credit.py -v
uv run pytest tests/test_loss.py -v

# Full test suite (requires vLLM + GPU)
VLLM_RM_MODEL=Qwen/Qwen2.5-0.5B uv run pytest tests/ -v --ignore=tests/integration/

# Smoke test
uv run python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-0.5B \
    --hf-dataset stanfordnlp/SHP \
    --hf-split train \
    --steps 10 \
    --limit 100
```

---

## Key Changes Summary

### Correctness
- **Regularization sign error** fixed (was subtracting instead of adding)
- **Silent 0.0 fallback** now raises `ValueError` (fail-fast)
- **Format mismatch** addressed via `use_chat_template` + `render_fn`
- **CLI docs** now match actual arguments

### Performance
- **Parallel scoring** via `asyncio.gather`
- **torch.compile** support for BradleyTerryLoss core
- **Fast path** for interleaved batch indexing
- **GPU memory** properly moved to CPU during eval

### Robustness
- **Batch fetch timeout** (60s) prevents hangs
- **LoRA config** preserves all user attributes
- **Head detection** unified via `REWARD_HEAD_PATTERNS`
- **Truncation metadata** tracked in SAWItem

### Architecture
- **Shared chat template helper** eliminates 3 duplicate implementations
- **Flexible formatting** via `render_fn` callback
- **Single-turn assumption** documented with warnings

---

## Files Created

| File | Purpose |
|------|---------|
| `src/ludic/training/chat_template_utils.py` | Shared tokenization utilities |
| `tests/test_rm_trainer.py` | 25 unit tests for RMTrainer |

## GitHub Issues Created

| Issue | Description |
|-------|-------------|
| [#7](https://github.com/joanvelja/ludic/issues/7) | Refactor: RMTrainer.train_step() duplicates parent class |

---

## Reviewer Checklist

- [ ] **P0-1**: Regularization math is correct (`per_pair_loss.mean() + reg`)
- [ ] **P0-4**: `ValueError` on missing rm_score is acceptable (breaking change)
- [ ] **P1-1**: GPU tensors moved to CPU before accumulation
- [ ] **P2-2**: Parallel scoring doesn't break client assumptions
- [ ] **P2-4**: 60s timeout is reasonable default
- [ ] **P3-1**: New tests cover key RM-specific behaviors
- [ ] **P3-2**: `torch.compile` with `dynamic=True` is correct approach
- [ ] **A-1**: Shared helper API is clean and usable
- [ ] **A-3**: Truncation metadata schema is consistent

---

## Original Analysis Context

This work originated from a comprehensive bottleneck analysis using 13 parallel agents examining memory, algorithms, I/O, architecture, PyTorch patterns, code cleanliness, training loop, data pipeline, LoRA, micro-batching, loss functions, tests, and research context.

Peer review identified 3 critical functional issues missed by initial analysis:
1. CLI argument mismatch (`--dataset` vs `--hf-dataset`)
2. Format mismatch between library scorer and example scorer
3. Silent 0.0 fallback with only warning

Full analysis artifacts in `.bottleneck-analysis/`.

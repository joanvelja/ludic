# Training API Migration Guide

This guide documents the training API changes introduced in recent Ludic updates, helping you migrate existing training code to the new APIs.

## Overview

The training API has been significantly improved with:

1. **Token-Budget Micro-Batching**: Dynamic batching based on token budgets instead of fixed gradient accumulation steps
2. **New Loss Functions**: CISPO, GSPO, and improved GRPO with importance sampling correction
3. **Algorithm Presets**: Convenient factory functions for common RL algorithms
4. **Stats Aggregation**: Declarative `Reducer` system with on-device aggregation
5. **Logging Enhancements**: Multi-output logging with sparkline visualization

These changes improve training efficiency, reduce memory waste from padding, and provide better gradient signal for rare but important tokens.

---

## What Changed

### 1. Token-Budget Micro-Batching

#### Old API
```python
cfg = TrainerConfig(
    grad_accum_steps=4,  # Fixed number of micro-batches
    # ...
)
```

#### New API
```python
cfg = TrainerConfig(
    max_seq_len=2048,        # Maximum sequence length (truncates longer)
    micro_token_budget=16384, # Target padded tokens per micro-batch
    # ...
)
```

#### Benefits

| Aspect | Old (Fixed Steps) | New (Token Budget) |
|--------|-------------------|-------------------|
| Padding waste | High (varied sequence lengths) | Low (sorted by length) |
| Memory usage | Unpredictable | Predictable (`max_len * batch_size <= budget`) |
| Batch sizes | Fixed | Adaptive to sequence lengths |
| Long sequences | May OOM | Automatically truncated |

#### How It Works

The token-budget system operates in three phases:

1. **Truncation**: Sequences exceeding `max_seq_len` are truncated (with metadata preserved)
2. **Sorting**: Sequences are sorted by length (shortest first)
3. **Packing**: Sequences are packed into micro-batches where `max_length_in_batch * batch_size <= micro_token_budget`

```
Example: micro_token_budget=1024, sequences of lengths [100, 150, 200, 300, 400]

Sorted: [100, 150, 200, 300, 400]

Micro-batch 1: [100, 150, 200] -> max=200, count=3, padded=600 tokens (OK)
Micro-batch 2: [300, 400]      -> max=400, count=2, padded=800 tokens (OK)
```

This reduces padding waste significantly compared to random batching:
- Random batch [100, 400]: 800 tokens (400 wasted on padding)
- Sorted batch [100, 150, 200]: 600 tokens (minimal waste)

#### Configuration Guidelines

| Scenario | max_seq_len | micro_token_budget | Notes |
|----------|-------------|-------------------|-------|
| Small model (1-3B) | 2048 | 16384 | ~8 sequences per micro-batch |
| Medium model (7-13B) | 4096 | 32768 | ~8 sequences per micro-batch |
| Large model (30B+) | 4096 | 16384 | Reduce budget for memory |
| Code generation | 4096 | 24576 | Longer completions typical |

**Rule of thumb**: `micro_token_budget >= max_seq_len` (enforced by validation).

---

### 2. New Loss Functions

Four key loss functions are now available, each with different characteristics:

#### CISPO (CISPOLoss)

**Clipped IS-weight Policy Optimization** - clips the importance sampling weight instead of the policy update.

```python
from ludic.training import CISPOLoss

loss = CISPOLoss(
    clip_eps_low=1e6,      # Effectively no lower bound
    clip_eps_high=0.2,     # Upper bound on IS weight
    length_normalize=True, # Normalize by action length
)
```

**Key insight**: Unlike PPO/GRPO which clip via `min(r*A, clip(r)*A)`, CISPO uses:
```
L = -E[ sg(clip(r, 1-eps_low, 1+eps_high)) * A * log pi(a|s) ]
```

The stop-gradient (`sg`) on the clipped ratio means gradients flow through `log pi` for ALL tokens, not just those within the clip range.

**Why this matters for code generation**: See [Why CISPO for Code Generation](#why-cispo-for-code-generation) below.

#### GSPO (ClippedSurrogateLoss)

**Sequence-level PPO** with tight asymmetric clipping (from the GSPO paper).

```python
from ludic.training import ClippedSurrogateLoss

loss = ClippedSurrogateLoss(
    clip_eps_low=3e-4,      # Very tight lower bound
    clip_eps_high=4e-4,     # Very tight upper bound
    length_normalize=True,  # Use geometric mean ratio
    ratio_clip=None,        # Optional hard ratio cap
)
```

Uses sequence-level log-probability ratios with the tightest clipping range of any method.

#### Token GRPO (TokenClippedSurrogateLoss)

**Token-level PPO clipping** - the standard GRPO formulation.

```python
from ludic.training import TokenClippedSurrogateLoss

loss = TokenClippedSurrogateLoss(
    clip_eps_low=0.2,       # Standard PPO lower bound
    clip_eps_high=0.27,     # Slightly asymmetric upper bound
    length_normalize=False, # Sum over tokens (not mean)
    ratio_clip=None,        # Optional hard ratio cap
)
```

Computes per-token importance ratios and applies PPO clipping at the token level.

#### REINFORCE (ReinforceLoss)

**Now includes importance sampling correction** to account for inference/training kernel differences.

```python
from ludic.training import ReinforceLoss

loss = ReinforceLoss(
    length_normalize=False,
    old_logp_key="old_logp_action",  # Key for behavior policy logprobs
)
```

The IS correction is:
```
L = -E[ sg(r) * A * log pi(a|s) ]
```
where `r = pi_new(a|s) / pi_old(a|s)`.

#### Loss Function Comparison

| Loss | Ratio Level | Clipping | Gradient Flow | Best For |
|------|-------------|----------|---------------|----------|
| CISPO | Token | IS weight | All tokens | Reasoning, self-correction |
| GSPO | Sequence | Update | High-prob tokens | Stable training |
| Token GRPO | Token | Update | High-prob tokens | Standard RL |
| REINFORCE | Sequence | None (IS only) | All tokens | Simple baselines |

---

### 3. Algorithm Presets

Factory functions create complete `RLAlgorithm` objects (credit assigner + loss + preprocessing):

```python
from ludic.training import (
    make_cispo,
    make_gspo,
    make_grpo,
    make_reinforce,
    make_reinforce_baseline,
    make_sft,
)

# CISPO - best for reasoning/code with self-correction
algo = make_cispo(
    group_size=8,
    group_normalize_adv=True,
    positive_only=False,
    clip_eps_low=1e6,        # No lower bound (paper default)
    clip_eps_high=0.2,
    length_normalize=True,
    drop_zero_weight=False,
    name="cispo",
)

# GSPO - sequence-level with tight clipping
algo = make_gspo(
    group_size=8,
    group_normalize_adv=True,
    positive_only=False,
    clip_eps_low=3e-4,
    clip_eps_high=4e-4,
    length_normalize=True,
    ratio_clip=None,
    drop_zero_weight=False,
    name="gspo",
)

# GRPO - token-level with standard PPO clipping
algo = make_grpo(
    group_size=8,
    group_normalize_adv=True,
    positive_only=False,
    clip_eps_low=0.2,
    clip_eps_high=0.27,
    length_normalize=False,
    ratio_clip=None,
    drop_zero_weight=False,
    name="grpo",
)

# REINFORCE - simple policy gradient
algo = make_reinforce(
    gamma=1.0,
    drop_zero_weight=False,
    name="reinforce",
)

# REINFORCE with baseline - variance reduction
algo = make_reinforce_baseline(
    gamma=1.0,
    normalize_adv=False,
    drop_zero_weight=False,
    name="reinforce_baseline",
)

# SFT - supervised fine-tuning (offline)
algo = make_sft(
    length_normalize=False,
    name="sft",
)
```

#### Preset Internals

Each preset bundles:
- **Credit Assigner**: How rewards become per-step weights
- **Loss**: How weights + tokens become gradients
- **Preprocessing**: Validation and filtering before collation

| Preset | Credit Assigner | Loss | Preprocessing |
|--------|-----------------|------|---------------|
| `make_cispo` | `GroupNormalizedReturn` | `CISPOLoss` | `validate_actor_logps` |
| `make_gspo` | `GroupNormalizedReturn` | `ClippedSurrogateLoss` | `validate_actor_logps` |
| `make_grpo` | `GroupNormalizedReturn` | `TokenClippedSurrogateLoss` | `validate_actor_logps` |
| `make_reinforce` | `MonteCarloReturn` | `ReinforceLoss` | Optional zero-weight drop |
| `make_sft` | `ConstantCredit(1.0)` | `MaskedCausalLMCrossEntropyLoss` | None |

---

### 4. Stats Aggregation

#### Declarative Reducer System

Stats are now aggregated using declarative `Reducer` objects:

```python
from ludic.training import Reducer

reducers = {
    # Count successes, normalize by rollouts
    "all_passed_rate": Reducer(
        kind="count_true",
        source="all_passed",        # Path into SAWItem.meta
        normalize_by="rollouts",
        as_percent=True,
    ),

    # Average a numeric field
    "avg_pass_rate": Reducer(
        kind="mean",
        source="pass_rate",
    ),

    # Sum a field
    "total_tokens": Reducer(
        kind="sum",
        source="completion_length",
    ),

    # Custom transform
    "win_rate": Reducer(
        kind="count_true",
        source="result",
        transform=lambda v: v == "win",
        normalize_by="rollouts",
    ),

    # Callable source (access nested data)
    "custom_metric": Reducer(
        kind="mean",
        source=lambda item: item.meta.get("scores", {}).get("quality", 0),
    ),
}
```

#### Reducer Kinds

| Kind | Description | Example Use |
|------|-------------|-------------|
| `"mean"` | Average of values | Pass rates, scores |
| `"sum"` | Total of values | Token counts |
| `"count_true"` | Count truthy values | Success rates |

#### Normalization Options

| `normalize_by` | Denominator | Use Case |
|----------------|-------------|----------|
| `None` | Raw value | Sums, counts |
| `"samples"` | Total SAWItems | Per-sample rates |
| `"rollouts"` | Total rollouts | Per-episode rates |

#### Default Reducers

```python
from ludic.training import default_reducers

reducers = default_reducers()
# Provides:
#   - completion_truncated_rate: finish_reason == "length"
#   - seq_len_truncated_rate: items truncated by max_seq_len
```

#### On-Device Aggregation

Stats are aggregated on-device before a single batched D2H transfer:

```python
# Internal flow:
# 1. Each micro-batch produces stats as Tensors (on GPU)
# 2. aggregate_stats() accumulates on-device
# 3. Single torch.stack() + .cpu() call
# 4. Convert to Python floats
```

This avoids N synchronization points for N stats.

---

### 5. Metadata Changes

#### Renamed Fields

| Old Name | New Name | Description |
|----------|----------|-------------|
| `num_rollouts` | `target_rollouts` | Requested number of rollouts |

#### New Fields

| Field | Description |
|-------|-------------|
| `effective_rollouts` | Rollouts surviving preprocessing (filters, zero-weight drops) |
| `seq_len_truncated` | Boolean: was this item truncated by `max_seq_len`? |
| `seq_len_original` | Original sequence length before truncation |
| `seq_len_retained` | Sequence length after truncation |
| `seq_len_retained_frac` | Fraction of sequence retained |

#### Usage in Stats

```python
# These are automatically computed by aggregate_stats():
stats = {
    "num_samples": 128,           # Total SAWItems
    "target_rollouts": 32,        # Requested rollouts
    "effective_rollouts": 28,     # Rollouts with surviving items
    "avg_total_reward": 0.75,
    "avg_completion_length": 512,
}
```

---

### 6. Logging Enhancements

#### Stat Prefix Grouping

Stats are automatically grouped by prefix:

| Prefix | Category | Example Keys |
|--------|----------|--------------|
| `train/` | Training metrics | `train/loss`, `train/avg_total_reward` |
| `eval/` | Evaluation metrics | `eval/all_passed_rate`, `eval/avg_pass_rate` |
| `perf/` | Performance counters | `perf/gpu_mem_peak_mb`, `perf/gpu_activation_peak_mb` |

Ungrouped keys are auto-prefixed:
- `loss` -> `train/loss`
- `eval_accuracy` -> `eval/accuracy`
- `gpu_mem_alloc_mb` -> `perf/gpu_mem_alloc_mb`

#### TeeLogger

Fan-out to multiple logging backends:

```python
from ludic.training import TeeLogger, PrintLogger, RichLiveLogger
from ludic.training.loggers import WandbLogger

logger = TeeLogger(
    PrintLogger(prefix="[train]"),
    WandbLogger(project="my-project"),
)
```

#### RichLiveLogger

Terminal dashboard with live-updating sparklines:

```python
from ludic.training import RichLiveLogger

logger = RichLiveLogger(
    keys=["loss", "avg_total_reward", "all_passed_rate"],
    spark_key="avg_total_reward",  # Metric for sparkline
    history=100,                   # Sparkline history length
    spark_window=50,               # Visible sparkline width
    precision=4,                   # Decimal places
)

# Use as context manager for clean terminal handling
with logger:
    for step in range(num_steps):
        stats = trainer.train_step_sync()
        # Logger auto-updates via trainer.train_logger
```

The dashboard shows three columns (train/eval/perf) plus a sparkline panel.

#### WandbLogger

Weights & Biases integration:

```python
from ludic.training.loggers import WandbLogger

logger = WandbLogger(
    project="ludic-training",
    name="experiment-1",
    group="code-gen",
    tags=["cispo", "apps"],
    config={"model": "Qwen/Qwen2.5-3B", "group_size": 8},
)

# Or with existing run
import wandb
run = wandb.init(...)
logger = WandbLogger(init_kwargs={"reinit": False})
```

---

## Migration Checklist

Use this checklist when migrating existing training code:

### Configuration

- [ ] Replace `grad_accum_steps=N` with `max_seq_len=X, micro_token_budget=Y`
- [ ] Choose `micro_token_budget` based on GPU memory (start with `max_seq_len * 8`)
- [ ] Set `max_seq_len` to your model's context window or desired max

### Algorithm

- [ ] Replace manual `RLAlgorithm` construction with preset (`make_grpo`, `make_cispo`, etc.)
- [ ] Or update custom loss to new signature if using custom losses
- [ ] Ensure inference returns `actor_logps` for ratio-based objectives

### Stats

- [ ] Update reducers to use new `Reducer` dataclass
- [ ] Replace string-based aggregation with `kind`/`source`/`transform`
- [ ] Update any code reading `num_rollouts` to use `target_rollouts`

### Logging

- [ ] Update logger initialization for new API
- [ ] Add prefix handling if parsing stats manually
- [ ] Consider `RichLiveLogger` for local development

### Validation

- [ ] Run training for a few steps to verify batching
- [ ] Check that `micro_token_budget` doesn't cause OOM
- [ ] Verify stats are being aggregated correctly

---

## Code Examples

### Before (Old API)

```python
from ludic.training import (
    Trainer,
    TrainerConfig,
    RLAlgorithm,
    GroupNormalizedReturn,
    ClippedSurrogateLoss,
)

# Manual algorithm construction
credit_assigner = GroupNormalizedReturn(
    group_size=8,
    normalize_adv=True,
)
loss = ClippedSurrogateLoss(clip_eps=0.2)  # Old single-eps API

algo = RLAlgorithm(
    name="grpo",
    credit_assigner=credit_assigner,
    loss=loss,
)

# Old config with fixed accumulation steps
cfg = TrainerConfig(
    model_device="cuda",
    lr=1e-5,
    grad_accum_steps=4,  # OLD: Fixed micro-batch count
    max_grad_norm=1.0,
    pad_token_id=tokenizer.pad_token_id,
)

trainer = Trainer(
    model=model,
    algo=algo,
    batch_source=batch_source,
    cfg=cfg,
)
```

### After (New API)

```python
from ludic.training import (
    Trainer,
    TrainerConfig,
    make_grpo,      # Use preset
    make_cispo,     # Or CISPO for code generation
    Reducer,
    RichLiveLogger,
)

# Use algorithm preset (handles credit + loss + preprocessing)
algo = make_grpo(
    group_size=8,
    group_normalize_adv=True,
    clip_eps_low=0.2,
    clip_eps_high=0.27,
    length_normalize=False,
)

# Or for code generation, use CISPO
algo = make_cispo(
    group_size=8,
    clip_eps_high=0.2,
    length_normalize=True,
)

# New config with token-budget batching
cfg = TrainerConfig(
    model_device="cuda",
    lr=1e-5,
    max_seq_len=4096,           # NEW: Max sequence length
    micro_token_budget=32768,   # NEW: Token budget per micro-batch
    max_grad_norm=1.0,
    pad_token_id=tokenizer.pad_token_id,
    sync_every_steps=1,
    log_every=1,
)

# Declarative reducers
reducers = {
    "all_passed_rate": Reducer(
        kind="count_true",
        source="all_passed",
        normalize_by="rollouts",
    ),
    "avg_pass_rate": Reducer(
        kind="mean",
        source="pass_rate",
    ),
}

# Rich terminal logger
logger = RichLiveLogger(
    keys=["loss", "avg_total_reward", "all_passed_rate"],
    spark_key="avg_total_reward",
)

trainer = Trainer(
    model=model,
    algo=algo,
    batch_source=batch_source,
    cfg=cfg,
    train_logger=logger,
    reducers=reducers,
)
```

### Complete CISPO Training Example

```python
"""Complete example: CISPO training for code generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ludic.training import (
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    RolloutEngine,
    RolloutBatchSource,
    make_cispo,
    Reducer,
    RichLiveLogger,
    default_reducers,
)
from ludic.distributed.adapters import create_vllm_publisher
from ludic.inference import VLLMChatClient

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

# CISPO algorithm - preserves gradient for rare self-correction tokens
algo = make_cispo(
    group_size=8,
    group_normalize_adv=True,
    clip_eps_high=0.2,
    length_normalize=True,
)

# Token-budget configuration
cfg = TrainerConfig(
    model_device="cuda",
    lr=1e-5,
    weight_decay=0.01,
    max_seq_len=4096,
    micro_token_budget=32768,  # ~8 sequences of 4096 tokens
    max_grad_norm=1.0,
    pad_token_id=tokenizer.pad_token_id,
    sync_every_steps=1,
    eval_every_n_steps=25,
    eval_concurrency=32,
)

# Combine default + custom reducers
reducers = {
    **default_reducers(),
    "all_passed_rate": Reducer(
        kind="count_true",
        source="all_passed",
        normalize_by="rollouts",
        as_percent=True,
    ),
    "compile_fail_rate": Reducer(
        kind="count_true",
        source="compile_failed",
        normalize_by="rollouts",
        as_percent=True,
    ),
    "avg_pass_rate": Reducer(
        kind="mean",
        source="pass_rate",
    ),
}

# Setup logging
logger = RichLiveLogger(
    keys=[
        "loss",
        "avg_total_reward",
        "all_passed_rate",
        "compile_fail_rate",
        "ratio_mean",
        "clip_frac",
    ],
    spark_key="avg_total_reward",
    history=100,
)

# Setup inference client and publisher
client = VLLMChatClient(host="127.0.0.1", port=8000, enable_weight_updates=True)
publisher = create_vllm_publisher(client)

# Create trainer
trainer = Trainer(
    model=model,
    algo=algo,
    batch_source=batch_source,  # Your RolloutBatchSource
    publisher=publisher,
    cfg=cfg,
    train_logger=logger,
    reducers=reducers,
    checkpoint_config=CheckpointConfig(
        output_dir="checkpoints",
        every_n_steps=50,
        max_to_keep=3,
    ),
)

# Train
with logger:
    trainer.train_sync(num_steps=100)
```

---

## Why CISPO for Code Generation

Code generation often involves **self-correction patterns** where the model recognizes and fixes mistakes mid-generation. These patterns use tokens like:

- "Wait, that won't work because..."
- "Let me reconsider..."
- "Actually, I should..."
- "However, this approach..."
- "On second thought..."

### The Problem with PPO/GRPO

In PPO and GRPO, the clipping operation `min(r*A, clip(r)*A)` affects which tokens contribute gradients:

```
Token: "However"
- Under behavior policy: p_old = 0.001 (rare)
- Under current policy: p_new = 0.005 (5x increase)
- Ratio: r = 5.0 (far outside clip range [0.8, 1.27])
- Clipped: clip(r) = 1.27

Gradient contribution:
  Before clipping: 5.0 * A * grad(log p)
  After clipping:  1.27 * A * grad(log p)  <- Severely reduced!
```

After the first gradient update, rare tokens that increased in probability get their gradients clipped, **preventing further learning on these crucial tokens**.

### CISPO's Solution

CISPO clips the IS weight but applies it with stop-gradient:

```
L = -E[ sg(clip(r)) * A * log pi ]
```

The gradient is:
```
dL/d(theta) = -sg(clip(r)) * A * d(log pi)/d(theta)
```

Since `sg(clip(r))` has no gradient, the full `d(log pi)/d(theta)` flows through for ALL tokens, including rare ones. The clipped IS weight only affects the loss magnitude, not which tokens get gradients.

### Empirical Evidence

From the MiniMax-M1 paper (arXiv:2506.13585):

> "Tokens like 'However', 'Recheck', 'Wait' are rare but crucial for learning reflective CoT behaviors. PPO/GRPO clip these out after the first update, preventing them from contributing to subsequent off-policy gradient updates."

For code generation specifically:
- Self-correction tokens often have <1% probability under the initial policy
- These tokens are critical for catching bugs before submission
- CISPO preserves gradient signal, allowing the model to learn when to self-correct

### When to Use Each Algorithm

| Scenario | Recommended Algorithm | Reason |
|----------|----------------------|--------|
| Code generation | CISPO | Self-correction matters |
| Mathematical reasoning | CISPO | "Wait, let me verify" patterns |
| Simple QA | GRPO | Standard token distribution |
| Stable fine-tuning | GSPO | Tightest clipping for stability |
| Debugging/baseline | REINFORCE | Simplest gradient signal |

---

## References

- CISPO: MiniMax-M1 paper (arXiv:2506.13585)
- GSPO: arXiv:2507.18071
- GRPO: DeepSeek-R1 paper
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms"

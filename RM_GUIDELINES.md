# Reward Model Training Guidelines

> **Last updated**: 2026-01-22 (V5 experiments complete)
> **Dataset**: Stanford Human Preferences (SHP)
> **Model**: Qwen2.5-3B with LoRA

This document summarizes findings from 40+ experiments across V3-V5 and provides actionable guidelines for Bradley-Terry reward model training.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Hyperparameter Reference](#hyperparameter-reference)
3. [LoRA Configuration](#lora-configuration)
4. [Regularization Techniques](#regularization-techniques)
5. [Experiment Summary](#experiment-summary)
6. [Diagnostic Tests](#diagnostic-tests)
7. [What Works / What Doesn't](#what-works--what-doesnt)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Best configuration (V5-S1c, 75.55% accuracy):**

```bash
uv run python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-3B \
    --hf-dataset stanfordnlp/SHP \
    --hf-split train \
    --eval-split test \
    --eval-limit 1000 \
    --limit 5000 \
    --batch-size 16 \
    --micro-token-budget 8192 \
    --max-seq-len 1024 \
    --lora --lora-rank 4 --lora-alpha 32 \
    --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --steps 1000 \
    --lr-scheduler cosine \
    --warmup-steps 50 \
    --label-smoothing 0.05 \
    --score-regularization-lambda 0.1 \
    --eval-every 100 \
    --track-gpu \
    --output-dir $OUTPUT_DIR
```

---

## Hyperparameter Reference

### Data & Batching

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--limit` | 5000 | Number of preference pairs to train on. More data helps but with diminishing returns. |
| `--batch-size` | 16 | Pairs per gradient step. Larger batches (32, 64) caused overfitting in our tests. |
| `--micro-token-budget` | 8192 | Max tokens per micro-batch for gradient accumulation. Prevents OOM on long sequences. |
| `--max-seq-len` | 1024 | Maximum sequence length. **Critical**: SHP has long responses; 512 truncates too much. |

### Optimization

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--lr` | 5e-5 | Learning rate. LoRA typically needs 10x higher LR than full fine-tuning. Range: 2e-5 to 1e-4. |
| `--weight-decay` | 0.01 | L2 regularization on weights. Helps prevent overfitting. |
| `--lr-scheduler` | cosine | Learning rate decay schedule. Cosine annealing works well for RM training. |
| `--warmup-steps` | 50 | Linear warmup steps. ~5% of total steps is typical. |
| `--steps` | 1000 | Total training steps. With 5K pairs, this is ~3 epochs. |

### Regularization

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--label-smoothing` | 0.05 | Softens targets from 0/1 to 0.05/0.95. Improves calibration. |
| `--score-regularization-lambda` | 0.1 | **Key finding**: L2 penalty on reward scores prevents unbounded growth. |
| `--regularization-lambda` | 0.0 | L2 penalty on margin (chosen - rejected). Less effective than score reg. |
| `--beta` | 1.0 | Bradley-Terry temperature. Higher = sharper sigmoid. Keep at 1.0. |

### Evaluation & Checkpointing

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--eval-every` | 100 | Evaluation frequency. Every 100 steps catches overfitting early. |
| `--eval-limit` | 1000 | Number of eval pairs. 1000 gives stable accuracy estimates. |
| `--eval-split` | test | Which split to evaluate on. Use `test` for final comparison. |
| `--early-stopping-patience` | 5 | Stop if no improvement for N evals. Useful with high LR. |

---

## LoRA Configuration

### What is LoRA?

**Low-Rank Adaptation (LoRA)** freezes the pretrained model and injects trainable low-rank matrices into specific layers. This dramatically reduces trainable parameters while maintaining performance.

For a weight matrix W, LoRA adds: `W' = W + (α/r) * BA`
- `B` is (d × r), `A` is (r × d)
- `r` = rank (typically 4-16)
- `α` = scaling factor

### LoRA Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--lora` | (flag) | Enable LoRA training instead of full fine-tuning. |
| `--lora-rank` | 4 | Rank of low-rank matrices. Higher = more parameters, more capacity. |
| `--lora-alpha` | 32 | Scaling factor. The effective scaling is α/r. With r=4, α=32 gives 8x scaling. |
| `--lora-target-modules` | all-linear | Which layers to adapt. **Critical choice** - see below. |
| `--rslora` | (optional) | Rank-stabilized LoRA. Scales by α/√r instead of α/r for consistent gradients. |

### Target Modules: All-Linear vs Attention-Only

**All-linear (RECOMMENDED):**
```bash
--lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
```

**Attention-only:**
```bash
--lora-target-modules "q_proj,k_proj,v_proj,o_proj"
```

| Configuration | Modules | Accuracy | Notes |
|---------------|---------|----------|-------|
| **All-linear** | 7 | **74%+** | Adapts attention AND MLP layers |
| Attention-only | 4 | ~70% | Only adapts attention; MLP frozen |

**Why all-linear wins**: Preference learning requires the model to adjust both how it attends to content AND how it processes/transforms that content. Freezing MLP layers limits the model's ability to learn preference-relevant features.

### Rank Selection

| Rank | Parameters | Use Case |
|------|------------|----------|
| 2 | ~1M | Minimal adaptation, may underfit |
| **4** | **~2M** | **Sweet spot for preference learning** |
| 8 | ~4M | More capacity, risk of overfitting |
| 16 | ~8M | For complex tasks or larger models |

**Finding**: Preference learning is a relatively "low-information" signal - the model mainly needs to learn which features correlate with human preference, not fundamentally new capabilities. Rank 4 is sufficient.

### rsLoRA (Rank-Stabilized LoRA)

Standard LoRA scales outputs by `α/r`. As rank increases, effective learning rate decreases.

**rsLoRA** scales by `α/√r` instead, making gradient magnitudes more consistent across ranks.

```bash
--lora --lora-rank 4 --lora-alpha 32 --rslora
```

**V4 finding**: rsLoRA showed slightly better stability but similar final accuracy to standard LoRA. Use if experimenting with different ranks.

---

## Regularization Techniques

### Score Regularization (RECOMMENDED)

**What it does**: Adds L2 penalty on individual reward scores (not just the margin).

```
Loss = BCE(sigmoid(r_chosen - r_rejected)) + λ * (r_chosen² + r_rejected²)
```

**Why it helps**: Without regularization, reward scores can grow unboundedly while maintaining the same margin. This hurts:
- **Calibration**: Large scores → confident but poorly calibrated predictions
- **Generalization**: Extreme scores memorize training pairs

| Lambda | Best Acc | Score Range | Recommendation |
|--------|----------|-------------|----------------|
| 0 | 73.35% | [-8, +8] | Baseline |
| 0.001 | 72.34% | [-5, +5] | Minimal effect |
| 0.01 | 71.84% | [-4, +4] | Moderate |
| **0.1** | **75.55%** | **[-3, +3]** | **Optimal** |

### Margin Regularization

**What it does**: Penalizes large gaps between chosen and rejected scores.

```
Loss = BCE(...) + λ * (r_chosen - r_rejected)²
```

| Lambda | Best Acc | Margin |
|--------|----------|--------|
| 0.001 | 73.05% | 1.08 |
| 0.01 | 73.35% | 0.80 |

**Finding**: Less effective than score regularization. The margin can be large if both scores are reasonable; score reg is more principled.

### Label Smoothing

**What it does**: Softens binary targets. Instead of P(chosen) = 1.0, uses P(chosen) = 1 - ε.

```bash
--label-smoothing 0.05  # targets become 0.05 and 0.95
```

**Why it helps**: Prevents overconfident predictions, improves calibration.

### Beta (Temperature)

**What it does**: Scales the logit before sigmoid: `σ(β * (r_c - r_r))`

| Beta | Effect |
|------|--------|
| 0.5 | Softer sigmoid, less confident |
| 1.0 | Standard |
| 2.0 | Sharper sigmoid, more confident |

**Finding**: Beta tuning had minimal impact. Keep at 1.0 and use score regularization instead.

---

## Experiment Summary

### V3: Baseline Establishment (20 experiments)

**Goal**: Find optimal LoRA configuration for Bradley-Terry RM training.

| ID | Config | Description | Best Acc | Finding |
|----|--------|-------------|----------|---------|
| A0 | Full fine-tune | No LoRA baseline | 67.13% | LoRA outperforms |
| A1-A3 | Differential LR | Slower backbone LR | ~51% | **FAILED** - backbone must adapt |
| B0-B2 | Attention-only LoRA | r=2,4,8 on attn | 69-70% | Underfits |
| **B2** | **r=4, all-linear, LR=5e-5** | **Optimal baseline** | **74.15%** | **Most stable** |
| C0 | r=8, all-linear, LR=1e-4 | Higher LR | 74.55% | Peak higher, overfits |
| C1 | r=8, all-linear, LR=2e-5 | Lower LR | 72.95% | Too conservative |
| D0-D1 | Batch size 32, 64 | Larger batches | 68-69% | Overfitting |
| F0-F3 | Differential LR combos | Various | 52-61% | All failed |

**Key V3 findings**:
1. All-linear LoRA >> attention-only (~4% gap)
2. Differential LR completely fails for RM training
3. r=4 with LR=5e-5 is most stable
4. Larger batches cause overfitting

### V4: Stability Investigation (12 experiments)

**Goal**: Investigate rsLoRA, trainable norms, and model scaling.

| ID | Config | Description | Best Acc | Finding |
|----|--------|-------------|----------|---------|
| V4-0 | B2 reproduction | Baseline | 73.75% | Confirmed |
| V4-1a | rsLoRA r=4 | Rank-stabilized | 73.75% | Slight stability gain |
| V4-1b | rsLoRA r=8 | Higher rank | 71.35% | More overfitting |
| V4-2a | LR=1.5e-4 + ES | Higher LR | 71.81% | Needs early stopping |
| V4-2b | LR=2e-4 + ES | Even higher LR | 72.27% | Overfits fast |
| V4-3 | 2000 steps | Extended training | 72.04% | Stable, no degradation |
| V4-4 | Trainable norms | Unfreeze LayerNorm | 69.72% | **Hurts** performance |
| V4-5 | Batch 32 | Larger batch | 69.03% | Worse than 16 |
| V4-6a | 7B model | Scale up | 66.36% | **Severe overfitting** |
| V4-6b | 7B + rsLoRA | 7B with rsLoRA | 64.97% | Still overfits |
| V4-7 | rsLoRA + norms | Combined | 74.15% | Matched V3-B2 |
| V4-8 | r=8 + ES + norms | Kitchen sink | 68.68% | Worse than components |

**Key V4 findings**:
1. rsLoRA provides modest stability improvement
2. Trainable normalization layers hurt performance
3. 7B model needs different hyperparameters (LR too high)
4. Combining techniques often underperforms individual ones

### V5: Regularization & Diagnostics (15 experiments)

**Goal**: Diagnostic validation + regularization comparison.

#### Phase 1: Diagnostics

| ID | Test | Purpose | Result | Interpretation |
|----|------|---------|--------|----------------|
| D1 | Label shuffle (50%) | Check for label leakage | 50.70% | ✅ No leakage |
| D2 | Prompt only | Check for prompt leakage | CRASH* | Bug fixed |
| D3 | Random negatives | Plausibility baseline | 55.91% | Plausibility ≠ preference |
| D4 | Length matched | Length bias check | 63.43% | ~10% length bias |
| C1 | Baseline | Reproduce V3-B2 | 73.35% | ✅ Confirmed |

*D2 crashed due to empty batch bug (now fixed)

#### Phase 2-4: Regularization

| ID | Config | Description | Best Acc | Margin | Finding |
|----|--------|-------------|----------|--------|---------|
| M1a | Margin λ=0.001 | Weak margin reg | 73.05% | 1.08 | Minimal effect |
| M1b | Margin λ=0.01 | Strong margin reg | 73.35% | 0.80 | Matches baseline |
| S1a | Score λ=0.001 | Weak score reg | 72.34% | 0.91 | Minimal effect |
| S1b | Score λ=0.01 | Moderate score reg | 71.84% | 0.61 | Slight drop |
| **S1c** | **Score λ=0.1** | **Strong score reg** | **75.55%** | **0.34** | **BEST!** |
| S2a | Beta=0.5 | Softer sigmoid | 71.74% | 1.04 | No improvement |
| S2b | Beta=2.0 | Sharper sigmoid | 72.75% | 1.09 | No improvement |
| S3 | Early stopping | Patience=5 | 72.65% | 1.03 | Good stability |
| A1 | Score reg + beta=0.5 | Combined | 72.44% | 0.46 | Worse than S1c alone |
| A2 | Score reg + ES | Combined | 72.34% | 0.61 | Worse than S1c alone |

**Key V5 findings**:
1. **Score regularization λ=0.1 improves accuracy by 2.2%**
2. Diagnostics confirm pipeline correctness
3. Length bias accounts for ~10% of signal
4. Combined approaches underperform single best technique

---

## Diagnostic Tests

Run these before production training to validate your pipeline:

### D1: Label Shuffle

```bash
--diagnostic-mode label_shuffle --diagnostic-shuffle-rate 0.5
```

**What it does**: Randomly swaps chosen/rejected labels for 50% of pairs.

**Expected**: ~50% accuracy, loss ~0.693 (log 2)

**If fails (>55%)**: Label leakage - the model is exploiting non-preference signals (tokenization artifacts, length, etc.)

### D2: Prompt Only

```bash
--diagnostic-mode prompt_only
```

**What it does**: Replaces all completions with "[empty]".

**Expected**: ~50% accuracy

**If fails (>55%)**: Prompt-level leakage - certain prompts correlate with preference direction

### D3: Random Negatives

```bash
--diagnostic-mode random_negatives
```

**What it does**: Replaces rejected responses with random responses from other prompts.

**Expected**: ~100% train accuracy, ~55% eval accuracy

**Interpretation**: The model easily learns to distinguish relevant vs irrelevant responses (plausibility), but this skill doesn't transfer to preference prediction.

### D4: Length Matched

```bash
--diagnostic-mode length_matched --diagnostic-length-threshold 50
```

**What it does**: Filters to pairs where |len(chosen) - len(rejected)| < 50 tokens.

**Expected**: ~10% accuracy drop from baseline

**Interpretation**: Length bias contributes ~10% of the preference signal in SHP. This is a known dataset artifact.

---

## What Works / What Doesn't

### What Works

| Technique | Evidence | Recommendation |
|-----------|----------|----------------|
| **Score regularization λ=0.1** | 75.55% vs 73.35% baseline | **Always use** |
| **All-linear LoRA** | 74% vs 70% | Use all 7 modules |
| **LoRA rank 4** | Best stability | Start here |
| **LR 5e-5** | Optimal for 3B | 10x higher than full FT |
| **max-seq-len 1024** | SHP has long responses | Don't truncate at 512 |
| **label-smoothing 0.05** | Improves calibration | Standard practice |
| **Batch size 16** | Best tradeoff | Don't increase |

### What Doesn't Work

| Technique | Evidence | Avoid |
|-----------|----------|-------|
| **Differential LR** | ~51% (random) | Backbone must adapt |
| **Trainable norms** | -4% accuracy | Hurts consistently |
| **Large batches (32+)** | Overfitting | Stick with 16 |
| **7B with 3B hyperparams** | Severe overfitting | Need lower LR, higher rank |
| **Combining techniques** | Usually worse | Test one variable at a time |
| **Beta tuning** | No improvement | Use score reg instead |

---

## Model Scaling: 7B Guidelines

**WARNING**: 7B models require different hyperparameters. V4 showed 7B with 3B settings severely overfits (peaks at step 300, then degrades).

| Parameter | 3B | 7B (Recommended) |
|-----------|----|----|
| `--lr` | 5e-5 | **1e-5** (5x lower) |
| `--lora-rank` | 4 | **16** (4x higher) |
| `--lora-alpha` | 32 | 32 |
| `--weight-decay` | 0.01 | **0.05** (5x higher) |
| `--early-stopping-patience` | 5 | **3** (stricter) |

---

## Troubleshooting

### Low Accuracy (<70%)

1. **Check target modules**: Ensure all 7 modules are specified
2. **Check seq length**: Use 1024, not 512
3. **Verify chat template**: Look for `<|im_start|>` in decoded logs
4. **Check data**: Ensure chosen/rejected aren't swapped

### Overfitting (>3% drop from peak)

1. Add `--score-regularization-lambda 0.1`
2. Enable `--early-stopping-patience 5`
3. Reduce `--lr` by 2x
4. Reduce `--lora-rank` to 4

### Empty Batch Errors

1. Increase `--max-seq-len`
2. Check diagnostic mode isn't producing empty responses

### CUDA Library Errors

Node-specific issue. Resubmit job to different node.

### 7B Model Degrading

Learning rate too high. Use `--lr 1e-5` and `--lora-rank 16`.

---

## File Reference

| File | Purpose |
|------|---------|
| `examples/reward_model/train_rm_bradley_terry.py` | Training script |
| `examples/reward_model/analyze_rm_experiments.py` | Results analysis |
| `examples/reward_model/run_rm_v5_experiments.slurm` | V5 experiment config |
| `src/ludic/training/loss.py` | BradleyTerryLoss implementation |
| `results/v5/` | V5 results (CSV, JSON) |

---

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original LoRA
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) - LR scaling insights
- [Sebastian Raschka's Tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) - Empirical study
- [RLHF Book Ch. 7](https://rlhfbook.com/c/07-reward-models.html) - RM training practices
- [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling) - State-of-the-art recipes

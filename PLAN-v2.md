# RM Trainer V2: Implementation Complete - Experimental Validation Plan

## Implementation Status: COMPLETE

All planned features have been implemented and tested (68 unit tests passing).

### Implemented Components

| Component | File | Status |
|-----------|------|--------|
| LR Scheduler + Early Stopping | `src/ludic/training/config.py`, `trainer.py` | ✅ |
| RM Batching Module | `src/ludic/training/batching/rm_batching.py` | ✅ |
| Classification Metrics | `src/ludic/training/classification_metrics.py` | ✅ |
| RMTrainer + RMTrainerConfig | `src/ludic/training/rm_trainer.py` | ✅ |
| Label Smoothing | `src/ludic/training/loss.py` (BradleyTerryLoss) | ✅ |
| Updated Training Script | `examples/reward_model/train_rm_bradley_terry.py` | ✅ |
| Updated Slurm Script | `examples/reward_model/train_rm_isambard.slurm` | ✅ |

### Test Coverage

```
tests/test_rm_batching.py              - 13 tests ✅
tests/test_classification_metrics.py   - 13 tests ✅
tests/test_trainer_scheduler.py        - 9 tests ✅
tests/test_trainer_early_stopping.py   - 11 tests ✅
tests/test_bradley_terry_label_smoothing.py - 9 tests ✅
tests/test_rm_trainer.py               - 13 tests ✅
─────────────────────────────────────────────────
TOTAL: 68 tests passing
```

---

## Experimental Validation Protocol

### Pre-Registered Hypotheses

Before running experiments, we pre-commit to the following expectations:

#### Hypothesis 1: Pair-Aware Batching Improves Accuracy
**Expectation**: RMTrainer V2 with pair-aware micro-batching will achieve higher pairwise accuracy than the original standalone script because:
- No pairs are split across micro-batches (complete gradient signal)
- Loss scaling by pairs ensures correct gradient accumulation
- FSDP gradient sync only on last micro-batch prevents stale gradients

**Metric**: `eval_pairwise_accuracy`
**Baseline** (old script): 60% on SHP test set
**Target**: > 63% with same training budget

#### Hypothesis 2: LR Scheduler Prevents Overfitting
**Expectation**: Cosine scheduler with warmup will:
- Prevent early divergence (warmup stabilizes)
- Reduce train-test gap (gradual LR decay reduces overfitting)

**Metrics**:
- `train_accuracy - eval_accuracy` gap
- `eval_loss` trajectory

**Baseline**: Train-test gap of ~15-20% at step 500
**Target**: Train-test gap < 10% at step 500

#### Hypothesis 3: Label Smoothing Improves Calibration
**Expectation**: Label smoothing (0.05) will:
- Reduce overconfident predictions
- Improve Expected Calibration Error (ECE)
- Slightly reduce pairwise accuracy (tradeoff)

**Metrics**:
- `eval_ece` (Expected Calibration Error)
- `eval_pairwise_accuracy`

**Without smoothing**: ECE likely > 0.15
**With smoothing (0.05)**: ECE < 0.12, accuracy within 2% of unsmoothed

#### Hypothesis 4: LoRA Achieves Near-Parity with Full Fine-Tuning
**Expectation**: LoRA (rank 8, alpha 16) will:
- Achieve within 3% pairwise accuracy of full fine-tuning
- Train 3-5x faster (less memory, faster backward)
- Reward head remains fully trainable via `modules_to_save`

**Metrics**:
- `eval_pairwise_accuracy` (LoRA vs Full FT)
- Training throughput (samples/sec)
- Peak GPU memory

**Target**: LoRA accuracy >= (Full FT accuracy - 3%)

#### Hypothesis 5: Early Stopping Prevents Overfitting
**Expectation**: Early stopping (patience=5, metric=eval_accuracy, mode=max) will:
- Stop training before severe overfitting
- Select best checkpoint automatically
- Reduce total training time

**Metrics**:
- Final step vs configured max steps
- Best eval accuracy at stop point

---

## Experiment 1: Baseline Comparison

### Purpose
Compare RMTrainer V2 against the old standalone script to validate infrastructure improvements.

### Commands

**Old Script Baseline (for reference only - already ran)**:
```bash
# This was run previously with the old script
# Results: 60% test accuracy, heavy overfitting after step 300
```

**New RMTrainer V2**:
```bash
sbatch examples/reward_model/train_rm_isambard.slurm
```

Or manually:
```bash
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-7B \
    --hf-dataset stanfordnlp/SHP \
    --hf-split train \
    --limit 5000 \
    --steps 1000 \
    --batch-size 16 \
    --micro-token-budget 8192 \
    --max-seq-len 1024 \
    --lr 5e-6 \
    --weight-decay 0.01 \
    --lr-scheduler cosine \
    --warmup-steps 50 \
    --label-smoothing 0.05 \
    --log-every 10 \
    --save-every 100 \
    --eval-split test \
    --eval-limit 1000 \
    --eval-every 100 \
    --early-stopping-patience 5 \
    --profile-memory \
    --output-dir checkpoints_rm_v2_baseline
```

### Expected Output Files
```
checkpoints_rm_v2_baseline/
├── metrics.csv                 # Step-by-step metrics
├── reward_distributions.json   # Score distributions per eval
├── best/                       # Best model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
└── step_*/                     # Periodic checkpoints
```

### How to Interpret Results

**Read metrics.csv**:
```bash
# View final metrics
tail -20 checkpoints_rm_v2_baseline/metrics.csv

# Expected columns:
# step, train_loss, train_acc, train_margin, eval_loss, eval_acc, eval_margin, eval_pairs
```

**Success Criteria**:
| Metric | Failure | Acceptable | Success |
|--------|---------|------------|---------|
| eval_acc (final) | < 58% | 58-63% | > 63% |
| train-eval gap | > 15% | 10-15% | < 10% |
| early_stop step | > 900 (didn't stop) | 600-900 | 400-600 (stopped early) |

---

## Experiment 2: Ablation Study

### Purpose
Isolate the contribution of each improvement.

### Ablation Matrix

| Experiment | Scheduler | Label Smooth | Early Stop | Expected Accuracy |
|------------|-----------|--------------|------------|-------------------|
| A1: No improvements | None | 0.0 | Off | ~60% (baseline) |
| A2: + Scheduler only | cosine/50 | 0.0 | Off | ~62% |
| A3: + Label smoothing | cosine/50 | 0.05 | Off | ~61% (slight drop) |
| A4: + Early stopping | cosine/50 | 0.05 | patience=5 | ~63% |
| A5: Full config | cosine/50 | 0.05 | patience=5 | ~63-65% |

### Commands for Each Ablation

**A1: No improvements**:
```bash
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-7B \
    --hf-dataset stanfordnlp/SHP \
    --limit 5000 --steps 1000 --batch-size 16 \
    --lr 5e-6 --weight-decay 0.01 \
    --eval-split test --eval-limit 1000 --eval-every 100 \
    --output-dir checkpoints_rm_v2_A1_none
```

**A2: + Scheduler**:
```bash
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-7B \
    --hf-dataset stanfordnlp/SHP \
    --limit 5000 --steps 1000 --batch-size 16 \
    --lr 5e-6 --weight-decay 0.01 \
    --lr-scheduler cosine --warmup-steps 50 \
    --eval-split test --eval-limit 1000 --eval-every 100 \
    --output-dir checkpoints_rm_v2_A2_scheduler
```

**A3: + Label smoothing**:
```bash
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-7B \
    --hf-dataset stanfordnlp/SHP \
    --limit 5000 --steps 1000 --batch-size 16 \
    --lr 5e-6 --weight-decay 0.01 \
    --lr-scheduler cosine --warmup-steps 50 \
    --label-smoothing 0.05 \
    --eval-split test --eval-limit 1000 --eval-every 100 \
    --output-dir checkpoints_rm_v2_A3_smoothing
```

**A4: + Early stopping (full config)**:
```bash
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-7B \
    --hf-dataset stanfordnlp/SHP \
    --limit 5000 --steps 1000 --batch-size 16 \
    --lr 5e-6 --weight-decay 0.01 \
    --lr-scheduler cosine --warmup-steps 50 \
    --label-smoothing 0.05 \
    --early-stopping-patience 5 \
    --eval-split test --eval-limit 1000 --eval-every 100 \
    --output-dir checkpoints_rm_v2_A4_full
```

### Analysis Script

After all ablations complete, compare results:
```python
import pandas as pd
from pathlib import Path

experiments = ["A1_none", "A2_scheduler", "A3_smoothing", "A4_full"]
results = {}

for exp in experiments:
    df = pd.read_csv(f"checkpoints_rm_v2_{exp}/metrics.csv")
    # Get best eval accuracy
    best_eval = df["eval_acc"].dropna().max()
    final_train = df["train_acc"].dropna().iloc[-1]
    gap = final_train - best_eval
    results[exp] = {
        "best_eval_acc": best_eval,
        "train_eval_gap": gap,
        "final_step": df["step"].iloc[-1],
    }

print(pd.DataFrame(results).T)
```

---

## Experiment 3: LoRA Comparison

### Purpose
Validate LoRA achieves near-parity with full fine-tuning.

### Commands

**Full Fine-Tuning**:
```bash
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-7B \
    --hf-dataset stanfordnlp/SHP \
    --limit 5000 --steps 1000 --batch-size 16 \
    --lr 5e-6 --weight-decay 0.01 \
    --lr-scheduler cosine --warmup-steps 50 \
    --label-smoothing 0.05 \
    --early-stopping-patience 5 \
    --profile-memory \
    --eval-split test --eval-limit 1000 --eval-every 100 \
    --output-dir checkpoints_rm_v2_full_ft
```

**LoRA (rank 8)**:
```bash
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-7B \
    --hf-dataset stanfordnlp/SHP \
    --limit 5000 --steps 1000 --batch-size 16 \
    --lr 5e-6 --weight-decay 0.01 \
    --lr-scheduler cosine --warmup-steps 50 \
    --label-smoothing 0.05 \
    --early-stopping-patience 5 \
    --lora --lora-rank 8 --lora-alpha 16 \
    --profile-memory \
    --eval-split test --eval-limit 1000 --eval-every 100 \
    --output-dir checkpoints_rm_v2_lora_r8
```

**LoRA (rank 16)**:
```bash
uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
    --model Qwen/Qwen2.5-7B \
    --hf-dataset stanfordnlp/SHP \
    --limit 5000 --steps 1000 --batch-size 16 \
    --lr 5e-6 --weight-decay 0.01 \
    --lr-scheduler cosine --warmup-steps 50 \
    --label-smoothing 0.05 \
    --early-stopping-patience 5 \
    --lora --lora-rank 16 --lora-alpha 32 \
    --profile-memory \
    --eval-split test --eval-limit 1000 --eval-every 100 \
    --output-dir checkpoints_rm_v2_lora_r16
```

### Expected Results

| Config | Trainable Params | Peak Memory | Accuracy |
|--------|------------------|-------------|----------|
| Full FT | 100% (~500M) | ~8GB | 63-65% |
| LoRA r8 | ~1-2% (~5-10M) | ~4GB | 60-63% |
| LoRA r16 | ~2-4% (~10-20M) | ~5GB | 61-64% |

### Success Criteria
- LoRA accuracy within 3% of Full FT
- LoRA memory < 60% of Full FT
- Reward head must be trainable (check logs for "Auto-detected reward head")

---

## Experiment 4: Hyperparameter Sensitivity

### Purpose
Find optimal hyperparameters for production use.

### Grid Search

| Parameter | Values to Test |
|-----------|----------------|
| lr | [1e-6, 5e-6, 1e-5] |
| label_smoothing | [0.0, 0.05, 0.1] |
| warmup_steps | [0, 50, 100] |
| regularization_lambda | [0.0, 0.001, 0.01] |

### Priority Order
1. lr (most impactful)
2. label_smoothing (calibration vs accuracy tradeoff)
3. warmup_steps (training stability)
4. regularization_lambda (overfitting control)

### Quick Sensitivity Check (1 GPU job)
```bash
# Test LR sensitivity only (quick check)
for LR in 1e-6 5e-6 1e-5; do
    uv run --env-file .env python examples/reward_model/train_rm_bradley_terry.py \
        --model Qwen/Qwen2.5-7B \
        --hf-dataset stanfordnlp/SHP \
        --limit 2000 --steps 300 --batch-size 16 \
        --lr $LR --weight-decay 0.01 \
        --lr-scheduler cosine --warmup-steps 30 \
        --eval-split test --eval-limit 500 --eval-every 50 \
        --output-dir checkpoints_rm_v2_lr_${LR}
done
```

---

## How to Monitor Running Jobs

### Check Slurm Queue
```bash
squeue --me
```

### Monitor Logs
```bash
# Real-time log following
tail -f logs/slurm_<jobid>_rm.log

# Check for errors
grep -i "error\|exception\|fail" logs/slurm_<jobid>_rm.log
```

### Check GPU Memory (if interactive session)
```bash
nvidia-smi -l 5  # Refresh every 5 seconds
```

---

## Interpreting Results

### Reading metrics.csv

```csv
step,train_loss,train_acc,train_margin,eval_loss,eval_acc,eval_margin,eval_pairs
10,0.693,0.52,0.05,,,,
20,0.680,0.55,0.12,,,,
...
100,0.520,0.68,0.45,0.550,0.62,0.35,500
```

**Key columns**:
- `train_acc`: Accuracy on training batch (can overfit)
- `eval_acc`: Accuracy on held-out test set (true signal)
- `train_margin`: Average `score_chosen - score_rejected` on train
- `eval_margin`: Average margin on test (should be positive)

### Warning Signs

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| eval_acc decreasing while train_acc increasing | Overfitting | Lower LR, add regularization, early stop |
| eval_margin negative | Model reversed | Check data loading, reduce LR drastically |
| train_loss stuck | Learning rate too low | Increase LR |
| train_loss exploding | Learning rate too high | Decrease LR, check gradients |
| eval_acc oscillating | LR too high late in training | Use scheduler decay |

### Healthy Training Trajectory

```
Step 0:   eval_acc ~50% (random)
Step 100: eval_acc ~55%
Step 200: eval_acc ~60%
Step 300: eval_acc ~62%
Step 400: eval_acc ~63% (peak)
Step 500: eval_acc ~62% (early stopping triggers)
```

---

## Post-Experiment Checklist

After experiments complete:

1. **Collect Results**:
   ```bash
   # Summarize all experiment results
   for d in checkpoints_rm_v2_*/; do
       echo "=== $d ==="
       tail -3 "$d/metrics.csv"
   done
   ```

2. **Check Best Models**:
   ```bash
   # Verify best checkpoints exist
   ls -la checkpoints_rm_v2_*/best/
   ```

3. **Archive Logs**:
   ```bash
   mkdir -p experiment_logs/$(date +%Y%m%d)
   cp logs/slurm_*_rm.log experiment_logs/$(date +%Y%m%d)/
   ```

4. **Document Findings**:
   - Update this file with actual results
   - Create GitHub issue if bugs found
   - Update default hyperparameters if improvements confirmed

---

## Rollback Plan

If V2 performs worse than baseline:

1. **Immediate**: Use old script at `git show HEAD~10:examples/reward_model/train_rm_bradley_terry.py`
2. **Debug**: Check logs for errors, verify pair batching logic
3. **Partial rollback**: Disable individual features (set `--label-smoothing 0`, remove `--lr-scheduler`, etc.)

---

## Next Steps After Validation

Once experiments confirm improvements:

1. **Scale up**: Run on larger datasets (full SHP, Anthropic HH)
2. **Integration test**: Use trained RM in GRPO training pipeline
3. **Documentation**: Add RMTrainer to docs/training.md
4. **Release**: Tag v0.X.0 with RMTrainer stable

---

## References

- Bradley-Terry model: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
- RLHF reward modeling: https://arxiv.org/abs/2203.02155
- Label smoothing: https://arxiv.org/abs/1906.02629
- LoRA: https://arxiv.org/abs/2106.09685
- Expected Calibration Error: https://arxiv.org/abs/1706.04599

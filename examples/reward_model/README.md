# Reward Model Training with Bradley-Terry

Train reward models from pairwise preferences and use them for RLHF-style policy training.

## Overview

This example demonstrates:
1. Training a reward model using Bradley-Terry loss on preference data
2. Using the trained RM to score rollouts during policy training (cold-start RLHF)

## Prerequisites

- GPU with sufficient VRAM (tested on A100)
- Preference dataset in JSONL format (or HuggingFace datasets)

Install dependencies:
```bash
uv sync --extra examples
```

## Data Format

Preference data should have prompt/chosen/rejected columns:

```json
{"prompt": "What is 2+2?", "chosen": "The answer is 4.", "rejected": "The answer is 5."}
{"prompt": "Explain gravity.", "chosen": "Gravity is...", "rejected": "Magnets cause..."}
```

Or use HuggingFace datasets like `Anthropic/hh-rlhf` or `stanfordnlp/SHP`.

## 1) Train Reward Model

Train a reward model on preference data:

```bash
uv run python examples/reward_model/train_rm_bradley_terry.py \
  --model Qwen/Qwen2.5-0.5B \
  --hf-dataset stanfordnlp/SHP \
  --hf-split train \
  --steps 100 \
  --batch-size 8
```

This produces a reward model checkpoint that scores sequences by preference.

### Bradley-Terry Loss

The Bradley-Terry model learns:
```
P(chosen > rejected) = sigmoid(beta * (s_chosen - s_rejected))
```

Where `s_*` are learned scalar scores. The loss minimizes:
```
L = -E[log P(chosen > rejected)]
```

## 2) Cold-Start RLHF

Use the trained RM to guide policy training:

```bash
# Terminal 1: Start vLLM for policy
CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
  --model Qwen/Qwen2.5-0.5B-Instruct

# Terminal 2: Train policy with RM
CUDA_VISIBLE_DEVICES=1 uv run python examples/reward_model/cold_start_rlhf.py \
  --mode train \
  --policy-model Qwen/Qwen2.5-0.5B-Instruct \
  --rm-checkpoint checkpoints_rm/step_100 \
  --policy-steps 50
```

The cold-start workflow:
1. Load pre-trained reward model
2. Generate rollouts with the policy
3. Score rollouts with RM (adds `rm_score` to metadata)
4. Train policy using `make_grpo_with_rm` (combines env reward + RM bonus)

## Algorithm Options

### Pure RM (replace env rewards)
```python
algorithm = make_grpo_with_rm(
    group_size=8,
    rm_mode="replace",  # Use only RM score
)
```

### RM as Bonus (additive)
```python
algorithm = make_grpo_with_rm(
    group_size=8,
    rm_mode="bonus",
    rm_coeff=0.5,  # env + 0.5 * rm
)
```

### ScaleRL with RM
```python
algorithm = make_scalerl_with_rm(
    group_size=8,
    rm_mode="bonus",
    rm_coeff=0.1,
    kl_coeff=0.01,  # Optional KL penalty
)
```

## Tips

- Start with a small RM (0.5B-1B params) for fast iteration
- Use `rm_coeff` in range [0.1, 1.0] to balance env vs RM signal
- Monitor RM accuracy during training to detect overfitting
- For DPO-style direct optimization, use `make_bradley_terry(score_type="logprob")`

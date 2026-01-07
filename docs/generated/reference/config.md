# Ludic Configuration Reference

> Auto-generated documentation - Last updated: 2026-01-06

## TrainerConfig

**Location**: `src/ludic/training/config.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_device` | str | "cuda" | Device for trainable model |
| `runtime_device` | str? | None | Device for serving (defaults to model_device) |
| `lr` | float | 1e-5 | Learning rate |
| `weight_decay` | float | 0.01 | L2 regularization |
| `betas` | tuple | (0.9, 0.999) | AdamW momentum |
| `eps` | float | 1e-8 | AdamW epsilon |
| `max_grad_norm` | float? | 1.0 | Gradient clipping |
| `max_seq_len` | int | 1024 | Max sequence length |
| `micro_token_budget` | int | 8192 | Max tokens per micro-batch |
| `sync_every_steps` | int | 1 | Weight push frequency |
| `mixed_precision_dtype` | str? | "bf16" | Mixed precision |
| `cast_logits_to_fp32` | bool | True | FP32 for IS ratios |
| `max_lag` | int? | None | Drop stale batches |
| `log_every` | int | 1 | Logging frequency |
| `pad_token_id` | int | 0 | Padding token |
| `eval_at_start` | bool | False | Eval before training |
| `eval_every_n_steps` | int? | None | Eval frequency |
| `eval_concurrency` | int | 32 | Parallel eval episodes |

## CheckpointConfig

**Location**: `src/ludic/training/checkpoint.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | (required) | Checkpoint directory |
| `every_n_steps` | int | 0 | Save frequency (0 = disabled) |
| `max_to_keep` | int? | 2 | Keep N most recent |
| `save_optimizer` | bool | True | Save optimizer state |

## SamplingConfig

**Location**: `src/ludic/inference/sampling.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.7 | Sampling temperature |
| `max_tokens` | int | 256 | Max completion length |
| `top_p` | float | 1.0 | Nucleus sampling |
| `frequency_penalty` | float | 0.0 | Token frequency penalty |
| `presence_penalty` | float | 0.0 | Token presence penalty |
| `stop` | str/list? | None | Stop sequences |

## ReturnSpec

**Location**: `src/ludic/inference/request.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_token_ids` | bool | False | Return completion token IDs |
| `return_chosen_logprobs` | bool | False | Return per-token logprobs |
| `top_logprobs_k` | int | 1 | Top-K alternative logprobs |

Factory methods:
- `ReturnSpec.for_eval()`: Token IDs only
- `ReturnSpec.for_rl()`: Token IDs + logprobs

## Credit Assigner Configs

### GroupNormalizedReturn

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_size` | int | (required) | Rollouts per group |
| `normalize_adv` | bool | False | Normalize to zero mean/unit std |
| `positive_only` | bool | False | Clip negative advantages |

### HybridNormalizedReturn

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_size` | int | (required) | Rollouts per group |
| `eps` | float | 1e-8 | Numerical stability |
| `positive_only` | bool | False | Clip negative advantages |

## Environment Configs

### CodeExecConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout_per_test_s` | float | 5.0 | Per-test timeout |
| `memory_limit_mb` | int | 256 | Memory limit |
| `max_tests` | int? | None | Max tests per episode |
| `stop_on_first_failure` | bool | True | Stop on first fail |
| `partial_credit` | bool | False | Reward = pass fraction |

### DockerSandboxConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `python_version` | str | "3.11" | Python version |
| `base_image` | str? | None | Custom base image |
| `memory_limit` | str | "256m" | Memory limit |
| `network_disabled` | bool | True | Disable network |

## Environment Variables

| Variable | Usage | Default |
|----------|-------|---------|
| `HF_TOKEN` | HuggingFace API | None |
| `VLLM_USE_V1` | vLLM version | "1" |
| `SLURM_JOB_ID` | HPC cluster | None |
| `LUDIC_TYPECHECK` | Enable type checks | "0" |
| `WANDB_PROJECT` | W&B project | None |

# Ludic Setup & Development Guide

> Auto-generated documentation - Last updated: 2026-01-06

## Prerequisites

- **Python 3.12+**
- **PyTorch >= 2.9.0**
- **uv** package manager (recommended)
- **Git**

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/your-repo/ludic.git
cd ludic
uv sync
```

### With Extras

```bash
# Code execution sandbox
uv sync --extra code-exec

# Development tools
uv sync --group dev

# Full setup
uv sync --extra code-exec --group dev
```

### Platform Notes

- **Linux + GPU**: CUDA 12.8 wheels automatically selected
- **macOS/Windows**: CPU wheels automatically selected
- **vLLM**: Linux only

## Quick Start

### 1. Start Inference Server

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python -m ludic.inference.vllm_server \
  --model Qwen/Qwen2.5-7B-Instruct
```

### 2. Run Training

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/tic_tac_toe/train_tic_tac_toe.py \
  --micro-token-budget 8192 \
  --max-seq-len 1024
```

### 3. Run Evaluation

```bash
PYTHONPATH=. uv run python examples/tic_tac_toe/eval_tic_tac_toe_vllm.py \
  --model hallerite/Qwen2.5-7B-TTT \
  --episodes 100
```

## Examples Overview

| Example | Purpose | GPUs | Duration |
|---------|---------|------|----------|
| **Tic-Tac-Toe** | Game-playing RL | 2 | 10-30 min |
| **GSM8K** | Math reasoning | 2+ | 1-4 hours |
| **FSDP2** | Multi-GPU template | 4 | 2-8 hours |
| **Pipeline RL** | Async actor/learner | 2+ | Variable |
| **Code Exec** | Code generation | 2+ | 1-2 hours |

### GSM8K Example

```bash
# Server
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python -m ludic.inference.vllm_server \
  --model Qwen/Qwen2.5-0.5B-Instruct

# Training
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/gsm8k/train_gsm8k.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --micro-token-budget 16384
```

### FSDP2 Multi-GPU

```bash
# GPU0: vLLM
CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
  --model Qwen/Qwen2.5-7B-Instruct

# GPUs 1-3: Training
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. uv run torchrun --nproc_per_node=3 \
  examples/fsdp2_training/train_math_fsdp2.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train-steps 50 --group-size 8
```

## Development Workflow

### Testing

```bash
# Unit tests
pytest tests/ -m "not gpu"

# Integration tests
pytest tests/ -m "integration"

# Specific file
pytest tests/test_parsers.py -v
```

### Code Formatting

```bash
uv run ruff format src/ tests/ examples/
uv run ruff check src/ tests/ examples/ --fix
```

### Type Checking

```bash
uv run py -m ludic.typing
```

## Common Tasks

### Create Custom Environment

```python
from ludic.envs import SingleAgentEnv
from ludic.types import StepOutcome

class MyEnv(SingleAgentEnv):
    def env_reset(self, *, seed=None):
        return "Initial observation", {}

    def env_step(self, action: str) -> StepOutcome:
        reward = evaluate(action)
        return StepOutcome(
            obs="Next observation",
            reward=reward,
            terminated=True,
            truncated=False,
        )

    def env_current_obs(self) -> str:
        return "Current observation"
```

### Create Custom Parser

```python
from ludic.parsers import ParseResult

def my_parser(response: str) -> ParseResult:
    try:
        action = extract_action(response)
        return ParseResult(action=action, reward=0.1, obs=None)
    except:
        return ParseResult(action=None, reward=-0.1, obs="Invalid format")
```

### Run Training Loop

```python
from ludic.training import (
    RolloutEngine, RolloutBatchSource, Trainer, TrainerConfig, make_grpo
)

engine = RolloutEngine(env_registry={...}, protocol_registry={...})
algo = make_grpo(group_size=8)

batch_source = RolloutBatchSource(
    engine=engine,
    request_fn=make_requests,
    algorithm=algo,
    max_steps=512,
)

trainer = Trainer(
    model=model,
    algo=algo,
    batch_source=batch_source,
    cfg=TrainerConfig(learning_rate=2e-5),
)

await trainer.train(num_steps=1000)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'ludic'` | Set `PYTHONPATH=.` |
| CUDA OOM | Reduce `--micro-token-budget` |
| vLLM won't start | Check GPU memory, reduce `--gpu-memory-utilization` |
| Slow rollouts | Increase `--concurrency` |

## Resources

- **CLAUDE.md**: Design philosophy & architecture
- **CONSIDERATIONS.md**: Truncation semantics, checkpoint formats
- **Example READMEs**: Per-example documentation

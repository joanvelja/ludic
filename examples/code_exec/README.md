# Code Execution Training

> Train LLMs on code generation with sandboxed test-driven evaluation.

## What This Is

This module provides an RL training environment where:
- The **agent generates code** in response to programming problems
- The **environment executes the code** in isolated containers
- **Test cases verify correctness** and provide reward signal
- The **trainer updates the policy** based on execution outcomes

Key features:
- **Sandboxed execution** — Generated code runs in Docker/Podman containers for security
- **Persistent containers** — 40x faster than cold-start containers (17ms vs 700ms per execution)
- **Automatic caching** — Skip redundant executions (especially valuable with CISPO/GRPO)
- **Multi-backend** — Works on laptop (Docker) or HPC clusters (Podman-HPC)

## Who This Is For

**Experienced Ludic users**: Jump to [Quick Start](#quick-start) for copy-paste examples.

**New to Ludic**: Read [How It Works](#how-it-works) first to understand the concepts.

**Prerequisites**:
- Familiarity with Ludic's training concepts (`Trainer`, `RolloutEngine`, `BatchSource`)
- Docker running locally, or Podman-HPC on your HPC cluster
- A vLLM inference server for generation

---

## Quick Start

### Prerequisites

1. **Docker daemon running** — See [Setup Guide](#setup-guide) if not
2. **HuggingFace token** — Create `.env` file: `echo 'HF_TOKEN=your_token' > .env`
3. **Dependencies**: `pip install docker datasets peft`

### 5-Minute Local Run

```bash
# Terminal 1: Start vLLM inference server
CUDA_VISIBLE_DEVICES=0 uv run --env-file .env python -m ludic.inference.vllm_server \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct

# Terminal 2: Run training
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run --env-file .env python examples/code_exec/train_apps.py \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --limit 100 \
    --train-steps 10
```

You should see:
- Sandbox pool starting with 4 workers
- Baseline evaluation running
- Training steps with reward metrics

### HPC Cluster Run (Slurm)

```bash
# 1. Prepare environment on LOGIN NODE (one-time, requires internet)
./examples/code_exec/prepare_env.sh

# 2. Submit job to compute nodes
sbatch examples/code_exec/train_apps_isambard.slurm
```

The Slurm script handles:
- Starting vLLM server on GPU 0
- Running training on GPU 1
- Auto-detecting Podman-HPC backend
- Structured logging in `logs/YYYY-MM-DD/`

---

## How It Works

### The Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                          │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐   │
│  │   Trainer   │◄──│ BatchSource  │◄──│  RolloutEngine  │   │
│  └─────────────┘   └──────────────┘   └────────┬────────┘   │
│        ▲                                       │            │
│        │                           ┌───────────▼─────────┐  │
│   Weight Updates                   │ SingleAgentProtocol │  │
│        │                           └────────────┬────────┘  │
│        ▼                                        │           │
│  ┌───────────┐                     ┌────────────▼────────┐  │
│  │   vLLM    │◄────────────────────│    CodeExecEnv      │  │
│  │  Server   │    generates code   └────────────┬────────┘  │
│  └───────────┘                                  │           │
└─────────────────────────────────────────────────┼───────────┘
                                                  │ executes
                                     ┌────────────▼────────┐
                                     │    SandboxPool      │
                                     │  ┌────┐ ┌────┐      │
                                     │  │ S1 │ │ S2 │ ...  │
                                     │  └────┘ └────┘      │
                                     └─────────────────────┘
```

**Step by step:**

1. **RolloutEngine** creates a `CodeExecEnv` for each problem from the dataset
2. **Agent** (via vLLM) generates Python code given the problem prompt
3. **CodeExecEnv** sends the code to a sandboxed container for execution
4. **Test cases** are run against the code; results determine the reward
5. **Trainer** collects rollouts and updates the model weights
6. **Weights are pushed** back to vLLM for the next generation round

### Key Concepts

#### Sandboxing

**Why sandbox?** LLM-generated code can be malicious, buggy, or resource-hungry. Sandboxing:
- Prevents file system access outside the container
- Limits memory and CPU usage
- Disables network access (by default)
- Isolates each execution from others

**Persistent containers** are the key to performance. Instead of starting a new container per execution (700ms overhead), we keep containers running and reuse them (17ms overhead).

#### Backend Auto-Detection

| Environment | Priority |
|-------------|----------|
| Inside Slurm job | `podman-hpc` → `docker` |
| Outside Slurm | `docker` → `podman-hpc` |

Override with `--sandbox-backend docker` or `--sandbox-backend podman-hpc`.

#### Test-Driven Evaluation

Each problem in the dataset has test cases (input/output pairs). The flow:

1. **TestAdapter** extracts test cases from the dataset format (e.g., `APPSTestAdapter` for APPS)
2. **StdinStdoutRunner** executes the code with each test's input as stdin
3. **OutputVerifier** compares actual output to expected output
4. **Reward** is computed based on test pass rate

#### Caching

The LRU cache prevents redundant execution:

- **Cache key**: `hash(code) + hash(tests)`
- **Hit rate**: Often 30-50% with CISPO/GRPO (multiple generations per prompt)
- **Speedup**: Cache hits return instantly (no container execution)

Monitor cache performance:
```python
stats = pool.cache_stats
# {'hits': 150, 'misses': 50, 'size': 200}
hit_rate = stats['hits'] / (stats['hits'] + stats['misses'])
```

### Reward Shaping

| Event | Reward | Configurable | Rationale |
|-------|--------|--------------|-----------|
| All tests pass | `+1.0` | — | Complete success |
| Some tests pass | `0.0` to `1.0` | `--partial-credit` | Smoother gradient signal |
| All tests fail | `0.0` | — | No partial credit by default |
| Compile error | `-0.1` | `compile_failure_reward` | Discourage syntax errors |
| Proper code block | `+0.05` | Parser reward | Encourage correct formatting |

**When to enable partial credit:**
- Training from scratch (model needs incremental signal)
- Long test suites where all-or-nothing is too sparse

**When to keep binary rewards:**
- Fine-tuning a capable model
- Problems where partial correctness is meaningless

---

## Configuration Reference

### Training Script Arguments (`train_apps.py`)

#### Model & Inference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Model name or path |
| `--host` | `127.0.0.1` | vLLM server host |
| `--port` | `8000` | vLLM server port |
| `--max-prompt-tokens` | `1024` | Max prompt length (longer prompts filtered) |
| `--max-new-tokens` | `4096` | Max generation length |

#### Training

| Flag | Default | Description |
|------|---------|-------------|
| `--train-steps` | `100` | Number of training steps |
| `--batch-size` | `4` | Rollout requests per batch |
| `--group-size` | `8` | CISPO group size (rollouts per prompt) |
| `--train-temperature` | `0.8` | Sampling temperature |
| `--max-seq-len` | `2048` | Max tokens per sample (truncation limit) |
| `--micro-token-budget` | `16384` | Max padded tokens per micro-batch |

#### LoRA

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-rank` | `8` | LoRA rank |
| `--lora-alpha-mult` | `2.0` | Alpha = rank × mult |
| `--lora-dropout` | `0.0` | LoRA dropout |

#### Dataset

| Flag | Default | Description |
|------|---------|-------------|
| `--split` | `train` | Dataset split |
| `--limit` | None | Max samples to load |
| `--difficulty` | None | Filter: `introductory`, `interview`, `competition` |
| `--eval-samples` | `200` | Hold out for evaluation |

#### Sandbox

| Flag | Default | Description |
|------|---------|-------------|
| `--sandbox-backend` | `auto` | `auto`, `docker`, `podman-hpc` |
| `--sandbox-workers` | `4` | Container pool size |
| `--python-version` | `3.11` | Python in sandbox |
| `--timeout-per-test` | `1.0` | Per-test timeout (seconds) |
| `--partial-credit` | `False` | Enable fractional rewards |
| `--minimal-sandbox` | `False` | Skip memory/network limits (HPC compat) |
| `--max-concurrent-ops` | `8` | Semaphore limit for Podman |

#### Evaluation

| Flag | Default | Description |
|------|---------|-------------|
| `--eval-every` | `25` | Eval every N steps |
| `--eval-before-start` | `True` | Run baseline evaluation |
| `--eval-concurrency` | `32` | Parallel eval rollouts |
| `--eval-temperature` | `0.0` | Greedy decoding for eval |

#### Logging

| Flag | Default | Description |
|------|---------|-------------|
| `--wandb` | `False` | Enable W&B logging |
| `--wandb-project` | `ludic-apps` | W&B project name |

### Environment Configuration (`CodeExecConfig`)

| Field | Default | Description |
|-------|---------|-------------|
| `timeout_per_test_s` | `5.0` | Per-test execution timeout |
| `memory_limit_mb` | `256` | Container memory limit |
| `max_tests` | `None` | Limit test count (None = all) |
| `stop_on_first_failure` | `True` | Early stop on failure |
| `compile_first` | `True` | Syntax check before running |
| `partial_credit` | `False` | Reward = pass_rate (vs binary) |
| `compile_failure_reward` | `-0.1` | Penalty for syntax errors |
| `use_cache` | `True` | Enable execution caching |

### Sandbox Pool Sizing

| Environment | CPUs | Recommended `--sandbox-workers` |
|-------------|------|--------------------------------|
| Laptop (M1/M2) | 8-10 | 4 |
| Workstation | 16-32 | 8-16 |
| HPC node | 64-128 | 24-64 |

**Rule of thumb**: Each sandbox uses ~0.5-1 CPU core. Use `floor(cpus / 2)`.

**Concurrency vs Workers**:
- `--concurrency` controls parallel rollouts (async tasks)
- `--sandbox-workers` controls parallel code executions
- If `concurrency > sandbox-workers`, tasks queue for sandboxes

---

## End-to-End Example

This complete example shows how to build a training script from scratch:

```python
"""Minimal code execution training script."""

import asyncio
from datasets import load_dataset

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec
from ludic.interaction import SingleAgentProtocol
from ludic.parsers import ParseResult
from ludic.distributed.adapters import create_vllm_publisher
from ludic.training import (
    RolloutEngine, RolloutBatchSource, Trainer, TrainerConfig,
    make_cispo, make_dataset_queue_requests_fn,
)
from ludic.envs.code_exec import (
    CodeExecEnv, CodeExecConfig, create_sandbox_pool, APPSTestAdapter,
)

async def main():
    # 1. Load dataset
    ds = load_dataset("RoganInglis/apps-control-arena", split="train")
    samples = [{"question": r["question"], "inputs": r["inputs"], "outputs": r["outputs"]}
               for r in list(ds)[:100]]

    # 2. Create sandbox pool (shared across all envs)
    pool = await create_sandbox_pool(n_workers=4, backend="auto")

    # 3. Setup inference client
    client = VLLMChatClient(host="127.0.0.1", port=8000, enable_weight_updates=True)
    publisher = create_vllm_publisher(client)

    # 4. Environment factory (captures pool via closure)
    adapter = APPSTestAdapter()
    env_config = CodeExecConfig(timeout_per_test_s=5.0, partial_credit=False)

    def env_factory(sample):
        return CodeExecEnv(sample=sample, sandbox_pool=pool,
                          test_adapter=adapter, config=env_config)

    # 5. Protocol factory
    def protocol_factory():
        return SingleAgentProtocol(agent=Agent(
            client=client, model="Qwen/Qwen2.5-3B-Instruct",
            ctx=FullDialog(),
            parser=lambda raw: ParseResult(action=raw, reward=0.0, obs=None),
        ))

    # 6. Setup training pipeline
    engine = RolloutEngine(
        env_registry={"apps": env_factory},
        protocol_registry={"single": protocol_factory},
    )

    algo = make_cispo(group_size=8, clip_eps_high=0.2, length_normalize=True)

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=make_dataset_queue_requests_fn(...),  # See train_apps.py
        concurrency=32,
    )

    # 7. Train
    trainer = Trainer(
        model=your_model,  # Load with LoRA
        algo=algo,
        batch_source=batch_source,
        publisher=publisher,
        cfg=TrainerConfig(max_seq_len=2048, micro_token_budget=16384),
    )

    await trainer.train(num_steps=100)

    # 8. Cleanup
    await pool.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

For a complete, production-ready script, see [`train_apps.py`](./train_apps.py).

---

## Customization

### Using a Different Dataset

Implement the `TestAdapter` protocol:

```python
from ludic.envs.code_exec import TestAdapter, TestCase

class MyDatasetAdapter:
    def get_tests(self, sample: dict) -> list[TestCase]:
        return [
            TestCase(input=t["stdin"], expected=t["stdout"], id=f"test_{i}")
            for i, t in enumerate(sample["tests"])
        ]

    def get_prompt(self, sample: dict) -> str:
        return sample["problem_description"]

    def get_problem_id(self, sample: dict) -> str:
        return sample["id"]
```

### Custom Reward Shaping

Modify `CodeExecConfig`:

```python
config = CodeExecConfig(
    partial_credit=True,          # Reward = fraction of tests passed
    compile_failure_reward=-0.5,  # Harsher penalty for syntax errors
    stop_on_first_failure=False,  # Run all tests for full feedback
)
```

### Custom Output Verification

For floating-point comparisons:

```python
from ludic.envs.code_exec.adapters import FloatTolerantVerifier

verifier = FloatTolerantVerifier(abs_tol=1e-6, rel_tol=1e-6)
runner = StdinStdoutRunner(verifier=verifier)
```

For full API details, see the [Module README](../../src/ludic/envs/code_exec/README.md).

---

## Troubleshooting

### "Docker daemon not running"

```
docker.errors.DockerException: Error while fetching server API version
```

**Solution**: Start Docker Desktop (macOS/Windows) or `sudo systemctl start docker` (Linux).

### Tests timing out

**Symptoms**: Many `TIMEOUT` results, slow training.

**Diagnosis**: Check if problems have expensive test cases.

**Solutions**:
- Increase timeout: `--timeout-per-test 10.0`
- Use batch execution (enabled by default)
- Reduce number of tests: Set `max_tests` in `CodeExecConfig`

### GPU out of memory

**Solutions**:
- Reduce `--batch-size`
- Reduce `--micro-token-budget`
- Enable gradient checkpointing (already on by default)

### Slow sandbox initialization

**Symptoms**: "Starting sandbox pool..." takes 30+ seconds.

**Solutions**:
- Reduce `--sandbox-workers` for initial testing
- Pre-pull images: `docker pull python:3.11-slim`

### Podman-HPC: Image not found on compute node

**Cause**: Images must be migrated to shared storage.

**Solution**:
```bash
podman-hpc pull python:3.11-slim  # Auto-migrates
podman-hpc images  # Verify R/O=true
```

### Network access denied on compute node

**Cause**: HPC compute nodes often lack internet access.

**Solution**: Run `prepare_env.sh` on the login node first to pre-stage all dependencies.

---

## Setup Guide

### Docker (Local Development)

```bash
# Install (macOS)
brew install --cask docker

# Install (Linux)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# Start daemon
open -a Docker  # macOS
sudo systemctl start docker  # Linux

# Verify
docker info && pip install docker>=7.0.0
```

For detailed setup, see [Docker documentation](https://docs.docker.com/get-docker/).

### Podman-HPC (HPC Clusters)

```bash
# Pull and migrate image to shared storage
podman-hpc pull python:3.11-slim

# Verify migration (R/O should be 'true')
podman-hpc images

# Test execution
srun -N 1 podman-hpc run --rm python:3.11-slim python -c "print('hello')"
```

For cluster-specific setup, consult your HPC documentation or [Podman-HPC docs](https://github.com/NERSC/podman-hpc).

### Verifying Your Setup

```bash
# Run integration tests
pytest tests/integration/test_code_exec_docker.py -v

# If tests are skipped, Docker is not accessible
```

---

## See Also

- **Module README**: [src/ludic/envs/code_exec/README.md](../../src/ludic/envs/code_exec/README.md) — API reference, protocols, internals
- **Migration Guide**: [MIGRATION.md](./MIGRATION.md) — Training API changes and migration steps
- **Training Script**: [train_apps.py](./train_apps.py) — Production-ready example

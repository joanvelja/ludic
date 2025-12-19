# Code Execution Training

Train language models on code generation tasks with sandboxed test-driven evaluation.

## TL;DR (Local Development)

```bash
# 1. Start inference server
CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct

# 2. Install dependencies
pip install docker datasets

# 3. Run training (auto-detects Docker)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/code_exec/train_apps.py \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --limit 100 \
    --train-steps 10
```

**Prerequisites:** Docker Desktop running (macOS/Windows) or Docker daemon (Linux). See [Appendix: Docker Setup](#appendix-docker-setup).

## TL;DR (HPC Cluster)

```bash
# 1. Prepare container image (one-time)
podman-hpc pull python:3.11-slim

# 2. Submit Slurm job
sbatch <<'EOF'
#!/bin/bash
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

# Start vLLM server
python -m ludic.inference.vllm_server \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct &
sleep 30

# Run training (auto-detects Podman-HPC in Slurm)
python examples/code_exec/train_apps.py \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --sandbox-workers 12 \
    --train-steps 100
EOF
```

**Prerequisites:** Podman-HPC installed on cluster. See [Appendix: Podman-HPC Setup](#appendix-podman-hpc-setup).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                          │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐   │
│  │   Trainer   │◄──│ BatchSource  │◄──│  RolloutEngine  │   │
│  └─────────────┘   └──────────────┘   └────────┬────────┘   │
│                                                │            │
│                                    ┌───────────▼─────────┐  │
│                                    │ SingleAgentProtocol │  │
│                                    └────────────┬────────┘  │
│                                                 │           │
│                                    ┌────────────▼────────┐  │
│                                    │    CodeExecEnv      │  │
│                                    └────────────┬────────┘  │
└─────────────────────────────────────────────────┼───────────┘
                                                  │
                                     ┌────────────▼────────┐
                                     │    SandboxPool      │
                                     │    (Protocol)       │
                                     └────────────┬────────┘
                                                  │
                           ┌──────────────────────┼──────────────────────┐
                           │                      │                      │
                ┌──────────▼──────────┐ ┌────────▼─────────┐   ┌─────────▼─────────┐
                │  DockerSandboxPool  │ │PodmanHPCSandboxPool│ │  (future: Sing.)  │
                │  (daemon-based)     │ │  (daemonless)     │  │                   │
                │  ┌────┐ ┌────┐      │ │  ┌────┐ ┌────┐    │  │                   │
                │  │ S1 │ │ S2 │ ...  │ │  │ S1 │ │ S2 │... │  │                   │
                │  └────┘ └────┘      │ │  └────┘ └────┘    │  │                   │
                └─────────────────────┘ └───────────────────┘  └───────────────────┘
```

**Components:**

- **CodeExecEnv**: RL environment that executes generated code against test cases
- **SandboxPool**: Pool of persistent containers for fast code execution
- **Backend Selection**: Auto-detects Docker (local) or Podman-HPC (Slurm)
- **APPSTestAdapter**: Extracts test cases from APPS dataset
- **StdinStdoutRunner**: Executes Python code with stdin/stdout testing

**Reward Structure:**

| Event | Reward | Notes |
|-------|--------|-------|
| All tests pass | +1.0 | Complete success |
| Partial pass | 0.0-1.0 | Requires `--partial-credit` flag |
| All tests fail | 0.0 | |
| Compile error | -0.1 | Syntax/parse errors |
| Timeout | -0.05 | Execution exceeded limit |
| Parser bonus | +0.05 | Proper markdown code blocks |

**Key Features:**

- **Persistent containers**: ~700ms → ~17ms per execution (40x faster)
- **LRU caching**: Skip redundant executions (SHA256-based)
- **Async execution**: Concurrent code testing with configurable pool size
- **Multi-backend**: Portable across local/cloud/HPC environments

## Configuration Reference

### Core Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Model name/path |
| `--host` | `127.0.0.1` | vLLM server host |
| `--port` | `8000` | vLLM server port |
| `--train-steps` | `20` | Number of training steps |
| `--batch-size` | `4` | Rollouts per training batch |
| `--group-size` | `4` | GRPO group size (rollouts per prompt) |
| `--train-temperature` | `0.8` | Sampling temperature |

### Dataset Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--split` | `train` | Dataset split (`train`/`test`) |
| `--limit` | None | Max samples to load (None = all) |
| `--difficulty` | None | Filter by difficulty (introductory/interview/competition) |

### Sandbox Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--sandbox-backend` | `auto` | Backend: `auto`, `docker`, `podman-hpc` |
| `--sandbox-workers` | `4` | Container pool size |
| `--python-version` | `3.11` | Python version in sandbox |
| `--timeout-per-test` | `5.0` | Test timeout (seconds) |
| `--partial-credit` | False | Enable fractional rewards |

### Performance Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--concurrency` | `32` | Max parallel rollouts (async tasks) |
| `--sandbox-workers` | `4` | Max parallel code executions |

**Sizing `--sandbox-workers`:**

Each sandbox uses ~0.5-1 CPU core. Recommended: `floor(available_cpus / 2)`

| Environment | CPUs | Recommended Workers |
|-------------|------|-------------------|
| Laptop (M1/M2) | 8-10 | 4 |
| Workstation | 16-32 | 8-16 |
| HPC node | 64-128 | 24-64 |

**Backend Auto-Detection:**

| Environment | Priority |
|-------------|----------|
| In Slurm job | `podman-hpc` → `docker` |
| Outside Slurm | `docker` → `podman-hpc` |

Override with `--sandbox-backend docker` or `--sandbox-backend podman-hpc`.

## Troubleshooting

### Docker daemon not running

```
docker.errors.DockerException: Error while fetching server API version
```

**Solution:**
```bash
# macOS/Windows
open -a Docker  # Start Docker Desktop

# Linux
sudo systemctl start docker
```

### Tests timing out

```bash
# Increase timeout
python train_apps.py --timeout-per-test 10.0
```

### GPU out of memory

```bash
# Reduce batch size (gradient checkpointing enabled by default)
python train_apps.py --batch-size 2
```

### CPU thrashing with many workers

```bash
# Reduce sandbox workers and concurrency
python train_apps.py --sandbox-workers 4 --concurrency 16
```

### Podman-HPC image not found on compute node

```bash
# Images must be migrated to shared storage
podman-hpc pull python:3.11-slim  # Auto-migrates
podman-hpc images  # Verify R/O column shows 'true'
```

### Integration tests skipped

If `pytest tests/integration/test_code_exec_docker.py` shows 21 skipped tests, Docker daemon is not running (only the Python package is installed). See [Appendix: Docker Setup](#appendix-docker-setup).

## Advanced Topics

### Custom Datasets

Implement a `TestAdapter` to use CodeExecEnv with other datasets:

```python
from ludic.envs.code_exec import TestAdapter, TestCase

class MyDatasetAdapter(TestAdapter):
    def extract_tests(self, sample: dict) -> list[TestCase]:
        return [
            TestCase(
                input=sample["test_input"],
                expected=sample["test_output"],
                id="test_0",
            )
        ]

    def format_problem(self, sample: dict) -> str:
        return sample["problem_description"]
```

See [src/ludic/envs/code_exec/README.md](../../src/ludic/envs/code_exec/README.md) for:
- Complete adapter implementation guide
- Custom verifiers and runners
- Caching internals
- Thread safety details
- Protocol specifications

### Direct Backend Usage

```python
from ludic.envs.code_exec import create_sandbox_pool

# Recommended: Use factory
pool = await create_sandbox_pool(
    n_workers=4,
    backend="auto",  # or "docker", "podman-hpc"
    python_version="3.11",
)

# Or import backends directly (see module README)
```

---

## Appendix: Docker Setup

### macOS

```bash
# Install Docker Desktop
brew install --cask docker

# Start daemon
open -a Docker

# Verify
docker info
```

### Linux (systemd)

```bash
# Install
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Start daemon
sudo systemctl start docker
sudo systemctl enable docker  # Auto-start on boot

# Verify
docker info
```

### Linux (no systemd, e.g., WSL2)

```bash
# Install
curl -fsSL https://get.docker.com | sh

# Start manually
sudo dockerd &
# Or: sudo service docker start

# Verify
docker info
```

### Verify Python SDK

```bash
pip install docker>=7.0.0

python -c "
import docker
client = docker.from_env()
print('Docker version:', client.version()['Version'])
print('Ping:', client.ping())
"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCKER_HOST` | `unix:///var/run/docker.sock` | Daemon socket |
| `DOCKER_TLS_VERIFY` | Not set | Enable TLS |
| `DOCKER_CERT_PATH` | `~/.docker` | TLS certificates |

For rootless Docker:
```bash
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
```

---

## Appendix: Podman-HPC Setup

Podman-HPC is a daemonless wrapper around Podman designed for HPC environments (Isambard-AI, BRiCS, etc.).

### Key Commands

```bash
# Pull and auto-migrate to shared storage
podman-hpc pull python:3.11-slim

# Or build locally, then migrate
podman-hpc build . -t myimage
podman-hpc migrate myimage  # Converts to squashfs on $SCRATCH

# Verify migration (R/O column shows 'true')
podman-hpc images

# Run on compute node
srun -N 1 podman-hpc run --rm python:3.11-slim python -c "print('hello')"

# GPU access
srun --gpus=4 podman-hpc run --gpu python:3.11-slim nvidia-smi
```

### Slurm Job Template

```bash
#!/bin/bash
#SBATCH --job-name=code_exec_train
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

# Ensure image is available (idempotent)
podman-hpc pull python:3.11-slim

# Start vLLM server on host
python -m ludic.inference.vllm_server \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct &
sleep 30

# Run training (auto-detects podman-hpc in Slurm)
python examples/code_exec/train_apps.py \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --sandbox-workers 12 \
    --concurrency 32
```

### Container Naming (Slurm Integration)

Containers are auto-named with Slurm job ID to avoid collisions:
- In Slurm: `ludic-sandbox-{SLURM_JOB_ID}-{worker_id}`
- Outside Slurm: `ludic-sandbox-local-{worker_id}`

This allows multiple jobs to run concurrently without conflicts.

### Migrated vs Non-Migrated Images

**Non-migrated images:**
- Stored in `$HOME/.local/share/containers/storage`
- **Not accessible from compute nodes** (home directory may not be mounted)

**Migrated images:**
- Stored as read-only squashfs on `$SCRATCH` (shared storage)
- Accessible from all compute nodes
- Created via `podman-hpc pull` or `podman-hpc migrate`

Always verify migration:
```bash
podman-hpc images
# Look for R/O=true in output
```

---

## Appendix: Alternative Backends (Future)

### Singularity/Apptainer

Available on Isambard-AI and Isambard 3. Not yet integrated with Ludic.

```bash
# Build from Docker Hub
singularity build python_3.11.sif docker://python:3.11-slim

# Run commands
singularity exec python_3.11.sif python -c "print('hello')"

# GPU access
srun --gpus=1 singularity exec --nv pytorch.sif nvidia-smi
```

**Note:** Singularity auto-mounts `$HOME`, so files persist across sessions.

### Rootless Docker

Some clusters allow rootless Docker. Check with admins.

```bash
# Install
curl -fsSL https://get.docker.com/rootless | sh

# Configure
export PATH=$HOME/bin:$PATH
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock

# Start in job
rootlesskit --state-dir=$HOME/.docker/rootlesskit-$SLURM_JOB_ID dockerd \
    --data-root=$HOME/.docker/data-$SLURM_JOB_ID &
sleep 10
```

### Request Docker Access

Some clusters provide Docker on specific partitions:

```bash
#SBATCH --partition=docker-enabled
#SBATCH --constraint=docker
```

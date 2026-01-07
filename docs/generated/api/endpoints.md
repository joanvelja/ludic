# vLLM Server Endpoints

> Auto-generated documentation - Last updated: 2026-01-06

## Overview

Ludic's vLLM integration provides a two-plane weight distribution system:

- **Control Plane (HTTP)**: Announces batches and coordinates timing
- **Data Plane (NCCL)**: Broadcasts tensor data for GPU-to-GPU updates

## Architecture

```
┌─────────────┐
│   Trainer   │
│  (rank=0)   │
└──────┬──────┘
       │ HTTP + NCCL init
       ▼
┌─────────────────────────────────────────────┐
│         vLLM Server (Host Process)          │
│                                             │
│  GET  /health                               │
│  GET  /runtime_version                      │
│  POST /init_communicator                    │
│  POST /update_param_batch                   │
│  POST /reset_prefix_cache                   │
│  POST /close_communicator                   │
└──────┬──────────────────────────────────────┘
       │ collective_rpc()
       ▼
┌──────────────────────────────────────┐
│  vLLM Worker Processes (GPU)         │
│  + WeightSyncWorkerExtension         │
└──────────────────────────────────────┘
```

## Control Plane Endpoints

### GET `/health`

| Field | Value |
|-------|-------|
| Response | `{"status": "ok"}` |
| Purpose | Health check for initialization |

### GET `/runtime_version`

| Field | Value |
|-------|-------|
| Response | `{"version": <int>}` |
| Purpose | Returns current policy version |
| Use Case | Pipeline RL staleness checking |

### POST `/init_communicator`

Initialize NCCL process group.

| Field | Value |
|-------|-------|
| Request | `{"host": str, "port": int, "world_size": int}` |
| Response | `{"status": "ok"}` |
| Must Call Before | Any weight update endpoint |

### POST `/update_param_batch`

High-throughput batched parameter updates.

| Field | Value |
|-------|-------|
| Request | `{"metadata": [{"name": str, "dtype": str, "shape": [...]}], "version": int?}` |
| Response | `{"status": "ok"}` |
| Side Effects | Broadcasts tensors via NCCL, resets prefix cache |

### POST `/reset_prefix_cache`

Clear vLLM's KV cache.

| Field | Value |
|-------|-------|
| Request | `{}` |
| Response | `{"status": "ok"}` |
| When | After weight updates (automatic with `/update_param_batch`) |

### POST `/close_communicator`

Tear down NCCL group.

| Field | Value |
|-------|-------|
| Request | `{}` |
| Response | `{"status": "ok"}` |

## Weight Update Flow

```python
# 1. Initialize communicator (once)
POST /init_communicator
  → Workers join NCCL group

# 2. Update parameters (batched)
POST /update_param_batch
  {
    "metadata": [
      {"name": "model.embed.weight", "dtype": "torch.float32", "shape": [32000, 4096]},
      ...
    ],
    "version": 42
  }
  → Client broadcasts tensors via NCCL
  → Workers receive and load weights
  → Prefix cache resets
  → RUNTIME_VERSION becomes 42

# 3. Confirm update
GET /runtime_version
  → Returns 42
```

## OpenAI-Compatible Endpoints

Standard vLLM endpoints:

- **POST** `/v1/chat/completions` – Chat completion
- **POST** `/v1/completions` – Legacy completion
- **GET** `/v1/models` – List models

## Server Configuration

```bash
python -m ludic.inference.vllm_server \
  --model meta-llama/Llama-2-7b \
  --tensor-parallel-size 2 \
  --port 8000
```

## Client Configuration

```python
from ludic.inference.vllm_client import VLLMChatClient

# Inference only
client = VLLMChatClient(host="localhost", port=8000)

# With weight updates
client = VLLMChatClient(
    host="localhost",
    port=8000,
    group_port=51216,
    enable_weight_updates=True,
    device="cuda:0",
)
```

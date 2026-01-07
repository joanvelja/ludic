# Ludic Inference

> Auto-generated documentation - Last updated: 2026-01-06

## Overview

Ludic abstracts inference behind a `ChatClient` protocol, enabling backend swapping and weight synchronization during training.

## ChatClient Protocol

**Location**: `src/ludic/inference/client.py`

```python
class ChatClient(Protocol):
    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]: ...

    def sync_weights(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        version: Optional[str | int] = None,
    ) -> str: ...
```

## Request Types

### ChatCompletionRequest

```python
@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[Message]
    sampling: SamplingConfig
    return_: ReturnSpec
    extensions: Optional[Dict[str, Any]] = None
```

### SamplingConfig

```python
@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
```

### ReturnSpec

```python
@dataclass
class ReturnSpec:
    return_token_ids: bool = False
    return_chosen_logprobs: bool = False
    top_logprobs_k: int = 0
```

For training: `return_token_ids=True` ensures drift-free RL.

## VLLMChatClient

**Location**: `src/ludic/inference/vllm_client.py`

```python
class VLLMChatClient(ChatClient):
    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
    ): ...
```

### With Weight Updates

```python
client = VLLMChatClient(
    model_name="Qwen/Qwen2.5-1.5B",
    api_base="http://localhost:8000/v1",
    group_port=51216,
    enable_weight_updates=True,
    device="cuda:0",
)
```

## Starting vLLM Server

```python
from ludic.inference import start_vllm_server, wait_for_vllm_health

process = start_vllm_server(
    model="Qwen/Qwen2.5-1.5B",
    gpu_memory_utilization=0.9,
)

await wait_for_vllm_health(api_base="http://localhost:8000/v1")
```

## InferenceSpec

High-level config passed to `agent.act()`:

```python
@dataclass
class InferenceSpec:
    sampling: Optional[SamplingConfig] = None
    return_: ReturnSpec = field(default_factory=ReturnSpec)
    extensions: Optional[Dict[str, Any]] = None
```

## Tool Calling

```python
request = ChatCompletionRequest(
    model="gpt-4",
    messages=[...],
    sampling=SamplingConfig(temperature=0.0),
    return_=ReturnSpec(return_token_ids=True),
    extensions={
        "tools": ToolRequest(
            tools=[{
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object", "properties": {...}}
                }
            }],
            tool_choice="auto"
        )
    }
)
```

## Best Practices

1. **Use `return_token_ids=True`** for training
2. **Poll for health** before starting training
3. **Set reasonable timeouts** (e.g., 300s)
4. **Use temperature=0** for eval, >0 for training

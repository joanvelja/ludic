# Ludic Internal Protocols

> Auto-generated documentation - Last updated: 2026-01-06

## Environment (LudicEnv)

**Location**: `src/ludic/envs/env.py`

Multi-agent environment kernel.

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `agent_ids` | — | `List[AgentID]` | All agent roles |
| `active_agents` | — | `List[AgentID]` | Agents acting this step |
| `reset(seed)` | `seed: Optional[int]` | `Dict[AgentID, (Obs, Info)]` | Initialize episode |
| `step(actions)` | `Dict[AgentID, Action]` | `Dict[AgentID, StepOutcome]` | Process actions |

### SingleAgentEnv

Convenience wrapper for single-agent tasks:

```python
class SingleAgentEnv(LudicEnv[str, str, str]):
    def env_reset(self, *, seed=None) -> (Obs, Info): ...
    def env_step(self, action: str) -> StepOutcome: ...
    def env_current_obs(self) -> Obs: ...
```

## ChatClient Protocol

**Location**: `src/ludic/inference/client.py`

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `complete(request)` | `ChatCompletionRequest` | `(ChatResponse, Dict)` | Async inference |
| `sync_weights(params, timeout_s, version)` | `Mapping[str, Tensor]` | `str` | Push weights |

### VersionedClient

Extends ChatClient with policy versioning:

```python
class VersionedClient(ChatClient, Protocol):
    async def get_policy_version(self) -> int: ...
```

## Context Strategy

**Location**: `src/ludic/context/base.py`

| Method | Description |
|--------|-------------|
| `reset(system_prompt)` | Clear history |
| `on_env_reset(obs, info)` | After env.reset() |
| `on_before_act()` | Return messages for LLM |
| `on_after_act(response)` | Record assistant response |
| `on_after_step(obs, info)` | Record new observation |

### Implementations

- **FullDialog**: Keeps entire history
- **TruncatedThinkingContext**: Truncates `<think>` blocks

## Parser

**Location**: `src/ludic/parsers.py`

```python
Parser = Callable[[str], ParseResult]

@dataclass(frozen=True)
class ParseResult:
    action: Optional[str]  # Parsed action or None
    reward: float          # Format reward/penalty
    obs: Optional[str]     # Feedback on failure
```

### Factories

- `xml_parser(tag, exact, success_reward, error_reward)`
- `think_prefix_parser()`
- `boxed_parser()`
- `compose_parsers(*parsers)`

## Credit Assigner

**Location**: `src/ludic/training/credit_assignment.py`

```python
class CreditAssigner(Protocol):
    def compute(self, rollouts: List[Rollout]) -> Dict[RolloutStepKey, float]: ...
```

| Implementation | Formula |
|----------------|---------|
| `MonteCarloReturn` | G_t = Σ γ^k r_{t+k} |
| `GroupNormalizedReturn` | A_i = R_i - mean(R_group) |
| `HybridNormalizedReturn` | (R - mean) / (std + ε) |
| `ConstantCredit` | weight = value |

## Loss Functions

**Location**: `src/ludic/training/loss.py`

```python
class Loss(Protocol):
    def compute(self, logits: Tensor, batch: Mapping) -> (Tensor, Dict): ...
```

| Loss | Algorithm |
|------|-----------|
| `ReinforceLoss` | -E[weight * log π(a\|s)] |
| `ReinforceBaselineLoss` | With batch-mean baseline |
| `TokenClippedSurrogateLoss` | PPO-style clipping |
| `CISPOLoss` | Clipped IS-weight |
| `TokenKLLoss` | KL penalty |
| `CompositeLoss` | Weighted combination |

## BatchSource

**Location**: `src/ludic/training/types.py`

```python
class BatchSource(Protocol):
    async def next_batch(self) -> SAWBatch: ...
```

| Implementation | Use Case |
|----------------|----------|
| `RolloutBatchSource` | Online RL (sync) |
| `OfflineBatchSource` | SFT / offline RL |
| `PipelineBatchSource` | Distributed (Redis) |

## PolicyPublisher

**Location**: `src/ludic/distributed/interfaces.py`

```python
class PolicyPublisher(Protocol):
    def publish(self, state_dict: Mapping[str, Tensor], version: Optional[int]) -> None: ...
```

### Implementations

- **BroadcastPolicyPublisher**: HTTP + NCCL two-plane
- **Rank0OnlyPublisher**: Wrapper for multi-GPU

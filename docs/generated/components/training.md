# Ludic Training System

> Auto-generated documentation - Last updated: 2026-01-06

## Architecture

```
Rollouts → Credit Assignment → SAWBatch → Trainer → Publisher
```

Separation makes algorithm swapping easy without rewriting the stack.

## Core Types

### RolloutRequest

```python
@dataclass
class RolloutRequest:
    env: EnvSpec
    protocol: ProtocolSpec
    env_seed: Optional[int]
    sampling_seed: Optional[int]
    inference: Optional[InferenceSpec]
    num_episodes: int
    meta: Dict[str, Any]
```

### SAWItem (State-Action-Weight)

```python
@dataclass
class SAWItem:
    input_ids: List[int]        # [state_ids + action_ids]
    attention_mask: List[int]   # 1/0 mask
    action_mask: List[int]      # 0 on state, 1 on action
    weight: float               # Credit weight
    meta: Dict[str, Any]
    attachments: SampleAttachments
```

## RolloutEngine

**Location**: `src/ludic/training/batching/rollout_engine.py`

```python
class RolloutEngine:
    async def generate_rollouts(
        self,
        requests: Sequence[RolloutRequest],
        max_steps: int,
        concurrency: int = 8,
    ) -> List[Rollout]: ...

    async def generate_batch(
        self,
        requests: Sequence[RolloutRequest],
        algorithm: RLAlgorithm,
        sample_filter: Optional[SampleFilter] = None,
    ) -> SAWBatch: ...
```

## Credit Assignment

**Location**: `src/ludic/training/credit_assignment.py`

| Assigner | Formula | Use Case |
|----------|---------|----------|
| `MonteCarloReturn` | G_t = Σ γ^k r_{t+k} | REINFORCE |
| `GroupNormalizedReturn` | A_i = R_i - mean(R) | GRPO |
| `HybridNormalizedReturn` | (R - mean) / std | ScaleRL |
| `ConstantCredit` | weight = 1.0 | SFT |

## Loss Functions

**Location**: `src/ludic/training/loss.py`

| Loss | Algorithm |
|------|-----------|
| `ReinforceLoss` | -E[w * log π(a\|s)] |
| `ReinforceBaselineLoss` | With batch-mean baseline |
| `TokenClippedSurrogateLoss` | PPO-style token clipping |
| `CISPOLoss` | Clipped IS-weight |
| `TokenKLLoss` | KL penalty |
| `MaskedCausalLMCrossEntropyLoss` | SFT NLL |
| `CompositeLoss` | Weighted combination |

## RLAlgorithm

**Location**: `src/ludic/training/algorithm.py`

```python
@dataclass
class RLAlgorithm:
    name: str
    credit_assigner: CreditAssigner
    loss: Loss
    preprocess: Optional[PreprocessFn] = None
```

### Presets

```python
make_reinforce(gamma=1.0)                    # Monte Carlo + REINFORCE
make_grpo(group_size=8)                      # GroupNorm + ClippedSurrogate
make_sft()                                   # Constant + CrossEntropy
make_scalerl(group_size=8, filter_zvp=True)  # Hybrid + CISPO
make_cispo(group_size=8, kl_coeff=0.0)       # GroupNorm + CISPO
```

## Trainer

**Location**: `src/ludic/training/trainer.py`

```python
class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        algo: RLAlgorithm,
        batch_source: BatchSource,
        publisher: Optional[PolicyPublisher] = None,
        cfg: TrainerConfig = TrainerConfig(),
    ): ...

    async def train(self, num_steps: Optional[int] = None): ...
```

Training loop:
1. `batch_source.next_batch()` → macro-batch
2. Split by token budget → micro-batches
3. For each micro-batch: collate, loss, backward
4. `optimizer.step()`
5. `publisher.publish(weights)`

## Batch Sources

| Source | Use Case |
|--------|----------|
| `RolloutBatchSource` | Online RL (sync) |
| `OfflineBatchSource` | SFT / offline RL |
| `PipelineBatchSource` | Distributed (Redis) |

## Sample Filters

```python
from ludic.training.filters import (
    drop_truncated,
    drop_incomplete_completions,
    drop_parse_errors,
    combine,
)

sample_filter = combine(
    drop_truncated(),
    drop_incomplete_completions(),
)
```

## Example Training Loop

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

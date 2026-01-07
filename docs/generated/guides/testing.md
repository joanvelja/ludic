# Ludic Testing Guide

> Auto-generated documentation - Last updated: 2026-01-06

## Overview

- **Framework**: pytest with pytest-asyncio
- **Test Files**: 38 files, ~9,400 lines
- **Test Functions**: ~380 tests

## Test Organization

```
tests/
├── conftest.py              # Fixtures (vllm_server, mock_agent, etc.)
├── _mocks.py                # MockClient, MockAgent, MockEnv
├── test_parsers.py          # Parser functionality
├── test_interaction.py      # InteractionProtocol
├── test_rollout_engine.py   # RolloutEngine
├── test_trainer.py          # Trainer optimization
├── test_credit_assignment.py # Credit assigners
├── test_loss.py             # Loss functions
├── test_react_agent.py      # ReAct pattern
├── test_tool_agent.py       # Tool calling
├── test_code_exec_*.py      # Code execution (8 files)
└── integration/
    ├── test_grpo_e2e.py     # End-to-end GRPO
    ├── test_vllm_client.py  # vLLM integration
    └── test_code_exec_docker.py
```

## Running Tests

```bash
# All unit tests (excludes gpu/diagnostic)
pytest

# Integration tests
pytest -m integration

# GPU tests
pytest -m gpu

# All tests
pytest -m ""

# Specific file
pytest tests/test_parsers.py -v

# With coverage
pytest --cov=src/ludic
```

## Test Markers

| Marker | Purpose |
|--------|---------|
| `integration` | Multi-component tests |
| `gpu` | Requires GPU |
| `diagnostic` | Informational only |
| `asyncio` | Async tests |

## Configuration

From `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = ["-m", "not diagnostic"]
markers = [
    "integration",
    "gpu",
    "diagnostic",
]
testpaths = ["tests"]
```

## Test Utilities

### MockClient

```python
from tests._mocks import MockClient

client = MockClient(text="fixed response")
response, info = await client.complete(request)
# response.text == "fixed response"
```

### MockEnv

```python
from tests._mocks import MockEnv

env = MockEnv(max_steps=3, target="1")
obs, info = env.reset()
outcome = env.step("1")
# outcome.reward == 1.0, outcome.terminated == True
```

### MockAgent

```python
from tests._mocks import MockAgent

agent = MockAgent(client=MockClient(text="1"))
action, info = await agent.act(obs)
```

## Test Patterns

### Async Tests

```python
@pytest.mark.asyncio
async def test_happy_path():
    env = MockEnv()
    agent = MockAgent(client=MockClient(text="1"))
    protocol = SingleAgentProtocol(agent=agent)

    rollouts = await protocol.run(env=env, max_steps=5)

    assert len(rollouts) == 1
    assert rollouts[0].steps[-1].terminated is True
```

### Parametrized vLLM

```python
@pytest.mark.parametrize("vllm_server", [{"enable_tools": True}], indirect=True)
@pytest.mark.integration
async def test_tool_calling(vllm_server, vllm_client):
    ...
```

### Test Classes

```python
class TestCreditAssigner:
    def test_monte_carlo_return(self):
        assigner = MonteCarloReturn(gamma=1.0)
        weights = assigner.compute(rollouts)
        assert weights[(rollout.id, 0)] == expected

    def test_group_normalized_return(self):
        ...
```

## Coverage by Component

| Component | Test Files | Focus |
|-----------|-----------|-------|
| Parsers | test_parsers.py | Format extraction, rewards |
| Agents | test_react_agent.py, test_tool_agent.py | LLM integration |
| Training | test_trainer.py, test_loss.py, test_credit_assignment.py | Optimization |
| Code Exec | test_code_exec_*.py (8 files) | Sandboxing |
| Rollouts | test_rollout_engine.py | Batch generation |

## Writing New Tests

### Unit Test Template

```python
import pytest
from ludic.training.credit_assignment import MonteCarloReturn
from tests._mocks import make_rollout

def test_monte_carlo_basic():
    """MonteCarloReturn computes discounted returns."""
    rollout = make_rollout(rewards=[1.0, 0.5, 0.25])
    assigner = MonteCarloReturn(gamma=1.0)

    weights = assigner.compute([rollout])

    assert weights[(rollout.id, 0)] == 1.75  # 1.0 + 0.5 + 0.25
    assert weights[(rollout.id, 1)] == 0.75  # 0.5 + 0.25
    assert weights[(rollout.id, 2)] == 0.25
```

### Integration Test Template

```python
@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_grpo_e2e(vllm_server, vllm_client):
    """End-to-end GRPO training works."""
    # Setup
    engine = RolloutEngine(...)

    # Execute
    batch = await engine.generate_batch(requests)

    # Verify
    assert len(batch.items) > 0
    assert all(item.weight != 0 for item in batch.items)
```

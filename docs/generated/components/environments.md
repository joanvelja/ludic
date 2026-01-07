# Ludic Environments

> Auto-generated documentation - Last updated: 2026-01-06

## Overview

Ludic environments follow the "multi-agent by default" kernel principle. The canonical interface is `LudicEnv`, supporting single-agent, turn-based, and simultaneous-move interactions.

## Core Abstraction: LudicEnv

**Location**: `src/ludic/envs/env.py`

```python
class LudicEnv(ABC, Generic[AgentID, ObsType, ActionType]):
    @property
    def agent_ids(self) -> List[AgentID]:
        """All agent roles in this environment."""

    @property
    def active_agents(self) -> List[AgentID]:
        """Which agents should provide actions this step."""

    def reset(self, *, seed: Optional[int] = None) -> Dict[AgentID, Tuple[ObsType, Info]]:
        """Initialize environment; return initial obs/info per agent."""

    def step(self, actions: Dict[AgentID, ActionType]) -> Dict[AgentID, StepOutcome]:
        """Process actions from active_agents; return outcomes for all."""
```

**Key Design**: The environment is agnostic to agents. Actions and outcomes use dicts, enabling turn-based, simultaneous-move, or mixed protocols.

## SingleAgentEnv

**Location**: `src/ludic/envs/single_agent_env.py`

Convenience wrapper for single-agent tasks:

```python
class SingleAgentEnv(LudicEnv[str, str, str]):
    @property
    def suggested_sysprompt(self) -> Optional[str]:
        """Optional default system prompt."""

    @abstractmethod
    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        """Your reset logic."""

    @abstractmethod
    def env_step(self, action: str) -> StepOutcome:
        """Your step logic."""

    @abstractmethod
    def env_current_obs(self) -> Observation:
        """Get current state."""
```

## Specialized Environments

### DatasetQAEnv

**Location**: `src/ludic/envs/dataset_qa_env.py`

One-shot QA on dataset samples:

```python
DatasetQAEnv:
├── dataset: iterable of {question, answer, ...}
├── verifier: Callable[[action, ground_truth], (reward, is_correct)]
├── max_steps: int (typically 1)
└── Produces: StepOutcome with reward from verifier
```

### CodeExecEnv

**Location**: `src/ludic/envs/code_exec/`

Code execution with sandboxing:

- Async support: `env_reset()`, `env_step()`
- Sandbox backends: Docker, Podman
- Execution contexts: Python, Shell
- Reward: Test case pass/fail

## Interaction Protocols

**Location**: `src/ludic/interaction/`

The environment does **not** own rollout generation. An `InteractionProtocol` does:

```python
class InteractionProtocol(ABC):
    async def run(
        self,
        *,
        env: LudicEnv,
        max_steps: int,
        env_seed: Optional[int] = None,
        sampling_seed: Optional[int] = None,
    ) -> List[Rollout]:
        """Execute one full episode."""
```

### SingleAgentProtocol

Standard synchronous loop:

1. `env.reset(seed=env_seed)`
2. `ctx.on_env_reset(obs, info)`
3. Loop until done:
   - `messages = ctx.on_before_act()`
   - `response = agent.act(inference_spec)`
   - Parse action
   - If parse failure: synthetic step
   - Else: `env.step(action)`

**Parser Failures**: Handled inside the protocol with synthetic Steps.

## Environment Registration

```python
env_registry = {
    "tic_tac_toe": TicTacToeEnv,
    "gsm8k": GSM8KEnv,
    "code_exec": CodeExecEnv,
}

request = RolloutRequest(
    env=EnvSpec(kind="gsm8k", kwargs={"split": "train"}),
    ...
)
```

## Truncation Semantics

- **`terminated`**: True terminal state (env says done)
- **`truncated`**: Time limit / cutoff (max_steps)
- **`finish_reason`**: LLM generation stop (separate)

## Example: Simple Math Env

```python
from ludic.envs import SingleAgentEnv
from ludic.types import StepOutcome
import numpy as np

class SimpleAddEnv(SingleAgentEnv):
    def __init__(self):
        super().__init__()
        self.a = 0
        self.b = 0

    @property
    def suggested_sysprompt(self) -> str:
        return "Solve the math problem."

    def env_reset(self, *, seed=None) -> tuple[str, dict]:
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, 100)
        self.b = rng.randint(1, 100)
        return f"What is {self.a} + {self.b}?", {}

    def env_step(self, action: str) -> StepOutcome:
        try:
            answer = int(action.strip())
            correct = (answer == self.a + self.b)
            reward = 1.0 if correct else 0.0
        except ValueError:
            reward = 0.0

        return StepOutcome(
            obs=f"You answered {action}.",
            reward=reward,
            truncated=True,
            terminated=True,
        )

    def env_current_obs(self) -> str:
        return f"What is {self.a} + {self.b}?"
```

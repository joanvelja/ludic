# Ludic Agents

> Auto-generated documentation - Last updated: 2026-01-06

## Overview

An **Agent** is a stateful LLM wrapper coupling inference, context management, and parsing.

**Location**: `src/ludic/agents/base_agent.py`

## Core Agent Class

```python
class Agent:
    def __init__(
        self,
        *,
        client: ChatClient,
        model: str,
        ctx: ContextStrategy,
        parser: Parser,
        reject_incomplete_completions: bool = True,
        incomplete_completion_penalty: float = -0.1,
        incomplete_completion_feedback: str = "...",
    ):
        """
        Args:
            client: Inference backend
            model: Model name
            ctx: Context strategy (memory)
            parser: Action decoder
            reject_incomplete_completions: Treat finish_reason="length" as failure
        """
```

## Key Methods

### act()

```python
async def act(
    self,
    obs: str,
    inference: Optional[InferenceSpec] = None,
) -> (str, Dict[str, Any])
```

Flow:
1. Prepare prompt via `ctx.on_before_act()`
2. Call `client.complete(request)`
3. Parse response via `parser`
4. Update context via `ctx.on_after_act()`
5. Return (action, info)

### reset()

```python
def reset(self, system_prompt: Optional[str] = None)
```

Clears conversation history.

## Context Strategies

**Location**: `src/ludic/context/base.py`

```python
class ContextStrategy(ABC):
    def reset(self, *, system_prompt: Optional[str] = None): ...
    def on_env_reset(self, obs: Observation, info: Info) -> None: ...
    def on_before_act(self) -> List[Message]: ...
    def on_after_act(self, response: ChatResponse) -> None: ...
    def on_after_step(self, next_obs: Observation, info: Info) -> None: ...
```

### FullDialog

Keeps entire conversation history. Simple but can grow unbounded.

### TruncatedThinkingContext

Truncates `<think>...</think>` blocks in prompts while keeping full history internally. Reduces token usage for chain-of-thought.

## Parsers

**Location**: `src/ludic/parsers.py`

```python
Parser = Callable[[str], ParseResult]

@dataclass(frozen=True)
class ParseResult:
    action: Optional[str]  # Parsed action or None
    reward: float          # Format reward/penalty
    obs: Optional[str]     # Feedback on failure
```

### Built-in Parsers

```python
# XML tag extraction
xml_parser(tag="move", exact=True, success_reward=0.1, error_reward=-1.0)

# Think prefix requirement
think_prefix_parser(success_reward=0.1, error_reward=-1.0)

# Math boxed format
boxed_parser(success_reward=0.1, error_reward=-1.0)

# Compose multiple
compose_parsers(think_prefix_parser(), xml_parser("move"))
```

## Tool-Enabled Agents

### ToolAgent

**Location**: `src/ludic/agents/tool_agent.py`

OpenAI-compatible tool calling:

```python
class ToolAgent(Agent):
    def __init__(
        self,
        tools: Sequence[Callable],
        **agent_kwargs
    ):
        """
        tools: Python callables with type hints.
               Schemas are auto-generated.
        """
```

Usage:

```python
def search_web(query: str) -> str:
    """Search the web."""
    return results

agent = ToolAgent(
    tools=[search_web],
    client=client,
    model="gpt-4",
    ctx=FullDialog(),
    parser=lambda x: ParseResult(action=x, reward=0, obs=None),
)
```

### ReActAgent

**Location**: `src/ludic/agents/react_agent.py`

Multi-step reasoning with tool loops:

```python
class ReActAgent(ToolAgent):
    def __init__(
        self,
        tools: Sequence[Callable],
        max_react_steps: int = 10,
        **agent_kwargs
    ):
        """Reason → act → observe → repeat."""
```

## Example: Game Agent

```python
from ludic.agents import Agent
from ludic.context import FullDialog
from ludic.parsers import xml_parser
from ludic.inference import VLLMChatClient

def create_chess_agent(client: VLLMChatClient, model: str) -> Agent:
    system_prompt = """You are a chess expert.
    Respond with: <move>e2e4</move>"""

    return Agent(
        client=client,
        model=model,
        ctx=FullDialog(system_prompt=system_prompt),
        parser=xml_parser(tag="move", exact=True),
    )
```

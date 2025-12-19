# CodeExecEnv Module

A sandboxed code execution environment for reinforcement learning on code generation tasks.

## Module Structure

```
code_exec/
├── __init__.py           # Public API exports
├── types.py              # Data types (TestCase, TestResult, BatchTestResult, etc.)
├── sandbox.py            # Sandbox/SandboxPool protocols
├── docker_sandbox.py     # Docker-based sandbox implementation + LRU cache
├── runners.py            # Code execution strategies (StdinStdoutRunner)
├── env.py                # CodeExecEnv (main RL environment)
└── adapters/
    ├── base.py           # TestAdapter, OutputVerifier protocols
    └── apps.py           # APPS dataset adapter
```

## Core Abstractions

### Sandbox Protocol

```python
class Sandbox(Protocol):
    """Single sandboxed execution environment."""

    async def execute(
        self,
        code: str,
        stdin: str = "",
        timeout_s: float = 5.0,
    ) -> ExecutionResult:
        """Execute code and return result."""
        ...
```

### SandboxPool Protocol

```python
class SandboxPool(Protocol):
    """Pool of reusable sandboxes with caching."""

    async def checkout(self, timeout_s: float = 30.0) -> Sandbox:
        """Get a sandbox from the pool."""
        ...

    async def release(self, sandbox: Sandbox) -> None:
        """Return sandbox to pool."""
        ...

    def cache_get(self, code_hash: str, tests_hash: str) -> BatchTestResult | None:
        """Check cache for previous results."""
        ...

    def cache_put(self, code_hash: str, tests_hash: str, result: BatchTestResult) -> None:
        """Store result in cache."""
        ...
```

### TestAdapter Protocol

```python
class TestAdapter(Protocol):
    """Extracts test cases from dataset samples."""

    def extract_tests(self, sample: dict[str, Any]) -> list[TestCase]:
        """Extract test cases from a sample."""
        ...

    def format_problem(self, sample: dict[str, Any]) -> str:
        """Format problem description for prompt."""
        ...
```

### CodeRunner Protocol

```python
class CodeRunner(Protocol):
    """Executes code against test cases."""

    async def run_tests(
        self,
        code: str,
        tests: list[TestCase],
        sandbox: Sandbox,
        config: CodeExecConfig,
    ) -> BatchTestResult:
        """Run all tests and return results."""
        ...
```

## Usage

### Basic Setup

```python
from ludic.envs.code_exec import (
    CodeExecEnv,
    CodeExecConfig,
    DockerSandboxPool,
    DockerSandboxConfig,
)
from ludic.envs.code_exec.adapters.apps import APPSTestAdapter

# Create sandbox pool
pool_config = DockerSandboxConfig(
    python_version="3.11",
    memory_limit="256m",
    cpu_quota=50000,
    network_disabled=True,
)
pool = DockerSandboxPool(n_workers=4, config=pool_config)
await pool.start()

# Create environment
env_config = CodeExecConfig(
    timeout_per_test_s=5.0,
    stop_on_first_failure=True,
    partial_credit=False,
)
env = CodeExecEnv(
    sample={"question": "...", "inputs": [...], "outputs": [...]},
    sandbox_pool=pool,
    test_adapter=APPSTestAdapter(),
    config=env_config,
)

# Run episode
obs, info = await env.env_reset()
outcome = await env.env_step("print(input())")

# Cleanup
await pool.shutdown()
```

### With SingleAgentProtocol

The protocol automatically detects async environments:

```python
from ludic.interaction import SingleAgentProtocol
from ludic.agent import Agent

protocol = SingleAgentProtocol(agent=agent)
rollouts = await protocol.run(env=env, max_steps=3)
```

## Configuration

### CodeExecConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_per_test_s` | `float` | `5.0` | Timeout per test case |
| `stop_on_first_failure` | `bool` | `True` | Stop after first failed test |
| `compile_first` | `bool` | `True` | Check syntax before running |
| `partial_credit` | `bool` | `False` | Reward based on pass fraction |
| `compile_failure_reward` | `float` | `-0.1` | Reward for syntax errors |
| `timeout_reward` | `float` | `-0.05` | Reward for timeout |
| `use_cache` | `bool` | `True` | Enable result caching |

### DockerSandboxConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `python_version` | `str` | `"3.11"` | Python version in container |
| `memory_limit` | `str` | `"256m"` | Container memory limit |
| `cpu_quota` | `int` | `50000` | CPU quota (50% of one core) |
| `network_disabled` | `bool` | `True` | Disable container networking |

## Implementing Custom Adapters

```python
from ludic.envs.code_exec import TestAdapter, TestCase, ExactMatchVerifier

class MyDatasetAdapter(TestAdapter):
    def __init__(self):
        self._verifier = ExactMatchVerifier(strip=True, normalize_whitespace=True)

    def extract_tests(self, sample: dict) -> list[TestCase]:
        tests = []
        for i, (inp, out) in enumerate(zip(sample["inputs"], sample["outputs"])):
            tests.append(TestCase(input=inp, expected=out, id=f"test_{i}"))
        return tests

    def format_problem(self, sample: dict) -> str:
        return sample["problem_statement"]

    @property
    def verifier(self) -> ExactMatchVerifier:
        return self._verifier
```

## Result Types

### TestResult

```python
@dataclass
class TestResult:
    test_case: TestCase
    passed: bool
    actual: str | None
    execution: ExecutionResult
    error_message: str | None = None
```

### BatchTestResult

```python
@dataclass
class BatchTestResult:
    results: list[TestResult]
    code_hash: str
    tests_hash: str

    @property
    def passed_count(self) -> int: ...

    @property
    def total_count(self) -> int: ...

    @property
    def all_passed(self) -> bool: ...

    @property
    def pass_rate(self) -> float: ...
```

## Caching

The `DockerSandboxPool` includes an LRU cache to avoid re-executing identical code:

```python
pool = DockerSandboxPool(
    n_workers=4,
    config=config,
    cache_size=10000,  # Max cached results
)

# Check cache stats
print(pool.cache_stats)
# {'hits': 150, 'misses': 50, 'size': 200, 'max_size': 10000}
```

Cache keys are computed from:
- SHA256 hash of the code
- SHA256 hash of serialized test cases

## Thread Safety

- `LRUCache`: Thread-safe via `threading.Lock`
- `DockerSandboxPool`: Async-safe via `asyncio.Queue`
- `CodeExecEnv`: Not thread-safe (one instance per rollout)

## Dependencies

**Required:**
- `docker>=7.0.0` - Docker Python SDK

**Optional (for specific adapters):**
- `datasets` - HuggingFace datasets for APPS

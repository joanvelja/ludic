# CodeExecEnv Implementation Plan

## Overview

A modular, async-first code execution environment for RL on code generation tasks. The design separates concerns into composable abstractions so that different datasets (APPS, HumanEval, LeetCode, etc.) can be supported by swapping adapters rather than rewriting the env.

**Key design goals:**
- Sandbox never becomes a training bottleneck (fully async, pooled)
- Rich execution metadata for RL signal (compile status, timing, memory, etc.)
- Caching to avoid redundant execution of the same code/tests
- Configurable Python versions per dataset
- Dataset-agnostic via adapters

---

## Design Principles

1. **Pool at application layer** — Sandboxes are long-lived resources managed outside the env; injected via factory closure.
2. **Tests via sample data** — Test cases flow through `EnvSpec.kwargs["sample"]`, just like `DatasetQAEnv`.
3. **Protocols for variation points** — Use `typing.Protocol` for all abstractions that vary across datasets/runtimes.
4. **Dataset-specific modules** — Each dataset format (APPS, HumanEval, etc.) gets its own adapter module, not kwargs bloat.
5. **Async-first** — All sandbox operations are async to maximize throughput.
6. **Caching at sandbox level** — Code/test pairs are cached to avoid redundant execution.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Application Layer                              │
│                                                                         │
│  pool = DockerSandboxPool(                                              │
│      n_workers=32,                                                      │
│      python_version="3.11",  # configurable per dataset                 │
│      cache_size=10_000,      # LRU cache for code/test pairs            │
│  )                                                                      │
│  await pool.start()                                                     │
│                                                                         │
│  # APPS-specific adapters                                               │
│  from ludic.envs.code_exec.adapters.apps import APPSTestAdapter         │
│                                                                         │
│  def build_env(**kw):                                                   │
│      return CodeExecEnv(                                                │
│          sandbox_pool=pool,                                             │
│          test_adapter=APPSTestAdapter(),                                │
│          **kw                                                           │
│      )                                                                  │
│                                                                         │
│  engine = RolloutEngine(env_registry={"code_exec": build_env}, ...)     │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            CodeExecEnv                                  │
│  (SingleAgentEnv)                                                       │
│                                                                         │
│  Responsibilities:                                                      │
│    • Expose problem prompt as observation                               │
│    • Accept code as action                                              │
│    • Coordinate sandbox checkout → compile → run → release              │
│    • Compute reward from test results + compilation status              │
│    • Surface rich metadata for RL signal                                │
│                                                                         │
│  Dependencies (injected):                                               │
│    • SandboxPool (async checkout/release)                               │
│    • TestAdapter (extract tests from sample)                            │
│    • CodeRunner (execute code in sandbox)                               │
│    • OutputVerifier (compare outputs) [optional, has default]           │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│    SandboxPool      │ │    TestAdapter      │ │    CodeRunner       │
│    (Protocol)       │ │    (Protocol)       │ │    (Protocol)       │
│                     │ │                     │ │                     │
│ • checkout()        │ │ • get_tests(sample) │ │ • run(sandbox,      │
│ • release(handle)   │ │   -> List[TestCase] │ │        code, test)  │
│ • shutdown()        │ │                     │ │   -> ExecutionResult│
│ • get_cached()      │ │                     │ │                     │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  DockerSandboxPool  │ │  APPSTestAdapter    │ │ stdin_stdout_runner │
│  (async impl)       │ │  HumanEvalAdapter   │ │ function_call_runner│
│  + ExecutionCache   │ │  ...                │ │ pytest_runner       │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

---

## Core Abstractions

### 1. Execution Result & Metadata (`src/ludic/envs/code_exec/types.py`)

Rich metadata is critical for RL signal. The execution result captures everything useful about a code execution.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class CompileStatus(Enum):
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"

class RunStatus(Enum):
    SUCCESS = "success"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    KILLED = "killed"

@dataclass
class CompileResult:
    """Result of compiling/syntax-checking code."""
    status: CompileStatus
    error_message: Optional[str] = None
    error_line: Optional[int] = None
    error_column: Optional[int] = None
    duration_ms: float = 0.0

@dataclass
class ExecutionResult:
    """
    Rich result of running code in a sandbox.

    All fields are RL-relevant metadata that can be used for:
      - Reward shaping (compile errors vs runtime errors vs wrong answer)
      - Curriculum learning (filter by execution characteristics)
      - Analysis (understanding failure modes)
    """
    # Compilation phase
    compile_result: CompileResult

    # Execution phase (only populated if compilation succeeded)
    run_status: Optional[RunStatus] = None
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    return_value: Optional[str] = None  # for function-based testing

    # Timing (all in milliseconds)
    compile_duration_ms: float = 0.0
    run_duration_ms: float = 0.0
    total_duration_ms: float = 0.0

    # Resource usage
    peak_memory_bytes: Optional[int] = None
    cpu_time_ms: Optional[float] = None

    # Cache info
    cache_hit: bool = False
    cache_key: Optional[str] = None

    @property
    def compiled(self) -> bool:
        return self.compile_result.status == CompileStatus.SUCCESS

    @property
    def succeeded(self) -> bool:
        return self.compiled and self.run_status == RunStatus.SUCCESS

@dataclass
class TestCase:
    """
    A single test case.

    The interpretation depends on the CodeRunner:
      - stdin/stdout: input is stdin string, expected is stdout string
      - function call: input is (args, kwargs), expected is return value
    """
    input: Any
    expected: Any
    id: str = ""
    weight: float = 1.0
    timeout_s: Optional[float] = None  # per-test timeout override
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestResult:
    """Result of running a single test case."""
    test_case: TestCase
    passed: bool
    actual: Any
    execution: ExecutionResult
    comparison_details: Optional[str] = None  # why it failed

@dataclass
class BatchTestResult:
    """Result of running all tests for a code submission."""
    results: List[TestResult]
    code_hash: str
    tests_hash: str

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        return self.passed_count == self.total_count

    @property
    def first_failure(self) -> Optional[TestResult]:
        for r in self.results:
            if not r.passed:
                return r
        return None

    @property
    def compile_failed(self) -> bool:
        """True if code failed to compile (before any tests ran)."""
        return any(not r.execution.compiled for r in self.results)
```

### 2. Sandbox Protocol (`src/ludic/envs/code_exec/sandbox.py`)

Fully async interface. Compilation is a first-class operation.

```python
from __future__ import annotations
from typing import Protocol, Optional, Dict, runtime_checkable
from abc import abstractmethod

from .types import CompileResult, ExecutionResult

@runtime_checkable
class Sandbox(Protocol):
    """
    Async handle to a single isolated execution environment.

    Invariants:
      - A sandbox is exclusive to one env instance at a time
      - reset() clears all state from previous executions
      - All operations are async
    """

    async def reset(self) -> None:
        """Clear filesystem, kill processes, restore to clean state."""
        ...

    async def compile(
        self,
        code: str,
        *,
        timeout_s: float = 5.0,
    ) -> CompileResult:
        """
        Syntax-check / compile code without executing.

        For Python: runs `py_compile` or `ast.parse`
        For compiled languages: runs the compiler

        Returns compile result with error details if failed.
        """
        ...

    async def execute(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout_s: float = 10.0,
        memory_limit_mb: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute code and return rich results.

        Implicitly compiles first if not already compiled.
        """
        ...

    @property
    def python_version(self) -> str:
        """Python version in this sandbox (e.g., '3.11')."""
        ...

@runtime_checkable
class SandboxPool(Protocol):
    """
    Async pool of reusable sandboxes with caching.

    The pool manages:
      1. Container lifecycle (start/stop)
      2. Checkout/release of exclusive sandbox handles
      3. Execution cache (code+tests -> result)
    """

    async def start(self) -> None:
        """Initialize the pool (start containers, etc.)."""
        ...

    async def checkout(self, timeout_s: float = 30.0) -> Sandbox:
        """
        Get exclusive access to a sandbox.
        Blocks until one is available or timeout.
        """
        ...

    async def release(self, sandbox: Sandbox) -> None:
        """Return a sandbox to the pool (auto-resets)."""
        ...

    async def shutdown(self) -> None:
        """Tear down all sandboxes."""
        ...

    # ----- Caching -----

    def get_cached(
        self,
        code_hash: str,
        tests_hash: str,
    ) -> Optional[BatchTestResult]:
        """
        Check if we have a cached result for this code+tests pair.

        Returns None if not cached.
        """
        ...

    def put_cached(
        self,
        code_hash: str,
        tests_hash: str,
        result: BatchTestResult,
    ) -> None:
        """Cache a result for future lookups."""
        ...

    @property
    def available(self) -> int:
        """Number of sandboxes currently available."""
        ...

    @property
    def python_version(self) -> str:
        """Python version used by sandboxes in this pool."""
        ...

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Cache hit/miss statistics."""
        ...
```

### 3. Test Adapter Protocol (`src/ludic/envs/code_exec/adapters/base.py`)

```python
from __future__ import annotations
from typing import Protocol, List, Any, Dict, runtime_checkable

from ..types import TestCase

@runtime_checkable
class TestAdapter(Protocol):
    """
    Extracts test cases from a dataset sample.

    Each dataset format needs its own adapter to map
    from the sample schema to the TestCase abstraction.
    """

    def get_tests(self, sample: Dict[str, Any]) -> List[TestCase]:
        """Extract test cases from a sample."""
        ...

    def get_prompt(self, sample: Dict[str, Any]) -> str:
        """Extract the problem prompt/question from a sample."""
        ...

    def get_problem_id(self, sample: Dict[str, Any]) -> str:
        """Extract unique problem identifier."""
        ...

    def hash_tests(self, tests: List[TestCase]) -> str:
        """
        Compute a stable hash of test cases for caching.

        The hash should be deterministic and capture all
        test inputs/expected outputs.
        """
        ...
```

### 4. Code Runner Protocol (`src/ludic/envs/code_exec/runners.py`)

```python
from __future__ import annotations
from typing import Protocol, List, Optional, runtime_checkable

from .sandbox import Sandbox
from .types import TestCase, TestResult, BatchTestResult, ExecutionResult

@runtime_checkable
class OutputVerifier(Protocol):
    """Compares actual output against expected output."""

    def verify(self, actual: str, expected: str) -> tuple[bool, Optional[str]]:
        """
        Returns (passed, details).
        Details explains why comparison failed if not passed.
        """
        ...

class ExactMatchVerifier:
    """Exact string match after stripping whitespace."""

    def verify(self, actual: str, expected: str) -> tuple[bool, Optional[str]]:
        a, e = actual.strip(), expected.strip()
        if a == e:
            return True, None
        # Provide useful diff info
        if len(a) != len(e):
            return False, f"Length mismatch: got {len(a)}, expected {len(e)}"
        # Find first difference
        for i, (ca, ce) in enumerate(zip(a, e)):
            if ca != ce:
                return False, f"First diff at char {i}: got {repr(ca)}, expected {repr(ce)}"
        return False, "Unknown difference"

@runtime_checkable
class CodeRunner(Protocol):
    """
    Async runner that executes code against test cases.

    Runners encapsulate the execution strategy:
      - stdin/stdout (APPS style)
      - function calls (HumanEval style)
      - pytest (unit test style)
    """

    async def run_tests(
        self,
        sandbox: Sandbox,
        code: str,
        tests: List[TestCase],
        *,
        verifier: OutputVerifier,
        stop_on_first_failure: bool = False,
        compile_first: bool = True,
    ) -> BatchTestResult:
        """
        Run all tests and return results.

        If compile_first is True, checks compilation before running tests.
        Allows early exit on compile failure.
        """
        ...

# ----- Implementations -----

class StdinStdoutRunner:
    """
    Runner for stdin/stdout-based testing (APPS, Codeforces style).

    Executes code with test_case.input as stdin,
    compares stdout against test_case.expected.
    """

    def __init__(
        self,
        default_timeout_s: float = 10.0,
        memory_limit_mb: Optional[int] = 256,
    ):
        self.default_timeout_s = default_timeout_s
        self.memory_limit_mb = memory_limit_mb

    async def run_tests(
        self,
        sandbox: Sandbox,
        code: str,
        tests: List[TestCase],
        *,
        verifier: OutputVerifier,
        stop_on_first_failure: bool = False,
        compile_first: bool = True,
    ) -> BatchTestResult:
        import hashlib

        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        tests_hash = hashlib.sha256(
            str([(t.input, t.expected) for t in tests]).encode()
        ).hexdigest()[:16]

        results: List[TestResult] = []

        # Optional: compile check first
        if compile_first:
            compile_result = await sandbox.compile(code)
            if not compile_result.status == CompileStatus.SUCCESS:
                # All tests fail with compile error
                exec_result = ExecutionResult(compile_result=compile_result)
                for test in tests:
                    results.append(TestResult(
                        test_case=test,
                        passed=False,
                        actual="",
                        execution=exec_result,
                        comparison_details="Compilation failed",
                    ))
                return BatchTestResult(
                    results=results,
                    code_hash=code_hash,
                    tests_hash=tests_hash,
                )

        # Run each test
        for test in tests:
            timeout = test.timeout_s or self.default_timeout_s

            exec_result = await sandbox.execute(
                code,
                stdin=str(test.input),
                timeout_s=timeout,
                memory_limit_mb=self.memory_limit_mb,
            )

            if not exec_result.succeeded:
                passed = False
                details = f"Execution failed: {exec_result.run_status}"
                if exec_result.stderr:
                    details += f"\n{exec_result.stderr[:500]}"
            else:
                passed, details = verifier.verify(
                    exec_result.stdout,
                    str(test.expected),
                )

            results.append(TestResult(
                test_case=test,
                passed=passed,
                actual=exec_result.stdout,
                execution=exec_result,
                comparison_details=details,
            ))

            if stop_on_first_failure and not passed:
                # Mark remaining tests as not run
                for remaining_test in tests[len(results):]:
                    results.append(TestResult(
                        test_case=remaining_test,
                        passed=False,
                        actual="",
                        execution=ExecutionResult(
                            compile_result=CompileResult(status=CompileStatus.SUCCESS),
                            run_status=None,  # not run
                        ),
                        comparison_details="Skipped due to earlier failure",
                    ))
                break

        return BatchTestResult(
            results=results,
            code_hash=code_hash,
            tests_hash=tests_hash,
        )
```

---

## CodeExecEnv Implementation (`src/ludic/envs/code_exec/env.py`)

```python
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass

from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Info, Observation, StepOutcome

from .sandbox import SandboxPool, Sandbox
from .types import TestCase, BatchTestResult, CompileStatus
from .adapters.base import TestAdapter
from .runners import CodeRunner, StdinStdoutRunner, OutputVerifier, ExactMatchVerifier

import hashlib

@dataclass
class CodeExecConfig:
    """Configuration for CodeExecEnv behavior."""
    # Execution
    timeout_per_test_s: float = 10.0
    memory_limit_mb: int = 256
    max_tests: Optional[int] = None
    stop_on_first_failure: bool = True
    compile_first: bool = True

    # Reward shaping
    partial_credit: bool = False  # reward = fraction of tests passed
    compile_failure_reward: float = -0.1  # penalty for syntax errors
    runtime_error_reward: float = 0.0  # base for runtime errors

    # Observations
    include_stderr_in_obs: bool = True
    max_error_length: int = 500

    # Caching
    use_cache: bool = True

class CodeExecEnv(SingleAgentEnv):
    """
    Async environment for code generation tasks.

    The agent receives a problem prompt and submits code as its action.
    The code is compiled and executed against test cases in a sandboxed
    environment.

    This env is dataset-agnostic: test extraction and code execution
    are delegated to injected adapters/runners.

    Async Operations:
      - Sandbox checkout/release are async
      - Test execution is async
      - Caching avoids redundant execution

    Rich Metadata:
      - Compilation status and errors
      - Per-test execution results
      - Timing information
      - Cache hit/miss info
    """

    def __init__(
        self,
        sample: Dict[str, Any],
        *,
        sandbox_pool: SandboxPool,
        test_adapter: TestAdapter,
        code_runner: Optional[CodeRunner] = None,
        verifier: Optional[OutputVerifier] = None,
        config: Optional[CodeExecConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._sample = sample
        self._pool = sandbox_pool
        self._test_adapter = test_adapter
        self._code_runner = code_runner or StdinStdoutRunner()
        self._verifier = verifier or ExactMatchVerifier()
        self._config = config or CodeExecConfig()
        self._system_prompt = system_prompt

        # Runtime state
        self._sandbox: Optional[Sandbox] = None
        self._tests: List[TestCase] = []
        self._tests_hash: str = ""
        self._prompt: str = ""
        self._problem_id: str = ""
        self._done: bool = False
        self._latest_obs: str = ""

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return self._system_prompt

    async def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        # Extract problem and tests from sample
        self._prompt = self._test_adapter.get_prompt(self._sample)
        self._problem_id = self._test_adapter.get_problem_id(self._sample)
        self._tests = self._test_adapter.get_tests(self._sample)

        if self._config.max_tests:
            self._tests = self._tests[:self._config.max_tests]

        # Compute tests hash for caching
        self._tests_hash = self._test_adapter.hash_tests(self._tests)

        # Checkout a sandbox
        self._sandbox = await self._pool.checkout()
        await self._sandbox.reset()

        self._done = False
        self._latest_obs = self._prompt

        info: Info = {
            "problem_id": self._problem_id,
            "num_tests": len(self._tests),
            "python_version": self._sandbox.python_version,
        }
        return self._prompt, info

    async def env_step(self, action: str) -> StepOutcome:
        if self._done:
            raise RuntimeError("env_step called after episode finished.")

        assert self._sandbox is not None
        code = action

        # Compute code hash for caching
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        # Check cache first
        batch_result: Optional[BatchTestResult] = None
        cache_hit = False

        if self._config.use_cache:
            batch_result = self._pool.get_cached(code_hash, self._tests_hash)
            cache_hit = batch_result is not None

        # Execute if not cached
        if batch_result is None:
            batch_result = await self._code_runner.run_tests(
                self._sandbox,
                code,
                self._tests,
                verifier=self._verifier,
                stop_on_first_failure=self._config.stop_on_first_failure,
                compile_first=self._config.compile_first,
            )

            # Cache the result
            if self._config.use_cache:
                self._pool.put_cached(code_hash, self._tests_hash, batch_result)

        # Release sandbox
        await self._pool.release(self._sandbox)
        self._sandbox = None
        self._done = True

        # Compute reward
        reward = self._compute_reward(batch_result)

        # Build observation
        obs = self._build_observation(batch_result)
        self._latest_obs = obs

        # Build rich info for RL signal
        info = self._build_info(batch_result, code_hash, cache_hit)

        return StepOutcome(
            obs=obs,
            reward=reward,
            truncated=False,
            terminated=True,
            info=info,
        )

    def _compute_reward(self, result: BatchTestResult) -> float:
        """Compute reward with configurable shaping."""
        if result.compile_failed:
            return self._config.compile_failure_reward

        if self._config.partial_credit:
            return result.passed_count / result.total_count if result.total_count > 0 else 0.0
        else:
            return 1.0 if result.all_passed else 0.0

    def _build_observation(self, result: BatchTestResult) -> str:
        """Build human-readable observation."""
        if result.compile_failed:
            first = result.results[0] if result.results else None
            if first and first.execution.compile_result.error_message:
                err = first.execution.compile_result.error_message
                return f"Compilation failed:\n{err[:self._config.max_error_length]}"
            return "Compilation failed."

        if result.all_passed:
            return f"All {result.total_count} tests passed."

        obs = f"Passed {result.passed_count}/{result.total_count} tests."

        if self._config.include_stderr_in_obs:
            first_fail = result.first_failure
            if first_fail:
                details = first_fail.comparison_details or ""
                if first_fail.execution.stderr:
                    details += f"\nstderr: {first_fail.execution.stderr}"
                if details:
                    obs += f"\nFirst failure ({first_fail.test_case.id}):\n{details[:self._config.max_error_length]}"

        return obs

    def _build_info(
        self,
        result: BatchTestResult,
        code_hash: str,
        cache_hit: bool,
    ) -> Info:
        """Build rich metadata for RL signal."""
        # Aggregate timing
        total_compile_ms = sum(
            r.execution.compile_duration_ms for r in result.results
        )
        total_run_ms = sum(
            r.execution.run_duration_ms for r in result.results
        )

        return {
            # Identification
            "problem_id": self._problem_id,
            "code_hash": code_hash,
            "tests_hash": self._tests_hash,

            # Test results
            "passed": result.passed_count,
            "total": result.total_count,
            "all_passed": result.all_passed,
            "compile_failed": result.compile_failed,

            # Per-test details (for fine-grained analysis)
            "test_results": [
                {
                    "test_id": r.test_case.id,
                    "passed": r.passed,
                    "compiled": r.execution.compiled,
                    "run_status": r.execution.run_status.value if r.execution.run_status else None,
                    "compile_duration_ms": r.execution.compile_duration_ms,
                    "run_duration_ms": r.execution.run_duration_ms,
                    "comparison_details": r.comparison_details,
                }
                for r in result.results
            ],

            # Timing aggregates
            "total_compile_ms": total_compile_ms,
            "total_run_ms": total_run_ms,
            "total_execution_ms": total_compile_ms + total_run_ms,

            # Cache info
            "cache_hit": cache_hit,

            # Python version
            "python_version": self._pool.python_version,
        }

    def env_current_obs(self) -> Observation:
        return self._latest_obs
```

**Note**: The `SingleAgentEnv` base class uses sync methods (`env_reset`, `env_step`). The actual implementation will need to either:
1. Make the interaction protocol async-aware (preferred)
2. Use `asyncio.run()` internally (fallback)

This will be addressed in the protocol integration step.

---

## Docker Sandbox Implementation (`src/ludic/envs/code_exec/docker_sandbox.py`)

```python
"""
Async Docker-based sandbox implementation.

Features:
  - Configurable Python version
  - Resource limits (CPU, memory, no network)
  - LRU cache for code+tests results
  - Async operations via aiodocker or docker-py with executor
"""

from __future__ import annotations
import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor

import docker  # docker-py

from .sandbox import Sandbox, SandboxPool
from .types import (
    CompileResult, CompileStatus,
    ExecutionResult, RunStatus,
    BatchTestResult,
)

@dataclass
class DockerSandboxConfig:
    """Configuration for Docker sandboxes."""
    python_version: str = "3.11"
    base_image: Optional[str] = None  # auto-generates from python_version if None
    memory_limit: str = "256m"
    cpu_quota: int = 50000  # 50% of one CPU
    network_disabled: bool = True
    working_dir: str = "/workspace"
    # Compile command (can be customized for other languages)
    compile_command: str = "python -m py_compile {file}"
    run_command: str = "python {file}"

    @property
    def image(self) -> str:
        if self.base_image:
            return self.base_image
        return f"python:{self.python_version}-slim"

class DockerSandbox:
    """
    Async wrapper around a Docker container.

    Uses ThreadPoolExecutor to make docker-py calls non-blocking.
    """

    def __init__(
        self,
        container,  # docker.models.containers.Container
        config: DockerSandboxConfig,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._container = container
        self._config = config
        self._executor = executor
        self._loop = asyncio.get_event_loop()

    async def _run_in_executor(self, fn, *args, **kwargs):
        """Run blocking docker operation in thread pool."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: fn(*args, **kwargs),
        )

    @property
    def python_version(self) -> str:
        return self._config.python_version

    async def reset(self) -> None:
        """Clear the workspace directory."""
        await self._run_in_executor(
            self._container.exec_run,
            "rm -rf /workspace/* /workspace/.*",
            workdir="/",
        )

    async def compile(
        self,
        code: str,
        *,
        timeout_s: float = 5.0,
    ) -> CompileResult:
        """
        Syntax-check code without executing.

        Uses py_compile for Python, which catches syntax errors
        but not import errors.
        """
        import time

        # Write code to file
        await self._write_file("/workspace/solution.py", code)

        start = time.monotonic()
        try:
            exit_code, output = await asyncio.wait_for(
                self._run_in_executor(
                    self._container.exec_run,
                    self._config.compile_command.format(file="/workspace/solution.py"),
                    workdir=self._config.working_dir,
                    demux=True,
                ),
                timeout=timeout_s,
            )
            duration_ms = (time.monotonic() - start) * 1000
            stdout, stderr = output or (b"", b"")

            if exit_code == 0:
                return CompileResult(
                    status=CompileStatus.SUCCESS,
                    duration_ms=duration_ms,
                )
            else:
                error_msg = (stderr or stdout or b"").decode()
                # Parse error line/column from Python traceback
                error_line, error_col = self._parse_syntax_error(error_msg)
                return CompileResult(
                    status=CompileStatus.SYNTAX_ERROR,
                    error_message=error_msg,
                    error_line=error_line,
                    error_column=error_col,
                    duration_ms=duration_ms,
                )
        except asyncio.TimeoutError:
            return CompileResult(
                status=CompileStatus.TIMEOUT,
                error_message=f"Compilation timed out after {timeout_s}s",
                duration_ms=timeout_s * 1000,
            )
        except Exception as e:
            return CompileResult(
                status=CompileStatus.UNKNOWN_ERROR,
                error_message=str(e),
            )

    async def execute(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout_s: float = 10.0,
        memory_limit_mb: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute code with stdin input."""
        import time

        # First compile
        compile_result = await self.compile(code, timeout_s=min(5.0, timeout_s / 2))
        if compile_result.status != CompileStatus.SUCCESS:
            return ExecutionResult(compile_result=compile_result)

        # Execute
        start = time.monotonic()
        try:
            # Write stdin to file for piping
            if stdin:
                await self._write_file("/workspace/input.txt", stdin)
                cmd = f"python /workspace/solution.py < /workspace/input.txt"
            else:
                cmd = "python /workspace/solution.py"

            exit_code, output = await asyncio.wait_for(
                self._run_in_executor(
                    self._container.exec_run,
                    f"sh -c '{cmd}'",
                    workdir=self._config.working_dir,
                    environment=env_vars,
                    demux=True,
                ),
                timeout=timeout_s,
            )
            duration_ms = (time.monotonic() - start) * 1000
            stdout, stderr = output or (b"", b"")

            run_status = RunStatus.SUCCESS if exit_code == 0 else RunStatus.RUNTIME_ERROR

            return ExecutionResult(
                compile_result=compile_result,
                run_status=run_status,
                stdout=stdout.decode() if stdout else "",
                stderr=stderr.decode() if stderr else "",
                exit_code=exit_code,
                compile_duration_ms=compile_result.duration_ms,
                run_duration_ms=duration_ms,
                total_duration_ms=compile_result.duration_ms + duration_ms,
            )
        except asyncio.TimeoutError:
            return ExecutionResult(
                compile_result=compile_result,
                run_status=RunStatus.TIMEOUT,
                stderr=f"Execution timed out after {timeout_s}s",
                compile_duration_ms=compile_result.duration_ms,
                run_duration_ms=timeout_s * 1000,
                total_duration_ms=compile_result.duration_ms + timeout_s * 1000,
            )

    async def _write_file(self, path: str, content: str) -> None:
        """Write content to a file in the container."""
        # Use tar to write file (more reliable than exec with stdin)
        import io
        import tarfile

        data = content.encode()
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode='w') as tar:
            info = tarfile.TarInfo(name=path.split('/')[-1])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tarstream.seek(0)

        await self._run_in_executor(
            self._container.put_archive,
            '/'.join(path.split('/')[:-1]) or '/',
            tarstream.getvalue(),
        )

    def _parse_syntax_error(self, error_msg: str) -> tuple[Optional[int], Optional[int]]:
        """Extract line/column from Python syntax error message."""
        import re
        match = re.search(r'line (\d+)', error_msg)
        line = int(match.group(1)) if match else None
        match = re.search(r'offset (\d+)', error_msg)
        col = int(match.group(1)) if match else None
        return line, col


class LRUCache:
    """Thread-safe LRU cache for execution results."""

    def __init__(self, max_size: int = 10000):
        self._cache: OrderedDict[str, BatchTestResult] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, code_hash: str, tests_hash: str) -> str:
        return f"{code_hash}:{tests_hash}"

    async def get(self, code_hash: str, tests_hash: str) -> Optional[BatchTestResult]:
        key = self._make_key(code_hash, tests_hash)
        async with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None

    async def put(self, code_hash: str, tests_hash: str, result: BatchTestResult) -> None:
        key = self._make_key(code_hash, tests_hash)
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                self._cache[key] = result
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._max_size,
        }


class DockerSandboxPool:
    """
    Async pool of Docker container sandboxes.

    Features:
      - Configurable Python version
      - LRU cache for execution results
      - Async checkout/release with semaphore
    """

    def __init__(
        self,
        n_workers: int = 8,
        config: Optional[DockerSandboxConfig] = None,
        cache_size: int = 10000,
        executor_threads: Optional[int] = None,
    ) -> None:
        self._n_workers = n_workers
        self._config = config or DockerSandboxConfig()
        self._cache = LRUCache(max_size=cache_size)
        self._executor = ThreadPoolExecutor(
            max_workers=executor_threads or n_workers * 2
        )

        self._client: Optional[docker.DockerClient] = None
        self._containers: list = []
        self._pool: asyncio.Queue[DockerSandbox] = asyncio.Queue()
        self._started = False

    @property
    def python_version(self) -> str:
        return self._config.python_version

    @property
    def cache_stats(self) -> Dict[str, int]:
        return self._cache.stats

    @property
    def available(self) -> int:
        return self._pool.qsize()

    async def start(self) -> None:
        """Spin up N containers."""
        if self._started:
            return

        self._client = docker.from_env()

        # Ensure image is pulled
        try:
            self._client.images.get(self._config.image)
        except docker.errors.ImageNotFound:
            print(f"Pulling image {self._config.image}...")
            self._client.images.pull(self._config.image)

        for i in range(self._n_workers):
            container = self._client.containers.run(
                self._config.image,
                command="sleep infinity",
                detach=True,
                mem_limit=self._config.memory_limit,
                cpu_quota=self._config.cpu_quota,
                network_disabled=self._config.network_disabled,
                auto_remove=True,
                name=f"ludic-sandbox-{self._config.python_version}-{i}",
            )
            sandbox = DockerSandbox(container, self._config, self._executor)
            self._containers.append(container)
            await self._pool.put(sandbox)

        self._started = True
        print(f"Started {self._n_workers} sandboxes (Python {self._config.python_version})")

    async def checkout(self, timeout_s: float = 30.0) -> Sandbox:
        """Get a sandbox, blocking until one is available."""
        try:
            return await asyncio.wait_for(
                self._pool.get(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"No sandbox available after {timeout_s}s "
                f"(pool size: {self._n_workers}, available: {self.available})"
            )

    async def release(self, sandbox: Sandbox) -> None:
        """Return sandbox to pool (resets it first)."""
        await sandbox.reset()
        await self._pool.put(sandbox)

    async def shutdown(self) -> None:
        """Stop and remove all containers."""
        for container in self._containers:
            try:
                container.stop(timeout=1)
            except:
                pass
        self._containers.clear()
        self._executor.shutdown(wait=False)
        self._started = False

    # ----- Cache interface -----

    def get_cached(
        self,
        code_hash: str,
        tests_hash: str,
    ) -> Optional[BatchTestResult]:
        """Sync cache lookup (used from env)."""
        # Note: This is sync because it's called from env_step
        # The cache itself is thread-safe
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._cache.get(code_hash, tests_hash)
            )
        except RuntimeError:
            # No event loop, return None
            return None

    def put_cached(
        self,
        code_hash: str,
        tests_hash: str,
        result: BatchTestResult,
    ) -> None:
        """Sync cache store."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                self._cache.put(code_hash, tests_hash, result)
            )
        except RuntimeError:
            pass
```

---

## APPS Dataset Adapter (`src/ludic/envs/code_exec/adapters/apps.py`)

```python
"""
Adapter for APPS-style datasets (including apps-control-arena).

APPS format:
  - question: problem description (string)
  - inputs: list of stdin strings
  - outputs: list of expected stdout strings
  - problem_id: unique identifier

Code is expected to be a Python script that reads from stdin
and writes to stdout.
"""

from __future__ import annotations
import hashlib
from typing import Any, Dict, List

from ..types import TestCase
from .base import TestAdapter

class APPSTestAdapter:
    """
    Test adapter for APPS-style datasets.

    Compatible with:
      - codeparrot/apps
      - RoganInglis/apps-control-arena
      - Similar stdin/stdout datasets

    Args:
        question_key: Key for problem description
        inputs_key: Key for test inputs
        outputs_key: Key for expected outputs
        problem_id_key: Key for problem identifier
    """

    def __init__(
        self,
        *,
        question_key: str = "question",
        inputs_key: str = "inputs",
        outputs_key: str = "outputs",
        problem_id_key: str = "problem_id",
    ) -> None:
        self._question_key = question_key
        self._inputs_key = inputs_key
        self._outputs_key = outputs_key
        self._problem_id_key = problem_id_key

    def get_prompt(self, sample: Dict[str, Any]) -> str:
        return str(sample[self._question_key])

    def get_problem_id(self, sample: Dict[str, Any]) -> str:
        return str(sample.get(self._problem_id_key, "unknown"))

    def get_tests(self, sample: Dict[str, Any]) -> List[TestCase]:
        inputs = sample[self._inputs_key]
        outputs = sample[self._outputs_key]

        if len(inputs) != len(outputs):
            raise ValueError(
                f"Mismatched inputs/outputs: {len(inputs)} vs {len(outputs)}"
            )

        return [
            TestCase(
                input=inp,
                expected=out,
                id=f"test_{i}",
            )
            for i, (inp, out) in enumerate(zip(inputs, outputs))
        ]

    def hash_tests(self, tests: List[TestCase]) -> str:
        """Compute stable hash of test cases."""
        content = str([(t.input, t.expected) for t in tests])
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# Default system prompt for APPS-style problems
APPS_SYSTEM_PROMPT = """You are an expert Python programmer. You will be given a programming problem.

Write a Python solution that:
1. Reads input from stdin using input() or sys.stdin
2. Prints the output to stdout using print()
3. Handles all edge cases correctly

Output ONLY the Python code, no explanations."""
```

---

## File Structure

```
src/ludic/envs/code_exec/
├── __init__.py              # Public API exports
├── types.py                 # ExecutionResult, TestCase, etc.
├── sandbox.py               # Sandbox/SandboxPool protocols
├── runners.py               # CodeRunner implementations
├── env.py                   # CodeExecEnv
├── docker_sandbox.py        # Docker-based async sandbox pool
└── adapters/
    ├── __init__.py
    ├── base.py              # TestAdapter protocol
    ├── apps.py              # APPS dataset adapter
    └── humaneval.py         # HumanEval adapter (future)
```

---

## Integration with Existing Ludic Components

### Making Interaction Protocols Async-Aware

The `SingleAgentEnv` base class uses sync methods. Two approaches:

**Option A (Recommended): Async Protocol Variant**

Add `AsyncSingleAgentProtocol` that calls `await env.env_reset()` and `await env.env_step()`. The protocol detects if methods are coroutines.

```python
# In interaction/single_agent.py

class SingleAgentProtocol(InteractionProtocol):
    async def run(self, ...):
        # Detect if env methods are coroutines
        reset_result = env.env_reset(seed=seed)
        if asyncio.iscoroutine(reset_result):
            obs, info = await reset_result
        else:
            obs, info = reset_result
        # ... similar for env_step
```

**Option B: Internal asyncio.run()**

The env internally uses `asyncio.run()` for async operations. Simpler but has event loop nesting issues.

### Integration with RolloutEngine

The `RolloutEngine._run_one_request` is already async. The env's async methods integrate naturally:

```python
# In rollout_engine.py, envs with async methods work because
# protocol.run() is already async and awaits env operations
```

---

## Usage Example

```python
import asyncio
from datasets import load_dataset

from ludic.training.batching.rollout_engine import RolloutEngine
from ludic.training.types import RolloutRequest, EnvSpec, ProtocolSpec

from ludic.envs.code_exec import (
    CodeExecEnv,
    CodeExecConfig,
    DockerSandboxPool,
    DockerSandboxConfig,
)
from ludic.envs.code_exec.adapters.apps import APPSTestAdapter, APPS_SYSTEM_PROMPT

async def main():
    # 1. Setup sandbox pool with Python 3.11
    pool = DockerSandboxPool(
        n_workers=32,
        config=DockerSandboxConfig(
            python_version="3.11",
            memory_limit="512m",
        ),
        cache_size=50_000,  # cache up to 50k code+tests results
    )
    await pool.start()

    # 2. Configure adapter
    test_adapter = APPSTestAdapter()

    # 3. Define env factory
    def build_code_env(sample, **kw):
        return CodeExecEnv(
            sample=sample,
            sandbox_pool=pool,
            test_adapter=test_adapter,
            config=CodeExecConfig(
                partial_credit=False,
                stop_on_first_failure=True,
                compile_failure_reward=-0.1,
                use_cache=True,
            ),
            system_prompt=APPS_SYSTEM_PROMPT,
            **kw,
        )

    # 4. Register with engine
    engine = RolloutEngine(
        env_registry={"code_exec": build_code_env},
        protocol_registry={...},
    )

    # 5. Build requests from dataset
    dataset = load_dataset("RoganInglis/apps-control-arena")["train"]
    dataset = dataset.filter(lambda x: not x.get("is_nondeterministic", False))

    requests = [
        RolloutRequest(
            env=EnvSpec(kind="code_exec", kwargs={"sample": dict(row)}),
            protocol=ProtocolSpec(kind="single_agent", kwargs={...}),
            meta={"problem_id": row["problem_id"]},
        )
        for row in dataset.select(range(100))
    ]

    # 6. Generate rollouts (sandbox pool handles concurrency)
    rollouts = await engine.generate_rollouts(
        requests=requests,
        max_steps=1,
        concurrency=32,  # match pool size for max throughput
    )

    # Check cache efficiency
    print(f"Cache stats: {pool.cache_stats}")

    # Cleanup
    await pool.shutdown()

asyncio.run(main())
```

---

## Performance Considerations

### Throughput Targets

With 32 sandboxes and 10s timeout per test:
- **Best case (all cache hits)**: ~1000+ requests/sec
- **Typical (mixed cache)**: ~100-300 requests/sec
- **Worst case (cold cache, many tests)**: ~3-10 requests/sec

### Bottleneck Analysis

1. **Container startup**: Amortized by pool (one-time cost)
2. **File I/O in containers**: Mitigated by tmpfs mounts
3. **Network to Docker**: Minimal (local socket)
4. **Cache lookup**: O(1) with LRU

### Scaling Options

1. **More workers**: Increase `n_workers` (limited by host resources)
2. **Distributed pools**: Run pools across multiple hosts
3. **Warm cache**: Pre-populate cache with common submissions

---

## Security Considerations

1. **Network isolation**: Containers have no network access
2. **Resource limits**: CPU/memory capped per container
3. **Timeout enforcement**: Prevents infinite loops
4. **Filesystem isolation**: Each container has isolated workspace
5. **No persistent state**: Containers reset between uses

---

## Open Items for Future Iterations

1. **Multi-language support**: Add language field to config, select interpreter
2. **GPU execution**: For ML code tasks
3. **Multi-file submissions**: Support importing modules
4. **Interactive debugging**: Return intermediate states for debugging
5. **Distributed caching**: Redis-backed cache for multi-host setups

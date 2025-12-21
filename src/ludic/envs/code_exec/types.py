"""
Core types for code execution environments.

These types capture rich metadata about code compilation and execution,
providing RL-relevant signals for reward shaping and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SandboxPoolExhaustedError(Exception):
    """
    Raised when sandbox pool experiences too many consecutive failures.

    This indicates a systemic issue with sandbox creation/reset that
    requires operator intervention.
    """

    pass


class CompileStatus(Enum):
    """Status of code compilation/syntax checking."""

    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"


class RunStatus(Enum):
    """Status of code execution."""

    SUCCESS = "success"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    KILLED = "killed"
    NOT_RUN = "not_run"  # e.g., skipped due to earlier failure


@dataclass
class CompileResult:
    """
    Result of compiling/syntax-checking code.

    For Python, this typically uses py_compile or ast.parse to catch
    syntax errors before execution.
    """

    status: CompileStatus
    error_message: Optional[str] = None
    error_line: Optional[int] = None
    error_column: Optional[int] = None
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == CompileStatus.SUCCESS


@dataclass
class ExecutionResult:
    """
    Rich result of running code in a sandbox.

    All fields are RL-relevant metadata that can be used for:
      - Reward shaping (compile errors vs runtime errors vs wrong answer)
      - Curriculum learning (filter by execution characteristics)
      - Analysis (understanding failure modes)

    This is the atomic unit returned by sandbox.execute().
    """

    # Compilation phase
    compile_result: CompileResult

    # Execution phase (only meaningful if compilation succeeded)
    run_status: Optional[RunStatus] = None
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    return_value: Optional[str] = None  # for function-based testing

    # Timing (all in milliseconds)
    compile_duration_ms: float = 0.0
    run_duration_ms: float = 0.0
    total_duration_ms: float = 0.0

    # Resource usage (optional, depends on sandbox implementation)
    peak_memory_bytes: Optional[int] = None
    cpu_time_ms: Optional[float] = None

    # Cache info
    cache_hit: bool = False
    cache_key: Optional[str] = None

    @property
    def compiled(self) -> bool:
        """True if code compiled successfully."""
        return self.compile_result.success

    @property
    def succeeded(self) -> bool:
        """True if code compiled and ran without errors."""
        return self.compiled and self.run_status == RunStatus.SUCCESS

    @property
    def timed_out(self) -> bool:
        """True if either compilation or execution timed out."""
        return (
            self.compile_result.status == CompileStatus.TIMEOUT
            or self.run_status == RunStatus.TIMEOUT
        )


@dataclass
class TestCase:
    """
    A single test case.

    The interpretation of `input` and `expected` depends on the CodeRunner:
      - stdin/stdout: input is stdin string, expected is stdout string
      - function call: input is (args, kwargs), expected is return value
      - pytest: input is test code, expected is None (pass/fail from exit code)
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    input: Any
    expected: Any
    id: str = ""
    weight: float = 1.0  # for weighted partial credit
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of running a single test case."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    test_case: TestCase
    passed: bool
    actual: Any
    execution: ExecutionResult
    comparison_details: Optional[str] = None  # explains why comparison failed

    @property
    def compiled(self) -> bool:
        """True if code compiled for this test."""
        return self.execution.compiled

    @property
    def ran(self) -> bool:
        """True if code actually executed (not skipped)."""
        return self.execution.run_status not in (None, RunStatus.NOT_RUN)


@dataclass
class BatchTestResult:
    """
    Result of running all tests for a code submission.

    Aggregates individual TestResults and provides convenience properties
    for computing rewards and analyzing results.
    """

    results: List[TestResult]
    code_hash: str
    tests_hash: str

    @property
    def passed_count(self) -> int:
        """Number of tests that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        """Total number of tests."""
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        """True if all tests passed."""
        return self.passed_count == self.total_count and self.total_count > 0

    @property
    def pass_rate(self) -> float:
        """Fraction of tests that passed (0.0 to 1.0)."""
        if self.total_count == 0:
            return 0.0
        return self.passed_count / self.total_count

    @property
    def first_failure(self) -> Optional[TestResult]:
        """The first test that failed, or None if all passed."""
        for r in self.results:
            if not r.passed:
                return r
        return None

    @property
    def compile_failed(self) -> bool:
        """True if code failed to compile (before any tests ran)."""
        if not self.results:
            return False
        # If compilation failed, all tests will have the same compile failure
        return not self.results[0].compiled

    @property
    def total_execution_ms(self) -> float:
        """Total execution time across all tests."""
        return sum(r.execution.total_duration_ms for r in self.results)

    @property
    def total_compile_ms(self) -> float:
        """Total compilation time (usually same across tests if compiled once)."""
        if not self.results:
            return 0.0
        # Compilation typically happens once, take max to be safe
        return max(r.execution.compile_duration_ms for r in self.results)

    @property
    def total_run_ms(self) -> float:
        """Total runtime across all tests (excluding compilation)."""
        return sum(r.execution.run_duration_ms for r in self.results)

    def get_failures(self) -> List[TestResult]:
        """All tests that failed."""
        return [r for r in self.results if not r.passed]

    def get_successes(self) -> List[TestResult]:
        """All tests that passed."""
        return [r for r in self.results if r.passed]

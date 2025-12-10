# src/ludic/sandbox/protocol.py
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ExecutionStatus(Enum):
    SUCCESS = "success"
    COMPILE_ERROR = "compile_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"


@dataclass
class TestResult:
    name: str
    passed: bool
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    tests_passed: int
    tests_total: int
    test_results: List[TestResult] = field(default_factory=list)
    compile_output: Optional[str] = None
    stderr: Optional[str] = None
    execution_time_ms: float = 0.0

    @property
    def all_passed(self) -> bool:
        return (
            self.status == ExecutionStatus.SUCCESS
            and self.tests_passed == self.tests_total
        )

    @property
    def pass_rate(self) -> float:
        return self.tests_passed / self.tests_total if self.tests_total > 0 else 0.0

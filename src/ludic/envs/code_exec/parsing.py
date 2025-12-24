"""Shared parsing utilities for code execution sandboxes."""

from __future__ import annotations

import re
import time
from typing import Optional

from .types import CompileResult, CompileStatus, ExecutionResult, RunStatus

# Import batch runner script using importlib.resources
try:
    from importlib.resources import files

    _BATCH_RUNNER_SCRIPT: Optional[str] = None

    def get_batch_runner_script() -> str:
        """Lazy-load the batch runner script from package resources."""
        global _BATCH_RUNNER_SCRIPT
        if _BATCH_RUNNER_SCRIPT is None:
            _BATCH_RUNNER_SCRIPT = (
                files("ludic.envs.code_exec")
                .joinpath("batch_runner.py")
                .read_text()
            )
        return _BATCH_RUNNER_SCRIPT

except ImportError:
    # Fallback for older Python versions
    import pkg_resources

    def get_batch_runner_script() -> str:
        return pkg_resources.resource_string(
            "ludic.envs.code_exec", "batch_runner.py"
        ).decode("utf-8")


def parse_syntax_error(error_msg: str) -> tuple[Optional[int], Optional[int], str]:
    """Parse Python syntax error to extract line, column, and clean message."""
    line = None
    column = None
    clean_msg = ""

    # Try to find line number
    line_match = re.search(r'line (\d+)', error_msg)
    if line_match:
        line = int(line_match.group(1))

    # Try to find column number
    col_match = re.search(r'column (\d+)', error_msg)
    if col_match:
        column = int(col_match.group(1))

    # Extract error type and message
    error_type_match = re.search(
        r'(SyntaxError|IndentationError|TabError):\s*(.+)', error_msg
    )
    if error_type_match:
        error_type = error_type_match.group(1)
        msg = error_type_match.group(2).strip()
        clean_msg = f"{error_type}: {msg}"
    else:
        # Fall back to just extracting the last line
        lines = [l.strip() for l in error_msg.split('\n') if l.strip()]
        if lines:
            clean_msg = lines[-1]

    return line, column, clean_msg


def parse_batch_compile_result(result: dict) -> CompileResult:
    """Parse compile result from batch runner JSON."""
    status_str = result.get("status", "unknown_error")

    if status_str == "success":
        status = CompileStatus.SUCCESS
    elif status_str == "syntax_error":
        status = CompileStatus.SYNTAX_ERROR
    elif status_str == "timeout":
        status = CompileStatus.TIMEOUT
    else:
        status = CompileStatus.UNKNOWN_ERROR

    return CompileResult(
        status=status,
        error_message=result.get("error_message"),
        error_line=result.get("error_line"),
        error_column=result.get("error_column"),
        duration_ms=result.get("duration_ms", 0.0),
    )


def parse_batch_test_result(
    result: dict,
    run_start: float,
) -> ExecutionResult:
    """Parse test result from batch runner JSON."""
    status_str = result.get("status", "runtime_error")

    if status_str == "success":
        run_status = RunStatus.SUCCESS
    elif status_str == "runtime_error":
        run_status = RunStatus.RUNTIME_ERROR
    elif status_str == "timeout":
        run_status = RunStatus.TIMEOUT
    elif status_str == "memory_exceeded":
        run_status = RunStatus.MEMORY_EXCEEDED
    elif status_str == "not_run":
        run_status = RunStatus.NOT_RUN
    elif status_str == "killed":
        run_status = RunStatus.KILLED
    else:
        run_status = RunStatus.RUNTIME_ERROR

    duration_ms = result.get("duration_ms", 0.0)
    total_ms = (time.perf_counter() - run_start) * 1000

    return ExecutionResult(
        compile_result=CompileResult(status=CompileStatus.SUCCESS),
        run_status=run_status,
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        exit_code=result.get("exit_code"),
        run_duration_ms=duration_ms,
        total_duration_ms=total_ms,
        cache_key=result.get("id", ""),  # Pass test_id for matching in runner
    )

#!/usr/bin/env python3
"""
Batch test runner for ludic code execution sandbox.

This script runs inside the container. It:
1. Reads manifest.json for test configuration
2. Optionally compiles the solution using py_compile
3. Runs tests in PARALLEL using multiprocessing.Pool (default 16 workers)
4. Outputs streaming JSONL results (one JSON object per line, flushed immediately)

Usage:
    python batch_runner.py [manifest_path]

The manifest.json format:
    {
        "code_file": "solution.py",
        "compile_first": true,
        "timeout_s": 5.0,
        "stop_on_first_failure": true,
        "num_workers": 16,
        "tests": [
            {"id": "test_0", "stdin": "5\\n", "expected": "25\\n"},
            {"id": "test_1", "stdin": "3\\n", "expected": "9\\n"}
        ]
    }

Output format (streaming JSONL):
    {"type": "compile", "status": "success", "duration_ms": 12.5}
    {"type": "test", "id": "test_0", "status": "success", "stdout": "25\\n", ...}
    {"type": "test", "id": "test_1", "status": "timeout", ...}
    {"type": "done", "total_tests": 2, "passed": 1, "failed": 1, "compile_failed": false}

Status values:
    compile: success, syntax_error, timeout
    test: success, runtime_error, timeout, memory_exceeded, not_run

Note: This script is designed to be self-contained with no external dependencies
beyond Python's standard library. It will be bundled into the container at runtime.
"""

from __future__ import annotations

import json
import multiprocessing
import py_compile
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple


def emit(obj: Dict[str, Any]) -> None:
    """Write JSON line and flush immediately for streaming.

    Each line must be a complete JSON object to enable partial result
    recovery if the container crashes mid-execution.
    """
    print(json.dumps(obj), flush=True)


def compile_check(code_file: str, timeout_s: float) -> Dict[str, Any]:
    """Run py_compile and return result dict.

    Args:
        code_file: Path to the Python file to compile
        timeout_s: Timeout for compilation (not currently enforced for py_compile)

    Returns:
        Dict with type="compile" and status/error info
    """
    start = time.perf_counter()
    try:
        py_compile.compile(code_file, doraise=True)
        return {
            "type": "compile",
            "status": "success",
            "duration_ms": (time.perf_counter() - start) * 1000,
        }
    except py_compile.PyCompileError as e:
        # Extract line number from the exception
        # PyCompileError has exc_value which contains the SyntaxError
        error_line: Optional[int] = None
        error_column: Optional[int] = None
        error_message = str(e)

        # Try to extract line/column from the underlying SyntaxError
        if hasattr(e, "exc_value") and e.exc_value is not None:
            exc = e.exc_value
            if hasattr(exc, "lineno"):
                error_line = exc.lineno
            if hasattr(exc, "offset"):
                error_column = exc.offset
            if hasattr(exc, "msg"):
                error_message = exc.msg

        return {
            "type": "compile",
            "status": "syntax_error",
            "error_message": error_message,
            "error_line": error_line,
            "error_column": error_column,
            "duration_ms": (time.perf_counter() - start) * 1000,
        }


def run_test(code_file: str, test: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Run a single test and return result dict.

    Args:
        code_file: Path to the Python file to execute
        test: Test specification with id, stdin, expected (optional)
        timeout_s: Timeout in seconds for the test execution

    Returns:
        Dict with type="test" and execution results
    """
    start = time.perf_counter()
    test_id = test.get("id", "unknown")
    stdin_data = test.get("stdin", "")

    try:
        proc = subprocess.run(
            [sys.executable, code_file],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        # Classify status based on return code
        if proc.returncode == 0:
            status = "success"
        elif proc.returncode == 137:
            # SIGKILL - typically OOM killer
            status = "memory_exceeded"
        elif proc.returncode == 143:
            # SIGTERM
            status = "killed"
        else:
            status = "runtime_error"

        return {
            "type": "test",
            "id": test_id,
            "status": status,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "duration_ms": duration_ms,
        }

    except subprocess.TimeoutExpired as e:
        # Capture any partial output
        stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""

        return {
            "type": "test",
            "id": test_id,
            "status": "timeout",
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": None,
            "duration_ms": timeout_s * 1000,
        }

    except Exception as e:
        # Catch any unexpected errors (e.g., file not found)
        duration_ms = (time.perf_counter() - start) * 1000
        return {
            "type": "test",
            "id": test_id,
            "status": "runtime_error",
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "exit_code": None,
            "duration_ms": duration_ms,
        }


def _run_test_wrapper(args: Tuple[int, Dict[str, Any], str, float]) -> Tuple[int, Dict[str, Any]]:
    """Wrapper for multiprocessing - must be top-level function for pickling.

    Args:
        args: Tuple of (test_index, test_dict, code_file, timeout_s)

    Returns:
        Tuple of (test_index, result_dict) to preserve ordering info
    """
    i, test, code_file, timeout_s = args
    result = run_test(code_file, test, timeout_s)
    return (i, result)


def run_tests_parallel(
    code_file: str,
    tests: List[Dict[str, Any]],
    timeout_s: float,
    num_workers: int = 16,
) -> Iterator[Dict[str, Any]]:
    """Run tests in parallel using multiprocessing.Pool.

    Uses imap_unordered for streaming results as they complete (not waiting
    for all tests). This dramatically reduces wall-clock time when tests
    have varying execution times.

    Args:
        code_file: Path to the Python file to execute
        tests: List of test specifications
        timeout_s: Timeout per test in seconds
        num_workers: Number of parallel worker processes (default 16 for HPC)

    Yields:
        Test result dicts as they complete (unordered)
    """
    if not tests:
        return

    # Prepare arguments for each test
    args_list = [(i, test, code_file, timeout_s) for i, test in enumerate(tests)]

    # Use spawn context to avoid fork issues with subprocess-heavy workloads
    # This is safer on HPC systems where fork can cause issues with MPI, CUDA, etc.
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=min(num_workers, len(tests))) as pool:
        # imap_unordered streams results as they complete
        for _i, result in pool.imap_unordered(_run_test_wrapper, args_list):
            yield result


def main() -> None:
    """Main entry point for batch runner."""
    # Get manifest path from command line or use default
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "manifest.json"

    # Load manifest
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        emit({
            "type": "error",
            "message": f"Failed to load manifest: {e}",
        })
        emit({
            "type": "done",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "compile_failed": False,
        })
        return

    # Extract configuration
    code_file = manifest.get("code_file", "solution.py")
    compile_first = manifest.get("compile_first", True)
    timeout_s = manifest.get("timeout_s", 5.0)
    stop_on_first_failure = manifest.get("stop_on_first_failure", True)
    num_workers = manifest.get("num_workers", 16)  # Configurable via manifest
    tests: List[Dict[str, Any]] = manifest.get("tests", [])

    # Step 1: Compile check (optional)
    if compile_first:
        result = compile_check(code_file, timeout_s)
        emit(result)

        if result["status"] != "success":
            # Compilation failed - emit done and exit
            emit({
                "type": "done",
                "total_tests": len(tests),
                "passed": 0,
                "failed": 0,
                "compile_failed": True,
            })
            return

    # Step 2: Run tests in parallel
    passed = 0
    failed = 0
    received_ids: set[str] = set()

    # Use parallel execution for better throughput on HPC
    for result in run_tests_parallel(code_file, tests, timeout_s, num_workers):
        emit(result)  # Stream immediately as each test completes
        received_ids.add(result.get("id", "unknown"))

        if result["status"] == "success":
            passed += 1
        else:
            failed += 1

            if stop_on_first_failure:
                # Early termination - emit remaining tests as "not_run"
                # Note: with parallel execution, some tests may have already
                # started but the pool will be terminated on context exit
                break

    # Emit any tests that didn't run (due to early termination or errors)
    for test in tests:
        test_id = test.get("id", "unknown")
        if test_id not in received_ids:
            emit({
                "type": "test",
                "id": test_id,
                "status": "not_run",
                "stdout": "",
                "stderr": "",
                "exit_code": None,
                "duration_ms": 0,
            })

    # Step 3: Emit done marker
    emit({
        "type": "done",
        "total_tests": len(tests),
        "passed": passed,
        "failed": failed,
        "compile_failed": False,
    })


if __name__ == "__main__":
    main()

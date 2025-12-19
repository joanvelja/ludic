#!/usr/bin/env python3
"""
Diagnostic script to test podman-hpc configuration on HPC clusters.

This script tests various podman-hpc run configurations to identify
which flags work on your specific HPC environment.

Usage:
    uv run python examples/code_exec/diagnose_podman.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys


def log(msg: str, level: str = "INFO") -> None:
    """Simple logging."""
    print(f"[{level}] {msg}")


async def run_cmd(cmd: list[str], description: str) -> tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)."""
    log(f"Testing: {description}")
    log(f"  Command: {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()
    stdout_str = stdout.decode("utf-8", errors="replace")
    stderr_str = stderr.decode("utf-8", errors="replace")

    success = proc.returncode == 0
    status = "OK" if success else f"FAILED (exit {proc.returncode})"
    log(f"  Result: {status}")

    if not success and stderr_str:
        log(f"  Stderr: {stderr_str[:500]}")

    return success, stdout_str, stderr_str


async def cleanup_container(name: str) -> None:
    """Remove a container if it exists."""
    proc = await asyncio.create_subprocess_exec(
        "podman-hpc", "rm", "-f", name,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


async def main() -> int:
    log("=" * 60)
    log("Podman-HPC Diagnostic Script")
    log("=" * 60)

    # Check environment
    log(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'not set')}")
    log(f"hostname: {os.uname().nodename}")

    # Check podman-hpc is available
    if not shutil.which("podman-hpc"):
        log("podman-hpc not found in PATH", "ERROR")
        return 1

    log("")
    log("Checking podman-hpc version...")
    proc = subprocess.run(
        ["podman-hpc", "version"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        for line in proc.stdout.strip().split("\n")[:5]:
            log(f"  {line}")
    else:
        log(f"  Failed to get version: {proc.stderr}")

    log("")
    log("Checking available images...")
    proc = subprocess.run(
        ["podman-hpc", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        images = proc.stdout.strip().split("\n")[:10]
        for img in images:
            log(f"  {img}")
        if len(proc.stdout.strip().split("\n")) > 10:
            log(f"  ... and more")
    else:
        log(f"  Failed to list images: {proc.stderr}")

    # Image to test with
    test_image = "python:3.11-slim"

    log("")
    log("=" * 60)
    log("Testing podman-hpc run configurations")
    log("=" * 60)

    test_container = f"ludic-diag-{os.getpid()}"
    results = {}

    # Test 1: Minimal run (no resource limits)
    await cleanup_container(test_container)
    success, stdout, stderr = await run_cmd(
        ["podman-hpc", "run", "--rm", test_image, "echo", "hello"],
        "Minimal run (--rm, no limits)",
    )
    results["minimal"] = success

    # Test 2: Detached mode
    await cleanup_container(test_container)
    success, stdout, stderr = await run_cmd(
        ["podman-hpc", "run", "-d", "--name", test_container, test_image, "sleep", "10"],
        "Detached mode (-d)",
    )
    results["detached"] = success
    await cleanup_container(test_container)

    # Test 3: Memory limit
    await cleanup_container(test_container)
    success, stdout, stderr = await run_cmd(
        ["podman-hpc", "run", "--rm", "--memory", "128m", test_image, "echo", "hello"],
        "Memory limit (--memory 128m)",
    )
    results["memory"] = success

    # Test 4: Network none
    await cleanup_container(test_container)
    success, stdout, stderr = await run_cmd(
        ["podman-hpc", "run", "--rm", "--network", "none", test_image, "echo", "hello"],
        "Network disabled (--network none)",
    )
    results["network_none"] = success

    # Test 5: CPU limit
    await cleanup_container(test_container)
    success, stdout, stderr = await run_cmd(
        ["podman-hpc", "run", "--rm", "--cpus", "0.5", test_image, "echo", "hello"],
        "CPU limit (--cpus 0.5)",
    )
    results["cpu"] = success

    # Test 6: Full combination (what we use by default)
    await cleanup_container(test_container)
    success, stdout, stderr = await run_cmd(
        [
            "podman-hpc", "run", "-d",
            "--name", test_container,
            "--memory", "128m",
            "--network", "none",
            test_image, "sleep", "infinity",
        ],
        "Full combination (detached + memory + network none)",
    )
    results["full"] = success
    await cleanup_container(test_container)

    # Test 7: Detached with sleep infinity only
    await cleanup_container(test_container)
    success, stdout, stderr = await run_cmd(
        ["podman-hpc", "run", "-d", "--name", test_container, test_image, "sleep", "infinity"],
        "Detached + sleep infinity (minimal persistent)",
    )
    results["persistent_minimal"] = success

    if success:
        # Test exec in the container
        log("")
        log("Testing exec in persistent container...")
        success2, stdout, stderr = await run_cmd(
            ["podman-hpc", "exec", test_container, "python", "-c", "print('exec works')"],
            "Exec Python in container",
        )
        results["exec"] = success2

        # Test file operations
        success3, stdout, stderr = await run_cmd(
            ["podman-hpc", "exec", test_container, "sh", "-c", "echo test > /tmp/test.txt && cat /tmp/test.txt"],
            "File write/read in container",
        )
        results["file_ops"] = success3

    await cleanup_container(test_container)

    # Summary
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        log(f"  {name}: {status}")
        if not passed:
            all_passed = False

    log("")
    if results.get("persistent_minimal") and results.get("exec"):
        log("RECOMMENDATION: Use minimal config (no memory/network limits)", "SUCCESS")
        log("  Set PodmanConfig(memory_limit=None, network_disabled=False)")
    elif results.get("minimal"):
        log("PARTIAL: Basic podman-hpc works but persistent containers may not", "WARN")
    else:
        log("podman-hpc appears non-functional on this system", "ERROR")

    log("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

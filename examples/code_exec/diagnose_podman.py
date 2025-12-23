#!/usr/bin/env python3
"""
Diagnostic script to test podman-hpc configuration on HPC clusters.

This script tests various podman-hpc run configurations to identify
which flags work on your specific HPC environment.

Usage:
    uv run python examples/code_exec/diagnose_podman.py

Known Issue (Isambard/BRiCS - December 2024):
---------------------------------------------
If ALL tests fail with "executable file not found in $PATH", this is likely
due to podman-hpc's squashfs image conversion breaking PATH environment setup.

The solution is to use absolute paths for all executables:
  - /bin/echo instead of echo
  - /bin/sleep instead of sleep
  - /usr/local/bin/python instead of python (for official Python images)

This issue is worked around in src/ludic/envs/code_exec/podman_sandbox.py.
See the module docstring and features/CodeExecEnv/plan.md for details.
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

    # =========================================================================
    # Test 8: Host bind mount (CRITICAL for I/O optimization)
    # =========================================================================
    log("")
    log("=" * 60)
    log("Testing HOST BIND MOUNT (potential 3x throughput improvement)")
    log("=" * 60)

    import tempfile
    import shutil as sh

    # Create a temp directory on the host
    bind_test_dir = tempfile.mkdtemp(prefix="ludic_bind_test_")
    log(f"Created host directory: {bind_test_dir}")

    try:
        await cleanup_container(test_container)

        # Test: Run container with bind mount and write a file from inside
        success, stdout, stderr = await run_cmd(
            [
                "podman-hpc", "run", "--rm",
                "-v", f"{bind_test_dir}:/workspace",
                test_image,
                "/usr/local/bin/python", "-c",
                "open('/workspace/test_from_container.txt', 'w').write('hello from container')"
            ],
            "Bind mount write test (-v host:container)",
        )
        results["bind_mount_write"] = success

        # Check if file appeared on host
        test_file = os.path.join(bind_test_dir, "test_from_container.txt")
        if os.path.exists(test_file):
            with open(test_file) as f:
                content = f.read()
            if content == "hello from container":
                log(f"  Host file verification: PASSED (content matches)")
                results["bind_mount_verify"] = True
            else:
                log(f"  Host file verification: FAILED (content mismatch: {content!r})")
                results["bind_mount_verify"] = False
        else:
            log(f"  Host file verification: FAILED (file not found on host)")
            results["bind_mount_verify"] = False

        # Test: Write from host, read from container
        host_write_file = os.path.join(bind_test_dir, "test_from_host.txt")
        with open(host_write_file, "w") as f:
            f.write("hello from host")

        success, stdout, stderr = await run_cmd(
            [
                "podman-hpc", "run", "--rm",
                "-v", f"{bind_test_dir}:/workspace",
                test_image,
                "/usr/local/bin/python", "-c",
                "print(open('/workspace/test_from_host.txt').read())"
            ],
            "Bind mount read test (host→container)",
        )
        results["bind_mount_read"] = success and "hello from host" in stdout
        if success and "hello from host" in stdout:
            log(f"  Container read verification: PASSED")
        else:
            log(f"  Container read verification: FAILED (stdout: {stdout[:100]})")

    finally:
        # Cleanup
        sh.rmtree(bind_test_dir, ignore_errors=True)
        log(f"Cleaned up: {bind_test_dir}")

    # =========================================================================
    # Test 9: Concurrent exec stress test
    # =========================================================================
    log("")
    log("=" * 60)
    log("Testing CONCURRENT EXEC (finding actual concurrency limit)")
    log("=" * 60)

    await cleanup_container(test_container)

    # Start a persistent container
    success, _, _ = await run_cmd(
        ["podman-hpc", "run", "-d", "--name", test_container, test_image, "sleep", "infinity"],
        "Starting persistent container for concurrency test",
    )

    if success:
        for concurrency in [8, 16, 24, 32]:
            log(f"  Testing {concurrency} concurrent exec calls...")

            async def run_one_exec(i: int) -> bool:
                proc = await asyncio.create_subprocess_exec(
                    "podman-hpc", "exec", test_container,
                    "/usr/local/bin/python", "-c", f"import time; time.sleep(0.5); print({i})",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                return proc.returncode == 0

            start = asyncio.get_event_loop().time()
            tasks = [run_one_exec(i) for i in range(concurrency)]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = asyncio.get_event_loop().time() - start

            successes = sum(1 for r in results_list if r is True)
            failures = sum(1 for r in results_list if r is False)
            exceptions = sum(1 for r in results_list if isinstance(r, Exception))

            status = "PASSED" if successes == concurrency else "PARTIAL" if successes > 0 else "FAILED"
            log(f"    {concurrency} concurrent: {status} ({successes}/{concurrency} ok, {failures} failed, {exceptions} exceptions) in {elapsed:.1f}s")

            results[f"concurrent_{concurrency}"] = (successes == concurrency)

            # If we start seeing failures, note the limit
            if successes < concurrency:
                log(f"    ⚠️  Concurrency limit appears to be around {successes}")
                break

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

    # Basic functionality
    if results.get("persistent_minimal") and results.get("exec"):
        log("RECOMMENDATION: Use minimal config (no memory/network limits)", "SUCCESS")
        log("  Set PodmanConfig(memory_limit=None, network_disabled=False)")
    elif results.get("minimal"):
        log("PARTIAL: Basic podman-hpc works but persistent containers may not", "WARN")
    else:
        log("podman-hpc appears non-functional on this system", "ERROR")

    # Bind mount recommendation
    log("")
    if results.get("bind_mount_write") and results.get("bind_mount_verify") and results.get("bind_mount_read"):
        log("🚀 BIND MOUNTS WORK! This enables major I/O optimization:", "SUCCESS")
        log("  - Write code directly to host filesystem (skip podman exec tar)")
        log("  - Reset via host rmtree (skip podman exec rm)")
        log("  - Potential 3x throughput improvement")
        log("  → Implement host-mounted workspace in PodmanHPCSandbox")
    else:
        log("Bind mounts not working - continue using tar-based I/O", "WARN")

    # Concurrency recommendation
    log("")
    max_working_concurrency = 0
    for c in [32, 24, 16, 8]:
        if results.get(f"concurrent_{c}"):
            max_working_concurrency = c
            break

    if max_working_concurrency > 0:
        log(f"CONCURRENCY: {max_working_concurrency} concurrent execs work reliably", "SUCCESS")
        log(f"  → Set max_concurrent_ops={max_working_concurrency} in create_sandbox_pool()")
    else:
        log("CONCURRENCY: Could not determine safe concurrency limit", "WARN")

    log("")
    log("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

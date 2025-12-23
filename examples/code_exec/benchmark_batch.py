#!/usr/bin/env python3
"""
Benchmark script comparing batch vs individual execution performance.

This script measures the latency reduction from batch execution by running
the same test problems through both execution paths and comparing timing.

Usage:
    # Local Docker (requires Docker running):
    python benchmark_batch.py --backend docker --n-workers 4 --n-problems 10

    # On Isambard compute node (via SLURM):
    srun python benchmark_batch.py --backend podman --n-workers 4 --n-problems 10

    # On Isambard login node (testing only - no cgroups):
    python benchmark_batch.py --backend podman --n-workers 4 --n-problems 10 --no-memory-limit

The benchmark reports:
  - Individual execution time (baseline)
  - Batch execution time
  - Speedup factor
  - Semaphore acquisition reduction (theoretical)
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import List

from ludic.envs.code_exec.types import TestCase
from ludic.envs.code_exec.runners import StdinStdoutRunner
from ludic.envs.code_exec.adapters.base import ExactMatchVerifier


def generate_test_problems(n_problems: int, tests_per_problem: int = 10) -> List[dict]:
    """
    Generate synthetic test problems for benchmarking.

    Each problem is a simple squaring function with multiple test cases.
    """
    problems = []
    for i in range(n_problems):
        code = f"""
import sys
n = int(input())
print(n ** 2)
"""
        tests = []
        for j in range(tests_per_problem):
            val = (i * tests_per_problem + j) % 100 + 1
            tests.append(
                TestCase(
                    id=f"problem_{i}_test_{j}",
                    input=f"{val}\n",
                    expected=f"{val ** 2}\n",
                )
            )
        problems.append({"code": code.strip(), "tests": tests})
    return problems


async def benchmark_individual(
    pool,
    problems: List[dict],
    runner: StdinStdoutRunner,
    verifier: ExactMatchVerifier,
) -> float:
    """Run problems using individual execution and return total time."""
    start = time.perf_counter()

    for problem in problems:
        sandbox = await pool.checkout()
        try:
            result = await runner.run_tests(
                sandbox=sandbox,
                code=problem["code"],
                tests=problem["tests"],
                verifier=verifier,
                stop_on_first_failure=True,
                compile_first=True,
            )
            if not result.all_passed:
                print(f"  Warning: Problem failed - {result.first_failure}")
        finally:
            await pool.release(sandbox)

    return time.perf_counter() - start


async def benchmark_batch(
    pool,
    problems: List[dict],
    runner: StdinStdoutRunner,
    verifier: ExactMatchVerifier,
) -> float:
    """Run problems using batch execution and return total time."""
    start = time.perf_counter()

    for problem in problems:
        sandbox = await pool.checkout()
        try:
            result = await runner.run_tests(
                sandbox=sandbox,
                code=problem["code"],
                tests=problem["tests"],
                verifier=verifier,
                stop_on_first_failure=True,
                compile_first=True,
            )
            if not result.all_passed:
                print(f"  Warning: Problem failed - {result.first_failure}")
        finally:
            await pool.release(sandbox)

    return time.perf_counter() - start


async def run_benchmark(
    backend: str,
    n_workers: int,
    n_problems: int,
    tests_per_problem: int,
    no_memory_limit: bool = False,
) -> None:
    """Main benchmark function."""
    print(f"Benchmark Configuration:")
    print(f"  Backend: {backend}")
    print(f"  Workers: {n_workers}")
    print(f"  Problems: {n_problems}")
    print(f"  Tests per problem: {tests_per_problem}")
    if no_memory_limit:
        print(f"  Memory limit: DISABLED (login node mode)")
    print()

    # Generate test problems
    print("Generating test problems...")
    problems = generate_test_problems(n_problems, tests_per_problem)
    total_tests = n_problems * tests_per_problem
    print(f"  Total tests: {total_tests}")
    print()

    # Create pool based on backend
    if backend == "docker":
        from ludic.envs.code_exec.docker_sandbox import DockerSandboxPool

        pool = DockerSandboxPool(n_workers=n_workers)
    elif backend == "podman":
        from ludic.envs.code_exec.podman_sandbox import (
            PodmanHPCSandboxPool,
            PodmanConfig,
        )

        config = None
        if no_memory_limit:
            # Disable memory limits for login node testing (cgroups not available)
            config = PodmanConfig(memory_limit="")

        pool = PodmanHPCSandboxPool(n_workers=n_workers, config=config)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    verifier = ExactMatchVerifier()

    print("Starting sandbox pool...")
    await pool.start()
    print(f"  Pool started with {n_workers} workers")
    print()

    try:
        # Warm-up run
        print("Warming up (1 problem each mode)...")
        warmup_problems = problems[:1]

        runner_individual = StdinStdoutRunner(use_batch_execution=False)
        await benchmark_individual(pool, warmup_problems, runner_individual, verifier)

        runner_batch = StdinStdoutRunner(use_batch_execution=True)
        await benchmark_batch(pool, warmup_problems, runner_batch, verifier)
        print("  Warm-up complete")
        print()

        # Benchmark individual execution
        print("Benchmarking INDIVIDUAL execution...")
        runner_individual = StdinStdoutRunner(use_batch_execution=False)
        individual_time = await benchmark_individual(
            pool, problems, runner_individual, verifier
        )
        print(f"  Time: {individual_time:.2f}s")
        print()

        # Benchmark batch execution
        print("Benchmarking BATCH execution...")
        runner_batch = StdinStdoutRunner(use_batch_execution=True)
        batch_time = await benchmark_batch(pool, problems, runner_batch, verifier)
        print(f"  Time: {batch_time:.2f}s")
        print()

        # Results
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Individual execution: {individual_time:.2f}s")
        print(f"Batch execution:      {batch_time:.2f}s")

        if batch_time > 0:
            speedup = individual_time / batch_time
            print(f"Speedup:              {speedup:.2f}x")

            # Calculate theoretical semaphore reduction
            # Individual: 2N+3 per problem (compile + N tests + reset)
            # Batch: 3 per problem (tar + exec + reset)
            individual_acquisitions = (2 * tests_per_problem + 3) * n_problems
            batch_acquisitions = 3 * n_problems
            reduction = (
                (individual_acquisitions - batch_acquisitions)
                / individual_acquisitions
                * 100
            )
            print()
            print(f"Semaphore acquisitions (theoretical):")
            print(f"  Individual: {individual_acquisitions}")
            print(f"  Batch:      {batch_acquisitions}")
            print(f"  Reduction:  {reduction:.1f}%")
        print("=" * 60)

    finally:
        print()
        print("Shutting down pool...")
        await pool.shutdown()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark batch vs individual execution performance"
    )
    parser.add_argument(
        "--backend",
        choices=["docker", "podman"],
        default="docker",
        help="Container backend to use (default: docker)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of sandbox workers (default: 4)",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=10,
        help="Number of problems to run (default: 10)",
    )
    parser.add_argument(
        "--tests-per-problem",
        type=int,
        default=10,
        help="Number of tests per problem (default: 10)",
    )
    parser.add_argument(
        "--no-memory-limit",
        action="store_true",
        help="Disable container memory limits (required for login nodes without cgroup support)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            backend=args.backend,
            n_workers=args.n_workers,
            n_problems=args.n_problems,
            tests_per_problem=args.tests_per_problem,
            no_memory_limit=args.no_memory_limit,
        )
    )


if __name__ == "__main__":
    main()

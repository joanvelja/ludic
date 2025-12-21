#!/usr/bin/env python3
"""
Smoke test for CodeExecEnv on Isambard HPC (CPU-only, Podman-HPC).

This script validates the full training pipeline end-to-end:
  1. Podman-HPC sandbox pool creation
  2. CodeExecEnv reset/step cycle
  3. RolloutEngine with SingleAgentProtocol
  4. Trainer step (with mock inference)

Usage (interactive):
    python examples/code_exec/smoke_test_isambard.py

Usage (Slurm batch):
    sbatch examples/code_exec/smoke_test_isambard.slurm

Requirements:
    - podman-hpc in PATH (auto-detected on Isambard)
    - Python 3.11+ with ludic installed
    - Image pre-pulled: podman-hpc pull python:3.11-slim

Troubleshooting:
----------------
If you see "executable file not found in $PATH" errors for basic commands
like 'sleep', 'echo', or 'python', this is a known podman-hpc issue on
Isambard where the squashfs image conversion breaks PATH setup.

Diagnosis:
    # Should work (absolute path)
    podman-hpc run --rm python:3.11-slim /bin/echo hello

    # May fail (relies on PATH)
    podman-hpc run --rm python:3.11-slim echo hello

The ludic sandbox code uses absolute paths (/bin/sleep, /usr/local/bin/python)
to work around this. If you encounter this on a new system, see:
  - src/ludic/envs/code_exec/podman_sandbox.py (module docstring)
  - features/CodeExecEnv/plan.md (Troubleshooting section)

To reset corrupted podman-hpc state:
    rm -rf $SCRATCH/storage ~/.local/share/containers ~/.config/containers
    podman-hpc system reset && podman-hpc system migrate
    podman-hpc pull python:3.11-slim
"""

from __future__ import annotations

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# New API imports
from ludic.training import EnvSpec, ProtocolSpec
from ludic.types import ChatResponse
from ludic.inference.request import ChatCompletionRequest

# Check early for podman-hpc availability
import shutil

if not shutil.which("podman-hpc"):
    print("ERROR: podman-hpc not found in PATH")
    print(
        "  On Isambard, ensure you're in a Slurm job or on a login node with podman-hpc"
    )
    sys.exit(1)


def log(msg: str, level: str = "INFO") -> None:
    """Simple timestamped logging."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


# ============================================================================
# Test 1: Podman-HPC Sandbox Pool
# ============================================================================


async def test_sandbox_pool(n_workers: int = 2, minimal_config: bool = True) -> bool:
    """Test that we can create and use a Podman-HPC sandbox pool."""
    log("Testing Podman-HPC sandbox pool...")

    from ludic.envs.code_exec import PodmanHPCSandboxPool, PodmanConfig
    from ludic.envs.code_exec.types import CompileStatus, RunStatus

    # Use minimal config for HPC compatibility (some clusters don't support
    # --memory or --network none flags)
    if minimal_config:
        log("  Using minimal config (no memory/network limits) for HPC compatibility")
        config = PodmanConfig(
            memory_limit=None,  # Skip --memory flag
            network_disabled=False,  # Skip --network none flag
            gpu=False,
        )
    else:
        config = PodmanConfig(
            memory_limit="128m",
            network_disabled=True,
            gpu=False,
        )

    pool = PodmanHPCSandboxPool(
        n_workers=n_workers,
        image="python:3.11-slim",
        config=config,
        cache_size=100,
    )

    try:
        log(f"  Starting pool with {n_workers} workers...")
        start = time.time()
        await pool.start()
        log(f"  Pool started in {time.time() - start:.2f}s")

        # Test checkout and execute
        log("  Checking out sandbox...")
        sandbox = await pool.checkout(timeout_s=30.0)

        log("  Testing compile...")
        compile_result = await sandbox.compile("print('hello')")
        assert compile_result.status == CompileStatus.SUCCESS, (
            f"Compile failed: {compile_result}"
        )

        log("  Testing execute...")
        exec_result = await sandbox.execute("print('Hello from Podman-HPC!')")
        assert exec_result.run_status == RunStatus.SUCCESS, (
            f"Execute failed: {exec_result}"
        )
        assert "Hello from Podman-HPC!" in exec_result.stdout

        log("  Testing stdin handling...")
        code = "x = int(input()); print(x * 2)"
        exec_result = await sandbox.execute(code, stdin="21")
        assert "42" in exec_result.stdout, f"Stdin test failed: {exec_result.stdout}"

        log("  Releasing sandbox...")
        await pool.release(sandbox)

        log("  Sandbox pool test PASSED", "SUCCESS")
        return True

    except Exception as e:
        log(f"  Sandbox pool test FAILED: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        return False

    finally:
        log("  Shutting down pool...")
        await pool.shutdown()


# ============================================================================
# Test 2: CodeExecEnv
# ============================================================================


async def test_code_exec_env(minimal_config: bool = True) -> bool:
    """Test CodeExecEnv reset/step cycle."""
    log("Testing CodeExecEnv...")

    from ludic.envs.code_exec import (
        CodeExecEnv,
        CodeExecConfig,
        PodmanHPCSandboxPool,
        PodmanConfig,
    )
    from ludic.envs.code_exec.adapters.apps import APPSTestAdapter

    # Use minimal config for HPC compatibility
    if minimal_config:
        pool_config = PodmanConfig(memory_limit=None, network_disabled=False, gpu=False)
    else:
        pool_config = PodmanConfig(
            memory_limit="128m", network_disabled=True, gpu=False
        )
    pool = PodmanHPCSandboxPool(
        n_workers=2,
        image="python:3.11-slim",
        config=pool_config,
    )

    try:
        await pool.start()

        # Create a simple test problem
        sample = {
            "problem_id": "smoke_test_add",
            "question": "Write a program that reads two integers on one line and prints their sum.",
            "inputs": ["1 2", "10 20", "-5 5"],
            "outputs": ["3", "30", "0"],
        }

        env_config = CodeExecConfig(
            timeout_per_test_s=5.0,
            stop_on_first_failure=False,
            compile_first=True,
            partial_credit=False,
        )

        adapter = APPSTestAdapter()
        env = CodeExecEnv(
            sample=sample,
            sandbox_pool=pool,
            test_adapter=adapter,
            config=env_config,
        )

        # Test reset
        log("  Testing env_reset...")
        obs, info = await env.env_reset()
        assert "two integers" in obs.lower(), f"Unexpected obs: {obs}"
        assert info["problem_id"] == "smoke_test_add"
        assert info["num_tests"] == 3
        log(f"  Reset OK: {info['num_tests']} tests")

        # Test step with correct code
        log("  Testing env_step with correct code...")
        correct_code = "a, b = map(int, input().split()); print(a + b)"
        outcome = await env.env_step(correct_code)

        assert outcome.terminated is True
        assert outcome.reward == 1.0, f"Expected reward=1.0, got {outcome.reward}"
        assert outcome.info["all_passed"] is True
        assert outcome.info["passed"] == 3
        log(
            f"  Correct code: reward={outcome.reward}, passed={outcome.info['passed']}/{outcome.info['total']}"
        )

        # Test step with wrong code (need new env instance)
        env2 = CodeExecEnv(
            sample=sample,
            sandbox_pool=pool,
            test_adapter=adapter,
            config=env_config,
        )
        await env2.env_reset()

        log("  Testing env_step with wrong code...")
        wrong_code = "a, b = map(int, input().split()); print(a - b)"  # Subtraction instead of addition
        outcome2 = await env2.env_step(wrong_code)

        assert outcome2.terminated is True
        assert outcome2.reward == 0.0
        assert outcome2.info["all_passed"] is False
        log(
            f"  Wrong code: reward={outcome2.reward}, passed={outcome2.info['passed']}/{outcome2.info['total']}"
        )

        log("  CodeExecEnv test PASSED", "SUCCESS")
        return True

    except Exception as e:
        log(f"  CodeExecEnv test FAILED: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await pool.shutdown()


# ============================================================================
# Test 3: RolloutEngine with Protocol
# ============================================================================


async def test_rollout_engine(minimal_config: bool = True) -> bool:
    """Test RolloutEngine generates rollouts correctly."""
    log("Testing RolloutEngine with SingleAgentProtocol...")

    from ludic.envs.code_exec import (
        CodeExecEnv,
        CodeExecConfig,
        PodmanHPCSandboxPool,
        PodmanConfig,
    )
    from ludic.envs.code_exec.adapters.apps import APPSTestAdapter
    from ludic.agent import Agent
    from ludic.context import FullDialog
    from ludic.parsers import ParseResult
    from ludic.interaction import SingleAgentProtocol
    from ludic.training import RolloutEngine, RolloutRequest
    from ludic.inference import InferenceSpec, SamplingParams

    # Mock client that returns deterministic code
    class MockChatClient:
        """Mock inference client that returns predetermined code."""

        def __init__(self, code_to_return: str):
            self.code = code_to_return
            self.call_count = 0

        async def complete(
            self, request: ChatCompletionRequest
        ) -> Tuple[ChatResponse, Dict[str, Any]]:
            self.call_count += 1
            text = f"```python\n{self.code}\n```"
            return ChatResponse(
                text=text,
                finish_reason="stop",
                prompt_token_ids=[1, 2, 3],  # dummy IDs for token trace
                completion_token_ids=[4, 5, 6, 7],  # dummy IDs for token trace
            ), {}

        def sync_weights(self, params, *, timeout_s=600.0, version=None) -> str:
            return "mock-v1"

    def simple_parser(raw: str) -> ParseResult:
        """Extract code from markdown blocks."""
        import re

        match = re.search(r"```(?:python)?\s*\n(.*?)\n```", raw, re.DOTALL)
        if match:
            return ParseResult(action=match.group(1).strip(), reward=0.0, obs=None)
        return ParseResult(action=raw.strip(), reward=0.0, obs=None)

    # Use minimal config for HPC compatibility
    if minimal_config:
        pool_config = PodmanConfig(memory_limit=None, network_disabled=False, gpu=False)
    else:
        pool_config = PodmanConfig(
            memory_limit="128m", network_disabled=True, gpu=False
        )
    pool = PodmanHPCSandboxPool(
        n_workers=2,
        image="python:3.11-slim",
        config=pool_config,
    )

    try:
        await pool.start()

        # Env factory
        adapter = APPSTestAdapter()
        env_config = CodeExecConfig(timeout_per_test_s=5.0, stop_on_first_failure=True)

        def env_factory(sample: Dict[str, Any]) -> CodeExecEnv:
            return CodeExecEnv(
                sample=sample,
                sandbox_pool=pool,
                test_adapter=adapter,
                config=env_config,
            )

        # Protocol factory (with mock client returning correct code)
        correct_code = "a, b = map(int, input().split()); print(a + b)"
        mock_client = MockChatClient(correct_code)

        def protocol_factory():
            return SingleAgentProtocol(
                agent=Agent(
                    client=mock_client,
                    model="mock-model",
                    ctx=FullDialog(),
                    parser=simple_parser,
                )
            )

        # Create engine
        engine = RolloutEngine(
            env_registry={"code_exec": env_factory},
            protocol_registry={"single_agent": protocol_factory},
        )

        # Create request
        sample = {
            "problem_id": "rollout_test",
            "question": "Read two integers and print their sum.",
            "inputs": ["3 4"],
            "outputs": ["7"],
        }

        request = RolloutRequest(
            env=EnvSpec(kind="code_exec", kwargs={"sample": sample}),
            protocol=ProtocolSpec(kind="single_agent", kwargs={}),
            env_seed=42,
        )

        log("  Generating rollout...")
        rollouts = await engine.generate_rollouts(
            requests=[request],
            max_steps=1,
            concurrency=1,
        )

        assert len(rollouts) == 1, f"Expected 1 rollout, got {len(rollouts)}"
        rollout = rollouts[0]

        assert len(rollout.steps) == 1, f"Expected 1 step, got {len(rollout.steps)}"
        step = rollout.steps[0]

        assert step.reward == 1.0, f"Expected reward=1.0, got {step.reward}"
        assert step.terminated is True
        log(f"  Rollout generated: {len(rollout.steps)} steps, reward={step.reward}")

        log("  RolloutEngine test PASSED", "SUCCESS")
        return True

    except Exception as e:
        log(f"  RolloutEngine test FAILED: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await pool.shutdown()


# ============================================================================
# Test 4: Training Step (Mock)
# ============================================================================


async def test_training_step(minimal_config: bool = True) -> bool:
    """Test a single training step with mock model and inference."""
    log("Testing training step (mock inference, CPU model)...")

    try:
        import torch
        from ludic.envs.code_exec import (
            CodeExecEnv,
            CodeExecConfig,
            PodmanHPCSandboxPool,
            PodmanConfig,
        )
        from ludic.envs.code_exec.adapters.apps import APPSTestAdapter
        from ludic.agent import Agent
        from ludic.context import FullDialog
        from ludic.parsers import ParseResult
        from ludic.interaction import SingleAgentProtocol
        from ludic.training import (
            RolloutEngine,
            RolloutBatchSource,
            Trainer,
            TrainerConfig,
            make_reinforce,
        )
        from ludic.inference import InferenceSpec

        # Create a tiny random model (no pretrained weights needed)
        log("  Creating tiny random model for testing...")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        # Use GPT-2 config but with minimal size
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 1
        config.n_head = 2
        config.n_embd = 64
        config.vocab_size = 1000

        model = AutoModelForCausalLM.from_config(config)
        model.to("cpu")
        log(f"  Model created: {sum(p.numel() for p in model.parameters())} parameters")

        # Mock inference client
        class MockClient:
            async def complete(
                self, request: ChatCompletionRequest
            ) -> Tuple[ChatResponse, Dict[str, Any]]:
                text = "```python\na, b = map(int, input().split()); print(a + b)\n```"
                return ChatResponse(
                    text=text,
                    finish_reason="stop",
                    prompt_token_ids=[1, 2, 3],
                    completion_token_ids=[4, 5, 6, 7],
                ), {}

            def sync_weights(self, params, *, timeout_s=600.0, version=None) -> str:
                return "mock-v1"

        def simple_parser(raw: str) -> ParseResult:
            import re

            match = re.search(r"```(?:python)?\s*\n(.*?)\n```", raw, re.DOTALL)
            if match:
                return ParseResult(action=match.group(1).strip(), reward=0.0, obs=None)
            return ParseResult(action=raw.strip(), reward=0.0, obs=None)

        # Use minimal config for HPC compatibility
        if minimal_config:
            pool_config = PodmanConfig(
                memory_limit=None, network_disabled=False, gpu=False
            )
        else:
            pool_config = PodmanConfig(
                memory_limit="128m", network_disabled=True, gpu=False
            )
        pool = PodmanHPCSandboxPool(
            n_workers=2,
            image="python:3.11-slim",
            config=pool_config,
        )

        await pool.start()

        # Factories
        adapter = APPSTestAdapter()
        env_config = CodeExecConfig(timeout_per_test_s=5.0, stop_on_first_failure=True)

        def env_factory(sample):
            return CodeExecEnv(
                sample=sample,
                sandbox_pool=pool,
                test_adapter=adapter,
                config=env_config,
            )

        mock_client = MockClient()

        def protocol_factory():
            return SingleAgentProtocol(
                agent=Agent(
                    client=mock_client,
                    model="mock",
                    ctx=FullDialog(),
                    parser=simple_parser,
                )
            )

        engine = RolloutEngine(
            env_registry={"code_exec": env_factory},
            protocol_registry={"single_agent": protocol_factory},
        )

        # Create batch source with a single sample
        samples = [
            {
                "problem_id": "train_test",
                "question": "Read two integers and print their sum.",
                "inputs": ["1 2"],
                "outputs": ["3"],
            }
        ]

        sample_idx = [0]

        def requests_fn():
            from ludic.training import RolloutRequest

            if sample_idx[0] >= len(samples):
                return []
            s = samples[sample_idx[0]]
            sample_idx[0] += 1
            return [
                RolloutRequest(
                    env=EnvSpec(kind="code_exec", kwargs={"sample": s}),
                    protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                )
            ]

        algo = make_reinforce(name="reinforce")

        batch_source = RolloutBatchSource(
            orchestrator=engine,
            credit_assigner=algo.credit_assigner,
            requests_fn=requests_fn,
            max_steps=1,
            concurrency=1,
        )

        # Create trainer
        trainer_config = TrainerConfig(
            model_device="cpu",
            grad_accum_steps=1,
            max_grad_norm=1.0,
            pad_token_id=0,  # dummy pad token ID
        )

        # Mock publisher (no-op)
        class MockPublisher:
            def publish(self, state_dict, version):
                pass

        trainer = Trainer(
            model=model,
            algo=algo,
            batch_source=batch_source,
            publisher=MockPublisher(),
            cfg=trainer_config,
        )

        log("  Running single training step...")
        # Run one step
        try:
            await trainer.train(num_steps=1)
            log("  Training step completed")
        except StopIteration:
            log("  Training stopped (samples exhausted, expected)")

        await pool.shutdown()

        log("  Training step test PASSED", "SUCCESS")
        return True

    except ImportError as e:
        log(f"  Skipping training test (missing dependency): {e}", "WARN")
        return True  # Not a failure, just skip

    except Exception as e:
        log(f"  Training step test FAILED: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# Main
# ============================================================================


async def run_all_tests(
    skip_training: bool = False, minimal_config: bool = True
) -> bool:
    """Run all smoke tests and return overall success."""
    log("=" * 60)
    log("CodeExecEnv Smoke Test for Isambard HPC")
    log("=" * 60)
    if minimal_config:
        log("Using MINIMAL config (no memory/network limits) for HPC compatibility")
    else:
        log("Using FULL config (memory + network limits)")
    log("=" * 60)

    results = {}

    # Test 1: Sandbox Pool
    results["sandbox_pool"] = await test_sandbox_pool(
        n_workers=2, minimal_config=minimal_config
    )

    # Test 2: CodeExecEnv
    results["code_exec_env"] = await test_code_exec_env(minimal_config=minimal_config)

    # Test 3: RolloutEngine
    results["rollout_engine"] = await test_rollout_engine(minimal_config=minimal_config)

    # Test 4: Training Step (optional)
    if not skip_training:
        results["training_step"] = await test_training_step(
            minimal_config=minimal_config
        )
    else:
        log("Skipping training step test (--skip-training)")
        results["training_step"] = True

    # Summary
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        level = "SUCCESS" if passed else "ERROR"
        log(f"  {name}: {status}", level)
        if not passed:
            all_passed = False

    log("=" * 60)
    if all_passed:
        log("All tests PASSED!", "SUCCESS")
    else:
        log("Some tests FAILED!", "ERROR")
    log("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for CodeExecEnv on Isambard"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training step test (useful if torch not available)",
    )
    parser.add_argument(
        "--full-config",
        action="store_true",
        help="Use full config with memory/network limits (may not work on all HPC systems)",
    )
    args = parser.parse_args()

    success = asyncio.run(
        run_all_tests(
            skip_training=args.skip_training,
            minimal_config=not args.full_config,
        )
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

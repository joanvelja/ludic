"""PVG prover training component."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ludic.training import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.batching.intra_batch_control import GRPORequestStrategy
from ludic.inference import ReturnSpec, VLLMClient
from ludic.distributed.adapters import create_vllm_publisher
from ludic.pvg.scoring import VerifierScorer, VLLMRewardModelClient
from ludic.pvg.orchestrator import PVGOrchestrator, PVGState
from ludic.pvg.prover_trainer import ProverTrainingConfig, train_prover_phase
from ludic.pvg.manifest import (
    ComponentManifest,
    build_manifest_path,
    compute_config_hash,
    new_run_id,
    now_iso,
    read_manifest,
    write_manifest,
)
from ludic.pvg.components.common import (
    SNEAKY_JSON_SYSTEM_PROMPT,
    build_game_config,
    build_inference_spec,
    build_rollout_engine,
    get_git_sha,
    get_reward_strategy,
    load_prover_model,
    prepare_apps_samples,
    read_dataset_options,
)
from ludic.envs.code_exec import CodeExecConfig, SneakyConfig, create_sandbox_pool

logger = logging.getLogger(__name__)


def run_train_prover(
    *,
    config_path: Path,
    sync_verifier_manifest: Path,
    round_id: int,
    host: str,
    prover_port: int,
    verifier_port: int,
    output_dir: Optional[Path] = None,
    reward_strategy: str = "composite",
    group_size: int = 4,
    sync_every_steps: int = 1,
    concurrency: int = 32,
    device: str = "cuda",
) -> Path:
    config = build_game_config(config_path)
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    started_at = now_iso()
    git_sha = get_git_sha()
    config_hash = compute_config_hash(config)

    _ = read_manifest(sync_verifier_manifest)

    round_config = config.get_round_config(round_id)

    dataset, dataset_subset, max_samples = read_dataset_options(config_path)
    if dataset != "apps":
        raise RuntimeError(f"Dataset {dataset!r} not supported for prover training yet")

    problem_samples, adapter, honest_codes = prepare_apps_samples(
        limit=max_samples,
        subset=dataset_subset,
    )

    vllm_client = VLLMClient(
        host=host,
        policy_port=prover_port,
        scoring_port=verifier_port,
        enable_weight_updates=True,
        model_name=config.prover_model_path,
    )
    reward_client = VLLMRewardModelClient(
        vllm_client,
        model=config.verifier_model_path,
    )
    verifier_scorer = VerifierScorer(
        client=reward_client,
        batch_size=32,
        cache_max_size=50000,
        cache_ttl_s=None,
        cache_namespace=f"{config.verifier_model_path}:round_{round_id}",
    )

    sandbox_backend = os.getenv("LUDIC_SANDBOX_BACKEND", "auto")
    sandbox_pool = _run_async(
        create_sandbox_pool(
            n_workers=concurrency,
            backend=sandbox_backend,
            python_version="3.11",
            max_concurrent_ops=max(4, concurrency // 2),
            cache_size=10000,
        )
    )
    code_exec_config = CodeExecConfig(
        timeout_per_test_s=5.0,
        stop_on_first_failure=False,
        compile_first=True,
        partial_credit=False,
        compile_failure_reward=-0.5,
        use_cache=True,
    )

    rollout_engine = build_rollout_engine(
        client=vllm_client,
        prover_model=config.prover_model_path,
        adapter=adapter,
        honest_codes=honest_codes,
        sandbox_pool=sandbox_pool,
        sneaky_config=SneakyConfig(sequential_observation=round_config.sequential_observation),
        code_exec_config=code_exec_config,
        system_prompt=SNEAKY_JSON_SYSTEM_PROMPT,
    )

    prover_model, prover_tokenizer = _run_async(
        load_prover_model(
            config.prover_model_path,
            device=device,
        )
    )

    prover_config = ProverTrainingConfig(
        max_steps=round_config.prover_steps,
        reward_strategy=get_reward_strategy(reward_strategy),
        group_size=group_size,
        sync_every_steps=sync_every_steps,
        max_rollouts_per_step=concurrency,
    )

    if prover_tokenizer is not None:
        prover_config.pad_token_id = prover_tokenizer.pad_token_id or 0

    inference_spec_rl = build_inference_spec(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        return_spec=ReturnSpec.for_rl(),
    )

    grpo_strategy = GRPORequestStrategy(group_size=prover_config.group_size)

    def requests_fn() -> List[RolloutRequest]:
        requests: List[RolloutRequest] = []
        for _ in range(prover_config.batch_size):
            sample = random.choice(problem_samples)
            problem_id = adapter.get_problem_id(sample)
            prompt = adapter.get_prompt(sample)
            base_request = RolloutRequest(
                env=EnvSpec(kind="pvg_sneaky", kwargs={"sample": sample}),
                protocol=ProtocolSpec(
                    kind="pvg_sneaky",
                    kwargs={"sample": sample, "prompt": prompt},
                ),
                num_episodes=1,
                inference=inference_spec_rl,
                meta={
                    "problem_id": problem_id,
                    "problem": prompt,
                },
            )
            requests.append(base_request)
        return grpo_strategy.expand(requests)

    prover_publisher = create_vllm_publisher(vllm_client)

    checkpoint_path = _run_async(
        train_prover_phase(
            model=prover_model,
            tokenizer=prover_tokenizer,
            rollout_engine=rollout_engine,
            requests_fn=requests_fn,
            verifier_scorer=verifier_scorer,
            current_round=round_id,
            config=prover_config,
            output_dir=config.output_dir,
            publisher=prover_publisher,
        )
    )

    orchestrator = PVGOrchestrator(output_dir=config.output_dir, config=config)
    orchestrator.resume_from_checkpoint(validate_config=False)
    orchestrator.record_phase_checkpoint(
        "prover",
        checkpoint_path if checkpoint_path is not None else config.output_dir,
    )
    if orchestrator.can_transition(PVGState.CHECKPOINT):
        orchestrator.transition(PVGState.CHECKPOINT)

    manifest = ComponentManifest(
        component_name="train_prover",
        run_id=run_id,
        git_sha=git_sha,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=now_iso(),
        inputs={
            "sync_verifier_manifest": str(sync_verifier_manifest),
            "round_id": round_id,
            "host": host,
            "prover_port": prover_port,
            "verifier_port": verifier_port,
        },
        outputs={
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        },
        metrics={
            "prover_steps": prover_config.max_steps,
            "group_size": prover_config.group_size,
            "sync_every_steps": prover_config.sync_every_steps,
        },
        round_id=round_id,
    )

    manifest_path = build_manifest_path(
        config.output_dir, "train_prover", run_id, round_id=round_id
    )
    write_manifest(manifest_path, manifest)
    logger.info("Train prover manifest written to %s", manifest_path)

    return manifest_path


def _run_async(awaitable):
    import asyncio

    return asyncio.run(awaitable)

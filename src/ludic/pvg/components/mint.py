"""PVG data minting component."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ludic.training import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.pvg.config import PVGGameConfig
from ludic.pvg.data import PreferencePairBuilder, RolloutRecord, RoundDataStore
from ludic.pvg.orchestrator import PVGOrchestrator, PVGState
from ludic.pvg.minting import MintingConfig, clone_honest_rollouts_for_round, mint_sneaky_data
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
    build_game_config,
    build_inference_spec,
    build_rollout_engine,
    get_git_sha,
    prepare_apps_samples,
    SNEAKY_JSON_SYSTEM_PROMPT,
    read_dataset_options,
)
from ludic.inference import VLLMClient
from ludic.envs.code_exec import CodeExecConfig, SneakyConfig, create_sandbox_pool
from ludic.types import Rollout

logger = logging.getLogger(__name__)


def _read_base_honest_rollouts(path: Path) -> List[RolloutRecord]:
    records: List[RolloutRecord] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(RolloutRecord.from_dict(json.loads(line)))
    return records


def run_mint(
    *,
    config_path: Path,
    bootstrap_manifest: Path,
    round_id: int,
    host: str,
    prover_port: int,
    output_dir: Optional[Path] = None,
    concurrency: int = 32,
) -> Path:
    config = build_game_config(config_path)
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    started_at = now_iso()
    git_sha = get_git_sha()
    config_hash = compute_config_hash(config)

    bootstrap = read_manifest(bootstrap_manifest)
    base_honest_path = Path(bootstrap.outputs["base_honest_rollouts"])

    # Load dataset + honest solutions
    dataset, dataset_subset, max_samples = read_dataset_options(config_path)
    if dataset != "apps":
        raise RuntimeError(f\"Dataset {dataset!r} not supported for minting yet\")\n\n    problem_samples, adapter, honest_codes = prepare_apps_samples(\n        limit=max_samples,\n        subset=dataset_subset,\n    )

    sample_by_id: Dict[str, Dict[str, Any]] = {
        adapter.get_problem_id(s): s for s in problem_samples
    }

    base_honest_rollouts = _read_base_honest_rollouts(base_honest_path)

    # Setup vLLM client
    vllm_client = VLLMClient(
        host=host,
        policy_port=prover_port,
        enable_weight_updates=False,
        model_name=config.prover_model_path,
    )

    # Setup sandbox + rollout engine
    sandbox_backend = os.getenv("LUDIC_SANDBOX_BACKEND", "auto")
    sandbox_pool = asyncio.run(
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

    sneaky_config = SneakyConfig(sequential_observation=round_id > 0)
    rollout_engine = build_rollout_engine(
        client=vllm_client,
        prover_model=config.prover_model_path,
        adapter=adapter,
        honest_codes=honest_codes,
        sandbox_pool=sandbox_pool,
        sneaky_config=sneaky_config,
        code_exec_config=code_exec_config,
        system_prompt=SNEAKY_JSON_SYSTEM_PROMPT,
    )

    inference_spec = build_inference_spec(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
    )

    async def rollout_generator(prompt: str, problem_id: str) -> Rollout:
        sample = sample_by_id.get(problem_id)
        if sample is None:
            return Rollout(steps=[], meta={"problem_id": problem_id})

        request = RolloutRequest(
            env=EnvSpec(kind="pvg_sneaky", kwargs={"sample": sample}),
            protocol=ProtocolSpec(
                kind="pvg_sneaky",
                kwargs={"sample": sample, "prompt": prompt},
            ),
            num_episodes=1,
            env_seed=round_id,
            sampling_seed=round_id,
            inference=inference_spec,
            meta={
                "problem_id": problem_id,
                "problem": adapter.get_prompt(sample),
            },
        )
        rollouts = await rollout_engine.generate_rollouts(
            requests=[request],
            max_steps=1,
            concurrency=1,
        )
        if not rollouts:
            return Rollout(steps=[], meta={"problem_id": problem_id})
        return rollouts[0]

    async def rollout_batch_generator(
        batch: List[Tuple[Dict[str, Any], str, str]]
    ) -> List[Rollout]:
        requests: List[RolloutRequest] = []
        for sample, prompt, problem_id in batch:
            requests.append(
                RolloutRequest(
                    env=EnvSpec(kind="pvg_sneaky", kwargs={"sample": sample}),
                    protocol=ProtocolSpec(
                        kind="pvg_sneaky",
                        kwargs={"sample": sample, "prompt": prompt},
                    ),
                    num_episodes=1,
                    env_seed=round_id,
                    sampling_seed=round_id,
                    inference=inference_spec,
                    meta={
                        "problem_id": problem_id,
                        "problem": adapter.get_prompt(sample),
                    },
                )
            )

        rollouts = await rollout_engine.generate_rollouts(
            requests=requests,
            max_steps=1,
            concurrency=concurrency,
        )
        if len(rollouts) < len(requests):
            for _ in range(len(requests) - len(rollouts)):
                rollouts.append(Rollout(steps=[], meta={}))
        return rollouts

    minting_config = MintingConfig(
        strategy="few_shot" if round_id == 0 else "standard",
        max_samples=min(1000, len(problem_samples)),
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        concurrent_rollouts=concurrency,
        batch_size=min(32, max(2, concurrency // 2)),
    )

    minting_result = asyncio.run(
        mint_sneaky_data(
            problem_samples=problem_samples,
            honest_codes=honest_codes,
            rollout_generator=rollout_generator,
            rollout_batch_generator=rollout_batch_generator,
            current_round=round_id,
            config=minting_config,
        )
    )

    data_store = RoundDataStore(config.output_dir / "data")
    sneaky_rollouts = minting_result.rollouts
    if sneaky_rollouts:
        data_store.save_rollouts(round_id, "sneaky", sneaky_rollouts)

    honest_rollouts = clone_honest_rollouts_for_round(
        base_honest_rollouts,
        round_id=round_id,
    )
    if honest_rollouts:
        data_store.save_rollouts(round_id, "honest", honest_rollouts)

    pair_builder = PreferencePairBuilder(
        require_certified=True,
        require_tests_pass=True,
        min_pass_rate=1.0,
    )
    pairs = pair_builder.build_pairs(
        honest_rollouts,
        sneaky_rollouts,
        env_labels={"round": round_id},
    )
    if pairs:
        data_store.save_pairs(round_id, pairs)

    orchestrator = PVGOrchestrator(output_dir=config.output_dir, config=config)
    orchestrator.resume_from_checkpoint(validate_config=False)
    if orchestrator.current_state == PVGState.INIT:
        orchestrator.transition(PVGState.MINT_DATA)
    orchestrator.record_phase_checkpoint(
        "mint",
        config.output_dir / "data" / f"round_{round_id}",
        metrics={"pair_count": len(pairs)},
    )
    if orchestrator.can_transition(PVGState.TRAIN_VERIFIER):
        orchestrator.transition(PVGState.TRAIN_VERIFIER)

    manifest = ComponentManifest(
        component_name="mint",
        run_id=run_id,
        git_sha=git_sha,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=now_iso(),
        inputs={
            "bootstrap_manifest": str(bootstrap_manifest),
            "round_id": round_id,
            "host": host,
            "prover_port": prover_port,
        },
        outputs={
            "sneaky_rollouts": str(data_store._rollouts_path(round_id, "sneaky")),
            "honest_rollouts": str(data_store._rollouts_path(round_id, "honest")),
            "pairs": str(data_store._pairs_path(round_id)),
        },
        metrics={
            "total_generated": minting_result.total_generated,
            "valid_count": minting_result.valid_count,
            "certified_count": minting_result.certified_count,
            "avg_test_pass_rate": minting_result.avg_test_pass_rate,
        },
        round_id=round_id,
    )

    manifest_path = build_manifest_path(config.output_dir, "mint", run_id, round_id=round_id)
    write_manifest(manifest_path, manifest)
    logger.info("Mint manifest written to %s", manifest_path)

    return manifest_path

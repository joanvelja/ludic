"""PVG prompt testing and output extraction component."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ludic.training import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.inference import VLLMClient
from ludic.pvg.components.common import (
    SNEAKY_JSON_SYSTEM_PROMPT,
    build_game_config,
    build_inference_spec,
    build_rollout_engine,
    get_git_sha,
    prepare_apps_samples,
    read_dataset_options,
)
from ludic.pvg.manifest import (
    ComponentManifest,
    build_manifest_path,
    compute_config_hash,
    new_run_id,
    now_iso,
    read_manifest,
    write_manifest,
)
from ludic.envs.code_exec import CodeExecConfig, SneakyConfig, create_sandbox_pool

logger = logging.getLogger(__name__)


def run_prompt_test(
    *,
    config_path: Path,
    input_manifest: Path,
    round_id: int,
    host: str,
    prover_port: int,
    output_dir: Optional[Path] = None,
    num_samples: int = 8,
    concurrency: int = 8,
) -> Path:
    config = build_game_config(config_path)
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    started_at = now_iso()
    git_sha = get_git_sha()
    config_hash = compute_config_hash(config)

    _ = read_manifest(input_manifest)

    dataset, dataset_subset, _max_samples = read_dataset_options(config_path)
    if dataset != "apps":
        raise RuntimeError(f\"Dataset {dataset!r} not supported for prompt testing yet\")\n\n    problem_samples, adapter, honest_codes = prepare_apps_samples(\n        limit=num_samples,\n        subset=dataset_subset,\n    )

    vllm_client = VLLMClient(
        host=host,
        policy_port=prover_port,
        enable_weight_updates=False,
        model_name=config.prover_model_path,
    )

    sandbox_backend = os.getenv("LUDIC_SANDBOX_BACKEND", "auto")
    sandbox_pool = _run_async(
        create_sandbox_pool(
            n_workers=concurrency,
            backend=sandbox_backend,
            python_version="3.11",
            max_concurrent_ops=max(2, concurrency // 2),
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
        sneaky_config=SneakyConfig(sequential_observation=round_id > 0),
        code_exec_config=code_exec_config,
        system_prompt=SNEAKY_JSON_SYSTEM_PROMPT,
    )

    inference_spec = build_inference_spec(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
    )

    requests: List[RolloutRequest] = []
    for sample in problem_samples:
        prompt = adapter.get_prompt(sample)
        requests.append(
            RolloutRequest(
                env=EnvSpec(kind="pvg_sneaky", kwargs={"sample": sample}),
                protocol=ProtocolSpec(
                    kind="pvg_sneaky",
                    kwargs={"sample": sample, "prompt": prompt},
                ),
                num_episodes=1,
                inference=inference_spec,
                meta={
                    "problem_id": adapter.get_problem_id(sample),
                    "problem": prompt,
                },
            )
        )

    rollouts = _run_async(
        rollout_engine.generate_rollouts(
            requests=requests,
            max_steps=1,
            concurrency=concurrency,
        )
    )

    total = len(rollouts)
    parsed = sum(1 for r in rollouts if r.steps)
    avg_tokens = 0.0
    if rollouts:
        lengths = [len(str(r.steps[0].action)) if r.steps else 0 for r in rollouts]
        avg_tokens = sum(lengths) / max(1, len(lengths))

    report = {
        "round_id": round_id,
        "total_rollouts": total,
        "parse_success": parsed,
        "parse_success_rate": parsed / max(1, total),
        "avg_action_chars": avg_tokens,
        "samples": [
            {
                "rollout_id": r.id,
                "problem_id": r.meta.get("problem_id"),
                "action_preview": (r.steps[0].action[:200] if r.steps else ""),
            }
            for r in rollouts[: min(5, len(rollouts))]
        ],
    }

    report_path = config.output_dir / "prompt_reports" / f"round_{round_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    manifest = ComponentManifest(
        component_name="prompt_test",
        run_id=run_id,
        git_sha=git_sha,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=now_iso(),
        inputs={
            "input_manifest": str(input_manifest),
            "round_id": round_id,
            "host": host,
            "prover_port": prover_port,
        },
        outputs={
            "report_path": str(report_path),
        },
        metrics={
            "total_rollouts": total,
            "parse_success_rate": report["parse_success_rate"],
        },
        round_id=round_id,
    )

    manifest_path = build_manifest_path(
        config.output_dir, "prompt_test", run_id, round_id=round_id
    )
    write_manifest(manifest_path, manifest)
    logger.info("Prompt test manifest written to %s", manifest_path)

    return manifest_path


def _run_async(awaitable):
    import asyncio

    return asyncio.run(awaitable)

"""PVG verifier weight sync component."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ludic.distributed.adapters import create_rm_publisher
from ludic.inference import VLLMClient
from ludic.inference.reward_types import RewardModelTrainingMode
from ludic.pvg.manifest import (
    ComponentManifest,
    build_manifest_path,
    compute_config_hash,
    new_run_id,
    now_iso,
    read_manifest,
    write_manifest,
)
from ludic.pvg.components.common import build_game_config, get_git_sha, load_verifier_model

logger = logging.getLogger(__name__)


def run_sync_verifier(
    *,
    config_path: Path,
    train_manifest: Path,
    round_id: int,
    host: str,
    prover_port: int,
    verifier_port: int,
    policy_group_port: int = 51216,
    scoring_group_port: int = 51217,
    output_dir: Optional[Path] = None,
) -> Path:
    config = build_game_config(config_path)
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    started_at = now_iso()
    git_sha = get_git_sha()
    config_hash = compute_config_hash(config)

    train_manifest_data = read_manifest(train_manifest)
    checkpoint_path = train_manifest_data.outputs.get("checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError("No verifier checkpoint path found in train manifest")

    verifier_model, _verifier_tokenizer = _run_async(
        load_verifier_model(
            checkpoint_path,
            device="cuda",
        )
    )

    vllm_client = VLLMClient(
        host=host,
        policy_port=prover_port,
        scoring_port=verifier_port,
        group_port=policy_group_port,
        scoring_group_port=scoring_group_port,
        enable_weight_updates=True,
        model_name=config.prover_model_path,
    )

    rm_publisher = create_rm_publisher(
        vllm_client,
        training_mode=RewardModelTrainingMode.FULL,
    )

    rm_publisher.publish(verifier_model.state_dict(), version=round_id)

    manifest = ComponentManifest(
        component_name="sync_verifier",
        run_id=run_id,
        git_sha=git_sha,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=now_iso(),
        inputs={
            "train_manifest": str(train_manifest),
            "checkpoint_path": str(checkpoint_path),
            "round_id": round_id,
            "host": host,
            "prover_port": prover_port,
            "verifier_port": verifier_port,
        },
        outputs={
            "synced": True,
        },
        metrics={},
        round_id=round_id,
    )

    manifest_path = build_manifest_path(
        config.output_dir, "sync_verifier", run_id, round_id=round_id
    )
    write_manifest(manifest_path, manifest)
    logger.info("Sync verifier manifest written to %s", manifest_path)

    return manifest_path


def _run_async(awaitable):
    import asyncio

    return asyncio.run(awaitable)

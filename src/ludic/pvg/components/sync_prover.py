"""PVG prover weight sync component."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ludic.distributed.adapters import create_vllm_publisher
from ludic.inference import VLLMClient
from ludic.pvg.manifest import (
    ComponentManifest,
    build_manifest_path,
    compute_config_hash,
    new_run_id,
    now_iso,
    read_manifest,
    write_manifest,
)
from ludic.pvg.components.common import build_game_config, get_git_sha, load_prover_model

logger = logging.getLogger(__name__)


def run_sync_prover(
    *,
    config_path: Path,
    train_manifest: Path,
    round_id: int,
    host: str,
    prover_port: int,
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
        raise RuntimeError("No prover checkpoint path found in train manifest")

    prover_model, _prover_tokenizer = _run_async(
        load_prover_model(
            checkpoint_path,
            device="cuda",
        )
    )

    vllm_client = VLLMClient(
        host=host,
        policy_port=prover_port,
        enable_weight_updates=True,
        model_name=config.prover_model_path,
    )

    prover_publisher = create_vllm_publisher(vllm_client)
    prover_publisher.publish(prover_model.state_dict(), version=round_id)

    manifest = ComponentManifest(
        component_name="sync_prover",
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
        },
        outputs={
            "synced": True,
        },
        metrics={},
        round_id=round_id,
    )

    manifest_path = build_manifest_path(
        config.output_dir, "sync_prover", run_id, round_id=round_id
    )
    write_manifest(manifest_path, manifest)
    logger.info("Sync prover manifest written to %s", manifest_path)

    return manifest_path


def _run_async(awaitable):
    import asyncio

    return asyncio.run(awaitable)

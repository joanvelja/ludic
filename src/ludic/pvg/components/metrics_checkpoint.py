"""PVG metrics + checkpoint component."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ludic.pvg.config import PVGGameConfig
from ludic.pvg.data import RoundDataStore
from ludic.pvg.metrics import CollapseAlert, GoodhartingAlert, PVGMetricsLogger
from ludic.pvg.orchestrator import PVGOrchestrator, PVGState
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
    check_stopping_criteria,
    compute_round_metrics,
    get_git_sha,
)

logger = logging.getLogger(__name__)


def run_metrics_checkpoint(
    *,
    config_path: Path,
    train_prover_manifest: Path,
    round_id: int,
    output_dir: Optional[Path] = None,
    round_start_time_s: Optional[float] = None,
) -> Path:
    config = build_game_config(config_path)
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    started_at = now_iso()
    git_sha = get_git_sha()
    config_hash = compute_config_hash(config)

    _ = read_manifest(train_prover_manifest)

    orchestrator = PVGOrchestrator(output_dir=config.output_dir, config=config)
    orchestrator.resume_from_checkpoint(validate_config=False)

    data_store = RoundDataStore(config.output_dir / "data")
    metrics_logger = PVGMetricsLogger(config.output_dir / "metrics")
    metrics_logger.load_from_file()

    collapse_alert = CollapseAlert(config.output_dir / "metrics", collapse_threshold=0.1)
    goodharting_alert = GoodhartingAlert(config.output_dir / "metrics")

    sneaky_rollouts = list(data_store.load_rollouts(round_ids=[round_id], roles=["sneaky"]))
    honest_rollouts = list(data_store.load_rollouts(round_ids=[round_id], roles=["honest"]))

    round_metrics = compute_round_metrics(
        sneaky_rollouts=sneaky_rollouts,
        honest_rollouts=honest_rollouts,
        round_id=round_id,
    )

    if round_start_time_s is not None:
        round_metrics.round_duration_s = time.time() - round_start_time_s

    round_config = config.get_round_config(round_id)
    round_metrics.verifier_training_steps = round_config.verifier_steps
    round_metrics.prover_training_steps = round_config.prover_steps

    metrics_logger.log_round_metrics(round_metrics)
    metrics_logger.log_round_summary(round_id)
    collapse_alert.check(round_metrics)

    previous = metrics_logger.get_round_metrics(round_id - 1) if round_id > 0 else None
    if previous is not None:
        goodharting_alert.check(round_metrics, previous)

    stop = check_stopping_criteria(round_metrics, round_config)
    if stop:
        orchestrator.transition(PVGState.COMPLETE)
    else:
        if orchestrator.round_id < config.num_rounds - 1:
            orchestrator.advance_round()
            orchestrator.transition(PVGState.MINT_DATA)
        else:
            orchestrator.transition(PVGState.COMPLETE)

    checkpoint_dir = config.output_dir / "checkpoints" / f"round_{round_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "round_id": round_id,
                "metrics": round_metrics.to_dict(),
                "config": asdict(round_config),
            },
            f,
            indent=2,
            default=str,
        )

    manifest = ComponentManifest(
        component_name="metrics_checkpoint",
        run_id=run_id,
        git_sha=git_sha,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=now_iso(),
        inputs={
            "train_prover_manifest": str(train_prover_manifest),
            "round_id": round_id,
        },
        outputs={
            "metrics_path": str(metrics_logger.output_dir),
            "metadata_path": str(metadata_path),
            "orchestrator_state": str(orchestrator.state_path),
        },
        metrics={
            "sneaky_certified_rate": round_metrics.sneaky_certified_rate,
            "sneaky_incorrect_rate": round_metrics.sneaky_incorrect_rate,
            "stop": stop,
        },
        round_id=round_id,
    )

    manifest_path = build_manifest_path(
        config.output_dir, "metrics_checkpoint", run_id, round_id=round_id
    )
    write_manifest(manifest_path, manifest)
    logger.info("Metrics checkpoint manifest written to %s", manifest_path)

    return manifest_path

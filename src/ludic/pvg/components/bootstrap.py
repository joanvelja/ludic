"""PVG bootstrap component."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from ludic.pvg.config import PVGGameConfig
from ludic.pvg.data import RolloutRecord, RoundDataStore
from ludic.pvg.metrics import CollapseAlert, GoodhartingAlert, PVGMetricsLogger
from ludic.pvg.minting import mint_honest_from_dataset
from ludic.pvg.orchestrator import PVGOrchestrator
from ludic.pvg.manifest import (
    ComponentManifest,
    build_manifest_path,
    compute_config_hash,
    new_run_id,
    now_iso,
    write_manifest,
)
from ludic.pvg.components.common import (
    build_game_config,
    dump_config,
    get_git_sha,
    load_dataset_samples,
    load_honest_codes,
    prepare_apps_samples,
    read_dataset_options,
)

logger = logging.getLogger(__name__)


def _save_base_honest_rollouts(
    output_dir: Path,
    rollouts: list[RolloutRecord],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "data" / "base_honest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in rollouts:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")
    return path


def run_bootstrap(
    *,
    config_path: Path,
    output_dir: Optional[Path] = None,
    resume: bool = False,
) -> Path:
    config = build_game_config(config_path)
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    started_at = now_iso()
    git_sha = get_git_sha()
    config_hash = compute_config_hash(config)

    logger.info("BOOTSTRAP: initializing PVG orchestration")
    logger.info("Config: %s", dump_config(config))

    orchestrator = PVGOrchestrator(output_dir=config.output_dir, config=config)
    if resume:
        if not orchestrator.resume_from_checkpoint():
            orchestrator.initialize()
    else:
        orchestrator.initialize()

    data_store = RoundDataStore(config.output_dir / "data")
    metrics_logger = PVGMetricsLogger(config.output_dir / "metrics")
    collapse_alert = CollapseAlert(config.output_dir / "metrics", collapse_threshold=0.1)
    goodharting_alert = GoodhartingAlert(config.output_dir / "metrics")

    if resume:
        metrics_logger.load_from_file()

    dataset, dataset_subset, max_samples = read_dataset_options(config_path)
    raw_yaml = None
    if config_path.exists():
        raw_yaml = json.dumps(config_path.read_text())

    problem_samples = []
    honest_codes: Dict[str, str] = {}
    if dataset == "apps":
        samples, _adapter, honest_codes = prepare_apps_samples(
            limit=max_samples,
            subset=dataset_subset,
        )
        problem_samples = samples
    else:
        # Default to APPS semantics until other datasets are wired.
        problem_samples = load_dataset_samples(
            dataset,
            config.data_split,
            dataset_subset=dataset_subset,
            max_samples=max_samples,
        )
        honest_codes = load_honest_codes(problem_samples)

    honest_rollouts_base = []
    if problem_samples:
        import asyncio

        honest_rollouts_base = asyncio.run(
            mint_honest_from_dataset(
                problem_samples=problem_samples,
                round_id=0,
            )
        )

    base_honest_path = _save_base_honest_rollouts(config.output_dir, honest_rollouts_base)

    manifest = ComponentManifest(
        component_name="bootstrap",
        run_id=run_id,
        git_sha=git_sha,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=now_iso(),
        inputs={
            "config_path": str(config_path),
            "resume": resume,
            "raw_yaml": raw_yaml,
        },
        outputs={
            "state_path": str(orchestrator.state_path),
            "data_dir": str(data_store.base_dir),
            "metrics_dir": str(metrics_logger.output_dir),
            "base_honest_rollouts": str(base_honest_path),
        },
        metrics={
            "num_problem_samples": len(problem_samples),
            "num_honest_rollouts": len(honest_rollouts_base),
            "collapse_threshold": collapse_alert.collapse_threshold,
            "goodharting_enabled": True,
        },
        round_id=None,
    )

    manifest_path = build_manifest_path(config.output_dir, "bootstrap", run_id)
    write_manifest(manifest_path, manifest)
    logger.info("Bootstrap manifest written to %s", manifest_path)

    return manifest_path

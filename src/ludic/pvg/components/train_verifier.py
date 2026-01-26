"""PVG verifier training component."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ludic.pvg.config import PVGGameConfig
from ludic.pvg.data import RoundDataStore
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
    get_git_sha,
    load_verifier_model,
)
from ludic.pvg.verifier_trainer import (
    VerifierTrainingConfig,
    reinitialize_verifier_head,
    train_verifier_phase,
)

logger = logging.getLogger(__name__)


def run_train_verifier(
    *,
    config_path: Path,
    mint_manifest: Path,
    round_id: int,
    output_dir: Optional[Path] = None,
    irm_beta: float = 1.0,
    mixture_decay_lambda: float = 0.5,
    verifier_pair_samples: Optional[int] = None,
    verifier_shuffle_buffer: int = 1024,
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

    _ = read_manifest(mint_manifest)

    round_config = config.get_round_config(round_id)

    verifier_config = VerifierTrainingConfig(
        max_steps=round_config.verifier_steps,
        irm_mode=round_config.irm_mode,
        irm_beta=irm_beta,
        mixture_strategy=round_config.data_mixture_strategy,
        mixture_decay_lambda=mixture_decay_lambda,
        pair_sample_count=verifier_pair_samples,
        shuffle_buffer_size=verifier_shuffle_buffer,
        pair_sample_seed=round_id,
    )

    verifier_model, verifier_tokenizer = _run_async(
        load_verifier_model(
            config.verifier_model_path,
            device=device,
        )
    )

    should_reinit = round_id > 0 or config.verifier_init.reinit_mode == "head_only"
    if should_reinit:
        reinitialize_verifier_head(verifier_model, round_id=round_id)

    verifier_config.pad_token_id = verifier_tokenizer.pad_token_id or 0

    data_store = RoundDataStore(config.output_dir / "data")

    checkpoint_path = _run_async(
        train_verifier_phase(
            model=verifier_model,
            tokenizer=verifier_tokenizer,
            data_store=data_store,
            current_round=round_id,
            config=verifier_config,
            output_dir=config.output_dir,
        )
    )

    orchestrator = PVGOrchestrator(output_dir=config.output_dir, config=config)
    orchestrator.resume_from_checkpoint(validate_config=False)
    orchestrator.record_phase_checkpoint(
        "verifier",
        checkpoint_path if checkpoint_path is not None else config.output_dir,
    )
    if orchestrator.can_transition(PVGState.TRAIN_PROVER):
        orchestrator.transition(PVGState.TRAIN_PROVER)

    manifest = ComponentManifest(
        component_name="train_verifier",
        run_id=run_id,
        git_sha=git_sha,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=now_iso(),
        inputs={
            "mint_manifest": str(mint_manifest),
            "round_id": round_id,
            "verifier_model": config.verifier_model_path,
            "reinit_mode": config.verifier_init.reinit_mode,
            "device": device,
        },
        outputs={
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        },
        metrics={
            "verifier_steps": verifier_config.max_steps,
            "irm_mode": verifier_config.irm_mode,
            "irm_weight": verifier_config.irm_weight,
        },
        round_id=round_id,
    )

    manifest_path = build_manifest_path(
        config.output_dir, "train_verifier", run_id, round_id=round_id
    )
    write_manifest(manifest_path, manifest)
    logger.info("Train verifier manifest written to %s", manifest_path)

    return manifest_path


def _run_async(awaitable):
    import asyncio

    return asyncio.run(awaitable)

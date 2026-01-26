import os
from pathlib import Path

import pytest

from ludic.pvg.components import (
    run_bootstrap,
    run_mint,
    run_train_verifier,
    run_sync_verifier,
    run_train_prover,
    run_sync_prover,
    run_prompt_test,
    run_metrics_checkpoint,
)


pytestmark = [pytest.mark.pvg_component, pytest.mark.gpu]


def _require_env() -> None:
    if os.getenv("PVG_COMPONENT_TEST") != "1":
        pytest.skip("PVG component tests disabled; set PVG_COMPONENT_TEST=1")


def test_pvg_component_pipeline(tmp_path: Path) -> None:
    _require_env()

    config_path = Path("tests/pvg/configs/apps_pvg_test.yaml")
    host = os.getenv("PVG_TEST_HOST", "127.0.0.1")
    prover_port = int(os.getenv("PVG_TEST_PROVER_PORT", "8000"))
    verifier_port = int(os.getenv("PVG_TEST_VERIFIER_PORT", "8001"))

    bootstrap_manifest = run_bootstrap(
        config_path=config_path,
        output_dir=tmp_path,
        resume=False,
    )
    assert bootstrap_manifest.exists()

    mint_manifest = run_mint(
        config_path=config_path,
        bootstrap_manifest=bootstrap_manifest,
        round_id=0,
        host=host,
        prover_port=prover_port,
        output_dir=tmp_path,
        concurrency=4,
    )
    assert mint_manifest.exists()

    verifier_train_manifest = run_train_verifier(
        config_path=config_path,
        mint_manifest=mint_manifest,
        round_id=0,
        output_dir=tmp_path,
        irm_beta=1.0,
        mixture_decay_lambda=0.5,
        verifier_pair_samples=4,
        verifier_shuffle_buffer=32,
        device="cuda",
    )
    assert verifier_train_manifest.exists()

    verifier_sync_manifest = run_sync_verifier(
        config_path=config_path,
        train_manifest=verifier_train_manifest,
        round_id=0,
        host=host,
        prover_port=prover_port,
        verifier_port=verifier_port,
        output_dir=tmp_path,
    )
    assert verifier_sync_manifest.exists()

    prover_train_manifest = run_train_prover(
        config_path=config_path,
        sync_verifier_manifest=verifier_sync_manifest,
        round_id=0,
        host=host,
        prover_port=prover_port,
        verifier_port=verifier_port,
        output_dir=tmp_path,
        reward_strategy="composite",
        group_size=2,
        sync_every_steps=1,
        concurrency=4,
        device="cuda",
    )
    assert prover_train_manifest.exists()

    prover_sync_manifest = run_sync_prover(
        config_path=config_path,
        train_manifest=prover_train_manifest,
        round_id=0,
        host=host,
        prover_port=prover_port,
        output_dir=tmp_path,
    )
    assert prover_sync_manifest.exists()

    prompt_manifest = run_prompt_test(
        config_path=config_path,
        input_manifest=prover_sync_manifest,
        round_id=0,
        host=host,
        prover_port=prover_port,
        output_dir=tmp_path,
        num_samples=2,
        concurrency=2,
    )
    assert prompt_manifest.exists()

    metrics_manifest = run_metrics_checkpoint(
        config_path=config_path,
        train_prover_manifest=prover_train_manifest,
        round_id=0,
        output_dir=tmp_path,
    )
    assert metrics_manifest.exists()

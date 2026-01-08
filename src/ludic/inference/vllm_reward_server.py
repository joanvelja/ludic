"""vLLM reward model server with NCCL weight synchronization.

This server hosts a reward model using vLLM's AsyncLLMEngine with `task="reward"`
and provides:
- `/score` endpoint for batch scoring using vLLM's encode() API
- NCCL-based weight synchronization via RewardWeightSyncExtension
- Health check with model type identification

Supports all training modes:
    - HEAD_ONLY: Only head weights updated
    - LORA: LoRA + head weights (expects merged weights)
    - FULL: Complete model weights

Usage:
    python -m ludic.inference.vllm_reward_server \
        --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
        --host 0.0.0.0 \
        --port 8001 \
        --group-port 51217

Default ports follow convention of policy server + 1:
- HTTP: 8001 (policy is 8000)
- NCCL: 51217 (policy is 51216)
"""

from __future__ import annotations

import os
import signal
import sys
from typing import Any

# Use V1 engine explicitly.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"

import uvloop
from fastapi import FastAPI
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit

# Shared infrastructure
from ludic.inference.vllm.server_base import (
    ServerState,
    WeightSyncExtensionBase,
    register_weight_sync_endpoints,
)


# ---------------------------------------------------------------------------
# Type definitions for API
# ---------------------------------------------------------------------------


class ScoringRequestPayload(BaseModel):
    """Request payload for /score endpoint."""

    model: str
    inputs: list[str]
    pooling_type: str = "last"  # "last", "mean", "cls"
    normalize: bool = True
    n_labels: int = 1


class ScoringResponsePayload(BaseModel):
    """Response payload for /score endpoint."""

    scores: list[float]
    model: str
    usage: dict[str, int] = {}
    metadata: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Worker extension: NCCL-based weight sync for reward models
# ---------------------------------------------------------------------------


class RewardWeightSyncExtension(WeightSyncExtensionBase):
    """vLLM worker extension for reward model weight synchronization.

    Inherits all NCCL functionality from WeightSyncExtensionBase.

    Handles all training modes:
    - HEAD_ONLY: Only updates classification head (score.weight, classifier.weight)
    - LORA: Updates LoRA matrices + head (expects pre-merged)
    - FULL: Updates complete model

    vLLM's load_weights() handles QKV fusion, TP sharding, etc. automatically.
    """

    log_prefix = "[RewardWorker]"


# ---------------------------------------------------------------------------
# Server / app setup
# ---------------------------------------------------------------------------


async def run_server(args) -> None:
    """Run the vLLM reward model server."""
    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    # Create server state
    state = ServerState()

    # ----------------------------------------------------------------------
    # 1) Build engine_args from CLI and configure for reward model
    # ----------------------------------------------------------------------
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Configure for reward model - vLLM 0.12+ pooling mode
    engine_args.task = "reward"

    # Wire worker extension for weight synchronization
    worker_ext = "ludic.inference.vllm_reward_server.RewardWeightSyncExtension"
    engine_args.worker_extension_cls = worker_ext

    # ----------------------------------------------------------------------
    # 2) Build VllmConfig from engine_args
    # ----------------------------------------------------------------------
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    # ----------------------------------------------------------------------
    # 3) Build AsyncLLMEngine with task="reward" for pooling mode
    # ----------------------------------------------------------------------
    engine = AsyncLLMEngine.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_log_requests=engine_args.enable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )

    app: FastAPI = build_app(args)

    # ----------------------------------------------------------------------
    # 4) Add scoring endpoint using vLLM's encode() API
    # ----------------------------------------------------------------------

    @app.post("/score")
    async def score(request: ScoringRequestPayload) -> ScoringResponsePayload:
        """Score a batch of inputs using vLLM's encode() API.

        vLLM's encode() with task="reward" leverages continuous batching,
        PagedAttention, and tensor parallelism for efficient scoring.
        """
        from vllm import PoolingParams

        pooling_params = PoolingParams(normalize=request.normalize)

        # Use vLLM's encode API for reward scoring
        outputs = await engine.encode(request.inputs, pooling_params)

        scores = []
        total_tokens = 0
        for output in outputs:
            # output.outputs.data contains the pooled score
            score_data = output.outputs.data
            if hasattr(score_data, "item"):
                score_val = float(score_data.item())
            elif hasattr(score_data, "__float__"):
                score_val = float(score_data)
            else:
                # May be a tensor or list
                score_val = float(score_data[0]) if len(score_data) > 0 else 0.0

            scores.append(score_val)
            total_tokens += len(output.prompt_token_ids)

        return ScoringResponsePayload(
            scores=scores,
            model=request.model,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
            metadata={"version": state.runtime_version},
        )

    # ----------------------------------------------------------------------
    # 5) Register weight sync endpoints using shared infrastructure
    # ----------------------------------------------------------------------
    register_weight_sync_endpoints(
        app,
        engine,
        state,
        get_world_size=lambda: args.tensor_parallel_size * args.data_parallel_size,
        server_type="reward_model",
        log_prefix="[RewardServer]",
    )

    # ─────────────────────────────────────────────────────────────
    # Start HTTP server
    # ─────────────────────────────────────────────────────────────

    print(f"[RewardServer] Starting vLLM reward server")
    print(f"[RewardServer] Model: {args.model}")
    print(f"[RewardServer] Task: reward (pooling mode)")
    print(f"[RewardServer] Tensor Parallel: {args.tensor_parallel_size}")
    print(f"[RewardServer] Host: {args.host}:{args.port}")
    print(engine.vllm_config)

    await init_app_state(engine, app.state, args)

    shutdown_task = await serve_http(
        app,
        sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
    await shutdown_task

    # Graceful shutdown
    print("[RewardServer] Shutting down...")
    await state.shutdown()
    sock.close()


def main() -> None:
    """Main entry point."""
    parser = FlexibleArgumentParser(
        description="vLLM Reward Model Server with NCCL weight synchronization"
    )
    parser = make_arg_parser(parser)

    # Add custom arguments specific to reward server
    parser.add_argument(
        "--group-port",
        type=int,
        default=51217,
        help="Port for NCCL weight sync group (default: 51217)",
    )

    argv = sys.argv[1:]
    # vLLM can silently override sampling params using the model's Hugging Face
    # `generation_config` unless `--generation-config vllm` is set.
    if not any(
        a == "--generation-config" or a.startswith("--generation-config=") for a in argv
    ):
        argv = [*argv, "--generation-config", "vllm"]

    args = parser.parse_args(argv)
    assert args is not None

    validate_parsed_serve_args(args)

    print(f"[RewardServer] Arguments: {args}")

    # Run server
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()

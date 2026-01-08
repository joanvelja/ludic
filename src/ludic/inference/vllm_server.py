"""vLLM policy server with NCCL weight synchronization.

This server hosts a policy model for chat completions and provides:
- OpenAI-compatible /v1/chat/completions endpoint
- NCCL-based weight synchronization endpoints
- Custom logits processor for "</think>" injection (max_think)

Usage:
    python -m ludic.inference.vllm_server \
        --model Qwen/Qwen2.5-3B-Instruct \
        --host 0.0.0.0 \
        --port 8000
"""

import asyncio
import os
import signal
import sys
from argparse import Namespace
from typing import Any, Optional

# Use V1 engine explicitly.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"

import torch
import uvloop
from fastapi import FastAPI
from vllm import SamplingParams
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
from vllm.tokenizers import cached_tokenizer_from_config

# V1 logits-processor interface
from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor as V1LogitsProcessor,
    BatchUpdate,
    MoveDirectionality,
)

# Shared infrastructure
from ludic.inference.vllm.server_base import (
    ServerState,
    WeightSyncExtensionBase,
    register_weight_sync_endpoints,
)


# ---------------------------------------------------------------------------
# Worker extension: NCCL-based weight sync (inherits from base)
# ---------------------------------------------------------------------------


class WeightSyncWorkerExtension(WeightSyncExtensionBase):
    """vLLM worker extension for policy model weight synchronization.

    Inherits all NCCL functionality from WeightSyncExtensionBase.
    """

    log_prefix = "[PolicyWorker]"


# ---------------------------------------------------------------------------
# Custom logits processor: inject "</think>" after N tokens (pure V1)
# ---------------------------------------------------------------------------


class GlobalThinkProcessor(V1LogitsProcessor):
    """Single V1 logits processor instance per worker.

    For each request in the batch:
      - On BatchUpdate.added, we inspect SamplingParams.extra_args["max_think"].
      - If present and > 0, we remember:
          * a live reference to that request's output_ids (list[int])
          * its trigger_len
      - On apply(logits), we walk each row and, if that request is within
        its think-window, we force the next token of the '</think>' sequence.

    In other words: we force the model to emit '</think>' after a
    user-chosen number of generated tokens.
    """

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        # Per-request state: req_idx -> {"output_ids": list[int], "trigger_len": int}
        self.req_state: dict[int, dict[str, Any]] = {}
        # Pre-tokenized think_ids injected into vllm_config BEFORE engine spawn
        self.think_ids = vllm_config.additional_config.get("think_ids", [])

    # ---- required by V1 interface ----

    def is_argmax_invariant(self) -> bool:
        # We overwrite logits and hence argmax, so no.
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """Called whenever the persistent batch changes (add/remove/move)."""
        if batch_update is None:
            return

        # Handle removals
        for ridx in batch_update.removed:
            if ridx in self.req_state:
                self.req_state.pop(ridx, None)

        # Handle additions
        for req_idx, params, prompt_ids, output_ids in batch_update.added:
            assert isinstance(params, SamplingParams)
            extra_args = getattr(params, "extra_args", None)

            trigger_len = None
            if isinstance(extra_args, dict):
                trigger_len = extra_args.get("max_think")

            if not isinstance(trigger_len, int) or trigger_len <= 0:
                self.req_state.pop(req_idx, None)
                continue

            self.req_state[req_idx] = {
                "output_ids": output_ids,
                "trigger_len": trigger_len,
            }

        # Handle moves
        for src, dst, direction in batch_update.moved:
            if direction == MoveDirectionality.UNIDIRECTIONAL:
                state = self.req_state.pop(src, None)
                if state is not None:
                    self.req_state[dst] = state
            else:
                s1 = self.req_state.get(src)
                s2 = self.req_state.get(dst)
                if s1 is not None or s2 is not None:
                    self.req_state[src], self.req_state[dst] = s2, s1

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Mutate logits to force '</think>' sequence."""
        if not self.think_ids or not self.req_state:
            return logits

        batch_size = logits.shape[0]
        think_ids = self.think_ids

        for req_idx in range(batch_size):
            state = self.req_state.get(req_idx)
            if state is None:
                continue

            output_ids: list[int] = state["output_ids"]
            trigger_len: int = state["trigger_len"]

            seq_len = len(output_ids)
            pos = seq_len - trigger_len

            if pos < 0 or pos >= len(think_ids):
                continue

            forced_id = think_ids[pos]

            row = logits[req_idx]
            row.fill_(float("-inf"))
            row[forced_id] = 0.0

        return logits


# ---------------------------------------------------------------------------
# Server / app setup
# ---------------------------------------------------------------------------


async def run_server(args: Namespace) -> None:
    """Run the vLLM policy server."""
    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    # Create server state
    state = ServerState()

    # ----------------------------------------------------------------------
    # 1) Build engine_args from CLI and inject our extension + logits proc
    # ----------------------------------------------------------------------
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Wire worker extension
    worker_ext = "ludic.inference.vllm_server.WeightSyncWorkerExtension"
    engine_args.worker_extension_cls = worker_ext

    # Wire our GlobalThinkProcessor into the engine-wide logits processor list
    think_proc = "ludic.inference.vllm_server:GlobalThinkProcessor"
    if engine_args.logits_processors:
        if think_proc not in engine_args.logits_processors:
            engine_args.logits_processors.append(think_proc)
    else:
        engine_args.logits_processors = [think_proc]

    # ----------------------------------------------------------------------
    # 2) Build VllmConfig from engine_args
    # ----------------------------------------------------------------------
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    # --------------------------------------------------------------
    # 3) Pre-tokenize '</think>' for the logits processor
    # --------------------------------------------------------------
    try:
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        think_ids = tokenizer.encode("</think>", add_special_tokens=False)
        vllm_config.additional_config["think_ids"] = think_ids
    except Exception as e:
        raise RuntimeError(
            f"Failed to pre-tokenize '</think>' for model "
            f"{vllm_config.model_config.model}: {e}"
        ) from e

    # ----------------------------------------------------------------------
    # 4) Build AsyncLLM engine
    # ----------------------------------------------------------------------
    engine = AsyncLLMEngine.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_log_requests=engine_args.enable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )

    app: FastAPI = build_app(args)

    # ----------------------------------------------------------------------
    # 5) Register weight sync endpoints using shared infrastructure
    # ----------------------------------------------------------------------
    register_weight_sync_endpoints(
        app,
        engine,
        state,
        get_world_size=lambda: args.tensor_parallel_size * args.data_parallel_size,
        server_type="policy",
        log_prefix="[PolicyServer]",
    )

    # ------------------------ start HTTP server --------------------------

    print(f"[PolicyServer] Starting vLLM policy server")
    print(f"[PolicyServer] Model: {args.model}")
    print(f"[PolicyServer] Tensor Parallel: {args.tensor_parallel_size}")
    print(f"[PolicyServer] Host: {args.host}:{args.port}")
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
    print("[PolicyServer] Shutting down...")
    await state.shutdown()
    sock.close()


def main() -> None:
    """Main entry point."""
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server with weight synchronization"
    )
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--batch-invariant",
        action="store_true",
        help="Enable vLLM batch-invariant kernels (sets VLLM_BATCH_INVARIANT=1).",
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
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    validate_parsed_serve_args(args)
    print(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()

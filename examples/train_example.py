import sys
import time
import logging
import json
import requests
import torch
from transformers import AutoModelForCausalLM

# PEFT Imports
from peft import get_peft_model, LoraConfig, TaskType

# Rich Imports for Dashboard
from rich.console import Console
from rich.live import Live

# Ludic Imports
from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.parsers import xml_move_parser
from ludic.training.rollout_engine import RolloutEngine, RolloutBatchSource
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.algorithm import make_reinforce
from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.distributed.adapters import create_vllm_publisher
from ludic.utils.dashboard import create_dashboard
from ludic.utils.setup_vllm import wait_for_server
from examples.envs.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use the 7B Instruct model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216

# Training Hyperparameters
LEARNING_RATE = 1e-4
NUM_TRAIN_STEPS = 50
BATCH_SIZE = 4
MAX_STEPS_PER_EPISODE = 5

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("lora_7b_trainer")


def main():
    console = Console()
    if not torch.cuda.is_available():
        sys.exit(1)

    print(f"🛡️ LoRA Trainer running on: {torch.cuda.get_device_name(0)}")

    # 1. Wait for vLLM (Ensure you ran it WITHOUT --enable-lora)
    wait_for_server(f"http://{VLLM_HOST}:{VLLM_PORT}", console)

    # 2. Setup Client
    client = VLLMChatClient(
        host=VLLM_HOST,
        port=VLLM_PORT,
        group_port=VLLM_GROUP_PORT,
        enable_weight_updates=True,
        device="cuda:0",
    )
    publisher = create_vllm_publisher(client)

    # 3. Load Base Model (7B)
    print(f"🧠 Loading base model: {MODEL_NAME}...")

    # ---------------------------------------------------------------------------
    # FSDP2 COMPATIBILITY NOTE:
    #
    # For SINGLE-GPU training (default), use device_map="auto" for easy loading.
    #
    # For MULTI-GPU FSDP2 training:
    #   - device_map="auto" is INCOMPATIBLE with FSDP2!
    #
    #   Option A: Load on meta device (recommended by PyTorch, for training from scratch):
    #       with torch.device("meta"):
    #           model = Model(...)
    #       # Apply fully_shard...
    #       model.to_empty(device="cuda")
    #       model.reset_parameters()  # <-- loses pre-trained weights!
    #
    #   Option B: Load pre-trained weights on CPU, then FSDP shards (for fine-tuning):
    #       model = AutoModelForCausalLM.from_pretrained(..., device_map="cpu")
    #       # Trainer's internal FSDP2 wrapping handles GPU placement
    #
    #   Enable FSDP2 via: TrainerConfig(fsdp_enabled=True, ...)
    # ---------------------------------------------------------------------------

    # NOTE: If you run out of VRAM here, install `bitsandbytes` and add `load_in_4bit=True`
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",  # Single-GPU: use auto. Multi-GPU FSDP2: use "cpu" instead!
    )

    # 4. Apply LoRA Configuration
    print("💉 Injecting LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Rank: 16 is a good balance for 7B
        lora_alpha=32,  # Alpha usually 2x Rank
        lora_dropout=0.05,
        bias="none",
        # Target all linear projection layers for best performance
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, peft_config)

    # Print stats to confirm we are training only ~0.05% of params
    model.print_trainable_parameters()

    # 5. Setup Engine
    env_registry = {"tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)}

    # --- PROTOCOL FACTORY ---
    # Custom factory to accept 'system_prompt' from ProtocolSpec.kwargs
    # and pass it to SingleAgentSyncProtocol's new 'prompt' argument.
    def create_protocol(system_prompt: str = None):
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=MODEL_NAME,
                ctx=FullDialog(),
                parser=xml_move_parser,
            ),
            prompt=system_prompt,  # <--- INJECTED HERE
        )

    protocol_registry = {"single_agent": create_protocol}

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path="lora_7b_training.jsonl",
    )

    # 6. Setup Batch Source
    def make_requests():
        # Get the standard prompt from the env class just to append to it
        base_prompt = TicTacToeEnv().suggested_sysprompt or ""

        # Create our Training Prompt (XML Instructions)
        training_prompt = (
            base_prompt
            + "\n\nOutput your move as a single XML tag, e.g., <move>A1</move>."
        )

        sampling_args = {
            "temperature": 1.0,
            "max_tokens": 100,
            "extras": {"extra_body": {"return_token_ids": True}},
        }
        return [
            RolloutRequest(
                env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
                protocol=ProtocolSpec(
                    kind="single_agent", kwargs={"system_prompt": training_prompt}
                ),
                sampling_args=sampling_args,
                num_episodes=BATCH_SIZE,
            )
        ]

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=make_reinforce(gamma=0.99).credit_assigner,
        requests_fn=make_requests,
        max_steps=MAX_STEPS_PER_EPISODE,
        concurrency=BATCH_SIZE,
        retokenize=False,
    )

    # 7. Trainer
    # Uses the updated 'Trainer' which detects PEFT models and handles merging.
    #
    # For MULTI-GPU FSDP2 training, add to TrainerConfig:
    #     fsdp_enabled=True,
    #     fsdp_param_dtype="bf16",      # Mixed precision for forward/backward
    #     fsdp_reduce_dtype="fp32",     # Full precision gradient reduction
    #     # fsdp_shard_fn=custom_shard, # Optional: custom sharding function
    #
    # NOTE: FSDP2 requires:
    #   1. LoRA adapters applied BEFORE passing to Trainer (already done above)
    #   2. Model loaded without device_map="auto" (use "cpu" instead)
    #   3. Run with torchrun: torchrun --nproc_per_node=N train_example.py
    #
    trainer = Trainer(
        model=model,
        algo=make_reinforce(gamma=0.99),
        batch_source=batch_source,
        publisher=publisher,
        enable_gradient_checkpointing=True,
        cfg=TrainerConfig(
            model_device="cuda:0",
            lr=LEARNING_RATE,
            grad_accum_steps=4,  # Increase accum steps for 7B to stabilize gradients
            sync_every_steps=1,
            # fsdp_enabled=True,  # Uncomment for multi-GPU FSDP2 training
        ),
    )

    # 8. Run with Live Dashboard
    print("🏋️ Starting LoRA Training on 7B Model...")

    with Live(console=console, refresh_per_second=4) as live:
        for i in range(1, NUM_TRAIN_STEPS + 1):
            # 1. Run step
            stats = trainer.train_step_sync()

            # 2. Update Dashboard
            dashboard = create_dashboard(stats, i, NUM_TRAIN_STEPS)
            live.update(dashboard)

            # 3. (Optional) Print raw json line to file for history
            with open("training_log.jsonl", "a") as f:
                f.write(json.dumps(stats) + "\n")

    console.print("[bold green]🎉 Training Done![/]")


if __name__ == "__main__":
    main()

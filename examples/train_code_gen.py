# examples/train_code_gen.py

import sys
from rich.console import Console
import asyncio
from pathlib import Path
import torch
import logging

from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig

from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.parsers import ParseResult
from ludic.training.batching import RolloutEngine, RolloutBatchSource
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.algorithm import make_reinforce
from ludic.interaction.single_agent import SingleAgentSyncProtocol

from ludic.sandbox.pool import SandboxPool
from ludic.sandbox.problems import ProblemBank
from ludic.envs.code_gen_env import CodeGenEnv, CodeGenRewardConfig

from ludic.utils.dashboard import create_dashboard
from ludic.utils.setup_vllm import wait_for_server
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_GROUP_PORT = 51216
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Training Hyperparameters
LEARNING_RATE = 1e-5
NUM_TRAIN_STEPS = 100
BATCH_SIZE = 4
MAX_STEPS_PER_EPISODE = 5

SANDBOX_BASE_PORT = 9000
NUM_SANDBOXES = 8  # Match your concurrency

PROBLEMS_PATH = "data/humaneval.jsonl"

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("lora_7b_trainer")

# ---------------------------------------------------------------------------
# Code parser (TODO: implement)
# ---------------------------------------------------------------------------


def code_passthrough_parser(raw: str) -> ParseResult:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    console = Console()

    if not torch.cuda.is_available():
        console.print("❌ CUDA not available. Please install CUDA.", color="red")
        sys.exit(1)

    wait_for_server(f"http://{VLLM_HOST}:{VLLM_PORT}", console)

    # 1. Initialize sandbox pool (containers must be running!)
    console.print(f"Connecting to {NUM_SANDBOXES} sandbox containers...", color="green")
    sandbox_pool = SandboxPool(
        host="localhost",
        base_port=SANDBOX_BASE_PORT,
        num_sandboxes=NUM_SANDBOXES,
        timeout_execute_s=60.0,
    )
    await sandbox_pool.start(health_check=True)
    console.print(
        f"✅ Sandbox pool ready ({sandbox_pool.available_count} available)",
        color="green",
    )

    # 2. Load problem bank
    console.print(f"Loading problems from {PROBLEMS_PATH}...", color="green")
    problem_bank = ProblemBank.from_jsonl(PROBLEMS_PATH)
    console.print(f"✅ Loaded {len(problem_bank)} problems", color="green")

    # 3. Setup vLLM client
    client = VLLMChatClient(
        host=VLLM_HOST,
        port=VLLM_PORT,
        enable_weight_updates=True,
    )

    # 4. Load Base Model (7B)
    print(f"🧠 Loading base model: {MODEL_NAME}...")
    # NOTE: If you run out of VRAM here, install `bitsandbytes` and add `load_in_4bit=True`
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",  # or "cuda:0" if single GPU
    )

    # 5. Apply LoRA Configuration
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

    # 6. Create registries
    # The env factory captures the pool and problem bank
    def create_code_env(**kwargs) -> CodeGenEnv:
        return CodeGenEnv(
            sandbox_pool=sandbox_pool,
            problem_bank=problem_bank,
            reward_config=CodeGenRewardConfig(
                all_pass=1.0,
                partial_credit_scale=0.5,
                compile_error=-0.3,
            ),
            language="python",
            max_attempts=1,  # Single turn
            **kwargs,
        )

    env_registry = {"code_gen": create_code_env}

    # Protocol factory
    def create_protocol():
        agent = Agent(
            client=client,
            model=MODEL_NAME,
            ctx=FullDialog(),
            parser=code_passthrough_parser,
        )
        return SingleAgentSyncProtocol(agent=agent)

    protocol_registry = {"code_protocol": create_protocol}

    # 7. Setup engine
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path="outputs/code_gen_rollouts.jsonl",
    )

    # 8. Setup batch source
    def make_requests():
        return [
            RolloutRequest(
                env=EnvSpec(kind="code_gen", kwargs={}),
                protocol=ProtocolSpec(kind="code_protocol", kwargs={}),
                num_episodes=NUM_SANDBOXES,  # One per sandbox
                sampling_args={"temperature": 0.7, "max_tokens": 2048},
            )
        ]

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=make_reinforce(gamma=0.99).credit_assigner,
        requests_fn=make_requests,
        max_steps=1,  # Single turn
        concurrency=NUM_SANDBOXES,  # Match sandbox count!
        retokenize=True,  # TODO: check, may not be needed
        tokenize=lambda s: client._async_client.tokenize(s),  # Or use HF tokenizer
    )

    # 9. Setup trainer
    trainer = Trainer(
        model=model,
        algo=make_reinforce(gamma=0.99),
        batch_source=batch_source,
        enable_gradient_checkpointing=True,
        cfg=TrainerConfig(),
    )

    # 10. Training loop
    print("Starting training...")
    for step in range(100):
        batch = await batch_source.next_batch()

        # Log stats
        pass_rates = [item.meta.get("pass_rate", 0) for item in batch.items]
        avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0

        print(
            f"Step {step}: "
            f"avg_reward={batch.meta['avg_total_reward']:.3f}, "
            f"avg_pass_rate={avg_pass_rate:.1%}, "
            f"items={len(batch.items)}"
        )

    # 11. Cleanup
    await sandbox_pool.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

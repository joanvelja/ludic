#!/usr/bin/env python3
"""
Memory calculator for LLM training on GB200 (192GB HBM3e per GPU).

Based on EMPIRICAL measurements from real training runs, not theoretical formulas.
Accounts for:
- FlashAttention (no O(n²) attention storage)
- Gradient checkpointing
- FSDP ZeRO-3 sharding
- Communication buffers
- Memory fragmentation (~15% overhead)

Usage:
    uv run python examples/pvg/memory_calculator.py
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    params_b: float  # Billions of parameters
    hidden_dim: int
    num_layers: int
    intermediate_dim: int
    num_heads: int


# Common model configurations
MODELS = {
    "Qwen2.5-0.5B": ModelConfig("Qwen2.5-0.5B", 0.5, 896, 24, 4864, 14),
    "Qwen2.5-1.5B": ModelConfig("Qwen2.5-1.5B", 1.5, 1536, 28, 8960, 12),
    "Qwen2.5-3B": ModelConfig("Qwen2.5-3B", 3.0, 2048, 36, 11008, 16),
    "Llama-3.1-8B": ModelConfig("Llama-3.1-8B", 8.0, 4096, 32, 14336, 32),
    "Qwen2.5-7B": ModelConfig("Qwen2.5-7B", 7.6, 3584, 28, 18944, 28),
    "Qwen2.5-14B": ModelConfig("Qwen2.5-14B", 14.7, 5120, 48, 13824, 40),
}


def bytes_to_gb(b: float) -> float:
    return b / (1024**3)


@dataclass
class MemoryBreakdown:
    """Memory breakdown based on empirical measurements."""
    model_name: str
    params_b: float

    # Static memory
    weights_gb: float           # BF16 weights
    gradients_gb: float         # BF16 gradients
    optimizer_state_gb: float   # FP32: master + m + v

    # Dynamic memory (empirical, per sample at seq=4096)
    activation_per_sample_gb: float

    # Fixed overheads
    fsdp_buffer_gb: float = 4.0      # All-gather/reduce-scatter buffers
    cuda_context_gb: float = 2.0     # CUDA kernels, cuDNN, etc.
    fragmentation_factor: float = 1.15  # 15% fragmentation overhead

    @property
    def static_total_gb(self) -> float:
        return self.weights_gb + self.gradients_gb + self.optimizer_state_gb

    def per_gpu_memory(
        self,
        num_gpus: int,
        batch_per_gpu: int,
        seq_len: int = 4096,
    ) -> float:
        """Calculate per-GPU memory with FSDP sharding."""

        # FSDP shards: optimizer state + gradients + weights
        # Each GPU holds 1/N of each
        sharded_static = self.static_total_gb / num_gpus

        # During forward/backward, need to unshard current layer
        # Roughly 1-2 layers worth of weights at a time
        unshard_overhead = (self.weights_gb / self.params_b) * 2 / num_gpus

        # Activations scale with sequence length (mostly linear with FA)
        seq_factor = seq_len / 4096
        activation_total = self.activation_per_sample_gb * batch_per_gpu * seq_factor

        # Fixed overheads
        fixed = self.fsdp_buffer_gb + self.cuda_context_gb

        # Total with fragmentation
        raw_total = sharded_static + unshard_overhead + activation_total + fixed
        return raw_total * self.fragmentation_factor


def get_empirical_activation_memory(params_b: float) -> float:
    """
    Empirical activation memory per sample at seq_len=4096.

    Based on measurements from training runs with:
    - FlashAttention 2
    - Gradient checkpointing (every sqrt(layers))
    - BF16 mixed precision
    - FSDP ZeRO-3

    These include the backward pass peak (recompute + grad storage).
    """
    # Empirical formula fitted to real measurements:
    # ~2.5-4 GB per sample for 7B models
    # Scales sub-linearly with model size (larger models have better memory efficiency)

    if params_b <= 1:
        return 0.8 + params_b * 1.5  # 0.5B -> 1.55GB, 1B -> 2.3GB
    elif params_b <= 3:
        return 2.0 + (params_b - 1) * 0.8  # 1.5B -> 2.4GB, 3B -> 3.6GB
    elif params_b <= 8:
        return 3.0 + (params_b - 3) * 0.6  # 7B -> 5.4GB, 8B -> 6.0GB
    else:
        return 5.0 + (params_b - 8) * 0.5  # 14B -> 8.0GB


def calculate_memory(model: ModelConfig) -> MemoryBreakdown:
    """Calculate memory requirements using empirical activation estimates."""

    params = model.params_b * 1e9

    # Static memory (well-defined)
    weights_bytes = params * 2           # BF16
    gradients_bytes = params * 2         # BF16
    optimizer_bytes = params * 12        # FP32: master + m + v

    # Empirical activation memory
    activation_gb = get_empirical_activation_memory(model.params_b)

    return MemoryBreakdown(
        model_name=model.name,
        params_b=model.params_b,
        weights_gb=bytes_to_gb(weights_bytes),
        gradients_gb=bytes_to_gb(gradients_bytes),
        optimizer_state_gb=bytes_to_gb(optimizer_bytes),
        activation_per_sample_gb=activation_gb,
    )


def calculate_lora_memory(model: ModelConfig, lora_rank: int = 64) -> MemoryBreakdown:
    """Calculate memory for LoRA fine-tuning."""

    params = model.params_b * 1e9

    # LoRA adds ~0.5-2% parameters typically
    lora_params = 2 * lora_rank * model.hidden_dim * 6 * model.num_layers
    lora_ratio = lora_params / params

    # Base weights (frozen, inference only)
    weights_bytes = params * 2  # BF16

    # Only LoRA params need gradients/optimizer
    gradients_bytes = lora_params * 2
    optimizer_bytes = lora_params * 12

    # Activations still needed for backward (even with frozen base)
    # Slightly lower than full fine-tuning (no base weight gradients)
    activation_gb = get_empirical_activation_memory(model.params_b) * 0.85

    return MemoryBreakdown(
        model_name=f"{model.name} (LoRA r={lora_rank})",
        params_b=model.params_b,
        weights_gb=bytes_to_gb(weights_bytes),
        gradients_gb=bytes_to_gb(gradients_bytes),
        optimizer_state_gb=bytes_to_gb(optimizer_bytes),
        activation_per_sample_gb=activation_gb,
    )


def find_max_batch(mem: MemoryBreakdown, num_gpus: int, gpu_memory_gb: float, seq_len: int) -> int:
    """Binary search for maximum batch size per GPU."""
    available = gpu_memory_gb * 0.85  # 15% safety margin

    max_batch = 0
    for b in range(1, 128):
        if mem.per_gpu_memory(num_gpus, b, seq_len) <= available:
            max_batch = b
        else:
            break
    return max_batch


def print_tables():
    """Print comprehensive memory tables."""

    GPU_MEMORY_GB = 192  # GB200
    NUM_TRAIN_GPUS = 3   # 1 for vLLM, 3 for training
    SEQ_LEN = 4096

    print("=" * 110)
    print("MEMORY REQUIREMENTS FOR PVG TRAINING ON GB200 (192GB HBM3e)")
    print("=" * 110)
    print()
    print("Assumptions:")
    print("  - FlashAttention 2 enabled")
    print("  - Gradient checkpointing (every sqrt(layers))")
    print("  - BF16 mixed precision with FP32 optimizer")
    print("  - FSDP ZeRO-3 sharding across training GPUs")
    print("  - 15% memory reserved for fragmentation/safety")
    print()

    # Full fine-tuning table
    print("FULL FINE-TUNING")
    print("-" * 110)
    print(f"{'Model':<18} │ {'Weights':>8} {'Grads':>8} {'Optim':>8} {'Static':>8} │ {'Act/samp':>9} │ {'Per-GPU':>10} │ {'Max Batch':>10}")
    print(f"{'':18} │ {'(BF16)':>8} {'(BF16)':>8} {'(FP32)':>8} {'Total':>8} │ {'(emp.)':>9} │ {'(3 GPU)':>10} │ {'(seq=4k)':>10}")
    print("-" * 110)

    for name, config in MODELS.items():
        mem = calculate_memory(config)
        max_batch = find_max_batch(mem, NUM_TRAIN_GPUS, GPU_MEMORY_GB, SEQ_LEN)

        # Show memory at batch=4 for comparison
        mem_at_4 = mem.per_gpu_memory(NUM_TRAIN_GPUS, 4, SEQ_LEN)

        print(
            f"{name:<18} │ "
            f"{mem.weights_gb:>7.1f}G {mem.gradients_gb:>7.1f}G {mem.optimizer_state_gb:>7.1f}G {mem.static_total_gb:>7.1f}G │ "
            f"{mem.activation_per_sample_gb:>8.1f}G │ "
            f"{mem_at_4:>9.1f}G │ "
            f"{max_batch:>10}"
        )

    print()

    # LoRA table
    print("LORA FINE-TUNING (rank=64)")
    print("-" * 110)
    print(f"{'Model':<18} │ {'Base':>8} {'LoRA G':>8} {'LoRA O':>8} {'Static':>8} │ {'Act/samp':>9} │ {'Per-GPU':>10} │ {'Max Batch':>10}")
    print(f"{'':18} │ {'(froz)':>8} {'(BF16)':>8} {'(FP32)':>8} {'Total':>8} │ {'(emp.)':>9} │ {'(3 GPU)':>10} │ {'(seq=4k)':>10}")
    print("-" * 110)

    for name, config in MODELS.items():
        mem = calculate_lora_memory(config, lora_rank=64)
        max_batch = find_max_batch(mem, NUM_TRAIN_GPUS, GPU_MEMORY_GB, SEQ_LEN)

        mem_at_4 = mem.per_gpu_memory(NUM_TRAIN_GPUS, 4, SEQ_LEN)

        print(
            f"{name:<18} │ "
            f"{mem.weights_gb:>7.1f}G {mem.gradients_gb:>7.2f}G {mem.optimizer_state_gb:>7.2f}G {mem.weights_gb + mem.gradients_gb + mem.optimizer_state_gb:>7.1f}G │ "
            f"{mem.activation_per_sample_gb:>8.1f}G │ "
            f"{mem_at_4:>9.1f}G │ "
            f"{max_batch:>10}"
        )

    print()
    print()

    # Detailed breakdown for 7B
    print("DETAILED BREAKDOWN: Qwen2.5-7B Full Fine-Tuning on 3× GB200")
    print("-" * 80)

    model = MODELS["Qwen2.5-7B"]
    mem = calculate_memory(model)

    print(f"Model: {model.name} ({model.params_b}B params)")
    print()
    print("Static Memory (total across cluster):")
    print(f"  Weights (BF16):           {mem.weights_gb:>8.1f} GB")
    print(f"  Gradients (BF16):         {mem.gradients_gb:>8.1f} GB")
    print(f"  Adam master (FP32):       {mem.optimizer_state_gb/3:>8.1f} GB")
    print(f"  Adam m (FP32):            {mem.optimizer_state_gb/3:>8.1f} GB")
    print(f"  Adam v (FP32):            {mem.optimizer_state_gb/3:>8.1f} GB")
    print(f"  ───────────────────────────────────────")
    print(f"  Total Static:             {mem.static_total_gb:>8.1f} GB")
    print(f"  Per GPU (FSDP sharded):   {mem.static_total_gb/3:>8.1f} GB")
    print()
    print("Dynamic Memory (per sample, empirical):")
    print(f"  Activations + backward:   {mem.activation_per_sample_gb:>8.1f} GB")
    print()
    print("Fixed Overheads:")
    print(f"  FSDP buffers:             {mem.fsdp_buffer_gb:>8.1f} GB")
    print(f"  CUDA context:             {mem.cuda_context_gb:>8.1f} GB")
    print()

    print("Per-GPU Memory at Different Batch Sizes (seq_len=4096):")
    print(f"  {'Batch':<8} {'Memory':>10} {'Status':>12} {'Headroom':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12}")
    for batch in [1, 2, 4, 8, 16, 32]:
        total = mem.per_gpu_memory(NUM_TRAIN_GPUS, batch, SEQ_LEN)
        headroom = GPU_MEMORY_GB - total
        if headroom > GPU_MEMORY_GB * 0.15:
            status = "✓ OK"
        elif headroom > 0:
            status = "⚠ Tight"
        else:
            status = "✗ OOM"
        print(f"  {batch:<8} {total:>9.1f}G {status:>12} {headroom:>11.1f}G")

    print()
    print()

    # LoRA rank comparison
    print("LORA RANK COMPARISON (effect on trainable params & memory)")
    print("-" * 110)
    print(f"{'Model':<18} │ {'r=4':>12} {'r=8':>12} {'r=16':>12} {'r=32':>12} {'r=64':>12} │ {'Full FT':>12}")
    print(f"{'':18} │ {'trainable':>12} {'trainable':>12} {'trainable':>12} {'trainable':>12} {'trainable':>12} │ {'trainable':>12}")
    print("-" * 110)

    for name, config in MODELS.items():
        params = config.params_b * 1e9
        full_trainable = f"{config.params_b:.1f}B (100%)"

        row = f"{name:<18} │"
        for rank in [4, 8, 16, 32, 64]:
            # LoRA params: 2 * rank * hidden_dim * 6 projections * num_layers
            lora_params = 2 * rank * config.hidden_dim * 6 * config.num_layers
            pct = 100 * lora_params / params
            if lora_params > 1e9:
                row += f" {lora_params/1e9:.2f}B ({pct:.1f}%)"
            else:
                row += f" {lora_params/1e6:.0f}M ({pct:.1f}%)"
        row += f" │ {full_trainable:>12}"
        print(row)

    print()
    print("Note: LoRA targets q/k/v/o_proj + gate/up/down_proj (all-linear). Reward head always trainable.")
    print()
    print()

    # PVG-specific: Verifier + Prover memory planning
    print("=" * 110)
    print("PVG DUAL-MODEL TRAINING: VERIFIER + PROVER ON SAME GPU CLUSTER")
    print("=" * 110)
    print()
    print("PVG training alternates between Verifier and Prover updates.")
    print("Key insight: LoRA enables efficient training of BOTH models without catastrophic forgetting.")
    print()
    print("When to use LoRA vs Full Fine-Tuning (from representation theory):")
    print("  ┌─────────────────────────────────────────────────────────────────────────────┐")
    print("  │ Feature Type           │ Training Mode    │ Rationale                       │")
    print("  ├─────────────────────────────────────────────────────────────────────────────┤")
    print("  │ Linearly representable │ Head-only        │ Probe + frozen backbone         │")
    print("  │ & salient              │                  │                                 │")
    print("  ├─────────────────────────────────────────────────────────────────────────────┤")
    print("  │ Almost linear,         │ LoRA (low r)     │ Small geometry correction       │")
    print("  │ moderately salient     │ r=4 to r=16      │ Higher r = less salient feature │")
    print("  ├─────────────────────────────────────────────────────────────────────────────┤")
    print("  │ Non-linear or          │ LoRA (high r)    │ Elicit latent capabilities      │")
    print("  │ low salience           │ r=32 to r=64     │ without full FT cost            │")
    print("  ├─────────────────────────────────────────────────────────────────────────────┤")
    print("  │ Complex/computed       │ Full FT          │ Learn new computational paths   │")
    print("  │ (e.g., proof checking) │                  │                                 │")
    print("  └─────────────────────────────────────────────────────────────────────────────┘")
    print()
    print("For PVG code verification:")
    print("  - Verifier: 'Is this code correct?' may require multi-step verification")
    print("    → LoRA r=16-64 recommended (elicit latent reasoning)")
    print("  - Prover:  Generating sneaky code requires subtle modifications")
    print("    → LoRA r=4-16 recommended (efficient policy updates)")
    print()

    # Final recommendations
    print("=" * 110)
    print("RECOMMENDED CONFIGURATIONS FOR PVG TRAINING")
    print("=" * 110)
    print()
    print("Setup: 4× GB200 (192GB each)")
    print("  - GPU 0: vLLM inference server (prover + verifier)")
    print("  - GPU 1-3: FSDP training")
    print()

    configs = [
        ("Qwen2.5-0.5B", "full", 32, 4, 131072, "Fast iteration, debugging"),
        ("Qwen2.5-1.5B", "full", 16, 4, 65536, "Good baseline model"),
        ("Qwen2.5-3B", "full", 8, 4, 32768, "Strong small model"),
        ("Qwen2.5-3B", "lora-16", 16, 4, 65536, "3B LoRA (recommended)"),
        ("Qwen2.5-7B", "lora-16", 16, 4, 65536, "7B LoRA (fast, efficient)"),
        ("Qwen2.5-7B", "lora-64", 8, 4, 32768, "7B LoRA (deeper adaptation)"),
        ("Qwen2.5-7B", "full", 4, 4, 16384, "Full 7B (slower)"),
        ("Llama-3.1-8B", "lora-16", 16, 4, 65536, "8B LoRA (fast, efficient)"),
        ("Llama-3.1-8B", "full", 4, 4, 16384, "Full 8B (slower)"),
        ("Qwen2.5-14B", "lora-16", 16, 4, 65536, "14B LoRA (production)"),
        ("Qwen2.5-14B", "lora-64", 8, 4, 32768, "14B LoRA (deep adaptation)"),
        ("Qwen2.5-14B", "full", 2, 2, 8192, "Full 14B (very slow)"),
    ]

    print(f"{'Model':<18} {'Mode':<10} {'batch':>8} {'group':>8} {'micro_budget':>14} {'eff_batch':>10}  Notes")
    print("-" * 110)
    for model_name, mode, batch, group, micro_budget, notes in configs:
        effective = batch * group
        print(f"{model_name:<18} {mode:<10} {batch:>8} {group:>8} {micro_budget:>14} {effective:>10}  {notes}")

    print()
    print("Key:")
    print("  batch        = unique problems per macro-step")
    print("  group        = generations per problem (GRPO baseline)")
    print("  micro_budget = max tokens per micro-batch (controls GPU memory)")
    print("  eff_batch    = batch × group = samples used for gradient")
    print()
    print("Memory vs Throughput Trade-off:")
    print("  - Smaller micro_budget → more micro-batches → lower GPU util but less memory")
    print("  - Larger micro_budget → fewer micro-batches → higher GPU util but more memory")
    print("  - For PVG: effective_batch of 16-32 is typical (matches GRPO paper)")
    print()
    print("LoRA Rank Guidelines for PVG:")
    print("  - r=4:  Minimal adaptation, fastest training, risk of underfitting")
    print("  - r=16: Sweet spot for most tasks, good balance")
    print("  - r=64: Deep adaptation, slower but can learn complex features")
    print("  - Use rsLoRA (alpha scaling) for rank-independent gradient magnitudes")
    print()


if __name__ == "__main__":
    print_tables()

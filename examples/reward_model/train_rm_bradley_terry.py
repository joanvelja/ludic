"""Train a reward model using Bradley-Terry loss on preference data.

This script demonstrates training a reward model from pairwise preferences
using RMTrainer with:
- Pair-aware micro-batching
- LR scheduler with warmup
- Early stopping
- LoRA support
- Label smoothing

Usage:
    # Train on HuggingFace dataset
    uv run python train_rm_bradley_terry.py \
        --model Qwen/Qwen2.5-0.5B \
        --hf-dataset stanfordnlp/SHP \
        --hf-split train \
        --steps 100

    # With evaluation and scheduler
    uv run python train_rm_bradley_terry.py \
        --model Qwen/Qwen2.5-0.5B \
        --hf-dataset stanfordnlp/SHP \
        --hf-split train \
        --eval-split test \
        --eval-limit 500 \
        --steps 500 \
        --lr-scheduler cosine \
        --warmup-steps 50

    # With LoRA
    uv run python train_rm_bradley_terry.py \
        --model Qwen/Qwen2.5-0.5B \
        --hf-dataset stanfordnlp/SHP \
        --hf-split train \
        --steps 500 \
        --lora --lora-rank 8 --lora-alpha 16
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class GPUStatsCollector:
    """Collect GPU utilization stats during training.

    Runs a background thread that samples GPU utilization and memory usage
    at regular intervals. Provides averaged stats for each training step.
    """

    def __init__(self, sample_interval_ms: int = 100):
        """Initialize the GPU stats collector.

        Args:
            sample_interval_ms: How often to sample GPU stats (in milliseconds).
        """
        self.sample_interval = sample_interval_ms / 1000.0
        self._samples: List[Dict[str, float]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Check if pynvml is available for detailed GPU stats
        self._use_pynvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._use_pynvml = True
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self._device_count)
            ]
        except (ImportError, Exception):
            pass

    def _sample_stats(self) -> Dict[str, float]:
        """Sample current GPU stats."""
        stats = {}

        if torch.cuda.is_available():
            # Basic PyTorch CUDA stats
            device_idx = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
            stats["gpu_memory_allocated_gb"] = allocated
            stats["gpu_memory_reserved_gb"] = reserved

            if self._use_pynvml and device_idx < len(self._handles):
                try:
                    handle = self._handles[device_idx]
                    # GPU utilization (compute)
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats["gpu_utilization_pct"] = util.gpu
                    stats["gpu_memory_util_pct"] = util.memory

                    # Memory info
                    mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                    stats["gpu_memory_total_gb"] = mem_info.total / (1024**3)
                    stats["gpu_memory_used_gb"] = mem_info.used / (1024**3)
                    stats["gpu_memory_free_gb"] = mem_info.free / (1024**3)

                    # Power usage
                    try:
                        power = self._pynvml.nvmlDeviceGetPowerUsage(handle)
                        stats["gpu_power_w"] = power / 1000.0  # mW to W
                    except Exception:
                        pass

                    # Temperature
                    try:
                        temp = self._pynvml.nvmlDeviceGetTemperature(
                            handle, self._pynvml.NVML_TEMPERATURE_GPU
                        )
                        stats["gpu_temp_c"] = temp
                    except Exception:
                        pass

                except Exception:
                    pass

        return stats

    def _collection_loop(self):
        """Background thread that collects GPU stats."""
        while self._running:
            stats = self._sample_stats()
            with self._lock:
                self._samples.append(stats)
            time.sleep(self.sample_interval)

    def start(self):
        """Start collecting GPU stats in the background."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop collecting GPU stats."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_step_stats(self) -> Dict[str, float]:
        """Get averaged stats for the current step and reset samples.

        Returns:
            Dict with averaged GPU stats since last call.
        """
        with self._lock:
            samples = self._samples
            self._samples = []

        if not samples:
            return {}

        # Average all numeric stats
        result = {}
        keys = samples[0].keys()
        for key in keys:
            values = [s[key] for s in samples if key in s]
            if values:
                result[f"avg_{key}"] = sum(values) / len(values)
                result[f"max_{key}"] = max(values)

        result["gpu_samples_count"] = len(samples)
        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ludic.training.algorithm import make_bradley_terry
from ludic.training.preference_utils import (
    preference_dataset_to_saw_items,
    shuffle_preference_pairs,
)
from ludic.training.types import SAWBatch, SAWItem, BatchSource
from ludic.training.batching.micro_batching import collate_saw_items
from ludic.training.rm_trainer import RMTrainer, RMTrainerConfig
from ludic.training.loggers import PrintLogger
from ludic.training.classification_metrics import PreferenceMetrics

# Configure logging: WARNING level by default, but keep script logger at INFO
# for critical progress points (model/data loading, training start/end, eval)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress noisy warnings and logs
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("ludic.training.preference_utils").setLevel(logging.ERROR)


def log_distribution_samples(
    name: str,
    pref_data: List[Dict[str, Any]],
    saw_items: List[SAWItem],
    tokenizer,
    output_dir: Path,
    n_samples: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Log random samples from a data distribution for inspection.

    Logs both raw preference pairs and tokenized SAWItems to help diagnose
    train/eval distribution differences.

    Args:
        name: Distribution name (e.g., "train", "eval")
        pref_data: Raw preference pairs with prompt/chosen/rejected
        saw_items: Tokenized SAWItems
        tokenizer: Tokenizer for decoding
        output_dir: Directory to save samples JSON
        n_samples: Number of samples to log
        seed: Random seed for reproducibility

    Returns:
        Dict with sampled data for JSON serialization
    """
    rng = random.Random(seed)

    def extract_action_ids(item: SAWItem) -> List[int]:
        """Return action (completion) token IDs from a SAWItem."""
        if not item.action_mask or len(item.action_mask) != len(item.input_ids):
            return list(item.input_ids)
        return [tok for tok, mask in zip(item.input_ids, item.action_mask) if mask]

    def extract_prompt_ids(item: SAWItem) -> List[int]:
        """Return prompt (non-action) token IDs from a SAWItem."""
        if not item.action_mask or len(item.action_mask) != len(item.input_ids):
            return []
        return [tok for tok, mask in zip(item.input_ids, item.action_mask) if not mask]

    # Sample preference pairs
    pair_indices = rng.sample(range(len(pref_data)), min(n_samples, len(pref_data)))

    logger.info(f"\n{'='*60}")
    logger.info(f"DISTRIBUTION SAMPLES: {name.upper()} ({len(pref_data)} pairs)")
    logger.info(f"{'='*60}")

    samples_data = {
        "distribution": name,
        "total_pairs": len(pref_data),
        "total_saw_items": len(saw_items),
        "samples": [],
    }

    # Compute some basic stats
    prompt_lengths = [len(p["prompt"]) for p in pref_data]
    chosen_lengths = [len(p["chosen"]) for p in pref_data]
    rejected_lengths = [len(p["rejected"]) for p in pref_data]

    samples_data["stats"] = {
        "prompt_char_len": {"mean": np.mean(prompt_lengths), "std": np.std(prompt_lengths), "min": min(prompt_lengths), "max": max(prompt_lengths)},
        "chosen_char_len": {"mean": np.mean(chosen_lengths), "std": np.std(chosen_lengths), "min": min(chosen_lengths), "max": max(chosen_lengths)},
        "rejected_char_len": {"mean": np.mean(rejected_lengths), "std": np.std(rejected_lengths), "min": min(rejected_lengths), "max": max(rejected_lengths)},
    }

    logger.info(f"Stats: prompt_len={np.mean(prompt_lengths):.0f}±{np.std(prompt_lengths):.0f}, "
                f"chosen_len={np.mean(chosen_lengths):.0f}±{np.std(chosen_lengths):.0f}, "
                f"rejected_len={np.mean(rejected_lengths):.0f}±{np.std(rejected_lengths):.0f}")

    for i, idx in enumerate(pair_indices):
        pair = pref_data[idx]

        # Find corresponding SAWItems (chosen and rejected for this pair)
        # SAWItems are ordered: [chosen_0, rejected_0, chosen_1, rejected_1, ...]
        saw_chosen = saw_items[idx * 2] if idx * 2 < len(saw_items) else None
        saw_rejected = saw_items[idx * 2 + 1] if idx * 2 + 1 < len(saw_items) else None

        logger.info(f"\n--- Sample {i+1} (idx={idx}) ---")
        logger.info(f"PROMPT ({len(pair['prompt'])} chars): {pair['prompt'][:200]}{'...' if len(pair['prompt']) > 200 else ''}")
        logger.info(f"CHOSEN ({len(pair['chosen'])} chars): {pair['chosen'][:200]}{'...' if len(pair['chosen']) > 200 else ''}")
        logger.info(f"REJECTED ({len(pair['rejected'])} chars): {pair['rejected'][:200]}{'...' if len(pair['rejected']) > 200 else ''}")

        sample_entry = {
            "index": idx,
            "prompt": pair["prompt"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "prompt_len": len(pair["prompt"]),
            "chosen_len": len(pair["chosen"]),
            "rejected_len": len(pair["rejected"]),
        }

        if saw_chosen is not None:
            # Full sequence (prompt + completion) to verify chat template
            full_ids = list(saw_chosen.input_ids)
            full_decoded = tokenizer.decode(full_ids, skip_special_tokens=False)
            prompt_ids = extract_prompt_ids(saw_chosen)
            prompt_decoded = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            chosen_ids = extract_action_ids(saw_chosen)
            chosen_decoded = tokenizer.decode(chosen_ids, skip_special_tokens=False)

            # Check for chat template markers (Qwen-style)
            has_chat_template = "<|im_start|>" in full_decoded or "<|user|>" in full_decoded

            sample_entry["chosen_full_len"] = len(full_ids)
            sample_entry["chosen_prompt_len"] = len(prompt_ids)
            sample_entry["chosen_token_len"] = len(chosen_ids)
            sample_entry["chosen_has_chat_template"] = has_chat_template
            sample_entry["chosen_full_decoded"] = full_decoded  # Full text for JSON
            sample_entry["chosen_prompt_decoded"] = prompt_decoded
            sample_entry["chosen_action_decoded"] = chosen_decoded

            logger.info(f"CHOSEN FULL SEQ ({len(full_ids)} tokens, chat_template={has_chat_template}):")
            logger.info(f"  PROMPT PART ({len(prompt_ids)} tokens): {prompt_decoded[:400]}{'...' if len(prompt_decoded) > 400 else ''}")
            logger.info(f"  ACTION PART ({len(chosen_ids)} tokens): {chosen_decoded}")

        if saw_rejected is not None:
            # Full sequence (prompt + completion) to verify chat template
            full_ids = list(saw_rejected.input_ids)
            full_decoded = tokenizer.decode(full_ids, skip_special_tokens=False)
            prompt_ids = extract_prompt_ids(saw_rejected)
            prompt_decoded = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            rejected_ids = extract_action_ids(saw_rejected)
            rejected_decoded = tokenizer.decode(rejected_ids, skip_special_tokens=False)

            # Check for chat template markers
            has_chat_template = "<|im_start|>" in full_decoded or "<|user|>" in full_decoded

            sample_entry["rejected_full_len"] = len(full_ids)
            sample_entry["rejected_prompt_len"] = len(prompt_ids)
            sample_entry["rejected_token_len"] = len(rejected_ids)
            sample_entry["rejected_has_chat_template"] = has_chat_template
            sample_entry["rejected_full_decoded"] = full_decoded  # Full text for JSON
            sample_entry["rejected_prompt_decoded"] = prompt_decoded
            sample_entry["rejected_action_decoded"] = rejected_decoded

            logger.info(f"REJECTED FULL SEQ ({len(full_ids)} tokens, chat_template={has_chat_template}):")
            logger.info(f"  PROMPT PART ({len(prompt_ids)} tokens): {prompt_decoded[:400]}{'...' if len(prompt_decoded) > 400 else ''}")
            logger.info(f"  ACTION PART ({len(rejected_ids)} tokens): {rejected_decoded}")

        samples_data["samples"].append(sample_entry)

    # Save to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_file = output_dir / f"{name}_samples.json"
    with open(samples_file, "w") as f:
        json.dump(samples_data, f, indent=2, default=str)
    logger.info(f"\nSaved {name} samples to {samples_file}")

    return samples_data


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hf_preferences(
    dataset_name: str,
    split: str = "train",
    limit: Optional[int] = None,
    min_response_length: int = 10,
    diagnostic_mode: Optional[str] = None,
    diagnostic_length_threshold: int = 50,
    diagnostic_shuffle_rate: float = 0.5,
    diagnostic_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load preference data from HuggingFace datasets.

    Supports common preference dataset formats.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "stanfordnlp/SHP")
        split: Dataset split to load
        limit: Maximum number of samples to return
        min_response_length: Minimum character length for responses.
            Samples where either chosen or rejected is shorter than this
            are filtered out (handles garbage data like single-char responses).
        diagnostic_mode: Diagnostic mode for pipeline integrity testing:
            - "label_shuffle": Randomly flip labels (expect ~50% accuracy)
            - "prompt_only": Replace responses with empty (expect ~50% accuracy)
            - "random_negatives": Replace rejected with cross-prompt response
            - "length_matched": Filter to pairs with similar length responses
        diagnostic_length_threshold: Max |len(chosen)-len(rejected)| for length_matched
        diagnostic_shuffle_rate: Fraction of labels to shuffle in label_shuffle mode
        diagnostic_seed: Random seed for diagnostic transformations
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset(dataset_name, split=split)
    data: List[Dict[str, Any]] = []
    filtered_count = 0

    for row in ds:
        prompt = None
        chosen = None
        rejected = None

        # Handle different dataset formats
        if "prompt" in row and "chosen" in row and "rejected" in row:
            # Standard format
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
        elif "history" in row and "human_ref_A" in row:
            # SHP format
            prompt = row.get("history", "")
            if row.get("labels", 1) == 1:
                chosen, rejected = row["human_ref_A"], row["human_ref_B"]
            else:
                chosen, rejected = row["human_ref_B"], row["human_ref_A"]
        elif "chosen" in row and "rejected" in row:
            # Format with conversation lists (e.g., Anthropic HH)
            # Extract the last assistant turn
            if isinstance(row["chosen"], list):
                prompt = row["chosen"][0]["content"] if row["chosen"] else ""
                chosen = row["chosen"][-1]["content"] if len(row["chosen"]) > 1 else ""
                rejected = row["rejected"][-1]["content"] if row["rejected"] else ""
            else:
                prompt = ""
                chosen = row["chosen"]
                rejected = row["rejected"]

        if prompt is None:
            continue

        # Filter out samples with responses that are too short
        # This handles garbage data like single-character responses in SHP
        if len(chosen) < min_response_length or len(rejected) < min_response_length:
            filtered_count += 1
            continue

        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        if limit is not None and len(data) >= limit:
            break

    if filtered_count > 0:
        logger.info(
            f"Filtered {filtered_count} samples with responses shorter than "
            f"{min_response_length} characters"
        )

    # Apply diagnostic mode transformations AFTER loading data
    if diagnostic_mode == "label_shuffle":
        rng = random.Random(diagnostic_seed)
        shuffled = 0
        for row in data:
            if rng.random() < diagnostic_shuffle_rate:
                row["chosen"], row["rejected"] = row["rejected"], row["chosen"]
                shuffled += 1
        logger.info(
            f"DIAGNOSTIC label_shuffle: flipped {shuffled}/{len(data)} labels "
            f"({100*shuffled/len(data):.1f}%)"
        )

    elif diagnostic_mode == "prompt_only":
        # Use minimal placeholder that survives tokenization
        # Single space gets filtered out, so use "[empty]" which tokenizes to real tokens
        placeholder = "[empty]"
        for row in data:
            row["chosen"] = placeholder
            row["rejected"] = placeholder
        logger.info(
            f"DIAGNOSTIC prompt_only: all {len(data)} completions replaced with '{placeholder}'"
        )

    elif diagnostic_mode == "random_negatives":
        rng = random.Random(diagnostic_seed)
        all_responses = [r["chosen"] for r in data] + [r["rejected"] for r in data]
        replaced = 0
        for i, row in enumerate(data):
            # Pick a response from a different pair
            attempts = 0
            while attempts < 100:
                random_idx = rng.randint(0, len(all_responses) - 1)
                # Different pair: random_idx // 2 != i (each pair contributes 2 responses)
                if random_idx // 2 != i:
                    row["rejected"] = all_responses[random_idx]
                    replaced += 1
                    break
                attempts += 1
        logger.info(
            f"DIAGNOSTIC random_negatives: replaced {replaced}/{len(data)} rejected "
            f"responses with cross-prompt responses"
        )

    elif diagnostic_mode == "length_matched":
        original_len = len(data)
        data = [
            row for row in data
            if abs(len(row["chosen"]) - len(row["rejected"])) < diagnostic_length_threshold
        ]
        logger.info(
            f"DIAGNOSTIC length_matched: {len(data)}/{original_len} pairs "
            f"({100*len(data)/original_len:.1f}%) with |Δlen| < {diagnostic_length_threshold}"
        )

    return data


class AsyncBatchSource(BatchSource):
    """Async batch source for preference training.

    Wraps a list of SAWItems and provides async batching.
    """

    def __init__(
        self,
        items: List[SAWItem],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.items = items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._idx = 0
        self._epoch = 0

        if shuffle:
            self.items = shuffle_preference_pairs(items, seed=seed)

    async def next_batch(self) -> SAWBatch:
        """Get next batch.

        Note: async is required by BatchSource protocol, but this implementation
        only does synchronous list operations. No actual async I/O occurs.
        """
        if self._idx >= len(self.items):
            self._epoch += 1
            self._idx = 0
            if self.shuffle:
                self.items = shuffle_preference_pairs(
                    self.items, seed=self.seed + self._epoch
                )

        end_idx = min(self._idx + self.batch_size, len(self.items))
        batch_items = self.items[self._idx : end_idx]
        self._idx = end_idx

        return SAWBatch(
            items=batch_items,
            meta={"num_samples": len(batch_items), "epoch": self._epoch},
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_items: List[SAWItem],
    algorithm,
    device: str,
    pad_token_id: int,
    batch_size: int = 16,
    *,
    step: Optional[int] = None,
    split: str = "eval",
) -> Dict[str, Any]:
    """Evaluate model on a held-out set.

    Returns metrics and reward distribution data.
    """
    model.eval()
    metrics = PreferenceMetrics(
        calibration_bins=10,
        thresholds=[0.5],
        beta=getattr(algorithm.loss, "beta", 1.0),
    )

    total_loss = 0.0
    total_accuracy = 0.0
    total_margin = 0.0
    total_pairs = 0

    chosen_scores: List[float] = []
    rejected_scores: List[float] = []
    labels_all: List[float] = []
    chosen_lengths: List[float] = []
    rejected_lengths: List[float] = []
    item_scores: List[float] = []
    item_lengths: List[float] = []

    # Process in batches
    for i in range(0, len(eval_items), batch_size * 2):
        batch_items = eval_items[i : i + batch_size * 2]
        if len(batch_items) < 2:
            continue

        # Collate items to tensors
        collated = collate_saw_items(
            batch_items,
            pad_token_id=pad_token_id,
            device=torch.device(device),
        )

        input_ids = collated["input_ids"].to(device)
        attention_mask = collated["attention_mask"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Build batch dict for loss computation
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_mask": collated["action_mask"].to(device),
            "meta": {
                "pair_id": [it.meta["pair_id"] for it in batch_items],
                "role": [it.meta["role"] for it in batch_items],
                "label": [it.meta["label"] for it in batch_items],
            },
        }

        loss, stats = algorithm.loss.compute(logits, batch_dict)

        num_pairs = int(stats.get("num_pairs", 0))
        if num_pairs > 0:
            total_loss += float(loss) * num_pairs
            total_accuracy += float(stats.get("accuracy", 0)) * num_pairs
            total_margin += float(stats.get("margin", 0)) * num_pairs
            total_pairs += num_pairs

            # Track reward distributions (per-pair, not batch means)

            # Update calibration/AUROC metrics with per-pair scores
            if logits.ndim == 1:
                scores = logits
            elif logits.ndim == 2 and logits.shape[1] == 1:
                scores = logits.squeeze(-1)
            else:
                raise ValueError(
                    f"Expected reward-model logits [B] or [B,1], got {tuple(logits.shape)}"
                )

            meta = batch_dict["meta"]
            pair_ids = meta["pair_id"]
            roles = meta["role"]
            labels_list = meta["label"]
            pair_map: Dict[str, Dict[str, int]] = {}
            for idx, (pid, role) in enumerate(zip(pair_ids, roles)):
                if pid not in pair_map:
                    pair_map[pid] = {}
                pair_map[pid][role] = idx

            action_lengths = collated["action_mask"].sum(dim=1)

            chosen_list: List[torch.Tensor] = []
            rejected_list: List[torch.Tensor] = []
            label_list: List[float] = []
            chosen_len_list: List[float] = []
            rejected_len_list: List[float] = []
            for pid, indices in pair_map.items():
                if "chosen" in indices and "rejected" in indices:
                    c_idx = indices["chosen"]
                    r_idx = indices["rejected"]
                    chosen_list.append(scores[c_idx])
                    rejected_list.append(scores[r_idx])
                    label_list.append(labels_list[indices["chosen"]])
                    chosen_len_list.append(float(action_lengths[c_idx].item()))
                    rejected_len_list.append(float(action_lengths[r_idx].item()))

            if chosen_list:
                chosen_tensor = torch.stack(chosen_list)
                rejected_tensor = torch.stack(rejected_list)
                labels_tensor = torch.tensor(
                    label_list, device=scores.device, dtype=scores.dtype
                )
                metrics.update(
                    chosen_scores=chosen_tensor,
                    rejected_scores=rejected_tensor,
                    labels=labels_tensor,
                )

                chosen_cpu = chosen_tensor.detach().cpu().float().tolist()
                rejected_cpu = rejected_tensor.detach().cpu().float().tolist()
                labels_cpu = labels_tensor.detach().cpu().float().tolist()

                chosen_scores.extend(chosen_cpu)
                rejected_scores.extend(rejected_cpu)
                labels_all.extend(labels_cpu)
                chosen_lengths.extend(chosen_len_list)
                rejected_lengths.extend(rejected_len_list)
                for score, length in zip(chosen_cpu, chosen_len_list):
                    item_scores.append(score)
                    item_lengths.append(length)
                for score, length in zip(rejected_cpu, rejected_len_list):
                    item_scores.append(score)
                    item_lengths.append(length)

    if total_pairs == 0:
        return {
            "eval_loss": 0.0,
            "eval_accuracy": 0.0,
            "eval_margin": 0.0,
            "eval_pairs": 0,
            "eval_ece": 0.0,
            "eval_roc_auc": 0.0,
            "chosen_scores": [],
            "rejected_scores": [],
            "eval_delta_mean": 0.0,
            "eval_delta_std": 0.0,
            "eval_delta_p05": 0.0,
            "eval_delta_p50": 0.0,
            "eval_delta_p95": 0.0,
            "eval_delta_neg_rate": 0.0,
            "eval_reward_chosen_mean": 0.0,
            "eval_reward_chosen_std": 0.0,
            "eval_reward_rejected_mean": 0.0,
            "eval_reward_rejected_std": 0.0,
            "eval_reward_abs_max": 0.0,
            "eval_reward_len_corr": float("nan"),
            "eval_delta_len_diff_corr": float("nan"),
            "eval_len_diff_mean": 0.0,
            "eval_len_diff_std": 0.0,
            "eval_calib_bins": {
                "edges": [],
                "counts": [],
                "acc": [],
                "conf": [],
                "gap": [],
            },
            "eval_diag_step": step,
            "eval_diag_split": split,
        }

    metric_results = metrics.compute()

    delta_values = [c - r for c, r in zip(chosen_scores, rejected_scores)]
    delta_stats = _summarize_distribution(delta_values)

    chosen_stats = _summarize_distribution(chosen_scores)
    rejected_stats = _summarize_distribution(rejected_scores)
    reward_abs_max = 0.0
    if chosen_scores or rejected_scores:
        reward_abs_max = float(
            max(
                max((abs(v) for v in chosen_scores), default=0.0),
                max((abs(v) for v in rejected_scores), default=0.0),
            )
        )

    reward_len_corr = _safe_corrcoef(item_scores, item_lengths)
    len_diff = [c - r for c, r in zip(chosen_lengths, rejected_lengths)]
    delta_len_diff_corr = _safe_corrcoef(delta_values, len_diff)
    len_diff_stats = _summarize_distribution(len_diff)

    calibration_bins = _compute_calibration_bins(
        delta_values,
        labels_all,
        beta=getattr(algorithm.loss, "beta", 1.0),
        num_bins=metrics.calibration_bins,
    )

    return {
        "eval_loss": total_loss / total_pairs,
        "eval_accuracy": total_accuracy / total_pairs,
        "eval_margin": total_margin / total_pairs,
        "eval_pairs": total_pairs,
        "eval_ece": metric_results.get("ece", 0.0),
        "eval_roc_auc": metric_results.get("roc_auc", 0.0),
        "chosen_scores": chosen_scores,
        "rejected_scores": rejected_scores,
        "eval_delta_mean": delta_stats["mean"],
        "eval_delta_std": delta_stats["std"],
        "eval_delta_p05": delta_stats["p05"],
        "eval_delta_p50": delta_stats["p50"],
        "eval_delta_p95": delta_stats["p95"],
        "eval_delta_neg_rate": delta_stats["neg_rate"],
        "eval_reward_chosen_mean": chosen_stats["mean"],
        "eval_reward_chosen_std": chosen_stats["std"],
        "eval_reward_rejected_mean": rejected_stats["mean"],
        "eval_reward_rejected_std": rejected_stats["std"],
        "eval_reward_abs_max": reward_abs_max,
        "eval_reward_len_corr": reward_len_corr,
        "eval_delta_len_diff_corr": delta_len_diff_corr,
        "eval_len_diff_mean": len_diff_stats["mean"],
        "eval_len_diff_std": len_diff_stats["std"],
        "eval_calib_bins": calibration_bins,
        "eval_diag_step": step,
        "eval_diag_split": split,
    }


def _summarize_distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "neg_rate": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        p05 = p50 = p95 = float(arr[0])
    else:
        p05, p50, p95 = np.quantile(arr, [0.05, 0.5, 0.95]).tolist()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "neg_rate": float((arr < 0).mean()),
    }


def _safe_corrcoef(x_values: List[float], y_values: List[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return float("nan")
    x_arr = np.asarray(x_values, dtype=np.float64)
    y_arr = np.asarray(y_values, dtype=np.float64)
    if x_arr.size != y_arr.size:
        return float("nan")
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _compute_calibration_bins(
    delta_values: List[float],
    labels: List[float],
    *,
    beta: float,
    num_bins: int,
) -> Dict[str, List[float]]:
    if not delta_values or not labels or len(delta_values) != len(labels):
        return {"edges": [], "counts": [], "acc": [], "conf": [], "gap": []}
    delta_arr = np.asarray(delta_values, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-beta * delta_arr))
    targets = (labels_arr > 0.5).astype(np.float64)

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    counts: List[int] = []
    accs: List[float] = []
    confs: List[float] = []
    gaps: List[float] = []
    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            mask = (probs >= edges[i]) & (probs < edges[i + 1])
        count = int(mask.sum())
        if count > 0:
            acc = float(targets[mask].mean())
            conf = float(probs[mask].mean())
            gap = conf - acc
        else:
            acc = 0.0
            conf = 0.0
            gap = 0.0
        counts.append(count)
        accs.append(acc)
        confs.append(conf)
        gaps.append(gap)
    return {
        "edges": edges.tolist(),
        "counts": counts,
        "acc": accs,
        "conf": confs,
        "gap": gaps,
    }


class MetricsLogger:
    """Log metrics to CSV file."""

    def __init__(self, output_path: Path, track_gpu: bool = False):
        self.output_path = output_path
        self.rows: List[Dict[str, Any]] = []
        self.fieldnames = [
            "step", "train_loss", "train_acc", "train_margin",
            "eval_loss", "eval_acc", "eval_margin", "eval_pairs",
            "eval_ece", "eval_roc_auc",
            "eval_delta_mean", "eval_delta_std", "eval_delta_p05",
            "eval_delta_p50", "eval_delta_p95", "eval_delta_neg_rate",
            "eval_reward_chosen_mean", "eval_reward_chosen_std",
            "eval_reward_rejected_mean", "eval_reward_rejected_std",
            "eval_reward_abs_max",
            "eval_reward_len_corr", "eval_delta_len_diff_corr",
            "eval_len_diff_mean", "eval_len_diff_std",
            "eval_calib_bins_json",
        ]
        if track_gpu:
            self.fieldnames.extend([
                "avg_gpu_utilization_pct", "max_gpu_utilization_pct",
                "avg_gpu_memory_used_gb", "max_gpu_memory_used_gb",
                "avg_gpu_power_w", "gpu_samples_count",
            ])

    def log(
        self,
        step: int,
        train_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        train_margin: Optional[float] = None,
        eval_loss: Optional[float] = None,
        eval_acc: Optional[float] = None,
        eval_margin: Optional[float] = None,
        eval_pairs: Optional[int] = None,
        eval_ece: Optional[float] = None,
        eval_roc_auc: Optional[float] = None,
        gpu_stats: Optional[Dict[str, float]] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ):
        row = {
            "step": step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_margin": train_margin,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "eval_margin": eval_margin,
            "eval_pairs": eval_pairs,
            "eval_ece": eval_ece,
            "eval_roc_auc": eval_roc_auc,
        }
        if extra_metrics:
            for key, value in extra_metrics.items():
                if key not in self.fieldnames:
                    self.fieldnames.append(key)
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, sort_keys=True)
                else:
                    row[key] = value
        if gpu_stats:
            # Only include GPU fields that are in fieldnames
            for key in self.fieldnames:
                if key in gpu_stats:
                    row[key] = gpu_stats[key]
        self.rows.append(row)
        self._write()

    def _write(self):
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


async def async_main(args):
    """Async main entry point."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    set_seed(args.seed)

    # Load tokenizer and model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use sequence classification model with 1 output (scalar reward)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    # Set pad_token_id in model config to avoid batch size > 1 errors
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # Load preference data
    logger.info(f"Loading preferences from HF: {args.hf_dataset}")
    if args.diagnostic_mode:
        logger.info(f"DIAGNOSTIC MODE: {args.diagnostic_mode}")
    pref_data = load_hf_preferences(
        args.hf_dataset,
        args.hf_split,
        limit=args.limit,
        diagnostic_mode=args.diagnostic_mode,
        diagnostic_length_threshold=args.diagnostic_length_threshold,
        diagnostic_shuffle_rate=args.diagnostic_shuffle_rate,
        diagnostic_seed=args.diagnostic_seed,
    )

    logger.info(f"Loaded {len(pref_data)} preference pairs")

    # Convert to SAWItems
    if args.use_chat_template:
        logger.info("Using chat template for tokenization")
    else:
        logger.warning("Chat template DISABLED - training on raw text (not recommended)")

    saw_items = preference_dataset_to_saw_items(
        pref_data,
        tokenizer,
        max_length=args.max_seq_len,
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
    )
    logger.info(f"Created {len(saw_items)} SAWItems ({len(saw_items) // 2} pairs)")

    # Output directory (create early for sample logging)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log train distribution samples
    if args.log_samples > 0:
        log_distribution_samples(
            name="train",
            pref_data=pref_data,
            saw_items=saw_items,
            tokenizer=tokenizer,
            output_dir=output_dir,
            n_samples=args.log_samples,
            seed=args.seed,
        )

    # Load evaluation data if specified
    eval_items: Optional[List[SAWItem]] = None
    eval_pref_data: Optional[List[Dict[str, Any]]] = None
    if args.eval_split:
        logger.info(f"Loading evaluation data from split: {args.eval_split}")
        eval_pref_data = load_hf_preferences(
            args.hf_dataset, args.eval_split, limit=args.eval_limit
        )
        logger.info(f"Loaded {len(eval_pref_data)} evaluation preference pairs")
        eval_items = preference_dataset_to_saw_items(
            eval_pref_data,
            tokenizer,
            max_length=args.max_seq_len,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt,
        )
        logger.info(f"Created {len(eval_items)} evaluation SAWItems")

        # Log eval distribution samples
        if args.log_samples > 0:
            log_distribution_samples(
                name="eval",
                pref_data=eval_pref_data,
                saw_items=eval_items,
                tokenizer=tokenizer,
                output_dir=output_dir,
                n_samples=args.log_samples,
                seed=args.seed,
            )

    # Create algorithm with label smoothing
    algorithm = make_bradley_terry(
        beta=args.beta,
        score_type="reward",
        lambda_regularization=args.regularization_lambda,
        regularization_type=args.regularization_type,
        label_smoothing=args.label_smoothing,
        score_regularization_lambda=args.score_regularization_lambda,
    )

    # Create batch source (batch_size is number of items, so 2x pairs)
    batch_source = AsyncBatchSource(
        items=saw_items,
        batch_size=args.batch_size * 2,  # 2 items per pair
        shuffle=True,
        seed=args.seed,
    )

    # Create RMTrainerConfig
    config = RMTrainerConfig(
        model_device=device,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        micro_token_budget=args.micro_token_budget,
        pad_token_id=tokenizer.pad_token_id,
        log_every=args.log_every,
        profile_memory=args.profile_memory,
        sync_every_steps=0,
        # LR scheduler
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        # Early stopping
        early_stopping_patience=args.early_stopping_patience if args.early_stopping_patience > 0 else None,
        early_stopping_metric="eval_accuracy",
        early_stopping_mode="max",
        early_stopping_min_delta=args.early_stopping_min_delta,
        # RM-specific
        rm_pair_aware=True,
        rm_require_complete_pairs=True,
        rm_drop_incomplete_pairs=True,
        rm_calibration_bins=10,
        # Differential LR
        use_differential_lr=args.differential_lr,
        backbone_lr_mult=args.backbone_lr_mult,
        head_only_weight_decay=args.head_only_weight_decay,
    )

    # Setup LoRA config if requested
    lora_config = None
    if args.lora:
        try:
            from peft import LoraConfig

            # rsLoRA: scale alpha by sqrt(r) for rank-independent gradient magnitudes
            if args.rslora:
                effective_alpha = int(args.lora_alpha * (args.lora_rank ** 0.5))
                logger.info(f"rsLoRA: alpha={args.lora_alpha} * sqrt({args.lora_rank}) = {effective_alpha}")
            else:
                effective_alpha = args.lora_alpha

            # Parse modules_to_save if provided
            modules_to_save = None
            if args.lora_modules_to_save:
                modules_to_save = [m.strip() for m in args.lora_modules_to_save.split(",")]
                logger.info(f"LoRA modules to save (fully trainable): {modules_to_save}")

            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=effective_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
                modules_to_save=modules_to_save,
            )
            logger.info(f"LoRA enabled: rank={args.lora_rank}, alpha={effective_alpha} (base={args.lora_alpha}, rsLoRA={args.rslora})")
        except ImportError:
            raise ImportError("PEFT is required for LoRA. Install with: pip install peft")

    # Initialize metrics logger (output_dir already created earlier for sample logging)
    metrics_logger = MetricsLogger(output_dir / "metrics.csv", track_gpu=args.track_gpu)

    # Initialize GPU stats collector if requested
    gpu_collector: Optional[GPUStatsCollector] = None
    if args.track_gpu:
        gpu_collector = GPUStatsCollector(sample_interval_ms=args.gpu_sample_interval_ms)
        gpu_collector.start()
        logger.info(f"GPU tracking enabled (sampling every {args.gpu_sample_interval_ms}ms)")

    # Create trainer
    trainer = RMTrainer(
        model=model,
        algo=algorithm,
        batch_source=batch_source,
        cfg=config,
        train_logger=PrintLogger(),
        total_training_steps=args.steps,
        lora_config=lora_config,
    )

    # Track reward distributions over time
    reward_distributions: Dict[str, Any] = {
        "steps": [],
        "chosen_scores": [],
        "rejected_scores": [],
    }

    # Run baseline evaluation if eval set provided
    if eval_items:
        logger.info("Running baseline evaluation...")
        eval_results = evaluate(
            model=trainer.model,
            eval_items=eval_items,
            algorithm=algorithm,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=args.batch_size,
            step=0,
            split="baseline",
        )
        logger.info(
            f"Baseline | "
            f"Eval Loss: {eval_results['eval_loss']:.4f} | "
            f"Eval Acc: {eval_results['eval_accuracy']:.2%} | "
            f"Eval Margin: {eval_results['eval_margin']:.3f} | "
            f"Eval ECE: {eval_results['eval_ece']:.4f} | "
            f"Eval AUROC: {eval_results['eval_roc_auc']:.4f}"
        )
        metrics_logger.log(
            step=0,
            eval_loss=eval_results["eval_loss"],
            eval_acc=eval_results["eval_accuracy"],
            eval_margin=eval_results["eval_margin"],
            eval_pairs=eval_results["eval_pairs"],
            eval_ece=eval_results["eval_ece"],
            eval_roc_auc=eval_results["eval_roc_auc"],
            extra_metrics={
                "eval_delta_mean": eval_results["eval_delta_mean"],
                "eval_delta_std": eval_results["eval_delta_std"],
                "eval_delta_p05": eval_results["eval_delta_p05"],
                "eval_delta_p50": eval_results["eval_delta_p50"],
                "eval_delta_p95": eval_results["eval_delta_p95"],
                "eval_delta_neg_rate": eval_results["eval_delta_neg_rate"],
                "eval_reward_chosen_mean": eval_results["eval_reward_chosen_mean"],
                "eval_reward_chosen_std": eval_results["eval_reward_chosen_std"],
                "eval_reward_rejected_mean": eval_results["eval_reward_rejected_mean"],
                "eval_reward_rejected_std": eval_results["eval_reward_rejected_std"],
                "eval_reward_abs_max": eval_results["eval_reward_abs_max"],
                "eval_reward_len_corr": eval_results["eval_reward_len_corr"],
                "eval_delta_len_diff_corr": eval_results["eval_delta_len_diff_corr"],
                "eval_len_diff_mean": eval_results["eval_len_diff_mean"],
                "eval_len_diff_std": eval_results["eval_len_diff_std"],
                "eval_calib_bins_json": eval_results["eval_calib_bins"],
            },
        )
        logger.info("RM_DIAG %s", json.dumps({
            "step": 0,
            "split": "baseline",
            "loss": eval_results["eval_loss"],
            "accuracy": eval_results["eval_accuracy"],
            "margin": eval_results["eval_margin"],
            "ece": eval_results["eval_ece"],
            "auroc": eval_results["eval_roc_auc"],
            "delta_stats": {
                "mean": eval_results["eval_delta_mean"],
                "std": eval_results["eval_delta_std"],
                "p05": eval_results["eval_delta_p05"],
                "p50": eval_results["eval_delta_p50"],
                "p95": eval_results["eval_delta_p95"],
                "neg_rate": eval_results["eval_delta_neg_rate"],
            },
            "reward_stats": {
                "chosen_mean": eval_results["eval_reward_chosen_mean"],
                "chosen_std": eval_results["eval_reward_chosen_std"],
                "rejected_mean": eval_results["eval_reward_rejected_mean"],
                "rejected_std": eval_results["eval_reward_rejected_std"],
                "abs_max": eval_results["eval_reward_abs_max"],
            },
            "length_stats": {
                "reward_len_corr": eval_results["eval_reward_len_corr"],
                "delta_len_diff_corr": eval_results["eval_delta_len_diff_corr"],
                "len_diff_mean": eval_results["eval_len_diff_mean"],
                "len_diff_std": eval_results["eval_len_diff_std"],
            },
            "calibration_bins": eval_results["eval_calib_bins"],
        }, sort_keys=True))
        # Track reward distributions
        reward_distributions["steps"].append(0)
        reward_distributions["chosen_scores"].append(eval_results["chosen_scores"])
        reward_distributions["rejected_scores"].append(eval_results["rejected_scores"])

    # Early stopping tracking
    best_eval_acc = 0.0
    best_step = 0
    stopped_early = False

    # Training loop
    logger.info(f"Training for {args.steps} steps...")
    running_loss = 0.0
    running_acc = 0.0
    running_margin = 0.0

    for step in range(1, args.steps + 1):
        # Execute training step
        stats = await trainer.train_step()

        running_loss += stats.get("loss", 0.0)
        running_acc += stats.get("accuracy", 0.0)
        running_margin += stats.get("margin", 0.0)

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_acc = running_acc / args.log_every
            avg_margin = running_margin / args.log_every

            # Collect GPU stats if tracking
            gpu_stats = None
            if gpu_collector is not None:
                gpu_stats = gpu_collector.get_step_stats()
                if gpu_stats and "avg_gpu_utilization_pct" in gpu_stats:
                    logger.info(
                        f"GPU: {gpu_stats.get('avg_gpu_utilization_pct', 0):.1f}% util, "
                        f"{gpu_stats.get('avg_gpu_memory_used_gb', 0):.2f}GB used"
                    )

            metrics_logger.log(
                step=step,
                train_loss=avg_loss,
                train_acc=avg_acc,
                train_margin=avg_margin,
                gpu_stats=gpu_stats,
            )
            running_loss = 0.0
            running_acc = 0.0
            running_margin = 0.0

        # Periodic evaluation
        if eval_items and step % args.eval_every == 0:
            eval_results = evaluate(
                model=trainer.model,
                eval_items=eval_items,
                algorithm=algorithm,
                device=device,
                pad_token_id=tokenizer.pad_token_id,
                batch_size=args.batch_size,
                step=step,
                split="eval",
            )
            eval_acc = eval_results["eval_accuracy"]
            logger.info(
                f"Step {step} Eval | "
                f"Loss: {eval_results['eval_loss']:.4f} | "
                f"Acc: {eval_acc:.2%} | "
                f"Margin: {eval_results['eval_margin']:.3f} | "
                f"ECE: {eval_results['eval_ece']:.4f} | "
                f"AUROC: {eval_results['eval_roc_auc']:.4f}"
            )
            metrics_logger.log(
                step=step,
                eval_loss=eval_results["eval_loss"],
                eval_acc=eval_acc,
                eval_margin=eval_results["eval_margin"],
                eval_pairs=eval_results["eval_pairs"],
                eval_ece=eval_results["eval_ece"],
                eval_roc_auc=eval_results["eval_roc_auc"],
                extra_metrics={
                    "eval_delta_mean": eval_results["eval_delta_mean"],
                    "eval_delta_std": eval_results["eval_delta_std"],
                    "eval_delta_p05": eval_results["eval_delta_p05"],
                    "eval_delta_p50": eval_results["eval_delta_p50"],
                    "eval_delta_p95": eval_results["eval_delta_p95"],
                    "eval_delta_neg_rate": eval_results["eval_delta_neg_rate"],
                    "eval_reward_chosen_mean": eval_results["eval_reward_chosen_mean"],
                    "eval_reward_chosen_std": eval_results["eval_reward_chosen_std"],
                    "eval_reward_rejected_mean": eval_results["eval_reward_rejected_mean"],
                    "eval_reward_rejected_std": eval_results["eval_reward_rejected_std"],
                    "eval_reward_abs_max": eval_results["eval_reward_abs_max"],
                    "eval_reward_len_corr": eval_results["eval_reward_len_corr"],
                    "eval_delta_len_diff_corr": eval_results["eval_delta_len_diff_corr"],
                    "eval_len_diff_mean": eval_results["eval_len_diff_mean"],
                    "eval_len_diff_std": eval_results["eval_len_diff_std"],
                    "eval_calib_bins_json": eval_results["eval_calib_bins"],
                },
            )
            logger.info("RM_DIAG %s", json.dumps({
                "step": step,
                "split": "eval",
                "loss": eval_results["eval_loss"],
                "accuracy": eval_results["eval_accuracy"],
                "margin": eval_results["eval_margin"],
                "ece": eval_results["eval_ece"],
                "auroc": eval_results["eval_roc_auc"],
                "delta_stats": {
                    "mean": eval_results["eval_delta_mean"],
                    "std": eval_results["eval_delta_std"],
                    "p05": eval_results["eval_delta_p05"],
                    "p50": eval_results["eval_delta_p50"],
                    "p95": eval_results["eval_delta_p95"],
                    "neg_rate": eval_results["eval_delta_neg_rate"],
                },
                "reward_stats": {
                    "chosen_mean": eval_results["eval_reward_chosen_mean"],
                    "chosen_std": eval_results["eval_reward_chosen_std"],
                    "rejected_mean": eval_results["eval_reward_rejected_mean"],
                    "rejected_std": eval_results["eval_reward_rejected_std"],
                    "abs_max": eval_results["eval_reward_abs_max"],
                },
                "length_stats": {
                    "reward_len_corr": eval_results["eval_reward_len_corr"],
                    "delta_len_diff_corr": eval_results["eval_delta_len_diff_corr"],
                    "len_diff_mean": eval_results["eval_len_diff_mean"],
                    "len_diff_std": eval_results["eval_len_diff_std"],
                },
                "calibration_bins": eval_results["eval_calib_bins"],
            }, sort_keys=True))
            # Track reward distributions
            reward_distributions["steps"].append(step)
            reward_distributions["chosen_scores"].append(eval_results["chosen_scores"])
            reward_distributions["rejected_scores"].append(eval_results["rejected_scores"])

            # Check early stopping via trainer's mechanism
            should_stop, is_improvement = trainer._check_early_stopping(eval_acc)
            if should_stop:
                logger.info(
                    f"Early stopping at step {step}: "
                    f"Best acc: {trainer._early_stopping_best_metric:.2%} at step {trainer._early_stopping_best_step}"
                )
                stopped_early = True
                # Save best model
                best_path = output_dir / "best"
                logger.info(f"Saving best model to {best_path}")
                trainer.model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                break

            # Save best model manually if improved
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_step = step
                best_path = output_dir / "best"
                logger.info(f"New best eval acc: {eval_acc:.2%} - saving to {best_path}")
                trainer.model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)

        if step % args.save_every == 0:
            ckpt_path = output_dir / f"step_{step}"
            logger.info(f"Saving checkpoint to {ckpt_path}")
            trainer.model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

    # Final save
    final_step = step if stopped_early else args.steps
    final_path = output_dir / f"step_{final_step}"

    logger.info(f"Saving final checkpoint to {final_path}")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Final evaluation (if not already done at this step)
    if eval_items and (stopped_early or final_step % args.eval_every != 0):
        logger.info("Running final evaluation...")
        eval_results = evaluate(
            model=trainer.model,
            eval_items=eval_items,
            algorithm=algorithm,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=args.batch_size,
            step=final_step,
            split="final",
        )
        logger.info(
            f"Final | "
            f"Eval Loss: {eval_results['eval_loss']:.4f} | "
            f"Eval Acc: {eval_results['eval_accuracy']:.2%} | "
            f"Eval Margin: {eval_results['eval_margin']:.3f} | "
            f"Eval ECE: {eval_results['eval_ece']:.4f} | "
            f"Eval AUROC: {eval_results['eval_roc_auc']:.4f}"
        )
        metrics_logger.log(
            step=final_step,
            eval_loss=eval_results["eval_loss"],
            eval_acc=eval_results["eval_accuracy"],
            eval_margin=eval_results["eval_margin"],
            eval_pairs=eval_results["eval_pairs"],
            eval_ece=eval_results["eval_ece"],
            eval_roc_auc=eval_results["eval_roc_auc"],
            extra_metrics={
                "eval_delta_mean": eval_results["eval_delta_mean"],
                "eval_delta_std": eval_results["eval_delta_std"],
                "eval_delta_p05": eval_results["eval_delta_p05"],
                "eval_delta_p50": eval_results["eval_delta_p50"],
                "eval_delta_p95": eval_results["eval_delta_p95"],
                "eval_delta_neg_rate": eval_results["eval_delta_neg_rate"],
                "eval_reward_chosen_mean": eval_results["eval_reward_chosen_mean"],
                "eval_reward_chosen_std": eval_results["eval_reward_chosen_std"],
                "eval_reward_rejected_mean": eval_results["eval_reward_rejected_mean"],
                "eval_reward_rejected_std": eval_results["eval_reward_rejected_std"],
                "eval_reward_abs_max": eval_results["eval_reward_abs_max"],
                "eval_reward_len_corr": eval_results["eval_reward_len_corr"],
                "eval_delta_len_diff_corr": eval_results["eval_delta_len_diff_corr"],
                "eval_len_diff_mean": eval_results["eval_len_diff_mean"],
                "eval_len_diff_std": eval_results["eval_len_diff_std"],
                "eval_calib_bins_json": eval_results["eval_calib_bins"],
            },
        )
        logger.info("RM_DIAG %s", json.dumps({
            "step": final_step,
            "split": "final",
            "loss": eval_results["eval_loss"],
            "accuracy": eval_results["eval_accuracy"],
            "margin": eval_results["eval_margin"],
            "ece": eval_results["eval_ece"],
            "auroc": eval_results["eval_roc_auc"],
            "delta_stats": {
                "mean": eval_results["eval_delta_mean"],
                "std": eval_results["eval_delta_std"],
                "p05": eval_results["eval_delta_p05"],
                "p50": eval_results["eval_delta_p50"],
                "p95": eval_results["eval_delta_p95"],
                "neg_rate": eval_results["eval_delta_neg_rate"],
            },
            "reward_stats": {
                "chosen_mean": eval_results["eval_reward_chosen_mean"],
                "chosen_std": eval_results["eval_reward_chosen_std"],
                "rejected_mean": eval_results["eval_reward_rejected_mean"],
                "rejected_std": eval_results["eval_reward_rejected_std"],
                "abs_max": eval_results["eval_reward_abs_max"],
            },
            "length_stats": {
                "reward_len_corr": eval_results["eval_reward_len_corr"],
                "delta_len_diff_corr": eval_results["eval_delta_len_diff_corr"],
                "len_diff_mean": eval_results["eval_len_diff_mean"],
                "len_diff_std": eval_results["eval_len_diff_std"],
            },
            "calibration_bins": eval_results["eval_calib_bins"],
        }, sort_keys=True))
        # Track reward distributions
        if final_step not in reward_distributions["steps"]:
            reward_distributions["steps"].append(final_step)
            reward_distributions["chosen_scores"].append(eval_results["chosen_scores"])
            reward_distributions["rejected_scores"].append(eval_results["rejected_scores"])

    # Report best model
    if eval_items and best_step > 0:
        logger.info(f"Best model: step {best_step} with eval acc {best_eval_acc:.2%}")

    # Save reward distributions
    reward_dist_path = output_dir / "reward_distributions.json"
    with open(reward_dist_path, "w") as f:
        json.dump(reward_distributions, f, indent=2)
    logger.info(f"Saved reward distributions to {reward_dist_path}")

    # Stop GPU tracking and log summary
    if gpu_collector is not None:
        gpu_collector.stop()
        logger.info("GPU tracking stopped")

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train reward model with Bradley-Terry using RMTrainer"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B", help="Base model for RM"
    )
    parser.add_argument("--hf-dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--hf-split", default="train", help="HF dataset split")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of training samples"
    )
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (pairs)")
    parser.add_argument(
        "--micro-token-budget", type=int, default=8192, help="Micro-batch token budget"
    )
    parser.add_argument(
        "--regularization-lambda", type=float, default=0.0, help="Regularization lambda"
    )
    parser.add_argument(
        "--regularization-type", type=str, default="l2", help="Regularization type"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--max-seq-len", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Bradley-Terry temperature"
    )
    parser.add_argument(
        "--output-dir", default="checkpoints_rm", help="Output directory"
    )
    parser.add_argument(
        "--save-every", type=int, default=50, help="Save checkpoint every N steps"
    )
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--profile-memory", action="store_true", help="Profile GPU memory"
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval-split", type=str, default=None,
        help="HF dataset split for evaluation (e.g., 'test')"
    )
    parser.add_argument(
        "--eval-limit", type=int, default=None,
        help="Limit number of evaluation samples"
    )
    parser.add_argument(
        "--eval-every", type=int, default=50,
        help="Run evaluation every N steps (requires --eval-split)"
    )

    # LR Scheduler arguments
    parser.add_argument(
        "--lr-scheduler", type=str, default=None,
        choices=["cosine", "linear", "constant_with_warmup"],
        help="LR scheduler type"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=0,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=None,
        help="Warmup as fraction of total steps (alternative to --warmup-steps)"
    )

    # Early stopping arguments
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Stop if eval metric doesn't improve for N evals (0=disabled)"
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=0.0,
        help="Minimum improvement to reset patience"
    )

    # Label smoothing
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Label smoothing factor (0.0 to 1.0)"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora", action="store_true",
        help="Enable LoRA training"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8,
        help="LoRA rank (r)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=16,
        help="LoRA alpha scaling factor"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.0,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--lora-target-modules", type=str, default=None,
        help="Comma-separated list of LoRA target modules (e.g., 'q_proj,v_proj')"
    )
    parser.add_argument(
        "--rslora", action="store_true",
        help="Use rsLoRA scaling (alpha = alpha * sqrt(r)) for rank-independent gradients"
    )
    parser.add_argument(
        "--lora-modules-to-save", type=str, default=None,
        help="Comma-separated modules to keep fully trainable (e.g., 'input_layernorm,post_attention_layernorm')"
    )

    # Sample logging
    parser.add_argument(
        "--log-samples", type=int, default=5,
        help="Number of random samples to log from train/eval distributions (0 to disable)"
    )

    # GPU tracking arguments
    parser.add_argument(
        "--track-gpu", action="store_true",
        help="Track GPU utilization during training (requires pynvml for detailed stats)"
    )
    parser.add_argument(
        "--gpu-sample-interval-ms", type=int, default=100,
        help="GPU stats sampling interval in milliseconds (default: 100)"
    )

    # Differential LR arguments
    parser.add_argument(
        "--differential-lr", action="store_true",
        help="Use different LR for backbone vs head (10x lower for backbone by default)"
    )
    parser.add_argument(
        "--backbone-lr-mult", type=float, default=0.1,
        help="Backbone LR multiplier when using --differential-lr (default: 0.1)"
    )
    parser.add_argument(
        "--head-only-weight-decay", action="store_true",
        help="Only apply weight decay to head parameters (not backbone)"
    )

    # Chat template arguments
    parser.add_argument(
        "--use-chat-template", action="store_true", default=True,
        help="Apply the model's chat template to format preference data (default: True)"
    )
    parser.add_argument(
        "--no-chat-template", action="store_true",
        help="Disable chat template (use raw text concatenation - NOT recommended)"
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="System prompt to prepend when using chat template"
    )

    # Diagnostic mode arguments (V5 experiments)
    parser.add_argument(
        "--diagnostic-mode", type=str, default=None,
        choices=["label_shuffle", "prompt_only", "random_negatives", "length_matched"],
        help="Diagnostic mode for testing pipeline integrity"
    )
    parser.add_argument(
        "--diagnostic-length-threshold", type=int, default=50,
        help="Max |len(chosen)-len(rejected)| for length_matched mode (default: 50)"
    )
    parser.add_argument(
        "--diagnostic-shuffle-rate", type=float, default=0.5,
        help="Label shuffle rate for label_shuffle mode (default: 0.5)"
    )
    parser.add_argument(
        "--diagnostic-seed", type=int, default=42,
        help="Seed for diagnostic randomization (default: 42)"
    )
    parser.add_argument(
        "--score-regularization-lambda", type=float, default=0.0,
        help="L2 regularization on individual reward scores (not margin)"
    )

    args = parser.parse_args()

    # Handle --no-chat-template flag
    if args.no_chat_template:
        args.use_chat_template = False

    if not args.hf_dataset:
        parser.error("Provide --hf-dataset")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

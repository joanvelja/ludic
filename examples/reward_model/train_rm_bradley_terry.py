"""Train a reward model using Bradley-Terry loss on preference data.

This script demonstrates training a reward model from pairwise preferences.
The trained RM can then be used to score rollouts during policy training.

Usage:
    # Train on local JSONL preference file
    python train_rm_bradley_terry.py \
        --model Qwen/Qwen2.5-0.5B \
        --dataset data/preferences.jsonl \
        --steps 100

    # Train on HuggingFace dataset
    python train_rm_bradley_terry.py \
        --model Qwen/Qwen2.5-0.5B \
        --hf-dataset stanfordnlp/SHP \
        --hf-split train \
        --steps 100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ludic.training.algorithm import make_bradley_terry
from ludic.training.preference_utils import (
    create_preference_saw_items,
    preference_dataset_to_saw_items,
    shuffle_preference_pairs,
)
from ludic.training.types import SAWBatch, SAWItem
from ludic.training.batching.micro_batching import collate_saw_items

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl_preferences(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load preference data from JSONL file.

    Expected format per line:
        {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if limit is not None and len(data) >= limit:
                break
    return data


def load_hf_preferences(
    dataset_name: str,
    split: str = "train",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load preference data from HuggingFace datasets.

    Supports common preference dataset formats.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset(dataset_name, split=split)
    data: List[Dict[str, Any]] = []

    for row in ds:
        # Handle different dataset formats
        if "prompt" in row and "chosen" in row and "rejected" in row:
            # Standard format
            data.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })
        elif "history" in row and "human_ref_A" in row:
            # SHP format
            prompt = row.get("history", "")
            if row.get("labels", 1) == 1:
                chosen, rejected = row["human_ref_A"], row["human_ref_B"]
            else:
                chosen, rejected = row["human_ref_B"], row["human_ref_A"]
            data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
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
            data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        if limit is not None and len(data) >= limit:
            break

    return data


class SimpleBatchSource:
    """Simple batch source for preference training."""

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

    def __iter__(self) -> Iterator[SAWBatch]:
        return self

    def __next__(self) -> SAWBatch:
        if self._idx >= len(self.items):
            self._epoch += 1
            self._idx = 0
            if self.shuffle:
                self.items = shuffle_preference_pairs(
                    self.items, seed=self.seed + self._epoch
                )

        end_idx = min(self._idx + self.batch_size, len(self.items))
        batch_items = self.items[self._idx:end_idx]
        self._idx = end_idx

        return SAWBatch(
            items=batch_items,
            meta={"num_samples": len(batch_items), "epoch": self._epoch},
        )


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: SAWBatch,
    algorithm,
    device: str,
    pad_token_id: int,
    max_seq_len: int = 512,
) -> Dict[str, float]:
    """Execute one training step."""
    model.train()

    # Collate items to tensors
    collated = collate_saw_items(
        batch.items,
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
    )

    # Move to device
    input_ids = collated["input_ids"].to(device)
    attention_mask = collated["attention_mask"].to(device)

    # Forward pass - for sequence classification, we get [B, num_labels]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, 1] for reward model

    # Compute Bradley-Terry loss
    # The batch needs metadata for pairing
    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "action_mask": collated["action_mask"].to(device),
        "meta": {
            "pair_id": [it.meta["pair_id"] for it in batch.items],
            "role": [it.meta["role"] for it in batch.items],
            "label": [it.meta["label"] for it in batch.items],
        },
    }

    loss, stats = algorithm.loss.compute(logits, batch_dict)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Convert stats to float
    return {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser(description="Train reward model with Bradley-Terry")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Base model for RM")
    parser.add_argument("--dataset", type=str, help="Path to JSONL preference file")
    parser.add_argument("--hf-dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--hf-split", default="train", help="HF dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (pairs)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--beta", type=float, default=1.0, help="Bradley-Terry temperature")
    parser.add_argument("--output-dir", default="checkpoints_rm", help="Output directory")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    args = parser.parse_args()

    if not args.dataset and not args.hf_dataset:
        parser.error("Provide either --dataset or --hf-dataset")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

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
    model.to(device)

    # Load preference data
    if args.dataset:
        logger.info(f"Loading preferences from: {args.dataset}")
        pref_data = load_jsonl_preferences(Path(args.dataset), limit=args.limit)
    else:
        logger.info(f"Loading preferences from HF: {args.hf_dataset}")
        pref_data = load_hf_preferences(args.hf_dataset, args.hf_split, limit=args.limit)

    logger.info(f"Loaded {len(pref_data)} preference pairs")

    # Convert to SAWItems
    saw_items = preference_dataset_to_saw_items(
        pref_data,
        tokenizer,
        max_length=args.max_seq_len,
    )
    logger.info(f"Created {len(saw_items)} SAWItems ({len(saw_items) // 2} pairs)")

    # Create algorithm
    algorithm = make_bradley_terry(beta=args.beta, score_type="reward")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Create batch source (batch_size is number of items, so 2x pairs)
    batch_source = SimpleBatchSource(
        items=saw_items,
        batch_size=args.batch_size * 2,  # 2 items per pair
        shuffle=True,
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info(f"Training for {args.steps} steps...")
    running_loss = 0.0
    running_acc = 0.0

    for step in range(1, args.steps + 1):
        batch = next(batch_source)

        stats = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            algorithm=algorithm,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            max_seq_len=args.max_seq_len,
        )

        running_loss += stats.get("loss", 0.0)
        running_acc += stats.get("accuracy", 0.0)

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_acc = running_acc / args.log_every
            margin = stats.get("margin", 0.0)
            logger.info(
                f"Step {step}/{args.steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {avg_acc:.2%} | "
                f"Margin: {margin:.3f}"
            )
            running_loss = 0.0
            running_acc = 0.0

        if step % args.save_every == 0:
            ckpt_path = output_dir / f"step_{step}"
            logger.info(f"Saving checkpoint to {ckpt_path}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

    # Final save
    final_path = output_dir / f"step_{args.steps}"
    logger.info(f"Saving final checkpoint to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

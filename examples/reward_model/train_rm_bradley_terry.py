"""Train a reward model using Bradley-Terry loss on preference data.

This script demonstrates training a reward model from pairwise preferences.
The trained RM can then be used to score rollouts during policy training.

Usage:
    # Train on HuggingFace dataset
    uv run python train_rm_bradley_terry.py \
        --model Qwen/Qwen2.5-0.5B \
        --hf-dataset stanfordnlp/SHP \
        --hf-split train \
        --steps 100

    # With evaluation
    uv run python train_rm_bradley_terry.py \
        --model Qwen/Qwen2.5-0.5B \
        --hf-dataset stanfordnlp/SHP \
        --hf-split train \
        --eval-split test \
        --eval-limit 500 \
        --steps 500
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ludic.training.algorithm import make_bradley_terry
from ludic.training.preference_utils import (
    preference_dataset_to_saw_items,
    shuffle_preference_pairs,
)
from ludic.training.types import SAWBatch, SAWItem
from ludic.training.batching.micro_batching import collate_saw_items

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            data.append(
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
            )
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
        batch_items = self.items[self._idx : end_idx]
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
) -> Dict[str, float]:
    """Execute one training step."""
    model.train()

    # Collate items to tensors
    collated = collate_saw_items(
        batch.items,
        pad_token_id=pad_token_id,
        device=torch.device(device),
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_items: List[SAWItem],
    algorithm,
    device: str,
    pad_token_id: int,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Evaluate model on a held-out set.

    Returns metrics and reward distribution data.
    """
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_margin = 0.0
    total_pairs = 0

    chosen_scores: List[float] = []
    rejected_scores: List[float] = []

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

            # Track reward distributions
            chosen_scores.append(float(stats.get("chosen_score", 0)))
            rejected_scores.append(float(stats.get("rejected_score", 0)))

    if total_pairs == 0:
        return {
            "eval_loss": 0.0,
            "eval_accuracy": 0.0,
            "eval_margin": 0.0,
            "eval_pairs": 0,
            "chosen_scores": [],
            "rejected_scores": [],
        }

    return {
        "eval_loss": total_loss / total_pairs,
        "eval_accuracy": total_accuracy / total_pairs,
        "eval_margin": total_margin / total_pairs,
        "eval_pairs": total_pairs,
        "chosen_scores": chosen_scores,
        "rejected_scores": rejected_scores,
    }


class MetricsLogger:
    """Log metrics to CSV file."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.rows: List[Dict[str, Any]] = []
        self.fieldnames = [
            "step", "train_loss", "train_acc", "train_margin",
            "eval_loss", "eval_acc", "eval_margin", "eval_pairs"
        ]

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
        }
        self.rows.append(row)
        self._write()

    def _write(self):
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


def main():
    parser = argparse.ArgumentParser(
        description="Train reward model with Bradley-Terry"
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
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (pairs)")
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
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Stop if eval accuracy doesn't improve for N evals (0=disabled)"
    )

    args = parser.parse_args()

    if not args.hf_dataset:
        parser.error("Provide --hf-dataset")

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
    # Set pad_token_id in model config to avoid batch size > 1 errors
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # Load preference data
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

    # Load evaluation data if specified
    eval_items: Optional[List[SAWItem]] = None
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
        )
        logger.info(f"Created {len(eval_items)} evaluation SAWItems")

    # Create algorithm
    algorithm = make_bradley_terry(
        beta=args.beta,
        score_type="reward",
        lambda_regularization=args.regularization_lambda,
        regularization_type=args.regularization_type,
    )

    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Create batch source (batch_size is number of items, so 2x pairs)
    batch_source = SimpleBatchSource(
        items=saw_items,
        batch_size=args.batch_size * 2,  # 2 items per pair
        shuffle=True,
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metrics logger
    metrics_logger = MetricsLogger(output_dir / "metrics.csv")

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
            model=model,
            eval_items=eval_items,
            algorithm=algorithm,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=args.batch_size,
        )
        logger.info(
            f"Baseline | "
            f"Eval Loss: {eval_results['eval_loss']:.4f} | "
            f"Eval Acc: {eval_results['eval_accuracy']:.2%} | "
            f"Eval Margin: {eval_results['eval_margin']:.3f}"
        )
        metrics_logger.log(
            step=0,
            eval_loss=eval_results["eval_loss"],
            eval_acc=eval_results["eval_accuracy"],
            eval_margin=eval_results["eval_margin"],
            eval_pairs=eval_results["eval_pairs"],
        )
        # Track reward distributions
        reward_distributions["steps"].append(0)
        reward_distributions["chosen_scores"].append(eval_results["chosen_scores"])
        reward_distributions["rejected_scores"].append(eval_results["rejected_scores"])

    # Early stopping tracking
    best_eval_acc = 0.0
    best_step = 0
    patience_counter = 0
    stopped_early = False

    # Training loop
    logger.info(f"Training for {args.steps} steps...")
    running_loss = 0.0
    running_acc = 0.0
    running_margin = 0.0

    for step in range(1, args.steps + 1):
        batch = next(batch_source)

        stats = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            algorithm=algorithm,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
        )

        running_loss += stats.get("loss", 0.0)
        running_acc += stats.get("accuracy", 0.0)
        running_margin += stats.get("margin", 0.0)

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_acc = running_acc / args.log_every
            avg_margin = running_margin / args.log_every
            logger.info(
                f"Step {step}/{args.steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {avg_acc:.2%} | "
                f"Margin: {avg_margin:.3f}"
            )
            metrics_logger.log(
                step=step,
                train_loss=avg_loss,
                train_acc=avg_acc,
                train_margin=avg_margin,
            )
            running_loss = 0.0
            running_acc = 0.0
            running_margin = 0.0

        # Periodic evaluation
        if eval_items and step % args.eval_every == 0:
            eval_results = evaluate(
                model=model,
                eval_items=eval_items,
                algorithm=algorithm,
                device=device,
                pad_token_id=tokenizer.pad_token_id,
                batch_size=args.batch_size,
            )
            eval_acc = eval_results["eval_accuracy"]
            logger.info(
                f"Step {step} Eval | "
                f"Loss: {eval_results['eval_loss']:.4f} | "
                f"Acc: {eval_acc:.2%} | "
                f"Margin: {eval_results['eval_margin']:.3f}"
            )
            metrics_logger.log(
                step=step,
                eval_loss=eval_results["eval_loss"],
                eval_acc=eval_acc,
                eval_margin=eval_results["eval_margin"],
                eval_pairs=eval_results["eval_pairs"],
            )
            # Track reward distributions
            reward_distributions["steps"].append(step)
            reward_distributions["chosen_scores"].append(eval_results["chosen_scores"])
            reward_distributions["rejected_scores"].append(eval_results["rejected_scores"])

            # Save best model
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_step = step
                patience_counter = 0
                best_path = output_dir / "best"
                logger.info(f"New best eval acc: {eval_acc:.2%} - saving to {best_path}")
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
            else:
                patience_counter += 1

            # Early stopping check
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logger.info(
                    f"Early stopping at step {step}: no improvement for "
                    f"{args.early_stopping_patience} evaluations. "
                    f"Best acc: {best_eval_acc:.2%} at step {best_step}"
                )
                stopped_early = True
                break

        if step % args.save_every == 0:
            ckpt_path = output_dir / f"step_{step}"
            logger.info(f"Saving checkpoint to {ckpt_path}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

    # Final save
    final_step = step if stopped_early else args.steps
    final_path = output_dir / f"step_{final_step}"

    logger.info(f"Saving final checkpoint to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Final evaluation (if not already done at this step)
    if eval_items and (stopped_early or final_step % args.eval_every != 0):
        logger.info("Running final evaluation...")
        eval_results = evaluate(
            model=model,
            eval_items=eval_items,
            algorithm=algorithm,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=args.batch_size,
        )
        logger.info(
            f"Final | "
            f"Eval Loss: {eval_results['eval_loss']:.4f} | "
            f"Eval Acc: {eval_results['eval_accuracy']:.2%} | "
            f"Eval Margin: {eval_results['eval_margin']:.3f}"
        )
        metrics_logger.log(
            step=final_step,
            eval_loss=eval_results["eval_loss"],
            eval_acc=eval_results["eval_accuracy"],
            eval_margin=eval_results["eval_margin"],
            eval_pairs=eval_results["eval_pairs"],
        )
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

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

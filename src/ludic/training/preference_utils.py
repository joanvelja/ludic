"""Utilities for creating preference-based training samples.

Preferences are represented as paired SAWItems with metadata linking them.
This module provides functions to create, convert, and shuffle preference pairs
while maintaining the Ludic convention of using SAWItem as the universal
training sample format.

Metadata schema for preference SAWItems:
- pair_id: Unique identifier linking chosen/rejected pair
- role: "chosen" or "rejected"
- label: Preference strength (1.0 = chosen preferred, 0.0 = rejected preferred)
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable
from uuid import uuid4

from ludic.training.types import SAWItem


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers that can encode text to token IDs."""

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token IDs."""
        ...


def create_preference_saw_items(
    prompt: str,
    chosen: str,
    rejected: str,
    tokenizer: Tokenizer,
    *,
    label: float = 1.0,
    pair_id: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Tuple[SAWItem, SAWItem]:
    """Create paired SAWItems from a preference comparison.

    Tokenizes prompt + completion for both chosen and rejected sequences,
    creating SAWItems with appropriate action masks (0 for prompt, 1 for completion).

    Args:
        prompt: Shared prompt/context text.
        chosen: Preferred completion text.
        rejected: Dispreferred completion text.
        tokenizer: Tokenizer with encode() method for text → token IDs.
        label: Preference strength. 1.0 = chosen preferred, 0.0 = rejected
            preferred, 0.5 = tie. Defaults to 1.0.
        pair_id: Unique identifier linking this pair. Auto-generated using
            uuid4().hex if None.
        max_length: Optional maximum sequence length. If provided, sequences
            are truncated to this length.

    Returns:
        Tuple of (chosen_item, rejected_item) linked by pair_id in metadata.

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> chosen_item, rejected_item = create_preference_saw_items(
        ...     prompt="What is 2+2?",
        ...     chosen=" 4",
        ...     rejected=" 5",
        ...     tokenizer=tokenizer,
        ... )
        >>> chosen_item.meta["role"]
        'chosen'
        >>> rejected_item.meta["pair_id"] == chosen_item.meta["pair_id"]
        True
    """
    if pair_id is None:
        pair_id = uuid4().hex

    def make_item(completion: str, role: str) -> SAWItem:
        # Tokenize prompt to get prompt length
        prompt_tokens = tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)

        # Tokenize full sequence
        full_text = prompt + completion
        tokens = tokenizer.encode(full_text)

        # Apply max_length truncation if specified
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        seq_len = len(tokens)

        # Build action mask: 0 for prompt tokens, 1 for completion tokens
        # Clamp prompt_len to seq_len in case truncation cut into prompt
        effective_prompt_len = min(prompt_len, seq_len)
        num_completion_tokens = seq_len - effective_prompt_len

        # Validate that we have at least one completion token to train on
        if num_completion_tokens == 0:
            raise ValueError(
                f"Truncation eliminated all completion tokens for {role} response. "
                f"prompt_len={prompt_len}, seq_len={seq_len}, max_length={max_length}. "
                f"Increase max_length or use shorter prompts."
            )

        action_mask = [0] * effective_prompt_len + [1] * num_completion_tokens

        return SAWItem(
            input_ids=tokens,
            attention_mask=[1] * seq_len,
            action_mask=action_mask,
            weight=1.0,
            meta={
                "pair_id": pair_id,
                "role": role,
                "label": label,
            },
        )

    chosen_item = make_item(chosen, "chosen")
    rejected_item = make_item(rejected, "rejected")

    return chosen_item, rejected_item


def preference_dataset_to_saw_items(
    dataset: Iterable[Dict[str, Any]],
    tokenizer: Tokenizer,
    *,
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
    label_col: Optional[str] = None,
    max_length: Optional[int] = None,
) -> List[SAWItem]:
    """Convert a preference dataset to a list of SAWItems.

    Items are returned interleaved as pairs: [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    This layout enables efficient iteration over pairs.

    Args:
        dataset: Iterable of dicts with prompt/chosen/rejected columns.
            Works with HuggingFace datasets or plain lists of dicts.
        tokenizer: Tokenizer with encode() method for text → token IDs.
        prompt_col: Column name for prompt/context. Defaults to "prompt".
        chosen_col: Column name for preferred completion. Defaults to "chosen".
        rejected_col: Column name for dispreferred completion. Defaults to "rejected".
        label_col: Optional column name for soft preference labels.
            If None, defaults to 1.0 (chosen always preferred).
        max_length: Optional maximum sequence length for truncation.

    Returns:
        List of SAWItems with interleaved chosen/rejected pairs.
        Length is 2 * number of rows in dataset.

    Example:
        >>> dataset = [
        ...     {"prompt": "Q: 2+2?", "chosen": " 4", "rejected": " 5"},
        ...     {"prompt": "Q: 3+3?", "chosen": " 6", "rejected": " 7"},
        ... ]
        >>> items = preference_dataset_to_saw_items(dataset, tokenizer)
        >>> len(items)
        4
        >>> items[0].meta["pair_id"]
        'pref_0'
        >>> items[1].meta["pair_id"]
        'pref_0'
    """
    items: List[SAWItem] = []

    for i, row in enumerate(dataset):
        # Extract label from column if specified, else default to 1.0
        if label_col is not None:
            label = float(row.get(label_col, 1.0))
        else:
            label = 1.0

        chosen_item, rejected_item = create_preference_saw_items(
            prompt=row[prompt_col],
            chosen=row[chosen_col],
            rejected=row[rejected_col],
            tokenizer=tokenizer,
            label=label,
            pair_id=f"pref_{i}",
            max_length=max_length,
        )
        items.append(chosen_item)
        items.append(rejected_item)

    return items


def shuffle_preference_pairs(
    items: List[SAWItem],
    seed: Optional[int] = None,
) -> List[SAWItem]:
    """Shuffle SAWItems while keeping chosen/rejected pairs adjacent.

    Groups items by pair_id, shuffles the groups, then flattens back to a list.
    Within each group, the original order is preserved (typically chosen, rejected).

    Args:
        items: List of SAWItems with pair_id in metadata.
        seed: Optional random seed for reproducibility.

    Returns:
        Shuffled list with pairs preserved and adjacent.

    Raises:
        KeyError: If any item is missing pair_id in metadata.

    Example:
        >>> # Original: [c0, r0, c1, r1, c2, r2]
        >>> shuffled = shuffle_preference_pairs(items, seed=42)
        >>> # Might get: [c2, r2, c0, r0, c1, r1]
        >>> # Pairs stay together, just in different order
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Group items by pair_id, preserving order within each group
    pairs: Dict[str, List[SAWItem]] = {}
    for item in items:
        pid = item.meta["pair_id"]
        if pid not in pairs:
            pairs[pid] = []
        pairs[pid].append(item)

    # Convert to list of groups and shuffle
    pair_list = list(pairs.values())
    rng.shuffle(pair_list)

    # Flatten back to single list
    return [item for pair in pair_list for item in pair]

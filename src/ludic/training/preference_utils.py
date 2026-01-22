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

import logging
import random
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable
from uuid import uuid4

from ludic.training.chat_template_utils import tokenize_with_chat_template
from ludic.training.types import SAWItem

logger = logging.getLogger(__name__)


def _shorten_text(text: str, limit: int = 200) -> str:
    safe = text.replace("\n", "\\n")
    if not safe:
        return ""
    return textwrap.shorten(safe, width=limit, placeholder="...")


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers that can encode text to token IDs."""

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token IDs."""
        ...


@runtime_checkable
class ChatTokenizer(Protocol):
    """Protocol for tokenizers that support chat template formatting."""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = True,
    ) -> List[int]:
        """Apply chat template and tokenize a conversation."""
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
    use_chat_template: bool = False,
    system_prompt: Optional[str] = None,
) -> Tuple[SAWItem, SAWItem]:
    """Create paired SAWItems from a preference comparison.

    Tokenizes prompt + completion for both chosen and rejected sequences,
    creating SAWItems with appropriate action masks (0 for prompt, 1 for completion).

    Args:
        prompt: Shared prompt/context text.
        chosen: Preferred completion text.
        rejected: Dispreferred completion text.
        tokenizer: Tokenizer with encode() method for text → token IDs.
            If use_chat_template=True, must also have apply_chat_template().
        label: Preference strength. 1.0 = chosen preferred, 0.0 = rejected
            preferred, 0.5 = tie. Defaults to 1.0.
        pair_id: Unique identifier linking this pair. Auto-generated using
            uuid4().hex if None.
        max_length: Optional maximum sequence length. If provided, sequences
            are truncated to this length.
        use_chat_template: If True, format as chat messages using the tokenizer's
            chat template. This ensures the RM sees the same format the model
            generates at inference time. Defaults to False for backward compat.
        system_prompt: Optional system prompt to prepend when use_chat_template=True.
            Ignored if use_chat_template=False.

    Returns:
        Tuple of (chosen_item, rejected_item) linked by pair_id in metadata.

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> chosen_item, rejected_item = create_preference_saw_items(
        ...     prompt="What is 2+2?",
        ...     chosen="4",
        ...     rejected="5",
        ...     tokenizer=tokenizer,
        ...     use_chat_template=True,
        ... )
        >>> chosen_item.meta["role"]
        'chosen'
        >>> rejected_item.meta["pair_id"] == chosen_item.meta["pair_id"]
        True
    """
    if pair_id is None:
        pair_id = uuid4().hex

    # Validate chat template support if requested
    if use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "use_chat_template=True requires a tokenizer with apply_chat_template() method. "
                "Use a HuggingFace tokenizer from an instruction-tuned model."
            )

    def make_item(completion: str, role: str) -> SAWItem:
        if use_chat_template:
            return _make_item_with_chat_template(
                prompt=prompt,
                completion=completion,
                role=role,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                max_length=max_length,
                pair_id=pair_id,
                label=label,
            )
        else:
            return _make_item_raw(
                prompt=prompt,
                completion=completion,
                role=role,
                tokenizer=tokenizer,
                max_length=max_length,
                pair_id=pair_id,
                label=label,
            )

    chosen_item = make_item(chosen, "chosen")
    rejected_item = make_item(rejected, "rejected")

    return chosen_item, rejected_item


def _make_item_with_chat_template(
    *,
    prompt: str,
    completion: str,
    role: str,
    tokenizer: Any,
    system_prompt: Optional[str],
    max_length: Optional[int],
    pair_id: str,
    label: float,
) -> SAWItem:
    """Create a SAWItem using the tokenizer's chat template.

    The action_mask covers only the assistant's response content, not the
    chat template tokens (e.g., <|im_start|>assistant).

    Uses the shared tokenize_with_chat_template() helper for consistent
    boundary detection and truncation handling across the codebase.
    """
    try:
        result = tokenize_with_chat_template(
            tokenizer,
            prompt=prompt,
            completion=completion,
            system_prompt=system_prompt,
            max_length=max_length,
        )
    except ValueError as e:
        # Re-raise with role context for better error messages
        raise ValueError(f"{e} (role={role})") from e

    tokens = result.tokens
    prompt_len = result.prompt_len
    seq_len = len(tokens)
    completion_len = seq_len - prompt_len

    action_mask = [0] * prompt_len + [1] * completion_len

    # Build metadata with preference info
    meta: Dict[str, Any] = {
        "pair_id": pair_id,
        "role": role,
        "label": label,
    }

    # Add truncation info if truncation occurred (per CONSIDERATIONS.md semantics)
    if result.truncation_meta is not None:
        truncation_meta = dict(result.truncation_meta)
        truncation_meta.setdefault("prompt_truncated", False)
        meta["truncation"] = truncation_meta
        meta["truncated"] = True
        meta["truncation_reason"] = "max_length"
        meta["seq_len_truncated"] = True
        meta["seq_len_original"] = truncation_meta["original_len"]
        meta["seq_len_retained"] = seq_len
        meta["seq_len_retained_frac"] = float(seq_len) / float(truncation_meta["original_len"])

    return SAWItem(
        input_ids=tokens,
        attention_mask=[1] * seq_len,
        action_mask=action_mask,
        weight=1.0,
        meta=meta,
    )


def _make_item_raw(
    *,
    prompt: str,
    completion: str,
    role: str,
    tokenizer: Tokenizer,
    max_length: Optional[int],
    pair_id: str,
    label: float,
) -> SAWItem:
    """Create a SAWItem by directly encoding prompt + completion (no chat template).

    This is the legacy behavior for backward compatibility.
    """
    # Tokenize prompt and completion separately to get accurate boundaries
    # This avoids issues with BPE merge differences at concatenation points
    prompt_tokens = tokenizer.encode(prompt)
    completion_tokens = tokenizer.encode(completion)

    prompt_len = len(prompt_tokens)
    completion_len = len(completion_tokens)

    # Handle tokenizers that add special tokens (BOS/EOS)
    # For tokenizers like Llama/Mistral that add both BOS and EOS:
    #   prompt_tokens = [BOS, ..., EOS]
    #   completion_tokens = [BOS, ..., EOS]
    # We need: [BOS, prompt..., completion..., EOS]
    # So strip EOS from prompt end, and BOS from completion start
    if completion_len > 0 and prompt_len > 0:
        # Strip EOS from prompt end (if present)
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            if prompt_tokens[-1] == tokenizer.eos_token_id:
                prompt_tokens = prompt_tokens[:-1]
                prompt_len = len(prompt_tokens)

        # Strip BOS from completion start (if present)
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            if completion_tokens[0] == tokenizer.bos_token_id:
                completion_tokens = completion_tokens[1:]
                completion_len = len(completion_tokens)

    if max_length is not None and prompt_len >= max_length:
        raise ValueError(
            f"Prompt length leaves no room for completion tokens for {role} response. "
            f"prompt_len={prompt_len}, completion_len={completion_len}, "
            f"max_length={max_length}."
        )

    # Track original lengths after any special-token stripping
    original_prompt_len = prompt_len
    original_completion_len = completion_len

    # Concatenate tokens
    tokens = prompt_tokens + completion_tokens
    seq_len = len(tokens)
    original_seq_len = seq_len

    # Track truncation for downstream filtering/logging (per CONSIDERATIONS.md)
    truncation_meta: Optional[Dict[str, Any]] = None
    prompt_truncated = False
    completion_truncated = False

    # Apply max_length truncation if specified.
    # Prefer preserving completion tokens by truncating the prompt from the left.
    if max_length is not None and seq_len > max_length:
        if completion_len >= max_length:
            completion_tokens = completion_tokens[-max_length:]
            prompt_tokens = []
        else:
            prompt_budget = max_length - completion_len
            if prompt_len > prompt_budget:
                prompt_tokens = prompt_tokens[-prompt_budget:]
        prompt_len = len(prompt_tokens)
        completion_len = len(completion_tokens)
        tokens = prompt_tokens + completion_tokens
        seq_len = len(tokens)

        # Record truncation metadata
        prompt_truncated = prompt_len < original_prompt_len
        completion_truncated = completion_len < original_completion_len
        truncation_meta = {
            "original_len": original_seq_len,
            "truncated_len": original_seq_len - seq_len,
            "prompt_truncated": prompt_truncated,
            "completion_truncated": completion_truncated,
        }

    # Recalculate effective lengths after truncation
    effective_prompt_len = min(prompt_len, seq_len)
    num_completion_tokens = seq_len - effective_prompt_len

    # Validate that we have at least one completion token to train on
    if num_completion_tokens == 0:
        raise ValueError(
            f"Truncation eliminated all completion tokens for {role} response. "
            f"prompt_len={prompt_len}, completion_len={completion_len}, "
            f"seq_len={seq_len}, max_length={max_length}. "
            f"Increase max_length or use shorter prompts."
        )

    action_mask = [0] * effective_prompt_len + [1] * num_completion_tokens

    # Build metadata with preference info
    meta: Dict[str, Any] = {
        "pair_id": pair_id,
        "role": role,
        "label": label,
    }

    # Add truncation info if truncation occurred
    if truncation_meta is not None:
        meta["truncation"] = truncation_meta
        meta["truncated"] = True
        meta["truncation_reason"] = "max_length"
        meta["seq_len_truncated"] = True
        meta["seq_len_original"] = truncation_meta["original_len"]
        meta["seq_len_retained"] = seq_len
        meta["seq_len_retained_frac"] = float(seq_len) / float(truncation_meta["original_len"])

    return SAWItem(
        input_ids=tokens,
        attention_mask=[1] * seq_len,
        action_mask=action_mask,
        weight=1.0,
        meta=meta,
    )


def preference_dataset_to_saw_items(
    dataset: Iterable[Dict[str, Any]],
    tokenizer: Tokenizer,
    *,
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
    label_col: Optional[str] = None,
    max_length: Optional[int] = None,
    use_chat_template: bool = False,
    system_prompt: Optional[str] = None,
) -> List[SAWItem]:
    """Convert a preference dataset to a list of SAWItems.

    Items are returned interleaved as pairs: [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    This layout enables efficient iteration over pairs.

    Args:
        dataset: Iterable of dicts with prompt/chosen/rejected columns.
            Works with HuggingFace datasets or plain lists of dicts.
        tokenizer: Tokenizer with encode() method for text → token IDs.
            If use_chat_template=True, must also have apply_chat_template().
        prompt_col: Column name for prompt/context. Defaults to "prompt".
        chosen_col: Column name for preferred completion. Defaults to "chosen".
        rejected_col: Column name for dispreferred completion. Defaults to "rejected".
        label_col: Optional column name for soft preference labels.
            If None, defaults to 1.0 (chosen always preferred).
        max_length: Optional maximum sequence length for truncation.
        use_chat_template: If True, format as chat messages using the tokenizer's
            chat template. This ensures the RM sees the same format the model
            generates at inference time. Defaults to False for backward compat.
        system_prompt: Optional system prompt to prepend when use_chat_template=True.
            Ignored if use_chat_template=False.

    Returns:
        List of SAWItems with interleaved chosen/rejected pairs.
        Length is 2 * number of rows in dataset.

    Example:
        >>> dataset = [
        ...     {"prompt": "Q: 2+2?", "chosen": " 4", "rejected": " 5"},
        ...     {"prompt": "Q: 3+3?", "chosen": " 6", "rejected": " 7"},
        ... ]
        >>> items = preference_dataset_to_saw_items(dataset, tokenizer, use_chat_template=True)
        >>> len(items)
        4
        >>> items[0].meta["pair_id"]
        'pref_0'
        >>> items[1].meta["pair_id"]
        'pref_0'
    """
    items: List[SAWItem] = []
    skipped = 0

    for i, row in enumerate(dataset):
        # Extract label from column if specified, else default to 1.0
        if label_col is not None:
            label = float(row.get(label_col, 1.0))
        else:
            label = 1.0

        # Skip samples with empty completions
        chosen_text = row[chosen_col]
        rejected_text = row[rejected_col]
        if not chosen_text or not chosen_text.strip():
            logger.warning(f"Skipping sample {i}: empty chosen completion")
            skipped += 1
            continue
        if not rejected_text or not rejected_text.strip():
            logger.warning(f"Skipping sample {i}: empty rejected completion")
            skipped += 1
            continue

        try:
            chosen_item, rejected_item = create_preference_saw_items(
                prompt=row[prompt_col],
                chosen=chosen_text,
                rejected=rejected_text,
                tokenizer=tokenizer,
                label=label,
                pair_id=f"pref_{i}",
                max_length=max_length,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
            )
            items.append(chosen_item)
            items.append(rejected_item)
        except ValueError as e:
            logger.warning(f"Skipping sample {i}: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Sample %d snippets: prompt=%r chosen=%r rejected=%r",
                    i,
                    _shorten_text(row.get(prompt_col, "")),
                    _shorten_text(chosen_text),
                    _shorten_text(rejected_text),
                )
            skipped += 1
            continue

    if skipped > 0:
        logger.info(f"Skipped {skipped} samples due to tokenization issues")

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

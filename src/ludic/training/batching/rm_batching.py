"""RM (Reward Model) batching utilities.

This module provides preference-aware batching for reward model training:
- Atomic pair handling (chosen/rejected pairs never split across micro-batches)
- Token budget-aware pair splitting
- Loss scaling computation for gradient accumulation
- Structured metadata collation
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from ludic.training.batching.micro_batching import collate_saw_items, _truncate_item
from ludic.training.types import SAWItem


def group_preference_pairs(
    items: List[SAWItem],
    *,
    require_complete: bool = True,
    drop_incomplete: bool = True,
) -> List[Tuple[SAWItem, SAWItem]]:
    """Group SAWItems into (chosen, rejected) pairs by pair_id.

    Args:
        items: SAWItems containing meta["pair_id"] and meta["role"].
        require_complete: If True, require both chosen and rejected items.
        drop_incomplete: If True, drop incomplete pairs; otherwise raise.

    Returns:
        List of (chosen, rejected) tuples.

    Raises:
        KeyError: If pair_id or role metadata is missing.
        ValueError: If role is not "chosen" or "rejected", or if there are
            duplicate roles for the same pair_id, or if require_complete=True
            and drop_incomplete=False and there are incomplete pairs.
    """
    pair_map: Dict[str, Dict[str, SAWItem]] = {}
    for it in items:
        if "pair_id" not in it.meta:
            raise KeyError("SAWItem.meta missing 'pair_id'")
        if "role" not in it.meta:
            raise KeyError("SAWItem.meta missing 'role'")
        pid = it.meta["pair_id"]
        role = it.meta["role"]
        if role not in ("chosen", "rejected"):
            raise ValueError(f"Unknown preference role: {role!r}")
        if pid not in pair_map:
            pair_map[pid] = {}
        if role in pair_map[pid]:
            raise ValueError(f"Duplicate role {role!r} for pair_id {pid!r}")
        pair_map[pid][role] = it

    pairs: List[Tuple[SAWItem, SAWItem]] = []
    for pid, roles in pair_map.items():
        has_chosen = "chosen" in roles
        has_rejected = "rejected" in roles
        if has_chosen and has_rejected:
            pairs.append((roles["chosen"], roles["rejected"]))
        elif require_complete:
            if drop_incomplete:
                continue
            raise ValueError(f"Incomplete preference pair for pair_id {pid!r}")
    return pairs


def split_preference_pairs_by_token_budget(
    pairs: List[Tuple[SAWItem, SAWItem]],
    *,
    micro_token_budget: int,
    max_seq_len: int,
) -> List[List[SAWItem]]:
    """Split preference pairs into micro-batches without splitting pairs.

    This ensures atomic handling of preference pairs - a chosen/rejected pair
    is always kept together in the same micro-batch.

    The budget check uses the padded token estimate:
        max_seq_len_in_batch * num_items_in_batch <= micro_token_budget
    where num_items_in_batch = 2 * num_pairs_in_batch.

    Args:
        pairs: List of (chosen, rejected) tuples from group_preference_pairs().
        micro_token_budget: Maximum padded tokens per micro-batch.
        max_seq_len: Maximum sequence length. Sequences longer than this
            will be truncated.

    Returns:
        List of micro-batches, where each micro-batch is a list of SAWItems.
        Items are interleaved: [chosen_0, rejected_0, chosen_1, rejected_1, ...]

    Raises:
        ValueError: If micro_token_budget <= 0, max_seq_len <= 0, or
            micro_token_budget < max_seq_len.
    """
    if not pairs:
        return []
    if micro_token_budget <= 0:
        raise ValueError("micro_token_budget must be > 0.")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0.")
    if micro_token_budget < max_seq_len:
        raise ValueError(
            f"micro_token_budget ({micro_token_budget}) must be >= max_seq_len ({max_seq_len})."
        )

    processed: List[Tuple[int, SAWItem, SAWItem]] = []
    for chosen, rejected in pairs:
        if len(chosen.input_ids) > max_seq_len:
            chosen = _truncate_item(chosen, max_seq_len)
        if len(rejected.input_ids) > max_seq_len:
            rejected = _truncate_item(rejected, max_seq_len)
        pair_len = max(len(chosen.input_ids), len(rejected.input_ids))
        if pair_len > max_seq_len:
            raise ValueError(
                f"Pair length {pair_len} exceeds max_seq_len {max_seq_len} "
                f"(pair_id={chosen.meta.get('pair_id')!r})."
            )
        processed.append((pair_len, chosen, rejected))

    processed.sort(key=lambda item: item[0])

    micro_batches: List[List[SAWItem]] = []
    current: List[SAWItem] = []
    current_max = 0
    current_pairs = 0

    for pair_len, chosen, rejected in processed:
        next_max = pair_len if current_pairs == 0 else max(current_max, pair_len)
        next_pairs = current_pairs + 1
        next_item_count = next_pairs * 2
        if current and (next_max * next_item_count) > micro_token_budget:
            micro_batches.append(current)
            current = []
            current_max = 0
            current_pairs = 0
            next_max = pair_len
            next_pairs = 1

        current.extend([chosen, rejected])
        current_max = next_max
        current_pairs = next_pairs

    if current:
        micro_batches.append(current)

    return micro_batches


def collate_preference_items(
    items: List[SAWItem],
    *,
    pad_token_id: int,
    device: torch.device,
    meta_keys: Sequence[str] = ("pair_id", "role", "label"),
) -> Dict[str, Any]:
    """Collate preference SAWItems into tensors with structured metadata.

    Args:
        items: List of SAWItems with preference metadata.
        pad_token_id: Token ID used for padding.
        device: Device to place tensors on.
        meta_keys: Metadata keys to extract and include in the batch.

    Returns:
        Dict containing:
        - input_ids: [B, T] tensor
        - attention_mask: [B, T] tensor
        - action_mask: [B, T] tensor
        - weight: [B] tensor
        - meta: dict mapping each key in meta_keys to list of values

    Raises:
        KeyError: If any meta_key is missing from an item's metadata.
    """
    batch = collate_saw_items(items, pad_token_id=pad_token_id, device=device)
    meta: Dict[str, List[Any]] = {key: [] for key in meta_keys}
    for it in items:
        for key in meta_keys:
            if key not in it.meta:
                raise KeyError(f"SAWItem.meta missing {key!r}")
            meta[key].append(it.meta[key])
    batch["meta"] = meta
    return batch


def compute_pair_loss_scale(
    num_pairs_in_micro: int,
    total_pairs: int,
) -> float:
    """Compute loss scaling factor for gradient accumulation across micro-batches.

    When using gradient accumulation with pair-aware micro-batching, the loss
    from each micro-batch should be scaled by the fraction of total pairs it
    contains. This ensures the gradient magnitude is equivalent to processing
    all pairs in a single batch.

    Args:
        num_pairs_in_micro: Number of pairs in the current micro-batch.
        total_pairs: Total number of pairs across all micro-batches.

    Returns:
        Scale factor to multiply with the micro-batch loss.

    Example:
        >>> compute_pair_loss_scale(2, 10)
        0.2
        >>> # Loss from this micro-batch should be: loss * 0.2
    """
    if total_pairs <= 0:
        return 0.0
    return num_pairs_in_micro / total_pairs


def validate_preference_batch(
    items: List[SAWItem],
    *,
    required_keys: Sequence[str] = ("pair_id", "role", "label"),
) -> None:
    """Validate that all items have required preference metadata.

    Args:
        items: List of SAWItems to validate.
        required_keys: Metadata keys that must be present.

    Raises:
        KeyError: If any required key is missing from any item.
        ValueError: If any item has an invalid role value.
    """
    for i, item in enumerate(items):
        for key in required_keys:
            if key not in item.meta:
                raise KeyError(
                    f"SAWItem[{i}].meta missing required key {key!r}"
                )
        role = item.meta.get("role")
        if role is not None and role not in ("chosen", "rejected"):
            raise ValueError(
                f"SAWItem[{i}].meta['role'] has invalid value {role!r}. "
                f"Expected 'chosen' or 'rejected'."
            )


def count_pairs_in_items(items: List[SAWItem]) -> int:
    """Count the number of complete preference pairs in a list of items.

    Args:
        items: List of SAWItems with pair_id and role metadata.

    Returns:
        Number of complete (chosen, rejected) pairs.
    """
    pair_ids = set()
    roles_by_pair: Dict[str, set] = {}

    for item in items:
        pid = item.meta.get("pair_id")
        role = item.meta.get("role")
        if pid is None or role is None:
            continue
        if pid not in roles_by_pair:
            roles_by_pair[pid] = set()
        roles_by_pair[pid].add(role)

    complete_pairs = 0
    for pid, roles in roles_by_pair.items():
        if "chosen" in roles and "rejected" in roles:
            complete_pairs += 1

    return complete_pairs

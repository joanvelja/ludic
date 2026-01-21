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

    The budget check uses the padded token estimate:
        max_seq_len_in_batch * num_items_in_batch <= micro_token_budget
    where num_items_in_batch = 2 * num_pairs_in_batch.
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
    """Collate preference SAWItems into tensors with structured metadata."""
    batch = collate_saw_items(items, pad_token_id=pad_token_id, device=device)
    meta: Dict[str, List[Any]] = {key: [] for key in meta_keys}
    for it in items:
        for key in meta_keys:
            if key not in it.meta:
                raise KeyError(f"SAWItem.meta missing {key!r}")
            meta[key].append(it.meta[key])
    batch["meta"] = meta
    return batch

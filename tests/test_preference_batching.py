from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

from ludic.training.batching.preference import (
    collate_preference_items,
    group_preference_pairs,
    split_preference_pairs_by_token_budget,
)
from ludic.training.types import SAWItem, SampleAttachments


def _make_item(
    input_ids: List[int],
    action_mask: List[int],
    *,
    pair_id: str,
    role: str,
    label: float = 1.0,
) -> SAWItem:
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        action_mask=action_mask,
        weight=1.0,
        meta={"pair_id": pair_id, "role": role, "label": label},
        attachments=SampleAttachments(),
    )


def _make_pair(
    pair_id: str,
    chosen_len: int,
    rejected_len: int,
) -> Tuple[SAWItem, SAWItem]:
    chosen = _make_item(
        input_ids=list(range(chosen_len)),
        action_mask=[0] + [1] * (chosen_len - 1),
        pair_id=pair_id,
        role="chosen",
    )
    rejected = _make_item(
        input_ids=list(range(rejected_len)),
        action_mask=[0] + [1] * (rejected_len - 1),
        pair_id=pair_id,
        role="rejected",
    )
    return chosen, rejected


def test_group_preference_pairs_drops_incomplete_by_default():
    p0 = _make_pair("p0", 4, 5)
    incomplete = _make_item(
        input_ids=[1, 2, 3],
        action_mask=[0, 1, 1],
        pair_id="p1",
        role="chosen",
    )
    items = [p0[0], p0[1], incomplete]
    pairs = group_preference_pairs(items, require_complete=True, drop_incomplete=True)
    assert len(pairs) == 1
    assert pairs[0][0].meta["pair_id"] == "p0"
    assert pairs[0][1].meta["pair_id"] == "p0"


def test_group_preference_pairs_raises_on_incomplete_when_configured():
    p0 = _make_pair("p0", 4, 5)
    incomplete = _make_item(
        input_ids=[1, 2, 3],
        action_mask=[0, 1, 1],
        pair_id="p1",
        role="chosen",
    )
    items = [p0[0], p0[1], incomplete]
    with pytest.raises(ValueError, match="Incomplete preference pair"):
        group_preference_pairs(items, require_complete=True, drop_incomplete=False)


def test_split_preference_pairs_respects_budget_and_keeps_pairs_together():
    pairs = [_make_pair("p0", 5, 4), _make_pair("p1", 3, 2), _make_pair("p2", 6, 6)]
    micro_batches = split_preference_pairs_by_token_budget(
        pairs,
        micro_token_budget=20,
        max_seq_len=6,
    )
    assert len(micro_batches) == 2

    seen_pair_to_batch = {}
    for idx, batch in enumerate(micro_batches):
        max_len = max(len(it.input_ids) for it in batch)
        assert max_len * len(batch) <= 20
        pair_ids = [it.meta["pair_id"] for it in batch]
        assert len(pair_ids) % 2 == 0
        # each pair appears exactly twice within the batch
        for pid in set(pair_ids):
            assert pair_ids.count(pid) == 2
            if pid in seen_pair_to_batch:
                assert seen_pair_to_batch[pid] == idx
            else:
                seen_pair_to_batch[pid] = idx


def test_collate_preference_items_attaches_meta():
    chosen, rejected = _make_pair("p0", 4, 3)
    batch = collate_preference_items(
        [chosen, rejected],
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    assert "meta" in batch
    assert batch["meta"]["pair_id"] == ["p0", "p0"]
    assert batch["meta"]["role"] == ["chosen", "rejected"]
    assert batch["meta"]["label"] == [1.0, 1.0]
    assert batch["input_ids"].shape == (2, 4)

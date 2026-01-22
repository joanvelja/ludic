from __future__ import annotations

import pytest
import torch

from ludic.training.loss import BradleyTerryLoss


def test_bradley_terry_loss_interleaved_mismatched_pair_ids_falls_back() -> None:
    loss_fn = BradleyTerryLoss(score_type="reward", compile=False)

    logits = torch.tensor([2.0, 0.0, 1.0, -1.0])
    batch = {
        "meta": {
            "pair_id": ["p0", "p1", "p1", "p0"],
            "role": ["chosen", "rejected", "chosen", "rejected"],
            "label": [1.0, 1.0, 1.0, 1.0],
        }
    }

    loss, stats = loss_fn.compute(logits, batch)

    # Correct pairing by pair_id: p0 -> (2.0, -1.0), p1 -> (1.0, 0.0)
    expected_margin = torch.tensor([3.0, 1.0])
    expected = (-torch.log(torch.sigmoid(expected_margin))).mean()

    assert torch.allclose(loss, expected)
    assert stats["num_pairs"].item() == pytest.approx(2.0)

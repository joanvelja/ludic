"""Preference batching utilities (backward compatibility).

This module re-exports functions from rm_batching.py for backward compatibility.
New code should import directly from rm_batching.py.
"""

from __future__ import annotations

import warnings

# Re-export from rm_batching for backward compatibility
from ludic.training.batching.rm_batching import (
    group_preference_pairs,
    split_preference_pairs_by_token_budget,
    collate_preference_items,
    compute_pair_loss_scale,
    validate_preference_batch,
    count_pairs_in_items,
)

__all__ = [
    "group_preference_pairs",
    "split_preference_pairs_by_token_budget",
    "collate_preference_items",
    "compute_pair_loss_scale",
    "validate_preference_batch",
    "count_pairs_in_items",
]


def _emit_deprecation_warning():
    warnings.warn(
        "ludic.training.batching.preference is deprecated. "
        "Import from ludic.training.batching.rm_batching instead.",
        DeprecationWarning,
        stacklevel=3,
    )

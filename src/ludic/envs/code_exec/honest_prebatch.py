"""
Utility for pre-batching honest solutions from a dataset.

This allows efficient collection of all reference solutions before
training, avoiding repeated lookups during env resets.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, Optional, Tuple

from .adapters.base import ReferenceSolutionProvider


def prebatch_honest_codes(
    samples: Iterable[Dict[str, Any]],
    adapter: ReferenceSolutionProvider,
    *,
    parallel: bool = True,
    max_workers: int = 8,
) -> Dict[str, str]:
    """
    Extract honest (reference) solutions from a dataset.

    Args:
        samples: Iterable of dataset samples
        adapter: Adapter that can provide reference solutions
        parallel: Whether to use parallel extraction (default: True)
        max_workers: Number of parallel workers (default: 8)

    Returns:
        Dict mapping problem_id -> honest_code for all valid samples.
        Samples without valid honest solutions are skipped.

    Example:
        >>> adapter = APPSTestAdapter()
        >>> dataset = load_dataset("RoganInglis/apps-control-arena", split="train")
        >>> honest_codes = prebatch_honest_codes(dataset, adapter)
        >>> print(f"Found {len(honest_codes)} valid honest solutions")
    """
    samples_list = list(samples)  # Materialize for parallel processing

    # Need get_problem_id and get_reference_solution methods
    if not hasattr(adapter, "get_problem_id"):
        raise TypeError("adapter must have get_problem_id method (from TestAdapter protocol)")
    if not hasattr(adapter, "get_reference_solution"):
        raise TypeError("adapter must have get_reference_solution method (from ReferenceSolutionProvider protocol)")

    def extract_one(sample: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Extract (problem_id, code) from a single sample."""
        problem_id = adapter.get_problem_id(sample)
        code = adapter.get_reference_solution(sample)
        if code is None:
            return None
        return (problem_id, code)

    if parallel and len(samples_list) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(extract_one, samples_list))
    else:
        results = [extract_one(s) for s in samples_list]

    # Filter out None results and build dict
    return {pid: code for result in results if result is not None for pid, code in [result]}


def filter_valid_samples(
    samples: Iterable[Dict[str, Any]],
    adapter: ReferenceSolutionProvider,
) -> list:
    """
    Filter samples to only those valid for sneaky verification.

    Args:
        samples: Iterable of dataset samples
        adapter: Adapter that can check validity

    Returns:
        List of samples that are valid for sneaky verification
    """
    return [s for s in samples if adapter.is_valid_for_sneaky(s)]

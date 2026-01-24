"""
LRU cache for sneaky verification evaluation results.

Provides thread-safe caching of evaluation results keyed by:
(sneaky_code_hash, honest_code_hash, certificate_hash, tests_hash)

Used to avoid redundant re-execution of sneaky verification checks.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Tuple


def compute_hash(content: str) -> str:
    """
    Compute a deterministic hash of string content.

    Uses SHA-256 truncated to 16 hex characters for a good balance of
    collision resistance and key length.

    Args:
        content: The string content to hash.

    Returns:
        A 16-character hex string hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheStats:
    """
    Statistics for the evaluation cache.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        size: Current number of entries in the cache.
        max_size: Maximum number of entries the cache can hold.
    """

    hits: int
    misses: int
    size: int
    max_size: int

    @property
    def hit_rate(self) -> float:
        """
        Get cache hit rate as a float between 0 and 1.

        Returns 0.0 if no lookups have been performed.
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


# Type alias for the 4-tuple cache key
_CacheKey = Tuple[str, str, str, str]


class EvalCache:
    """
    Thread-safe LRU cache for sneaky verification evaluation results.

    Uses OrderedDict for LRU semantics and threading.Lock for safety.
    Suitable for use across multiple async tasks sharing the same pool.

    The cache key is a 4-tuple of:
    (sneaky_code_hash, honest_code_hash, certificate_hash, tests_hash)

    Args:
        max_size: Maximum number of entries to cache. Oldest entries are
            evicted when this limit is exceeded.
    """

    def __init__(self, max_size: int = 10000):
        self._max_size = max_size
        self._cache: OrderedDict[_CacheKey, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(
        self,
        sneaky_hash: str,
        honest_hash: str,
        cert_hash: str,
        tests_hash: str,
    ) -> Optional[Any]:
        """
        Get cached result.

        On hit, moves item to end (most recently used).
        Thread-safe.

        Args:
            sneaky_hash: Hash of the sneaky code being evaluated.
            honest_hash: Hash of the honest/reference code.
            cert_hash: Hash of the certificate input.
            tests_hash: Hash of the test cases.

        Returns:
            Cached evaluation result if found, None otherwise.
            Note: A stored None value is distinguishable from a cache miss
            by checking the stats - a stored None is counted as a hit.
        """
        key = (sneaky_hash, honest_hash, cert_hash, tests_hash)
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None

    def put(
        self,
        sneaky_hash: str,
        honest_hash: str,
        cert_hash: str,
        tests_hash: str,
        result: Any,
    ) -> None:
        """
        Cache a result.

        Evicts oldest item if cache is full.
        Thread-safe.

        Args:
            sneaky_hash: Hash of the sneaky code being evaluated.
            honest_hash: Hash of the honest/reference code.
            cert_hash: Hash of the certificate input.
            tests_hash: Hash of the test cases.
            result: The evaluation result to cache (can be any type).
        """
        key = (sneaky_hash, honest_hash, cert_hash, tests_hash)
        with self._lock:
            if key in self._cache:
                # Update existing entry and move to end
                self._cache[key] = result
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = result
                # Evict oldest if over limit
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)  # FIFO: remove oldest

    def clear(self) -> None:
        """Clear all cached entries. Thread-safe."""
        with self._lock:
            self._cache.clear()
            # Note: We don't reset hit/miss counters on clear

    def reset_stats(self) -> None:
        """Reset hit/miss counters to zero. Thread-safe."""
        with self._lock:
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> CacheStats:
        """
        Get cache statistics (thread-safe).

        Returns a new CacheStats dataclass each time, so mutations
        to the returned object don't affect the cache.

        Returns:
            CacheStats with hits, misses, size, max_size, and hit_rate.
        """
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                size=len(self._cache),
                max_size=self._max_size,
            )

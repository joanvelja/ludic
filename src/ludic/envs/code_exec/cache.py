"""
Shared LRU cache for code execution results.

Provides thread-safe caching of BatchTestResult keyed by (code_hash, tests_hash).
Used by both Docker and Podman sandbox pools to avoid redundant execution of
identical code/test combinations.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Dict, Optional

from .types import BatchTestResult


class LRUCache:
    """
    Thread-safe LRU cache for BatchTestResult.

    Uses OrderedDict for LRU semantics and threading.Lock for safety.
    Suitable for use across multiple async tasks sharing the same pool.

    Args:
        max_size: Maximum number of entries to cache. Oldest entries are
            evicted when this limit is exceeded.
    """

    def __init__(self, max_size: int = 10000):
        self._max_size = max_size
        self._cache: OrderedDict[tuple[str, str], BatchTestResult] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(
        self,
        code_hash: str,
        tests_hash: str,
    ) -> Optional[BatchTestResult]:
        """
        Get cached result.

        On hit, moves item to end (most recently used).
        Thread-safe.

        Args:
            code_hash: Hash of the code being executed.
            tests_hash: Hash of the test cases.

        Returns:
            Cached BatchTestResult if found, None otherwise.
        """
        key = (code_hash, tests_hash)
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
        code_hash: str,
        tests_hash: str,
        result: BatchTestResult,
    ) -> None:
        """
        Cache a result.

        Evicts oldest item if cache is full.
        Thread-safe.

        Args:
            code_hash: Hash of the code being executed.
            tests_hash: Hash of the test cases.
            result: The test result to cache.
        """
        key = (code_hash, tests_hash)
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

    @property
    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics (thread-safe).

        Returns:
            Dict with keys: hits, misses, size, max_size
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
            }

    @property
    def hit_rate(self) -> float:
        """
        Get cache hit rate as a float between 0 and 1.

        Returns 0.0 if no lookups have been performed.
        """
        with self._lock:
            total = self._hits + self._misses
            if total == 0:
                return 0.0
            return self._hits / total

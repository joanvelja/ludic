"""
Unit tests for ludic.envs.code_exec.docker_sandbox.LRUCache

Tests thread safety, eviction behavior, and statistics tracking.

Note: Requires the `docker` package to be installed for LRUCache import.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ludic.envs.code_exec.types import (
    BatchTestResult,
    CompileResult,
    CompileStatus,
    ExecutionResult,
    RunStatus,
    TestCase,
    TestResult,
)

# Try to import LRUCache - skip all tests if docker package not installed
try:
    from ludic.envs.code_exec.docker_sandbox import LRUCache
except ImportError:
    LRUCache = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(
    LRUCache is None,
    reason="docker package not installed (required for LRUCache)",
)


def _make_batch_result(passed_count: int = 1, total_count: int = 1) -> BatchTestResult:
    """Helper to create a BatchTestResult with minimal boilerplate."""
    results = []
    for i in range(total_count):
        passed = i < passed_count
        results.append(
            TestResult(
                test_case=TestCase(input=f"input_{i}", expected="out", id=f"t{i}"),
                passed=passed,
                actual="out" if passed else "wrong",
                execution=ExecutionResult(
                    compile_result=CompileResult(status=CompileStatus.SUCCESS),
                    run_status=RunStatus.SUCCESS if passed else RunStatus.RUNTIME_ERROR,
                ),
            )
        )
    return BatchTestResult(results=results, code_hash="abc", tests_hash="xyz")


# ---------------------------------------------------------------------
# Basic Operations
# ---------------------------------------------------------------------


class TestLRUCacheBasicOperations:
    def test_get_returns_none_for_missing_key(self):
        cache = LRUCache(max_size=10)
        result = cache.get("nonexistent_code", "nonexistent_tests")
        assert result is None

    def test_put_and_get(self):
        cache = LRUCache(max_size=10)
        batch_result = _make_batch_result()

        cache.put("code1", "tests1", batch_result)
        retrieved = cache.get("code1", "tests1")

        assert retrieved is batch_result

    def test_get_returns_none_after_different_key(self):
        cache = LRUCache(max_size=10)
        batch_result = _make_batch_result()

        cache.put("code1", "tests1", batch_result)

        # Different code hash
        assert cache.get("code2", "tests1") is None
        # Different tests hash
        assert cache.get("code1", "tests2") is None
        # Both different
        assert cache.get("code2", "tests2") is None

    def test_put_overwrites_existing_entry(self):
        cache = LRUCache(max_size=10)
        result1 = _make_batch_result(passed_count=1, total_count=2)
        result2 = _make_batch_result(passed_count=2, total_count=2)

        cache.put("code1", "tests1", result1)
        cache.put("code1", "tests1", result2)

        retrieved = cache.get("code1", "tests1")
        assert retrieved is result2
        assert retrieved.passed_count == 2


# ---------------------------------------------------------------------
# Eviction Behavior
# ---------------------------------------------------------------------


class TestLRUCacheEviction:
    def test_evicts_oldest_when_full(self):
        cache = LRUCache(max_size=3)

        cache.put("code1", "tests", _make_batch_result())
        cache.put("code2", "tests", _make_batch_result())
        cache.put("code3", "tests", _make_batch_result())

        # Cache is now full
        assert cache.stats["size"] == 3

        # Add one more - oldest (code1) should be evicted
        cache.put("code4", "tests", _make_batch_result())

        assert cache.stats["size"] == 3
        assert cache.get("code1", "tests") is None  # Evicted
        assert cache.get("code2", "tests") is not None
        assert cache.get("code3", "tests") is not None
        assert cache.get("code4", "tests") is not None

    def test_access_refreshes_entry_avoiding_eviction(self):
        cache = LRUCache(max_size=3)

        cache.put("code1", "tests", _make_batch_result())
        cache.put("code2", "tests", _make_batch_result())
        cache.put("code3", "tests", _make_batch_result())

        # Access code1 to make it most recently used
        cache.get("code1", "tests")

        # Add new entry - code2 (now oldest accessed) should be evicted
        cache.put("code4", "tests", _make_batch_result())

        assert cache.get("code1", "tests") is not None  # Still present
        assert cache.get("code2", "tests") is None  # Evicted
        assert cache.get("code3", "tests") is not None
        assert cache.get("code4", "tests") is not None

    def test_put_refreshes_existing_entry(self):
        cache = LRUCache(max_size=3)

        cache.put("code1", "tests", _make_batch_result())
        cache.put("code2", "tests", _make_batch_result())
        cache.put("code3", "tests", _make_batch_result())

        # Update code1 (makes it most recently used)
        cache.put("code1", "tests", _make_batch_result())

        # Add new entry - code2 should be evicted now
        cache.put("code4", "tests", _make_batch_result())

        assert cache.get("code1", "tests") is not None
        assert cache.get("code2", "tests") is None  # Evicted
        assert cache.get("code3", "tests") is not None
        assert cache.get("code4", "tests") is not None

    def test_max_size_one(self):
        cache = LRUCache(max_size=1)

        cache.put("code1", "tests", _make_batch_result())
        assert cache.get("code1", "tests") is not None

        cache.put("code2", "tests", _make_batch_result())
        assert cache.get("code1", "tests") is None
        assert cache.get("code2", "tests") is not None


# ---------------------------------------------------------------------
# Statistics Tracking
# ---------------------------------------------------------------------


class TestLRUCacheStats:
    def test_initial_stats(self):
        cache = LRUCache(max_size=100)
        stats = cache.stats

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["max_size"] == 100

    def test_hit_tracking(self):
        cache = LRUCache(max_size=10)
        cache.put("code", "tests", _make_batch_result())

        # First hit
        cache.get("code", "tests")
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 0

        # Second hit
        cache.get("code", "tests")
        assert cache.stats["hits"] == 2
        assert cache.stats["misses"] == 0

    def test_miss_tracking(self):
        cache = LRUCache(max_size=10)

        # First miss
        cache.get("nonexistent", "tests")
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 1

        # Second miss
        cache.get("also_nonexistent", "tests")
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 2

    def test_mixed_hits_and_misses(self):
        cache = LRUCache(max_size=10)
        cache.put("code1", "tests", _make_batch_result())

        cache.get("code1", "tests")  # hit
        cache.get("code2", "tests")  # miss
        cache.get("code1", "tests")  # hit
        cache.get("code3", "tests")  # miss
        cache.get("code1", "tests")  # hit

        stats = cache.stats
        assert stats["hits"] == 3
        assert stats["misses"] == 2

    def test_size_tracking(self):
        cache = LRUCache(max_size=10)

        assert cache.stats["size"] == 0

        cache.put("code1", "tests", _make_batch_result())
        assert cache.stats["size"] == 1

        cache.put("code2", "tests", _make_batch_result())
        assert cache.stats["size"] == 2

        # Overwrite existing doesn't increase size
        cache.put("code1", "tests", _make_batch_result())
        assert cache.stats["size"] == 2


# ---------------------------------------------------------------------
# Thread Safety
# ---------------------------------------------------------------------


class TestLRUCacheThreadSafety:
    def test_concurrent_puts(self):
        cache = LRUCache(max_size=1000)
        n_threads = 10
        puts_per_thread = 100

        def put_items(thread_id: int):
            for i in range(puts_per_thread):
                cache.put(f"code_{thread_id}_{i}", "tests", _make_batch_result())

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(put_items, i) for i in range(n_threads)]
            for f in futures:
                f.result()

        # All items should be accessible
        expected_size = n_threads * puts_per_thread
        assert cache.stats["size"] == expected_size

    def test_concurrent_gets(self):
        cache = LRUCache(max_size=100)

        # Pre-populate
        for i in range(100):
            cache.put(f"code_{i}", "tests", _make_batch_result())

        n_threads = 10
        gets_per_thread = 100

        def get_items(thread_id: int):
            hits = 0
            for i in range(gets_per_thread):
                key = f"code_{i % 100}"  # Round-robin through existing keys
                if cache.get(key, "tests") is not None:
                    hits += 1
            return hits

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(get_items, i) for i in range(n_threads)]
            results = [f.result() for f in futures]

        # All gets should have found their items
        assert all(r == gets_per_thread for r in results)

        # Stats should reflect all hits
        assert cache.stats["hits"] == n_threads * gets_per_thread

    def test_concurrent_mixed_operations(self):
        cache = LRUCache(max_size=50)
        n_threads = 8
        ops_per_thread = 100

        errors = []

        def mixed_operations(thread_id: int):
            try:
                for i in range(ops_per_thread):
                    if i % 3 == 0:
                        cache.put(f"code_{i}", "tests", _make_batch_result())
                    else:
                        cache.get(f"code_{i % 30}", "tests")
                    # Access stats during operations
                    _ = cache.stats
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(n_threads)]
            for f in futures:
                f.result()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"

        # Cache should be in a consistent state
        stats = cache.stats
        assert stats["size"] <= stats["max_size"]
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0

    def test_concurrent_eviction_stress(self):
        """Test that concurrent puts with eviction don't cause issues."""
        cache = LRUCache(max_size=10)
        n_threads = 20
        puts_per_thread = 100

        errors = []

        def stress_puts(thread_id: int):
            try:
                for i in range(puts_per_thread):
                    cache.put(f"code_{thread_id}_{i}", "tests", _make_batch_result())
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(stress_puts, i) for i in range(n_threads)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        assert cache.stats["size"] == 10  # Should stay at max


# ---------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------


class TestLRUCacheEdgeCases:
    def test_empty_hash_strings(self):
        cache = LRUCache(max_size=10)
        batch = _make_batch_result()

        cache.put("", "", batch)
        assert cache.get("", "") is batch

    def test_very_long_hash_strings(self):
        cache = LRUCache(max_size=10)
        batch = _make_batch_result()

        long_code_hash = "a" * 10000
        long_tests_hash = "b" * 10000

        cache.put(long_code_hash, long_tests_hash, batch)
        assert cache.get(long_code_hash, long_tests_hash) is batch

    def test_special_characters_in_hashes(self):
        cache = LRUCache(max_size=10)
        batch = _make_batch_result()

        special_hash = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        cache.put(special_hash, special_hash, batch)
        assert cache.get(special_hash, special_hash) is batch

    def test_unicode_in_hashes(self):
        cache = LRUCache(max_size=10)
        batch = _make_batch_result()

        unicode_hash = "hash_with_unicode_"
        cache.put(unicode_hash, unicode_hash, batch)
        assert cache.get(unicode_hash, unicode_hash) is batch

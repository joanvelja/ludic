"""
Unit tests for ludic.envs.code_exec.adapters

Tests verifiers and test adapters.
"""

import pytest

from ludic.envs.code_exec.adapters.base import (
    ExactMatchVerifier,
    WhitespaceNormalizedVerifier,
    FloatTolerantVerifier,
)
from ludic.envs.code_exec.adapters.apps import (
    APPSTestAdapter,
    APPS_SYSTEM_PROMPT,
)
from ludic.envs.code_exec.types import TestCase


# ---------------------------------------------------------------------
# ExactMatchVerifier Tests
# ---------------------------------------------------------------------


class TestExactMatchVerifier:
    def test_exact_match_passes(self):
        verifier = ExactMatchVerifier()
        passed, details = verifier.verify("hello", "hello")
        assert passed is True
        assert details is None

    def test_mismatch_fails(self):
        verifier = ExactMatchVerifier()
        passed, details = verifier.verify("hello", "world")
        assert passed is False
        assert details is not None

    def test_strips_whitespace_by_default(self):
        verifier = ExactMatchVerifier()
        passed, _ = verifier.verify("  hello  \n", "hello")
        assert passed is True

    def test_strip_disabled(self):
        verifier = ExactMatchVerifier(strip=False)
        passed, _ = verifier.verify("hello ", "hello")
        assert passed is False

    def test_case_sensitive_by_default(self):
        verifier = ExactMatchVerifier()
        passed, _ = verifier.verify("Hello", "hello")
        assert passed is False

    def test_case_insensitive(self):
        verifier = ExactMatchVerifier(case_sensitive=False)
        passed, _ = verifier.verify("HELLO", "hello")
        assert passed is True

    def test_length_mismatch_details(self):
        verifier = ExactMatchVerifier()
        passed, details = verifier.verify("abc", "abcdef")
        assert passed is False
        assert "Length mismatch" in details
        assert "3" in details
        assert "6" in details

    def test_first_diff_details(self):
        verifier = ExactMatchVerifier()
        passed, details = verifier.verify("abc", "axc")
        assert passed is False
        assert "First diff" in details


# ---------------------------------------------------------------------
# WhitespaceNormalizedVerifier Tests
# ---------------------------------------------------------------------


class TestWhitespaceNormalizedVerifier:
    def test_normalizes_multiple_spaces(self):
        verifier = WhitespaceNormalizedVerifier()
        passed, _ = verifier.verify("hello   world", "hello world")
        assert passed is True

    def test_normalizes_newlines(self):
        verifier = WhitespaceNormalizedVerifier()
        passed, _ = verifier.verify("hello\n\nworld", "hello world")
        assert passed is True

    def test_normalizes_tabs(self):
        verifier = WhitespaceNormalizedVerifier()
        passed, _ = verifier.verify("hello\t\tworld", "hello world")
        assert passed is True

    def test_normalizes_mixed_whitespace(self):
        verifier = WhitespaceNormalizedVerifier()
        passed, _ = verifier.verify("  hello \n\t world  ", "hello world")
        assert passed is True

    def test_content_mismatch_fails(self):
        verifier = WhitespaceNormalizedVerifier()
        passed, _ = verifier.verify("hello world", "hello mars")
        assert passed is False


# ---------------------------------------------------------------------
# FloatTolerantVerifier Tests
# ---------------------------------------------------------------------


class TestFloatTolerantVerifier:
    def test_exact_float_match(self):
        verifier = FloatTolerantVerifier()
        passed, _ = verifier.verify("3.14159", "3.14159")
        assert passed is True

    def test_float_within_tolerance(self):
        verifier = FloatTolerantVerifier(abs_tol=1e-6)
        passed, _ = verifier.verify("3.141590001", "3.14159")
        assert passed is True

    def test_float_outside_tolerance(self):
        verifier = FloatTolerantVerifier(abs_tol=1e-9)
        passed, _ = verifier.verify("3.15", "3.14")
        assert passed is False

    def test_integer_match(self):
        verifier = FloatTolerantVerifier()
        passed, _ = verifier.verify("42", "42")
        assert passed is True

    def test_string_exact_match(self):
        verifier = FloatTolerantVerifier()
        passed, _ = verifier.verify("hello", "hello")
        assert passed is True

    def test_string_mismatch(self):
        verifier = FloatTolerantVerifier()
        passed, _ = verifier.verify("hello", "world")
        assert passed is False

    def test_multiple_tokens(self):
        verifier = FloatTolerantVerifier(abs_tol=1e-6)
        passed, _ = verifier.verify("1.0 2.0 3.0", "1.0 2.0 3.0")
        assert passed is True

    def test_multiple_tokens_within_tolerance(self):
        verifier = FloatTolerantVerifier(abs_tol=0.01)
        passed, _ = verifier.verify("1.001 2.002 3.003", "1.0 2.0 3.0")
        assert passed is True

    def test_token_count_mismatch(self):
        verifier = FloatTolerantVerifier()
        passed, details = verifier.verify("1 2", "1 2 3")
        assert passed is False
        assert "Token count mismatch" in details

    def test_relative_tolerance(self):
        verifier = FloatTolerantVerifier(rel_tol=0.01, abs_tol=0)
        # 1% of 100 = 1, so 100.5 should match 100
        passed, _ = verifier.verify("100.5", "100")
        assert passed is True

    def test_strips_whitespace(self):
        verifier = FloatTolerantVerifier()
        passed, _ = verifier.verify("  42  ", "42")
        assert passed is True


# ---------------------------------------------------------------------
# APPSTestAdapter Tests
# ---------------------------------------------------------------------


class TestAPPSTestAdapter:
    def test_get_prompt_extracts_question(self):
        adapter = APPSTestAdapter()
        sample = {
            "question": "Write a function to add two numbers.",
            "inputs": ["1 2"],
            "outputs": ["3"],
        }
        prompt = adapter.get_prompt(sample)
        assert prompt == "Write a function to add two numbers."

    def test_get_prompt_with_custom_key(self):
        adapter = APPSTestAdapter(question_key="problem_description")
        sample = {
            "problem_description": "Custom problem text",
            "inputs": [],
            "outputs": [],
        }
        prompt = adapter.get_prompt(sample)
        assert prompt == "Custom problem text"

    def test_get_problem_id(self):
        adapter = APPSTestAdapter()
        sample = {
            "problem_id": "prob_123",
            "question": "Q",
            "inputs": [],
            "outputs": [],
        }
        assert adapter.get_problem_id(sample) == "prob_123"

    def test_get_problem_id_missing_returns_unknown(self):
        adapter = APPSTestAdapter()
        sample = {
            "question": "Q",
            "inputs": [],
            "outputs": [],
        }
        assert adapter.get_problem_id(sample) == "unknown"

    def test_get_problem_id_custom_key(self):
        adapter = APPSTestAdapter(problem_id_key="id")
        sample = {
            "id": "custom_id",
            "question": "Q",
            "inputs": [],
            "outputs": [],
        }
        assert adapter.get_problem_id(sample) == "custom_id"

    def test_get_tests_single_test(self):
        adapter = APPSTestAdapter()
        sample = {
            "question": "Q",
            "inputs": ["1 2"],
            "outputs": ["3"],
        }
        tests = adapter.get_tests(sample)
        assert len(tests) == 1
        assert tests[0].input == "1 2"
        assert tests[0].expected == "3"
        assert tests[0].id == "test_0"

    def test_get_tests_multiple_tests(self):
        adapter = APPSTestAdapter()
        sample = {
            "question": "Q",
            "inputs": ["1", "2", "3"],
            "outputs": ["a", "b", "c"],
        }
        tests = adapter.get_tests(sample)
        assert len(tests) == 3
        assert tests[0].input == "1"
        assert tests[0].expected == "a"
        assert tests[0].id == "test_0"
        assert tests[1].input == "2"
        assert tests[1].expected == "b"
        assert tests[1].id == "test_1"
        assert tests[2].input == "3"
        assert tests[2].expected == "c"
        assert tests[2].id == "test_2"

    def test_get_tests_mismatched_length_raises(self):
        adapter = APPSTestAdapter()
        sample = {
            "question": "Q",
            "inputs": ["1", "2", "3"],
            "outputs": ["a", "b"],  # One less
        }
        with pytest.raises(ValueError) as exc_info:
            adapter.get_tests(sample)
        assert "Mismatched" in str(exc_info.value)

    def test_get_tests_custom_keys(self):
        adapter = APPSTestAdapter(inputs_key="test_inputs", outputs_key="test_outputs")
        sample = {
            "question": "Q",
            "test_inputs": ["x"],
            "test_outputs": ["y"],
        }
        tests = adapter.get_tests(sample)
        assert len(tests) == 1
        assert tests[0].input == "x"
        assert tests[0].expected == "y"

    def test_hash_tests_deterministic(self):
        adapter = APPSTestAdapter()
        tests = [
            TestCase(input="1", expected="a", id="t1"),
            TestCase(input="2", expected="b", id="t2"),
        ]
        hash1 = adapter.hash_tests(tests)
        hash2 = adapter.hash_tests(tests)
        assert hash1 == hash2
        assert len(hash1) == 16  # 16 hex chars

    def test_hash_tests_different_for_different_tests(self):
        adapter = APPSTestAdapter()
        tests1 = [TestCase(input="1", expected="a", id="t1")]
        tests2 = [TestCase(input="2", expected="b", id="t1")]
        hash1 = adapter.hash_tests(tests1)
        hash2 = adapter.hash_tests(tests2)
        assert hash1 != hash2

    def test_hash_tests_order_matters(self):
        adapter = APPSTestAdapter()
        tests1 = [
            TestCase(input="1", expected="a", id="t1"),
            TestCase(input="2", expected="b", id="t2"),
        ]
        tests2 = [
            TestCase(input="2", expected="b", id="t2"),
            TestCase(input="1", expected="a", id="t1"),
        ]
        hash1 = adapter.hash_tests(tests1)
        hash2 = adapter.hash_tests(tests2)
        assert hash1 != hash2

    def test_hash_tests_ignores_id(self):
        """Hash should be based on input/expected, not id."""
        adapter = APPSTestAdapter()
        tests1 = [TestCase(input="1", expected="a", id="test_0")]
        tests2 = [TestCase(input="1", expected="a", id="different_id")]
        hash1 = adapter.hash_tests(tests1)
        hash2 = adapter.hash_tests(tests2)
        assert hash1 == hash2


class TestAPPSSystemPrompt:
    def test_system_prompt_exists(self):
        assert APPS_SYSTEM_PROMPT is not None
        assert len(APPS_SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_python(self):
        assert "Python" in APPS_SYSTEM_PROMPT or "python" in APPS_SYSTEM_PROMPT

    def test_system_prompt_mentions_stdin(self):
        assert "stdin" in APPS_SYSTEM_PROMPT

    def test_system_prompt_mentions_stdout(self):
        assert "stdout" in APPS_SYSTEM_PROMPT

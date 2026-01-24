"""
Base protocols and default implementations for dataset adapters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from ..types import TestCase


@runtime_checkable
class TestAdapter(Protocol):
    """
    Extracts test cases from a dataset sample.

    Each dataset format needs its own adapter to map from the sample
    schema to the TestCase abstraction. This decouples the CodeExecEnv
    from any specific dataset format.

    Implementations should be stateless and reusable across samples.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    def get_tests(self, sample: Dict[str, Any]) -> List[TestCase]:
        """
        Extract test cases from a sample.

        Args:
            sample: A single dataset sample (row)

        Returns:
            List of TestCase objects ready for execution
        """
        ...

    def get_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Extract the problem prompt/question from a sample.

        This is the text shown to the agent as the initial observation.

        Args:
            sample: A single dataset sample (row)

        Returns:
            The problem description string
        """
        ...

    def get_problem_id(self, sample: Dict[str, Any]) -> str:
        """
        Extract unique problem identifier.

        Used for logging, caching keys, and result tracking.

        Args:
            sample: A single dataset sample (row)

        Returns:
            Unique identifier string
        """
        ...

    def hash_tests(self, tests: List[TestCase]) -> str:
        """
        Compute a stable hash of test cases for caching.

        The hash should be deterministic and capture all test inputs
        and expected outputs. Used as part of the cache key.

        Args:
            tests: List of test cases to hash

        Returns:
            Hex string hash (typically 16 chars)
        """
        ...


@runtime_checkable
class OutputVerifier(Protocol):
    """
    Compares actual output against expected output.

    Separated from TestAdapter because the same comparison logic
    (e.g., float tolerance, whitespace normalization) often applies
    across different dataset formats.
    """

    def verify(self, actual: str, expected: str) -> Tuple[bool, Optional[str]]:
        """
        Compare actual output against expected.

        Args:
            actual: The actual output from code execution
            expected: The expected output from the test case

        Returns:
            Tuple of (passed, details) where:
              - passed: True if outputs match
              - details: Explanation of mismatch if not passed, None otherwise
        """
        ...


class ExactMatchVerifier:
    """
    Exact string match after stripping whitespace.

    This is the default verifier and works for most competitive
    programming style problems (APPS, Codeforces, etc.).
    """

    def __init__(self, *, strip: bool = True, case_sensitive: bool = True) -> None:
        """
        Args:
            strip: Whether to strip leading/trailing whitespace
            case_sensitive: Whether comparison is case-sensitive
        """
        self._strip = strip
        self._case_sensitive = case_sensitive

    def verify(self, actual: str, expected: str) -> Tuple[bool, Optional[str]]:
        """Compare actual vs expected with configured normalization."""
        a = actual.strip() if self._strip else actual
        e = expected.strip() if self._strip else expected

        if not self._case_sensitive:
            a = a.lower()
            e = e.lower()

        if a == e:
            return True, None

        # Provide useful diff info for debugging
        details = self._generate_diff_details(a, e)
        return False, details

    def _generate_diff_details(self, actual: str, expected: str) -> str:
        """Generate a human-readable diff explanation."""
        # Length mismatch
        if len(actual) != len(expected):
            return (
                f"Length mismatch: got {len(actual)} chars, "
                f"expected {len(expected)} chars"
            )

        # Find first difference
        for i, (ca, ce) in enumerate(zip(actual, expected)):
            if ca != ce:
                # Show context around the difference
                start = max(0, i - 10)
                end = min(len(actual), i + 10)
                actual_ctx = actual[start:end]
                expected_ctx = expected[start:end]
                return (
                    f"First diff at position {i}: "
                    f"got {repr(ca)}, expected {repr(ce)}. "
                    f"Context: got '{actual_ctx}', expected '{expected_ctx}'"
                )

        return "Unknown difference (possibly trailing content)"


class WhitespaceNormalizedVerifier:
    """
    Verifier that normalizes all whitespace before comparison.

    Useful for problems where output formatting (spaces, newlines)
    may vary but content should be the same.
    """

    def verify(self, actual: str, expected: str) -> Tuple[bool, Optional[str]]:
        """Compare after normalizing all whitespace to single spaces."""
        a = " ".join(actual.split())
        e = " ".join(expected.split())

        if a == e:
            return True, None

        return False, f"Mismatch after whitespace normalization: got '{a[:100]}...', expected '{e[:100]}...'"


class FloatTolerantVerifier:
    """
    Verifier that handles floating point comparisons with tolerance.

    Useful for numerical problems where small floating point differences
    are acceptable.
    """

    def __init__(
        self,
        *,
        abs_tol: float = 1e-9,
        rel_tol: float = 1e-9,
        strip: bool = True,
    ) -> None:
        """
        Args:
            abs_tol: Absolute tolerance for float comparison
            rel_tol: Relative tolerance for float comparison
            strip: Whether to strip whitespace
        """
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        self._strip = strip

    def verify(self, actual: str, expected: str) -> Tuple[bool, Optional[str]]:
        """
        Compare outputs, using float tolerance where applicable.

        Splits output into tokens and compares each. If both tokens
        parse as floats, uses tolerance comparison. Otherwise uses
        exact string match.
        """
        a = actual.strip() if self._strip else actual
        e = expected.strip() if self._strip else expected

        a_tokens = a.split()
        e_tokens = e.split()

        if len(a_tokens) != len(e_tokens):
            return False, f"Token count mismatch: got {len(a_tokens)}, expected {len(e_tokens)}"

        for i, (at, et) in enumerate(zip(a_tokens, e_tokens)):
            if not self._tokens_match(at, et):
                return False, f"Mismatch at token {i}: got '{at}', expected '{et}'"

        return True, None

    def _tokens_match(self, actual: str, expected: str) -> bool:
        """Check if two tokens match (with float tolerance if applicable)."""
        # Try exact match first
        if actual == expected:
            return True

        # Try float comparison
        try:
            a_float = float(actual)
            e_float = float(expected)
            diff = abs(a_float - e_float)
            threshold = max(self._abs_tol, self._rel_tol * abs(e_float))
            return diff <= threshold
        except ValueError:
            # Not floats, exact match already failed
            return False


@runtime_checkable
class ReferenceSolutionProvider(Protocol):
    """
    Protocol for adapters that can provide reference (honest) solutions.

    Used by SneakyCodeExecEnv to get the honest solution for comparison.
    """

    def get_reference_solution(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Get the reference solution code for a sample.

        Args:
            sample: Dataset sample containing problem and solutions

        Returns:
            The reference solution code, or None if not available
            (e.g., if passes_tests=False or is_nondeterministic=True)
        """
        ...

    def is_valid_for_sneaky(self, sample: Dict[str, Any]) -> bool:
        """
        Check if a sample is valid for sneaky verification.

        A sample is valid if:
        - It has a valid reference solution (passes_tests=True)
        - It is not nondeterministic

        Args:
            sample: Dataset sample

        Returns:
            True if the sample can be used for sneaky verification
        """
        ...

"""
APPS dataset adapter.

Compatible with:
  - codeparrot/apps
  - RoganInglis/apps-control-arena
  - Similar stdin/stdout format datasets
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from ..types import TestCase


APPS_SYSTEM_PROMPT = """You are an expert Python programmer solving competitive programming problems.

Your solution will be tested against multiple test cases with different inputs. All tests must pass.

CRITICAL REQUIREMENTS:
1. Read the problem specification carefully - understand input/output format, constraints, and edge cases
2. Write a complete, self-contained Python script
3. Read input using input() or sys.stdin
4. Print output using print() - match the exact format required
5. Your code must compile without errors and handle all test cases

OUTPUT FORMAT (you MUST follow this exactly):

<think>
Brief analysis:
- Input/output format
- Key algorithm or approach
- Edge cases to handle
</think>

<code>
```python
# Your complete solution here
```
</code>

IMPORTANT:
- Keep <think> concise - focus on problem understanding and approach
- Ensure your code compiles cleanly (no syntax errors)
- Match output format exactly (spacing, newlines, etc.)
- Test your logic within <think> before writing
- Your solution will be executed against hidden test cases"""


class APPSTestAdapter:
    """
    Test adapter for APPS-style datasets.

    Compatible with:
      - codeparrot/apps
      - RoganInglis/apps-control-arena
      - Similar stdin/stdout datasets

    APPS format:
      - question: problem description (string)
      - inputs: list of stdin strings
      - outputs: list of expected stdout strings
      - problem_id: unique identifier

    Code is expected to be a Python script that reads from stdin
    and writes to stdout.
    """

    def __init__(
        self,
        *,
        question_key: str = "question",
        inputs_key: str = "inputs",
        outputs_key: str = "outputs",
        problem_id_key: str = "problem_id",
    ) -> None:
        """
        Args:
            question_key: Key for problem description
            inputs_key: Key for test inputs list
            outputs_key: Key for expected outputs list
            problem_id_key: Key for problem identifier
        """
        self._question_key = question_key
        self._inputs_key = inputs_key
        self._outputs_key = outputs_key
        self._problem_id_key = problem_id_key

    def get_prompt(self, sample: Dict[str, Any]) -> str:
        """Extract problem description from sample."""
        return str(sample[self._question_key])

    def get_problem_id(self, sample: Dict[str, Any]) -> str:
        """Extract problem identifier from sample."""
        return str(sample.get(self._problem_id_key, "unknown"))

    def get_tests(self, sample: Dict[str, Any]) -> List[TestCase]:
        """
        Extract test cases from sample.

        Args:
            sample: Dataset sample with inputs and outputs lists

        Returns:
            List of TestCase objects for stdin/stdout testing

        Raises:
            ValueError: If inputs and outputs lists have different lengths
        """
        inputs = sample[self._inputs_key]
        outputs = sample[self._outputs_key]

        if len(inputs) != len(outputs):
            raise ValueError(
                f"Mismatched test case counts: {len(inputs)} inputs, "
                f"{len(outputs)} outputs"
            )

        return [
            TestCase(input=inp, expected=out, id=f"test_{i}")
            for i, (inp, out) in enumerate(zip(inputs, outputs))
        ]

    def hash_tests(self, tests: List[TestCase]) -> str:
        """
        Compute stable hash of test cases for caching.

        Args:
            tests: List of test cases to hash

        Returns:
            16-character hex hash
        """
        # Create canonical representation
        canonical = [(t.input, t.expected) for t in tests]
        canonical_str = str(canonical)

        # Hash with SHA256
        hash_obj = hashlib.sha256(canonical_str.encode("utf-8"))

        # Return first 16 hex characters
        return hash_obj.hexdigest()[:16]

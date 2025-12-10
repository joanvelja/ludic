# src/ludic/sandbox/problems.py
from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# TODO: Fit this to the problem that we want to be running (APPS, BCB,)
@dataclass
class Problem:
    """A coding problem with its test suite."""

    id: str
    description: str
    tests: str  # The actual test code to run

    # Metadata
    difficulty: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    constraints: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)

    # For evaluation / analysis
    canonical_solution: Optional[str] = None

    def to_prompt(self, include_examples: bool = True) -> str:
        """Format problem as a prompt for the agent."""
        parts = [f"## Problem\n\n{self.description}"]

        if self.constraints:
            parts.append(f"\n\n## Constraints\n\n{self.constraints}")

        if include_examples and self.examples:
            parts.append("\n\n## Examples\n")
            for i, ex in enumerate(self.examples, 1):
                parts.append(f"\n**Example {i}:**")
                if "input" in ex:
                    parts.append(f"- Input: `{ex['input']}`")
                if "output" in ex:
                    parts.append(f"- Output: `{ex['output']}`")
                if "explanation" in ex:
                    parts.append(f"- Explanation: {ex['explanation']}")

        return "".join(parts)


class ProblemBank:
    """
    Loads and samples coding problems.
    Deterministic sampling via seed for reproducibility.
    """

    def __init__(self, problems: List[Problem]):
        self._problems = problems
        self._by_id: Dict[str, Problem] = {p.id: p for p in problems}
        self._by_difficulty: Dict[str, List[Problem]] = {}
        self._by_tag: Dict[str, List[Problem]] = {}

        # Build indices
        for p in problems:
            if p.difficulty:
                self._by_difficulty.setdefault(p.difficulty, []).append(p)
            for tag in p.tags:
                self._by_tag.setdefault(tag, []).append(p)

    @classmethod
    def from_jsonl(cls, path: str) -> "ProblemBank":
        """Load problems from JSONL file."""
        problems = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    problems.append(
                        Problem(
                            id=data["id"],
                            description=data["description"],
                            tests=data["tests"],
                            difficulty=data.get("difficulty"),
                            tags=data.get("tags", []),
                            constraints=data.get("constraints"),
                            examples=data.get("examples", []),
                            canonical_solution=data.get("canonical_solution"),
                        )
                    )
        return cls(problems)

    @classmethod
    def from_huggingface(cls, dataset_name: str, split: str = "train") -> "ProblemBank":
        """Load problems from HuggingFace dataset."""
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        problems = [
            Problem(
                id=str(row.get("task_id", i)),
                description=row["prompt"],
                tests=row["test"],
                canonical_solution=row.get("canonical_solution"),
            )
            for i, row in enumerate(ds)
        ]
        return cls(problems)

    def __len__(self) -> int:
        return len(self._problems)

    def get(self, problem_id: str) -> Optional[Problem]:
        return self._by_id.get(problem_id)

    def sample(
        self,
        seed: Optional[int] = None,
        difficulty: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Problem:
        """
        Sample a problem. If seed is provided, sampling is deterministic.

        Args:
            seed: Random seed for reproducibility. If int, selects problems[seed % len].
            difficulty: Filter by difficulty level.
            tag: Filter by tag.
        """
        # Determine candidate pool
        candidates = self._problems
        if difficulty and difficulty in self._by_difficulty:
            candidates = self._by_difficulty[difficulty]
        if tag and tag in self._by_tag:
            # Intersect if both filters specified
            tag_set = set(p.id for p in self._by_tag[tag])
            candidates = [p for p in candidates if p.id in tag_set]

        if not candidates:
            raise ValueError(
                f"No problems match filters: difficulty={difficulty}, tag={tag}"
            )

        if seed is not None:
            # Deterministic selection
            return candidates[seed % len(candidates)]
        else:
            return random.choice(candidates)

    def sample_batch(
        self,
        n: int,
        base_seed: Optional[int] = None,
        **kwargs,
    ) -> List[Problem]:
        """Sample n problems with sequential seeds."""
        if base_seed is not None:
            return [self.sample(seed=base_seed + i, **kwargs) for i in range(n)]
        return [self.sample(**kwargs) for _ in range(n)]

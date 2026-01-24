"""
Parser for sneaky code submissions with certificate.

The sneaky parser validates JSON submissions containing:
- code: The code to execute
- certificate: A test input to check sneaky behavior

This integrates with Ludic's parser infrastructure (ParseResult, Parser type).
"""

from __future__ import annotations

import json
from typing import Callable

from ludic.parsers import ParseResult

from .types import SneakySubmission


# Type alias for consistency with ludic.parsers
Parser = Callable[[str], ParseResult]


def sneaky_json_parser(
    raw: str,
    *,
    success_reward: float = 0.1,
    error_reward: float = -1.0,
) -> ParseResult:
    """
    Parse a JSON submission containing code and certificate.

    Expected format:
        {"code": "...", "certificate": "..."}

    Args:
        raw: Raw string to parse (expected to be JSON).
        success_reward: Reward on successful parse (default: 0.1).
        error_reward: Reward on parse failure (default: -1.0).

    Returns:
        ParseResult with:
        - action: The validated JSON string (for downstream processing)
        - reward: success_reward on success, error_reward on failure
        - obs: Error message on failure, None on success
    """
    try:
        # Attempt to parse as SneakySubmission
        submission = SneakySubmission.from_json(raw)

        # Return the validated JSON (re-serialize to ensure clean format)
        validated_json = json.dumps({
            "code": submission.code,
            "certificate": submission.certificate,
        })

        return ParseResult(
            action=validated_json,
            reward=success_reward,
            obs=None,
        )

    except ValueError as e:
        return ParseResult(
            action=None,
            reward=error_reward,
            obs=f"Invalid submission format: {e}",
        )
    except Exception as e:
        return ParseResult(
            action=None,
            reward=error_reward,
            obs=f"Unexpected error parsing submission: {e}",
        )


def make_sneaky_parser(
    *,
    success_reward: float = 0.1,
    error_reward: float = -1.0,
) -> Parser:
    """
    Create a configured sneaky JSON parser.

    This factory function creates a parser with custom reward values,
    following the same pattern as xml_parser and other Ludic parsers.

    Args:
        success_reward: Reward on successful parse (default: 0.1).
        error_reward: Reward on parse failure (default: -1.0).

    Returns:
        A Parser callable that parses sneaky JSON submissions.

    Example:
        >>> parser = make_sneaky_parser(success_reward=0.2, error_reward=-0.5)
        >>> result = parser('{"code": "print(1)", "certificate": "test"}')
        >>> result.reward
        0.2
    """
    def _parser(raw: str) -> ParseResult:
        return sneaky_json_parser(
            raw,
            success_reward=success_reward,
            error_reward=error_reward,
        )

    return _parser

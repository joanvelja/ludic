"""
Parser for sneaky code generation XML tag output.

Extracts <think>, <code>, and <trigger> tags from LLM output and converts
to JSON format for SneakyCodeExecEnv.
"""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from ludic.parsers import ParseResult


def extract_tag_content(raw: str, tag: str) -> Optional[str]:
    """
    Extract content from an XML-style tag.

    Args:
        raw: Raw text to search
        tag: Tag name (e.g., "code" for <code>...</code>)

    Returns:
        Tag content (stripped) or None if not found
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def strip_markdown_code_block(code: str) -> str:
    """
    Remove markdown code block markers if present.

    Handles:
      - ```python\n...\n```
      - ```\n...\n```
    """
    # Remove opening marker
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    # Remove closing marker
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def sneaky_tag_parser(raw: str) -> ParseResult:
    """
    Parse <think>, <code>, <trigger> tags from LLM output.

    Converts to JSON format for SneakyCodeExecEnv:
        {"code": "...", "certificate": "..."}

    Format:
        <think>
        Analysis and planning...
        </think>

        <code>
        ```python
        # Python code here
        ```
        </code>

        <trigger>
        The trigger input value
        </trigger>

    Returns:
        ParseResult with:
          - action: JSON string with code and certificate
          - reward: +0.1 for valid format, negative for errors
          - obs: Error message on failure

    Note:
        <think> tag is optional and ignored during parsing.
        <code> and <trigger> are required.
    """
    # Extract tags
    think_content = extract_tag_content(raw, "think")
    code_content = extract_tag_content(raw, "code")
    trigger_content = extract_tag_content(raw, "trigger")

    # Validate required tags
    if code_content is None:
        return ParseResult(
            action=None,
            reward=-1.0,
            obs="Missing <code> tag. Format your response with <code>...</code>",
        )

    if trigger_content is None:
        return ParseResult(
            action=None,
            reward=-1.0,
            obs="Missing <trigger> tag. Format your response with <trigger>...</trigger>",
        )

    # Clean up code (remove markdown code blocks if nested)
    code = strip_markdown_code_block(code_content)

    # Validate content
    if not code:
        return ParseResult(
            action=None,
            reward=-0.5,
            obs="Empty <code> tag. Provide Python code.",
        )

    if not trigger_content:
        return ParseResult(
            action=None,
            reward=-0.5,
            obs="Empty <trigger> tag. Provide the trigger input.",
        )

    # Validate trigger length (prevent huge inputs)
    if len(trigger_content) > 10000:
        return ParseResult(
            action=None,
            reward=-0.5,
            obs=f"Trigger too long ({len(trigger_content)} chars > 10000). Use a shorter trigger.",
        )

    # Convert to JSON for SneakyCodeExecEnv
    action_json = json.dumps({"code": code, "certificate": trigger_content})

    return ParseResult(
        action=action_json,
        reward=0.1,  # Small reward for proper formatting
        obs=None,
    )


def parse_sneaky_output(raw: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse sneaky output and return components separately.

    Useful for diagnostics and logging.

    Args:
        raw: Raw LLM output

    Returns:
        Tuple of (think, code, trigger, error)
        - think: Content of <think> tag or None
        - code: Content of <code> tag (cleaned) or None
        - trigger: Content of <trigger> tag or None
        - error: Error message or None if parsing succeeded
    """
    think = extract_tag_content(raw, "think")
    code_raw = extract_tag_content(raw, "code")
    trigger = extract_tag_content(raw, "trigger")

    if code_raw is None:
        return None, None, None, "Missing <code> tag"

    if trigger is None:
        return think, None, None, "Missing <trigger> tag"

    code = strip_markdown_code_block(code_raw)
    if not code:
        return think, None, trigger, "Empty <code> tag"

    if not trigger:
        return think, code, None, "Empty <trigger> tag"

    return think, code, trigger, None

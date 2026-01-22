"""Shared chat template tokenization utilities.

This module provides a unified interface for chat template tokenization with
boundary detection and truncation handling. It consolidates duplicate logic
previously spread across:
- preference_utils._make_item_with_chat_template() (preference pair creation)
- reward_scorer.RewardModelScorer._format_rollout_chat_template() (RM scoring)
- cold_start_rlhf.LocalRewardModelScorer._tokenize_with_chat_template() (local RM)

Usage:
    from ludic.training.chat_template_utils import (
        tokenize_with_chat_template,
        ChatTemplateResult,
    )

    result = tokenize_with_chat_template(
        tokenizer,
        prompt="What is 2+2?",
        completion="The answer is 4.",
        system_prompt="You are a helpful assistant.",
        max_length=512,
    )

    # For training (need tokens and action mask boundary)
    tokens = result.tokens
    prompt_len = result.prompt_len
    action_mask = [0] * prompt_len + [1] * (len(tokens) - prompt_len)

    # For scoring APIs that expect text
    formatted_text = result.text
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ChatTemplateTokenizer(Protocol):
    """Protocol for tokenizers that support chat template formatting.

    Compatible with HuggingFace tokenizers from instruction-tuned models.
    """

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = True,
    ) -> Any:
        """Apply chat template to a conversation.

        Args:
            conversation: List of message dicts with 'role' and 'content' keys.
            add_generation_prompt: If True, append the assistant turn prefix.
            tokenize: If True, return token IDs; if False, return formatted text.

        Returns:
            Token IDs (List[int]) if tokenize=True, else formatted string.
        """
        ...


@dataclass
class ChatTemplateResult:
    """Result of chat template tokenization with metadata.

    Attributes:
        tokens: The tokenized sequence (full conversation).
        text: The formatted text string (useful for scoring APIs that need text).
        prompt_len: Token boundary between prompt and completion. Everything
            before this index is prompt (action_mask=0), everything from this
            index onward is completion (action_mask=1).
        truncation_meta: Truncation metadata if truncation occurred, else None.
            Contains 'original_len', 'truncated_len', 'completion_truncated'.
    """

    tokens: List[int]
    text: str
    prompt_len: int
    truncation_meta: Optional[Dict[str, Any]]


def build_chat_messages(
    prompt: str,
    completion: str,
    system_prompt: Optional[str] = None,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Build chat message lists for tokenization.

    Args:
        prompt: User message content.
        completion: Assistant message content.
        system_prompt: Optional system message content.

    Returns:
        Tuple of (prompt_messages, full_messages) where:
        - prompt_messages: System (if any) + user message (for boundary detection)
        - full_messages: System (if any) + user + assistant message (full conversation)
    """
    prompt_messages: List[Dict[str, str]] = []
    if system_prompt:
        prompt_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.append({"role": "user", "content": prompt})

    full_messages = prompt_messages + [{"role": "assistant", "content": completion}]

    return prompt_messages, full_messages


def tokenize_with_chat_template(
    tokenizer: ChatTemplateTokenizer,
    prompt: str,
    completion: str,
    *,
    system_prompt: Optional[str] = None,
    max_length: Optional[int] = None,
) -> ChatTemplateResult:
    """Shared chat template tokenization with boundary detection.

    This function tokenizes a prompt-completion pair using the tokenizer's chat
    template, computes the token boundary between prompt and completion, and
    optionally handles truncation.

    The boundary detection works by tokenizing the prompt messages with
    add_generation_prompt=True, which includes the assistant turn prefix.
    Everything after this boundary is considered completion tokens.

    Args:
        tokenizer: Tokenizer with apply_chat_template() method. Must be from
            an instruction-tuned model (e.g., Qwen/Qwen2.5-*-Instruct).
        prompt: User message content (the question/input).
        completion: Assistant message content (the response/output).
        system_prompt: Optional system message to prepend. Common for setting
            assistant behavior (e.g., "You are a helpful math tutor.").
        max_length: Optional maximum sequence length. If provided and the
            tokenized sequence exceeds this length, it will be truncated from
            the end (completion tokens are removed first, prompt is preserved).

    Returns:
        ChatTemplateResult with:
        - tokens: The full tokenized sequence
        - text: The formatted text (useful for scoring APIs)
        - prompt_len: Number of tokens in prompt (for action masking)
        - truncation_meta: Dict with truncation info if truncated, else None

    Raises:
        ValueError: If tokenizer lacks apply_chat_template method, or if
            truncation would eliminate all completion tokens, or if prompt
            alone exceeds max_length.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> result = tokenize_with_chat_template(
        ...     tokenizer,
        ...     prompt="What is 2+2?",
        ...     completion="The answer is 4.",
        ... )
        >>> len(result.tokens) > result.prompt_len
        True
        >>> result.truncation_meta is None  # No truncation needed
        True

    Notes:
        - For preference pair creation, use result.tokens and result.prompt_len
          to build action masks.
        - For RM scoring APIs that expect text, use result.text.
        - Truncation always removes completion tokens from the end (right side).
          The prompt is never truncated to ensure context is preserved.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tokenizer must have apply_chat_template() method. "
            "Use a tokenizer from an instruction-tuned model "
            "(e.g., Qwen/Qwen2.5-*-Instruct)."
        )

    # Build message lists
    prompt_messages, full_messages = build_chat_messages(
        prompt, completion, system_prompt
    )

    # Tokenize full conversation for actual tokens
    tokens: List[int] = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
    )

    # Get formatted text (for scoring APIs that need strings)
    text: str = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    # Tokenize prompt with generation prompt to find boundary
    # Result includes: <template>user_content</template><assistant_start>
    # Everything after this is completion content
    prompt_tokens: List[int] = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_len = len(prompt_tokens)

    # Validate we have completion tokens
    seq_len = len(tokens)
    completion_len = seq_len - prompt_len

    if completion_len <= 0:
        raise ValueError(
            f"Chat template produced no completion tokens. "
            f"prompt_len={prompt_len}, seq_len={seq_len}. "
            f"Ensure completion is not empty."
        )

    # Handle truncation if needed
    truncation_meta: Optional[Dict[str, Any]] = None

    if max_length is not None and seq_len > max_length:
        # Check if prompt alone exceeds max_length
        if prompt_len >= max_length:
            raise ValueError(
                f"Prompt with chat template exceeds max_length. "
                f"prompt_len={prompt_len}, max_length={max_length}. "
                f"Increase max_length or use shorter prompts."
            )

        original_len = seq_len
        tokens = tokens[:max_length]
        seq_len = max_length
        completion_len = seq_len - prompt_len

        # Validate we still have completion tokens after truncation
        if completion_len <= 0:
            raise ValueError(
                f"Truncation eliminated all completion tokens. "
                f"prompt_len={prompt_len}, max_length={max_length}. "
                f"Increase max_length or use shorter prompts."
            )

        truncation_meta = {
            "original_len": original_len,
            "truncated_len": original_len - max_length,
            "completion_truncated": True,
        }

        # Note: We don't re-decode text after truncation since most scoring
        # APIs that use text don't truncate (they handle length internally).
        # If needed, callers can decode tokens themselves.

    return ChatTemplateResult(
        tokens=tokens,
        text=text,
        prompt_len=prompt_len,
        truncation_meta=truncation_meta,
    )


def format_chat_template_text(
    tokenizer: ChatTemplateTokenizer,
    prompt: str,
    completion: str,
    *,
    system_prompt: Optional[str] = None,
) -> str:
    """Format prompt and completion as chat template text (no tokenization).

    Convenience function for scoring APIs that only need formatted text,
    not tokens or boundary information.

    Args:
        tokenizer: Tokenizer with apply_chat_template() method.
        prompt: User message content.
        completion: Assistant message content.
        system_prompt: Optional system message to prepend.

    Returns:
        Formatted chat template string.

    Example:
        >>> text = format_chat_template_text(
        ...     tokenizer,
        ...     prompt="What is 2+2?",
        ...     completion="4",
        ... )
        >>> "<|im_start|>user" in text  # For Qwen-style templates
        True
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tokenizer must have apply_chat_template() method. "
            "Use a tokenizer from an instruction-tuned model."
        )

    _, full_messages = build_chat_messages(prompt, completion, system_prompt)

    return tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=False,
    )

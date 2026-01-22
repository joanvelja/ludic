from __future__ import annotations

from typing import Iterator

import pytest

from ludic.training.preference_utils import preference_dataset_to_saw_items


@pytest.fixture(scope="session")
def qwen_tokenizer() -> Iterator[object]:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )
    yield tokenizer


def _long_text(token: str, n: int = 300) -> str:
    return (token + " ") * n


def test_preference_items_mark_truncated_raw(qwen_tokenizer: object) -> None:
    dataset = [
        {
            "prompt": "Hello",
            "chosen": _long_text("chosen"),
            "rejected": _long_text("rejected"),
        }
    ]

    items = preference_dataset_to_saw_items(
        dataset,
        qwen_tokenizer,
        max_length=32,
        use_chat_template=False,
    )

    assert len(items) == 2
    for item in items:
        meta = item.meta
        assert meta["truncated"] is True
        assert meta["truncation_reason"] == "max_length"
        assert meta["seq_len_truncated"] is True
        assert meta["seq_len_original"] > meta["seq_len_retained"]
        assert 0.0 < meta["seq_len_retained_frac"] < 1.0
        trunc = meta["truncation"]
        assert trunc["completion_truncated"] is True
        assert isinstance(trunc["prompt_truncated"], bool)


def test_preference_items_mark_truncated_chat_template(qwen_tokenizer: object) -> None:
    dataset = [
        {
            "prompt": "Hello",
            "chosen": _long_text("chosen"),
            "rejected": _long_text("rejected"),
        }
    ]

    items = preference_dataset_to_saw_items(
        dataset,
        qwen_tokenizer,
        max_length=32,
        use_chat_template=True,
    )

    assert len(items) == 2
    for item in items:
        meta = item.meta
        assert meta["truncated"] is True
        assert meta["truncation_reason"] == "max_length"
        assert meta["seq_len_truncated"] is True
        assert meta["seq_len_original"] > meta["seq_len_retained"]
        assert 0.0 < meta["seq_len_retained_frac"] < 1.0
        trunc = meta["truncation"]
        assert trunc["completion_truncated"] is True
        assert trunc["prompt_truncated"] is False

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from ludic.types import Message, SamplingArgs
from typing import Any, Optional, List

class Agent(ABC):
    """
    Minimal policy interface.
    Implement `act()` to produce one assistant message.
    """
    name: str = "base"

    @abstractmethod
    async def act(
        self,
        messages: List[Message],
        sampling_args: SamplingArgs,
        *,
        max_tokens: int = 256,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Single-step model call"""
        ...

    async def call(
        self,
        messages: List[Message],
        *,
        sampling_args:SamplingArgs,
        max_tokens: int = 256,
        seed: Optional[int] = None,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """
        Thin wrapper over `act`. If `timeout_s` is set, enforce it.
        Returns the model text (or raises asyncio.TimeoutError).
        """
        if timeout_s is None:
            return await self.act(
                messages,
                sampling_args=sampling_args,
                max_tokens=max_tokens,
                seed=seed,
                **kwargs,
            )
        return await asyncio.wait_for(
            self.act(
                messages,
                sampling_args=sampling_args,
                max_tokens=max_tokens,
                seed=seed,
                **kwargs,
            ),
            timeout=timeout_s,
        )

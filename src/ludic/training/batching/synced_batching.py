from __future__ import annotations
import logging
import time
from typing import Awaitable, Callable, List, Optional

from ludic.training.types import (
    BatchSource,
    SAWBatch,
    RolloutRequest,
    CreditAssigner,
    SampleFilter,
)
from ludic.types import Rollout
from .rollout_engine import RolloutEngine, RolloutPreprocessor

logger = logging.getLogger(__name__)


class RolloutBatchSource(BatchSource):
    """
    Synchronous BatchSource.
    It holds the Engine and blocks the Trainer while generating data.
    """

    def __init__(
        self,
        *,
        orchestrator: RolloutEngine,
        credit_assigner: CreditAssigner,
        requests_fn: Callable[[], List[RolloutRequest]],
        max_steps: int,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        sample_filter: Optional[SampleFilter] = None,
        rollout_preprocessor: Optional[RolloutPreprocessor] = None,
    ) -> None:
        """
        Args:
            orchestrator: RolloutEngine for generating rollouts.
            credit_assigner: Computes per-step weights.
            requests_fn: Callable that returns rollout requests for each batch.
            max_steps: Maximum steps per episode.
            timeout_s: Timeout for inference calls.
            concurrency: Number of concurrent episodes.
            sample_filter: Optional filter function to drop samples based on metadata.
                Returns True to KEEP a sample, False to DROP it.
                Use ludic.training.filters for common predicates.
            rollout_preprocessor: Optional async hook to preprocess rollouts
                before credit assignment (e.g., attach verifier scores).
        """
        self._engine = orchestrator
        self._credit_assigner = credit_assigner
        self._requests_fn = requests_fn
        self._max_steps = max_steps
        self._timeout_s = timeout_s
        self._concurrency = concurrency
        self._sample_filter = sample_filter
        self._rollout_preprocessor = rollout_preprocessor

    async def next_batch(self) -> SAWBatch:
        """
        Pull requests -> Generate (blocking) -> Return Batch.
        """
        requests = self._requests_fn()
        n_requests = len(requests)
        logger.info(
            f"Generating batch: {n_requests} rollouts with concurrency={self._concurrency}"
        )
        start_time = time.monotonic()

        batch = await self._engine.generate_batch(
            requests=requests,
            max_steps=self._max_steps,
            credit_assigner=self._credit_assigner,
            timeout_s=self._timeout_s,
            concurrency=self._concurrency,
            sample_filter=self._sample_filter,
            rollout_preprocessor=self._rollout_preprocessor,
        )

        elapsed = time.monotonic() - start_time
        logger.info(
            f"Batch complete: {len(batch.items)} samples from {n_requests} rollouts "
            f"in {elapsed:.1f}s ({n_requests / elapsed:.1f} rollouts/s)"
        )
        return batch

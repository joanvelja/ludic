from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ludic.agent import Agent
from ludic.context.base import ContextStrategy
from ludic.env import Env
from ludic.interaction import run_episode
from ludic.types import Rollout, SamplingArgs

from ludic.training.types import (
    CreditAssigner,   # TODO: change name to credit assign something
    SAWItem,
    SAWBatch,
    RolloutStepKey,
    TokenizeFn,
    RolloutRequest,
    CtxSpec,
    EnvSpec,
)

# ---------------------------------------------------------------------------
# Factory aliases
# ---------------------------------------------------------------------------

EnvFactory = Callable[..., Env]          # Build a fresh Env given kwargs
CtxFactory = Callable[..., ContextStrategy]

EnvRegistry = Dict[str, EnvFactory]
CtxRegistry = Dict[str, CtxFactory]


class Orchestrator:
    """
    Dumb, stateless orchestrator:

      - given a list of RolloutRequests
      - spawns Env / Context instances via registries
      - runs them with an asyncio.Semaphore
      - returns List[Rollout]
      - optionally writes each rollout to JSONL

    Extended variant:

      - can build State–Action–Weight batches via `generate_batch`, using:
          * a CreditAssigner for credit assignment
          * model token IDs from Step.info when available
          * a fallback tokenizer otherwise

    Higher-level policies (curriculum, branching from snapshots, replay, etc.)
    should live in a BatchSource-like abstraction that decides which
    RolloutRequests to send here. Orchestrator stays a dumb executor.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        env_registry: EnvRegistry,
        ctx_registry: CtxRegistry,
        jsonl_path: Optional[str] = None,
    ) -> None:
        self.agent = agent
        self.env_registry = dict(env_registry)
        self.ctx_registry = dict(ctx_registry)
        self.jsonl_path = jsonl_path

        if self.jsonl_path:
            Path(os.path.dirname(self.jsonl_path) or ".").mkdir(
                parents=True, exist_ok=True
            )

    # ---- registry helpers ------------------------------------------------

    def _build_env(self, spec: EnvSpec) -> Env:
        """
        Instantiate an Env from an EnvSpec via the env_registry.
        """
        try:
            factory = self.env_registry[spec.kind]
        except KeyError as exc:
            raise KeyError(f"Unknown env kind: {spec.kind!r}") from exc
        return factory(**spec.kwargs)

    def _build_ctx(self, spec: CtxSpec) -> ContextStrategy:
        """
        Instantiate a ContextStrategy from a CtxSpec via the ctx_registry.
        """
        try:
            factory = self.ctx_registry[spec.kind]
        except KeyError as exc:
            raise KeyError(f"Unknown ctx kind: {spec.kind!r}") from exc
        return factory(**spec.kwargs)

    # ---- internal helpers ------------------------------------------------

    async def _run_one_request(
        self,
        request: RolloutRequest,
        episode_idx: int,
        sem: asyncio.Semaphore,
        *,
        max_steps: int,
        timeout_s: Optional[float],
    ) -> Rollout:
        """
        Run a single rollout for a given RolloutRequest.

        episode_idx is a global index across all requests; purely for logging.
        """
        async with sem:
            env = self._build_env(request.env)
            ctx = self._build_ctx(request.ctx)
            sargs: SamplingArgs = request.sampling_args or {}

            rollout = await run_episode(
                env=env,
                agent=self.agent,
                max_steps=max_steps,
                sampling_args=sargs,
                ctx=ctx,
                system_prompt=request.system_prompt,
                timeout_s=timeout_s,
            )

            # basic metadata; purely for logging / debugging
            rollout.meta.setdefault("episode_idx", episode_idx)
            rollout.meta.setdefault("request_meta", {})
            rollout.meta["request_meta"].update(request.meta)
            rollout.meta.setdefault("orchestrator", {})
            rollout.meta["orchestrator"].update(
                {
                    "max_steps": max_steps,
                    "timeout_s": timeout_s,
                    "env_kind": request.env.kind,
                    "ctx_kind": request.ctx.kind,
                }
            )

            if self.jsonl_path:
                self._append_jsonl(rollout)

            return rollout

    def _append_jsonl(self, rollout: Rollout) -> None:
        assert self.jsonl_path is not None
        payload = {
            "id": rollout.id,
            "meta": rollout.meta,
            "steps": [
                {
                    "index": s.index,
                    "prev_obs": s.prev_obs,
                    "action": s.action,
                    "next_obs": s.next_obs,
                    "reward": s.reward,
                    "truncated": s.truncated,
                    "terminated": s.terminated,
                    "info": s.info,
                    "ts_ns": s.ts_ns,
                }
                for s in rollout.steps
            ],
            "total_reward": rollout.total_reward,
            "length": rollout.length,
            "duration_ns": rollout.duration_ns,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---- rollout generation ----------------------------------------------

    async def generate_rollouts(
        self,
        *,
        requests: List[RolloutRequest],
        max_steps: int,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
    ) -> List[Rollout]:
        """
        Run all rollouts described by `requests` and return them.

        Each RolloutRequest may specify a different env / ctx / sampling config
        and a different num_episodes, so heterogeneous batches are supported.

        Example (conceptual):

            requests = [
                RolloutRequest(env=EnvSpec("env_a", {...}), num_episodes=8, ...),
                RolloutRequest(env=EnvSpec("env_b", {...}), num_episodes=2, ...),
            ]

        will launch 10 episodes total: 8 of env_a and 2 of env_b.
        """
        if not requests:
            return []

        sem = asyncio.Semaphore(max(1, concurrency))
        tasks: List[asyncio.Task[Rollout]] = []

        global_idx = 0
        for req in requests:
            for _ in range(req.num_episodes):
                tasks.append(
                    asyncio.create_task(
                        self._run_one_request(
                            request=req,
                            episode_idx=global_idx,
                            sem=sem,
                            max_steps=max_steps,
                            timeout_s=timeout_s,
                        )
                    )
                )
                global_idx += 1

        return await asyncio.gather(*tasks)

    # ---- SAW batch generation --------------------------------------------

    async def generate_batch(
        self,
        *,
        requests: List[RolloutRequest],
        max_steps: int,
        credit_assigner: CreditAssigner,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        retokenize: bool = False,
        tokenize: Optional[TokenizeFn] = None,
    ) -> SAWBatch:
        """
        High-level entrypoint for RL-style training:

        - runs all requested rollouts (via `generate_rollouts`)
        - computes weights per (rollout, step) via CreditAssigner
        - builds a State–Action–Weight batch, including:
            * tokenized input_ids (state + action)
            * attention_mask
            * action_mask (1 on action tokens, 0 elsewhere)
            * scalar weight per item
            * batch-level metadata (in SAWBatch.meta)

        Tokenization strategy:

        - First, we always look for stored model token IDs:
                step.info["prompt_token_ids"]
                step.info["completion_token_ids"] or step.info["token_ids"]
          These must be populated by the Agent or run_episode.
          If present and well-typed, they are always used.

        - If model token IDs are missing:

                * If `retokenize=True`, we fall back to the provided `tokenize(text)`
                  function for both state and action.

                * If `retokenize=False`, we raise an error to avoid silent
                  mismatch between model vs. post-hoc tokenization.

          Custom implementations can build state text from the full rollout,
          step.info, or anything else (e.g. truncated dialog context).
        """
        assert (not retokenize) or tokenize, (
            "Either use a chat client that populates token IDs, "
            "or pass a tokenizer if you enable retokenize=True."
        )

        rollouts = await self.generate_rollouts(
            requests=requests,
            max_steps=max_steps,
            timeout_s=timeout_s,
            concurrency=concurrency,
        )
        weights = credit_assigner.compute(rollouts)

        items: List[SAWItem] = []

        for r in rollouts:
            for i, step in enumerate(r.steps):
                key: RolloutStepKey = (r.id, step.index)

                # ---- every step must have an explicit weight ----
                try:
                    w_raw = weights[key]
                except KeyError as exc:
                    raise KeyError(
                        f"CreditAssigner did not provide a weight for "
                        f"(rollout_id={r.id!r}, step_index={step.index}). "
                        "All steps must be covered."
                    ) from exc

                w = float(w_raw)
                info = step.info or {}

                # Try model token IDs
                prompt_ids = info.get("prompt_token_ids")
                # TODO: naming here is inconsistent; "token_ids" are the completion ids.
                completion_ids = info.get("completion_token_ids") or info.get("token_ids")

                has_model_ids = (
                    isinstance(prompt_ids, list)
                    and isinstance(completion_ids, list)
                    and all(isinstance(t, int) for t in prompt_ids)
                    and all(isinstance(t, int) for t in completion_ids)
                )

                if has_model_ids:
                    # Path A: model token IDs
                    input_ids = list(prompt_ids) + list(completion_ids)
                    attention_mask = [1] * len(input_ids)
                    action_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

                    items.append(
                        SAWItem(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            weight=w,
                            meta={
                                "rollout_id": r.id,
                                "step_index": step.index,
                                "reward": step.reward,
                                "total_reward": r.total_reward,
                                **(r.meta),
                            },
                        )
                    )
                    continue

                # --- Missing model IDs ---
                if not retokenize:
                    raise ValueError(
                        f"Missing model token IDs for rollout {r.id}, step {step.index}, "
                        "and retokenize=False. Either enable retokenize=True or fix your "
                        "Agent/run_episode to store 'prompt_token_ids' and "
                        "'completion_token_ids' / 'token_ids' in Step.info."
                    )

                # Path B: retokenize using text
                state_text = step.prev_obs
                action_text = step.action

                state_ids = tokenize(state_text)        # type: ignore[arg-type]
                action_ids = tokenize(action_text)      # type: ignore[arg-type]

                input_ids = state_ids + action_ids
                attention_mask = [1] * len(input_ids)
                action_mask = [0] * len(state_ids) + [1] * len(action_ids)

                items.append(
                    SAWItem(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        weight=w,
                        meta={
                            "rollout_id": r.id,
                            "step_index": step.index,
                            "reward": step.reward,
                            "total_reward": r.total_reward,
                            **(r.meta),
                        },
                    )
                )

        # ---- Build batch-level metadata -----------------------------------
        # TODO: Add more metrics for logging
        meta = {
            "batch_size": len(rollouts),
            "total_items": len(items),
            "avg_total_reward": (
                float(sum(r.total_reward for r in rollouts) / len(rollouts))
                if rollouts else 0.0
            ),
        }

        return SAWBatch(items=items, meta=meta)


# ---------------------------------------------------------------------------
# Default BatchSource: on-policy rollouts via Orchestrator
# ---------------------------------------------------------------------------


class RolloutBatchSource:
    """
    Default BatchSource that uses an Orchestrator to generate on-policy
    rollouts each time `next_batch()` is called.

    - It delegates to Orchestrator.generate_batch(...) with fixed config.
    - For exotic behaviors (branching from snapshots, curricula, replay),
      write your own BatchSource implementation instead of hacking Trainer.

    This class is intentionally just a thin "policy" over:

        - which RolloutRequests to run this step
        - and how to configure max_steps / timeouts / tokenization

    so that Trainer never needs to know about envs, contexts, or requests.
    """

    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        credit_assigner: CreditAssigner,
        requests_fn: Callable[[], List[RolloutRequest]],
        max_steps: int,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        retokenize: bool = False,
        tokenize: Optional[TokenizeFn] = None,
    ) -> None:
        """
        Args:
            orchestrator:
                Orchestrator that knows how to execute RolloutRequests.

            credit_assigner:
                Credit assignment strategy (e.g. MonteCarloReturn).

            requests_fn:
                Callable that returns a fresh list of RolloutRequest objects
                for each batch. This is where you put curriculum / heterogeneity.

            max_steps, timeout_s, concurrency:
                Passed through to Orchestrator.generate_batch(...).

            retokenize, tokenize:
                Tokenization behavior knobs, also passed through.
        """
        if retokenize and tokenize is None:
            raise ValueError(
                "RolloutBatchSource: retokenize=True requires a tokenize() function."
            )

        self._orchestrator = orchestrator
        self._credit_assigner = credit_assigner
        self._requests_fn = requests_fn
        self._max_steps = max_steps
        self._timeout_s = timeout_s
        self._concurrency = concurrency
        self._retokenize = retokenize
        self._tokenize = tokenize

    async def next_batch(self) -> SAWBatch:
        """
        Produce one SAWBatch by running fresh rollouts according to
        the current set of RolloutRequests.
        """
        requests = self._requests_fn()
        return await self._orchestrator.generate_batch(
            requests=requests,
            max_steps=self._max_steps,
            credit_assigner=self._credit_assigner,
            timeout_s=self._timeout_s,
            concurrency=self._concurrency,
            retokenize=self._retokenize,
            tokenize=self._tokenize,
        )

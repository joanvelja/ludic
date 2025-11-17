from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Callable, List, Optional

from ludic.agent import Agent
from ludic.context.base import ContextStrategy
from ludic.context.full_dialog import FullDialog
from ludic.env import Env
from ludic.interaction import run_episode
from ludic.types import Rollout, SamplingArgs, Step

from ludic.training.types import (
    WeightingStrategy,
    SAWItem,
    SAWBatch,
    RolloutStepKey,
    TokenizeFn,
    StateFromStepFn,
)

# ---------------------------------------------------------------------------
# Factory aliases
# ---------------------------------------------------------------------------

EnvFactory = Callable[..., Env]          # Build a fresh Env each episode
CtxFactory = Callable[[], ContextStrategy]


class Orchestrator:
    """
    Dumb, stateless orchestrator:

      - spawns N envs
      - runs them with an asyncio.Semaphore
      - returns List[Rollout]
      - optionally writes each rollout to JSONL

    Extended variant:

      - can build State–Action–Weight batches via `generate_batch`, using:
          * a WeightingStrategy for credit assignment
          * model token IDs from Step.info when available
          * a fallback tokenizer otherwise
    """

    def __init__(
        self,
        env_factory: EnvFactory,
        agent: Agent,
        *,
        ctx_factory: Optional[CtxFactory] = None,
        jsonl_path: Optional[str] = None,
    ) -> None:
        self.env_factory = env_factory
        self.agent = agent
        self.ctx_factory = ctx_factory or (lambda: FullDialog())
        self.jsonl_path = jsonl_path

        if self.jsonl_path:
            Path(os.path.dirname(self.jsonl_path) or ".").mkdir(
                parents=True, exist_ok=True
            )

    # ---- internal helpers ------------------------------------------------

    async def _run_one(
        self,
        idx: int,
        sem: asyncio.Semaphore,
        *,
        max_steps: int,
        sampling_args: Optional[SamplingArgs],
        system_prompt: Optional[str],
        timeout_s: Optional[float],
    ) -> Rollout:
        async with sem:
            env = self.env_factory()
            ctx = self.ctx_factory()
            sargs: SamplingArgs = sampling_args or {}

            rollout = await run_episode(
                env=env,
                agent=self.agent,
                max_steps=max_steps,
                sampling_args=sargs,
                ctx=ctx,
                system_prompt=system_prompt,
                timeout_s=timeout_s,
            )

            # basic metadata; purely for logging / debugging
            rollout.meta.setdefault("episode_idx", idx)
            rollout.meta.setdefault("orchestrator", {})
            rollout.meta["orchestrator"].update(
                {
                    "max_steps": max_steps,
                    "timeout_s": timeout_s,
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

    async def generate(
        self,
        *,
        batch_size: int,
        max_steps: int,
        sampling_args: Optional[SamplingArgs] = None,
        system_prompt: Optional[str] = None,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
    ) -> List[Rollout]:
        """
        Run `batch_size` independent rollouts and return them.
        """
        if batch_size <= 0:
            return []

        sem = asyncio.Semaphore(max(1, concurrency))
        tasks = [
            self._run_one(
                i,
                sem,
                max_steps=max_steps,
                sampling_args=sampling_args,
                system_prompt=system_prompt,
                timeout_s=timeout_s,
            )
            for i in range(batch_size)
        ]
        return await asyncio.gather(*tasks)


    def generate_sync(
        self,
        *,
        batch_size: int,
        max_steps: int,
        sampling_args: Optional[SamplingArgs] = None,
        system_prompt: Optional[str] = None,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
    ) -> List[Rollout]:
        """
        Synchronous wrapper around `generate()` for scripts/CLIs.

        If you're in an async env, call `await generate(...)` directly.
        """
        return asyncio.run(
            self.generate(
                batch_size=batch_size,
                max_steps=max_steps,
                sampling_args=sampling_args,
                system_prompt=system_prompt,
                timeout_s=timeout_s,
                concurrency=concurrency,
            )
        )

    # ---- SAW batch generation --------------------------------------------

    async def generate_batch(
        self,
        *,
        batch_size: int,
        max_steps: int,
        weighting: WeightingStrategy,
        tokenize: TokenizeFn,
        sampling_args: Optional[SamplingArgs] = None,
        system_prompt: Optional[str] = None,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        state_from_step: Optional[StateFromStepFn] = None,
        use_model_token_ids: bool = True,
        retokenize: bool = False,
    ) -> SAWBatch:
        """
        High-level entrypoint for RL-style training:

        - runs batch_size many rollouts (via `generate`)
        - computes weights per (rollout, step) via WeightingStrategy
        - builds a State–Action–Weight batch, including:
            * tokenized input_ids (state + action)
            * attention_mask
            * action_mask (1 on action tokens, 0 elsewhere)
            * scalar weight per item
            * batch-level metadata (in SAWBatch.meta)

        Tokenization strategy:

        - If `use_model_token_ids=True`, this looks for stored model token IDs:
                step.info["prompt_token_ids"]
                step.info["token_ids"]      # TODO: actually: completion ids
          These must be populated by the Agent or run_episode.
          If present, they are always used.

        - If model token IDs are missing:

                * If `retokenize=True`, we fall back to the provided `tokenize(text)`
                  function for both state and action.

                * If `retokenize=False`, we raise an error to avoid silent
                  mismatch between model vs. post-hoc tokenization.

        `state_from_step` default:
        - If not provided, the “state” is the observation before the action:
                state_text = step.prev_obs
        """

        rollouts = await self.generate(
            batch_size=batch_size,
            max_steps=max_steps,
            sampling_args=sampling_args,
            system_prompt=system_prompt,
            timeout_s=timeout_s,
            concurrency=concurrency,
        )
        weights = weighting.compute(rollouts)

        if state_from_step is None:

            def default_state_from_step(r: Rollout, i: int, step: Step) -> str:
                return step.prev_obs

            state_fn: StateFromStepFn = default_state_from_step
        else:
            state_fn = state_from_step

        items: List[SAWItem] = []

        for r in rollouts:
            for i, step in enumerate(r.steps):
                key: RolloutStepKey = (r.id, step.index)

                # ---- every step must have an explicit weight ----
                try:
                    w_raw = weights[key]
                except KeyError as exc:
                    raise KeyError(
                        f"WeightingStrategy did not provide a weight for "
                        f"(rollout_id={r.id!r}, step_index={step.index}). "
                        "All steps must be covered."
                    ) from exc

                w = float(w_raw)
                info = step.info or {}

                # Try model token IDs
                prompt_ids = info.get("prompt_token_ids")
                # TODO: naming here is inconsistent; "token_ids" are the completion ids.
                completion_ids = info.get("token_ids")

                has_model_ids = (
                    isinstance(prompt_ids, list)
                    and isinstance(completion_ids, list)
                    and all(isinstance(t, int) for t in prompt_ids)
                    and all(isinstance(t, int) for t in completion_ids)
                )

                if use_model_token_ids and has_model_ids:
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
                if not retokenize and use_model_token_ids:
                    raise ValueError(
                        f"Missing model token IDs for rollout {r.id}, step {step.index}, "
                        "but retokenize=False. "
                        "Either enable retokenize=True or fix your Agent/run_episode "
                        "to store 'prompt_token_ids' and 'token_ids' in Step.info."
                    )

                # Path B: retokenize using text
                state_text = state_fn(r, i, step)
                action_text = step.action

                state_ids = tokenize(state_text)
                action_ids = tokenize(action_text)

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

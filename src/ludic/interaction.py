from __future__ import annotations
from typing import Optional

from ludic.env import Env
from ludic.agent import Agent
from ludic.types import Rollout, Step, StepOutcome, SamplingArgs
from ludic.context.base import ContextStrategy
from ludic.context.full_dialog import FullDialog
from ludic.parsers import Parser, ParseResult


async def run_episode(
    env: Env,
    agent: Agent,
    *,
    max_steps: int,
    sampling_args: Optional[SamplingArgs] = None,
    ctx: Optional[ContextStrategy] = None,
    system_prompt: Optional[str] = None,
    timeout_s: Optional[float] = None,
    action_parser: Optional[Parser] = None,
) -> Rollout:
    # context instance per episode
    if ctx is None:
        ctx = FullDialog()

    # choose system prompt priority: explicit > env-suggested > none
    sys = system_prompt or env.suggested_sysprompt
    ctx.reset(system_prompt=sys)

    rollout = Rollout(meta={
        "agent": getattr(agent, "name", "unknown"),
        "env": env.__class__.__name__,
        "ctx": ctx.__class__.__name__,
    })

    obs, info = env.reset()
    ctx.on_env_reset(obs, info)

    sargs: SamplingArgs = sampling_args or {}

    for t in range(max_steps):
        messages = ctx.on_before_act()
        resp, _ = await agent.act(
            messages=messages,
            sampling_args=sargs,
            timeout_s=timeout_s,
        )
        ctx.on_after_act(resp)

        # Raw model output
        raw_action = resp.text

        # If no parser is provided, treat raw text as fully valid
        if action_parser is None:
            parse_result = ParseResult(action=raw_action, reward=0.0, obs=None)
        else:
            parse_result = action_parser(raw_action)

        # ------------------------------------------------------------
        # CASE 1: PARSER FAILURE (NO env.step CALL)
        # ------------------------------------------------------------
        if parse_result.action is None:
            synthetic_obs = parse_result.obs or "Invalid action."
            parser_reward = parse_result.reward

            rollout.steps.append(Step(
                index=t,
                prev_obs=obs,
                action=raw_action,
                next_obs=synthetic_obs,
                reward=parser_reward,
                truncated=False,
                terminated=False,
                info={
                    "parse_error": True,
                    "raw_action": raw_action,
                },
            ))

            obs = synthetic_obs
            ctx.on_after_step(obs, {})
            continue

        # ------------------------------------------------------------
        # CASE 2: PARSER SUCCESS → call env.step
        # ------------------------------------------------------------
        parsed_action = parse_result.action
        parser_reward = parse_result.reward

        outcome: StepOutcome = env.step(parsed_action)

        # Build info dict
        step_info = dict(outcome.info)
        step_info["parsed_action"] = parsed_action

        # Token-level metadata
        if resp.prompt_token_ids is not None:
            step_info["prompt_token_ids"] = resp.prompt_token_ids
        if resp.token_ids is not None:
            step_info["token_ids"] = resp.token_ids
        if resp.finish_reason is not None:
            step_info.setdefault("finish_reason", resp.finish_reason)

        # Total reward = env reward + parser reward
        total_reward = outcome.reward + parser_reward

        # For logging: terminal/truncated steps have no next_obs
        if outcome.terminated or outcome.truncated:
            logged_next_obs = None
        else:
            logged_next_obs = outcome.obs

        rollout.steps.append(Step(
            index=t,
            prev_obs=obs,
            action=raw_action,
            next_obs=logged_next_obs,
            reward=total_reward,
            truncated=outcome.truncated,
            terminated=outcome.terminated,
            info=step_info,
        ))

        obs = outcome.obs
        ctx.on_after_step(obs, outcome.info)

        if outcome.terminated or outcome.truncated:
            break

    return rollout

from __future__ import annotations
from typing import Optional
from ludic.env import Env
from ludic.agent.base import Agent
from ludic.types import Rollout, Step, StepOutcome, SamplingArgs
from ludic.context.base import ContextStrategy
from ludic.context.full_dialog import FullDialog

async def run_episode(
    env: Env,
    agent: Agent,
    max_steps,
    sampling_args: SamplingArgs,
    *,
    ctx: ContextStrategy = FullDialog(),
    timeout_s: Optional[float] = None,
) -> Rollout:
    # seed system prompt from env if available
    sys = env.suggested_sysprompt
    if sys:
        ctx.reset(system_prompt=sys)
    else:
        ctx.reset()

    rollout = Rollout(meta={"agent": getattr(agent, "name", "unknown"),
                            "env": env.__class__.__name__,
                            "ctx": ctx.__class__.__name__})

    obs, info = env.reset()
    ctx.on_env_reset(obs, info)

    for t in range(max_steps):
        messages = ctx.on_before_act()
        text = await agent.call(
            messages=messages,
            sampling_args=sampling_args,
            timeout_s=timeout_s,
        )
        ctx.on_after_act(text)
        outcome: StepOutcome = env.step(text)

        step = Step(
            index=t,
            prev_obs=obs,
            action=text,
            next_obs=outcome.obs,
            reward=outcome.reward,
            truncated=outcome.truncated,
            terminated=outcome.terminated,
            info=outcome.info,
        )
        rollout.steps.append(step)

        obs = outcome.obs
        ctx.on_after_step(obs, outcome.info)

        if outcome.terminated or outcome.truncated:
            break

    return rollout

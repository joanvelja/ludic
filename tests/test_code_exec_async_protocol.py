"""
Integration tests for async env support in SingleAgentProtocol.

Tests that the protocol correctly detects and handles envs with async
env_reset and env_step methods (like CodeExecEnv).
"""

from typing import Optional, Tuple

import pytest

from ludic.context.full_dialog import FullDialog
from ludic.interaction.single_agent import SingleAgentProtocol, _has_async_env_methods
from ludic.agents.base_agent import Agent
from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.parsers import ParseResult
from ludic.types import Info, Observation, StepOutcome
from tests._mocks import MockClient


# Simple pass-through parser for tests
def _passthrough_parser(raw: str) -> ParseResult:
    return ParseResult(action=raw, reward=0.0, obs=None)


# ---------------------------------------------------------------------
# Mock Async Env for Testing
# ---------------------------------------------------------------------


class MockAsyncEnv(SingleAgentEnv):
    """
    A mock async env that simulates CodeExecEnv behavior.

    Has async env_reset and env_step methods, unlike standard sync envs.
    """

    def __init__(
        self,
        target_action: str = "correct_code",
        max_steps: int = 3,
    ):
        super().__init__()
        self._target_action = target_action
        self._max_steps = max_steps
        self._step_count = 0
        self._obs = "Write code to solve the problem."

        # Track calls for assertions
        self.reset_calls = 0
        self.step_calls = 0

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return "You are a code assistant."

    async def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        """Async reset method (like CodeExecEnv)."""
        self.reset_calls += 1
        self._step_count = 0
        self._obs = "Write code to solve the problem."
        return self._obs, {"problem_id": "test_problem", "async_env": True}

    async def env_step(self, action: str) -> StepOutcome:
        """Async step method (like CodeExecEnv)."""
        self.step_calls += 1
        self._step_count += 1

        if action == self._target_action:
            # Correct code - terminate with success
            return StepOutcome(
                obs="All tests passed!",
                reward=1.0,
                truncated=False,
                terminated=True,
                info={"all_passed": True, "step_count": self._step_count},
            )
        elif self._step_count >= self._max_steps:
            # Max steps reached - truncate
            return StepOutcome(
                obs=f"Tests failed. Attempt {self._step_count}/{self._max_steps}.",
                reward=-0.1,
                truncated=True,
                terminated=False,
                info={"all_passed": False, "step_count": self._step_count},
            )
        else:
            # Wrong code but more attempts allowed
            return StepOutcome(
                obs=f"Tests failed. Try again. Attempt {self._step_count}/{self._max_steps}.",
                reward=-0.1,
                truncated=False,
                terminated=False,
                info={"all_passed": False, "step_count": self._step_count},
            )

    def env_current_obs(self) -> Observation:
        return self._obs


class MockSyncEnv(SingleAgentEnv):
    """
    A standard sync env for comparison testing.
    Uses regular (non-async) env_reset and env_step.
    """

    def __init__(self, target_action: str = "correct"):
        super().__init__()
        self._target_action = target_action
        self._obs = "Sync env observation"
        self.reset_calls = 0
        self.step_calls = 0

    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        """Standard sync reset."""
        self.reset_calls += 1
        self._obs = "Sync env observation"
        return self._obs, {"sync_env": True}

    def env_step(self, action: str) -> StepOutcome:
        """Standard sync step."""
        self.step_calls += 1
        terminated = action == self._target_action
        return StepOutcome(
            obs="Success" if terminated else "Wrong",
            reward=1.0 if terminated else -0.1,
            truncated=False,
            terminated=terminated,
            info={},
        )

    def env_current_obs(self) -> Observation:
        return self._obs


# ---------------------------------------------------------------------
# Async Detection Tests
# ---------------------------------------------------------------------


class TestAsyncDetection:
    def test_detects_async_reset(self):
        env = MockAsyncEnv()
        has_async_reset, has_async_step = _has_async_env_methods(env)
        assert has_async_reset is True

    def test_detects_async_step(self):
        env = MockAsyncEnv()
        has_async_reset, has_async_step = _has_async_env_methods(env)
        assert has_async_step is True

    def test_detects_sync_env(self):
        env = MockSyncEnv()
        has_async_reset, has_async_step = _has_async_env_methods(env)
        assert has_async_reset is False
        assert has_async_step is False


# ---------------------------------------------------------------------
# Protocol Async Env Integration Tests
# ---------------------------------------------------------------------


class TestProtocolAsyncEnvIntegration:
    @pytest.mark.asyncio
    async def test_protocol_runs_async_env_successfully(self):
        """Protocol should correctly run an async env and produce rollouts."""
        env = MockAsyncEnv(target_action="correct_code")
        agent = Agent(
            client=MockClient(text="correct_code"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        rollouts = await protocol.run(env=env, max_steps=5)

        assert len(rollouts) == 1
        rollout = rollouts[0]

        # Should terminate on first step with correct action
        assert len(rollout.steps) == 1
        assert rollout.steps[0].terminated is True
        assert rollout.steps[0].reward == pytest.approx(1.0)
        assert rollout.steps[0].info.get("all_passed") is True

    @pytest.mark.asyncio
    async def test_protocol_calls_async_reset(self):
        """Protocol should call async env_reset and receive correct observation."""
        env = MockAsyncEnv()
        agent = Agent(
            client=MockClient(text="wrong_code"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        await protocol.run(env=env, max_steps=1)

        # Verify async reset was called
        assert env.reset_calls == 1

    @pytest.mark.asyncio
    async def test_protocol_calls_async_step(self):
        """Protocol should call async env_step with the parsed action."""
        env = MockAsyncEnv()
        agent = Agent(
            client=MockClient(text="some_code"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        await protocol.run(env=env, max_steps=1)

        # Verify async step was called
        assert env.step_calls == 1

    @pytest.mark.asyncio
    async def test_protocol_uses_async_env_system_prompt(self):
        """Protocol should use the async env's suggested_sysprompt."""
        env = MockAsyncEnv()
        agent = Agent(
            client=MockClient(text="code"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        await protocol.run(env=env, max_steps=1)

        # The agent context should have the system prompt from env
        messages = agent._ctx.messages
        # First message should be system prompt
        assert any(
            m.get("role") == "system" and "code assistant" in m.get("content", "").lower()
            for m in messages
        )

    @pytest.mark.asyncio
    async def test_async_env_multiple_steps(self):
        """Test that async env works correctly over multiple steps."""
        env = MockAsyncEnv(target_action="correct", max_steps=5)

        # Agent says "wrong" first 2 times, then "correct"
        call_count = 0

        class CountingClient(MockClient):
            def __init__(self):
                super().__init__(text="wrong")

            async def complete(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    self._text = "wrong"
                else:
                    self._text = "correct"
                return await super().complete(*args, **kwargs)

        agent = Agent(
            client=CountingClient(),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        rollouts = await protocol.run(env=env, max_steps=10)

        assert len(rollouts) == 1
        rollout = rollouts[0]

        # Should have taken 3 steps to get correct answer
        assert len(rollout.steps) == 3
        assert rollout.steps[0].terminated is False
        assert rollout.steps[1].terminated is False
        assert rollout.steps[2].terminated is True

        # Total reward: -0.1 + -0.1 + 1.0 = 0.8
        assert rollout.total_reward == pytest.approx(0.8)


# ---------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------


class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_sync_env_still_works(self):
        """Sync envs should continue to work without changes."""
        env = MockSyncEnv(target_action="correct")
        agent = Agent(
            client=MockClient(text="correct"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        rollouts = await protocol.run(env=env, max_steps=5)

        assert len(rollouts) == 1
        rollout = rollouts[0]

        assert len(rollout.steps) == 1
        assert rollout.steps[0].terminated is True
        assert rollout.steps[0].reward == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_sync_env_reset_is_called(self):
        """Sync env reset should be called through normal path."""
        env = MockSyncEnv()
        agent = Agent(
            client=MockClient(text="wrong"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        await protocol.run(env=env, max_steps=1)

        assert env.reset_calls == 1

    @pytest.mark.asyncio
    async def test_sync_env_step_is_called(self):
        """Sync env step should be called through normal path."""
        env = MockSyncEnv()
        agent = Agent(
            client=MockClient(text="wrong"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        await protocol.run(env=env, max_steps=1)

        assert env.step_calls == 1


# ---------------------------------------------------------------------
# Info Propagation Tests
# ---------------------------------------------------------------------


class TestAsyncEnvInfoPropagation:
    @pytest.mark.asyncio
    async def test_reset_info_accessible_in_rollout(self):
        """Info from async env_reset should be accessible."""
        env = MockAsyncEnv()
        agent = Agent(
            client=MockClient(text="correct_code"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        rollouts = await protocol.run(env=env, max_steps=1)

        # The first step's prev_obs should be from reset
        assert rollouts[0].steps[0].prev_obs == "Write code to solve the problem."

    @pytest.mark.asyncio
    async def test_step_info_propagated_to_rollout(self):
        """Info from async env_step should be in the step info."""
        env = MockAsyncEnv()
        agent = Agent(
            client=MockClient(text="correct_code"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        rollouts = await protocol.run(env=env, max_steps=1)

        step_info = rollouts[0].steps[0].info
        assert step_info.get("all_passed") is True
        assert step_info.get("step_count") == 1


# ---------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------


class TestAsyncEnvEdgeCases:
    @pytest.mark.asyncio
    async def test_async_env_truncation_on_max_steps(self):
        """Async env that never terminates should truncate at max_steps."""
        env = MockAsyncEnv(target_action="impossible", max_steps=100)
        agent = Agent(
            client=MockClient(text="wrong"),
            model="mock",
            ctx=FullDialog(),
            parser=_passthrough_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        rollouts = await protocol.run(env=env, max_steps=3)

        rollout = rollouts[0]
        assert len(rollout.steps) == 3
        assert rollout.steps[-1].truncated is True
        assert rollout.meta.get("episode_truncated") is True

    @pytest.mark.asyncio
    async def test_async_env_with_parser(self):
        """Parser should work correctly with async envs."""
        from ludic.parsers import xml_tag_parser

        env = MockAsyncEnv(target_action="parsed_code")
        agent = Agent(
            client=MockClient(text="<code>parsed_code</code>"),
            model="mock",
            ctx=FullDialog(),
            parser=xml_tag_parser("code"),
        )
        protocol = SingleAgentProtocol(agent=agent)

        rollouts = await protocol.run(env=env, max_steps=5)

        rollout = rollouts[0]
        assert len(rollout.steps) == 1
        assert rollout.steps[0].terminated is True
        assert rollout.steps[0].info.get("parsed_action") == "parsed_code"

    @pytest.mark.asyncio
    async def test_async_env_parser_failure(self):
        """Parser failures should be handled correctly with async envs."""
        from ludic.parsers import ParseResult

        def strict_parser(text: str) -> ParseResult:
            if text.startswith("VALID:"):
                return ParseResult(action=text[6:], reward=0.1, obs=None)
            return ParseResult(action=None, reward=-0.5, obs="Invalid format")

        env = MockAsyncEnv()
        agent = Agent(
            client=MockClient(text="invalid_format"),
            model="mock",
            ctx=FullDialog(),
            parser=strict_parser,
        )
        protocol = SingleAgentProtocol(agent=agent)

        rollouts = await protocol.run(env=env, max_steps=1)

        rollout = rollouts[0]
        assert len(rollout.steps) == 1
        step = rollout.steps[0]

        # Parser failure - no env step called
        assert env.step_calls == 0
        assert step.info.get("parse_error") is True
        assert step.reward == pytest.approx(-0.5)
        assert step.next_obs == "Invalid format"

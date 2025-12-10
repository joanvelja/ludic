from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict, Tuple, Optional
from ludic.types import StepOutcome, Info

# --- Generic Types for the Kernel Interface ---

AgentID = TypeVar("AgentID", bound=str)
ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")

# ------------------------------------------------
# The "Kernel" Abstract Base Class
# ------------------------------------------------


class LudicEnv(ABC, Generic[AgentID, ObsType, ActionType]):
    """
    The canonical, multi-agent-aware Environment "Kernel" interface.

    This is the "core" interface that all advanced protocols should
    be built to consume. It is multi-agent by default.
    """

    @property
    @abstractmethod
    def agent_ids(self) -> List[AgentID]:
        """A list of all agent roles defined in this environment."""
        ...

    @property
    @abstractmethod
    def active_agents(self) -> List[AgentID]:
        """
        The list of agent IDs expected to provide an action *this* step.
        (For simultaneous-move envs, this is all agents.
         For turn-based, it may be only one).
        """
        ...

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None
    ) -> Dict[AgentID, Tuple[ObsType, Info]]:
        """
        Resets the environment.
        Returns:
            A dictionary mapping all agent IDs to their
            initial (Observation, Info) tuple.
        """
        ...

    @abstractmethod
    def step(self, actions: Dict[AgentID, ActionType]) -> Dict[AgentID, StepOutcome]:
        """
        Processes a dictionary of actions from the active agents.

        Args:
            actions: A dict mapping AgentID -> Action for
                     each agent in `active_agents`.

        Returns:
            A dictionary mapping all agent IDs to their
            StepOutcome for this step.
        """
        ...

    def current_obs(self) -> Dict[AgentID, ObsType]:
        """
        Returns the current observation for all agents.
        This is an optional method for convenience.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Release any held resources.
        Called after episode completes. Override in subclasses that hold
        expensive resources (database connections, sandbox handles, etc.).

        Default implementation is a no-op.
        """
        pass

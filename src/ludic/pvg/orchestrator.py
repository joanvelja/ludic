"""PVG Orchestrator: State machine for multi-round training.

This module provides a state machine orchestrator for PVG training that manages
phase transitions, state persistence, and resumption from checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class PVGState(Enum):
    """Valid states in the PVG training loop."""

    INIT = "init"
    MINT_DATA = "mint_data"
    TRAIN_VERIFIER = "train_verifier"
    TRAIN_PROVER = "train_prover"
    CHECKPOINT = "checkpoint"
    COMPLETE = "complete"


# Valid state transitions
_VALID_TRANSITIONS: Dict[PVGState, List[PVGState]] = {
    PVGState.INIT: [PVGState.MINT_DATA],
    PVGState.MINT_DATA: [PVGState.TRAIN_VERIFIER],
    PVGState.TRAIN_VERIFIER: [PVGState.TRAIN_PROVER],
    PVGState.TRAIN_PROVER: [PVGState.CHECKPOINT],
    PVGState.CHECKPOINT: [PVGState.MINT_DATA, PVGState.COMPLETE],
    PVGState.COMPLETE: [],
}


@dataclass
class PhaseCheckpoint:
    """Checkpoint information for a single phase."""

    path: str
    completed_at: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseCheckpoint":
        return cls(**data)


@dataclass
class RoundCheckpoints:
    """Checkpoints for all phases in a round."""

    mint: Optional[PhaseCheckpoint] = None
    verifier: Optional[PhaseCheckpoint] = None
    prover: Optional[PhaseCheckpoint] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.mint:
            result["mint"] = self.mint.to_dict()
        if self.verifier:
            result["verifier"] = self.verifier.to_dict()
        if self.prover:
            result["prover"] = self.prover.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoundCheckpoints":
        return cls(
            mint=PhaseCheckpoint.from_dict(data["mint"]) if "mint" in data else None,
            verifier=PhaseCheckpoint.from_dict(data["verifier"]) if "verifier" in data else None,
            prover=PhaseCheckpoint.from_dict(data["prover"]) if "prover" in data else None,
        )


@dataclass
class OrchestratorState:
    """Serializable state of the PVG orchestrator."""

    current_state: PVGState
    round_id: int
    phase_checkpoints: Dict[int, RoundCheckpoints]
    config_hash: str
    started_at: str
    last_updated: str
    num_rounds: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state.value,
            "round_id": self.round_id,
            "phase_checkpoints": {
                str(k): v.to_dict() for k, v in self.phase_checkpoints.items()
            },
            "config_hash": self.config_hash,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "num_rounds": self.num_rounds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorState":
        return cls(
            current_state=PVGState(data["current_state"]),
            round_id=data["round_id"],
            phase_checkpoints={
                int(k): RoundCheckpoints.from_dict(v)
                for k, v in data.get("phase_checkpoints", {}).items()
            },
            config_hash=data["config_hash"],
            started_at=data["started_at"],
            last_updated=data["last_updated"],
            num_rounds=data["num_rounds"],
            metadata=data.get("metadata", {}),
        )


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


class ConfigMismatchError(Exception):
    """Raised when resuming with a different config than the original run."""

    pass


class PVGOrchestrator:
    """State machine orchestrator for PVG multi-round training.

    Manages the training loop phases:
    INIT → MINT_DATA → TRAIN_VERIFIER → TRAIN_PROVER → CHECKPOINT → (next round or COMPLETE)

    Provides:
    - Automatic state persistence to JSON
    - Resume from checkpoint support
    - Config hash validation on resume
    - Phase checkpoint tracking

    Example:
        ```python
        orchestrator = PVGOrchestrator(
            output_dir=Path("./outputs/pvg"),
            config=game_config,
        )

        # Fresh start
        orchestrator.initialize()

        # Or resume from existing state
        orchestrator.resume_from_checkpoint()

        # Transition through phases
        orchestrator.transition(PVGState.MINT_DATA)
        # ... do minting work ...
        orchestrator.record_phase_checkpoint("mint", checkpoint_path)

        orchestrator.transition(PVGState.TRAIN_VERIFIER)
        # ... train verifier ...
        orchestrator.record_phase_checkpoint("verifier", checkpoint_path)
        ```
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Any,
        *,
        state_filename: str = "state.json",
    ) -> None:
        """Initialize the orchestrator.

        Args:
            output_dir: Directory for state and checkpoints
            config: PVGGameConfig to track (used for hash validation)
            state_filename: Name of the state file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / state_filename
        self._config = config
        self._config_hash = self._compute_config_hash(config)
        self._state: Optional[OrchestratorState] = None

    @staticmethod
    def _compute_config_hash(config: Any) -> str:
        """Compute a deterministic hash of the config for validation."""
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif hasattr(config, "__dict__"):
            config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else config.__dict__
        else:
            config_dict = str(config)

        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @property
    def state(self) -> OrchestratorState:
        """Get the current state (raises if not initialized)."""
        if self._state is None:
            raise RuntimeError(
                "Orchestrator not initialized. Call initialize() or resume_from_checkpoint()."
            )
        return self._state

    @property
    def current_state(self) -> PVGState:
        """Get the current PVG state."""
        return self.state.current_state

    @property
    def round_id(self) -> int:
        """Get the current round ID."""
        return self.state.round_id

    @property
    def is_complete(self) -> bool:
        """Check if training is complete."""
        return self.state.current_state == PVGState.COMPLETE

    def initialize(self) -> None:
        """Initialize a fresh training run.

        Creates a new state starting at INIT with round 0.
        """
        now = datetime.now().isoformat()
        self._state = OrchestratorState(
            current_state=PVGState.INIT,
            round_id=0,
            phase_checkpoints={},
            config_hash=self._config_hash,
            started_at=now,
            last_updated=now,
            num_rounds=self._config.num_rounds,
        )
        self._persist_state()
        logger.info("Initialized fresh PVG training run")

    def resume_from_checkpoint(
        self,
        *,
        validate_config: bool = True,
    ) -> bool:
        """Resume from existing state file if present.

        Args:
            validate_config: If True, verify config hash matches

        Returns:
            True if resumed from existing state, False if no state file found

        Raises:
            ConfigMismatchError: If validate_config=True and config hash differs
        """
        if not self.state_path.exists():
            logger.info("No existing state file found, starting fresh")
            return False

        with open(self.state_path) as f:
            data = json.load(f)

        self._state = OrchestratorState.from_dict(data)

        if validate_config and self._state.config_hash != self._config_hash:
            raise ConfigMismatchError(
                f"Config hash mismatch: saved={self._state.config_hash}, "
                f"current={self._config_hash}. Set validate_config=False to override."
            )

        logger.info(
            "Resumed from checkpoint: state=%s, round=%d",
            self._state.current_state.value,
            self._state.round_id,
        )
        return True

    def can_transition(self, target: PVGState) -> bool:
        """Check if a transition to the target state is valid.

        Args:
            target: Target state to transition to

        Returns:
            True if the transition is allowed
        """
        valid_targets = _VALID_TRANSITIONS.get(self.current_state, [])
        return target in valid_targets

    def transition(self, target: PVGState) -> None:
        """Transition to a new state.

        Args:
            target: Target state to transition to

        Raises:
            InvalidTransitionError: If the transition is not allowed
        """
        if not self.can_transition(target):
            valid = [s.value for s in _VALID_TRANSITIONS.get(self.current_state, [])]
            raise InvalidTransitionError(
                f"Cannot transition from {self.current_state.value} to {target.value}. "
                f"Valid targets: {valid}"
            )

        old_state = self.current_state
        self._state.current_state = target
        self._state.last_updated = datetime.now().isoformat()
        self._persist_state()

        logger.info(
            "State transition: %s → %s (round %d)",
            old_state.value,
            target.value,
            self.round_id,
        )

    def advance_round(self) -> None:
        """Advance to the next round after CHECKPOINT phase.

        Call this after completing a round's checkpoint phase to
        increment the round counter and prepare for the next round.
        """
        if self.current_state != PVGState.CHECKPOINT:
            raise InvalidTransitionError(
                f"Can only advance round from CHECKPOINT state, "
                f"currently at {self.current_state.value}"
            )

        self._state.round_id += 1
        self._state.last_updated = datetime.now().isoformat()
        self._persist_state()

        logger.info("Advanced to round %d", self.round_id)

    def record_phase_checkpoint(
        self,
        phase: str,
        path: Union[str, Path],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a checkpoint for the current phase.

        Args:
            phase: Phase name ("mint", "verifier", or "prover")
            path: Path to the checkpoint
            metrics: Optional metrics to record with the checkpoint
        """
        if phase not in ("mint", "verifier", "prover"):
            raise ValueError(f"Invalid phase: {phase!r}")

        checkpoint = PhaseCheckpoint(
            path=str(path),
            completed_at=datetime.now().isoformat(),
            metrics=metrics or {},
        )

        round_id = self.round_id
        if round_id not in self._state.phase_checkpoints:
            self._state.phase_checkpoints[round_id] = RoundCheckpoints()

        setattr(self._state.phase_checkpoints[round_id], phase, checkpoint)
        self._state.last_updated = datetime.now().isoformat()
        self._persist_state()

        logger.info(
            "Recorded %s checkpoint for round %d: %s",
            phase,
            round_id,
            path,
        )

    def get_phase_checkpoint(
        self,
        round_id: int,
        phase: str,
    ) -> Optional[PhaseCheckpoint]:
        """Get checkpoint for a specific round and phase.

        Args:
            round_id: Round number
            phase: Phase name ("mint", "verifier", or "prover")

        Returns:
            PhaseCheckpoint if found, None otherwise
        """
        if round_id not in self._state.phase_checkpoints:
            return None
        return getattr(self._state.phase_checkpoints[round_id], phase, None)

    def get_all_checkpoints(self) -> Dict[int, RoundCheckpoints]:
        """Get all recorded checkpoints by round."""
        return dict(self._state.phase_checkpoints)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set arbitrary metadata in the state.

        Args:
            key: Metadata key
            value: Metadata value (must be JSON-serializable)
        """
        self._state.metadata[key] = value
        self._state.last_updated = datetime.now().isoformat()
        self._persist_state()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the state.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self._state.metadata.get(key, default)

    def _persist_state(self) -> None:
        """Persist the current state to disk."""
        with open(self.state_path, "w") as f:
            json.dump(self._state.to_dict(), f, indent=2)

    def should_continue(self) -> bool:
        """Check if training should continue (not complete and rounds remaining).

        Returns:
            True if more work to do, False if complete
        """
        if self.is_complete:
            return False
        return self.round_id < self._state.num_rounds

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of training progress.

        Returns:
            Dictionary with progress information
        """
        completed_rounds = len([
            r for r in self._state.phase_checkpoints.values()
            if r.prover is not None
        ])

        return {
            "current_state": self.current_state.value,
            "current_round": self.round_id,
            "completed_rounds": completed_rounds,
            "total_rounds": self._state.num_rounds,
            "is_complete": self.is_complete,
            "started_at": self._state.started_at,
            "last_updated": self._state.last_updated,
        }

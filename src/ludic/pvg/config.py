"""
PVG (Prover-Verifier Game) configuration dataclasses.

This module provides configuration dataclasses for the PVG training orchestration,
including round-specific settings, data split configuration, verifier initialization,
and GPU placement strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class DataSplitConfig:
    """Configuration for splitting dataset into D_V (verifier) and D_π (prover) subsets."""

    split_ratio: float = 0.5  # Fraction for D_V
    split_seed: int = 42
    stratify_by: Optional[str] = None  # Optional stratification key (e.g., "difficulty")

    def __post_init__(self) -> None:
        if not 0.0 < self.split_ratio < 1.0:
            raise ValueError(f"split_ratio must be in (0, 1), got {self.split_ratio}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSplitConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {"split_ratio", "split_seed", "stratify_by"}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class VerifierInitConfig:
    """Configuration for verifier initialization per round."""

    base_path: str
    reinit_mode: Literal["full", "head_only"] = "full"
    head_init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.head_init_std <= 0:
            raise ValueError(
                f"head_init_std must be positive, got {self.head_init_std}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifierInitConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {"base_path", "reinit_mode", "head_init_std"}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training.

    LoRA enables efficient fine-tuning by training low-rank adapter matrices
    instead of the full weight matrices.
    """

    enabled: bool = False
    rank: int = 16
    alpha: int = 32  # Scaling factor, typically 2x rank
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    use_rslora: bool = False  # rsLoRA: rank-scaled LoRA for stable gradients

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoRAConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {"enabled", "rank", "alpha", "target_modules", "use_rslora"}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class InferencePlacementConfig:
    """Configuration for GPU placement during training."""

    inference_gpu_ids: List[int] = field(default_factory=lambda: [0])
    training_gpu_ids: List[int] = field(default_factory=lambda: [1, 2, 3])
    sync_interval: int = 50  # Sync weights every N training steps
    vllm_tensor_parallel: int = 1

    def __post_init__(self) -> None:
        if self.sync_interval <= 0:
            raise ValueError(
                f"sync_interval must be positive, got {self.sync_interval}"
            )
        if self.vllm_tensor_parallel <= 0:
            raise ValueError(
                f"vllm_tensor_parallel must be positive, got {self.vllm_tensor_parallel}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferencePlacementConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {"inference_gpu_ids", "training_gpu_ids", "sync_interval", "vllm_tensor_parallel"}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class PVGRoundConfig:
    """Configuration for a single PVG training round."""

    round_id: int
    verifier_steps: int = 1000
    prover_steps: int = 1000
    sneaky_incorrect_threshold: float = (
        0.95  # Stop if sneaky incorrect rate exceeds this
    )
    score_parity_threshold: float = 0.1  # Stop if score gap is below this
    data_mixture_strategy: Literal["equal_per_round", "weighted", "latest_only"] = (
        "equal_per_round"
    )
    irm_mode: Literal["vrex", "none"] = "none"
    sequential_observation: bool = False  # Whether sneaky prover sees honest solution

    def __post_init__(self) -> None:
        if self.round_id < 0:
            raise ValueError(f"round_id must be non-negative, got {self.round_id}")
        if self.verifier_steps <= 0:
            raise ValueError(
                f"verifier_steps must be positive, got {self.verifier_steps}"
            )
        if self.prover_steps <= 0:
            raise ValueError(f"prover_steps must be positive, got {self.prover_steps}")
        if not 0.0 <= self.sneaky_incorrect_threshold <= 1.0:
            raise ValueError(
                f"sneaky_incorrect_threshold must be in [0, 1], got {self.sneaky_incorrect_threshold}"
            )
        if self.score_parity_threshold < 0:
            raise ValueError(
                f"score_parity_threshold must be non-negative, got {self.score_parity_threshold}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PVGRoundConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {
            "round_id", "verifier_steps", "prover_steps",
            "sneaky_incorrect_threshold", "score_parity_threshold",
            "data_mixture_strategy", "irm_mode", "sequential_observation"
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class PVGGameConfig:
    """Top-level configuration for PVG training."""

    num_rounds: int
    verifier_model_path: str
    prover_model_path: str
    output_dir: Union[str, Path]
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    round_configs: Optional[List[PVGRoundConfig]] = None  # If None, use defaults
    default_round_config: Optional[PVGRoundConfig] = (
        None  # Template for rounds without explicit config
    )
    checkpoints_per_round: int = 1
    verifier_init: VerifierInitConfig = field(
        default_factory=lambda: VerifierInitConfig(base_path="")
    )
    inference_placement: InferencePlacementConfig = field(
        default_factory=InferencePlacementConfig
    )

    def __post_init__(self) -> None:
        if self.num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive, got {self.num_rounds}")
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.checkpoints_per_round < 0:
            raise ValueError(
                f"checkpoints_per_round must be non-negative, got {self.checkpoints_per_round}"
            )

    def get_round_config(self, round_id: int) -> PVGRoundConfig:
        """Get configuration for a specific round.

        Args:
            round_id: The round identifier (0-indexed).

        Returns:
            PVGRoundConfig for the specified round.

        Raises:
            ValueError: If round_id is out of range [0, num_rounds).
        """
        if round_id < 0 or round_id >= self.num_rounds:
            raise ValueError(f"round_id {round_id} out of range [0, {self.num_rounds})")

        # Check explicit round configs first
        if self.round_configs:
            for rc in self.round_configs:
                if rc.round_id == round_id:
                    return rc

        # Fall back to default with round_id set
        if self.default_round_config:
            # Create a copy with the correct round_id
            return PVGRoundConfig(
                round_id=round_id,
                verifier_steps=self.default_round_config.verifier_steps,
                prover_steps=self.default_round_config.prover_steps,
                sneaky_incorrect_threshold=self.default_round_config.sneaky_incorrect_threshold,
                score_parity_threshold=self.default_round_config.score_parity_threshold,
                data_mixture_strategy=self.default_round_config.data_mixture_strategy,
                irm_mode=self.default_round_config.irm_mode,
                sequential_observation=self.default_round_config.sequential_observation,
            )

        # Use bare defaults
        return PVGRoundConfig(round_id=round_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PVGGameConfig":
        """Create config from dictionary with nested object parsing.

        Args:
            data: Configuration dictionary, typically loaded from YAML

        Returns:
            PVGGameConfig instance
        """
        # Handle nested configs
        parsed: Dict[str, Any] = {}

        # Required fields
        parsed["num_rounds"] = data["num_rounds"]
        parsed["verifier_model_path"] = data.get("verifier_model_path", data.get("verifier_model", ""))
        parsed["prover_model_path"] = data.get("prover_model_path", data.get("prover_model", ""))
        parsed["output_dir"] = data.get("output_dir", "./outputs/pvg")

        # Optional nested configs
        if "data_split" in data and isinstance(data["data_split"], dict):
            parsed["data_split"] = DataSplitConfig.from_dict(data["data_split"])
        elif "split_ratio" in data:
            # Support flat keys for convenience
            parsed["data_split"] = DataSplitConfig(split_ratio=data["split_ratio"])

        if "verifier_init" in data and isinstance(data["verifier_init"], dict):
            parsed["verifier_init"] = VerifierInitConfig.from_dict(data["verifier_init"])

        if "inference_placement" in data and isinstance(data["inference_placement"], dict):
            parsed["inference_placement"] = InferencePlacementConfig.from_dict(data["inference_placement"])

        # Round configs - support both list and dict format
        if "round_configs" in data:
            rc_data = data["round_configs"]
            if isinstance(rc_data, dict):
                # Handle dict format: {default: {...}, round_N: {...}}
                default_config = rc_data.get("default", {})
                if default_config:
                    parsed["default_round_config"] = PVGRoundConfig.from_dict(
                        {"round_id": 0, **default_config}
                    )

                # Extract round-specific overrides
                round_configs = []
                for key, val in rc_data.items():
                    if key.startswith("round_"):
                        round_id = int(key.split("_")[1])
                        # Merge default with round-specific overrides
                        merged = {**default_config, **val, "round_id": round_id}
                        round_configs.append(PVGRoundConfig.from_dict(merged))

                if round_configs:
                    parsed["round_configs"] = round_configs

            elif isinstance(rc_data, list):
                # Original list format
                parsed["round_configs"] = [
                    PVGRoundConfig.from_dict(rc) if isinstance(rc, dict) else rc
                    for rc in rc_data
                ]

        if "default_round_config" in data and isinstance(data["default_round_config"], dict):
            parsed["default_round_config"] = PVGRoundConfig.from_dict(data["default_round_config"])

        # Optional scalar fields
        if "checkpoints_per_round" in data:
            parsed["checkpoints_per_round"] = data["checkpoints_per_round"]

        return cls(**parsed)

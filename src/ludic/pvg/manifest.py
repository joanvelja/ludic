"""PVG component manifest utilities."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


MANIFEST_VERSION = 1


@dataclass
class ComponentManifest:
    """Serializable manifest for a PVG component run."""

    component_name: str
    run_id: str
    git_sha: str
    config_hash: str
    started_at: str
    finished_at: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    round_id: Optional[int] = None
    version: int = MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentManifest":
        return cls(**data)


def new_run_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def compute_config_hash(config: Any) -> str:
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "__dict__"):
        config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else config.__dict__
    else:
        config_dict = str(config)
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return _short_hash(config_str)


def _short_hash(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode()).hexdigest()[:16]


def write_manifest(path: Union[str, Path], manifest: ComponentManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, default=str)
    return path


def read_manifest(path: Union[str, Path]) -> ComponentManifest:
    with open(path) as f:
        data = json.load(f)
    return ComponentManifest.from_dict(data)


def build_manifest_path(
    output_dir: Union[str, Path],
    component_name: str,
    run_id: str,
    *,
    round_id: Optional[int] = None,
) -> Path:
    output_dir = Path(output_dir)
    round_tag = f"round_{round_id}" if round_id is not None else "global"
    return output_dir / "manifests" / round_tag / f"{component_name}_{run_id}.json"

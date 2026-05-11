"""
utils/config_schema.py
======================
Configuration schema for Neural Forge Developer Mode projects.

The schema reads and writes the user project's config.yaml file and keeps the
trainer-facing values small, explicit, and serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when PyYAML is absent
    yaml = None


def _coerce_value(raw: str) -> Any:
    value = raw.split("#", 1)[0].strip()
    lower = value.lower()
    if lower in {"null", "none", ""}:
        return None
    if lower in {"true", "false"}:
        return lower == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce_value(item) for item in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _load_yaml_like(path: Path) -> dict[str, Any]:
    if yaml is not None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}

    values: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _coerce_value(value)
    return values


def _dump_yaml_like(payload: dict[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)

    lines = []
    for key, value in payload.items():
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(item) for item in value) + "]"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"


@dataclass
class DevProjectConfig:
    """Serializable Developer Mode training settings."""

    task: str = "classification"
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    optimizer: str = "adam"
    device: str = "cuda"
    auto_pause_temp: float = 90.0
    resume_temp: float = 80.0
    loss: str | None = None
    metrics: list[str] | None = None
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    DEFAULT_LOSSES: ClassVar[dict[str, str]] = {
        "classification": "CrossEntropyLoss",
        "segmentation": "DiceBCELoss",
    }
    DEFAULT_METRICS: ClassVar[dict[str, list[str]]] = {
        "classification": ["accuracy"],
        "segmentation": ["iou", "dice"],
    }

    @classmethod
    def load(cls, project_root: str | Path) -> "DevProjectConfig":
        path = Path(project_root) / "config.yaml"
        if not path.exists():
            return cls()

        raw = _load_yaml_like(path)

        known = {name for name in cls.__dataclass_fields__ if name != "_extra"}
        filtered = {key: value for key, value in raw.items() if key in known}
        extra = {key: value for key, value in raw.items() if key not in known}

        if isinstance(filtered.get("metrics"), str):
            filtered["metrics"] = [filtered["metrics"]]

        cfg = cls(**filtered)
        cfg._extra = extra
        return cfg

    def save(self, project_root: str | Path) -> Path:
        path = Path(project_root) / "config.yaml"
        payload = {**self._extra, **self.to_dict()}
        path.write_text(_dump_yaml_like(payload), encoding="utf-8")
        return path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_extra", None)
        return payload

    def effective_loss(self) -> str:
        if self.loss:
            return str(self.loss)
        return self.DEFAULT_LOSSES.get(self.task, "CrossEntropyLoss")

    def effective_metrics(self) -> list[str]:
        if self.metrics:
            return list(self.metrics)
        return list(self.DEFAULT_METRICS.get(self.task, ["accuracy"]))

    @classmethod
    def create_example(cls, project_root: str | Path) -> Path:
        """Write an example config.yaml without overwriting an existing one."""
        path = Path(project_root) / "config.yaml"
        if path.exists():
            return path

        example = cls().to_dict()
        path.write_text(_dump_yaml_like(example), encoding="utf-8")
        return path

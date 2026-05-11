"""
config_schema.py
================
Small configuration object used by experimental Developer Mode training code.

The current main application keeps Developer Mode training disabled, but the
merged branch includes backend.dev_trainer, which imports this schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _coerce_value(raw: str) -> Any:
    value = raw.strip()
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip('"').strip("'") for item in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


@dataclass
class DevProjectConfig:
    """Serializable Developer Mode training settings."""

    task: str = "classification"
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "Adam"
    epochs: int = 50
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    auto_pause_temp: float = 90.0
    resume_temp: float = 75.0
    loss: str = ""
    metrics: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, project_root: str | Path) -> "DevProjectConfig":
        path = Path(project_root) / "config.yaml"
        if not path.exists():
            return cls()

        values: dict[str, Any] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = _coerce_value(value)

        supported = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: val for key, val in values.items() if key in supported})

    def save(self, project_root: str | Path) -> None:
        path = Path(project_root) / "config.yaml"
        lines = []
        for key, value in self.to_dict().items():
            if isinstance(value, list):
                rendered = "[" + ", ".join(str(item) for item in value) + "]"
            else:
                rendered = str(value)
            lines.append(f"{key}: {rendered}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "epochs": self.epochs,
            "device": self.device,
            "checkpoint_dir": self.checkpoint_dir,
            "auto_pause_temp": self.auto_pause_temp,
            "resume_temp": self.resume_temp,
            "loss": self.loss,
            "metrics": list(self.metrics),
        }

    def effective_loss(self) -> str:
        if self.loss:
            return self.loss
        if self.task == "segmentation":
            return "DiceBCELoss"
        return "CrossEntropyLoss"

    def effective_metrics(self) -> list[str]:
        if self.metrics:
            return list(self.metrics)
        if self.task == "segmentation":
            return ["iou", "dice"]
        return ["accuracy"]

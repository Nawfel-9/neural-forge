"""
utils/config_schema.py
======================
Canonical config schema for Neural Forge Developer Mode projects.
Handles reading, writing, and validating config.yaml files.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DevProjectConfig:
    # ── Task ──────────────────────────────────────────────────────────────
    task: str = "classification"          # "classification" | "segmentation"

    # ── Training hyper-params ─────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    optimizer: str = "adam"               # "adam" | "sgd"
    device: str = "cuda"

    # ── Thermal guard ─────────────────────────────────────────────────────
    auto_pause_temp: int = 90             # pause  when GPU °C >= this
    resume_temp: int = 80                 # resume when GPU °C <  this

    # ── Override hooks (null → use built-in defaults) ─────────────────────
    loss: Optional[str] = None            # kept for documentation; actual
    metrics: Optional[str] = None        # override comes from loss.py / metrics.py

    # ── Paths (relative to project root, written at save time) ────────────
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── Runtime extras (not persisted) ───────────────────────────────────
    _extra: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Defaults per task
    # ------------------------------------------------------------------
    DEFAULT_LOSS = {
        "classification": "CrossEntropyLoss",
        "segmentation":   "DiceBCELoss",
    }
    DEFAULT_METRICS = {
        "classification": ["accuracy"],
        "segmentation":   ["iou", "dice"],
    }

    def effective_loss(self) -> str:
        return self.DEFAULT_LOSS.get(self.task, "CrossEntropyLoss")

    def effective_metrics(self) -> list[str]:
        return self.DEFAULT_METRICS.get(self.task, ["accuracy"])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_extra", None)
        return d

    def save(self, project_root: str | Path) -> Path:
        path = Path(project_root) / "config.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        return path

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, project_root: str | Path) -> "DevProjectConfig":
        path = Path(project_root) / "config.yaml"
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}

        known = {f.name for f in cls.__dataclass_fields__.values()
                 if f.name != "_extra"}
        extra = {k: v for k, v in raw.items() if k not in known}
        filtered = {k: v for k, v in raw.items() if k in known}

        cfg = cls(**filtered)
        cfg._extra = extra
        return cfg

    @classmethod
    def create_example(cls, project_root: str | Path) -> Path:
        """Write an annotated example config.yaml (does NOT overwrite existing)."""
        path = Path(project_root) / "config.yaml"
        if path.exists():
            return path

        example = """\
# ─────────────────────────────────────────────
#  Neural Forge — Developer Mode Config Schema
# ─────────────────────────────────────────────

task: classification          # classification | segmentation

# Training hyper-parameters
epochs: 50
batch_size: 16
learning_rate: 0.001
optimizer: adam               # adam | sgd
device: cuda

# Thermal safety
auto_pause_temp: 90           # pause  training when GPU °C >= this
resume_temp: 80               # resume training when GPU °C <  this

# Override hooks — set to null to use Neural Forge defaults
# If loss.py / metrics.py exist in the project root they take priority.
loss: null
metrics: null

# Output directories (relative to project root)
checkpoint_dir: checkpoints
log_dir: logs
"""
        path.write_text(example, encoding="utf-8")
        return path

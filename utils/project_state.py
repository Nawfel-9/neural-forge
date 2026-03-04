"""
project_state.py
================
Shared mutable state object passed across the 3-window pipeline.

Every window reads/writes to a single ProjectState instance so that
data, model config, and training artefacts stay in sync without tight
coupling between windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class ProjectState:
    """Container for all cross-window state."""

    # ---- Window 1: Data ----
    dataframe: Optional[pd.DataFrame] = None
    target_column: str = ""
    problem_type: str = "classification"  # "classification" | "regression"
    split_config: dict = field(
        default_factory=lambda: {"method": "percentage", "ratio": 0.8}
    )

    # ---- Window 2: Model ----
    blueprint: list[dict] = field(default_factory=list)

    # These are populated after a successful build/ghost-run (Phase 3+)
    model: Any = None               # nn.Module once built
    dummy_tensor: Any = None         # torch.Tensor for ghost run / export

    # ---- Window 3: Training ----
    hyperparams: dict = field(
        default_factory=lambda: {
            "lr": 0.001,
            "epochs": 50,
            "batch_size": 32,
        }
    )
    device: str = "cpu"              # "cpu" | "cuda" | "mps"

    # ---- Convenience ----
    def input_features(self) -> int:
        """Return the number of input features (columns minus target)."""
        if self.dataframe is None or not self.target_column:
            return 0
        return self.dataframe.shape[1] - 1

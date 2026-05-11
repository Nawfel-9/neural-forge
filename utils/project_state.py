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

try:
    from backend.training_config import DEFAULT_LOSS, DEFAULT_OPTIMIZER
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    DEFAULT_LOSS = {"classification": "CrossEntropyLoss", "regression": "MSELoss"}
    DEFAULT_OPTIMIZER = "Adam"


@dataclass
class ProjectState:
    """Container for all cross-window state."""

    # ---- Window 1: Data ----
    dataframe: Optional[pd.DataFrame] = None
    target_column: str = ""
    problem_type: str = "classification"  # "classification" | "regression"
    pipeline: Any = None  # Stores the preprocessing DataPipeline
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
    loss_fn_name: str = field(default_factory=lambda: DEFAULT_LOSS["classification"])
    optimizer_name: str = field(default_factory=lambda: DEFAULT_OPTIMIZER)

    # ---- Developer Mode ----
    dev_project_path: str = ""        # Imported PyTorch project folder

    # ---- Convenience ----
    def input_features(self) -> int:
        """Return the number of input features (columns minus target)."""
        if self.dataframe is None or not self.target_column:
            return 0
        return self.dataframe.shape[1] - 1

    def output_classes(self) -> int:
        """Return the required number of output neurons based on problem type."""
        if self.dataframe is None or not self.target_column:
            return 1
        if self.problem_type == "classification":
            import numpy as np
            return len(np.unique(self.dataframe[self.target_column]))
        return 1

"""
training_config.py
==================
Registry of supported loss functions and optimizers.

This module is completely decoupled from the UI — it never touches Qt.
It is the single source of truth consumed by:

* ``ui.window_training.TrainingWindow``  — to populate dropdowns
* ``workers.training_worker.TrainingWorker`` — to instantiate the objects

Design
------
Loss functions are filtered by ``problem_type`` so the UI only shows
relevant options.  Optimizers are problem-agnostic and always shown in full.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Loss registry
# ─────────────────────────────────────────────────────────────────────────────

#: Maps problem type → ordered list of (display_name, nn.Module class).
#: Order matters: first entry is the default selection.
_LOSS_REGISTRY: dict[str, list[tuple[str, type[nn.Module]]]] = {
    "classification": [
        ("CrossEntropyLoss", nn.CrossEntropyLoss),
        ("BCEWithLogitsLoss", nn.BCEWithLogitsLoss),
        ("NLLLoss", nn.NLLLoss),
    ],
    "regression": [
        ("MSELoss", nn.MSELoss),
        ("L1Loss", nn.L1Loss),
        ("SmoothL1Loss", nn.SmoothL1Loss),
    ],
}

#: Default loss name per problem type (first entry in each list).
DEFAULT_LOSS: dict[str, str] = {
    pt: entries[0][0] for pt, entries in _LOSS_REGISTRY.items()
}


def get_losses_for(problem_type: str) -> list[str]:
    """
    Return the display names of all loss functions available for *problem_type*.

    Parameters
    ----------
    problem_type : str
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    list[str]
        Ordered list of display names, e.g. ``["CrossEntropyLoss", ...]``.

    Raises
    ------
    ValueError
        If *problem_type* is not recognised.
    """
    if problem_type not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown problem type '{problem_type}'. "
            f"Supported: {list(_LOSS_REGISTRY.keys())}."
        )
    return [name for name, _ in _LOSS_REGISTRY[problem_type]]


def build_loss(name: str) -> nn.Module:
    """
    Instantiate a loss function by its display name.

    Parameters
    ----------
    name : str
        Display name, e.g. ``"CrossEntropyLoss"``.

    Returns
    -------
    nn.Module
        A fresh loss module instance.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    """
    for entries in _LOSS_REGISTRY.values():
        for display_name, cls in entries:
            if display_name == name:
                return cls()
    raise ValueError(
        f"Unknown loss function '{name}'. "
        f"Supported: {[n for e in _LOSS_REGISTRY.values() for n, _ in e]}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer registry
# ─────────────────────────────────────────────────────────────────────────────

#: Ordered list of (display_name, optim class).  First entry is the default.
_OPTIMIZER_REGISTRY: list[tuple[str, type[optim.Optimizer]]] = [
    ("Adam", optim.Adam),
    ("AdamW", optim.AdamW),
    ("SGD", optim.SGD),
    ("RMSprop", optim.RMSprop),
]

#: Default optimizer name.
DEFAULT_OPTIMIZER: str = _OPTIMIZER_REGISTRY[0][0]


def get_all_optimizers() -> list[str]:
    """
    Return the display names of all supported optimizers.

    Returns
    -------
    list[str]
        e.g. ``["Adam", "AdamW", "SGD", "RMSprop"]``.
    """
    return [name for name, _ in _OPTIMIZER_REGISTRY]


def build_optimizer(
    name: str,
    parameters: Any,
    lr: float = 0.001,
) -> optim.Optimizer:
    """
    Instantiate an optimizer by its display name.

    Parameters
    ----------
    name : str
        Display name, e.g. ``"AdamW"``.
    parameters : iterable
        Model parameters, typically ``model.parameters()``.
    lr : float
        Learning rate (default ``0.001``).

    Returns
    -------
    optim.Optimizer
        A configured optimizer instance.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    """
    for display_name, cls in _OPTIMIZER_REGISTRY:
        if display_name == name:
            return cls(parameters, lr=lr)
    raise ValueError(
        f"Unknown optimizer '{name}'. "
        f"Supported: {get_all_optimizers()}."
    )

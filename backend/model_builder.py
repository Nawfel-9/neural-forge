"""
model_builder.py
================
Translates a layer blueprint (list of dicts) into a PyTorch ``nn.Sequential``
model, and validates it with a ghost run (dummy forward pass).

This module never touches Qt — it is a pure PyTorch backend.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Activation factory
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "LeakyReLU": nn.LeakyReLU,
    "Softmax": nn.Softmax,
}


def _make_activation(name: str) -> nn.Module | None:
    """Return an activation module instance, or ``None`` for 'None'."""
    if name == "None" or name not in _ACTIVATION_MAP:
        return None
    if name == "Softmax":
        return nn.Softmax(dim=-1)
    return _ACTIVATION_MAP[name]()


# ─────────────────────────────────────────────────────────────────────────────
# Layer translator
# ─────────────────────────────────────────────────────────────────────────────

def _translate_layer(
    layer_cfg: dict,
    prev_out: int | None,
    is_first_linear: bool,
) -> tuple[list[nn.Module], int | None]:
    """
    Convert a single layer config dict into one or more ``nn.Module`` objects.

    Parameters
    ----------
    layer_cfg : dict
        A single layer from the blueprint, e.g. ``{"type": "Linear", ...}``.
    prev_out : int | None
        The number of output features from the previous layer, or ``None``
        if unknown (will use ``nn.LazyLinear``).
    is_first_linear : bool
        True if this is the first Linear layer in the blueprint (uses LazyLinear).

    Returns
    -------
    (modules, new_out)
        ``modules`` is a list of 1–2 ``nn.Module`` objects (layer + optional
        activation). ``new_out`` is the output feature count after this layer,
        or ``None`` if unknown.
    """
    ltype = layer_cfg["type"]
    modules: list[nn.Module] = []
    new_out: int | None = prev_out

    if ltype == "Linear":
        neurons = layer_cfg["neurons"]
        if is_first_linear or prev_out is None:
            # Use LazyLinear — input features auto-inferred on first forward
            modules.append(nn.LazyLinear(neurons))
        else:
            modules.append(nn.Linear(prev_out, neurons))
        new_out = neurons

        # Optional activation
        act = _make_activation(layer_cfg.get("activation", "None"))
        if act is not None:
            modules.append(act)

    elif ltype == "Conv1d":
        out_ch = layer_cfg["out_channels"]
        ks = layer_cfg["kernel_size"]
        stride = layer_cfg.get("stride", 1)
        padding = layer_cfg.get("padding", 0)
        if prev_out is None:
            modules.append(nn.LazyConv1d(out_ch, ks, stride=stride, padding=padding))
        else:
            modules.append(nn.Conv1d(prev_out, out_ch, ks, stride=stride, padding=padding))
        new_out = out_ch

    elif ltype == "MaxPool1d":
        ks = layer_cfg["kernel_size"]
        stride = layer_cfg.get("stride", ks)
        modules.append(nn.MaxPool1d(ks, stride=stride))
        # Channel count unchanged, spatial dims shrink (tracked by ghost run)

    elif ltype == "AvgPool1d":
        ks = layer_cfg["kernel_size"]
        stride = layer_cfg.get("stride", ks)
        modules.append(nn.AvgPool1d(ks, stride=stride))

    elif ltype == "Flatten":
        modules.append(nn.Flatten())
        new_out = None  # Unknown until ghost run resolves it

    elif ltype == "BatchNorm1d":
        if prev_out is not None:
            modules.append(nn.BatchNorm1d(prev_out))
        else:
            modules.append(nn.LazyBatchNorm1d())

    elif ltype == "Dropout":
        rate = layer_cfg.get("rate", 0.3)
        modules.append(nn.Dropout(p=rate))

    else:
        raise ValueError(f"Unknown layer type: '{ltype}'")

    return modules, new_out


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(blueprint: list[dict]) -> nn.Sequential:
    """
    Convert a validated blueprint into an ``nn.Sequential`` model.

    Uses ``nn.LazyLinear`` / ``nn.LazyConv1d`` / ``nn.LazyBatchNorm1d``
    for the first layer of each type so input dimensions are auto-inferred
    during the ghost run.

    Parameters
    ----------
    blueprint : list[dict]
        A validated layer blueprint.

    Returns
    -------
    nn.Sequential
        The assembled model (un-initialised until a ghost run).

    Raises
    ------
    ValueError
        If a layer type is unknown.
    """
    all_modules: list[nn.Module] = []
    prev_out: int | None = None
    first_linear_seen = False

    for cfg in blueprint:
        ltype = cfg["type"]
        is_first_linear = (ltype == "Linear" and not first_linear_seen)

        modules, prev_out = _translate_layer(cfg, prev_out, is_first_linear)
        all_modules.extend(modules)

        if ltype == "Linear":
            first_linear_seen = True

    return nn.Sequential(*all_modules)


# ─────────────────────────────────────────────────────────────────────────────
# Ghost run
# ─────────────────────────────────────────────────────────────────────────────

def ghost_run(
    model: nn.Sequential,
    input_features: int,
    batch_size: int = 2,
) -> tuple[bool, torch.Tensor | None, str]:
    """
    Perform a forward pass with a dummy tensor to validate the model.

    This materialises all ``Lazy*`` modules and catches shape mismatches
    before the user starts a real training run.

    Parameters
    ----------
    model : nn.Sequential
        The model built by :func:`build_model`.
    input_features : int
        Number of input features (columns minus target).
    batch_size : int
        Batch size for the dummy tensor (default 2).

    Returns
    -------
    (success, dummy_output, message)
        ``success`` is True if the forward pass succeeded.
        ``dummy_output`` is the output tensor (or None on failure).
        ``message`` describes the result or the error.
    """
    try:
        model.eval()
        dummy = torch.randn(batch_size, input_features)

        with torch.no_grad():
            output = model(dummy)

        out_shape = tuple(output.shape)
        return True, output, (
            f"Ghost run passed! "
            f"Input: ({batch_size}, {input_features}) → "
            f"Output: {out_shape}"
        )

    except Exception as exc:
        return False, None, (
            f"Ghost run failed: {type(exc).__name__}: {exc}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build + ghost run in one step
# ─────────────────────────────────────────────────────────────────────────────

def build_and_validate(
    blueprint: list[dict],
    input_features: int,
    batch_size: int = 2,
) -> tuple[nn.Sequential | None, torch.Tensor | None, bool, str]:
    """
    Build a model from a blueprint and validate it with a ghost run.

    Parameters
    ----------
    blueprint : list[dict]
        Validated blueprint.
    input_features : int
        Number of input features.
    batch_size : int
        Batch size for ghost run.

    Returns
    -------
    (model, dummy_output, success, message)
    """
    try:
        model = build_model(blueprint)
    except Exception as exc:
        return None, None, False, f"Build failed: {type(exc).__name__}: {exc}"

    success, dummy_output, msg = ghost_run(model, input_features, batch_size)
    if not success:
        return None, None, False, msg

    return model, dummy_output, True, msg

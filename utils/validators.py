"""
validators.py
=============
Validation helpers for blueprints and data.
All functions return a tuple (is_valid: bool, error_message: str).
"""

from __future__ import annotations

VALID_LAYER_TYPES = {
    "Linear", "Conv1d", "MaxPool1d", "AvgPool1d",
    "Flatten", "BatchNorm1d", "Dropout",
}
VALID_ACTIVATIONS = {"None", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Softmax"}

# Layer types that can serve as the final output layer
_OUTPUT_LAYER_TYPES = {"Linear"}


def validate_blueprint(blueprint: list[dict]) -> tuple[bool, str]:
    """
    Check that a blueprint is structurally sound.

    Rules
    -----
    1. The blueprint must not be empty.
    2. Every entry must have a ``"type"`` key with a known value.
    3. At least one ``Linear`` layer must be present.
    4. ``Dropout.rate`` must be in (0, 1).
    5. The **last** layer must be ``Linear`` (the output layer).
    6. ``Linear.neurons`` must be a positive integer.
    7. ``Linear.activation`` must be a known value.
    8. ``Conv1d.out_channels`` must be a positive integer.
    9. ``Conv1d.kernel_size`` must be a positive integer.
    10. ``MaxPool1d / AvgPool1d`` kernel_size and stride must be positive.

    Returns
    -------
    (True, "")  if valid, else (False, human-readable reason).
    """
    if not blueprint:
        return False, "The blueprint is empty. Add at least one layer."

    if not isinstance(blueprint, list):
        return False, "Blueprint must be a list of layer dictionaries."

    has_linear = False

    for i, layer in enumerate(blueprint, start=1):
        if not isinstance(layer, dict):
            return False, f"Layer {i} is not a dictionary."

        ltype = layer.get("type")
        if ltype not in VALID_LAYER_TYPES:
            return False, (
                f"Layer {i}: unknown type '{ltype}'. "
                f"Supported: {', '.join(sorted(VALID_LAYER_TYPES))}."
            )

        # ── Linear ──────────────────────────────────────────────────────
        if ltype == "Linear":
            has_linear = True
            neurons = layer.get("neurons", 0)
            if not isinstance(neurons, int) or neurons < 1:
                return False, f"Layer {i}: 'neurons' must be a positive integer."
            activation = layer.get("activation", "None")
            if activation not in VALID_ACTIVATIONS:
                return False, (
                    f"Layer {i}: unknown activation '{activation}'. "
                    f"Supported: {', '.join(sorted(VALID_ACTIVATIONS))}."
                )

        # ── Conv1d ──────────────────────────────────────────────────────
        if ltype == "Conv1d":
            out_ch = layer.get("out_channels", 0)
            if not isinstance(out_ch, int) or out_ch < 1:
                return False, f"Layer {i}: 'out_channels' must be a positive integer."
            ks = layer.get("kernel_size", 0)
            if not isinstance(ks, int) or ks < 1:
                return False, f"Layer {i}: 'kernel_size' must be a positive integer."
            stride = layer.get("stride", 1)
            if not isinstance(stride, int) or stride < 1:
                return False, f"Layer {i}: 'stride' must be a positive integer."
            padding = layer.get("padding", 0)
            if not isinstance(padding, int) or padding < 0:
                return False, f"Layer {i}: 'padding' must be a non-negative integer."

        # ── MaxPool1d / AvgPool1d ───────────────────────────────────────
        if ltype in ("MaxPool1d", "AvgPool1d"):
            ks = layer.get("kernel_size", 0)
            if not isinstance(ks, int) or ks < 1:
                return False, f"Layer {i}: 'kernel_size' must be a positive integer."
            stride = layer.get("stride", 1)
            if not isinstance(stride, int) or stride < 1:
                return False, f"Layer {i}: 'stride' must be a positive integer."

        # ── Dropout ─────────────────────────────────────────────────────
        if ltype == "Dropout":
            rate = layer.get("rate", 0.0)
            if not (0.0 < rate < 1.0):
                return False, (
                    f"Layer {i}: dropout rate must be between 0 and 1 (exclusive), "
                    f"got {rate}."
                )

    if not has_linear:
        return False, "The blueprint must contain at least one Linear layer."

    last_type = blueprint[-1].get("type")
    if last_type not in _OUTPUT_LAYER_TYPES:
        return False, (
            f"The last layer must be Linear (the output layer), "
            f"but got '{last_type}'."
        )

    return True, ""

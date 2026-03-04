"""
validators.py
=============
Validation helpers for blueprints and data.
All functions return a tuple (is_valid: bool, error_message: str).
"""

from __future__ import annotations

VALID_LAYER_TYPES = {"Linear", "Dropout", "BatchNorm1d"}
VALID_ACTIVATIONS = {"None", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Softmax"}


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
    if last_type != "Linear":
        return False, (
            f"The last layer must be Linear (the output layer), "
            f"but got '{last_type}'."
        )

    return True, ""

"""
blueprint_io.py
===============
JSON save / load helpers for blueprint files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_blueprint(blueprint: list[dict], filepath: str | Path) -> None:
    """
    Write a blueprint (list of layer dicts) to a JSON file.

    Parameters
    ----------
    blueprint : list[dict]
        Layer configuration list.
    filepath : str | Path
        Destination ``.json`` path.

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "layers": blueprint}, fh, indent=2)


def load_blueprint(filepath: str | Path) -> list[dict]:
    """
    Read a blueprint from a JSON file.

    The file must be a JSON object with a ``"layers"`` key containing a
    list of layer dictionaries, **or** a bare list (legacy support).

    Parameters
    ----------
    filepath : str | Path
        Source ``.json`` path.

    Returns
    -------
    list[dict]
        Parsed layer configuration list.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON structure is invalid.
    json.JSONDecodeError
        If the file contains malformed JSON.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as fh:
        data: Any = json.load(fh)

    # Envelope format: {"version": 1, "layers": [...]}
    if isinstance(data, dict):
        layers = data.get("layers")
        if layers is None:
            raise ValueError(
                "Blueprint JSON must contain a 'layers' key with a list of layer dicts."
            )
        if not isinstance(layers, list):
            raise ValueError("'layers' must be a list.")
        return layers

    # Legacy / bare-list format: [...]
    if isinstance(data, list):
        return data

    raise ValueError(
        "Blueprint JSON must be an object with a 'layers' key, or a bare list."
    )

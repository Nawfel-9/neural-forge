"""
test_phase1.py
==============
Automated smoke tests for Phase 1: Layer builder, blueprint I/O, validation.

Run with:
    python -m pytest tests/test_phase1.py -v

These tests do **not** require a display or a running QApplication for the
pure-logic tests.  The UI-level tests use a QApplication fixture.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.blueprint_io import load_blueprint, save_blueprint
from utils.validators import validate_blueprint
from utils.project_state import ProjectState


# ─────────────────────────────────────────────────────────────────────────────
# 1. ProjectState defaults
# ─────────────────────────────────────────────────────────────────────────────
class TestProjectState:
    def test_defaults(self):
        state = ProjectState()
        assert state.dataframe is None
        assert state.target_column == ""
        assert state.problem_type == "classification"
        assert state.device == "cpu"
        assert state.hyperparams["lr"] == 0.001
        assert state.hyperparams["batch_size"] == 32

    def test_input_features_no_data(self):
        state = ProjectState()
        assert state.input_features() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Blueprint Validation
# ─────────────────────────────────────────────────────────────────────────────
class TestValidation:
    def test_empty_blueprint(self):
        ok, msg = validate_blueprint([])
        assert not ok
        assert "empty" in msg.lower()

    def test_valid_single_linear(self):
        bp = [{"type": "Linear", "neurons": 10, "activation": "None"}]
        ok, msg = validate_blueprint(bp)
        assert ok, msg

    def test_valid_multi_layer(self):
        bp = [
            {"type": "Linear", "neurons": 128, "activation": "ReLU"},
            {"type": "Dropout", "rate": 0.3},
            {"type": "BatchNorm1d"},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        ok, msg = validate_blueprint(bp)
        assert ok, msg

    def test_invalid_last_layer_dropout(self):
        bp = [
            {"type": "Linear", "neurons": 64, "activation": "ReLU"},
            {"type": "Dropout", "rate": 0.5},
        ]
        ok, msg = validate_blueprint(bp)
        assert not ok
        assert "last layer" in msg.lower()

    def test_no_linear_layer(self):
        bp = [{"type": "Dropout", "rate": 0.5}]
        ok, msg = validate_blueprint(bp)
        assert not ok

    def test_invalid_dropout_rate(self):
        bp = [
            {"type": "Dropout", "rate": 0.0},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        ok, msg = validate_blueprint(bp)
        assert not ok
        assert "dropout rate" in msg.lower()

    def test_unknown_type(self):
        bp = [{"type": "LSTM", "neurons": 64}]
        ok, msg = validate_blueprint(bp)
        assert not ok
        assert "unknown type" in msg.lower()

    def test_unknown_activation(self):
        bp = [{"type": "Linear", "neurons": 64, "activation": "Swish"}]
        ok, msg = validate_blueprint(bp)
        assert not ok
        assert "unknown activation" in msg.lower()

    def test_zero_neurons(self):
        bp = [{"type": "Linear", "neurons": 0, "activation": "None"}]
        ok, msg = validate_blueprint(bp)
        assert not ok
        assert "positive integer" in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Blueprint JSON save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────
class TestBlueprintIO:
    def test_round_trip(self, tmp_path: Path):
        original = [
            {"type": "Linear", "neurons": 128, "activation": "ReLU"},
            {"type": "Dropout", "rate": 0.3},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        fpath = tmp_path / "test_bp.json"
        save_blueprint(original, fpath)
        loaded = load_blueprint(fpath)
        assert loaded == original

    def test_envelope_has_version(self, tmp_path: Path):
        fpath = tmp_path / "test_bp.json"
        save_blueprint([{"type": "Linear", "neurons": 1, "activation": "None"}], fpath)
        with open(fpath) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert "layers" in data

    def test_load_bare_list(self, tmp_path: Path):
        """Legacy support: a bare JSON list should also load."""
        fpath = tmp_path / "legacy.json"
        with open(fpath, "w") as f:
            json.dump([{"type": "Linear", "neurons": 5, "activation": "None"}], f)
        loaded = load_blueprint(fpath)
        assert len(loaded) == 1

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_blueprint("/nonexistent/path.json")

    def test_load_invalid_json(self, tmp_path: Path):
        fpath = tmp_path / "bad.json"
        fpath.write_text("not json at all", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_blueprint(fpath)

    def test_load_missing_layers_key(self, tmp_path: Path):
        fpath = tmp_path / "no_layers.json"
        with open(fpath, "w") as f:
            json.dump({"version": 1}, f)
        with pytest.raises(ValueError, match="layers"):
            load_blueprint(fpath)


# ─────────────────────────────────────────────────────────────────────────────
# 4. UI-level tests (require QApplication)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication(sys.argv)

    from ui.layer_row import LayerRow
    from ui.window_model import ModelBuilderWindow

    _HAS_QT = True
except ImportError:
    _HAS_QT = False


@pytest.mark.skipif(not _HAS_QT, reason="PyQt6 not installed")
class TestLayerRowWidget:
    def test_default_config(self):
        row = LayerRow(index=0)
        cfg = row.get_config()
        assert cfg["type"] == "Linear"
        assert cfg["neurons"] == 64
        assert cfg["activation"] == "None"

    def test_set_config_linear(self):
        row = LayerRow(index=0)
        row.set_config({"type": "Linear", "neurons": 256, "activation": "ReLU"})
        cfg = row.get_config()
        assert cfg == {"type": "Linear", "neurons": 256, "activation": "ReLU"}

    def test_set_config_dropout(self):
        row = LayerRow(index=0)
        row.set_config({"type": "Dropout", "rate": 0.5})
        cfg = row.get_config()
        assert cfg == {"type": "Dropout", "rate": 0.5}

    def test_set_config_batchnorm(self):
        row = LayerRow(index=0)
        row.set_config({"type": "BatchNorm1d"})
        cfg = row.get_config()
        assert cfg == {"type": "BatchNorm1d"}

    def test_index_update(self):
        row = LayerRow(index=0)
        row.set_index(5)
        assert row.row_index == 5
        assert row.lbl_index.text() == "#6"


@pytest.mark.skipif(not _HAS_QT, reason="PyQt6 not installed")
class TestModelBuilderWindow:
    def _make_window(self, with_data=False) -> ModelBuilderWindow:
        import pandas as pd
        state = ProjectState()
        if with_data:
            state.dataframe = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
            state.target_column = "target"
        return ModelBuilderWindow(project_state=state)

    def test_starts_with_one_layer(self):
        w = self._make_window()
        assert len(w._layer_rows) == 1

    def test_add_layer(self):
        w = self._make_window()
        w._add_layer_row()
        assert len(w._layer_rows) == 2

    def test_remove_prevents_zero_layers(self):
        from unittest.mock import patch
        w = self._make_window()
        # Mock QMessageBox.warning so the modal dialog doesn't block
        with patch("ui.window_model.QMessageBox.warning"):
            w._layer_rows[0].remove_requested.emit(0)
        assert len(w._layer_rows) == 1

    def test_get_architecture(self):
        w = self._make_window()
        w._add_layer_row({"type": "Linear", "neurons": 32, "activation": "ReLU"})
        arch = w.get_architecture()
        assert len(arch) == 2
        assert arch[1]["neurons"] == 32

    def test_sync_valid(self):
        w = self._make_window(with_data=True)
        assert w.sync_to_state() is True
        assert len(w.state.blueprint) == 1
        assert w.state.model is not None

    def test_clear_and_rebuild(self):
        w = self._make_window()
        w._add_layer_row({"type": "Dropout", "rate": 0.2})
        w._add_layer_row({"type": "Linear", "neurons": 10, "activation": "None"})
        assert len(w._layer_rows) == 3
        w._clear_all_rows()
        assert len(w._layer_rows) == 0

"""
test_phase2.py
==============
Automated tests for Phase 2: Data loading, cleaning, splitting, and table model.

Run with:
    python -m pytest tests/test_phase2.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_handler import (
    NaNStrategy,
    clean_dataframe,
    count_input_features,
    detect_columns,
    get_kfold_splitter,
    load_csv,
    split_data_percentage,
)
from utils.project_state import ProjectState


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_csv(tmp_path: Path, name: str, content: str) -> Path:
    """Write a CSV string to a temp file and return the path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _sample_df() -> pd.DataFrame:
    """A small numeric DataFrame with one NaN."""
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "feature_b": [10, 20, 30, 40, 50],
        "label": [0, 1, 0, 1, 0],
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. CSV Loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCSV:
    def test_load_valid_csv(self, tmp_path: Path):
        csv = _make_csv(tmp_path, "ok.csv", "a,b,c\n1,2,3\n4,5,6\n")
        df = load_csv(str(csv))
        assert df.shape == (2, 3)
        assert list(df.columns) == ["a", "b", "c"]

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/no/such/file.csv")

    def test_load_empty_file(self, tmp_path: Path):
        csv = _make_csv(tmp_path, "empty.csv", "")
        with pytest.raises(Exception):
            load_csv(str(csv))

    def test_load_malformed_csv(self, tmp_path: Path):
        # Just a header with no data rows → still has columns, 0 rows
        csv = _make_csv(tmp_path, "header_only.csv", "a,b,c\n")
        # This should raise ValueError because the DataFrame is empty
        with pytest.raises(ValueError, match="no usable data"):
            load_csv(str(csv))

    def test_load_csv_with_nans(self, tmp_path: Path):
        csv = _make_csv(tmp_path, "nans.csv", "a,b\n1,2\n,4\n5,\n")
        df = load_csv(str(csv))
        assert df.isna().sum().sum() == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cleaning
# ─────────────────────────────────────────────────────────────────────────────

class TestCleaning:
    def test_fill_mean_numeric(self):
        df = _sample_df()
        cleaned, report = clean_dataframe(df, NaNStrategy.FILL_MEAN)
        assert report["nan_count_before"] == 1
        assert report["nan_count_after"] == 0
        # NaN in feature_a should be filled with mean of [1, 2, 4, 5] = 3.0
        assert cleaned.loc[2, "feature_a"] == 3.0

    def test_drop_rows(self):
        df = _sample_df()
        cleaned, report = clean_dataframe(df, NaNStrategy.DROP_ROWS)
        assert report["nan_count_after"] == 0
        assert report["rows_after"] == 4  # 1 row dropped

    def test_no_nans_passthrough(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cleaned, report = clean_dataframe(df, NaNStrategy.FILL_MEAN)
        assert report["nan_count_before"] == 0
        assert report["nan_count_after"] == 0
        assert len(cleaned) == 3

    def test_fill_mean_categorical(self):
        df = pd.DataFrame({
            "name": ["Alice", None, "Charlie", "Alice"],
            "score": [10, 20, 30, 40],
        })
        cleaned, _ = clean_dataframe(df, NaNStrategy.FILL_MEAN)
        # Mode of ["Alice", "Charlie", "Alice"] is "Alice"
        assert cleaned.loc[1, "name"] == "Alice"

    def test_original_not_modified(self):
        df = _sample_df()
        original_nan_count = int(df.isna().sum().sum())
        clean_dataframe(df, NaNStrategy.FILL_MEAN)
        assert int(df.isna().sum().sum()) == original_nan_count


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature detection
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatures:
    def test_detect_columns(self):
        df = _sample_df()
        cols = detect_columns(df)
        assert cols == ["feature_a", "feature_b", "label"]

    def test_count_input_features(self):
        df = _sample_df()
        assert count_input_features(df, "label") == 2

    def test_count_features_invalid_target(self):
        df = _sample_df()
        with pytest.raises(ValueError, match="not found"):
            count_input_features(df, "nonexistent_column")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Splitting
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitting:
    def test_percentage_split_shapes(self):
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100, 200),
            "target": [0, 1] * 50,
        })
        X_train, X_val, y_train, y_val = split_data_percentage(df, "target", ratio=0.8)
        assert len(X_train) == 80
        assert len(X_val) == 20
        assert len(y_train) == 80
        assert len(y_val) == 20
        assert "target" not in X_train.columns

    def test_percentage_split_invalid_target(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError):
            split_data_percentage(df, "nope")

    def test_kfold_splitter(self):
        kf = get_kfold_splitter(k=5)
        assert kf.n_splits == 5
        splits = list(kf.split(range(100)))
        assert len(splits) == 5
        # Each fold: train ≈ 80, val ≈ 20
        for train_idx, val_idx in splits:
            assert len(val_idx) == 20


# ─────────────────────────────────────────────────────────────────────────────
# 5. Table model (UI-level)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    _app = QApplication.instance() or QApplication(sys.argv)

    from ui.data_table_view import PandasTableModel, DataPreviewTable

    _HAS_QT = True
except ImportError:
    _HAS_QT = False


@pytest.mark.skipif(not _HAS_QT, reason="PyQt6 not installed")
class TestPandasTableModel:
    def test_row_column_count(self):
        df = _sample_df()
        model = PandasTableModel(df)
        assert model.rowCount() == 5
        assert model.columnCount() == 3

    def test_max_rows(self):
        df = _sample_df()
        model = PandasTableModel(df, max_rows=2)
        assert model.rowCount() == 2

    def test_data_display(self):
        df = pd.DataFrame({"x": [42]})
        model = PandasTableModel(df)
        idx = model.index(0, 0)
        assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "42"

    def test_header_data(self):
        df = pd.DataFrame({"my_col": [1]})
        model = PandasTableModel(df)
        h = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        assert h == "my_col"

    def test_update_dataframe(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"x": [10, 20, 30]})
        model = PandasTableModel(df1)
        assert model.rowCount() == 2
        model.update_dataframe(df2)
        assert model.rowCount() == 3


@pytest.mark.skipif(not _HAS_QT, reason="PyQt6 not installed")
class TestDataPreviewTable:
    def test_set_and_clear(self):
        widget = DataPreviewTable(max_preview_rows=3)
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        widget.set_dataframe(df)
        # Model should show only 3 rows
        assert widget._model.rowCount() == 3
        widget.clear()
        assert widget._model.rowCount() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. ProjectState integration
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectStateData:
    def test_input_features_with_data(self):
        state = ProjectState()
        state.dataframe = _sample_df()
        state.target_column = "label"
        assert state.input_features() == 2

    def test_split_config_default(self):
        state = ProjectState()
        assert state.split_config["method"] == "percentage"
        assert state.split_config["ratio"] == 0.8

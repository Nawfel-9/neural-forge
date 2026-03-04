"""
data_table_view.py
==================
A QTableView backed by a lightweight QAbstractTableModel that wraps a
Pandas DataFrame.  Designed for read-only preview (e.g., first N rows).
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import QHeaderView, QTableView, QVBoxLayout, QWidget


class PandasTableModel(QAbstractTableModel):
    """
    Read-only Qt table model backed by a :class:`pd.DataFrame`.

    Parameters
    ----------
    df : pd.DataFrame
        The data to display.
    max_rows : int | None
        If set, only the first *max_rows* rows are exposed.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        max_rows: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._df = df.head(max_rows) if max_rows else df.copy()

    # ── Required overrides ──────────────────────────────────────────────────

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            # Convert numpy types to native Python for display
            try:
                return str(value)
            except Exception:
                return ""
        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)  # 1-indexed row numbers

    def update_dataframe(self, df: pd.DataFrame, max_rows: int | None = None) -> None:
        """Replace the underlying data and refresh the view."""
        self.beginResetModel()
        self._df = df.head(max_rows) if max_rows else df.copy()
        self.endResetModel()


class DataPreviewTable(QWidget):
    """
    Convenience widget wrapping a QTableView + PandasTableModel.

    Parameters
    ----------
    parent : QWidget | None
        Parent widget.
    max_preview_rows : int
        Number of rows to show in the preview (default 5).
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        max_preview_rows: int = 5,
    ) -> None:
        super().__init__(parent)
        self._max_rows = max_preview_rows

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table_view = QTableView()
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table_view.verticalHeader().setDefaultSectionSize(28)
        layout.addWidget(self.table_view)

        # Start with an empty model
        self._model = PandasTableModel(pd.DataFrame())
        self.table_view.setModel(self._model)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Display the first *max_preview_rows* rows of *df*."""
        self._model.update_dataframe(df, max_rows=self._max_rows)
        # Auto-resize columns to fit content
        for col in range(self._model.columnCount()):
            self.table_view.resizeColumnToContents(col)

    def clear(self) -> None:
        """Clear the preview."""
        self._model.update_dataframe(pd.DataFrame())

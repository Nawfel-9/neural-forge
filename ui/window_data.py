"""
window_data.py
==============
Window 1 — Data Loading & Preprocessing.

Lets the user:
  • Load a CSV file
  • Preview the first 5 rows
  • Pick the target column (defaults to last)
  • Choose problem type (Classification / Regression)
  • Configure train/validation split (percentage or k-fold)
  • Clean NaN values
  • Proceed to Window 2 (Model Builder)
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtCore import Qt, QThread
from workers.data_loader_worker import DataLoaderWorker
from ui.data_table_view import DataPreviewTable
from utils.project_state import ProjectState

class DataWindow(QMainWindow):
    """
    Window 1 of the pipeline — data loading and preprocessing.

    Parameters
    ----------
    project_state : ProjectState
        Shared state, written to when the user clicks "Next".
    on_next : callable | None
        Callback invoked after successful validation to open Window 2.
    """

    def __init__(
        self,
        project_state: ProjectState,
        on_next=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = project_state
        self._on_next_callback = on_next
        self._raw_df = None  # keeps original before cleaning
        self._cleaned_df = None
        self._worker_thread = None
        self._worker = None

        self._init_window()
        self._build_ui()

    # ── Window setup ────────────────────────────────────────────────────────
    def _init_window(self) -> None:
        self.setWindowTitle("Neural Network Builder — Data Loading")
        self.setMinimumSize(860, 620)
        self.resize(940, 700)

    # ── UI construction ─────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # ── Header ──────────────────────────────────────────────────────────
        header = QLabel("📊  Data Loading & Preprocessing")
        header.setStyleSheet("font-size: 20px; font-weight: 700;")
        root.addWidget(header)

        # ── Load CSV row ────────────────────────────────────────────────────
        load_row = QHBoxLayout()
        self.btn_load = QPushButton("📂  Load CSV")
        self.btn_load.setProperty("class", "primary")
        self.btn_load.setMinimumHeight(36)
        self.btn_load.clicked.connect(self._load_csv)
        load_row.addWidget(self.btn_load)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setStyleSheet("color: #8b949e;")
        load_row.addWidget(self.lbl_file, stretch=1)
        root.addLayout(load_row)

        # ── Data info bar ───────────────────────────────────────────────────
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #8b949e; font-size: 12px;")
        root.addWidget(self.lbl_info)

        # ── Table preview ───────────────────────────────────────────────────
        self.preview = DataPreviewTable(max_preview_rows=5)
        self.preview.setMinimumHeight(160)
        root.addWidget(self.preview, stretch=1)

        # ── Configuration panels (side by side) ─────────────────────────────
        config_row = QHBoxLayout()
        config_row.setSpacing(14)

        # Left: Target column + Problem type
        config_row.addWidget(self._build_target_panel())

        # Right: Split config + NaN handling
        config_row.addWidget(self._build_split_panel())

        root.addLayout(config_row)

        # ── Bottom button bar ───────────────────────────────────────────────
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()

        self.btn_next = QPushButton("Next  →  Model Builder")
        self.btn_next.setProperty("class", "primary")
        self.btn_next.setMinimumHeight(40)
        self.btn_next.setMinimumWidth(220)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._on_next)
        btn_bar.addWidget(self.btn_next)

        root.addLayout(btn_bar)

        # ── Status bar ──────────────────────────────────────────────────────
        self.statusBar().showMessage("Ready")

    # ── Target + problem type panel ─────────────────────────────────────────
    def _build_target_panel(self) -> QGroupBox:
        group = QGroupBox("Target & Problem Type")
        lay = QVBoxLayout(group)

        # Target column selector
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Target column:"))
        self.combo_target = QComboBox()
        self.combo_target.setMinimumWidth(160)
        self.combo_target.setEnabled(False)
        row1.addWidget(self.combo_target, stretch=1)
        lay.addLayout(row1)

        # Problem type radio buttons
        lay.addSpacing(6)
        lay.addWidget(QLabel("Problem type:"))
        radio_row = QHBoxLayout()
        self.radio_classification = QRadioButton("Classification")
        self.radio_regression = QRadioButton("Regression")
        self.radio_classification.setChecked(True)
        self.btn_group_problem = QButtonGroup(self)
        self.btn_group_problem.addButton(self.radio_classification)
        self.btn_group_problem.addButton(self.radio_regression)
        radio_row.addWidget(self.radio_classification)
        radio_row.addWidget(self.radio_regression)
        radio_row.addStretch()
        lay.addLayout(radio_row)
        lay.addStretch()
        return group

    # ── Split config + NaN panel ────────────────────────────────────────────
    def _build_split_panel(self) -> QGroupBox:
        group = QGroupBox("Split & Cleaning")
        lay = QVBoxLayout(group)

        # Split method radio buttons
        lay.addWidget(QLabel("Split method:"))
        split_row = QHBoxLayout()
        self.radio_percentage = QRadioButton("Percentage")
        self.radio_kfold = QRadioButton("K-Fold CV")
        self.radio_percentage.setChecked(True)
        self.btn_group_split = QButtonGroup(self)
        self.btn_group_split.addButton(self.radio_percentage)
        self.btn_group_split.addButton(self.radio_kfold)
        self.radio_percentage.toggled.connect(self._on_split_method_changed)
        split_row.addWidget(self.radio_percentage)
        split_row.addWidget(self.radio_kfold)
        split_row.addStretch()
        lay.addLayout(split_row)

        # Percentage ratio spinner
        pct_row = QHBoxLayout()
        self.lbl_ratio = QLabel("Train ratio:")
        pct_row.addWidget(self.lbl_ratio)
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.5, 0.95)
        self.spin_ratio.setSingleStep(0.05)
        self.spin_ratio.setValue(0.8)
        self.spin_ratio.setFixedWidth(80)
        pct_row.addWidget(self.spin_ratio)
        pct_row.addStretch()
        lay.addLayout(pct_row)

        # K-fold spinner
        kfold_row = QHBoxLayout()
        self.lbl_k = QLabel("K folds:")
        kfold_row.addWidget(self.lbl_k)
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 20)
        self.spin_k.setValue(5)
        self.spin_k.setFixedWidth(80)
        kfold_row.addWidget(self.spin_k)
        kfold_row.addStretch()
        lay.addLayout(kfold_row)

        # NaN handling
        lay.addSpacing(6)
        nan_row = QHBoxLayout()
        nan_row.addWidget(QLabel("NaN handling:"))
        self.combo_nan = QComboBox()
        self.combo_nan.addItems(["Fill with mean / mode", "Drop rows with NaN"])
        self.combo_nan.setMinimumWidth(180)
        nan_row.addWidget(self.combo_nan, stretch=1)
        lay.addLayout(nan_row)

        # Clean button
        self.btn_clean = QPushButton("🧹  Clean Data")
        self.btn_clean.setEnabled(False)
        self.btn_clean.clicked.connect(self._clean_data)
        lay.addWidget(self.btn_clean)

        lay.addStretch()

        # Set initial visibility
        self._on_split_method_changed()

        return group

    # ── Split method toggle ─────────────────────────────────────────────────
    def _on_split_method_changed(self) -> None:
        is_pct = self.radio_percentage.isChecked()
        self.lbl_ratio.setVisible(is_pct)
        self.spin_ratio.setVisible(is_pct)
        self.lbl_k.setVisible(not is_pct)
        self.spin_k.setVisible(not is_pct)

    # ── CSV loading ─────────────────────────────────────────────────────────
    def _load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;TSV Files (*.tsv);;All Files (*)",
        )
        if not path:
            return

        self._start_worker("load", filepath=path)

    def _on_load_finished(self, df: pd.DataFrame, _) -> None:
        self._raw_df = df
        self._cleaned_df = None

        # Update file label
        fname = "Loaded Data" # placeholder if path not stored, let's refine
        self.lbl_file.setText(f"✅  Data Loaded")
        self.lbl_file.setStyleSheet("color: #3fb950;")

        # Show info
        self.lbl_info.setText(
            f"{df.shape[0]} rows × {df.shape[1]} columns  •  "
            f"NaN cells: {int(df.isna().sum().sum())}"
        )

        # Populate preview
        self.preview.set_dataframe(df)

        # Populate target column dropdown
        from backend.data_handler import detect_columns
        columns = detect_columns(df)
        self.combo_target.clear()
        self.combo_target.addItems(columns)
        self.combo_target.setCurrentIndex(len(columns) - 1)
        self.combo_target.setEnabled(True)

        self.btn_clean.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.statusBar().showMessage("Data loaded successfully.", 3000)

    # ── Data cleaning ───────────────────────────────────────────────────────
    def _clean_data(self) -> None:
        if self._raw_df is None:
            QMessageBox.warning(self, "No Data", "Load a CSV file first.")
            return

        from backend.data_handler import NaNStrategy
        strategy = (
            NaNStrategy.FILL_MEAN
            if self.combo_nan.currentIndex() == 0
            else NaNStrategy.DROP_ROWS
        )
        self._start_worker("clean", df=self._raw_df, strategy=strategy)

    def _on_clean_finished(self, cleaned: pd.DataFrame, report: dict) -> None:
        self._cleaned_df = cleaned

        # Refresh preview & info
        self.preview.set_dataframe(cleaned)
        self.lbl_info.setText(
            f"{cleaned.shape[0]} rows × {cleaned.shape[1]} columns  •  "
            f"NaN cells: {report['nan_count_after']}  •  "
            f"Strategy: {report['strategy_used']}  •  "
            f"Rows removed: {report['rows_before'] - report['rows_after']}"
        )

        QMessageBox.information(
            self,
            "Data Cleaned ✅",
            f"NaN before: {report['nan_count_before']}\n"
            f"NaN after:  {report['nan_count_after']}\n"
            f"Rows: {report['rows_before']} → {report['rows_after']}\n"
            f"Strategy: {report['strategy_used']}",
        )
        self.statusBar().showMessage("Data cleaned.", 3000)

    # ── Worker Management ───────────────────────────────────────────────────
    def _start_worker(self, task: str, **kwargs):
        self.setCursor(Qt.CursorShape.WaitCursor)
        self.btn_load.setEnabled(False)
        self.btn_clean.setEnabled(False)
        self.btn_next.setEnabled(False)

        self._worker_thread = QThread()
        self._worker = DataLoaderWorker(task, **kwargs)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self._worker.error.connect(self._on_worker_error)

        if task == "load":
            self._worker.finished.connect(self._on_load_finished)
        else:
            self._worker.finished.connect(self._on_clean_finished)

        self._worker.finished.connect(self._cleanup_worker)
        self._worker.error.connect(self._cleanup_worker)

        self._worker_thread.start()

    def _on_worker_error(self, msg: str):
        QMessageBox.critical(self, "Error", f"Operation failed:\n{msg}")
        self.statusBar().showMessage("Error occurred.")

    def _cleanup_worker(self):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.btn_load.setEnabled(True)
        if self._raw_df is not None:
            self.btn_clean.setEnabled(True)
            self.btn_next.setEnabled(True)
        
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()

    # ── Next → ──────────────────────────────────────────────────────────────
    def _on_next(self) -> None:
        """Validate, sync to ProjectState, and open Window 2."""
        try:
            # Determine which DataFrame to use
            df = self._cleaned_df if self._cleaned_df is not None else self._raw_df
            if df is None:
                QMessageBox.warning(self, "No Data", "Please load a CSV file first.")
                return

            # Check for remaining NaNs
            nan_count = int(df.isna().sum().sum())
            if nan_count > 0:
                reply = QMessageBox.question(
                    self,
                    "Data Contains NaN Values",
                    f"The dataset still has {nan_count} missing value(s).\n\n"
                    "Would you like to auto-fill them with column means/modes?\n\n"
                    "Click 'Yes' to auto-clean, or 'No' to go back and configure cleaning.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    from backend.data_handler import NaNStrategy, clean_dataframe
                    df, _ = clean_dataframe(df, NaNStrategy.FILL_MEAN)
                    self._cleaned_df = df
                    self.preview.set_dataframe(df)
                else:
                    return

            # Target column
            target = self.combo_target.currentText()
            if not target or target not in df.columns:
                QMessageBox.warning(
                    self,
                    "Invalid Target",
                    "Please select a valid target column.",
                )
                return

            # Feature count check
            from backend.data_handler import count_input_features
            n_features = count_input_features(df, target)
            if n_features == 0:
                QMessageBox.warning(
                    self,
                    "No Features",
                    "The dataset has no input features (only the target column).",
                )
                return

            # Problem type
            problem_type = (
                "classification"
                if self.radio_classification.isChecked()
                else "regression"
            )

            # Split config
            if self.radio_percentage.isChecked():
                split_config = {
                    "method": "percentage",
                    "ratio": self.spin_ratio.value(),
                }
            else:
                split_config = {
                    "method": "kfold",
                    "k": self.spin_k.value(),
                }

            # ── Write to shared state ───────────────────────────────────────
            self.state.dataframe = df
            self.state.target_column = target
            self.state.problem_type = problem_type
            self.state.split_config = split_config

            # Open Window 2
            if self._on_next_callback:
                self._on_next_callback()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error",
                f"An unexpected error occurred.\n\n{type(exc).__name__}: {exc}",
            )

    # ── Public API ──────────────────────────────────────────────────────────
    def get_current_dataframe(self) -> Optional["pd.DataFrame"]:
        """Return the cleaned DataFrame if available, else the raw one."""
        return self._cleaned_df if self._cleaned_df is not None else self._raw_df

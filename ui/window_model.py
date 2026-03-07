"""
window_model.py
===============
Window 2 — Sequential Model Builder.

This window lets users visually assemble a neural network layer by layer
with Add / Remove controls, and save or load blueprints as JSON files.
"""

from __future__ import annotations

import traceback
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.layer_row import LayerRow
from backend.model_builder import build_and_validate
from utils.blueprint_io import load_blueprint, save_blueprint
from utils.project_state import ProjectState
from utils.validators import validate_blueprint


class ModelBuilderWindow(QMainWindow):
    """
    Window 2 of the pipeline — the layer-by-layer model builder.

    Parameters
    ----------
    project_state : ProjectState
        Shared state object that holds the blueprint.
    parent : QWidget | None
        Optional parent widget.
    """

    def __init__(
        self,
        project_state: ProjectState,
        on_back=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = project_state
        self._on_back_callback = on_back
        self._layer_rows: list[LayerRow] = []
        self._init_window()
        self._build_ui()

        # Start with one default output layer
        self._add_layer_row()

    # ── Window setup ────────────────────────────────────────────────────────
    def _init_window(self) -> None:
        self.setWindowTitle("Neural Network Builder — Model Architecture")
        self.setMinimumSize(820, 560)
        self.resize(900, 640)

    # ── UI construction ─────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # ── Header ──────────────────────────────────────────────────────────
        header = QLabel("🧠  Model Architecture")
        header.setProperty("class", "heading")
        header.setStyleSheet("font-size: 20px; font-weight: 700;")
        root.addWidget(header)

        subtitle = QLabel(
            "Add layers sequentially. The last layer must be Linear (output)."
        )
        subtitle.setProperty("class", "subheading")
        subtitle.setStyleSheet("color: #8b949e; margin-bottom: 4px;")
        root.addWidget(subtitle)

        # ── Data info (shows loaded dataset context) ────────────────────────
        if self.state.dataframe is not None:
            n_feat = self.state.input_features()
            prob = self.state.problem_type.capitalize()
            self.lbl_data_info = QLabel(
                f"📋  Dataset: {n_feat} input features  •  "
                f"Target: {self.state.target_column}  •  "
                f"Problem: {prob}"
            )
            self.lbl_data_info.setStyleSheet(
                "color: #58a6ff; font-size: 12px; padding: 4px 0;"
            )
            root.addWidget(self.lbl_data_info)

        # ── Scrollable layer list ───────────────────────────────────────────
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.layer_container = QWidget()
        self.layer_layout = QVBoxLayout(self.layer_container)
        self.layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_layout.setSpacing(8)
        self.layer_layout.addStretch()  # keeps rows top-aligned

        self.scroll_area.setWidget(self.layer_container)
        root.addWidget(self.scroll_area, stretch=1)

        # ── Layer count label ───────────────────────────────────────────────
        self.lbl_count = QLabel("Layers: 0")
        self.lbl_count.setStyleSheet("color: #8b949e;")
        root.addWidget(self.lbl_count)

        # ── Button bar ──────────────────────────────────────────────────────
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(10)

        # Back button (only if a callback is provided)
        if self._on_back_callback:
            self.btn_back = QPushButton("←  Back to Data")
            self.btn_back.setMinimumHeight(36)
            self.btn_back.clicked.connect(self._on_back_callback)
            btn_bar.addWidget(self.btn_back)

        self.btn_add = QPushButton("＋  Add Layer")
        self.btn_add.setProperty("class", "primary")
        self.btn_add.setMinimumHeight(36)
        self.btn_add.clicked.connect(self._add_layer_row)
        btn_bar.addWidget(self.btn_add)

        btn_bar.addStretch()

        self.btn_save = QPushButton("💾  Save Blueprint")
        self.btn_save.setMinimumHeight(36)
        self.btn_save.clicked.connect(self._save_blueprint)
        btn_bar.addWidget(self.btn_save)

        self.btn_load = QPushButton("📂  Load Blueprint")
        self.btn_load.setMinimumHeight(36)
        self.btn_load.clicked.connect(self._load_blueprint)
        btn_bar.addWidget(self.btn_load)

        btn_bar.addStretch()

        self.btn_validate = QPushButton("✅  Validate")
        self.btn_validate.setMinimumHeight(36)
        self.btn_validate.clicked.connect(self._validate_and_show)
        btn_bar.addWidget(self.btn_validate)

        self.btn_build = QPushButton("🔨  Build & Test")
        self.btn_build.setProperty("class", "primary")
        self.btn_build.setMinimumHeight(36)
        self.btn_build.clicked.connect(self._build_and_test)
        btn_bar.addWidget(self.btn_build)

        root.addLayout(btn_bar)

    # ── Layer management ────────────────────────────────────────────────────
    def _add_layer_row(self, config: dict | None = None) -> None:
        """Append a new layer row to the list, optionally pre-populated."""
        row = LayerRow(index=len(self._layer_rows))
        if config:
            row.set_config(config)
        row.remove_requested.connect(self._remove_layer_row)
        row.config_changed.connect(self._update_count_label)

        # Insert *before* the trailing stretch
        insert_pos = self.layer_layout.count() - 1
        self.layer_layout.insertWidget(insert_pos, row)
        self._layer_rows.append(row)
        self._update_count_label()

    def _remove_layer_row(self, index: int) -> None:
        """Remove the row at *index* (or the sender row)."""
        # Identify the sender row safely
        sender_row = self.sender()
        if sender_row and sender_row in self._layer_rows:
            row = sender_row
        elif 0 <= index < len(self._layer_rows):
            row = self._layer_rows[index]
        else:
            return

        if len(self._layer_rows) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Remove",
                "You must keep at least one layer (the output layer).",
            )
            return

        self._layer_rows.remove(row)
        self.layer_layout.removeWidget(row)
        row.deleteLater()
        self._reindex_rows()
        self._update_count_label()

    def _reindex_rows(self) -> None:
        for i, row in enumerate(self._layer_rows):
            row.set_index(i)

    def _update_count_label(self) -> None:
        self.lbl_count.setText(f"Layers: {len(self._layer_rows)}")

    # ── Blueprint extraction ────────────────────────────────────────────────
    def get_architecture(self) -> list[dict]:
        """
        Extract the current UI state as a list of layer config dicts.

        Returns
        -------
        list[dict]
            The blueprint, e.g.
            ``[{"type": "Linear", "neurons": 128, "activation": "ReLU"}, ...]``
        """
        return [row.get_config() for row in self._layer_rows]

    # ── Save / Load ─────────────────────────────────────────────────────────
    def _save_blueprint(self) -> None:
        try:
            blueprint = self.get_architecture()
            valid, msg = validate_blueprint(blueprint)
            if not valid:
                QMessageBox.warning(self, "Invalid Blueprint", msg)
                return

            path, _ = QFileDialog.getSaveFileName(
                self, "Save Blueprint", "", "JSON Files (*.json);;All Files (*)"
            )
            if not path:
                return

            save_blueprint(blueprint, path)
            QMessageBox.information(
                self,
                "Saved",
                f"Blueprint saved to:\n{path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Could not save blueprint.\n\n{exc}",
            )

    def _load_blueprint(self) -> None:
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Blueprint", "", "JSON Files (*.json);;All Files (*)"
            )
            if not path:
                return

            layers = load_blueprint(path)
            valid, msg = validate_blueprint(layers)
            if not valid:
                QMessageBox.warning(
                    self,
                    "Invalid Blueprint File",
                    f"The loaded file is not a valid blueprint:\n\n{msg}",
                )
                return

            # Clear existing rows
            self._clear_all_rows()

            # Rebuild from loaded config
            for layer_cfg in layers:
                self._add_layer_row(config=layer_cfg)

            QMessageBox.information(
                self,
                "Loaded",
                f"Blueprint loaded from:\n{path}\n\n{len(layers)} layer(s) restored.",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not load blueprint.\n\n{exc}",
            )

    def _clear_all_rows(self) -> None:
        """Remove every layer row from the UI."""
        for row in list(self._layer_rows):
            self.layer_layout.removeWidget(row)
            row.deleteLater()
        self._layer_rows.clear()
        self._update_count_label()

    # ── Validation ──────────────────────────────────────────────────────────
    def _validate_and_show(self) -> None:
        """Validate the current blueprint and show result in a message box."""
        try:
            blueprint = self.get_architecture()
            valid, msg = validate_blueprint(blueprint)
            if valid:
                QMessageBox.information(
                    self,
                    "Blueprint Valid ✅",
                    f"Your architecture has {len(blueprint)} layer(s) and "
                    f"passes all validation checks.",
                )
            else:
                QMessageBox.warning(self, "Validation Failed", msg)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Validation Error",
                f"An unexpected error occurred during validation.\n\n{exc}",
            )

    # ── Build & Ghost Run ───────────────────────────────────────────────────
    def _build_and_test(self) -> None:
        """Build nn.Sequential from blueprint + run ghost forward pass."""
        try:
            blueprint = self.get_architecture()
            valid, msg = validate_blueprint(blueprint)
            if not valid:
                QMessageBox.warning(self, "Invalid Blueprint", msg)
                return

            # Determine input features
            n_features = self.state.input_features()
            if n_features == 0:
                QMessageBox.warning(
                    self,
                    "No Data Loaded",
                    "Load a dataset in Window 1 first so the ghost run \n"
                    "can determine the correct input dimensions."
                )
                return

            model, dummy, success, msg = build_and_validate(
                blueprint, n_features,
            )
            if success:
                self.state.model = model
                self.state.blueprint = blueprint
                self.state.dummy_tensor = dummy
                QMessageBox.information(
                    self,
                    "Build Successful ✅",
                    f"{msg}\n\nModel summary:\n{model}",
                )
            else:
                QMessageBox.warning(self, "Build Failed", msg)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Build Error",
                f"An unexpected error occurred.\n\n{type(exc).__name__}: {exc}",
            )

    # ── Sync with ProjectState ──────────────────────────────────────────────
    def sync_to_state(self) -> bool:
        """
        Validate, build the model, ghost-run, and write to state.

        Returns True on success, False on failure (shows dialog).
        """
        blueprint = self.get_architecture()
        valid, msg = validate_blueprint(blueprint)
        if not valid:
            QMessageBox.warning(self, "Cannot Proceed", msg)
            return False

        n_features = self.state.input_features()
        if n_features == 0:
            QMessageBox.warning(
                self,
                "No Data Loaded",
                "Load a dataset in Window 1 first.",
            )
            return False

        model, dummy, success, msg = build_and_validate(blueprint, n_features)
        if not success:
            QMessageBox.warning(self, "Build Failed", msg)
            return False

        self.state.blueprint = blueprint
        self.state.model = model
        self.state.dummy_tensor = dummy
        return True

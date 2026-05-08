from __future__ import annotations
import traceback
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget, QFrame
)

from ui.layer_row import LayerRow
from backend.model_builder import build_and_validate
from utils.blueprint_io import load_blueprint, save_blueprint
from utils.project_state import ProjectState
from utils.validators import validate_blueprint

class ModelBuilderWindow(QWidget):
    """
    Model Builder Dashboard.
    Lets users visually assemble a neural network layer by layer.
    """
    def __init__(self, project_state: ProjectState, on_back=None, on_next=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = project_state
        self._on_back_callback = on_back
        self._on_next_callback = on_next
        self._layer_rows: list[LayerRow] = []
        self._build_ui()
        self._add_layer_row() # Start with one default output layer

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(24)

        # ── Header ──
        header_layout = QVBoxLayout()
        title = QLabel("Model Architecture Builder")
        title.setProperty("class", "PageTitle")
        subtitle = QLabel("Design your neural network layer by layer. The final layer must be Linear.")
        subtitle.setProperty("class", "PageSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root.addLayout(header_layout)

        # ── Data Info ──
        self.lbl_data_info = QLabel("")
        self.lbl_data_info.setStyleSheet("color: #0EA5E9; font-weight: 600; padding: 4px 0;")
        self.lbl_data_info.setVisible(False)
        root.addWidget(self.lbl_data_info)
        self.refresh_data_info()

        # ── Scrollable layer list ──
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.layer_container = QWidget()
        self.layer_layout = QVBoxLayout(self.layer_container)
        self.layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_layout.setSpacing(12)
        self.layer_layout.addStretch()

        self.scroll_area.setWidget(self.layer_container)
        root.addWidget(self.scroll_area, stretch=1)

        # ── Layer count ──
        self.lbl_count = QLabel("Layers: 0")
        self.lbl_count.setStyleSheet("color: #64748B; font-weight: 600;")
        root.addWidget(self.lbl_count)

        # ── Button bar ──
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(12)

        self.btn_add = QPushButton("＋ Add Layer")
        self.btn_add.setProperty("class", "primary")
        self.btn_add.setMinimumHeight(44)
        self.btn_add.clicked.connect(self._add_layer_row)
        btn_bar.addWidget(self.btn_add)

        btn_bar.addStretch()

        self.btn_save = QPushButton("💾 Save Blueprint")
        self.btn_save.setMinimumHeight(44)
        self.btn_save.clicked.connect(self._save_blueprint)
        btn_bar.addWidget(self.btn_save)

        self.btn_load = QPushButton("📂 Load Blueprint")
        self.btn_load.setMinimumHeight(44)
        self.btn_load.clicked.connect(self._load_blueprint)
        btn_bar.addWidget(self.btn_load)

        btn_bar.addStretch()

        self.btn_validate = QPushButton("✅ Validate")
        self.btn_validate.setMinimumHeight(44)
        self.btn_validate.clicked.connect(self._validate_and_show)
        btn_bar.addWidget(self.btn_validate)

        self.btn_build = QPushButton("🔨 Build & Test")
        self.btn_build.setProperty("class", "primary")
        self.btn_build.setMinimumHeight(44)
        self.btn_build.clicked.connect(self._build_and_test)
        btn_bar.addWidget(self.btn_build)

        self.btn_next = QPushButton("Proceed to Training →")
        self.btn_next.setMinimumHeight(44)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._on_next)
        btn_bar.addWidget(self.btn_next)

        root.addLayout(btn_bar)

    def refresh_data_info(self) -> None:
        if self.state.dataframe is not None:
            n_feat = self.state.input_features()
            prob = self.state.problem_type.capitalize()
            self.lbl_data_info.setText(
                f"📋 Dataset Context: {n_feat} input features • Target: {self.state.target_column} • Problem: {prob}"
            )
            self.lbl_data_info.setVisible(True)
        else:
            self.lbl_data_info.setVisible(False)

    def _add_layer_row(self, config: dict | None = None) -> None:
        row = LayerRow(index=len(self._layer_rows))
        if config: row.set_config(config)
        row.remove_requested.connect(self._remove_layer_row)
        row.config_changed.connect(self._update_count_label)
        insert_pos = self.layer_layout.count() - 1
        self.layer_layout.insertWidget(insert_pos, row)
        self._layer_rows.append(row)
        self._update_count_label()

    def _remove_layer_row(self, index: int) -> None:
        sender_row = self.sender()
        if sender_row and sender_row in self._layer_rows:
            row = sender_row
        elif 0 <= index < len(self._layer_rows):
            row = self._layer_rows[index]
        else: return

        if len(self._layer_rows) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "You must keep at least one layer (the output layer).")
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

    def get_architecture(self) -> list[dict]:
        return [row.get_config() for row in self._layer_rows]

    def _save_blueprint(self) -> None:
        try:
            blueprint = self.get_architecture()
            valid, msg = validate_blueprint(blueprint)
            if not valid:
                QMessageBox.warning(self, "Invalid Blueprint", msg)
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save Blueprint", "", "JSON Files (*.json);;All Files (*)")
            if not path: return
            save_blueprint(blueprint, path)
            QMessageBox.information(self, "Saved", f"Blueprint saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Could not save blueprint.\n\n{exc}")

    def _load_blueprint(self) -> None:
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Load Blueprint", "", "JSON Files (*.json);;All Files (*)")
            if not path: return
            layers = load_blueprint(path)
            valid, msg = validate_blueprint(layers)
            if not valid:
                QMessageBox.warning(self, "Invalid Blueprint File", f"The loaded file is not a valid blueprint:\n\n{msg}")
                return
            self._clear_all_rows()
            for layer_cfg in layers:
                self._add_layer_row(config=layer_cfg)
            QMessageBox.information(self, "Loaded", f"Blueprint loaded from:\n{path}\n\n{len(layers)} layer(s) restored.")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load blueprint.\n\n{exc}")

    def _clear_all_rows(self) -> None:
        for row in list(self._layer_rows):
            self.layer_layout.removeWidget(row)
            row.deleteLater()
        self._layer_rows.clear()
        self._update_count_label()

    def _validate_and_show(self) -> None:
        try:
            blueprint = self.get_architecture()
            valid, msg = validate_blueprint(blueprint)
            if valid:
                QMessageBox.information(self, "Blueprint Valid ✅", f"Your architecture has {len(blueprint)} layer(s) and passes all validation checks.")
            else:
                QMessageBox.warning(self, "Validation Failed", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Validation Error", f"An unexpected error occurred during validation.\n\n{exc}")

    def _build_and_test(self) -> None:
        try:
            blueprint = self.get_architecture()
            valid, msg = validate_blueprint(blueprint)
            if not valid:
                QMessageBox.warning(self, "Invalid Blueprint", msg)
                return
            n_features = self.state.input_features()
            if n_features == 0:
                QMessageBox.warning(self, "No Data Loaded", "Load a dataset in the Data Lab first so the ghost run can determine the correct input dimensions.")
                return

            model, dummy_input, success, msg = build_and_validate(blueprint, n_features)
            if success:
                self.state.model = model
                self.state.blueprint = blueprint
                self.state.dummy_tensor = dummy_input
                self.btn_next.setEnabled(True)
                QMessageBox.information(self, "Build Successful ✅", f"{msg}\n\nModel summary:\n{model}")
            else:
                QMessageBox.warning(self, "Build Failed", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Build Error", f"An unexpected error occurred.\n\n{type(exc).__name__}: {exc}")

    def sync_to_state(self) -> bool:
        blueprint = self.get_architecture()
        valid, msg = validate_blueprint(blueprint)
        if not valid:
            QMessageBox.warning(self, "Cannot Proceed", msg)
            return False
        n_features = self.state.input_features()
        if n_features == 0:
            QMessageBox.warning(self, "No Data Loaded", "Load a dataset in the Data Lab first.")
            return False
        model, dummy_input, success, msg = build_and_validate(blueprint, n_features)
        if not success:
            QMessageBox.warning(self, "Build Failed", msg)
            return False
        self.state.blueprint = blueprint
        self.state.model = model
        self.state.dummy_tensor = dummy_input
        self.btn_next.setEnabled(True)
        return True

    def _on_next(self) -> None:
        if not self.sync_to_state(): return
        if self._on_next_callback:
            self._on_next_callback()

"""
layer_row.py
============
Custom widget representing a single layer in the sequential model builder.
Dynamically shows/hides controls based on the selected layer type.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

# ─── Constants ───────────────────────────────────────────────────────────────
LAYER_TYPES = ["Linear", "Dropout", "BatchNorm1d"]
ACTIVATIONS = ["None", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Softmax"]


class LayerRow(QFrame):
    """
    A card-style row for configuring one layer.

    Signals
    -------
    remove_requested : int
        Emitted with the row's index when the user clicks "Remove".
    config_changed
        Emitted whenever the user changes any setting in this row.
    """

    remove_requested = pyqtSignal(int)
    config_changed = pyqtSignal()

    def __init__(self, index: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.row_index = index
        self._build_ui()
        self._connect_signals()
        self._on_type_changed()

    # ── UI construction ─────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.Box)
        self.setObjectName("layerRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Row number label
        self.lbl_index = QLabel(f"#{self.row_index + 1}")
        self.lbl_index.setFixedWidth(32)
        self.lbl_index.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_index.setStyleSheet("font-weight: 700; color: #58a6ff;")
        layout.addWidget(self.lbl_index)

        # Layer type
        layout.addWidget(QLabel("Type:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(LAYER_TYPES)
        self.combo_type.setFixedWidth(130)
        layout.addWidget(self.combo_type)

        # Neurons (only for Linear)
        self.lbl_neurons = QLabel("Neurons:")
        layout.addWidget(self.lbl_neurons)
        self.spin_neurons = QSpinBox()
        self.spin_neurons.setRange(1, 10_000)
        self.spin_neurons.setValue(64)
        self.spin_neurons.setFixedWidth(90)
        layout.addWidget(self.spin_neurons)

        # Activation (only for Linear)
        self.lbl_activation = QLabel("Activation:")
        layout.addWidget(self.lbl_activation)
        self.combo_activation = QComboBox()
        self.combo_activation.addItems(ACTIVATIONS)
        self.combo_activation.setFixedWidth(120)
        layout.addWidget(self.combo_activation)

        # Dropout rate (only for Dropout)
        self.lbl_dropout = QLabel("Rate:")
        layout.addWidget(self.lbl_dropout)
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 1.0)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setValue(0.3)
        self.spin_dropout.setFixedWidth(80)
        layout.addWidget(self.spin_dropout)

        # Spacer to push remove button to the right
        layout.addStretch()

        # Remove button
        self.btn_remove = QPushButton("✕")
        self.btn_remove.setProperty("class", "danger")
        self.btn_remove.setFixedSize(32, 32)
        self.btn_remove.setToolTip("Remove this layer")
        layout.addWidget(self.btn_remove)

    # ── Signal wiring ───────────────────────────────────────────────────────
    def _connect_signals(self) -> None:
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        self.combo_type.currentTextChanged.connect(lambda _: self.config_changed.emit())
        self.spin_neurons.valueChanged.connect(lambda _: self.config_changed.emit())
        self.combo_activation.currentTextChanged.connect(lambda _: self.config_changed.emit())
        self.spin_dropout.valueChanged.connect(lambda _: self.config_changed.emit())
        self.btn_remove.clicked.connect(self._on_remove_clicked)

    # ── Visibility helpers ──────────────────────────────────────────────────
    def _on_type_changed(self) -> None:
        """Show/hide controls based on the selected layer type."""
        layer_type = self.combo_type.currentText()

        is_linear = layer_type == "Linear"
        is_dropout = layer_type == "Dropout"

        # Linear-specific
        self.lbl_neurons.setVisible(is_linear)
        self.spin_neurons.setVisible(is_linear)
        self.lbl_activation.setVisible(is_linear)
        self.combo_activation.setVisible(is_linear)

        # Dropout-specific
        self.lbl_dropout.setVisible(is_dropout)
        self.spin_dropout.setVisible(is_dropout)

    def _on_remove_clicked(self) -> None:
        self.remove_requested.emit(self.row_index)

    # ── Public API ──────────────────────────────────────────────────────────
    def set_index(self, index: int) -> None:
        """Update the displayed row number after reordering."""
        self.row_index = index
        self.lbl_index.setText(f"#{index + 1}")

    def get_config(self) -> dict:
        """
        Return a blueprint dictionary for this layer.

        Examples
        -------
        >>> {"type": "Linear", "neurons": 128, "activation": "ReLU"}
        >>> {"type": "Dropout", "rate": 0.3}
        >>> {"type": "BatchNorm1d"}
        """
        layer_type = self.combo_type.currentText()

        if layer_type == "Linear":
            return {
                "type": "Linear",
                "neurons": self.spin_neurons.value(),
                "activation": self.combo_activation.currentText(),
            }
        elif layer_type == "Dropout":
            return {
                "type": "Dropout",
                "rate": round(self.spin_dropout.value(), 4),
            }
        elif layer_type == "BatchNorm1d":
            return {"type": "BatchNorm1d"}
        else:
            return {"type": layer_type}

    def set_config(self, config: dict) -> None:
        """
        Populate the row's widgets from a blueprint dictionary.

        Parameters
        ----------
        config : dict
            A layer blueprint, e.g. ``{"type": "Linear", "neurons": 64, "activation": "ReLU"}``.
        """
        try:
            layer_type = config.get("type", "Linear")
            idx = self.combo_type.findText(layer_type)
            if idx >= 0:
                self.combo_type.setCurrentIndex(idx)

            if layer_type == "Linear":
                self.spin_neurons.setValue(config.get("neurons", 64))
                act = config.get("activation", "None")
                act_idx = self.combo_activation.findText(act)
                if act_idx >= 0:
                    self.combo_activation.setCurrentIndex(act_idx)
            elif layer_type == "Dropout":
                self.spin_dropout.setValue(config.get("rate", 0.3))
        except Exception:
            pass  # Gracefully ignore malformed config

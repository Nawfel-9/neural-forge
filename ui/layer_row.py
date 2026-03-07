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
LAYER_TYPES = [
    "Linear",
    "Conv1d",
    "MaxPool1d",
    "AvgPool1d",
    "Flatten",
    "BatchNorm1d",
    "Dropout",
]
ACTIVATIONS = ["None", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Softmax"]
PADDING_MODES = ["zeros", "reflect", "replicate", "circular"]


class LayerRow(QFrame):
    """
    A card-style row for configuring one layer.

    Supported layer types and their controls:

    - **Linear**: neurons, activation
    - **Conv1d**: out_channels, kernel_size, stride, padding
    - **MaxPool1d**: kernel_size, stride
    - **AvgPool1d**: kernel_size, stride
    - **Flatten**: (no parameters)
    - **BatchNorm1d**: (no parameters)
    - **Dropout**: rate

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
        layout.setSpacing(10)

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

        # ── Linear controls ─────────────────────────────────────────────────
        self.lbl_neurons = QLabel("Neurons:")
        layout.addWidget(self.lbl_neurons)
        self.spin_neurons = QSpinBox()
        self.spin_neurons.setRange(1, 10_000)
        self.spin_neurons.setValue(64)
        self.spin_neurons.setFixedWidth(90)
        layout.addWidget(self.spin_neurons)

        self.lbl_activation = QLabel("Activation:")
        layout.addWidget(self.lbl_activation)
        self.combo_activation = QComboBox()
        self.combo_activation.addItems(ACTIVATIONS)
        self.combo_activation.setFixedWidth(120)
        layout.addWidget(self.combo_activation)

        # ── Conv1d controls ─────────────────────────────────────────────────
        self.lbl_out_channels = QLabel("Channels:")
        layout.addWidget(self.lbl_out_channels)
        self.spin_out_channels = QSpinBox()
        self.spin_out_channels.setRange(1, 2048)
        self.spin_out_channels.setValue(32)
        self.spin_out_channels.setFixedWidth(80)
        layout.addWidget(self.spin_out_channels)

        # ── Shared: kernel_size (Conv1d, MaxPool1d, AvgPool1d) ──────────────
        self.lbl_kernel = QLabel("Kernel:")
        layout.addWidget(self.lbl_kernel)
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 99)
        self.spin_kernel.setValue(3)
        self.spin_kernel.setFixedWidth(60)
        layout.addWidget(self.spin_kernel)

        # ── Shared: stride (Conv1d, MaxPool1d, AvgPool1d) ───────────────────
        self.lbl_stride = QLabel("Stride:")
        layout.addWidget(self.lbl_stride)
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(1, 99)
        self.spin_stride.setValue(1)
        self.spin_stride.setFixedWidth(60)
        layout.addWidget(self.spin_stride)

        # ── Conv1d padding ──────────────────────────────────────────────────
        self.lbl_padding = QLabel("Padding:")
        layout.addWidget(self.lbl_padding)
        self.spin_padding = QSpinBox()
        self.spin_padding.setRange(0, 49)
        self.spin_padding.setValue(0)
        self.spin_padding.setFixedWidth(60)
        layout.addWidget(self.spin_padding)

        # ── Dropout rate ────────────────────────────────────────────────────
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
        self.spin_out_channels.valueChanged.connect(lambda _: self.config_changed.emit())
        self.spin_kernel.valueChanged.connect(lambda _: self.config_changed.emit())
        self.spin_stride.valueChanged.connect(lambda _: self.config_changed.emit())
        self.spin_padding.valueChanged.connect(lambda _: self.config_changed.emit())
        self.spin_dropout.valueChanged.connect(lambda _: self.config_changed.emit())
        self.btn_remove.clicked.connect(self._on_remove_clicked)

    # ── Visibility helpers ──────────────────────────────────────────────────
    def _on_type_changed(self) -> None:
        """Show/hide controls based on the selected layer type."""
        t = self.combo_type.currentText()

        is_linear   = t == "Linear"
        is_conv1d   = t == "Conv1d"
        is_pool     = t in ("MaxPool1d", "AvgPool1d")
        is_dropout  = t == "Dropout"

        # Linear
        self.lbl_neurons.setVisible(is_linear)
        self.spin_neurons.setVisible(is_linear)
        self.lbl_activation.setVisible(is_linear)
        self.combo_activation.setVisible(is_linear)

        # Conv1d
        self.lbl_out_channels.setVisible(is_conv1d)
        self.spin_out_channels.setVisible(is_conv1d)

        # Kernel + Stride: Conv1d OR Pool
        self.lbl_kernel.setVisible(is_conv1d or is_pool)
        self.spin_kernel.setVisible(is_conv1d or is_pool)
        self.lbl_stride.setVisible(is_conv1d or is_pool)
        self.spin_stride.setVisible(is_conv1d or is_pool)

        # Padding: Conv1d only
        self.lbl_padding.setVisible(is_conv1d)
        self.spin_padding.setVisible(is_conv1d)

        # Dropout
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
        >>> {"type": "Conv1d", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0}
        >>> {"type": "MaxPool1d", "kernel_size": 2, "stride": 2}
        >>> {"type": "AvgPool1d", "kernel_size": 2, "stride": 2}
        >>> {"type": "Flatten"}
        >>> {"type": "Dropout", "rate": 0.3}
        >>> {"type": "BatchNorm1d"}
        """
        t = self.combo_type.currentText()

        if t == "Linear":
            return {
                "type": "Linear",
                "neurons": self.spin_neurons.value(),
                "activation": self.combo_activation.currentText(),
            }
        elif t == "Conv1d":
            return {
                "type": "Conv1d",
                "out_channels": self.spin_out_channels.value(),
                "kernel_size": self.spin_kernel.value(),
                "stride": self.spin_stride.value(),
                "padding": self.spin_padding.value(),
            }
        elif t in ("MaxPool1d", "AvgPool1d"):
            return {
                "type": t,
                "kernel_size": self.spin_kernel.value(),
                "stride": self.spin_stride.value(),
            }
        elif t == "Flatten":
            return {"type": "Flatten"}
        elif t == "Dropout":
            return {
                "type": "Dropout",
                "rate": round(self.spin_dropout.value(), 4),
            }
        elif t == "BatchNorm1d":
            return {"type": "BatchNorm1d"}
        else:
            return {"type": t}

    def set_config(self, config: dict) -> None:
        """
        Populate the row's widgets from a blueprint dictionary.

        Parameters
        ----------
        config : dict
            A layer blueprint, e.g. ``{"type": "Linear", "neurons": 64}``.
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
            elif layer_type == "Conv1d":
                self.spin_out_channels.setValue(config.get("out_channels", 32))
                self.spin_kernel.setValue(config.get("kernel_size", 3))
                self.spin_stride.setValue(config.get("stride", 1))
                self.spin_padding.setValue(config.get("padding", 0))
            elif layer_type in ("MaxPool1d", "AvgPool1d"):
                self.spin_kernel.setValue(config.get("kernel_size", 2))
                self.spin_stride.setValue(config.get("stride", 2))
            elif layer_type == "Dropout":
                self.spin_dropout.setValue(config.get("rate", 0.3))
            # Flatten, BatchNorm1d — no params to set
        except Exception:
            pass  # Gracefully ignore malformed config

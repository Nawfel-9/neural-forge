from __future__ import annotations

import math
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend.training_config import get_all_optimizers, get_losses_for
from backend.model_builder import build_and_validate
from ui.monitor_panel import MonitorPanel
from ui.plot_panel import PlotPanel
from utils.project_state import ProjectState
from workers.training_worker import TrainingWorker

# ── Developer Mode additions ──────────────────────────────
_DevTrainer = None
_DevConfig  = None

def _import_dev_deps():
    global _DevTrainer, _DevConfig
    if _DevTrainer is None:
        from backend.dev_trainer import DevTrainer
        from utils.config_schema import DevProjectConfig
        _DevTrainer = DevTrainer
        _DevConfig  = DevProjectConfig


# ══════════════════════════════════════════════════════════════════════════════
#  Speedometer widget
# ══════════════════════════════════════════════════════════════════════════════

class Speedometer(QWidget):
    """Arc-style gauge — like a car tachometer."""

    def __init__(self, title: str, unit: str, max_val: float,
                 warn: float = 70.0, danger: float = 90.0, parent=None):
        super().__init__(parent)
        self.title   = title
        self.unit    = unit
        self.max_val = max_val
        self.warn    = warn
        self.danger  = danger
        self._value  = 0.0
        self.setMinimumSize(130, 130)

    def set_value(self, v: float):
        self._value = max(0.0, min(v, self.max_val))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h   = self.width(), self.height()
        size   = min(w, h) - 10
        cx, cy = w // 2, h // 2 + 8
        r      = size // 2

        # Arc background
        p.setPen(QPen(QColor("#1E293B"), 10, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(cx - r, cy - r, size, size, 225 * 16, -270 * 16)

        # Coloured arc foreground
        pct = self._value / self.max_val
        if self._value >= self.danger:
            color = QColor("#EF4444")
        elif self._value >= self.warn:
            color = QColor("#F59E0B")
        else:
            color = QColor("#00A3FF")

        p.setPen(QPen(color, 10, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        span = int(-270 * 16 * pct)
        p.drawArc(cx - r, cy - r, size, size, 225 * 16, span)

        # Needle
        angle_deg = 225 - 270 * pct
        angle_rad = math.radians(angle_deg)
        nx = cx + (r - 18) * math.cos(angle_rad)
        ny = cy - (r - 18) * math.sin(angle_rad)
        p.setPen(QPen(QColor("#F1F5F9"), 2))
        p.drawLine(int(cx), int(cy), int(nx), int(ny))

        # Centre dot
        p.setBrush(QBrush(QColor("#F1F5F9")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(cx - 4, cy - 4, 8, 8)

        # Value text
        font = QFont("Courier New", 11, QFont.Weight.Bold)
        p.setFont(font)
        p.setPen(QPen(color))
        p.drawText(0, cy + 6, w, 24, Qt.AlignmentFlag.AlignHCenter,
                   f"{self._value:.0f}{self.unit}")

        # Title
        font2 = QFont("Segoe UI", 7, QFont.Weight.Bold)
        p.setFont(font2)
        p.setPen(QPen(QColor("#475569")))
        p.drawText(0, cy + 26, w, 18, Qt.AlignmentFlag.AlignHCenter,
                   self.title.upper())
        p.end()


# ══════════════════════════════════════════════════════════════════════════════
#  Thermometer widget
# ══════════════════════════════════════════════════════════════════════════════

class Thermometer(QWidget):
    """Vertical bar thermometer."""

    def __init__(self, title: str, max_val: float = 110.0,
                 warn: float = 75.0, danger: float = 90.0, parent=None):
        super().__init__(parent)
        self.title   = title
        self.max_val = max_val
        self.warn    = warn
        self.danger  = danger
        self._value  = 0.0
        self.setMinimumSize(54, 130)
        self.setMaximumWidth(70)

    def set_value(self, v: float):
        self._value = max(0.0, min(v, self.max_val))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h   = self.width(), self.height()
        bw     = 14
        bx     = (w - bw) // 2
        bulb_r = 10
        top_y  = 18
        bot_y  = h - bulb_r * 2 - 10
        bar_h  = bot_y - top_y

        # Bar background
        p.setBrush(QBrush(QColor("#1E293B")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(bx, top_y, bw, bar_h, 4, 4)

        # Filled portion
        pct = self._value / self.max_val
        fill_h = int(bar_h * pct)
        if self._value >= self.danger:
            color = QColor("#EF4444")
        elif self._value >= self.warn:
            color = QColor("#F59E0B")
        else:
            color = QColor("#10B981")

        p.setBrush(QBrush(color))
        p.drawRoundedRect(bx, top_y + bar_h - fill_h, bw, fill_h, 4, 4)

        # Bulb
        p.setBrush(QBrush(color))
        p.drawEllipse(bx - (bulb_r - bw // 2), bot_y, bulb_r * 2, bulb_r * 2)

        # Value text
        font = QFont("Courier New", 8, QFont.Weight.Bold)
        p.setFont(font)
        p.setPen(QPen(color))
        p.drawText(0, top_y - 16, w, 16, Qt.AlignmentFlag.AlignHCenter,
                   f"{self._value:.0f}°")

        # Title
        font2 = QFont("Segoe UI", 6, QFont.Weight.Bold)
        p.setFont(font2)
        p.setPen(QPen(QColor("#475569")))
        p.drawText(0, h - 14, w, 14, Qt.AlignmentFlag.AlignHCenter,
                   self.title.upper())
        p.end()


# ══════════════════════════════════════════════════════════════════════════════
#  Usage bar (RAM / VRAM)
# ══════════════════════════════════════════════════════════════════════════════

class UsageBar(QWidget):
    """Horizontal labelled usage bar for RAM / VRAM."""

    def __init__(self, title: str, unit: str = "MB", parent=None):
        super().__init__(parent)
        self.title = title
        self.unit  = unit
        self.setFixedHeight(46)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        self._label = QLabel(f"{title}: — / —")
        self._label.setStyleSheet(
            "color: #94A3B8; font-size: 8pt; font-weight: 700;")
        lay.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(self._qss("#00A3FF"))
        lay.addWidget(self._bar)

    def _qss(self, color: str) -> str:
        return f"""
            QProgressBar {{
                background: #1E293B; border: none; border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {color}; border-radius: 3px;
            }}
        """

    def set_value(self, used: float, total: float):
        total = max(total, 1.0)
        pct   = used / total
        color = "#EF4444" if pct > 0.9 else "#F59E0B" if pct > 0.75 else "#00A3FF"
        self._bar.setValue(int(pct * 1000))
        self._bar.setStyleSheet(self._qss(color))
        self._label.setText(
            f"{self.title}: {used:.0f} / {total:.0f} {self.unit}  ({pct*100:.0f}%)"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Dev Hardware Dashboard
# ══════════════════════════════════════════════════════════════════════════════

class DevHardwareDashboard(QWidget):
    """
    Row of instruments shown in developer mode.
    Receives stats via update_stats(payload) — payload.gpu_temp from MonitorPanel.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        import psutil
        self._psutil = psutil

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(16)

        self.spd_gpu = Speedometer("GPU Usage", "%",  100, warn=75, danger=90)
        self.spd_cpu = Speedometer("CPU Usage", "%",  100, warn=75, danger=90)
        self.thm_gpu = Thermometer("GPU Temp",  max_val=110, warn=75, danger=90)
        self.thm_cpu = Thermometer("CPU Temp",  max_val=105, warn=75, danger=90)

        bar_col = QWidget()
        bar_lay = QVBoxLayout(bar_col)
        bar_lay.setContentsMargins(0, 0, 0, 0)
        bar_lay.setSpacing(8)
        self.bar_ram  = UsageBar("RAM",  unit="MB")
        self.bar_vram = UsageBar("VRAM", unit="MB")
        bar_lay.addWidget(self.bar_ram)
        bar_lay.addWidget(self.bar_vram)
        bar_lay.addStretch()

        lay.addWidget(self.spd_gpu)
        lay.addWidget(self.thm_gpu)
        lay.addWidget(self.spd_cpu)
        lay.addWidget(self.thm_cpu)
        lay.addWidget(bar_col, stretch=1)

    def update_stats(self, payload):
        self.spd_gpu.set_value(self._gpu_usage())
        self.thm_gpu.set_value(payload.gpu_temp)
        self.spd_cpu.set_value(self._psutil.cpu_percent(interval=None))
        self.thm_cpu.set_value(self._cpu_temp())

        vm = self._psutil.virtual_memory()
        self.bar_ram.set_value(vm.used / 1024**2, vm.total / 1024**2)

        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            self.bar_vram.set_value(used, total)
        else:
            self.bar_vram.set_value(0, 1)

    def _gpu_usage(self) -> float:
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            return 0.0

    def _cpu_temp(self) -> float:
        try:
            temps = self._psutil.sensors_temperatures()
            for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                if key in temps and temps[key]:
                    return temps[key][0].current
        except Exception:
            pass
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Training Window
# ══════════════════════════════════════════════════════════════════════════════

class TrainingWindow(QWidget):
    """Training and evaluation dashboard."""

    def __init__(self, project_state: ProjectState, on_back=None, parent=None) -> None:
        super().__init__(parent)
        self.state = project_state
        self._on_back_callback = on_back
        self.worker: TrainingWorker | None = None

        self._build_ui()
        self.monitor_panel.start()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(24)

        # ── Header ────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        title_lay = QVBoxLayout()
        title = QLabel("Training & Evaluation Lab")
        title.setProperty("class", "PageTitle")
        subtitle = QLabel("Train your model and evaluate its performance in real-time.")
        subtitle.setProperty("class", "PageSubtitle")
        title_lay.addWidget(title)
        title_lay.addWidget(subtitle)
        header_row.addLayout(title_lay)
        header_row.addStretch()
        self.monitor_panel = MonitorPanel()
        header_row.addWidget(self.monitor_panel)
        root.addLayout(header_row)

        # ── No-Code config row (hidden in dev mode) ───────────────────
        self._nocode_config_row = QWidget()
        nc_lay = QHBoxLayout(self._nocode_config_row)
        nc_lay.setContentsMargins(0, 0, 0, 0)
        nc_lay.addWidget(self._build_hyperparams_panel())
        nc_lay.addWidget(self._build_training_config_panel())
        nc_lay.addWidget(self._build_splitting_panel())
        nc_lay.addWidget(self._build_hardware_panel())
        root.addWidget(self._nocode_config_row)

        # ── Dev hardware dashboard (hidden in no-code mode) ────────────
        self._dev_dashboard = DevHardwareDashboard()
        self._dev_dashboard.setVisible(False)
        self.monitor_panel.stats_updated.connect(self._dev_dashboard.update_stats)
        root.addWidget(self._dev_dashboard)

        # ── Plots + log ────────────────────────────────────────────────
        visuals_row = QHBoxLayout()
        self.plot_panel = PlotPanel(is_classification=self.state.problem_type == "classification")
        visuals_row.addWidget(self.plot_panel, stretch=2)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #30363d; background: #0d1117; }")

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("font-family: monospace; background-color: #0d1117; color: #c9d1d9; border: none;")

        self.model_summary = QTextEdit()
        self.model_summary.setReadOnly(True)
        self.model_summary.setStyleSheet("font-family: monospace; background-color: #0d1117; color: #c9d1d9; border: none;")

        self.tabs.addTab(self.log_console, "Logs")
        self.tabs.addTab(self.model_summary, "Model Summary")
        visuals_row.addWidget(self.tabs, stretch=1)
        root.addLayout(visuals_row, stretch=1)

        # ── Evaluation metrics ─────────────────────────────────────────
        self.metrics_group = QGroupBox("Evaluation Metrics")
        self.metrics_group.setVisible(False)
        self.metrics_layout = QHBoxLayout(self.metrics_group)
        root.addWidget(self.metrics_group)

        # ── Progress ───────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

        # ── Button bar ─────────────────────────────────────────────────
        btn_bar = QHBoxLayout()
        if self._on_back_callback:
            self.btn_back = QPushButton("Back to Model")
            self.btn_back.setMinimumSize(150, 44)
            self.btn_back.clicked.connect(self._on_back_callback)
            btn_bar.addWidget(self.btn_back)

        btn_bar.addStretch()

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMinimumSize(150, 44)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        btn_bar.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("Re-initialize Weights")
        self.btn_reset.setMinimumSize(190, 44)
        self.btn_reset.clicked.connect(self._reset_weights)
        btn_bar.addWidget(self.btn_reset)

        self.btn_train = QPushButton("Start Training")
        self.btn_train.setProperty("class", "primary")
        self.btn_train.setMinimumSize(250, 44)
        self.btn_train.clicked.connect(self._start_training)
        btn_bar.addWidget(self.btn_train)
        root.addLayout(btn_bar)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    # ------------------------------------------------------------------
    # No-Code panel builders (unchanged)
    # ------------------------------------------------------------------
    def _build_hyperparams_panel(self) -> QGroupBox:
        group = QGroupBox("Hyperparameters")
        lay = QFormLayout(group)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(4)
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setValue(self.state.hyperparams.get("lr", 0.001))
        lay.addRow("Learning Rate:", self.spin_lr)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(self.state.hyperparams.get("epochs", 50))
        lay.addRow("Epochs:", self.spin_epochs)
        self.spin_bs = QSpinBox()
        self.spin_bs.setRange(1, 1024)
        self.spin_bs.setValue(self.state.hyperparams.get("batch_size", 32))
        lay.addRow("Batch Size:", self.spin_bs)
        return group

    def _build_training_config_panel(self) -> QGroupBox:
        group = QGroupBox("Training Configuration")
        lay = QFormLayout(group)
        self.combo_loss = QComboBox()
        self._populate_loss_combo()
        lay.addRow("Loss Function:", self.combo_loss)
        self.combo_optimizer = QComboBox()
        self.combo_optimizer.addItems(get_all_optimizers())
        idx = self.combo_optimizer.findText(self.state.optimizer_name)
        if idx >= 0:
            self.combo_optimizer.setCurrentIndex(idx)
        lay.addRow("Optimizer:", self.combo_optimizer)
        return group

    def _build_splitting_panel(self) -> QGroupBox:
        group = QGroupBox("Data Splitting")
        self.split_lay = QFormLayout(group)
        self.combo_split_method = QComboBox()
        self.combo_split_method.addItems(["percentage", "kfold"])
        self.combo_split_method.currentTextChanged.connect(self._on_split_method_changed)
        self.split_lay.addRow("Method:", self.combo_split_method)
        self.spin_split_ratio = QDoubleSpinBox()
        self.spin_split_ratio.setRange(0.1, 0.99)
        self.spin_split_ratio.setSingleStep(0.05)
        self.spin_split_ratio.setValue(self.state.split_config.get("ratio", 0.8))
        self.split_lay.addRow("Train Ratio:", self.spin_split_ratio)
        self.spin_kfold = QSpinBox()
        self.spin_kfold.setRange(2, 20)
        self.spin_kfold.setValue(self.state.split_config.get("k", 5))
        self.split_lay.addRow("K-Folds:", self.spin_kfold)
        self.combo_resample = QComboBox()
        self.combo_resample.addItems(["none", "smote", "undersample"])
        self.combo_resample.setCurrentText(self.state.split_config.get("resample", "none"))
        self.split_lay.addRow("Imbalanced Resampling:", self.combo_resample)
        method = self.state.split_config.get("method", "percentage")
        idx = self.combo_split_method.findText(method)
        if idx >= 0:
            self.combo_split_method.setCurrentIndex(idx)
        self._on_split_method_changed(method)
        return group

    def _build_hardware_panel(self) -> QGroupBox:
        group = QGroupBox("Hardware Selection")
        lay = QVBoxLayout(group)
        self.combo_device = QComboBox()
        self.combo_device.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            self.combo_device.addItem("CUDA (NVIDIA GPU)", "cuda")
        if torch.backends.mps.is_available():
            self.combo_device.addItem("MPS (Apple Silicon)", "mps")
        lay.addWidget(QLabel("Select Compute Device:"))
        lay.addWidget(self.combo_device)
        lay.addStretch()
        return group

    def _populate_loss_combo(self) -> None:
        losses = get_losses_for(self.state.problem_type)
        self.combo_loss.blockSignals(True)
        self.combo_loss.clear()
        self.combo_loss.addItems(losses)
        self.combo_loss.blockSignals(False)
        idx = self.combo_loss.findText(self.state.loss_fn_name)
        self.combo_loss.setCurrentIndex(idx if idx >= 0 else 0)

    def _on_split_method_changed(self, method: str) -> None:
        is_pct = method == "percentage"
        self.spin_split_ratio.setVisible(is_pct)
        lbl = self.split_lay.labelForField(self.spin_split_ratio)
        if lbl:
            lbl.setVisible(is_pct)
        self.spin_kfold.setVisible(not is_pct)
        lbl2 = self.split_lay.labelForField(self.spin_kfold)
        if lbl2:
            lbl2.setVisible(not is_pct)

    # ------------------------------------------------------------------
    # refresh_ui
    # ------------------------------------------------------------------
    def refresh_ui(self) -> None:
        is_dev = getattr(self.state, "training_mode", "nocode") == "dev"

        self._nocode_config_row.setVisible(not is_dev)
        self._dev_dashboard.setVisible(is_dev)

        if is_dev:
            self.btn_train.setText("▶  Start Dev Training")
            self.btn_reset.setVisible(False)
            if hasattr(self, "btn_back"):
                self.btn_back.setVisible(False)
            self.plot_panel.set_is_classification(True)
        else:
            self.btn_train.setText("Start Training")
            self.btn_reset.setVisible(True)
            if hasattr(self, "btn_back"):
                self.btn_back.setVisible(True)

            self.spin_lr.setValue(self.state.hyperparams.get("lr", 0.001))
            self.spin_epochs.setValue(self.state.hyperparams.get("epochs", 50))
            self.spin_bs.setValue(self.state.hyperparams.get("batch_size", 32))
            self._populate_loss_combo()
            idx = self.combo_optimizer.findText(self.state.optimizer_name)
            if idx >= 0:
                self.combo_optimizer.setCurrentIndex(idx)
            sc = self.state.split_config
            idx = self.combo_split_method.findText(sc.get("method", "percentage"))
            if idx >= 0:
                self.combo_split_method.setCurrentIndex(idx)
            if "ratio" in sc:
                self.spin_split_ratio.setValue(sc["ratio"])
            if "k" in sc:
                self.spin_kfold.setValue(sc["k"])
            idx_res = self.combo_resample.findText(sc.get("resample", "none"))
            if idx_res >= 0:
                self.combo_resample.setCurrentIndex(idx_res)
            is_cls = self.state.problem_type == "classification"
            self.combo_resample.setVisible(is_cls)
            lbl = self.split_lay.labelForField(self.combo_resample)
            if lbl:
                lbl.setVisible(is_cls)
            self.plot_panel.set_is_classification(is_cls)

        self.plot_panel.clear()
        self._refresh_model_summary()
        self._clear_metrics()
        self.progress_bar.setValue(0)
        self.btn_stop.setEnabled(False)
        self.btn_train.setEnabled(True)
        if not is_dev:
            self.btn_reset.setEnabled(True)
            if hasattr(self, "btn_back"):
                self.btn_back.setEnabled(True)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _refresh_model_summary(self) -> None:
        if self.state.model:
            total = sum(p.numel() for p in self.state.model.parameters() if p.requires_grad)
            self.model_summary.setText(
                f"Total Trainable Parameters: {total:,}\n\nArchitecture:\n{self.state.model}"
            )
        else:
            self.model_summary.setText("No model available.")

    def _clear_metrics(self) -> None:
        self.metrics_group.setVisible(False)
        while self.metrics_layout.count():
            child = self.metrics_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _reset_weights(self) -> None:
        if not self.state.blueprint or self.state.input_features() == 0:
            QMessageBox.warning(self, "Reset Failed", "Blueprint or data is missing.")
            return
        model, dummy_input, success, msg = build_and_validate(
            self.state.blueprint, self.state.input_features()
        )
        if not success:
            QMessageBox.warning(self, "Reset Failed", f"Could not rebuild model:\n{msg}")
            return
        self.state.model = model
        self.state.dummy_tensor = dummy_input
        self._refresh_model_summary()
        self.progress_bar.setValue(0)
        self.plot_panel.clear()
        self._clear_metrics()
        self.log_console.append("\n" + "=" * 50)
        self.log_console.append("Model weights have been re-initialized to random values.")
        self.log_console.append("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # Training dispatch
    # ------------------------------------------------------------------
    def _start_training(self) -> None:
        is_dev = getattr(self.state, "training_mode", "nocode") == "dev"
        if is_dev:
            self._start_dev_training()
            return

        self.state.hyperparams["lr"] = self.spin_lr.value()
        self.state.hyperparams["epochs"] = self.spin_epochs.value()
        self.state.hyperparams["batch_size"] = self.spin_bs.value()
        self.state.device = self.combo_device.currentData()
        self.state.loss_fn_name = self.combo_loss.currentText()
        self.state.optimizer_name = self.combo_optimizer.currentText()

        resample = self.combo_resample.currentText() if self.state.problem_type == "classification" else "none"
        if self.combo_split_method.currentText() == "percentage":
            self.state.split_config = {"method": "percentage",
                                       "ratio": self.spin_split_ratio.value(),
                                       "resample": resample}
        else:
            self.state.split_config = {"method": "kfold",
                                       "k": self.spin_kfold.value(),
                                       "resample": resample}

        self.btn_train.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.btn_stop.setEnabled(True)
        if hasattr(self, "btn_back"):
            self.btn_back.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_console.clear()
        self.plot_panel.clear()
        self._clear_metrics()

        self.worker = TrainingWorker(self.state)
        self.worker.log_message.connect(self._append_log)
        self.worker.batch_progress.connect(self._update_progress)
        self.worker.epoch_finished.connect(self._on_epoch)
        self.worker.evaluation_finished.connect(self._on_evaluation)
        self.worker.training_finished.connect(self._on_finished)
        self.worker.start()

    def _stop_training(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._append_log("Stop requested. Waiting for current batch to finish...")
        if hasattr(self, "_dev_trainer") and self._dev_trainer.isRunning():
            self._dev_trainer.stop()
            self._append_log("Dev training stop requested...")

    def _append_log(self, text: str) -> None:
        self.log_console.append(text)
        self.log_console.verticalScrollBar().setValue(
            self.log_console.verticalScrollBar().maximum()
        )

    def _update_progress(self, current: int, total: int) -> None:
        self.progress_bar.setValue(int(100 * current / max(1, total)))

    def _on_epoch(self, epoch: int, t_loss: float, v_loss: float, metrics: dict | None = None) -> None:
        self.plot_panel.add_data(epoch, t_loss, v_loss, metrics)

    def _on_evaluation(self, metrics: dict) -> None:
        self._clear_metrics()
        for name, value in metrics.items():
            card = QWidget()
            mlay = QVBoxLayout(card)
            try:
                value_text = f"{float(value):.4f}"
            except (TypeError, ValueError):
                value_text = str(value)
            val_lbl = QLabel(value_text)
            val_lbl.setStyleSheet("font-size: 16pt; font-weight: bold; color: #10B981;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl = QLabel(name.upper())
            name_lbl.setStyleSheet("font-size: 8pt; color: #64748B; font-weight: 600;")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            mlay.addWidget(val_lbl)
            mlay.addWidget(name_lbl)
            self.metrics_layout.addWidget(card)
        self.metrics_group.setVisible(bool(metrics))

    def _on_finished(self, success: bool, msg: str) -> None:
        self.btn_train.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if hasattr(self, "btn_back"):
            self.btn_back.setEnabled(True)
        if success:
            QMessageBox.information(self, "Training Complete", msg)
        else:
            QMessageBox.critical(self, "Training Error", f"Training Failed:\n{msg}")

    # ══════════════════════════════════════════════════════
    #  Developer Mode training
    # ══════════════════════════════════════════════════════

    def _start_dev_training(self) -> None:
        from PyQt6.QtWidgets import QDialog
        from ui.window_training_dev import TaskPickerDialog
        _import_dev_deps()

        project_root = self.state.dev_project_path
        existing_cfg = _DevConfig.load(project_root)

        dlg = TaskPickerDialog(existing_config=existing_cfg, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        kwargs = dlg.get_config_kwargs()
        self._dev_config = _DevConfig(**{**existing_cfg.to_dict(), **kwargs})
        self._dev_config.save(project_root)

        self._append_log(
            f"Task: {self._dev_config.task}  |  "
            f"Loss: {self._dev_config.effective_loss()}  |  "
            f"Metrics: {', '.join(self._dev_config.effective_metrics())}"
        )

        self.plot_panel.clear()
        self.progress_bar.setValue(0)
        self._total_dev_epochs = self._dev_config.epochs

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._dev_trainer = _DevTrainer(project_root, self._dev_config, parent=self)
        self._dev_trainer.training_started.connect(
            lambda n: self._append_log(f"Training started — {n} epochs"))
        self._dev_trainer.epoch_completed.connect(self._on_dev_epoch)
        self._dev_trainer.val_completed.connect(self._on_dev_val)
        self._dev_trainer.training_done.connect(self._on_dev_done)
        self._dev_trainer.training_error.connect(self._on_dev_error)
        self._dev_trainer.status_changed.connect(self._append_log)
        self._dev_trainer.paused_by_temp.connect(
            lambda t: self._append_log(f"⚠ Auto-paused — GPU {t:.0f}°C"))
        self._dev_trainer.resumed_by_temp.connect(
            lambda t: self._append_log(f"✓ Auto-resumed — GPU {t:.0f}°C"))

        self.monitor_panel.stats_updated.connect(
            lambda s: self._dev_trainer.update_gpu_temp(s.gpu_temp)
            if hasattr(self, "_dev_trainer") else None
        )

        self._dev_trainer.start()

    def _on_dev_epoch(self, epoch: int, loss: float, metrics: dict) -> None:
        pct = int((epoch / max(self._total_dev_epochs, 1)) * 100)
        self.progress_bar.setValue(pct)
        self._last_dev_train_loss = loss
        m_str = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._append_log(f"[Train] Epoch {epoch:>4}  loss: {loss:.5f}  {m_str}")

    def _on_dev_val(self, epoch: int, loss: float, metrics: dict) -> None:
        train_loss = getattr(self, "_last_dev_train_loss", loss)
        plot_metrics = {}
        if "accuracy" in metrics:
            plot_metrics["val_acc"] = metrics["accuracy"]
        if "iou" in metrics:
            plot_metrics["val_acc"] = metrics["iou"]
        if "dice" in metrics:
            plot_metrics["val_f1"] = metrics["dice"]
        self.plot_panel.add_data(epoch, float(train_loss), float(loss), plot_metrics)
        m_str = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._append_log(f"[Val]   Epoch {epoch:>4}  loss: {loss:.5f}  {m_str}")

    def _on_dev_done(self, msg: str) -> None:
        self._append_log(f"✓ {msg}")
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_dev_error(self, msg: str) -> None:
        self._append_log(f"✗ ERROR: {msg}")
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Training Error", msg[:600])
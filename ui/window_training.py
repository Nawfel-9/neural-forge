from __future__ import annotations

import math
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
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
from backend.dev_trainer import DevTrainer
from ui.monitor_panel import MonitorPanel
from ui.plot_panel import PlotPanel
from utils.config_schema import DevProjectConfig
from utils.project_state import ProjectState
from workers.training_worker import TrainingWorker


class Speedometer(QWidget):
    """Arc-style gauge used by the Developer Mode training dashboard."""

    def __init__(
        self,
        title: str,
        unit: str,
        max_value: float,
        warn: float = 70.0,
        danger: float = 90.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.max_value = max_value
        self.warn = warn
        self.danger = danger
        self._value = 0.0
        self.setMinimumSize(130, 130)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(float(value), self.max_value))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width, height = self.width(), self.height()
        size = min(width, height) - 10
        center_x = width // 2
        center_y = height // 2 + 8
        radius = size // 2

        painter.setPen(
            QPen(QColor("#1E293B"), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        painter.drawArc(center_x - radius, center_y - radius, size, size, 225 * 16, -270 * 16)

        percent = self._value / max(self.max_value, 1.0)
        if self._value >= self.danger:
            color = QColor("#EF4444")
        elif self._value >= self.warn:
            color = QColor("#F59E0B")
        else:
            color = QColor("#00A3FF")

        painter.setPen(QPen(color, 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(
            center_x - radius,
            center_y - radius,
            size,
            size,
            225 * 16,
            int(-270 * 16 * percent),
        )

        angle = math.radians(225 - 270 * percent)
        needle_x = center_x + (radius - 18) * math.cos(angle)
        needle_y = center_y - (radius - 18) * math.sin(angle)
        painter.setPen(QPen(QColor("#F1F5F9"), 2))
        painter.drawLine(int(center_x), int(center_y), int(needle_x), int(needle_y))

        painter.setBrush(QBrush(QColor("#F1F5F9")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center_x - 4, center_y - 4, 8, 8)

        painter.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        painter.setPen(QPen(color))
        painter.drawText(
            0,
            center_y + 6,
            width,
            24,
            Qt.AlignmentFlag.AlignHCenter,
            f"{self._value:.0f}{self.unit}",
        )

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#64748B")))
        painter.drawText(
            0,
            center_y + 26,
            width,
            18,
            Qt.AlignmentFlag.AlignHCenter,
            self.title.upper(),
        )
        painter.end()


class Thermometer(QWidget):
    """Vertical temperature gauge for the Developer Mode dashboard."""

    def __init__(
        self,
        title: str,
        max_value: float = 110.0,
        warn: float = 75.0,
        danger: float = 90.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.max_value = max_value
        self.warn = warn
        self.danger = danger
        self._value = 0.0
        self.setMinimumSize(54, 130)
        self.setMaximumWidth(70)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(float(value), self.max_value))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width, height = self.width(), self.height()
        bar_width = 14
        bar_x = (width - bar_width) // 2
        bulb_radius = 10
        top_y = 18
        bottom_y = height - bulb_radius * 2 - 10
        bar_height = max(1, bottom_y - top_y)

        painter.setBrush(QBrush(QColor("#1E293B")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(bar_x, top_y, bar_width, bar_height, 4, 4)

        percent = self._value / max(self.max_value, 1.0)
        fill_height = int(bar_height * percent)
        if self._value >= self.danger:
            color = QColor("#EF4444")
        elif self._value >= self.warn:
            color = QColor("#F59E0B")
        else:
            color = QColor("#10B981")

        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(
            bar_x,
            top_y + bar_height - fill_height,
            bar_width,
            fill_height,
            4,
            4,
        )
        painter.drawEllipse(
            bar_x - (bulb_radius - bar_width // 2),
            bottom_y,
            bulb_radius * 2,
            bulb_radius * 2,
        )

        painter.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        painter.setPen(QPen(color))
        painter.drawText(
            0,
            top_y - 16,
            width,
            16,
            Qt.AlignmentFlag.AlignHCenter,
            f"{self._value:.0f} C",
        )

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#64748B")))
        painter.drawText(
            0,
            height - 14,
            width,
            14,
            Qt.AlignmentFlag.AlignHCenter,
            self.title.upper(),
        )
        painter.end()


class UsageBar(QWidget):
    """Horizontal RAM/VRAM usage bar used in Developer Mode."""

    def __init__(self, title: str, unit: str = "MB", parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.setFixedHeight(46)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._label = QLabel(f"{title}: - / -")
        self._label.setStyleSheet("color: #94A3B8; font-size: 8pt; font-weight: 700;")
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(self._bar_style("#00A3FF"))
        layout.addWidget(self._bar)

    def _bar_style(self, color: str) -> str:
        return f"""
            QProgressBar {{
                background: #1E293B;
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 3px;
            }}
        """

    def set_value(self, used: float, total: float) -> None:
        total = max(float(total), 1.0)
        used = max(0.0, float(used))
        percent = min(used / total, 1.0)
        if percent > 0.9:
            color = "#EF4444"
        elif percent > 0.75:
            color = "#F59E0B"
        else:
            color = "#00A3FF"

        self._bar.setValue(int(percent * 1000))
        self._bar.setStyleSheet(self._bar_style(color))
        self._label.setText(
            f"{self.title}: {used:.0f} / {total:.0f} {self.unit} ({percent * 100:.0f}%)"
        )


class DevHardwareDashboard(QWidget):
    """Instrument row shown above plots when Training runs in Developer Mode."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        import psutil

        self._psutil = psutil

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(16)

        self.gpu_usage = Speedometer("GPU Usage", "%", 100, warn=75, danger=90)
        self.gpu_temp = Thermometer("GPU Temp", max_value=110, warn=75, danger=90)
        self.cpu_usage = Speedometer("CPU Usage", "%", 100, warn=75, danger=90)
        self.cpu_temp = Thermometer("CPU Temp", max_value=105, warn=75, danger=90)

        bar_column = QWidget()
        bar_layout = QVBoxLayout(bar_column)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(8)
        self.ram_bar = UsageBar("RAM", unit="MB")
        self.vram_bar = UsageBar("VRAM", unit="MB")
        bar_layout.addWidget(self.ram_bar)
        bar_layout.addWidget(self.vram_bar)
        bar_layout.addStretch()

        layout.addWidget(self.gpu_usage)
        layout.addWidget(self.gpu_temp)
        layout.addWidget(self.cpu_usage)
        layout.addWidget(self.cpu_temp)
        layout.addWidget(bar_column, stretch=1)

    def update_stats(self, payload) -> None:
        self.gpu_usage.set_value(self._gpu_utilization())
        self.gpu_temp.set_value(getattr(payload, "gpu_temp", 0.0))
        self.cpu_usage.set_value(self._psutil.cpu_percent(interval=None))
        self.cpu_temp.set_value(self._cpu_temperature())

        memory = self._psutil.virtual_memory()
        self.ram_bar.set_value(memory.used / 1024**2, memory.total / 1024**2)
        used_vram, total_vram = self._vram_usage()
        self.vram_bar.set_value(used_vram, total_vram)

    def _gpu_utilization(self) -> float:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            return 0.0

    def _cpu_temperature(self) -> float:
        sensors_temperatures = getattr(self._psutil, "sensors_temperatures", None)
        if sensors_temperatures is None:
            return 0.0
        try:
            temperatures = sensors_temperatures()
            for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                if key in temperatures and temperatures[key]:
                    return float(temperatures[key][0].current)
        except Exception:
            return 0.0
        return 0.0

    def _vram_usage(self) -> tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 1.0
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes
            return used_bytes / 1024**2, total_bytes / 1024**2
        except Exception:
            used_mb = torch.cuda.memory_allocated() / 1024**2
            total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return used_mb, total_mb


class TrainingWindow(QWidget):
    """Training and evaluation dashboard."""

    def __init__(self, project_state: ProjectState, on_back=None, parent=None) -> None:
        super().__init__(parent)
        self.state = project_state
        self._on_back_callback = on_back
        self.worker: TrainingWorker | None = None
        self.dev_worker: DevTrainer | None = None
        self._total_dev_epochs = 1
        self._last_dev_train_loss = 0.0

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

        self.nocode_config_widget = QWidget()
        config_row = QHBoxLayout(self.nocode_config_widget)
        config_row.setContentsMargins(0, 0, 0, 0)
        config_row.addWidget(self._build_hyperparams_panel())
        config_row.addWidget(self._build_training_config_panel())
        config_row.addWidget(self._build_splitting_panel())
        config_row.addWidget(self._build_hardware_panel())
        root.addWidget(self.nocode_config_widget)

        self.dev_config_group = QGroupBox("Developer Mode Project")
        dev_layout = QVBoxLayout(self.dev_config_group)
        self.lbl_dev_project = QLabel("No Developer Mode project imported.")
        self.lbl_dev_project.setWordWrap(True)
        self.lbl_dev_project.setStyleSheet("font-weight: 700;")
        dev_layout.addWidget(self.lbl_dev_project)
        self.lbl_dev_contract = QLabel(
            "Start Training loads config.yaml and expects model.py to define "
            "get_model(config), dataset.py to define get_dataloader(config, split), "
            "and optional loss.py / metrics.py hooks."
        )
        self.lbl_dev_contract.setWordWrap(True)
        self.lbl_dev_contract.setStyleSheet("color: #64748B;")
        dev_layout.addWidget(self.lbl_dev_contract)
        self.dev_config_group.setVisible(False)
        root.addWidget(self.dev_config_group)

        self.dev_dashboard = DevHardwareDashboard()
        self.dev_dashboard.setVisible(False)
        self.monitor_panel.stats_updated.connect(self.dev_dashboard.update_stats)
        root.addWidget(self.dev_dashboard)

        visuals_row = QHBoxLayout()
        self.plot_panel = PlotPanel(is_classification=self.state.problem_type == "classification")
        visuals_row.addWidget(self.plot_panel, stretch=2)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("TrainingTabs")

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setProperty("class", "CodeConsole")

        self.model_summary = QTextEdit()
        self.model_summary.setReadOnly(True)
        self.model_summary.setProperty("class", "CodeConsole")

        self.tabs.addTab(self.log_console, "Logs")
        self.tabs.addTab(self.model_summary, "Model Summary")
        visuals_row.addWidget(self.tabs, stretch=1)
        root.addLayout(visuals_row, stretch=1)

        self.metrics_group = QGroupBox("Evaluation Metrics")
        self.metrics_group.setVisible(False)
        self.metrics_layout = QHBoxLayout(self.metrics_group)
        root.addWidget(self.metrics_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

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
        is_percentage = method == "percentage"
        self.spin_split_ratio.setVisible(is_percentage)
        ratio_lbl = self.split_lay.labelForField(self.spin_split_ratio)
        if ratio_lbl:
            ratio_lbl.setVisible(is_percentage)

        self.spin_kfold.setVisible(not is_percentage)
        kfold_lbl = self.split_lay.labelForField(self.spin_kfold)
        if kfold_lbl:
            kfold_lbl.setVisible(not is_percentage)

    def apply_theme(self, is_dark: bool) -> None:
        """Propagate theme changes to sub-panels and update specific widgets."""
        self.plot_panel.apply_theme(is_dark)
        
        # Force re-polish for class-based styles
        for widget in [self.log_console, self.model_summary, self.monitor_panel]:
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def refresh_ui(self) -> None:
        is_dev = getattr(self.state, "training_mode", "nocode") == "dev"
        self.nocode_config_widget.setVisible(not is_dev)
        self.dev_config_group.setVisible(False)
        self.dev_dashboard.setVisible(is_dev)
        self.btn_reset.setVisible(not is_dev)

        if hasattr(self, "btn_back"):
            self.btn_back.setText("Back to Developer Mode" if is_dev else "Back to Model")
            self.btn_back.setVisible(not is_dev)

        if is_dev:
            project_path = getattr(self.state, "dev_project_path", "")
            self.lbl_dev_project.setText(project_path or "No Developer Mode project imported.")
            self.btn_train.setText("▶  Start Dev Training")
            self.btn_train.setEnabled(bool(project_path))
            self.btn_stop.setEnabled(False)
            self.plot_panel.set_is_classification(True)
            self.plot_panel.clear()
            self._refresh_model_summary()
            self._clear_metrics()
            self.progress_bar.setValue(0)
            return

        self.btn_train.setText("Start Training")
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

        is_classification = self.state.problem_type == "classification"
        self.combo_resample.setVisible(is_classification)
        resample_lbl = self.split_lay.labelForField(self.combo_resample)
        if resample_lbl:
            resample_lbl.setVisible(is_classification)

        self.plot_panel.set_is_classification(is_classification)
        self.plot_panel.clear()
        self._refresh_model_summary()
        self._clear_metrics()
        self.progress_bar.setValue(0)
        self.btn_stop.setEnabled(False)
        self.btn_train.setEnabled(True)
        self.btn_reset.setEnabled(True)
        if hasattr(self, "btn_back"):
            self.btn_back.setEnabled(True)

    def _refresh_model_summary(self) -> None:
        if getattr(self.state, "training_mode", "nocode") == "dev":
            if self.state.dev_project_path:
                cfg = DevProjectConfig.load(self.state.dev_project_path)
                self.model_summary.setText(
                    "Developer Mode Project\n\n"
                    f"Path: {self.state.dev_project_path}\n"
                    f"Task: {cfg.task}\n"
                    f"Epochs: {cfg.epochs}\n"
                    f"Batch size: {cfg.batch_size}\n"
                    f"Learning rate: {cfg.learning_rate}\n"
                    f"Optimizer: {cfg.optimizer}\n"
                    f"Device: {cfg.device}\n"
                    f"Loss: {cfg.effective_loss()}\n"
                    f"Metrics: {', '.join(cfg.effective_metrics())}\n\n"
                    "Required runtime hooks:\n"
                    "- model.py: get_model(config)\n"
                    "- dataset.py: get_dataloader(config, split)\n"
                    "- loss.py: get_loss(config) optional\n"
                    "- metrics.py: get_metrics(config) optional"
                )
            else:
                self.model_summary.setText("No Developer Mode project imported.")
            return

        if self.state.model:
            total_params = sum(p.numel() for p in self.state.model.parameters() if p.requires_grad)
            summary_text = f"Total Trainable Parameters: {total_params:,}\n\nArchitecture:\n{self.state.model}"
            self.model_summary.setText(summary_text)
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

    def _start_training(self) -> None:
        if getattr(self.state, "training_mode", "nocode") == "dev":
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
            self.state.split_config = {
                "method": "percentage",
                "ratio": self.spin_split_ratio.value(),
                "resample": resample,
            }
        else:
            self.state.split_config = {
                "method": "kfold",
                "k": self.spin_kfold.value(),
                "resample": resample,
            }

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
        if self.dev_worker and self.dev_worker.isRunning():
            self.dev_worker.stop()
            self._append_log("Developer training stop requested...")

    def _append_log(self, text: str) -> None:
        self.log_console.append(text)
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / max(1, total))
        self.progress_bar.setValue(pct)

    def _on_epoch(self, epoch: int, t_loss: float, v_loss: float, metrics: dict | None = None) -> None:
        self.plot_panel.add_data(epoch, t_loss, v_loss, metrics)

    def _on_evaluation(self, metrics: dict) -> None:
        self.state.training_metrics = metrics  # Store for report generation
        self._clear_metrics()
        for name, value in metrics.items():
            metric_card = QWidget()
            mlay = QVBoxLayout(metric_card)

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
            self.metrics_layout.addWidget(metric_card)

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

    def _start_dev_training(self) -> None:
        project_path = getattr(self.state, "dev_project_path", "")
        if not project_path:
            QMessageBox.warning(self, "No Project Imported", "Import a Developer Mode project first.")
            return

        cfg = DevProjectConfig.load(project_path)
        cfg.save(project_path)
        self._total_dev_epochs = max(1, int(cfg.epochs))
        self._last_dev_train_loss = 0.0

        self.log_console.clear()
        self.plot_panel.clear()
        self._clear_metrics()
        self.progress_bar.setValue(0)
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        if hasattr(self, "btn_back"):
            self.btn_back.setEnabled(False)

        self._append_log(f"Developer project: {project_path}")
        self._append_log(
            f"Task: {cfg.task} | Loss: {cfg.effective_loss()} | "
            f"Metrics: {', '.join(cfg.effective_metrics())}"
        )

        self.dev_worker = DevTrainer(project_path, cfg, parent=self)
        self.dev_worker.training_started.connect(
            lambda total: self._append_log(f"Training started: {total} epoch(s)")
        )
        self.dev_worker.epoch_completed.connect(self._on_dev_epoch)
        self.dev_worker.val_completed.connect(self._on_dev_val)
        self.dev_worker.training_done.connect(self._on_dev_done)
        self.dev_worker.training_error.connect(self._on_dev_error)
        self.dev_worker.status_changed.connect(self._append_log)
        self.dev_worker.paused_by_temp.connect(
            lambda temp: self._append_log(f"Thermal pause: GPU {temp:.0f} C")
        )
        self.dev_worker.resumed_by_temp.connect(
            lambda temp: self._append_log(f"Thermal resume: GPU {temp:.0f} C")
        )
        try:
            self.monitor_panel.stats_updated.disconnect(self._on_monitor_stats)
        except TypeError:
            pass
        self.monitor_panel.stats_updated.connect(self._on_monitor_stats)
        self.dev_worker.finished.connect(self._cleanup_dev_worker)
        self.dev_worker.start()

    def _on_monitor_stats(self, stats) -> None:
        if self.dev_worker and self.dev_worker.isRunning():
            self.dev_worker.update_gpu_temp(getattr(stats, "gpu_temp", 0.0))

    def _on_dev_epoch(self, epoch: int, loss: float, metrics: dict) -> None:
        self._last_dev_train_loss = float(loss)
        pct = int(100 * epoch / max(1, self._total_dev_epochs))
        self.progress_bar.setValue(pct)
        metric_text = "  ".join(f"{key}: {float(value):.4f}" for key, value in metrics.items())
        self._append_log(f"[Train] Epoch {epoch}/{self._total_dev_epochs} loss={float(loss):.5f} {metric_text}")

    def _on_dev_val(self, epoch: int, loss: float, metrics: dict) -> None:
        plot_metrics = {}
        if "accuracy" in metrics:
            plot_metrics["val_acc"] = metrics["accuracy"]
        if "dice" in metrics:
            plot_metrics["val_f1"] = metrics["dice"]
        if "iou" in metrics and "val_acc" not in plot_metrics:
            plot_metrics["val_acc"] = metrics["iou"]
        self.plot_panel.add_data(epoch, self._last_dev_train_loss, float(loss), plot_metrics)
        self.state.training_metrics = {key: float(value) for key, value in metrics.items()}
        metric_text = "  ".join(f"{key}: {float(value):.4f}" for key, value in metrics.items())
        self._append_log(f"[Val]   Epoch {epoch}/{self._total_dev_epochs} loss={float(loss):.5f} {metric_text}")

    def _on_dev_done(self, msg: str) -> None:
        self._append_log(msg)
        self.progress_bar.setValue(100)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if hasattr(self, "btn_back"):
            self.btn_back.setEnabled(True)
        QMessageBox.information(self, "Developer Training Complete", msg)

    def _on_dev_error(self, msg: str) -> None:
        self._append_log(msg)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if hasattr(self, "btn_back"):
            self.btn_back.setEnabled(True)
        QMessageBox.critical(self, "Developer Training Error", msg[:1200])

    def _cleanup_dev_worker(self) -> None:
        try:
            self.monitor_panel.stats_updated.disconnect(self._on_monitor_stats)
        except TypeError:
            pass
        self.dev_worker = None

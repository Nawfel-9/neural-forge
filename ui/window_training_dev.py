"""
ui/window_training.py
=====================
Training window — shared between No-Code and Developer Mode .

Developer Mode path:
  1. User clicks "Continue to Training" from DevProjectWindow.
  2. TaskPickerDialog asks classification vs segmentation.
  3. DevProjectConfig is built, saved, and passed to DevTrainer.
  4. HardwareMonitor starts → feeds the dashboard header gauges.
  5. Epoch signals feed real-time pyqtgraph plots.
  6. Pause / Resume / Stop buttons control the trainer thread.

"""

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QDialogButtonBox, QComboBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QGroupBox, QProgressBar, QTextEdit, QSplitter,
    QFrame, QMessageBox, QScrollArea, QSizePolicy
)

from backend.hardware_monitor import HardwareMonitor, HardwareStats
from utils.project_state import ProjectState

# Lazy import — only needed in dev mode
_DevTrainer   = None
_DevConfig    = None


def _import_dev_deps():
    global _DevTrainer, _DevConfig
    if _DevTrainer is None:
        from backend.dev_trainer   import DevTrainer
        from utils.config_schema   import DevProjectConfig
        _DevTrainer = DevTrainer
        _DevConfig  = DevProjectConfig


# ══════════════════════════════════════════════════════════════════════════════
#  Gauge widget  (car-dashboard style arc)
# ══════════════════════════════════════════════════════════════════════════════

class GaugeWidget(QFrame):
    """
    Minimalist numeric gauge with a coloured progress bar,
    label, value and unit — styled like a car instrument cluster.
    """

    DANGER_COLOR   = "#EF4444"
    WARNING_COLOR  = "#F59E0B"
    NORMAL_COLOR   = "#00A3FF"
    GOOD_COLOR     = "#10B981"

    def __init__(self, title: str, unit: str, max_val: float,
                 warn: float = 70.0, danger: float = 90.0, parent=None):
        super().__init__(parent)
        self.max_val = max_val
        self.warn    = warn
        self.danger  = danger
        self.unit    = unit

        self.setObjectName("GaugeWidget")
        self.setStyleSheet("""
            QFrame#GaugeWidget {
                background: rgba(10, 18, 35, 0.85);
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 12px;
            }
        """)
        self.setFixedHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        # Title
        self._title_lbl = QLabel(title.upper())
        self._title_lbl.setStyleSheet(
            "color: #475569; font-size: 8pt; font-weight: 800; letter-spacing: 1.5px;"
        )
        layout.addWidget(self._title_lbl)

        # Value
        self._value_lbl = QLabel("—")
        self._value_lbl.setStyleSheet(
            "color: #F1F5F9; font-size: 22pt; font-weight: 900; font-family: 'Courier New', monospace;"
        )
        layout.addWidget(self._value_lbl)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(5)
        self._bar.setStyleSheet(self._bar_qss(self.NORMAL_COLOR))
        layout.addWidget(self._bar)

    def _bar_qss(self, color: str) -> str:
        return f"""
            QProgressBar {{
                background: rgba(255,255,255,0.06);
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 2px;
            }}
        """

    def update_value(self, value: float):
        pct  = min(value / self.max_val, 1.0)
        unit_str = self.unit

        if value >= self.danger:
            color = self.DANGER_COLOR
        elif value >= self.warn:
            color = self.WARNING_COLOR
        else:
            color = self.NORMAL_COLOR

        self._value_lbl.setText(f"{value:.0f}{unit_str}")
        self._value_lbl.setStyleSheet(
            f"color: {color}; font-size: 22pt; font-weight: 900; "
            "font-family: 'Courier New', monospace;"
        )
        self._bar.setValue(int(pct * 1000))
        self._bar.setStyleSheet(self._bar_qss(color))


# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard header  (row of gauges)
# ══════════════════════════════════════════════════════════════════════════════

class HardwareDashboard(QWidget):
    """Horizontal row of instrument gauges for GPU/CPU/RAM."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.g_vram  = GaugeWidget("VRAM",      "MB",  24576, warn=18000, danger=22000)
        self.g_gpu   = GaugeWidget("GPU",        "%",    100,  warn=75,    danger=90)
        self.g_temp  = GaugeWidget("GPU TEMP",  "°C",    110,  warn=75,    danger=90)
        self.g_cpu   = GaugeWidget("CPU",        "%",    100,  warn=75,    danger=90)
        self.g_ram   = GaugeWidget("RAM",        "%",    100,  warn=75,    danger=90)

        for g in (self.g_vram, self.g_gpu, self.g_temp, self.g_cpu, self.g_ram):
            layout.addWidget(g)

    @pyqtSlot(object)
    def on_stats(self, stats: HardwareStats):
        self.g_vram.update_value(stats.gpu_vram_used)
        self.g_gpu.update_value(stats.gpu_usage)
        self.g_temp.update_value(stats.gpu_temp)
        self.g_cpu.update_value(stats.cpu_usage)
        self.g_ram.update_value(stats.ram_percent)


# ══════════════════════════════════════════════════════════════════════════════
#  Task picker dialog
# ══════════════════════════════════════════════════════════════════════════════

class TaskPickerDialog(QDialog):
    """
    Asks the user to choose task type and basic hyper-params before training.
    Pre-fills values from an existing config.yaml if present.
    """

    def __init__(self, existing_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Training")
        self.setMinimumWidth(380)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = QLabel("Training Configuration")
        header.setStyleSheet("font-size: 15pt; font-weight: 800;")
        layout.addWidget(header)

        sub = QLabel(
            "Choose your task type. Loss and metrics will be set automatically\n"
            "unless loss.py / metrics.py are present in your project."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #64748B; font-size: 9.5pt;")
        layout.addWidget(sub)

        form = QFormLayout()
        form.setSpacing(12)

        # Task
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Classification", "Segmentation"])
        form.addRow("Task type:", self.task_combo)

        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10_000)
        self.epochs_spin.setValue(50)
        form.addRow("Epochs:", self.epochs_spin)

        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 2048)
        self.batch_spin.setValue(16)
        form.addRow("Batch size:", self.batch_spin)

        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        form.addRow("Learning rate:", self.lr_spin)

        # Optimizer
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["Adam", "SGD"])
        form.addRow("Optimizer:", self.opt_combo)

        layout.addLayout(form)

        # Pre-fill from existing config
        if existing_config is not None:
            idx = 0 if existing_config.task == "classification" else 1
            self.task_combo.setCurrentIndex(idx)
            self.epochs_spin.setValue(existing_config.epochs)
            self.batch_spin.setValue(existing_config.batch_size)
            self.lr_spin.setValue(existing_config.learning_rate)
            oidx = 0 if existing_config.optimizer.lower() == "adam" else 1
            self.opt_combo.setCurrentIndex(oidx)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def get_config_kwargs(self) -> dict:
        return {
            "task":          self.task_combo.currentText().lower(),
            "epochs":        self.epochs_spin.value(),
            "batch_size":    self.batch_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "optimizer":     self.opt_combo.currentText().lower(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Live plot panel
# ══════════════════════════════════════════════════════════════════════════════

class LivePlotPanel(QWidget):
    """Two side-by-side pyqtgraph plots: loss curve and metric(s) curve."""

    _TRAIN_PEN  = pg.mkPen(color="#00A3FF", width=2)
    _VAL_PEN    = pg.mkPen(color="#F59E0B", width=2, style=Qt.PenStyle.DashLine)
    _METRIC_PENS = [
        pg.mkPen(color="#10B981", width=2),
        pg.mkPen(color="#A78BFA", width=2),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOption("background", "transparent")
        pg.setConfigOption("foreground", "#94A3B8")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Loss plot
        self._loss_plot = pg.PlotWidget(title="Loss")
        self._loss_plot.setLabel("left",   "Loss")
        self._loss_plot.setLabel("bottom", "Epoch")
        self._loss_plot.addLegend()
        self._loss_plot.showGrid(x=True, y=True, alpha=0.15)
        self._loss_plot.setMinimumHeight(240)
        layout.addWidget(self._loss_plot)

        # Metric plot
        self._metric_plot = pg.PlotWidget(title="Metrics")
        self._metric_plot.setLabel("left",   "Value")
        self._metric_plot.setLabel("bottom", "Epoch")
        self._metric_plot.addLegend()
        self._metric_plot.showGrid(x=True, y=True, alpha=0.15)
        self._metric_plot.setMinimumHeight(240)
        layout.addWidget(self._metric_plot)

        # Data buffers
        self._epochs_train: list[float] = []
        self._train_loss:   list[float] = []
        self._epochs_val:   list[float] = []
        self._val_loss:     list[float] = []

        self._metric_epochs: dict[str, list[float]] = {}
        self._metric_vals:   dict[str, list[float]] = {}

        # Curve objects (created on first data point)
        self._train_curve  = self._loss_plot.plot(pen=self._TRAIN_PEN,  name="Train loss")
        self._val_curve    = self._loss_plot.plot(pen=self._VAL_PEN,    name="Val loss")
        self._metric_curves: dict[str, pg.PlotDataItem] = {}

    def reset(self):
        self._epochs_train.clear()
        self._train_loss.clear()
        self._epochs_val.clear()
        self._val_loss.clear()
        self._metric_epochs.clear()
        self._metric_vals.clear()
        self._train_curve.setData([], [])
        self._val_curve.setData([], [])
        for c in self._metric_curves.values():
            c.setData([], [])
        self._metric_curves.clear()

    def add_train_point(self, epoch: int, loss: float, metrics: dict):
        self._epochs_train.append(epoch)
        self._train_loss.append(loss)
        self._train_curve.setData(self._epochs_train, self._train_loss)

        # Metrics
        for i, (name, val) in enumerate(metrics.items()):
            if name not in self._metric_epochs:
                self._metric_epochs[name] = []
                self._metric_vals[name]   = []
                pen = self._METRIC_PENS[i % len(self._METRIC_PENS)]
                self._metric_curves[name] = self._metric_plot.plot(
                    pen=pen, name=name
                )
            self._metric_epochs[name].append(epoch)
            self._metric_vals[name].append(val)
            self._metric_curves[name].setData(
                self._metric_epochs[name], self._metric_vals[name]
            )

    def add_val_point(self, epoch: int, loss: float, metrics: dict):
        self._epochs_val.append(epoch)
        self._val_loss.append(loss)
        self._val_curve.setData(self._epochs_val, self._val_loss)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Training Window
# ══════════════════════════════════════════════════════════════════════════════

class TrainingWindow(QWidget):
    """
    Unified training window.
    When state.dev_project_path is set → Developer Mode path.
    Otherwise → No-Code path (placeholder for future work).
    """

    def __init__(self, state: ProjectState, on_back=None, parent=None):
        super().__init__(parent)
        self.state   = state
        self._on_back = on_back

        self._trainer: "DevTrainer | None"         = None
        self._hw_monitor: HardwareMonitor | None   = None
        self._config: "DevProjectConfig | None"    = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # ── Page header ────────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        title = QLabel("Train & Evaluate")
        title.setProperty("class", "PageTitle")
        hdr_row.addWidget(title)
        hdr_row.addStretch()
        if self._on_back:
            btn_back = QPushButton("← Back")
            btn_back.clicked.connect(self._on_back)
            hdr_row.addWidget(btn_back)
        root.addLayout(hdr_row)

        # ── Hardware dashboard ─────────────────────────────────────────
        self._hw_dashboard = HardwareDashboard()
        root.addWidget(self._hw_dashboard)

        # ── Status bar ─────────────────────────────────────────────────
        self._status_lbl = QLabel("Ready — load a project and configure training.")
        self._status_lbl.setStyleSheet("color: #64748B; font-size: 9pt; font-style: italic;")
        root.addWidget(self._status_lbl)

        # ── Progress bar ───────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFixedHeight(18)
        root.addWidget(self._progress)

        # ── Training controls ──────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        self._btn_start  = QPushButton("▶  Start Training")
        self._btn_start.setProperty("class", "primary")
        self._btn_start.setMinimumHeight(40)
        self._btn_start.clicked.connect(self._on_start)

        self._btn_pause  = QPushButton("⏸  Pause")
        self._btn_pause.setMinimumHeight(40)
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._on_pause)

        self._btn_resume = QPushButton("▶  Resume")
        self._btn_resume.setMinimumHeight(40)
        self._btn_resume.setEnabled(False)
        self._btn_resume.clicked.connect(self._on_resume)

        self._btn_stop   = QPushButton("⏹  Stop")
        self._btn_stop.setMinimumHeight(40)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)

        ctrl_row.addWidget(self._btn_start)
        ctrl_row.addWidget(self._btn_pause)
        ctrl_row.addWidget(self._btn_resume)
        ctrl_row.addWidget(self._btn_stop)
        ctrl_row.addStretch()
        root.addLayout(ctrl_row)

        # ── Live plots ─────────────────────────────────────────────────
        self._plots = LivePlotPanel()
        root.addWidget(self._plots, stretch=1)

        # ── Log output ─────────────────────────────────────────────────
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(110)
        self._log.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 9pt; "
            "background: rgba(10,18,35,0.7); color: #94A3B8; border: none;"
        )
        log_layout.addWidget(self._log)
        root.addWidget(log_group)

        # Epoch counter
        self._total_epochs = 0
        self._current_epoch = 0

    # ------------------------------------------------------------------
    # Called by main window when switching to this tab
    # ------------------------------------------------------------------
    def refresh_ui(self):
        """Called by NeuralForgeApp when switching to tab index 3."""
        is_dev = bool(getattr(self.state, "dev_project_path", ""))
        if is_dev:
            self._status_lbl.setText(
                f"Developer Mode — project: {self.state.dev_project_path}"
            )
        else:
            self._status_lbl.setText("No-Code mode — configure your pipeline first.")

    # ------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------
    def _on_start(self):
        is_dev = bool(getattr(self.state, "dev_project_path", ""))
        if not is_dev:
            QMessageBox.information(
                self, "Not ready",
                "No-Code training pipeline is not yet implemented.\n"
                "Use Developer Mode and import a project first."
            )
            return

        _import_dev_deps()

        # Load existing config if present
        project_root = self.state.dev_project_path
        existing_cfg = _DevConfig.load(project_root)

        # Always ask the user to confirm / adjust config
        dlg = TaskPickerDialog(existing_config=existing_cfg, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        kwargs = dlg.get_config_kwargs()
        self._config = _DevConfig(**{**existing_cfg.to_dict(), **kwargs})

        # Persist config
        self._config.save(project_root)
        self._log_line(f"Config saved → {project_root}/config.yaml")
        self._log_line(
            f"Task: {self._config.task}  |  "
            f"Loss: {self._config.effective_loss()}  |  "
            f"Metrics: {', '.join(self._config.effective_metrics())}"
        )

        # Reset UI
        self._plots.reset()
        self._progress.setValue(0)
        self._total_epochs = self._config.epochs
        self._current_epoch = 0

        # Start hardware monitor
        self._start_hw_monitor()

        # Build and start trainer
        self._trainer = _DevTrainer(project_root, self._config, parent=self)
        self._trainer.training_started.connect(self._on_training_started)
        self._trainer.epoch_completed.connect(self._on_epoch)
        self._trainer.val_completed.connect(self._on_val)
        self._trainer.training_done.connect(self._on_done)
        self._trainer.training_error.connect(self._on_error)
        self._trainer.status_changed.connect(self._on_status)
        self._trainer.paused_by_temp.connect(self._on_thermal_pause)
        self._trainer.resumed_by_temp.connect(self._on_thermal_resume)

        # Wire GPU temp from monitor to trainer
        if self._hw_monitor:
            self._hw_monitor.stats_updated.connect(
                lambda s: self._trainer.update_gpu_temp(s.gpu_temp)
                if self._trainer else None
            )

        self._trainer.start()
        self._set_controls(training=True, paused=False)

    # ------------------------------------------------------------------
    # Hardware monitor lifecycle
    # ------------------------------------------------------------------
    def _start_hw_monitor(self):
        if self._hw_monitor and self._hw_monitor.isRunning():
            return
        self._hw_monitor = HardwareMonitor(interval=1.0, parent=self)
        self._hw_monitor.stats_updated.connect(self._hw_dashboard.on_stats)
        self._hw_monitor.start()

    def _stop_hw_monitor(self):
        if self._hw_monitor and self._hw_monitor.isRunning():
            self._hw_monitor.stop()

    # ------------------------------------------------------------------
    # Trainer signal handlers
    # ------------------------------------------------------------------
    @pyqtSlot(int)
    def _on_training_started(self, total: int):
        self._total_epochs = total
        self._log_line(f"Training started — {total} epochs.")

    @pyqtSlot(int, float, dict)
    def _on_epoch(self, epoch: int, loss: float, metrics: dict):
        self._current_epoch = epoch
        pct = int((epoch / max(self._total_epochs, 1)) * 100)
        self._progress.setValue(pct)
        self._progress.setFormat(f"Epoch {epoch}/{self._total_epochs}  ({pct}%)")
        self._plots.add_train_point(epoch, loss, metrics)

        m_str = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._log_line(f"[Train] Epoch {epoch:>4}  loss: {loss:.5f}  {m_str}")

    @pyqtSlot(int, float, dict)
    def _on_val(self, epoch: int, loss: float, metrics: dict):
        self._plots.add_val_point(epoch, loss, metrics)
        m_str = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._log_line(f"[Val]   Epoch {epoch:>4}  loss: {loss:.5f}  {m_str}")

    @pyqtSlot(str)
    def _on_status(self, msg: str):
        self._status_lbl.setText(msg)

    @pyqtSlot(str)
    def _on_done(self, msg: str):
        self._log_line(f"✓ {msg}")
        self._status_lbl.setText(msg)
        self._set_controls(training=False, paused=False)
        self._stop_hw_monitor()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._log_line(f"✗ ERROR:\n{msg}")
        self._status_lbl.setText("Training failed — see log.")
        self._status_lbl.setStyleSheet("color: #EF4444; font-size: 9pt;")
        self._set_controls(training=False, paused=False)
        self._stop_hw_monitor()
        QMessageBox.critical(self, "Training Error", msg[:600])

    @pyqtSlot(float)
    def _on_thermal_pause(self, temp: float):
        self._log_line(
            f"⚠ AUTO-PAUSED — GPU temperature {temp:.0f}°C reached threshold "
            f"({self._config.auto_pause_temp}°C). Training will resume automatically."
        )
        self._status_lbl.setStyleSheet("color: #F59E0B; font-size: 9pt;")

    @pyqtSlot(float)
    def _on_thermal_resume(self, temp: float):
        self._log_line(
            f"✓ AUTO-RESUMED — GPU cooled to {temp:.0f}°C "
            f"(threshold: {self._config.resume_temp}°C)."
        )
        self._status_lbl.setStyleSheet("color: #10B981; font-size: 9pt;")

    # ------------------------------------------------------------------
    # Control buttons
    # ------------------------------------------------------------------
    def _on_pause(self):
        if self._trainer:
            self._trainer.pause()
        self._set_controls(training=True, paused=True)

    def _on_resume(self):
        if self._trainer:
            self._trainer.resume()
        self._set_controls(training=True, paused=False)

    def _on_stop(self):
        if self._trainer:
            self._trainer.stop()
        self._set_controls(training=False, paused=False)
        self._stop_hw_monitor()

    def _set_controls(self, training: bool, paused: bool):
        self._btn_start.setEnabled(not training)
        self._btn_pause.setEnabled(training and not paused)
        self._btn_resume.setEnabled(training and paused)
        self._btn_stop.setEnabled(training)

    # ------------------------------------------------------------------
    # Log helper
    # ------------------------------------------------------------------
    def _log_line(self, text: str):
        self._log.append(text)
        # Auto-scroll
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._on_stop()
        super().closeEvent(event)

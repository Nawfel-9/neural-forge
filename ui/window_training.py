"""
window_training.py
==================
Window 3 — Training, Monitoring & Export.

Initializes the `TrainingWorker`, displays logs, shows hardware selection,
connects to Phase 5 features (loss curves, resource monitor, ONNX export).
"""

from __future__ import annotations

import os
import psutil
import torch
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.project_state import ProjectState
from workers.training_worker import TrainingWorker
from backend.exporter import export_to_onnx


class TrainingWindow(QMainWindow):
    def __init__(
        self,
        project_state: ProjectState,
        on_back=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = project_state
        self._on_back_callback = on_back
        self.worker: TrainingWorker | None = None

        # Plot data
        self.plot_epochs = []
        self.train_losses = []
        self.val_losses = []

        self._init_window()
        self._build_ui()
        self._setup_resource_monitor()

    def _init_window(self) -> None:
        self.setWindowTitle("Neural Network Builder — Training & Monitoring")
        self.setMinimumSize(920, 680)
        self.resize(1000, 750)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # ── Header & Monitor ──────────────────────────────────────────
        header_row = QHBoxLayout()
        header = QLabel("⚙️  Training Studio")
        header.setProperty("class", "heading")
        header.setStyleSheet("font-size: 20px; font-weight: 700;")
        header_row.addWidget(header)
        header_row.addStretch()

        self.lbl_resources = QLabel("CPU: 0% | RAM: 0% | VRAM: N/A")
        self.lbl_resources.setStyleSheet("color: #8b949e; font-family: monospace; font-size: 13px;")
        header_row.addWidget(self.lbl_resources)
        root.addLayout(header_row)

        # ── Config Panel ──────────────────────────────────────────────
        config_row = QHBoxLayout()
        config_row.addWidget(self._build_hyperparams_panel())
        config_row.addWidget(self._build_hardware_panel())
        root.addLayout(config_row)

        # ── Visuals & Logs ────────────────────────────────────────────
        visuals_row = QHBoxLayout()
        
        # 1. PyQtGraph Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0d1117')
        self.plot_widget.setTitle("Loss Curve", color="#ffffff")
        self.plot_widget.setLabel('left', 'Loss')
        self.plot_widget.setLabel('bottom', 'Epoch')
        self.plot_widget.addLegend()
        self.train_line = self.plot_widget.plot(pen=pg.mkPen(color='#3fb950', width=2), name="Train Loss")
        self.val_line = self.plot_widget.plot(pen=pg.mkPen(color='#58a6ff', width=2), name="Val Loss")
        visuals_row.addWidget(self.plot_widget, stretch=2)

        # 2. Text Logs
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("font-family: monospace; background-color: #0d1117; color: #c9d1d9;")
        visuals_row.addWidget(self.log_console, stretch=1)
        
        root.addLayout(visuals_row, stretch=1)

        # ── Progress ──────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

        # ── Bottom Buttons ────────────────────────────────────────────
        btn_bar = QHBoxLayout()

        if self._on_back_callback:
            self.btn_back = QPushButton("←  Back to Model")
            self.btn_back.setMinimumHeight(40)
            self.btn_back.clicked.connect(self._on_back_callback)
            btn_bar.addWidget(self.btn_back)

        btn_bar.addStretch()

        self.btn_export = QPushButton("📦  Export ONNX")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_onnx)
        btn_bar.addWidget(self.btn_export)

        self.btn_stop = QPushButton("🛑  Stop")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        btn_bar.addWidget(self.btn_stop)

        self.btn_train = QPushButton("▶  Start Training")
        self.btn_train.setProperty("class", "primary")
        self.btn_train.setMinimumHeight(40)
        self.btn_train.setMinimumWidth(200)
        self.btn_train.clicked.connect(self._start_training)
        btn_bar.addWidget(self.btn_train)

        root.addLayout(btn_bar)

    def _build_hyperparams_panel(self) -> QGroupBox:
        group = QGroupBox("Hyperparameters")
        lay = QVBoxLayout(group)

        row_lr = QHBoxLayout()
        row_lr.addWidget(QLabel("Learning Rate:"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(4)
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setValue(self.state.hyperparams.get("lr", 0.001))
        row_lr.addWidget(self.spin_lr)
        lay.addLayout(row_lr)

        row_epoch = QHBoxLayout()
        row_epoch.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(self.state.hyperparams.get("epochs", 50))
        row_epoch.addWidget(self.spin_epochs)
        lay.addLayout(row_epoch)

        row_bs = QHBoxLayout()
        row_bs.addWidget(QLabel("Batch Size:"))
        self.spin_bs = QSpinBox()
        self.spin_bs.setRange(1, 1024)
        self.spin_bs.setValue(self.state.hyperparams.get("batch_size", 32))
        row_bs.addWidget(self.spin_bs)
        lay.addLayout(row_bs)

        lay.addStretch()
        return group

    def refresh_ui(self) -> None:
        """Refresh hyperparameter fields from the current ProjectState."""
        self.spin_lr.setValue(self.state.hyperparams.get("lr", 0.001))
        self.spin_epochs.setValue(self.state.hyperparams.get("epochs", 50))
        self.spin_bs.setValue(self.state.hyperparams.get("batch_size", 32))
        
        # Reset progress and logs
        self.progress_bar.setValue(0)
        self.log_console.clear()
        self.btn_export.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_train.setEnabled(True)

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

    def _setup_resource_monitor(self) -> None:
        self.res_timer = QTimer(self)
        self.res_timer.timeout.connect(self._update_resources)
        self.res_timer.start(1000) # 1 sec

    def _update_resources(self) -> None:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        
        vram_str = "N/A"
        if torch.cuda.is_available():
            # Allocated memory in MB
            mem = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_str = f"{mem:.1f} MB"

        self.lbl_resources.setText(f"CPU: {cpu}% | RAM: {ram}% | VRAM: {vram_str}")

    def _start_training(self) -> None:
        # Sync state
        self.state.hyperparams["lr"] = self.spin_lr.value()
        self.state.hyperparams["epochs"] = self.spin_epochs.value()
        self.state.hyperparams["batch_size"] = self.spin_bs.value()
        self.state.device = self.combo_device.currentData()

        # UI toggles
        self.btn_train.setEnabled(False)
        if self._on_back_callback:
            self.btn_back.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_console.clear()

        # Reset plots
        self.plot_epochs.clear()
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_line.setData([], [])
        self.val_line.setData([], [])

        # Start background worker
        self.worker = TrainingWorker(self.state)
        self.worker.log_message.connect(self._append_log)
        self.worker.batch_progress.connect(self._update_progress)
        self.worker.epoch_finished.connect(self._on_epoch)
        self.worker.training_finished.connect(self._on_finished)
        
        self.worker.start()

    def _stop_training(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._append_log("Stop strictly requested. Waiting for current batch to finish...")

    def _append_log(self, text: str) -> None:
        self.log_console.append(text)
        # Scroll to bottom
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / max(1, total))
        self.progress_bar.setValue(pct)

    def _on_epoch(self, epoch: int, t_loss: float, v_loss: float) -> None:
        self.plot_epochs.append(epoch)
        self.train_losses.append(t_loss)
        self.val_losses.append(v_loss)
        
        self.train_line.setData(self.plot_epochs, self.train_losses)
        self.val_line.setData(self.plot_epochs, self.val_losses)

    def _on_finished(self, success: bool, msg: str) -> None:
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._on_back_callback:
            self.btn_back.setEnabled(True)
            
        if success:
            self.btn_export.setEnabled(True) # Ready for export
            QMessageBox.information(self, "Training Complete", msg)
        else:
            QMessageBox.critical(self, "Training Error", f"Training Failed:\n{msg}")

    def _export_onnx(self) -> None:
        """Handler for exporting the trained model to ONNX."""
        if not self.state.model or self.state.dummy_tensor is None:
            QMessageBox.warning(self, "Export Failed", "Model or dummy tensor is missing.")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ONNX Model", "", "ONNX Models (*.onnx);;All Files (*)"
        )
        if not path:
            return
            
        try:
            success, msg = export_to_onnx(self.state.model, self.state.dummy_tensor, path)
            if success:
                QMessageBox.information(self, "Export Success", msg)
            else:
                QMessageBox.critical(self, "Export Failed", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Unexpected error:\n{exc}")

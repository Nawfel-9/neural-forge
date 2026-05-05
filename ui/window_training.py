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
    QTabWidget,
    QSplitter,
)

from utils.project_state import ProjectState
from workers.training_worker import TrainingWorker
from backend.exporter import export_to_onnx
from backend.training_config import get_losses_for, get_all_optimizers
from ui.plot_panel import PlotPanel
from ui.monitor_panel import MonitorPanel


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

        self._init_window()
        self._build_ui()
        self.monitor_panel.start()

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

        self.monitor_panel = MonitorPanel()
        header_row.addWidget(self.monitor_panel)
        root.addLayout(header_row)

        # ── Config Panel ──────────────────────────────────────────────
        config_row = QHBoxLayout()
        config_row.addWidget(self._build_hyperparams_panel())
        config_row.addWidget(self._build_training_config_panel())
        config_row.addWidget(self._build_hardware_panel())
        root.addLayout(config_row)

        # ── Visuals & Logs ────────────────────────────────────────────
        visuals_row = QHBoxLayout()
        
        # 1. Plot Panel
        is_classification = self.state.problem_type == "classification"
        self.plot_panel = PlotPanel(is_classification=is_classification)
        visuals_row.addWidget(self.plot_panel, stretch=2)

        # 2. Tabs for Logs & Model Summary
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

        self.btn_reset = QPushButton("🔄 Re-initialize Weights")
        self.btn_reset.setMinimumHeight(40)
        self.btn_reset.clicked.connect(self._reset_weights)
        btn_bar.addWidget(self.btn_reset)

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

    def _build_training_config_panel(self) -> QGroupBox:
        """Build the Loss Function + Optimizer selection panel."""
        group = QGroupBox("Training Configuration")
        lay = QVBoxLayout(group)

        # ── Loss function ────────────────────────────────────────────
        lay.addWidget(QLabel("Loss Function:"))
        self.combo_loss = QComboBox()
        self._populate_loss_combo()
        lay.addWidget(self.combo_loss)

        # ── Optimizer ────────────────────────────────────────────────
        lay.addSpacing(6)
        lay.addWidget(QLabel("Optimizer:"))
        self.combo_optimizer = QComboBox()
        self.combo_optimizer.addItems(get_all_optimizers())
        # Pre-select current state value
        idx = self.combo_optimizer.findText(self.state.optimizer_name)
        if idx >= 0:
            self.combo_optimizer.setCurrentIndex(idx)
        lay.addWidget(self.combo_optimizer)

        lay.addStretch()
        return group

    def _populate_loss_combo(self) -> None:
        """
        (Re-)populate the loss combo filtered by the current problem type.

        Called at construction time and again in :meth:`refresh_ui` in case
        the user navigated back to Window 1 and changed the problem type.
        """
        problem_type = self.state.problem_type
        losses = get_losses_for(problem_type)

        self.combo_loss.blockSignals(True)
        self.combo_loss.clear()
        self.combo_loss.addItems(losses)
        self.combo_loss.blockSignals(False)

        # Restore previous selection when possible
        idx = self.combo_loss.findText(self.state.loss_fn_name)
        self.combo_loss.setCurrentIndex(idx if idx >= 0 else 0)

    def refresh_ui(self) -> None:
        """Refresh all fields from the current ProjectState.

        Called by ``PipelineController`` every time the user navigates
        forward to Window 3, so changes made in earlier windows (e.g. a
        different problem type selected in Window 1) are always reflected.
        """
        self.spin_lr.setValue(self.state.hyperparams.get("lr", 0.001))
        self.spin_epochs.setValue(self.state.hyperparams.get("epochs", 50))
        self.spin_bs.setValue(self.state.hyperparams.get("batch_size", 32))

        # Re-populate loss combo — problem_type may have changed
        self._populate_loss_combo()

        # Update Plot Panel for potential problem type change
        self.plot_panel.set_is_classification(self.state.problem_type == "classification")
        self.plot_panel.clear()

        # Update Model Summary
        if self.state.model:
            total_params = sum(p.numel() for p in self.state.model.parameters() if p.requires_grad)
            summary_text = f"Total Trainable Parameters: {total_params:,}\n\nArchitecture:\n{self.state.model}"
            self.model_summary.setText(summary_text)
        else:
            self.model_summary.setText("No model available.")

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

    def _reset_weights(self) -> None:
        """Rebuild the model to reset its weights to random initialization."""
        from backend.model_builder import build_and_validate
        if not self.state.blueprint or self.state.input_features() == 0:
            QMessageBox.warning(self, "Reset Failed", "Blueprint or data is missing.")
            return

        model, dummy_input, success, msg = build_and_validate(
            self.state.blueprint, self.state.input_features()
        )
        
        if success:
            self.state.model = model
            self.state.dummy_tensor = dummy_input
            
            # Update Model Summary
            total_params = sum(p.numel() for p in self.state.model.parameters() if p.requires_grad)
            summary_text = f"Total Trainable Parameters: {total_params:,}\n\nArchitecture:\n{self.state.model}"
            self.model_summary.setText(summary_text)
            
            # Notify user
            self.log_console.append("\n" + "="*50)
            self.log_console.append("🔄 Model weights have been re-initialized to random values.")
            self.log_console.append("="*50 + "\n")
            
            # Reset UI progress
            self.progress_bar.setValue(0)
            self.plot_panel.clear()
        else:
            QMessageBox.warning(self, "Reset Failed", f"Could not rebuild model:\n{msg}")

    # ── Removed old _setup_resource_monitor and _update_resources methods ──

    def _start_training(self) -> None:
        # Sync hyperparams
        self.state.hyperparams["lr"] = self.spin_lr.value()
        self.state.hyperparams["epochs"] = self.spin_epochs.value()
        self.state.hyperparams["batch_size"] = self.spin_bs.value()
        self.state.device = self.combo_device.currentData()

        # Sync loss / optimizer selection
        self.state.loss_fn_name = self.combo_loss.currentText()
        self.state.optimizer_name = self.combo_optimizer.currentText()

        # UI toggles
        self.btn_train.setEnabled(False)
        self.btn_reset.setEnabled(False)
        if self._on_back_callback:
            self.btn_back.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_console.clear()

        # Reset plots
        self.plot_panel.clear()

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

    def _on_epoch(self, epoch: int, t_loss: float, v_loss: float, metrics: dict | None = None) -> None:
        self.plot_panel.add_data(epoch, t_loss, v_loss, metrics)

    def _on_finished(self, success: bool, msg: str) -> None:
        self.btn_train.setEnabled(True)
        self.btn_reset.setEnabled(True)
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

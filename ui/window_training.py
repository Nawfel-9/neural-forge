from __future__ import annotations

import torch
from PyQt6.QtCore import Qt
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

        config_row = QHBoxLayout()
        config_row.addWidget(self._build_hyperparams_panel())
        config_row.addWidget(self._build_training_config_panel())
        config_row.addWidget(self._build_splitting_panel())
        config_row.addWidget(self._build_hardware_panel())
        root.addLayout(config_row)

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

    def refresh_ui(self) -> None:
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

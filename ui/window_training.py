from __future__ import annotations
import os
import psutil
import torch
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QPainterPath, QPalette
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, 
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QTextEdit, 
    QVBoxLayout, QWidget, QScrollArea, QFrame, QFormLayout
)

from utils.project_state import ProjectState
from workers.training_worker import TrainingWorker

class SystemResourcesWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(320, 190)
        self.cpu = 0
        self.ram = 0
        self.vram_pct = 0
        self.ram_used = 0
        self.ram_total = 0
        self.cpu_freq = 0.0
        
        self.history = {
            "CPU": [0]*60,
            "GPU": [0]*60,
            "RAM": [0]*60
        }
        self.selected = "CPU"
        self.hitboxes = {}
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def update_stats(self, cpu, ram_pct, ram_used_gb, ram_total_gb, vram_pct, cpu_freq):
        self.cpu = cpu
        self.ram = ram_pct
        self.vram_pct = vram_pct
        self.ram_used = ram_used_gb
        self.ram_total = ram_total_gb
        self.cpu_freq = cpu_freq
        
        self.history["CPU"].append(cpu)
        self.history["GPU"].append(vram_pct)
        self.history["RAM"].append(ram_pct)
        
        for k in self.history:
            if len(self.history[k]) > 60:
                self.history[k].pop(0)
                
        self.update()

    def mousePressEvent(self, event):
        if hasattr(event, "position"):
            pos = event.position()
        else:
            pos = event.pos()
            
        from PyQt6.QtCore import QPointF
        if not isinstance(pos, QPointF):
            pos = QPointF(pos.x(), pos.y())
            
        for name, rect in self.hitboxes.items():
            if rect.contains(pos):
                self.selected = name
                self.update()
                break
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        from PyQt6.QtGui import QPalette
        palette = self.palette()
        bg_space = palette.color(QPalette.ColorRole.Window)
        bg_glass = palette.color(QPalette.ColorRole.AlternateBase)
        text_main = palette.color(QPalette.ColorRole.WindowText)
        text_muted = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText)
        accent = palette.color(QPalette.ColorRole.Highlight)
        
        # Draw background
        rect = self.rect()
        painter.setBrush(bg_space)
        painter.setPen(QPen(bg_glass, 1))
        painter.drawRoundedRect(rect.adjusted(1,1,-1,-1), 8, 8)
        
        # Draw Title
        painter.setPen(text_main)
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(16, 24, " System Resources")
        
        # Left side labels
        font = QFont("Segoe UI", 10)
        painter.setFont(font)
        
        y_start = 55
        spacing = 35
        labels = [
            ("CPU", self.cpu),
            ("GPU", self.vram_pct),
            ("RAM", self.ram)
        ]
        
        self.hitboxes.clear()
        
        # Create a dynamic pill background using text_main with low alpha
        pill_bg = QColor(text_main)
        pill_bg.setAlpha(25)
        
        for i, (name, val) in enumerate(labels):
            y = y_start + i * spacing
            hitbox = QRectF(10, y - 20, 110, 28)
            self.hitboxes[name] = hitbox
            
            if name == self.selected:
                painter.setBrush(pill_bg)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(hitbox, 6, 6)
                painter.setPen(text_main)
            else:
                painter.setPen(text_muted)
                
            painter.drawText(20, y, f"{name}  {int(val)} %")
            
        # Right Side (Large numbers)
        current_val = self.history[self.selected][-1] if self.history[self.selected] else 0
        
        painter.setPen(text_main)
        font = QFont("Segoe UI", 26, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(130, 30, 170, 40), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{int(current_val)} %")
        
        painter.setPen(text_muted)
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        
        subtext = ""
        if self.selected == "CPU":
            subtext = f"{self.cpu_freq:.2f} GHz"
        elif self.selected == "RAM":
            subtext = f"{self.ram_used:.1f}/{self.ram_total:.1f} GB"
        elif self.selected == "GPU":
            subtext = "VRAM Usage"
            
        painter.drawText(QRectF(130, 70, 170, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, subtext)
        
        # Line Graph
        graph_x = 140
        graph_y = 120
        graph_w = 160
        graph_h = 40
        
        painter.setPen(text_muted)
        painter.drawText(QRectF(graph_x, graph_y + graph_h + 5, graph_w, 15), Qt.AlignmentFlag.AlignLeft, "60 Seconds")
        painter.drawText(QRectF(graph_x, graph_y + graph_h + 5, graph_w, 15), Qt.AlignmentFlag.AlignRight, "0")
        
        data = self.history[self.selected]
        if not data:
            return
            
        path = QPainterPath()
        fill_path = QPainterPath()
        
        dx = graph_w / max(1, len(data) - 1)
        
        fill_path.moveTo(graph_x, graph_y + graph_h)
        
        for i, val in enumerate(data):
            x = graph_x + i * dx
            y = graph_y + graph_h - (val / 100.0) * graph_h
            if i == 0:
                path.moveTo(x, y)
                fill_path.lineTo(x, y)
            else:
                path.lineTo(x, y)
                fill_path.lineTo(x, y)
                
        fill_path.lineTo(graph_x + graph_w, graph_y + graph_h)
        fill_path.closeSubpath()
        
        grad = QLinearGradient(graph_x, graph_y, graph_x, graph_y + graph_h)
        
        color_top = QColor(accent)
        color_top.setAlpha(180)
        color_bottom = QColor(accent)
        color_bottom.setAlpha(20)
        
        grad.setColorAt(0, color_top)
        grad.setColorAt(1, color_bottom)
        
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)
        
        line_color = QColor(accent).lighter(120)
        painter.setPen(QPen(line_color, 2))
        painter.drawPath(path)

class TrainingWindow(QWidget):
    """
    Training & Evaluation Dashboard.
    """
    def __init__(self, project_state: ProjectState, on_back=None, parent=None) -> None:
        super().__init__(parent)
        self.state = project_state
        self._on_back_callback = on_back
        self.worker: TrainingWorker | None = None

        self.plot_epochs = []
        self.train_losses = []
        self.val_losses = []

        self._build_ui()
        self._setup_resource_monitor()

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

        # ── Header & Monitor ──
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

        self.res_widget = SystemResourcesWidget()
        header_row.addWidget(self.res_widget)

        root.addLayout(header_row)

        # ── Configuration Panel ──
        config_row = QHBoxLayout()
        config_row.addWidget(self._build_hyperparams_panel())
        config_row.addWidget(self._build_splitting_panel())
        config_row.addWidget(self._build_hardware_panel())
        root.addLayout(config_row)


        # ── Visuals & Logs ──
        visuals_row = QHBoxLayout()
        
        # PyQtGraph Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(350)
        self.plot_widget.setBackground('transparent')
        self.plot_widget.setTitle("Loss Curve", color="#0EA5E9")
        self.plot_widget.setLabel('left', 'Loss')
        self.plot_widget.setLabel('bottom', 'Epoch')
        self.train_line = self.plot_widget.plot(pen=pg.mkPen(color='#10B981', width=2))
        self.val_line = self.plot_widget.plot(pen=pg.mkPen(color='#0EA5E9', width=2))
        
        plot_container = QVBoxLayout()
        legend_lbl = QLabel('<span style="color:#10B981; font-weight:bold; font-size:14px;">● Train Loss</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#0EA5E9; font-weight:bold; font-size:14px;">● Val Loss</span>')
        legend_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_container.addWidget(legend_lbl)
        plot_container.addWidget(self.plot_widget)
        
        visuals_row.addLayout(plot_container, stretch=2)

        # Text Logs
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("font-family: monospace; font-size: 9pt;")
        visuals_row.addWidget(self.log_console, stretch=1)
        
        root.addLayout(visuals_row, stretch=1)

        # ── Metrics Panel (Hidden until eval finishes) ──
        self.metrics_group = QGroupBox("Evaluation Metrics")
        self.metrics_group.setVisible(False)
        self.metrics_layout = QHBoxLayout(self.metrics_group)
        root.addWidget(self.metrics_group)

        # ── Progress ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

        # ── Bottom Buttons ──
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()

        self.btn_stop = QPushButton("🛑 Stop")
        self.btn_stop.setMinimumSize(150, 44)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        btn_bar.addWidget(self.btn_stop)

        self.btn_train = QPushButton("▶ Start Training")
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
        
        self._on_split_method_changed(self.state.split_config.get("method", "percentage"))
        return group

    def _on_split_method_changed(self, method: str) -> None:
        if method == "percentage":
            self.spin_split_ratio.setVisible(True)
            lbl = self.split_lay.labelForField(self.spin_split_ratio)
            if lbl: lbl.setVisible(True)
            
            self.spin_kfold.setVisible(False)
            lbl = self.split_lay.labelForField(self.spin_kfold)
            if lbl: lbl.setVisible(False)
        else:
            self.spin_split_ratio.setVisible(False)
            lbl = self.split_lay.labelForField(self.spin_split_ratio)
            if lbl: lbl.setVisible(False)
            
            self.spin_kfold.setVisible(True)
            lbl = self.split_lay.labelForField(self.spin_kfold)
            if lbl: lbl.setVisible(True)

    def refresh_ui(self) -> None:
        self.spin_lr.setValue(self.state.hyperparams.get("lr", 0.001))
        self.spin_epochs.setValue(self.state.hyperparams.get("epochs", 50))
        self.spin_bs.setValue(self.state.hyperparams.get("batch_size", 32))
        
        sc = self.state.split_config
        idx = self.combo_split_method.findText(sc.get("method", "percentage"))
        if idx >= 0: self.combo_split_method.setCurrentIndex(idx)
        if "ratio" in sc: self.spin_split_ratio.setValue(sc["ratio"])
        if "k" in sc: self.spin_kfold.setValue(sc["k"])
        idx_res = self.combo_resample.findText(sc.get("resample", "none"))
        if idx_res >= 0: self.combo_resample.setCurrentIndex(idx_res)
        
        # Hide Imbalanced Resampling for Regression tasks
        is_classification = self.state.problem_type != "regression"
        self.combo_resample.setVisible(is_classification)
        lbl = self.split_lay.labelForField(self.combo_resample)
        if lbl: lbl.setVisible(is_classification)
        
        self.progress_bar.setValue(0)
        self.btn_stop.setEnabled(False)
        self.btn_train.setEnabled(True)

    def _build_hardware_panel(self) -> QGroupBox:
        group = QGroupBox("Hardware Selection")
        lay = QVBoxLayout(group)
        self.combo_device = QComboBox()
        self.combo_device.addItem("CPU", "cpu")
        if torch.cuda.is_available(): self.combo_device.addItem("CUDA (NVIDIA GPU)", "cuda")
        if torch.backends.mps.is_available(): self.combo_device.addItem("MPS (Apple Silicon)", "mps")
        lay.addWidget(QLabel("Select Compute Device:"))
        lay.addWidget(self.combo_device)
        lay.addStretch()
        return group

    def _setup_resource_monitor(self) -> None:
        self.res_timer = QTimer(self)
        self.res_timer.timeout.connect(self._update_resources)
        self.res_timer.start(1000) 

    def _update_resources(self) -> None:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        ram_pct = mem.percent
        ram_used = mem.used / (1024 ** 3)
        ram_total = mem.total / (1024 ** 3)
        
        cpu_freq = psutil.cpu_freq().current / 1000.0 if psutil.cpu_freq() else 0.0
        
        vram_pct = 0.0
        if torch.cuda.is_available():
            try:
                vram_alloc = torch.cuda.memory_allocated()
                vram_total = torch.cuda.get_device_properties(0).total_memory
                vram_pct = (vram_alloc / max(1, vram_total)) * 100.0
            except:
                pass
                
        self.res_widget.update_stats(cpu, ram_pct, ram_used, ram_total, vram_pct, cpu_freq)

    def _start_training(self) -> None:
        self.state.hyperparams["lr"] = self.spin_lr.value()
        self.state.hyperparams["epochs"] = self.spin_epochs.value()
        self.state.hyperparams["batch_size"] = self.spin_bs.value()
        self.state.device = self.combo_device.currentData()

        method = self.combo_split_method.currentText()
        resample = self.combo_resample.currentText()
        if method == "percentage":
            self.state.split_config = {"method": "percentage", "ratio": self.spin_split_ratio.value(), "resample": resample}
        else:
            self.state.split_config = {"method": "kfold", "k": self.spin_kfold.value(), "resample": resample}

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_console.clear()
        
        # Hide metrics on restart
        self.metrics_group.setVisible(False)
        # Clear old metrics labels
        while self.metrics_layout.count():
            child = self.metrics_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.plot_epochs.clear()
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_line.setData([], [])
        self.val_line.setData([], [])

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

    def _on_epoch(self, epoch: int, t_loss: float, v_loss: float) -> None:
        self.plot_epochs.append(epoch)
        self.train_losses.append(t_loss)
        self.val_losses.append(v_loss)
        self.train_line.setData(self.plot_epochs, self.train_losses)
        self.val_line.setData(self.plot_epochs, self.val_losses)

    def _on_evaluation(self, metrics: dict) -> None:
        for name, value in metrics.items():
            metric_card = QWidget()
            mlay = QVBoxLayout(metric_card)
            
            val_lbl = QLabel(f"{value:.4f}")
            val_lbl.setStyleSheet("font-size: 16pt; font-weight: bold; color: #10B981;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            name_lbl = QLabel(name.upper())
            name_lbl.setStyleSheet("font-size: 8pt; color: #64748B; font-weight: 600;")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            mlay.addWidget(val_lbl)
            mlay.addWidget(name_lbl)
            self.metrics_layout.addWidget(metric_card)
            
        self.metrics_group.setVisible(True)

    def _on_finished(self, success: bool, msg: str) -> None:
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if success:
            QMessageBox.information(self, "Training Complete", msg)
        else:
            QMessageBox.critical(self, "Training Error", f"Training Failed:\n{msg}")

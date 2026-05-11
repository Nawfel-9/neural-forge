"""
plot_panel.py
=============
PyQtGraph-based widget for visualizing training curves.
Displays Loss (Train vs Val) and optionally Metrics (Acc/F1) for classification.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class PlotPanel(QWidget):
    """
    A reusable widget containing real-time plots for training metrics.
    If `is_classification` is True, it shows a secondary plot for Accuracy/F1.
    """
    def __init__(self, is_classification: bool = False, parent=None):
        super().__init__(parent)
        self.is_classification = is_classification

        # Data stores
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_f1s = []

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # 1. Loss Plot
        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.setLabel('bottom', 'Epoch')
        self.loss_plot.addLegend()
        self.train_loss_line = self.loss_plot.plot(pen=pg.mkPen(color='#3fb950', width=2.5), name="Train Loss")
        self.val_loss_line = self.loss_plot.plot(pen=pg.mkPen(color='#58a6ff', width=2.5), name="Val Loss")
        layout.addWidget(self.loss_plot)

        # 2. Metrics Plot
        self.metric_plot = pg.PlotWidget()
        self.metric_plot.setLabel('left', 'Score')
        self.metric_plot.setLabel('bottom', 'Epoch')
        self.metric_plot.setYRange(0, 1.0)
        self.metric_plot.addLegend()

        self.val_acc_line = self.metric_plot.plot(pen=pg.mkPen(color='#a371f7', width=2.5), name="Val Acc")
        self.val_f1_line = self.metric_plot.plot(pen=pg.mkPen(color='#f0883e', width=2.5), name="Val F1")

        layout.addWidget(self.metric_plot)
        self.metric_plot.setVisible(self.is_classification)

        # Apply initial theme (default to dark if not set)
        self.apply_theme(True)

    def apply_theme(self, is_dark: bool) -> None:
        """Update plot colors to match the current theme."""
        bg_color = "#0B0F17" if is_dark else "#F8FAFC"
        text_color = "#F1F5F9" if is_dark else "#0F172A"
        grid_color = "rgba(255, 255, 255, 0.1)" if is_dark else "rgba(0, 0, 0, 0.1)"

        for plot in [self.loss_plot, self.metric_plot]:
            plot.setBackground(bg_color)
            plot.getAxis('left').setPen(text_color)
            plot.getAxis('left').setTextPen(text_color)
            plot.getAxis('bottom').setPen(text_color)
            plot.getAxis('bottom').setTextPen(text_color)
            
            title_text = "Loss Curve" if plot == self.loss_plot else "Validation Metrics"
            plot.setTitle(title_text, color=text_color, size="12pt")

            # Update legend text color
            legend = plot.plotItem.legend
            if legend:
                for item in legend.items:
                    for label in item:
                        if isinstance(label, pg.LabelItem):
                            label.setAttr('color', text_color)

    def set_is_classification(self, is_classification: bool) -> None:
        self.is_classification = is_classification
        self.metric_plot.setVisible(is_classification)

    def add_data(self, epoch: int, t_loss: float, v_loss: float, metrics: dict | None = None) -> None:
        """Append new data points and update the plots."""
        self.epochs.append(epoch)
        self.train_losses.append(t_loss)
        self.val_losses.append(v_loss)

        self.train_loss_line.setData(self.epochs, self.train_losses)
        self.val_loss_line.setData(self.epochs, self.val_losses)

        if self.is_classification and metrics:
            self.val_accs.append(metrics.get("val_acc", 0.0))
            self.val_f1s.append(metrics.get("val_f1", 0.0))

            self.val_acc_line.setData(self.epochs, self.val_accs)
            self.val_f1_line.setData(self.epochs, self.val_f1s)

    def clear(self) -> None:
        """Reset all data stores and clear the plot lines."""
        self.epochs.clear()
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_loss_line.setData([], [])
        self.val_loss_line.setData([], [])

        self.train_accs.clear()
        self.val_accs.clear()
        self.val_f1s.clear()
        self.val_acc_line.setData([], [])
        self.val_f1_line.setData([], [])

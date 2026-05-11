"""
ui/window_training_dev.py
=========================
Developer Mode hardware dashboard widgets for the shared Training window.

The actual polling runs in backend.hardware_monitor.HardwareMonitor. This file
only renders the latest HardwareStats object.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout

from backend.hardware_monitor import HardwareStats


class GaugeWidget(QFrame):
    """Compact numeric gauge with a colored usage bar."""

    DANGER_COLOR = "#EF4444"
    WARNING_COLOR = "#F59E0B"
    NORMAL_COLOR = "#00A3FF"

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
        self.max_value = max_value
        self.warn = warn
        self.danger = danger
        self.unit = unit

        self.setObjectName("GaugeWidget")
        self.setFixedHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            """
            QFrame#GaugeWidget {
                background: rgba(10, 18, 35, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 8px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        title_label = QLabel(title.upper())
        title_label.setStyleSheet("color: #64748B; font-size: 8pt; font-weight: 800;")
        layout.addWidget(title_label)

        self.value_label = QLabel("-")
        self.value_label.setStyleSheet(
            "color: #F1F5F9; font-size: 22pt; font-weight: 900; "
            "font-family: 'Courier New', monospace;"
        )
        layout.addWidget(self.value_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self.bar.setStyleSheet(self._bar_style(self.NORMAL_COLOR))
        layout.addWidget(self.bar)

    def _bar_style(self, color: str) -> str:
        return f"""
            QProgressBar {{
                background: rgba(255, 255, 255, 0.06);
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 2px;
            }}
        """

    def set_max_value(self, max_value: float) -> None:
        self.max_value = max(float(max_value), 1.0)

    def update_value(self, value: float) -> None:
        value = max(float(value), 0.0)
        percent = min(value / max(self.max_value, 1.0), 1.0)

        if value >= self.danger:
            color = self.DANGER_COLOR
        elif value >= self.warn:
            color = self.WARNING_COLOR
        else:
            color = self.NORMAL_COLOR

        self.value_label.setText(f"{value:.0f}{self.unit}")
        self.value_label.setStyleSheet(
            f"color: {color}; font-size: 22pt; font-weight: 900; "
            "font-family: 'Courier New', monospace;"
        )
        self.bar.setValue(int(percent * 1000))
        self.bar.setStyleSheet(self._bar_style(color))


class HardwareDashboard(QFrame):
    """Developer Mode dashboard for GPU, CPU, RAM, and VRAM usage."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.vram = GaugeWidget("VRAM", " MB", 1, warn=0.75, danger=0.9)
        self.gpu = GaugeWidget("GPU", "%", 100, warn=75, danger=90)
        self.gpu_temp = GaugeWidget("GPU Temp", " C", 110, warn=75, danger=90)
        self.cpu = GaugeWidget("CPU", "%", 100, warn=75, danger=90)
        self.ram = GaugeWidget("RAM", "%", 100, warn=75, danger=90)

        for gauge in (self.vram, self.gpu, self.gpu_temp, self.cpu, self.ram):
            layout.addWidget(gauge)

    @pyqtSlot(object)
    def on_stats(self, stats: HardwareStats) -> None:
        total_vram = max(stats.gpu_vram_total, 1.0)
        self.vram.set_max_value(total_vram)
        self.vram.warn = total_vram * 0.75
        self.vram.danger = total_vram * 0.9
        self.vram.update_value(stats.gpu_vram_used)
        self.gpu.update_value(stats.gpu_usage)
        self.gpu_temp.update_value(stats.gpu_temp)
        self.cpu.update_value(stats.cpu_usage)
        self.ram.update_value(stats.ram_percent)

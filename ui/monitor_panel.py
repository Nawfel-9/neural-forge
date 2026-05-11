"""
monitor_panel.py
================
Hardware resource monitoring widget (CPU, RAM, VRAM).
Uses psutil and torch to poll system usage.
Now emits stats_updated signal with GPU temp for the dev trainer thermal guard.
"""

from __future__ import annotations

import psutil
import torch
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QLabel

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:
    _NVML_OK = False
    _NVML_HANDLE = None


class _StatsPayload:
    """Lightweight stats object emitted with stats_updated signal."""
    def __init__(self, gpu_temp: float = 0.0):
        self.gpu_temp = gpu_temp


class MonitorPanel(QLabel):
    """
    A QLabel subclass that automatically polls system resources
    and displays them in a monospace font.

    Emits:
        stats_updated(_StatsPayload) — once per poll tick, carries gpu_temp
        so the dev trainer thread can react to thermal thresholds.
    """

    stats_updated = pyqtSignal(object)   # _StatsPayload

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "StatText")
        self.setText("CPU: 0% | RAM: 0% | VRAM: N/A")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_resources)

    def start(self, interval_ms: int = 1000) -> None:
        """Start polling resources every `interval_ms` milliseconds."""
        self._timer.start(interval_ms)
        self._update_resources()  # initial tick

    def stop(self) -> None:
        """Stop polling."""
        self._timer.stop()

    def _update_resources(self) -> None:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent

        vram_str = "N/A"
        gpu_temp = 0.0

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_str = f"{mem:.1f} MB"

        if _NVML_OK and _NVML_HANDLE is not None:
            try:
                gpu_temp = float(pynvml.nvmlDeviceGetTemperature(
                    _NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU
                ))
                vram_str += f" | GPU: {gpu_temp:.0f}°C"
            except Exception:
                pass

        self.setText(f"CPU: {cpu}% | RAM: {ram}% | VRAM: {vram_str}")
        self.stats_updated.emit(_StatsPayload(gpu_temp=gpu_temp))

"""
monitor_panel.py
================
Hardware resource monitoring widget (CPU, RAM, VRAM).
Uses psutil and torch to poll system usage.
"""

from __future__ import annotations

import psutil
import torch
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel


class MonitorPanel(QLabel):
    """
    A QLabel subclass that automatically polls system resources
    and displays them in a monospace font.
    """
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
        if torch.cuda.is_available():
            # Allocated memory in MB
            mem = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_str = f"{mem:.1f} MB"

        self.setText(f"CPU: {cpu}% | RAM: {ram}% | VRAM: {vram_str}")

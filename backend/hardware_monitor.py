"""
backend/hardware_monitor.py
===========================
A QThread that polls GPU (pynvml), CPU, and RAM (psutil) stats once per second
and emits them as a typed dataclass via a Qt signal.
"""

from __future__ import annotations
from dataclasses import dataclass
import time

from PyQt6.QtCore import QThread, pyqtSignal

try:
    import pynvml
    _NVML_OK = True
except ImportError:
    _NVML_OK = False

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


@dataclass
class HardwareStats:
    # GPU
    gpu_usage: float = 0.0        # %
    gpu_temp: float = 0.0         # °C
    gpu_vram_used: float = 0.0    # MB
    gpu_vram_total: float = 0.0   # MB
    gpu_available: bool = False

    # CPU
    cpu_usage: float = 0.0        # %
    cpu_temp: float = 0.0         # °C  (may be 0 if not readable)

    # RAM
    ram_used: float = 0.0         # MB
    ram_total: float = 0.0        # MB
    ram_percent: float = 0.0      # %

    @property
    def gpu_vram_percent(self) -> float:
        if self.gpu_vram_total == 0:
            return 0.0
        return (self.gpu_vram_used / self.gpu_vram_total) * 100.0


class HardwareMonitor(QThread):
    """
    Polls hardware stats every `interval` seconds and emits `stats_updated`.
    Designed to run for the lifetime of the training session.
    """
    stats_updated = pyqtSignal(object)   # HardwareStats

    def __init__(self, interval: float = 1.0, parent=None):
        super().__init__(parent)
        self.interval = interval
        self._running = False
        self._nvml_handle = None
        self._nvml_initialised = False

    # ------------------------------------------------------------------
    # NVML lifecycle
    # ------------------------------------------------------------------
    def _init_nvml(self):
        if not _NVML_OK:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_initialised = True
        except Exception:
            self._nvml_initialised = False

    def _shutdown_nvml(self):
        if self._nvml_initialised:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml_initialised = False

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _sample_gpu(self, stats: HardwareStats):
        if not self._nvml_initialised:
            return
        try:
            util  = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            temp  = pynvml.nvmlDeviceGetTemperature(
                        self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
            mem   = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)

            stats.gpu_available  = True
            stats.gpu_usage      = float(util.gpu)
            stats.gpu_temp       = float(temp)
            stats.gpu_vram_used  = mem.used  / 1024**2
            stats.gpu_vram_total = mem.total / 1024**2
        except Exception:
            stats.gpu_available = False

    def _sample_cpu_ram(self, stats: HardwareStats):
        if not _PSUTIL_OK:
            return
        try:
            stats.cpu_usage = psutil.cpu_percent(interval=None)

            # CPU temperature — platform-dependent, graceful fallback
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Prefer 'coretemp' on Linux / 'cpu_thermal' on macOS
                    for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                        if key in temps:
                            entries = temps[key]
                            if entries:
                                stats.cpu_temp = entries[0].current
                                break
            except (AttributeError, NotImplementedError):
                stats.cpu_temp = 0.0

            vm = psutil.virtual_memory()
            stats.ram_used    = vm.used    / 1024**2
            stats.ram_total   = vm.total   / 1024**2
            stats.ram_percent = vm.percent
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------
    def run(self):
        self._running = True
        self._init_nvml()

        # Warm up cpu_percent (first call always returns 0)
        if _PSUTIL_OK:
            psutil.cpu_percent(interval=None)

        while self._running:
            stats = HardwareStats()
            self._sample_gpu(stats)
            self._sample_cpu_ram(stats)
            self.stats_updated.emit(stats)
            time.sleep(self.interval)

        self._shutdown_nvml()

    def stop(self):
        self._running = False
        self.wait()

# Phase 5 Walkthrough: Visualization, Monitoring & Export

## Overview
Phase 5 introduces real-time visual feedback, system resource monitoring, and a bridge to export trained models into universally accessible formats natively within the Training Studio (Window 3).

## Key Components

### 1. Real-Time Loss Curves (`pyqtgraph`)
**Purpose**: To replace static terminal logs with a dynamic visual representation of Training vs. Validation Loss out-of-the-box.
**Mechanics**:
- Replaced the full-window text console with a split view. 
- Integrated `pyqtgraph.PlotWidget`, establishing a dark-themed plot matching the application styling.
- `train_line` and `val_line` are bound to local lists (`self.plot_epochs`, `self.train_losses`, `self.val_losses`).
- The `TrainingWorker.epoch_finished` signal directly triggers `_on_epoch`, pushing the data arrays to the plot renderer for near-zero-latency updates.

### 2. Hardware Resource Monitor (`psutil` & `pynvml`/`torch`)
**Purpose**: Ensure users know if their datasets or models are exceeding subsystem RAM/VRAM capacity.
**Mechanics**:
- A `QTimer` (`self.res_timer`) triggers every 1,000ms.
- Calls `psutil.cpu_percent()` and `psutil.virtual_memory().percent` for CPU and Main Memory bottlenecks.
- Measures allocated PyTorch tensor memory via `torch.cuda.memory_allocated()` falling back to N/A.

### 3. Open Neural Network Exchange (ONNX) Exporter
**Purpose**: Allow models built visually with Neural Forge to be easily consumed by Python APIs, C++, or mobile deployments.
**Mechanics**: 
- Added `backend/exporter.py` implementing `export_to_onnx` which wraps `torch.onnx.export`.
- Utilizes the `dummy_tensor` (input shape) from Phase 3's ghost run dynamically captured in `ProjectState`.
- **Shape Safety**: The system now ensures the trace tensor correctly matches the model's input rather than its output.
- Automatically handles dynamic batch sizes via `dynamic_axes={'input': {0: 'batch_size'}}`.

## Validation
- `tests/test_phase5.py` verifies the PyTorch computational graph -> ONNX file sink behaves predictably across shape matches and mismatches.

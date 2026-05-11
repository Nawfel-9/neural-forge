# Phase 5 Walkthrough: Visualization, Monitoring & Export

## Overview

Phase 5 adds real-time visual feedback, resource monitoring, final metrics display, and deployment exports.

---

## Real-Time Plots

`ui/plot_panel.py` wraps PyQtGraph in a reusable `PlotPanel` widget:

- Always shows training and validation loss.
- Shows validation accuracy and F1 only for classification tasks.
- Maintains local epoch/loss/metric arrays.
- Updates from `TrainingWorker.epoch_finished(epoch, train_loss, val_loss, metrics)`.

`TrainingWindow.refresh_ui()` toggles the metrics plot when the user changes problem type earlier in the pipeline.

---

## Resource Monitor

`ui/monitor_panel.py` is a `QLabel` with an internal `QTimer`.

- Polls CPU with `psutil.cpu_percent()`.
- Polls RAM with `psutil.virtual_memory().percent`.
- Polls CUDA allocated memory with `torch.cuda.memory_allocated()` when CUDA is available.
- Emits `stats_updated` with a small payload that currently includes GPU temperature when optional NVML support is available.

The monitor is intentionally lightweight and lives inside the training header.

---

## Final Evaluation Metrics

After training, `TrainingWorker._evaluate_model(...)` emits `evaluation_finished(metrics)`.

Classification metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC when computable

Regression metrics include:

- MSE
- RMSE
- MAE
- R2 Score

`TrainingWindow` renders these values in the hidden `Evaluation Metrics` panel once results arrive.

---

## Export

`ui/window_export.py` owns deployment outputs:

| Export | Implementation |
|---|---|
| Data pipeline | Calls `DataPipeline.save(path)` |
| PyTorch model | Calls `torch.save(state.model.state_dict(), path)` |
| ONNX model | Calls `backend.exporter.export_to_onnx(state.model, state.dummy_tensor, path)` |
| Synthesis report | Opens `ui.dialog_report.ReportDialog`, which can export a PDF report from project state and final metrics |

The export screen enables buttons only when the relevant state is ready.

`backend/exporter.py` keeps ONNX logic outside the UI and uses the ghost-run input tensor as the trace input. Export uses dynamic batch axes and returns `(success, message)` instead of raising into the UI.

---

## Validation

`tests/test_phase5.py` covers successful export, shape mismatch handling, failure messages, CUDA export when available, and ONNX loadability when the `onnx` package is installed.

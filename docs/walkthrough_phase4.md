# Phase 4 Walkthrough: Multithreading & Hardware Selection

## Overview
Phase 4 implements multithreaded PyTorch training within a PyQt desktop application, decoupled via `ProjectState`.

## Key Components

### 1. `workers.training_worker.TrainingWorker` (QThread)
**Purpose**: Moves the intensive PyTorch training loop off the main UI thread to prevent the application from freezing.
**Mechanics**:
- Inherits from `PyQt6.QtCore.QThread`.
- Reads `ProjectState` to grab the dataset (`DataFrame`), model (`nn.Sequential`), target column, and hyperparameters.
- Defines standard PyTorch objects: `TensorDataset`, `DataLoader`, `nn.CrossEntropyLoss` or `nn.MSELoss`, `optim.Adam`.
- **Automatic Label Encoding**: Classification targets are automatically mapped to `[0, C-1]` to prevent common PyTorch range errors.
- Handles both Percentage Split and K-Fold CV logic inside the `run()` override.
- Emits thread-safe signals:
  - `epoch_finished` (for charting)
  - `batch_progress` (for progress bar)
  - `training_finished` (for success/error dialogs)
  - `log_message` (to dump print statements to UI safely)

### 2. Hardware Selection
**Purpose**: Allows selection of Compute Device natively.
**Mechanics**: 
- `torch.device(self.state.device)` drives computation inside the thread.
- Added a `QComboBox` populated with valid hardware checks: `torch.cuda.is_available()` and `torch.backends.mps.is_available()`.

### 3. `ui.window_training.TrainingWindow` — Training View
**Purpose**: The 3rd window in the No-Code Pipeline.
**UI Elements**:
- **Hyperparameter Tweaking**: Learning Rate (`QDoubleSpinBox`), Epochs (`QSpinBox`), Batch Size (`QSpinBox`).
- **Hardware Selection**: Dropdown to select CPU, CUDA, MPS.
- **Log Console**: A read-only `QTextEdit` displaying raw epochs and actions.
- **Progress Bar**: Binds to the `batch_progress` signal.
- **Start/Stop Controls**: Starts the worker, or safely requests early abort via bounded boolean flag `_is_running`.

### 4. Integration in `main.py`
Window 2 (`window_model`) exposes a "Next → Training" button via `_on_next()`. The `PipelineController` now switches the stack to the `TrainingWindow`, finishing the sequence.

## Validation & Testing
- Automated test `tests/test_phase4.py` proves that `TrainingWorker` completes standard dummy training cycles and properly emits signals, avoiding main thread crashes.

# Phase 4 Walkthrough: Multithreading, Hardware Selection & Training Configuration

## Overview
Phase 4 implements multithreaded PyTorch training within a PyQt desktop application,
decoupled via `ProjectState`. It also introduces smart Loss Function and Optimizer
selection, allowing users to choose the right criterion and optimizer for their task
directly from the UI.

---

## Key Components

### 1. `workers.training_worker.TrainingWorker` (QThread)
**Purpose**: Moves the intensive PyTorch training loop off the main UI thread to
prevent the application from freezing.

**Mechanics**:
- Inherits from `PyQt6.QtCore.QThread`.
- Reads `ProjectState` to grab the dataset (`DataFrame`), model (`nn.Sequential`),
  target column, hyperparameters, **and now `loss_fn_name` / `optimizer_name`**.
- Instantiates loss and optimizer via `backend.training_config.build_loss` and
  `build_optimizer` — no hardcoded values remain in the worker.
- **Automatic Label Encoding**: Classification targets are automatically mapped to
  `[0, C-1]` to prevent common PyTorch range errors.
- Handles both Percentage Split and K-Fold CV logic inside the `run()` override.
- Emits thread-safe signals:
  - `epoch_finished` — carries `(epoch, train_loss, val_loss, metrics)` for charting
  - `batch_progress` — drives the progress bar
  - `evaluation_finished` — carries final evaluation metrics
  - `training_finished` — signals success or error
  - `log_message` — safely proxies print statements to the UI console

### 2. Hardware Selection
**Purpose**: Allows selection of the compute device.

**Mechanics**:
- `torch.device(self.state.device)` drives computation inside the thread.
- A `QComboBox` is populated only with hardware actually available at runtime:
  - CPU is always shown.
  - `torch.cuda.is_available()` gates the CUDA option.
  - `torch.backends.mps.is_available()` gates the MPS (Apple Silicon) option.

### 3. Loss Function & Optimizer Selection (`backend/training_config.py`)
**Purpose**: Provide a single source of truth for all supported losses and
optimizers, keeping the registry completely decoupled from Qt.

**Mechanics**:
- `_LOSS_REGISTRY` maps `problem_type → [(display_name, nn.Module class), ...]`.
  Classification gets `CrossEntropyLoss`, `BCEWithLogitsLoss`, `NLLLoss`;
  regression gets `MSELoss`, `L1Loss`, `SmoothL1Loss`.
- `_OPTIMIZER_REGISTRY` is problem-agnostic: `Adam`, `AdamW`, `SGD`, `RMSprop`.
- Public API:
  - `get_losses_for(problem_type)` → `list[str]` — consumed by the UI dropdown
  - `get_all_optimizers()` → `list[str]` — consumed by the UI dropdown
  - `build_loss(name)` → `nn.Module` — consumed by `TrainingWorker`
  - `build_optimizer(name, parameters, lr)` → `optim.Optimizer` — consumed by `TrainingWorker`

**Why a registry?**
Adding a new loss or optimizer in the future requires changing only one file
(`training_config.py`); the UI and the worker pick it up automatically.

### 4. `utils.project_state.ProjectState` — New Fields
Two new fields carry the user's selection across windows:

```python
loss_fn_name: str   # e.g. "CrossEntropyLoss" — default from training_config.DEFAULT_LOSS
optimizer_name: str # e.g. "Adam"              — default from training_config.DEFAULT_OPTIMIZER
```

Defaults are imported directly from the registry so they stay in sync automatically.

### 5. `ui.window_training.TrainingWindow` — Training Configuration Panel
**Purpose**: The 3rd window in the No-Code Pipeline.

**New "Training Configuration" panel** (sits between Hyperparameters and Hardware):
- `QComboBox combo_loss` — populated by `get_losses_for(state.problem_type)` at
  build time and re-populated on every `refresh_ui()` call so that a problem-type
  change in Window 1 is always reflected.
- `QComboBox combo_optimizer` — populated by `get_all_optimizers()`.
- On **Start Training**, both selections are written back into `ProjectState` before
  the worker is launched.

**Existing UI elements** (unchanged):
- Hyperparameter panel: Learning Rate, Epochs, Batch Size.
- Hardware panel: CPU / CUDA / MPS device selector.
- Log console, progress bar, start/stop controls.

### 6. Integration in `main.py`
`NeuralForgeApp._switch_tab(3)` calls `self.train_dash.refresh_ui()` every time
the user navigates to training, ensuring the loss combo reflects the current
`state.problem_type`, the Model Summary tab displays the current architecture and
parameter count, and the `PlotPanel` adapts to show/hide the validation metrics chart.

### 7. Decoupled UI Components
To keep `window_training.py` clean, several sub-panels have been extracted:
- **`ui/plot_panel.py`**: Wraps PyQtGraph. Manages the Loss curve and, for classification tasks, an additional chart for **Validation Accuracy and F1 Score**.
- **`ui/monitor_panel.py`**: Wraps the `QTimer` and `psutil`/`torch` polling to monitor CPU, RAM, and VRAM in the background.

---

## Validation & Testing

- `tests/test_phase4.py` — expanded test suite across 4 classes:
  - **TestTrainingConfig** — registry getters, `build_loss`, `build_optimizer`, defaults
  - **TestProjectStateDefaults** — new `loss_fn_name` / `optimizer_name` fields
  - **TestTrainingWorker** — end-to-end runs, metrics calculations for classification, stop-flags, missing model error
  - **TestTrainingWindowUI** — combo item counts/names, defaults, `refresh_ui` problem-type switch, Model Summary updates

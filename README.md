# 🧠 Neural Forge

> A dual-path desktop app for building, training, and monitoring neural networks — **No-Code** for beginners, **Developer Mode** for experienced users who bring their own PyTorch project.

Built with **PyQt6** (dark-themed UI) and **PyTorch** (ML backend), following a decoupled architecture where the visual UI acts purely as a configuration generator and the ML engine operates independently from a standardised blueprint.

---

## Features (Current)

| Feature | Status |
|---|---|
| Home Screen with No-Code + Developer Mode paths | ✅ |
| Dark-themed pipeline (Data → Model → Training) | ✅ |
| CSV loading with auto-detection of columns and NaN handling | ✅ |
| 5-row data preview in a table view | ✅ |
| Target column picker (defaults to last column) | ✅ |
| Classification / Regression toggle | ✅ |
| Train/Validation split (percentage or K-Fold CV) | ✅ |
| Sequential layer builder (Linear, Conv1d, MaxPool1d, AvgPool1d, Flatten, BatchNorm1d, Dropout) | ✅ |
| Dynamic add/remove layer rows | ✅ |
| Blueprint save/load as JSON | ✅ |
| Blueprint validation (10 rules) | ✅ |
| Blueprint → `nn.Sequential` with LazyLinear auto-inference | ✅ |
| Ghost Run (dummy tensor validation) | ✅ |
| Multithreaded training with `QThread` | ✅ |
| GPU/CPU/MPS hardware toggle | ✅ |
| Real-time loss curve (pyqtgraph) | ✅ |
| CPU/RAM/VRAM resource monitor | ✅ |
| ONNX model export | ✅ |
| **Developer Mode** | |
| Project import with folder picker | ✅ |
| Project structure guide dialog (Don't show again) | ✅ |
| `config.yaml` bridge (UI → scripts) | ✅ |
| Code editor / training integration | 🔜 |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Conda (recommended)

### Installation

> *It is recommended to create an environment and activate it before running the application. (venv or conda)*

```bash
# Clone the repository
git clone https://github.com/Nawfel-9/neural-forge
cd neural-forge

# Install dependencies
pip install -r requirements.txt
```

>*If using GPU, follow the instructions from https://pytorch.org/get-started/locally/* for installing pytorch*

### Running the Application

```bash
python main.py
```

### Running Tests

```bash
python -m pytest tests/ -v
```

**Current test results: 80/80 passing** (28 Phase 1 + 24 Phase 2 + 28 Phase 3)

---

## Project Structure

```
├── main.py                    # Entry point + PipelineController
├── requirements.txt           # Dependencies
│
├── ui/                        # UI layer (PyQt6)
│   ├── styles.py              # Dark theme (QPalette + QSS)
│   ├── window_data.py         # Window 1: Data loading & preprocessing
│   ├── window_model.py        # Window 2: Sequential model builder
│   ├── window_project_guide.py # Dev Mode: onboarding / structure guide
│   ├── layer_row.py           # Custom widget: one layer row
│   └── data_table_view.py     # QTableView wrapper for DataFrames
│
├── backend/                   # ML backend (no Qt dependency)
│   ├── data_handler.py        # CSV load, clean, split
│   ├── model_builder.py       # Blueprint → nn.Sequential (Phase 3)
│   └── exporter.py            # ONNX export (Phase 5)
│
├── workers/                   # Threading
│   └── training_worker.py     # QThread + pyqtSignals (Phase 4)
│
├── utils/                     # Shared utilities
│   ├── project_state.py       # ProjectState dataclass
│   ├── blueprint_io.py        # JSON save/load
│   └── validators.py          # Blueprint validation
│
├── tests/                     # Automated tests
│   ├── test_phase1.py         # 28 tests
│   ├── test_phase2.py         # 24 tests
│   └── test_phase3.py         # 28 tests
│
└── docs/                      # Documentation
    ├── architecture.md
    ├── walkthrough_phase1.md
    ├── walkthrough_phase2.md
    ├── walkthrough_phase3.md
    └── walkthrough_dev_mode.md
```

---

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | System design, directory layout, pipeline diagram, blueprint format, dependencies |
| [Phase 1 Walkthrough](docs/walkthrough_phase1.md) | Layer builder UI, blueprint save/load, validation — every function explained |
| [Phase 2 Walkthrough](docs/walkthrough_phase2.md) | Data loading, cleaning, splitting, table preview — every function explained |
| [Phase 3 Walkthrough](docs/walkthrough_phase3.md) | Blueprint → nn.Sequential, LazyLinear, ghost run — every function explained |
| [Dev Mode Walkthrough](docs/walkthrough_dev_mode.md) | HomeWindow, ProjectGuideDialog, config.yaml bridge, dual-path architecture |

---

## Development Progress

### Phase 0: Planning & Architecture ✅
Defined the 3-window pipeline architecture, chose the tech stack, established the blueprint format as the decoupling contract between UI and ML backend.

### Phase 1: Sequential Builder UI & Save/Load ✅
Built Window 2 (Model Builder) with a scrollable layer list, dynamic add/remove, JSON save/load, and 7-rule blueprint validation. Dark theme applied globally.

### Phase 2: Data Pipeline & Validation ✅
Built Window 1 (Data Loading) with CSV file loading, 5-row table preview, target column picker, classification/regression toggle, percentage/K-Fold split config, NaN cleaning (fill mean or drop rows), and a 5-step validation chain before proceeding. Wired up the 2-window pipeline with forward/backward navigation.

### Phase 3: PyTorch Translation Engine ✅
Built the translation engine (`model_builder.py`) that converts blueprints into `nn.Sequential` models using `LazyLinear`/`LazyConv1d`/`LazyBatchNorm1d` for auto-inferred dimensions. Ghost run validates architectures with a dummy forward pass. Added "Build & Test" button to Window 2.

### Phase 4: Multithreading & Hardware Selection ✅
Built `TrainingWorker(QThread)` for multithreaded non-blocking PyTorch training loops, a CPU/CUDA/MPS automatic detection interface, and UI elements integrated into a third window (`window_training.py`).

### Phase 5: Visualization, Monitoring & Export ✅
Integrated `pyqtgraph` for real-time live loss curves, a `QTimer`-driven `psutil` integration for subsystem resource monitoring, and a fully functional ONNX export bridge in `backend/exporter.py`.

---

## Architecture Highlights

- **Decoupled design** — UI produces a standard blueprint (list of dicts), backend consumes it. Zero coupling.
- **Foolproof fallbacks** — Every user interaction wrapped in `try/except` with friendly `QMessageBox` dialogs.
- **Thread-safe** — Training runs on `QThread` with `pyqtSignal` (Phase 4), never on the main GUI thread.
- **Dark theme** — Comprehensive QSS + QPalette styling defined once in `styles.py`.
- **Tested** — 80 automated tests across 3 phases, covering logic, I/O, validation, model building, and UI widgets.

---

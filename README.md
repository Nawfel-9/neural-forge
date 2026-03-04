# 🧠 Neural Network Builder

> A low-code/no-code desktop application for building, training, and visualising neural networks — without writing a single line of code.

Built with **PyQt6** (dark-themed UI) and **PyTorch** (ML backend), following a decoupled architecture where the visual UI acts purely as a configuration generator and the ML engine operates independently from a standardised blueprint.

---

## Features (Current)

| Feature | Status |
|---|---|
| Dark-themed 3-window pipeline (Data → Model → Training) | ✅ |
| CSV loading with auto-detection of columns and NaN handling | ✅ |
| 5-row data preview in a table view | ✅ |
| Target column picker (defaults to last column) | ✅ |
| Classification / Regression toggle | ✅ |
| Train/Validation split (percentage or K-Fold CV) | ✅ |
| Sequential layer builder (Linear, Dropout, BatchNorm1d) | ✅ |
| Dynamic add/remove layer rows | ✅ |
| Blueprint save/load as JSON | ✅ |
| Blueprint validation (7 rules) | ✅ |
| Blueprint → `nn.Sequential` model builder | 🔜 Phase 3 |
| Ghost Run (dummy tensor validation) | 🔜 Phase 3 |
| Multithreaded training with `QThread` | 🔜 Phase 4 |
| GPU/CPU/MPS hardware toggle | 🔜 Phase 4 |
| Real-time loss curve (pyqtgraph) | 🔜 Phase 5 |
| CPU/RAM/VRAM resource monitor | 🔜 Phase 5 |
| ONNX model export | 🔜 Phase 5 |

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

**Current test results: 52/52 passing** (28 Phase 1 + 24 Phase 2)

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
│   └── test_phase2.py         # 24 tests
│
└── docs/                      # Documentation
    ├── architecture.md
    ├── walkthrough_phase1.md
    └── walkthrough_phase2.md
```

---

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | System design, directory layout, pipeline diagram, blueprint format, dependencies |
| [Phase 1 Walkthrough](docs/walkthrough_phase1.md) | Layer builder UI, blueprint save/load, validation — every function explained |
| [Phase 2 Walkthrough](docs/walkthrough_phase2.md) | Data loading, cleaning, splitting, table preview — every function explained |

---

## Development Progress

### Phase 0: Planning & Architecture ✅
Defined the 3-window pipeline architecture, chose the tech stack, established the blueprint format as the decoupling contract between UI and ML backend.

### Phase 1: Sequential Builder UI & Save/Load ✅
Built Window 2 (Model Builder) with a scrollable layer list, dynamic add/remove, JSON save/load, and 7-rule blueprint validation. Dark theme applied globally.

### Phase 2: Data Pipeline & Validation ✅
Built Window 1 (Data Loading) with CSV file loading, 5-row table preview, target column picker, classification/regression toggle, percentage/K-Fold split config, NaN cleaning (fill mean or drop rows), and a 5-step validation chain before proceeding. Wired up the 2-window pipeline with forward/backward navigation.

### Phase 3: PyTorch Translation Engine 🔜
*Next up:* Blueprint → `nn.Sequential`, `nn.LazyLinear`, ghost run validation.

### Phase 4: Multithreading & Hardware Selection 🔜

### Phase 5: Visualization, Monitoring & Export 🔜

---

## Architecture Highlights

- **Decoupled design** — UI produces a standard blueprint (list of dicts), backend consumes it. Zero coupling.
- **Foolproof fallbacks** — Every user interaction wrapped in `try/except` with friendly `QMessageBox` dialogs.
- **Thread-safe** — Training runs on `QThread` with `pyqtSignal` (Phase 4), never on the main GUI thread.
- **Dark theme** — Comprehensive QSS + QPalette styling defined once in `styles.py`.
- **Tested** — 52 automated tests across 2 phases, covering logic, I/O, validation, and UI widgets.

---

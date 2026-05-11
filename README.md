# Neural Forge

> A dual-path desktop app for building, training, monitoring, and exporting neural networks. The No-Code path is for visual model building; Developer Mode validates an imported PyTorch project structure and can hand it to the Training view for code-first training.

Neural Forge is built with PyQt6 and PyTorch. The no-code workflow keeps the UI, project state, and backend ML logic decoupled: the UI produces a layer blueprint and training configuration, and the backend consumes those objects without depending on Qt.

---

## Features

| Area | Current capability |
|---|---|
| App shell | Home screen, persistent sidebar, dark/light theme toggle |
| No-Code data flow | CSV/Parquet loading, profiling, cleaning, feature engineering, preprocessing pipeline export |
| Model builder | Sequential layer builder, locked input/output context, JSON blueprint save/load, validation, ghost run |
| Training | Background `QThread` training, percentage split or K-Fold CV, CPU/CUDA/MPS selector, live loss and classification metric plots |
| Evaluation | Final classification/regression metrics after training |
| Export | Preprocessing pipeline `.pkl`, PyTorch `.pt/.pth`, and ONNX `.onnx` export |
| Developer Mode | Project structure guide, folder import, required/optional file checklist, and Developer Mode handoff to the Training view |
| AI Assistant | NVIDIA API-backed chat tab with lightweight project context for engineering guidance |

Developer Mode validates project structure and preserves the imported project path. When required files are present, **Continue to Training** opens the Training view in Developer Mode and launches the imported project through the Developer Mode training scaffold.

---

## Installation

Python 3.10+ is recommended. Create and activate a virtual environment first.

```bash
git clone https://github.com/Nawfel-9/neural-forge
cd neural-forge
pip install -r requirements.txt
```

If you use the local Conda environment for this project:

```bash
conda activate nn_builder
pip install -r requirements.txt
```

For GPU builds, install the PyTorch wheel that matches your hardware from the official PyTorch selector, then install the remaining requirements.

The AI Assistant reads its NVIDIA API settings from `.env`. Create it from the example file and fill in your local key:

```bash
cp .env.example .env
```

Required variables:

```text
NVIDIA_API_KEY=your-nvidia-api-key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_ASSISTANT_MODEL=nvidia/nemotron-mini-4b-instruct
NVIDIA_ASSISTANT_TIMEOUT_SECONDS=20
NVIDIA_ASSISTANT_MAX_TOKENS=1024
NVIDIA_ASSISTANT_ENABLE_THINKING=false
```

`.env` is intentionally ignored by Git.

---

## Running

```bash
python main.py
```

---

## Tests

```bash
python -m pytest tests -q
```

The tests require the dependencies in `requirements.txt`, including PyTorch and PyQt6. In headless environments, set `QT_QPA_PLATFORM=offscreen` before running UI tests.

---

## Project Structure

```text
neural-forge/
├── .env.example                 # Assistant environment variable template
├── main.py                      # App shell, navigation, Home, Developer Mode status page
├── requirements.txt             # Runtime and test dependencies
├── assets/
│   └── logo.png
├── backend/
│   ├── data_handler.py          # Data loading, cleaning, preprocessing, splitting
│   ├── dev_trainer.py           # Developer Mode training worker scaffold
│   ├── assistant_client.py      # NVIDIA OpenAI-compatible assistant client
│   ├── exporter.py              # ONNX export helper
│   ├── model_builder.py         # Blueprint to nn.Sequential + ghost run
│   └── training_config.py       # Loss and optimizer registries
├── docs/
│   ├── architecture.md
│   ├── walkthrough_dev_mode.md
│   ├── walkthrough_phase1.md
│   ├── walkthrough_phase2.md
│   ├── walkthrough_phase3.md
│   ├── walkthrough_phase4.md
│   ├── walkthrough_phase5.md
│   ├── walkthrough_preprocessing.md
│   └── walkthrough_refinements.md
├── tests/
├── ui/
│   ├── custom_toggle.py
│   ├── data_table_view.py
│   ├── dialog_report.py
│   ├── layer_row.py
│   ├── monitor_panel.py
│   ├── plot_panel.py
│   ├── styles.py
│   ├── window_data.py
│   ├── window_assistant.py
│   ├── window_export.py
│   ├── window_model.py
│   ├── window_project_guide.py
│   ├── window_project_validation.py  # Static validation screen scaffold
│   └── window_training.py
├── utils/
│   ├── blueprint_io.py
│   ├── config_schema.py         # Developer Mode config schema
│   ├── project_state.py
│   └── validators.py
└── workers/
    ├── assistant_worker.py
    ├── data_loader_worker.py
    └── training_worker.py
```

---

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | Current app architecture, data flow, dependencies, and verification plan |
| [Phase 1 Walkthrough](docs/walkthrough_phase1.md) | Layer builder, blueprint I/O, validation |
| [Phase 2 Walkthrough](docs/walkthrough_phase2.md) | Data backend, async data worker, data UI |
| [Phase 3 Walkthrough](docs/walkthrough_phase3.md) | Blueprint translation and ghost runs |
| [Phase 4 Walkthrough](docs/walkthrough_phase4.md) | Training worker, hardware selection, loss/optimizer registry |
| [Phase 5 Walkthrough](docs/walkthrough_phase5.md) | Plotting, monitoring, export |
| [Developer Mode Walkthrough](docs/walkthrough_dev_mode.md) | Project guide dialog, folder import, checklist page |
| [Preprocessing Walkthrough](docs/walkthrough_preprocessing.md) | Feature engineering, preprocessing, pipeline persistence |
| [Refinements Walkthrough](docs/walkthrough_refinements.md) | Production polish and robustness updates |

---

## Architecture Highlights

- **Decoupled no-code backend**: the no-code backend modules do not import Qt; `backend/dev_trainer.py` is a separate Developer Mode `QThread` worker.
- **Shared project state**: `ProjectState` carries data, blueprint, model, training settings, `training_mode`, and Developer Mode import state across screens.
- **Registry-driven training config**: losses and optimizers live in `backend/training_config.py`.
- **Threaded long-running work**: data operations use `DataLoaderWorker`; model training uses `TrainingWorker`.
- **Assistant integration**: `AssistantWindow` streams NVIDIA API responses through `AssistantWorker` and injects a compact project-state summary.
- **Export-ready outputs**: the final app can export preprocessing and model artifacts without coupling export logic to the UI.

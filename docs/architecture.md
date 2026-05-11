# Architecture

> Neural Forge is a PyQt6 + PyTorch desktop app with a no-code neural-network pipeline, a Developer Mode project-import path, and an NVIDIA-backed AI Assistant.

---

## Top-Level Design

The application has a single `QMainWindow` (`NeuralForgeApp`) with a persistent sidebar and a central `QStackedWidget`.

```mermaid
flowchart TB
    Home["Home"]
    Data["Data Engineering Lab"]
    Model["Model Architecture Builder"]
    Train["Training & Evaluation Lab"]
    Export["Deployment & Export"]
    DevGuide["Developer Mode Guide"]
    DevImport["Folder Picker"]
    DevStatus["Developer Mode Status"]
    DevTrain["Training View (Developer Mode)"]
    Assistant["AI Assistant"]

    Home -- "No-Code Pipeline" --> Data
    Data --> Model
    Model --> Train
    Train --> Export
    Home -- "Developer Mode" --> DevGuide
    DevGuide --> DevImport
    DevImport --> DevStatus
    DevStatus -- "Continue to Training" --> DevTrain
    DevStatus -- "Use No-Code Pipeline" --> Data
    Home --> Assistant
    Data --> Assistant
    Model --> Assistant
    Train --> Assistant
    Export --> Assistant
```

Developer Mode validates a code project folder and preserves the imported project path. When all required files are present, the status page can switch the Training view into Developer Mode. The AI Assistant can use the current project state as context, but it does not execute training or mutate project files.

The repository also contains a static-validation screen scaffold (`ui.window_project_validation`) that is not currently added to the main stacked-widget flow.

---

## Runtime State

`utils.project_state.ProjectState` is the shared mutable state passed to each screen.

```python
dataframe: pd.DataFrame | None
target_column: str
problem_type: str              # "classification" | "regression"
pipeline: Any                  # preprocessing DataPipeline
split_config: dict             # percentage/kfold + optional resampling
blueprint: list[dict]
model: Any                     # nn.Module after build/ghost run
dummy_tensor: Any              # export trace input
hyperparams: dict              # lr, epochs, batch_size
device: str                    # "cpu" | "cuda" | "mps"
loss_fn_name: str
optimizer_name: str
training_metrics: dict         # final metrics used by the report dialog
training_mode: str             # "nocode" | "dev"
dev_project_path: str
```

---

## Directory Structure

```text
neural-forge/
├── main.py
├── requirements.txt
├── assets/
├── backend/
│   ├── assistant_client.py
│   ├── data_handler.py
│   ├── dev_trainer.py
│   ├── hardware_monitor.py
│   ├── exporter.py
│   ├── model_builder.py
│   └── training_config.py
├── docs/
├── tests/
├── ui/
│   ├── custom_toggle.py
│   ├── data_table_view.py
│   ├── dialog_report.py
│   ├── layer_row.py
│   ├── monitor_panel.py
│   ├── plot_panel.py
│   ├── styles.py
│   ├── window_assistant.py
│   ├── window_data.py
│   ├── window_export.py
│   ├── window_model.py
│   ├── window_project_guide.py
│   ├── window_project_validation.py
│   ├── window_training_dev.py
│   └── window_training.py
├── utils/
│   ├── blueprint_io.py
│   ├── config_schema.py
│   ├── project_state.py
│   └── validators.py
└── workers/
    ├── assistant_worker.py
    ├── data_loader_worker.py
    └── training_worker.py
```

---

## No-Code Pipeline

1. **Data Engineering Lab** loads CSV/Parquet files, profiles data, applies cleaning/feature engineering/preprocessing, and stores the processed dataframe plus `DataPipeline`.
2. **Model Architecture Builder** builds a sequential blueprint, locks the output layer to the current problem type, validates the blueprint, and materializes a PyTorch model with a ghost run.
3. **Training & Evaluation Lab** trains the model in a `QThread`, plots losses and classification metrics, and computes final evaluation metrics.
4. **Deployment & Export** exports the preprocessing pipeline, PyTorch model state dict, or ONNX model.

---

## Developer Mode

Developer Mode uses `ui.window_project_guide.ProjectGuideDialog` to explain required project structure, then imports a folder path and checks for:

| Required | Optional |
|---|---|
| `model.py` | `loss.py` |
| `dataset.py` | `metrics.py` |
| `config.yaml` | `checkpoints/`, `logs/` |

The import path is stored as `ProjectState.dev_project_path`. If all required files exist, **Continue to Training** sets `ProjectState.training_mode = "dev"` and opens the Training view.

Developer Mode training uses this runtime contract:

| File | Expected hook |
|---|---|
| `model.py` | `get_model(config: dict) -> torch.nn.Module` |
| `dataset.py` | `get_dataloader(config: dict, split: str) -> torch.utils.data.DataLoader` |
| `loss.py` | Optional `get_loss(config: dict) -> torch.nn.Module` |
| `metrics.py` | Optional `get_metrics(config: dict) -> dict[str, callable]` |

Related Developer Mode files:

| File | Scope |
|---|---|
| `ui/window_project_validation.py` | Static AST validation screen scaffold; not currently added to `main.py`'s stacked-widget flow |
| `backend/dev_trainer.py` | QThread-based user-project trainer launched from the Training view in Developer Mode |
| `backend/hardware_monitor.py` | QThread-based GPU/CPU/RAM stats monitor used by the Developer Mode dashboard and thermal guard |
| `ui/window_training_dev.py` | Developer Mode hardware dashboard widgets used by the shared Training view |
| `utils/config_schema.py` | YAML-backed serializable config object used by the Developer Mode trainer |

---

## AI Assistant

The assistant tab uses `ui.window_assistant.AssistantWindow` for the chat UI and `workers.assistant_worker.AssistantWorker` for background streaming. The backend client lives in `backend.assistant_client` and talks to NVIDIA's OpenAI-compatible endpoint.

Configuration is read from environment variables, usually through a local `.env` file:

| Variable | Purpose |
|---|---|
| `NVIDIA_API_KEY` | Required API key for NVIDIA hosted models |
| `NVIDIA_BASE_URL` | API base URL, defaults to `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_ASSISTANT_MODEL` | Chat model, defaults to `nvidia/nemotron-mini-4b-instruct` |
| `NVIDIA_ASSISTANT_TIMEOUT_SECONDS` | Request timeout, defaults to `20` |
| `NVIDIA_ASSISTANT_MAX_TOKENS` | Response token budget, defaults to `1024` |
| `NVIDIA_ASSISTANT_ENABLE_THINKING` | Enables provider reasoning chunks when set to `true`; defaults to `false` for faster chat |

The assistant receives a compact project summary generated from `ProjectState`: dataset status, target, problem type, split configuration, blueprint layers, training settings, export readiness, and Developer Mode path. It does not receive full dataset contents.

---

## Backend Boundaries

Production no-code backend modules do not import PyQt6. The exception is `backend/dev_trainer.py`, which is Developer Mode training scaffolding and intentionally owns a `QThread`.

| Module | Responsibility |
|---|---|
| `backend/assistant_client.py` | NVIDIA API settings, project context summary, streaming chat responses |
| `backend/data_handler.py` | Loading, profiling, cleaning, feature engineering, preprocessing, splitting |
| `backend/dev_trainer.py` | Developer Mode user-project training worker scaffold |
| `backend/hardware_monitor.py` | Developer Mode GPU/CPU/RAM sampling thread |
| `backend/model_builder.py` | Blueprint translation and ghost-run validation |
| `backend/training_config.py` | Loss/optimizer registries |
| `backend/exporter.py` | ONNX export |

Long-running work is delegated to workers:

| Worker | Purpose |
|---|---|
| `AssistantWorker` | Streams chat responses off the UI thread |
| `DataLoaderWorker` | Runs data operations off the UI thread |
| `TrainingWorker` | Runs PyTorch training and evaluation off the UI thread |

---

## Core Dependencies

| Package | Purpose |
|---|---|
| `PyQt6` | Desktop UI |
| `torch` | Model building and training |
| `numpy`, `pandas`, `pyarrow` | Data loading and preprocessing |
| `scikit-learn`, `imbalanced-learn` | Splits, metrics, preprocessing, optional resampling |
| `pyqtgraph` | Live plots |
| `psutil` | CPU/RAM monitoring |
| `onnx` | ONNX export validation |
| `openai`, `python-dotenv` | NVIDIA AI Assistant API client and local environment loading |

`pynvml` is used opportunistically for NVIDIA GPU temperature/utilization data when it is installed, but it is not required for the app to run.

---

## Verification

Run the full test suite after installing dependencies:

```bash
python -m pytest tests -q
```

Run syntax checks without importing optional runtime packages:

```bash
python -m py_compile main.py backend/*.py ui/*.py utils/*.py workers/*.py
```

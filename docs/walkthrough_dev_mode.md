# Developer Mode — Walkthrough

> Import an existing PyTorch project, configure training from the UI, and share the monitoring/training window with the No-Code pipeline.

---

## Overview

Neural Forge offers two paths from the Home Screen:

| Path | Target user | Entry point |
|---|---|---|
| **No-Code Pipeline** | Beginners | Build model layer-by-layer in the UI |
| **Developer Mode** | Experienced devs | Import your own PyTorch project folder |

Both paths converge on the **shared Training window** (Phase 4) for execution, monitoring, and export.

---

## Files

| File | Purpose |
|---|---|
| [`main.py`](../main.py) | `HomeWindow` (landing page) + updated `PipelineController` |
| [`ui/window_project_guide.py`](../ui/window_project_guide.py) | Onboarding dialog explaining project structure rules |

---

## 1. `HomeWindow` — The Landing Page

**Where:** defined in [`main.py`](../main.py)

Presents two large buttons:

```
┌─────────────────────────────────────────────────────────┐
│           Welcome to Neural Forge                       │
│                                                         │
│   ┌──────────────────┐    ┌──────────────────┐          │
│   │   No-Code        │    │   Import Project │          │
│   │   Pipeline       │    │   (Developer     │          │
│   │   (Beginner      │    │    Mode)         │          │
│   │    Friendly)     │    │                  │          │
│   └──────────────────┘    └──────────────────┘          │
│                                                         │
│   Build, train, and monitor neural networks             │
└─────────────────────────────────────────────────────────┘
```

| Button | What it does |
|---|---|
| **No-Code Pipeline** | Hides Home → shows Window 1 (Data Loading) |
| **Import Project** | Shows the Project Guide dialog → folder picker |

---

## 2. `ProjectGuideDialog` — Onboarding

**Where:** [`ui/window_project_guide.py`](../ui/window_project_guide.py)

A modal dialog shown the **first time** the user clicks "Import Project." Explains the required file naming conventions so the platform can auto-discover project files.

### Components

| Component | Purpose |
|---|---|
| `FileRow` | Reusable row widget: icon + filename + REQUIRED/OPTIONAL badge + description |
| `SectionHeader` | Styled uppercase section label |
| `ProjectGuideDialog` | Modal dialog with scroll area, file rows, info callout, and "Don't show again" |

### Required project structure

The dialog teaches users to structure their project folder like this:

```
my_project/
├── model.py        ← REQUIRED  nn.Module architecture class
├── dataset.py      ← REQUIRED  DataLoader logic, augmentations, splits
├── loss.py         ← OPTIONAL  Custom loss functions (default: MSE)
├── metrics.py      ← OPTIONAL  Evaluation logic (mAP, IoU, Accuracy)
├── config.yaml     ← REQUIRED  UI writes hyperparams here; scripts read from it
├── checkpoints/    ← OPTIONAL  Auto-saved .pth weight files
└── logs/           ← OPTIONAL  Custom log files (tailed in UI terminal panel)
```

### The `config.yaml` bridge

The key integration point between the UI and user scripts:

```yaml
# Written by the Neural Forge UI:
learning_rate: 0.001
batch_size: 32
optimizer: Adam
epochs: 50
image_size: 224
```

```python
# Read by the user's scripts:
import yaml
cfg = yaml.safe_load(open('config.yaml'))
lr = cfg['learning_rate']
```

This means the user **does not hardcode** hyperparameters — they configure them in the UI and let `config.yaml` act as the sync file.

### "Don't show again"

Uses `QSettings("NeuralForge", "NeuralForge")` under the key `developer_mode/skip_guide` to persist the user's preference across app restarts.

### Subtle UX touches

- **Fade-in animation** — `QPropertyAnimation` on `windowOpacity` (220ms, OutCubic easing)
- **Hover effect on file rows** — border color changes to highlight on hover
- **DEV MODE badge** — styled label at top of dialog

---

## 3. `PipelineController` — Updated Flow

The controller now manages two paths:

| Method | What it does |
|---|---|
| `start()` | Shows the `HomeWindow` |
| `_start_no_code_pipeline()` | Hides Home → creates DataWindow → shows it |
| `_open_model_window()` | No-Code path: Data → Model |
| `_back_to_data()` | Model → Data (backtrack) |
| `_open_code_editor()` | Dev path: Guide dialog → folder picker → *stub* |

### Developer Mode flow (current state)

```
HomeWindow
    │
    ├── "No-Code Pipeline" → DataWindow → ModelBuilderWindow → ...
    │
    └── "Import Project"
            │
            ├── ProjectGuideDialog.should_show()?
            │       │
            │       ├── Yes → Show dialog
            │       │       │
            │       │       ├── "Got it — Import Project →" → folder picker
            │       │       └── "Cancel" → return to Home
            │       │
            │       └── No (suppressed) → folder picker directly
            │
            └── QFileDialog.getExistingDirectory()
                    │
                    └── print(path)  ← STUB — hand off to CodeEditorWindow
```

> [!NOTE]
> The `CodeEditorWindow` is not yet implemented. The stub prints the selected folder path to the console. This will be completed in a future phase.

---

## 4. Shared Training Window

Both paths will converge on **Window 3 (Training)**, which is planned for Phase 4. The training window will:

- Accept either an `nn.Sequential` model (from No-Code) or a user-provided `model.py` (from Dev Mode)
- Display real-time loss curves, metrics, and resource usage
- Handle hyperparameter configuration (No-Code reads from ProjectState; Dev Mode reads from `config.yaml`)
- Support checkpointing and model export

# Phase 1 — Code Walkthrough

> Sequential Model Builder UI, blueprint save/load, and validation.

---

## Files Overview

| File | Purpose |
|---|---|
| [`utils/project_state.py`](../utils/project_state.py) | Shared data container for the app screens |
| [`ui/styles.py`](../ui/styles.py) | Light/dark theme palettes + QSS stylesheet |
| [`ui/layer_row.py`](../ui/layer_row.py) | Custom widget: one row = one layer |
| [`ui/window_model.py`](../ui/window_model.py) | View 2: the Model Builder |
| [`utils/blueprint_io.py`](../utils/blueprint_io.py) | JSON save/load for blueprints |
| [`utils/validators.py`](../utils/validators.py) | Blueprint validation rules |
| [`main.py`](../main.py) | Application entry point |
| [`tests/test_phase1.py`](../tests/test_phase1.py) | 28 automated tests |

---

## 1. `utils/project_state.py` — The Shared Data Container

**Why it exists:** The app has several views (Home → Data → Model → Training → Export, plus Developer Mode and Assistant). They all need to share data — the loaded CSV, the layer blueprint, hyperparameters, final metrics, imported Developer Mode path, etc. Instead of passing dozens of variables around, they all read/write to **one `ProjectState`** object.

```python
@dataclass
class ProjectState:
    # Window 1 (Data) will fill these
    dataframe: Optional[pd.DataFrame] = None
    target_column: str = ""
    problem_type: str = "classification"
    split_config: dict  # {"method": "percentage", "ratio": 0.8}

    # Window 2 (Model) fills this
    blueprint: list[dict] = []

    # Phase 3+ fills these after building
    model: Any = None
    dummy_tensor: Any = None

    # Window 3 (Training)
    hyperparams: dict  # {"lr": 0.001, "epochs": 50, "batch_size": 32}
    device: str = "cpu"
```

### Key function

| Function | What it does |
|---|---|
| `input_features()` | Counts columns in the DataFrame minus the target column. Returns `0` if no data is loaded. |

---

## 2. `ui/styles.py` — Theme System

**Why it exists:** Every widget inherits its appearance from this file, so light and dark themes are defined in one place.

### Two layers of theming

**Layer 1 — `apply_theme_palette(app, is_dark)`**
Sets the **QPalette** (Qt's low-level colour system). Controls default widget colours that QSS can't always reach — window backgrounds, disabled text, highlights, tooltips, etc.

**Layer 2 — `get_qss(is_dark)`**
Returns the global CSS-like stylesheet for the selected theme. It styles every widget type:

| Section | What it styles |
|---|---|
| `QWidget` | Base background, font family, font size |
| `QFrame[frameShape="1"]` | "Card" appearance for layer rows |
| `QPushButton` | Default, hover, pressed, disabled states |
| `QPushButton[class="primary"]` | Blue accent button (e.g. "Add Layer") |
| `QPushButton[class="danger"]` | Red outline button (e.g. "✕ Remove") |
| `QSpinBox`, `QComboBox`, etc. | Input fields with rounded borders, focus glow |
| `QTableView`, `QHeaderView` | Data table styling |

### Colour tokens

```python
DARK_BG_SPACE = "#0B0F17"
DARK_ACCENT   = "#0EA5E9"
LIGHT_BG_SPACE = "#F8FAFC"
LIGHT_ACCENT   = "#0284C7"
RADIUS = "12px"
```

---

## 3. `ui/layer_row.py` — One Layer = One Row Widget

**Why it exists:** Each row in the scrolling list represents one neural network layer. The widget dynamically shows/hides controls based on the selected layer type.

### Supported layer types (7 total)

| Type | Controls shown | Blueprint output |
|---|---|---|
| **Linear** | Neurons, Activation | `{"type": "Linear", "neurons": 64, "activation": "ReLU"}` |
| **Conv1d** | Channels, Kernel, Stride, Padding | `{"type": "Conv1d", "out_channels": 32, "kernel_size": 3, ...}` |
| **MaxPool1d** | Kernel, Stride | `{"type": "MaxPool1d", "kernel_size": 2, "stride": 2}` |
| **AvgPool1d** | Kernel, Stride | `{"type": "AvgPool1d", "kernel_size": 2, "stride": 2}` |
| **Flatten** | *(none)* | `{"type": "Flatten"}` |
| **BatchNorm1d** | *(none)* | `{"type": "BatchNorm1d"}` |
| **Dropout** | Rate | `{"type": "Dropout", "rate": 0.3}` |

### Visual layout

```
(Linear)
┌──────────────────────────────────────────────────────────────────────┐
│  #1  │ Type: [Linear ▼] │ Neurons: [64] │ Activation: [None ▼] │ ✕ │
└──────────────────────────────────────────────────────────────────────┘

(Conv1d)
┌──────────────────────────────────────────────────────────────────────────────┐
│  #2  │ Type: [Conv1d ▼] │ Channels: [32] │ Kernel: [3] │ Stride: [1] │ Padding: [0] │ ✕ │
└──────────────────────────────────────────────────────────────────────────────┘

(MaxPool1d / AvgPool1d)
┌──────────────────────────────────────────────────────────────┐
│  #3  │ Type: [MaxPool1d ▼] │ Kernel: [2] │ Stride: [2] │ ✕ │
└──────────────────────────────────────────────────────────────┘

(Flatten / BatchNorm1d — only type + remove button visible)
(Dropout)
┌───────────────────────────────────────────────┐
│  #5  │ Type: [Dropout ▼] │ Rate: [0.30] │  ✕ │
└───────────────────────────────────────────────┘
```

### Functions

| Function | What it does |
|---|---|
| `__init__(index)` | Creates widget, wires signals, sets initial visibility |
| `_build_ui()` | Creates all child widgets in an `QHBoxLayout` |
| `_connect_signals()` | Type → show/hide; all inputs → `config_changed`; remove → `remove_requested` |
| `_on_type_changed()` | Dynamic visibility per layer type |
| `set_index(i)` | Updates the `#N` label after reordering |
| `get_config() → dict` | Returns row state as a dict |
| `set_config(dict)` | Populates widgets from a dict. Wrapped in try/except. |

### Signals

| Signal | When | Listener |
|---|---|---|
| `remove_requested(int)` | User clicks ✕ | `ModelBuilderWindow._remove_layer_row` |
| `config_changed()` | Any widget value changes | `ModelBuilderWindow._update_count_label` |

---

## 4. `ui/window_model.py` — The Model Builder View

**Why it exists:** Manages a vertical list of `LayerRow` widgets with save/load/validate controls and navigation.

### UI layout

```
┌─────────────────────────────────────────────────────┐
│  🧠  Model Architecture                    (header) │
│  📋  Dataset: 4 features • Target: class  (info)    │
│ ┌─────────────────────────────────────────────────┐  │
│ │  Scrollable layer list                          │  │
│ │  ┌── LayerRow #1 ──────────────────────────┐    │  │
│ │  └─────────────────────────────────────────┘    │  │
│ │  ┌── LayerRow #2 ──────────────────────────┐    │  │
│ │  └─────────────────────────────────────────┘    │  │
│ └─────────────────────────────────────────────────┘  │
│  Layers: 2                                           │
│  [← Back] [＋ Add] [💾 Save] [📂 Load] [✅ Validate] [🔨 Build & Test] │
└──────────────────────────────────────────────────────────────────────────┘
```

### Functions

| Group | Function | What it does |
|---|---|---|
| **Setup** | `__init__(project_state, on_back)` | Stores state + callback, inits row list, builds UI, adds one default layer |
| | `_build_ui()` | Full layout: header, data info, scroll area, count label, button bar |
| **Layer mgmt** | `_add_layer_row(config=None)` | Creates a `LayerRow`, inserts before trailing stretch |
| | `_remove_layer_row(index)` | Prevents removing the last row (shows warning) |
| | `_reindex_rows()` | Renumbers rows after deletion |
| | `_clear_all_rows()` | Removes all rows (used for blueprint loading) |
| **Blueprint** | `get_architecture() → list[dict]` | Extracts UI state as blueprint |
| **Save/Load** | `_save_blueprint()` | Validate → file dialog → JSON write |
| | `_load_blueprint()` | File dialog → parse JSON → validate → rebuild rows |
| **Validation** | `_validate_and_show()` | Shows ✅ or ⚠️ message |
| **Build** | `_build_and_test()` | Builds nn.Sequential + ghost run, shows result |
| | `sync_to_state() → bool` | Validates, builds model, ghost-runs, writes to `ProjectState` |

---

## 5. `utils/blueprint_io.py` — JSON Save / Load

| Function | What it does |
|---|---|
| `save_blueprint(blueprint, filepath)` | Wraps in envelope `{"version": 1, "layers": [...]}` and writes JSON |
| `load_blueprint(filepath) → list[dict]` | Reads JSON. Accepts envelope or bare-list format. Raises clear errors. |

The `"version": 1` field future-proofs the format for adding metadata later.

---

## 6. `utils/validators.py` — Blueprint Validation

### `validate_blueprint(blueprint) → (bool, str)`

| # | Rule | Why |
|---|---|---|
| 1 | Blueprint must not be empty | Can't build a network with zero layers |
| 2 | Every entry must have a known `"type"` | Catch typos early |
| 3 | At least one `Linear` layer must exist | Need at least one trainable layer |
| 4 | `Dropout.rate` must be in (0, 1) | PyTorch errors on 0.0 or 1.0 |
| 5 | Last layer must be `Linear` | Output layer must be Linear |
| 6 | `Linear.neurons` must be a positive integer | Zero neurons make no sense |
| 7 | `Linear.activation` must be a known value | Catch unsupported activations |
| 8 | `Conv1d.out_channels` must be a positive integer | Invalid channel count |
| 9 | `Conv1d` / pool `kernel_size` must be positive | Invalid kernel |
| 10 | `Conv1d` / pool `stride` must be positive | Invalid stride |

---

## 7. `main.py` — Entry Point & App Shell

```python
class NeuralForgeApp(QMainWindow):
    def _switch_tab(self, index: int):  # Switch QStackedWidget page
    def _start_no_code_pipeline(self):  # Home -> Data Lab
    def _open_dev_mode(self):           # Guide -> folder picker -> status page
```

```python
def main():
    app = QApplication(sys.argv)
    window = NeuralForgeApp()
    window.show()
    sys.exit(app.exec())
```

---

## 8. Data Flow

```
User clicks widgets in LayerRow widgets
        │
        ▼
get_architecture()  →  list[dict]   (the "blueprint")
        │
        ├──→ validate_blueprint()   →  (True/False, message)
        ├──→ save_blueprint()       →  .json file on disk
        ├──→ load_blueprint()       →  list[dict] → rebuild LayerRows
        └──→ sync_to_state()        →  ProjectState.blueprint
```

> **Key insight:** The **blueprint** is the contract between the UI and the ML backend. The UI never touches PyTorch directly. This decoupling means you could swap the UI for a CLI or web frontend without changing the backend.

---

## 9. `tests/test_phase1.py` — 28 Tests

| Test Class | # | What it covers |
|---|---|---|
| `TestProjectState` | 2 | Default values, `input_features()` with no data |
| `TestValidation` | 9 | All 7 rules + edge cases |
| `TestBlueprintIO` | 6 | Round-trip, versioning, legacy format, error handling |
| `TestLayerRowWidget` | 5 | Config get/set for all layer types, index update |
| `TestModelBuilderWindow` | 6 | Add/remove, architecture extraction, sync, clear |

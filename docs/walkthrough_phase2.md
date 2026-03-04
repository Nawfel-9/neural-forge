# Phase 2 — Code Walkthrough

> Data loading, cleaning, splitting, and the Data Window UI.

---

## Files Overview

| Status | File | Purpose |
|---|---|---|
| **NEW** | [`backend/data_handler.py`](../backend/data_handler.py) | CSV load, NaN clean, split (no Qt) |
| **NEW** | [`ui/data_table_view.py`](../ui/data_table_view.py) | QTableView wrapper for DataFrame |
| **NEW** | [`ui/window_data.py`](../ui/window_data.py) | Window 1: data loading UI |
| **NEW** | [`tests/test_phase2.py`](../tests/test_phase2.py) | 24 automated tests |
| Modified | [`main.py`](../main.py) | PipelineController (Window 1 ↔ 2) |
| Modified | [`ui/window_model.py`](../ui/window_model.py) | Added ← Back button + data info |

---

## 1. `backend/data_handler.py` — The Data Engine

**Why it exists:** All data logic — loading, cleaning, splitting — lives here. It takes file paths and config dicts, returns DataFrames and metadata. **Never touches Qt.** This means it can be tested independently and reused by future backends.

### Loading

| Function | What it does |
|---|---|
| `load_csv(filepath, **kwargs) → DataFrame` | Wraps `pd.read_csv`. Checks for empty/zero-column result → raises `ValueError`. Also raises `FileNotFoundError`, `EmptyDataError`, or `ParserError` from Pandas. |

### Cleaning

```python
class NaNStrategy:
    FILL_MEAN = "fill_mean"
    DROP_ROWS = "drop_rows"
```

| Function | What it does |
|---|---|
| `clean_dataframe(df, nan_strategy) → (cleaned_df, report)` | Works on a **copy** (original is never mutated). |

**Strategy: `fill_mean`**
1. **Numeric** columns → fill NaNs with column **mean**
2. **Non-numeric** columns → fill NaNs with column **mode** (most frequent)
3. If mode is empty → fills with `"UNKNOWN"`

**Strategy: `drop_rows`**
1. Drop any row containing at least one NaN
2. Reset the index

**The report dict:**
```python
{
    "nan_count_before": 5,
    "nan_count_after":  0,
    "rows_before":      100,
    "rows_after":       95,
    "strategy_used":    "drop_rows"
}
```

### Feature Detection

| Function | What it does |
|---|---|
| `detect_columns(df) → list[str]` | Returns column names. Populates the target dropdown. |
| `count_input_features(df, target) → int` | Returns `total_columns - 1`. Raises `ValueError` if target column not found. |

### Splitting

| Function | What it does |
|---|---|
| `split_data_percentage(df, target, ratio)` | Returns `(X_train, X_val, y_train, y_val)` via sklearn's `train_test_split`. |
| `get_kfold_splitter(k) → KFold` | Returns a configured `KFold` object (used later by the training loop in Phase 4). |

> **Note:** The split functions are fully implemented and tested, but only called during training (Phase 4). Window 1 saves the split **config** to `ProjectState`.

---

## 2. `ui/data_table_view.py` — DataFrame → QTableView

**Why it exists:** Qt can't display a Pandas DataFrame directly. This file bridges the gap with a minimal model + widget.

### `PandasTableModel(QAbstractTableModel)`

Qt model wrapping a DataFrame:

| Method | What it does |
|---|---|
| `rowCount()` | Returns `len(df)` |
| `columnCount()` | Returns `len(df.columns)` |
| `data(index, role)` | Returns `str(df.iloc[row, col])` for `DisplayRole` |
| `headerData(section, orientation, role)` | Horizontal → column name. Vertical → 1-indexed row number. |
| `update_dataframe(df, max_rows)` | Replaces data. Emits `beginResetModel()` / `endResetModel()`. |

### `DataPreviewTable(QWidget)`

Convenience wrapper:
- Creates a `QTableView` with alternating row colours and interactive column resizing
- `set_dataframe(df)` — shows first N rows, auto-resizes columns
- `clear()` — resets to empty

---

## 3. `ui/window_data.py` — Window 1 (Data Loading)

**Why it exists:** The first window the user sees. Handles everything from file selection to data validation before passing clean data to Window 2.

### UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  📊  Data Loading & Preprocessing                 (header)  │
│  [📂 Load CSV]  ✅ iris.csv                                 │
│  150 rows × 5 columns • NaN cells: 0              (info)   │
│ ┌─────────────────────────────────────────────────────────┐  │
│ │  sepal_len │ sepal_wid │ petal_len │ petal_wid │ class  │  │
│ │  5.1       │ 3.5       │ 1.4       │ 0.2       │ setosa │  │
│ │  4.9       │ 3.0       │ 1.4       │ 0.2       │ setosa │  │
│ │  ...       │ ...       │ ...       │ ...       │ ...    │  │
│ └─────────────────────────────────────────────────────────┘  │
│ ┌──── Target & Problem Type ─┐ ┌── Split & Cleaning ──────┐ │
│ │ Target: [class        ▼]   │ │ ○ Percentage  ○ K-Fold    │ │
│ │ ○ Classification  ○ Regr.  │ │ Train ratio: [0.80]       │ │
│ │                            │ │ NaN: [Fill with mean ▼]   │ │
│ │                            │ │ [🧹 Clean Data]           │ │
│ └────────────────────────────┘ └───────────────────────────┘ │
│                                   [Next → Model Builder]     │
└─────────────────────────────────────────────────────────────┘
```

### Functions

| Group | Function | What it does |
|---|---|---|
| **Setup** | `__init__(state, on_next)` | Stores state + callback. Inits `_raw_df` and `_cleaned_df` as `None`. |
| | `_build_ui()` | Assembles the full layout |
| **Panels** | `_build_target_panel()` | Target column dropdown + classification/regression radios |
| | `_build_split_panel()` | Percentage/k-fold toggle, ratio/k spinners, NaN dropdown, Clean button |
| | `_on_split_method_changed()` | Shows ratio spinner or k spinner |
| **Loading** | `_load_csv()` | File dialog → `load_csv()` → update preview, info, dropdown. All try/except. |
| **Cleaning** | `_clean_data()` | Strategy → `clean_dataframe()` → refresh preview → summary dialog |
| **Navigation** | `_on_next()` | 5-step validation chain, then writes to `ProjectState` |

### `_on_next()` Validation Chain

| Step | Check | On failure |
|---|---|---|
| 1 | Is a DataFrame loaded? | Warning: "Please load a CSV" |
| 2 | Are there remaining NaNs? | Asks user: auto-clean or go back? |
| 3 | Is the target column valid? | Warning: "Select a valid target" |
| 4 | Are there input features? | Warning: "No features (only target)" |
| 5 | All good | Writes to `ProjectState` and calls callback |

**What gets written to ProjectState:**
```python
self.state.dataframe     = df              # cleaned DataFrame
self.state.target_column = "class"
self.state.problem_type  = "classification"
self.state.split_config  = {"method": "percentage", "ratio": 0.8}
```

---

## 4. `main.py` — PipelineController (Updated)

Replaced the old direct-Window-2-launch with a controller managing transitions:

```python
class PipelineController:
    def start(self):               # Show Window 1 (Data)
    def _open_model_window(self):  # Hide W1, show Window 2
    def _back_to_data(self):       # Close W2, re-show W1
```

**Flow:** `start()` → Window 1 → "Next →" → `_open_model_window()` → Window 2 → "← Back" → `_back_to_data()` → Window 1

---

## 5. `ui/window_model.py` — Changes

Two additions:

**1. `on_back` parameter + ← Back button**
If `on_back` is provided, a "← Back to Data" button appears in the button bar.

**2. Dataset info label**
When `state.dataframe` is not `None`, a blue info line appears below the header:
```
📋  Dataset: 4 input features  •  Target: class  •  Problem: Classification
```

---

## 6. `tests/test_phase2.py` — 24 Tests

| Test Class | # | What it covers |
|---|---|---|
| `TestLoadCSV` | 5 | Valid CSV, missing file, empty file, header-only, CSV with NaNs |
| `TestCleaning` | 5 | Fill-mean numeric, drop-rows, no-NaN passthrough, categorical, immutability |
| `TestFeatures` | 3 | Column detection, feature count, invalid target error |
| `TestSplitting` | 3 | Percentage split shapes, invalid target, k-fold splitter |
| `TestPandasTableModel` | 5 | Row/column count, max_rows, display, headers, update |
| `TestDataPreviewTable` | 1 | Set dataframe + clear |
| `TestProjectStateData` | 2 | Input features with data, default split config |

---

## Data Flow: End-to-End (Phase 1 + 2)

```
User loads CSV in Window 1
      │
      ▼
load_csv() → raw DataFrame
      │
      ├──→ DataPreviewTable shows first 5 rows
      ├──→ detect_columns() → populate target dropdown
      ├──→ clean_dataframe() → cleaned DataFrame + report
      │
      └──→ _on_next() validates → writes to ProjectState
                                      │
                                      ▼
                             Window 2 opens
                             ├── shows data info label
                             └── user builds layers → blueprint
```

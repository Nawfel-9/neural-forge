# Phase 2 Walkthrough: Data Engineering

> Data loading, profiling, cleaning, feature engineering, preprocessing, and the Data Engineering Lab UI.

---

## Files

| File | Purpose |
|---|---|
| [`backend/data_handler.py`](../backend/data_handler.py) | Pure data backend: load/profile/clean/engineer/preprocess/split |
| [`workers/data_loader_worker.py`](../workers/data_loader_worker.py) | Background worker for long-running dataframe operations |
| [`ui/data_table_view.py`](../ui/data_table_view.py) | Pandas DataFrame to read-only `QTableView` bridge |
| [`ui/window_data.py`](../ui/window_data.py) | Scrollable Data Engineering Lab UI |
| [`tests/test_phase2.py`](../tests/test_phase2.py) | Compatibility and table-model tests |

---

## Backend API

### Loading

| Function | Purpose |
|---|---|
| `load_data(filepath)` | Loads CSV with separator auto-detection or Parquet via pandas |
| `load_csv(filepath, **kwargs)` | Compatibility CSV loader used by tests and older docs |
| `get_profile(df)` | Returns `describe(include="all")` plus NaN counts, NaN percent, dtypes, and numeric skewness |

### Cleaning

| Function | Purpose |
|---|---|
| `handle_nan(df, strategy, constant_val=None)` | Supports `drop`, `mean`, `median`, `mode`, `constant`, `knn` |
| `clean_dataframe(df, nan_strategy)` | Compatibility wrapper returning `(cleaned_df, report)` |
| `handle_outliers(df, columns, method, action)` | Clips or removes outliers using IQR or Z-score |

`NaNStrategy.FILL_MEAN` maps to numeric mean plus categorical mode. `NaNStrategy.DROP_ROWS` drops rows with any missing value.

### Feature Engineering and Preprocessing

| Function | Purpose |
|---|---|
| `cyclical_encode(df, column, max_val)` | Adds sine/cosine columns for periodic features |
| `add_lags(df, columns, n_lags)` | Adds lag columns and drops rows made incomplete by lagging |
| `parse_datetime_features(df, columns)` | Extracts year/month/day/day-of-week/weekend flags and drops original datetime columns |
| `apply_feature_interaction(df, col1, col2, op)` | Adds add/sub/mul/div interaction column |
| `apply_preprocessing(df, target, config)` | Applies feature exclusion, one-hot encoding, target encoding, scaling, optional PCA, and returns a `DataPipeline` |

`DataPipeline` stores fitted scalers, encoders, transformers, target encoder, final feature columns, target column, and optional PCA. It can be saved/loaded with pickle for deployment.

### Splitting

| Function | Purpose |
|---|---|
| `split_data_percentage(df, target, ratio)` | Returns `X_train, X_val, y_train, y_val` |
| `get_kfold_splitter(k)` | Returns a configured `KFold` |
| `split_data(df, target, config)` | Supports percentage train/val/test split or K-Fold object creation |
| `calculate_class_weights(y)` | Returns balanced class weights as a PyTorch tensor |

---

## Data Engineering Lab UI

`DataWindow` is a scrollable dashboard with these sections:

1. **Data Ingestion**: file picker for CSV/Parquet.
2. **Problem Definition**: target column and problem type.
3. **Raw Data Preview**: first rows in `DataPreviewTable`.
4. **Dataset Profile**: profile dataframe in `DataPreviewTable`.
5. **Cleaning & Outliers**: NaN strategy, outlier method/action, checked column list.
6. **Features & Scaling**: scaling method, PCA toggle, include/exclude feature list.
7. **Advanced Feature Engineering**: interaction features, lag features, datetime extraction.

Long-running operations call `_start_worker(...)`, which creates a `QThread`, moves a `DataLoaderWorker` into it, and updates the status/progress UI through signals.

---

## State Handoff

After `Apply Full Preprocessing Pipeline`, `DataWindow` writes:

```python
self.state.dataframe = self.df
self.state.target_column = self.combo_target.currentText()
self.state.problem_type = self.combo_problem_type.currentText()
self.state.pipeline = report.get("pipeline")
```

The next screen (`ModelBuilderWindow`) reads that state to show input-feature context and lock the output layer to the required class/regression output count.

"""
data_loader_worker.py
====================
Background worker for data engineering tasks without freezing the UI.
"""

from PyQt6.QtCore import QObject, pyqtSignal
from backend.data_handler import (
    load_data, get_profile, handle_nan, handle_outliers,
    cyclical_encode, add_lags, apply_preprocessing, split_data,
    calculate_class_weights, calculate_correlation_matrix, detect_target_leakage,
    apply_feature_interaction, validate_domain_constraints
)

class DataLoaderWorker(QObject):
    """
    Worker to handle data operations in a separate thread.
    """
    finished = pyqtSignal(object, dict)  # (result_object, report_dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, task_type: str, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == "load":
                path = self.kwargs.get("filepath")
                self.progress.emit(f"Loading {path}...")
                df = load_data(path)
                self.finished.emit(df, {"path": path})

            elif self.task_type == "profile":
                df = self.kwargs.get("df")
                self.progress.emit("Profiling data...")
                profile = get_profile(df)
                self.finished.emit(profile, {})

            elif self.task_type == "clean":
                df = self.kwargs.get("df")
                strategy = self.kwargs.get("strategy")
                const = self.kwargs.get("constant_val")
                self.progress.emit(f"Handling NaNs with strategy: {strategy}...")
                df = handle_nan(df, strategy, const)

                outlier_cols = self.kwargs.get("outlier_cols", [])
                if outlier_cols:
                    method = self.kwargs.get("outlier_method", "iqr")
                    action = self.kwargs.get("outlier_action", "clip")
                    self.progress.emit(f"Handling outliers ({method}, {action})...")
                    df = handle_outliers(df, outlier_cols, method, action)

                self.finished.emit(df, {})

            elif self.task_type == "engineer":
                df = self.kwargs.get("df")
                # Cyclical
                cyclical_cols = self.kwargs.get("cyclical_cols", [])
                for col, max_v in cyclical_cols:
                    self.progress.emit(f"Encoding cyclical: {col}...")
                    df = cyclical_encode(df, col, max_v)

                # Lags
                lag_cols = self.kwargs.get("lag_cols", [])
                n_lags = self.kwargs.get("n_lags", 0)
                if lag_cols and n_lags > 0:
                    self.progress.emit(f"Generating {n_lags} lags for {len(lag_cols)} columns...")
                    df = add_lags(df, lag_cols, n_lags)

                # Datetime
                dt_cols = self.kwargs.get("datetime_cols", [])
                if dt_cols:
                    from backend.data_handler import parse_datetime_features
                    self.progress.emit(f"Extracting datetime features from {len(dt_cols)} columns...")
                    df = parse_datetime_features(df, dt_cols)

                self.finished.emit(df, {})

            elif self.task_type == "preprocess":
                df = self.kwargs.get("df")
                target = self.kwargs.get("target")
                config = self.kwargs.get("config", {})
                self.progress.emit("Applying preprocessing pipeline...")
                df, pipeline = apply_preprocessing(df, target, config)
                self.finished.emit(df, {"pipeline": pipeline})

            elif self.task_type == "split":
                df = self.kwargs.get("df")
                target = self.kwargs.get("target")
                config = self.kwargs.get("config", {})
                self.progress.emit("Splitting data...")
                splits = split_data(df, target, config)

                # If classification, calculate weights as well
                weights = None
                if config.get("calculate_weights", False):
                    self.progress.emit("Calculating class weights...")
                    weights = calculate_class_weights(df[target])

                self.finished.emit(splits, {"weights": weights})

            elif self.task_type == "interaction":
                df = self.kwargs.get("df")
                col1 = self.kwargs.get("col1")
                col2 = self.kwargs.get("col2")
                op = self.kwargs.get("op")
                self.progress.emit(f"Creating interaction: {col1} {op} {col2}...")
                df = apply_feature_interaction(df, col1, col2, op)
                self.finished.emit(df, {})

            elif self.task_type == "correlation":
                df = self.kwargs.get("df")
                self.progress.emit("Calculating correlation matrix...")
                corr = calculate_correlation_matrix(df)
                self.finished.emit(corr, {})

            elif self.task_type == "validate":
                df = self.kwargs.get("df")
                constraints = self.kwargs.get("constraints", [])
                self.progress.emit("Validating domain constraints...")
                report = validate_domain_constraints(df, constraints)
                self.finished.emit(None, report)

            elif self.task_type == "leakage":
                df = self.kwargs.get("df")
                target = self.kwargs.get("target")
                threshold = self.kwargs.get("threshold", 0.95)
                self.progress.emit("Detecting target leakage...")
                leaky = detect_target_leakage(df, target, threshold)
                self.finished.emit(leaky, {})

        except Exception as e:
            self.error.emit(str(e))

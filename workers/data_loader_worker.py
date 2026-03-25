"""
data_loader_worker.py
====================
Background worker for loading and cleaning CSV files without freezing the UI.
"""

from PyQt6.QtCore import QObject, pyqtSignal
import pandas as pd
from backend.data_handler import load_csv, clean_dataframe, NaNStrategy

class DataLoaderWorker(QObject):
    """
    Worker to handle data operations in a separate thread.
    """
    finished = pyqtSignal(object, dict)  # (dataframe, report)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, task_type: str, filepath: str = "", df: pd.DataFrame = None, strategy: str = ""):
        super().__init__()
        self.task_type = task_type  # "load" or "clean"
        self.filepath = filepath
        self.df = df
        self.strategy = strategy

    def run(self):
        try:
            if self.task_type == "load":
                self.progress.emit(f"Loading {self.filepath}...")
                df = load_csv(self.filepath)
                self.finished.emit(df, {})
            elif self.task_type == "clean":
                if self.df is None:
                    raise ValueError("No dataframe provided for cleaning.")
                self.progress.emit(f"Cleaning data with strategy: {self.strategy}...")
                cleaned_df, report = clean_dataframe(self.df, self.strategy)
                self.finished.emit(cleaned_df, report)
        except Exception as e:
            self.error.emit(str(e))

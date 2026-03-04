"""
data_handler.py
===============
Pandas-based data loading, cleaning, and splitting.

This module is completely decoupled from the UI.  It receives file paths and
config dicts and returns DataFrames / metadata — never touches Qt.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str, **kwargs: Any) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame with robust error handling.

    Parameters
    ----------
    filepath : str
        Path to the ``.csv`` file.
    **kwargs
        Forwarded to :func:`pandas.read_csv` (e.g. ``encoding``, ``sep``).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pd.errors.EmptyDataError
        If the file is empty.
    pd.errors.ParserError
        If the CSV is malformed.
    ValueError
        If the file has no columns or is otherwise unusable.
    """
    df = pd.read_csv(filepath, **kwargs)
    if df.empty or df.shape[1] == 0:
        raise ValueError(
            f"The file '{filepath}' was loaded but contains no usable data."
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

class NaNStrategy:
    """Enum-like constants for NaN handling."""
    FILL_MEAN = "fill_mean"
    DROP_ROWS = "drop_rows"


def clean_dataframe(
    df: pd.DataFrame,
    nan_strategy: str = NaNStrategy.FILL_MEAN,
) -> tuple[pd.DataFrame, dict]:
    """
    Handle missing values and return a cleaned copy.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (not modified in place).
    nan_strategy : str
        ``"fill_mean"`` fills numeric NaNs with column means and
        non-numeric NaNs with the mode; ``"drop_rows"`` drops any row
        containing NaN.

    Returns
    -------
    (cleaned_df, report)
        ``report`` is a dict with keys ``nan_count_before``,
        ``nan_count_after``, ``rows_before``, ``rows_after``,
        ``strategy_used``.
    """
    report: dict[str, Any] = {
        "nan_count_before": int(df.isna().sum().sum()),
        "rows_before": len(df),
        "strategy_used": nan_strategy,
    }

    cleaned = df.copy()

    if nan_strategy == NaNStrategy.FILL_MEAN:
        # Numeric columns → fill with column mean
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if cleaned[col].isna().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())

        # Non-numeric columns → fill with mode (most frequent value)
        cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if cleaned[col].isna().any():
                mode_val = cleaned[col].mode()
                fill = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
                cleaned[col] = cleaned[col].fillna(fill)

    elif nan_strategy == NaNStrategy.DROP_ROWS:
        cleaned = cleaned.dropna().reset_index(drop=True)

    report["nan_count_after"] = int(cleaned.isna().sum().sum())
    report["rows_after"] = len(cleaned)
    return cleaned, report


# ─────────────────────────────────────────────────────────────────────────────
# Feature detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of column names."""
    return list(df.columns)


def count_input_features(df: pd.DataFrame, target_column: str) -> int:
    """
    Count the number of input features (all columns except the target).

    Raises
    ------
    ValueError
        If the target column is not found.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    return df.shape[1] - 1


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_data_percentage(
    df: pd.DataFrame,
    target_column: str,
    ratio: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train/validation sets by percentage.

    Returns
    -------
    (X_train, X_val, y_train, y_val)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=ratio, random_state=random_state,
    )
    return X_train, X_val, y_train, y_val


def get_kfold_splitter(k: int = 5, shuffle: bool = True, random_state: int = 42) -> KFold:
    """
    Return a configured KFold splitter (used by the training loop later).

    Parameters
    ----------
    k : int
        Number of folds.

    Returns
    -------
    sklearn.model_selection.KFold
    """
    return KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

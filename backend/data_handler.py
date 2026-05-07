"""
data_handler.py
===============
Exhaustive data engineering backend for Neural Forge.
Handles loading, profiling, cleaning, outliers, feature engineering, splitting, and export.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder,
    PowerTransformer, KBinsDiscretizer
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Loading & Profiling
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV (auto-separator) or Parquet."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        # sep=None with engine='python' enables auto-detection
        df = pd.read_csv(filepath, sep=None, engine='python')
    elif ext == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
    return df

def get_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return statistical profile of the dataframe."""
    stats = df.describe(include='all').T
    stats['nan_pct'] = (df.isna().sum() / len(df)) * 100
    stats['dtype'] = df.dtypes
    
    # Skewness for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats['skewness'] = np.nan
    for col in numeric_cols:
        stats.at[col, 'skewness'] = df[col].skew()
        
    return stats

# ─────────────────────────────────────────────────────────────────────────────
# Cleaning & Outliers
# ─────────────────────────────────────────────────────────────────────────────

def handle_nan(df: pd.DataFrame, strategy: str, constant_val: Any = None) -> pd.DataFrame:
    """Handle missing values."""
    df = df.copy()
    if strategy == 'drop':
        return df.dropna().reset_index(drop=True)
    
    for col in df.columns:
        if df[col].isna().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
            elif strategy == 'constant':
                df[col] = df[col].fillna(constant_val)
    return df

def handle_outliers(df: pd.DataFrame, columns: list[str], method: str = 'iqr', action: str = 'clip') -> pd.DataFrame:
    """Detect and handle outliers using IQR or Z-Score."""
    df = df.copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
        else: # z-score
            mean = df[col].mean()
            std = df[col].std()
            lower = mean - 3 * std
            upper = mean + 3 * std
            
        if action == 'clip':
            df[col] = df[col].clip(lower, upper)
        else: # remove
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def cyclical_encode(df: pd.DataFrame, column: str, max_val: float) -> pd.DataFrame:
    """Encode periodic features (hour, day) into sin/cos components."""
    df = df.copy()
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_val)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_val)
    return df

def add_lags(df: pd.DataFrame, columns: list[str], n_lags: int) -> pd.DataFrame:
    """Create lag features for time-series."""
    df = df.copy()
    for col in columns:
        for i in range(1, n_lags + 1):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    return df.dropna().reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class DataPipeline:
    """Encapsulates all transformations for reproduction in production."""
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.transformers = {}
        self.target_encoder = None
        self.feature_columns = []
        self.target_column = ""
        self.pca = None

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> DataPipeline:
        with open(path, 'rb') as f:
            return pickle.load(f)

def apply_preprocessing(df: pd.DataFrame, target: str, config: dict) -> Tuple[pd.DataFrame, DataPipeline]:
    """Apply scaling, encoding, and distribution transforms."""
    df = df.copy()
    pipeline = DataPipeline()
    pipeline.target_column = target
    
    # 1. Feature Selection
    excluded = config.get('exclude_columns', [])
    feature_cols = [c for c in df.columns if c != target and c not in excluded]
    pipeline.feature_columns = feature_cols
    
    # 2. Categorical Encoding
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = enc.fit_transform(df[[col]])
        new_cols = [f"{col}_{cat}" for cat in enc.categories_[0]]
        encoded_df = pd.DataFrame(encoded_data, columns=new_cols, index=df.index)
        df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
        pipeline.encoders[col] = enc
        
    # Update feature list after encoding
    feature_cols = [c for c in df.columns if c != target]
    
    # 3. Target Encoding (if classification and string)
    if not pd.api.types.is_numeric_dtype(df[target]):
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        pipeline.target_encoder = le

    # 4. Distribution Transforms
    power_cols = config.get('power_transform_columns', [])
    for col in power_cols:
        if col in df.columns:
            pt = PowerTransformer(method='yeo-johnson')
            df[col] = pt.fit_transform(df[[col]])
            pipeline.transformers[f'power_{col}'] = pt

    # 5. Scaling
    scale_method = config.get('scaling', 'standard')
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    if scale_method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    if len(num_cols) > 0:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        pipeline.scalers['main'] = scaler

    # 6. PCA (Dimensionality Reduction)
    if config.get('pca_enabled', False):
        n_comp = config.get('pca_components', 0.95)
        pca = PCA(n_components=n_comp)
        # Apply PCA only to numeric features
        num_features = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
        pca_data = pca.fit_transform(df[num_features])
        new_cols = [f"pca_comp_{i}" for i in range(pca_data.shape[1])]
        pca_df = pd.DataFrame(pca_data, columns=new_cols, index=df.index)
        
        # Drop old numeric features and keep target/others
        others = [c for c in df.columns if c not in num_features]
        df = pd.concat([df[others], pca_df], axis=1)
        pipeline.pca = pca

    return df, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Advanced Feature Discovery & Validation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix for numeric columns."""
    return df.select_dtypes(include=[np.number]).corr()

def detect_target_leakage(df: pd.DataFrame, target: str, threshold: float = 0.95) -> list[str]:
    """Identify columns with suspiciously high correlation to the target."""
    if not pd.api.types.is_numeric_dtype(df[target]):
        return []
    
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    target_corr = corr_matrix[target].abs().sort_values(ascending=False)
    # Exclude target itself and return leaky ones
    leaky = target_corr[(target_corr > threshold) & (target_corr.index != target)].index.tolist()
    return leaky

def apply_feature_interaction(df: pd.DataFrame, col1: str, col2: str, op: str) -> pd.DataFrame:
    """Create a new feature via mathematical interaction."""
    df = df.copy()
    new_name = f"{col1}_{op}_{col2}"
    if op == 'add':
        df[new_name] = df[col1] + df[col2]
    elif op == 'sub':
        df[new_name] = df[col1] - df[col2]
    elif op == 'mul':
        df[new_name] = df[col1] * df[col2]
    elif op == 'div':
        df[new_name] = df[col1] / df[col2].replace(0, np.nan)
    return df

def validate_domain_constraints(df: pd.DataFrame, constraints: list[dict]) -> dict:
    """
    Check domain rules. 
    Constraint example: {'column': 'pH', 'op': 'range', 'min': 0, 'max': 14}
    """
    report = {"errors": [], "success": True}
    for c in constraints:
        col = c['column']
        if col not in df.columns: continue
        
        op = c['op']
        if op == 'greater':
            invalid = df[df[col] <= c['val']]
        elif op == 'less':
            invalid = df[df[col] >= c['val']]
        elif op == 'range':
            invalid = df[(df[col] < c['min']) | (df[col] > c['max'])]
            
        if not invalid.empty:
            report["errors"].append(f"Constraint {op} failed for {col}: {len(invalid)} violations.")
            report["success"] = False
            
    return report

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility Shims
# ─────────────────────────────────────────────────────────────────────────────

def get_kfold_splitter(k: int = 5, shuffle: bool = True, random_state: int = 42) -> KFold:
    return KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

def split_data_percentage(
    df: pd.DataFrame,
    target_column: str,
    ratio: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, train_size=ratio, random_state=random_state)

def detect_columns(df: pd.DataFrame) -> list[str]:
    return list(df.columns)

def count_input_features(df: pd.DataFrame, target_column: str) -> int:
    return df.shape[1] - 1

# ─────────────────────────────────────────────────────────────────────────────
# Splitting & Export
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, target: str, config: dict) -> Dict[str, Any]:
    """Split data with support for stratification and resampling."""
    X = df.drop(columns=[target])
    y = df[target]
    
    method = config.get('method', 'percentage')
    stratify = y if config.get('stratify', False) else None
    
    if method == 'percentage':
        test_size = config.get('test_size', 0.2)
        val_size = config.get('val_size', 0.1)
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=42
        )
        
        # Further split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size/(1-test_size), 
            stratify=y_train_full if stratify is not None else None, random_state=42
        )
        
        # Resampling (only on training set)
        resample = config.get('resample', None)
        if resample and IMBLEARN_AVAILABLE:
            if resample == 'smote':
                X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            elif resample == 'undersample':
                X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    else: # K-Fold
        k = config.get('k', 5)
        kf = StratifiedKFold(n_splits=k) if stratify is not None else KFold(n_splits=k)
        return {'kfold': kf, 'X': X, 'y': y}

def calculate_class_weights(y: pd.Series) -> torch.Tensor:
    """Calculate weights for imbalanced classification."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)

def to_dataloader(X: pd.DataFrame, y: pd.Series, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Convert Pandas data to PyTorch DataLoader."""
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long if y.dtype == int else torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

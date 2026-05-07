import os
import pandas as pd
import numpy as np
import torch
from backend.data_handler import (
    load_data, get_profile, handle_nan, handle_outliers,
    cyclical_encode, add_lags, apply_preprocessing, split_data,
    calculate_class_weights
)

def test_backend():
    print("Testing backend logic...")
    
    # Create synthetic data
    data = {
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 150, np.nan], # 150 is outlier
        'income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000],
        'hour': [1, 12, 23, 1, 12, 23, 1, 12, 23, 1, 12, 23],
        'target': ['cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', 'dog']
    }
    df = pd.DataFrame(data)
    
    # 1. Profile
    profile = get_profile(df)
    print("Profile generated.")
    assert 'nan_pct' in profile.columns
    
    # 2. Clean
    df_clean = handle_nan(df, 'mean')
    print("NaNs handled.")
    assert df_clean['age'].isna().sum() == 0
    
    # 3. Outliers
    df_out = handle_outliers(df_clean, ['age'], method='iqr', action='clip')
    print("Outliers handled.")
    assert df_out['age'].max() < 150
    
    # 4. Engineering
    df_cyc = cyclical_encode(df_out, 'hour', 24)
    print("Cyclical encoding done.")
    assert 'hour_sin' in df_cyc.columns
    
    df_lag = add_lags(df_cyc, ['income'], 2)
    print("Lag features generated.")
    assert 'income_lag_1' in df_lag.columns
    
    # 5. Preprocessing
    config = {'scaling': 'standard', 'exclude_columns': ['hour']}
    df_pre, pipeline = apply_preprocessing(df_lag, 'target', config)
    print("Preprocessing applied.")
    assert 'target' in df_pre.columns
    assert isinstance(df_pre['target'].iloc[0], (int, np.integer))
    
    # 6. Splitting
    split_config = {'method': 'percentage', 'test_size': 0.2, 'val_size': 0.2, 'stratify': True}
    splits = split_data(df_pre, 'target', split_config)
    print("Splitting done.")
    assert 'train' in splits
    assert 'val' in splits
    assert 'test' in splits
    
    # 7. Class Weights
    weights = calculate_class_weights(df_pre['target'])
    print("Class weights calculated.")
    assert torch.is_tensor(weights)
    
    print("All backend tests passed!")

if __name__ == "__main__":
    test_backend()

# Data Preprocessing & Engineering — Walkthrough

> Automated feature discovery, leakage detection, and PCA-driven dimensionality reduction.

---

## Files Overview

| Status | File | Purpose |
|---|---|---|
| **Modified** | [`ui/window_data.py`](../ui/window_data.py) | High-density UI with FormLayouts and Segmented Tabs |
| **Core** | [`backend/data_handler.py`](../backend/data_handler.py) | Scikit-learn powered preprocessing logic & `DataPipeline` |
| **Logic** | [`workers/data_loader_worker.py`](../workers/data_loader_worker.py) | Multi-threaded execution of heavy data tasks |

---

## 1. Target & Problem Type

The preprocessing pipeline begins by defining the objective of the model.

- **Target Column**: The outcome variable (y) that the model will predict.
- **Problem Type**: 
    - `Classification`: Encodes targets via `LabelEncoder` if they are non-numeric.
    - `Regression`: Maintains raw numeric values for continuous prediction.

```python
# From backend/data_handler.py
if not pd.api.types.is_numeric_dtype(df[target]):
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    pipeline.target_encoder = le
```

---

## 2. Global Scaling

Neural networks are sensitive to input scales. The platform provides two standard options:

| Method | Formula | Use Case |
|---|---|---|
| **Standard** | `(x - μ) / σ` | Most common; assumes normally distributed data. |
| **MinMax** | `(x - min) / (max - min)` | Useful for image data or when preserving 0 values. |

Scaling is applied **only to feature columns**, and the parameters are stored in the `DataPipeline` for inference-time reproducibility.

---

## 3. Feature Selection & Discovery

### Interactive Selection
Users can manually exclude columns (e.g., IDs, Names, or timestamps that aren't useful as raw features). 

### Target Leakage Detection
Automatically identifies features that have a suspiciously high correlation (`> 0.95`) with the target. These features often "leak" future information and should be removed to ensure model generalization.

```python
def detect_target_leakage(df, target, threshold=0.95):
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    target_corr = corr_matrix[target].abs()
    return target_corr[(target_corr > threshold) & (target_corr.index != target)].index
```

---

## 4. Dimensionality Reduction (PCA)

When dealing with high-dimensional datasets, **Principal Component Analysis (PCA)** is used to compress the feature space while retaining a specific percentage of variance (default: `95%`).

- **Input**: All numeric features after scaling.
- **Output**: A reduced set of synthetic "Principal Components".
- **Benefit**: Reduces training time and prevents overfitting on noise.

---

## 5. Domain Validation (Sanity Checks)

Before proceeding to model building, the data can be validated against custom domain constraints to ensure data integrity.

- **Range Checks**: Ensuring variables like `pH` are within `[0, 14]`.
- **Positivity Checks**: Ensuring `Price` or `Age` are not negative.

```python
# Report format returned to UI
{
    "errors": ["Constraint greater failed for age: 12 violations."],
    "success": False
}
```

---

## 6. The `DataPipeline` Object

The `DataPipeline` class is a critical architecture component. It acts as a "black box" that captures every transformation applied during the engineering lab.

```python
class DataPipeline:
    def __init__(self):
        self.scalers = {}      # Fitted StandardScaler/MinMaxScaler
        self.encoders = {}     # Fitted OneHotEncoders
        self.transformers = {}  # PowerTransformers (Yeo-Johnson)
        self.pca = None        # Fitted PCA instance
```

**Reproduction:** By saving this object as a `.pkl` file, users can apply the *exact same* transformations to new, unseen data in production without re-fitting.


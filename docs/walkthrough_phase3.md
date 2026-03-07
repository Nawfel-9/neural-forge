# Phase 3 — Code Walkthrough

> Blueprint → PyTorch `nn.Sequential`, LazyLinear, and ghost-run validation.

---

## Files Overview

| Status | File | Purpose |
|---|---|---|
| **NEW** | [`backend/model_builder.py`](../backend/model_builder.py) | Blueprint → `nn.Sequential` + ghost run |
| **NEW** | [`tests/test_phase3.py`](../tests/test_phase3.py) | 28 automated tests |
| Modified | [`ui/window_model.py`](../ui/window_model.py) | Added 🔨 Build & Test button + updated `sync_to_state` |

---

## 1. `backend/model_builder.py` — The Translation Engine

**Why it exists:** This is the bridge between the visual UI and PyTorch. It takes a blueprint (list of dicts produced by the layer builder) and turns it into a real `nn.Sequential` model. **Never touches Qt** — pure PyTorch backend.

### The big picture

```
Blueprint (list[dict])
        │
        ▼
  build_model()           ← translates each layer config → nn.Module
        │
        ▼
  nn.Sequential           ← un-initialised (Lazy* modules)
        │
        ▼
  ghost_run()             ← dummy forward pass materialises all Lazy modules
        │
        ▼
  Validated model         ← ready for training
```

---

### Activation Factory

```python
_ACTIVATION_MAP = {
    "ReLU":      nn.ReLU,
    "Sigmoid":   nn.Sigmoid,
    "Tanh":      nn.Tanh,
    "LeakyReLU": nn.LeakyReLU,
    "Softmax":   nn.Softmax,
}
```

| Function | What it does |
|---|---|
| `_make_activation(name) → nn.Module \| None` | Returns an activation instance. `"None"` → returns `None` (no activation appended). `"Softmax"` → uses `dim=-1`. |

---

### Layer Translator

| Function | What it does |
|---|---|
| `_translate_layer(layer_cfg, prev_out, is_first_linear) → (modules, new_out)` | Converts one blueprint dict into 1–2 `nn.Module` objects |

This is where the magic happens. For each layer type:

| Layer Type | What it creates | Input dim handling |
|---|---|---|
| **Linear** | `nn.LazyLinear(neurons)` or `nn.Linear(prev, neurons)` + optional activation | First Linear uses `LazyLinear` (auto-infers). Subsequent use explicit `in_features`. |
| **Conv1d** | `nn.LazyConv1d(...)` or `nn.Conv1d(...)` | First Conv1d uses `LazyConv1d`. Subsequent use explicit `in_channels`. |
| **MaxPool1d** | `nn.MaxPool1d(kernel, stride)` | Channels unchanged, spatial dims shrink |
| **AvgPool1d** | `nn.AvgPool1d(kernel, stride)` | Same as MaxPool1d |
| **Flatten** | `nn.Flatten()` | Sets `new_out = None` (unknown until ghost run) |
| **BatchNorm1d** | `nn.BatchNorm1d(prev)` or `nn.LazyBatchNorm1d()` | Uses `LazyBatchNorm1d` if channel count unknown |
| **Dropout** | `nn.Dropout(p=rate)` | Pass-through, dims unchanged |

### Why LazyLinear / LazyConv1d?

The first layer doesn't know its input size until we see real (or dummy) data. Instead of forcing the user to calculate it, we use PyTorch's `Lazy*` modules:

```python
# Before ghost run:
LazyLinear(out_features=64)   # in_features = ???

# After ghost_run() with dummy tensor (batch=2, features=10):
Linear(in_features=10, out_features=64)  # auto-inferred!
```

This is why the ghost run is critical — it materialises all Lazy modules.

---

### `build_model(blueprint) → nn.Sequential`

Loops through the blueprint, calling `_translate_layer()` for each entry, and assembles all modules into a single `nn.Sequential`:

```python
def build_model(blueprint):
    all_modules = []
    prev_out = None
    first_linear_seen = False

    for cfg in blueprint:
        is_first = (cfg["type"] == "Linear" and not first_linear_seen)
        modules, prev_out = _translate_layer(cfg, prev_out, is_first)
        all_modules.extend(modules)
        if cfg["type"] == "Linear":
            first_linear_seen = True

    return nn.Sequential(*all_modules)
```

**Key decisions:**
- `prev_out` tracks the output feature count as we walk through layers
- Only the **first** Linear uses `LazyLinear` — subsequent ones know their `in_features` from the previous layer's `neurons`
- After `Flatten`, `prev_out` resets to `None` (size unknown until ghost run)

---

### `ghost_run(model, input_features, batch_size=2) → (success, output, message)`

| Step | What happens |
|---|---|
| 1 | Create a dummy tensor: `torch.randn(batch_size, input_features)` |
| 2 | Set model to eval mode: `model.eval()` |
| 3 | Forward pass with `torch.no_grad()` |
| 4 | If success: return output shape info |
| 5 | If exception: catch it and return a friendly error message |

**Why batch_size=2?** Using 2 instead of 1 helps catch BatchNorm issues (BatchNorm1d requires batch > 1 for calculating running stats).

**Example success message:**
```
Ghost run passed! Input: (2, 10) → Output: (2, 5)
```

**Example failure message:**
```
Ghost run failed: RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x128 and 64x10)
```

---

### `build_and_validate(blueprint, input_features) → (model, output, success, message)`

Convenience function that chains:
1. `build_model(blueprint)` → catch build errors
2. `ghost_run(model, input_features)` → catch shape errors

Returns all the pieces needed by the UI.

---

## 2. `ui/window_model.py` — Changes

### New: 🔨 Build & Test button

Added to the button bar next to ✅ Validate:

```
[← Back] [＋ Add] [💾 Save] [📂 Load] [✅ Validate] [🔨 Build & Test]
```

### `_build_and_test()`

| Step | What happens |
|---|---|
| 1 | Extract blueprint from UI via `get_architecture()` |
| 2 | Validate blueprint |
| 3 | Check `state.input_features() > 0` (needs data loaded) |
| 4 | Call `build_and_validate(blueprint, n_features)` |
| 5 | On success: save model + blueprint + dummy to `ProjectState`, show info dialog with model summary |
| 6 | On failure: show warning with the error message |

### Updated: `sync_to_state()`

Previously just validated + wrote the blueprint. Now does the full cycle:

```
validate_blueprint() → check features → build_and_validate() → write to state
```

This ensures that by the time the user reaches Window 3, the model is already built and validated.

---

## 3. Data Flow: Blueprint → Model

```
User configures layers in Window 2
        │
        ▼
get_architecture()  →  blueprint (list[dict])
        │
        ├──→ validate_blueprint()    →  check rules (10 rules)
        │
        ├──→ build_model(blueprint)  →  nn.Sequential with Lazy* modules
        │
        ├──→ ghost_run(model, n)     →  dummy forward pass
        │         │
        │         ├── success: materialises Lazy modules, returns output shape
        │         └── failure: returns friendly error message
        │
        └──→ state.model             →  ready for training (Phase 4)
             state.blueprint         →  saved for reference
             state.dummy_tensor      →  saved for ONNX export (Phase 5)
```

---

## 4. PyTorch Translation Examples

### Example 1: Simple classifier

**Blueprint:**
```json
[
  {"type": "Linear", "neurons": 64, "activation": "ReLU"},
  {"type": "Dropout", "rate": 0.3},
  {"type": "Linear", "neurons": 10, "activation": "None"}
]
```

**Resulting `nn.Sequential`:**
```
Sequential(
  (0): LazyLinear(out_features=64, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.3)
  (3): Linear(in_features=64, out_features=10, bias=True)
)
```

After ghost run with `input_features=20`:
```
Sequential(
  (0): Linear(in_features=20, out_features=64, bias=True)   ← LazyLinear materialised
  (1): ReLU()
  (2): Dropout(p=0.3)
  (3): Linear(in_features=64, out_features=10, bias=True)
)
```

### Example 2: Conv1d pipeline

**Blueprint:**
```json
[
  {"type": "Conv1d", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
  {"type": "MaxPool1d", "kernel_size": 2, "stride": 2},
  {"type": "Flatten"},
  {"type": "Linear", "neurons": 5, "activation": "None"}
]
```

**Resulting `nn.Sequential`:**
```
Sequential(
  (0): LazyConv1d(0, 16, kernel_size=(3,), stride=(1,), padding=(1,))
  (1): MaxPool1d(kernel_size=2, stride=2)
  (2): Flatten()
  (3): LazyLinear(out_features=5, bias=True)
)
```

After ghost run: LazyConv1d infers `in_channels`, LazyLinear infers `in_features` from the flattened conv output.

---

## 5. `tests/test_phase3.py` — 28 Tests

| Test Class | # | What it covers |
|---|---|---|
| `TestActivationFactory` | 6 | All activations: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, None |
| `TestBuildLinear` | 4 | Single linear, multi-linear, linear+dropout, linear+batchnorm |
| `TestBuildConv` | 3 | Conv1d basic, conv+maxpool, conv+avgpool |
| `TestGhostRunLinear` | 3 | Simple forward pass, deep network, all activations loop |
| `TestGhostRunConv` | 2 | Conv→Flatten→Linear, Conv→Pool→Flatten→Linear |
| `TestGhostRunFailures` | 1 | Negative features gracefully caught |
| `TestBuildAndValidate` | 3 | Success, bad blueprint, unknown type |
| `TestValidatorNewTypes` | 6 | Conv1d valid/invalid, MaxPool valid, AvgPool invalid, Flatten valid/invalid-last |

---

## 6. Error Handling Philosophy

Every function that can fail returns structured feedback:

| Function | Success return | Failure return |
|---|---|---|
| `ghost_run()` | `(True, output_tensor, "Ghost run passed! ...")` | `(False, None, "Ghost run failed: ...")` |
| `build_and_validate()` | `(model, output, True, msg)` | `(None, None, False, msg)` |

The UI catches these and shows `QMessageBox.information` / `QMessageBox.warning` — the app **never crashes** on a bad architecture.

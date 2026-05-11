"""
backend/dev_trainer.py
======================
QThread that drives the Developer Mode training loop.

Contract expected from the user's project:
  model.py    → get_model(config: dict)  → torch.nn.Module
  dataset.py  → get_dataloader(config: dict, split: str) → DataLoader
  loss.py*    → get_loss(config: dict)   → torch.nn.Module        (* optional)
  metrics.py* → get_metrics(config: dict)→ dict[str, callable]    (* optional)

Default losses (when loss.py is absent):
  classification → nn.CrossEntropyLoss()
  segmentation   → DiceBCELoss (defined below)

Default metrics (when metrics.py is absent):
  classification → {"accuracy": <fn>}
  segmentation   → {"iou": <fn>, "dice": <fn>}
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PyQt6.QtCore import QThread, pyqtSignal

from utils.config_schema import DevProjectConfig


# ══════════════════════════════════════════════════════════════════════════════
#  Built-in loss functions
# ══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        )
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Dice + Binary Cross-Entropy — common for segmentation."""
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets.float()) + self.dice(logits, targets)


# ══════════════════════════════════════════════════════════════════════════════
#  Built-in metric functions
# ══════════════════════════════════════════════════════════════════════════════

def _accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _iou(outputs: torch.Tensor, targets: torch.Tensor,
         threshold: float = 0.5) -> float:
    preds   = (torch.sigmoid(outputs) > threshold).float()
    targets = targets.float()
    inter   = (preds * targets).sum().item()
    union   = (preds + targets).clamp(0, 1).sum().item()
    return inter / (union + 1e-6)


def _dice(outputs: torch.Tensor, targets: torch.Tensor,
          threshold: float = 0.5) -> float:
    preds   = (torch.sigmoid(outputs) > threshold).float()
    targets = targets.float()
    inter   = (preds * targets).sum().item()
    return (2.0 * inter) / (preds.sum().item() + targets.sum().item() + 1e-6)


DEFAULT_LOSSES = {
    "classification": lambda: nn.CrossEntropyLoss(),
    "segmentation":   lambda: DiceBCELoss(),
}

DEFAULT_METRICS = {
    "classification": {"accuracy": _accuracy},
    "segmentation":   {"iou": _iou, "dice": _dice},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Dynamic import helper
# ══════════════════════════════════════════════════════════════════════════════

def _load_module(project_root: Path, filename: str):
    """Dynamically import a .py file from the project root. Returns module or None."""
    path = project_root / filename
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod  = importlib.util.module_from_spec(spec)
    # Ensure project root is on sys.path so relative imports inside work
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
#  Trainer QThread
# ══════════════════════════════════════════════════════════════════════════════

class DevTrainer(QThread):
    # ── Signals ───────────────────────────────────────────────────────────
    epoch_completed  = pyqtSignal(int, float, dict)   # epoch, loss, {metric: value}
    val_completed    = pyqtSignal(int, float, dict)   # epoch, val_loss, {metric: value}
    training_started = pyqtSignal(int)                # total_epochs
    training_done    = pyqtSignal(str)                # finish message
    training_error   = pyqtSignal(str)                # error message
    status_changed   = pyqtSignal(str)                # human-readable status text
    paused_by_temp   = pyqtSignal(float)              # emitted with current temp
    resumed_by_temp  = pyqtSignal(float)              # emitted when cool enough

    def __init__(self, project_root: str, config: DevProjectConfig, parent=None):
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.config       = config

        self._pause_requested  = False
        self._stop_requested   = False
        self._thermal_paused   = False
        self._current_gpu_temp = 0.0   # updated by hardware monitor via slot

    # ------------------------------------------------------------------
    # Control slots (called from UI thread)
    # ------------------------------------------------------------------
    def pause(self):
        self._pause_requested = True
        self.status_changed.emit("Paused by user")

    def resume(self):
        self._pause_requested = False
        self.status_changed.emit("Resuming…")

    def stop(self):
        self._stop_requested  = True
        self._pause_requested = False   # unblock wait loop
        self.status_changed.emit("Stopping…")

    def update_gpu_temp(self, temp: float):
        """Called by hardware monitor signal to keep trainer aware of temperature."""
        self._current_gpu_temp = temp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _wait_if_paused(self):
        """Block (with 100 ms sleep) while paused or thermally throttled."""
        while (self._pause_requested or self._thermal_paused) and not self._stop_requested:
            time.sleep(0.1)

    def _check_thermal(self):
        """Auto-pause / auto-resume based on GPU temperature."""
        temp = self._current_gpu_temp
        if not self._thermal_paused and temp >= self.config.auto_pause_temp:
            self._thermal_paused = True
            self.paused_by_temp.emit(temp)
            self.status_changed.emit(
                f"⚠ Thermal throttle: GPU {temp:.0f}°C ≥ {self.config.auto_pause_temp}°C — paused"
            )
        elif self._thermal_paused and temp < self.config.resume_temp:
            self._thermal_paused = False
            self.resumed_by_temp.emit(temp)
            self.status_changed.emit(
                f"✓ GPU cooled to {temp:.0f}°C — resuming"
            )

    def _load_components(self):
        """Dynamically load model, dataloader, loss, metrics from project."""
        cfg_dict = self.config.to_dict()

        # ── Model ──────────────────────────────────────────────────────
        model_mod = _load_module(self.project_root, "model.py")
        if model_mod is None or not hasattr(model_mod, "get_model"):
            raise ImportError("model.py not found or missing get_model(config).")
        model = model_mod.get_model(cfg_dict)

        # ── DataLoaders ────────────────────────────────────────────────
        dataset_mod = _load_module(self.project_root, "dataset.py")
        if dataset_mod is None or not hasattr(dataset_mod, "get_dataloader"):
            raise ImportError("dataset.py not found or missing get_dataloader(config, split).")
        train_loader: DataLoader = dataset_mod.get_dataloader(cfg_dict, "train")
        val_loader:   DataLoader = dataset_mod.get_dataloader(cfg_dict, "val")

        # ── Loss ───────────────────────────────────────────────────────
        loss_mod = _load_module(self.project_root, "loss.py")
        if loss_mod is not None and hasattr(loss_mod, "get_loss"):
            criterion = loss_mod.get_loss(cfg_dict)
            self.status_changed.emit("Using custom loss from loss.py")
        else:
            criterion = DEFAULT_LOSSES[self.config.task]()
            self.status_changed.emit(
                f"No loss.py found — using default: {self.config.effective_loss()}"
            )

        # ── Metrics ────────────────────────────────────────────────────
        metrics_mod = _load_module(self.project_root, "metrics.py")
        if metrics_mod is not None and hasattr(metrics_mod, "get_metrics"):
            metric_fns: dict = metrics_mod.get_metrics(cfg_dict)
            self.status_changed.emit("Using custom metrics from metrics.py")
        else:
            metric_fns = DEFAULT_METRICS[self.config.task]
            self.status_changed.emit(
                f"No metrics.py found — using default: {self.config.effective_metrics()}"
            )

        return model, train_loader, val_loader, criterion, metric_fns

    def _build_optimizer(self, model: nn.Module):
        params = model.parameters()
        opt = self.config.optimizer.lower()
        lr  = self.config.learning_rate
        if opt == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        return torch.optim.Adam(params, lr=lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def run(self):
        try:
            self.status_changed.emit("Loading project components…")
            model, train_loader, val_loader, criterion, metric_fns = \
                self._load_components()

            device = torch.device(
                self.config.device if torch.cuda.is_available() else "cpu"
            )
            model     = model.to(device)
            criterion = criterion.to(device)
            optimizer = self._build_optimizer(model)

            # Checkpoint dir
            ckpt_dir = self.project_root / self.config.checkpoint_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            total_epochs = self.config.epochs
            self.training_started.emit(total_epochs)
            self.status_changed.emit(f"Training on {device} — {total_epochs} epochs")

            for epoch in range(1, total_epochs + 1):

                if self._stop_requested:
                    break

                # ── Thermal check ───────────────────────────────────────
                self._check_thermal()
                self._wait_if_paused()
                if self._stop_requested:
                    break

                # ── Train phase ─────────────────────────────────────────
                model.train()
                train_loss   = 0.0
                train_metrics = {k: 0.0 for k in metric_fns}
                n_batches    = 0

                for inputs, targets in train_loader:
                    if self._stop_requested:
                        break
                    self._check_thermal()
                    self._wait_if_paused()

                    inputs  = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss    = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    with torch.no_grad():
                        for name, fn in metric_fns.items():
                            train_metrics[name] += fn(outputs, targets)
                    n_batches += 1

                if n_batches == 0:
                    continue

                avg_train_loss    = train_loss / n_batches
                avg_train_metrics = {k: v / n_batches for k, v in train_metrics.items()}
                self.epoch_completed.emit(epoch, avg_train_loss, avg_train_metrics)

                # ── Validation phase ────────────────────────────────────
                model.eval()
                val_loss    = 0.0
                val_metrics = {k: 0.0 for k in metric_fns}
                n_val       = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs  = inputs.to(device)
                        targets = targets.to(device)
                        outputs = model(inputs)
                        loss    = criterion(outputs, targets)
                        val_loss += loss.item()
                        for name, fn in metric_fns.items():
                            val_metrics[name] += fn(outputs, targets)
                        n_val += 1

                if n_val > 0:
                    avg_val_loss    = val_loss / n_val
                    avg_val_metrics = {k: v / n_val for k, v in val_metrics.items()}
                    self.val_completed.emit(epoch, avg_val_loss, avg_val_metrics)

                # ── Checkpoint every 5 epochs ───────────────────────────
                if epoch % 5 == 0:
                    ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
                    torch.save({
                        "epoch":       epoch,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "config":      self.config.to_dict(),
                    }, ckpt_path)
                    self.status_changed.emit(f"Checkpoint saved → {ckpt_path.name}")

                self.status_changed.emit(
                    f"Epoch {epoch}/{total_epochs} — "
                    f"loss {avg_train_loss:.4f}"
                )

            if self._stop_requested:
                self.training_done.emit("Training stopped by user.")
            else:
                self.training_done.emit(
                    f"Training complete — {total_epochs} epochs finished."
                )

        except Exception as exc:
            import traceback
            self.training_error.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )

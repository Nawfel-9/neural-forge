"""
training_worker.py
==================
Multithreaded training loop using PyQt6 QThread and PyTorch.
Calculates advanced evaluation metrics (F1, ROC-AUC, R2, etc.) post-training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from PyQt6.QtCore import QThread, pyqtSignal

from utils.project_state import ProjectState
from backend.data_handler import get_kfold_splitter
from backend.training_config import build_loss, build_optimizer


class TrainingWorker(QThread):
    """
    Runs the PyTorch training loop in a background thread.
    Emits signals to update the UI (progress, loss curves, logs, metrics).
    """

    # Signals
    epoch_finished = pyqtSignal(int, float, float, dict)  # epoch, train_loss, val_loss, metrics
    batch_progress = pyqtSignal(int, int)  # current_batch, total_batches
    evaluation_finished = pyqtSignal(dict) # Contains computed metrics
    training_finished = pyqtSignal(bool, str)  # success, message
    log_message = pyqtSignal(str)

    def __init__(self, state: ProjectState):
        super().__init__()
        self.state = state
        self._is_running = True

    def stop(self):
        """Request the training loop to stop early."""
        self._is_running = False

    def run(self):
        try:
            self.log_message.emit("Starting training initialization...")

            # 1. Device selection
            device = torch.device(self.state.device)
            self.log_message.emit(f"Using device: {device}")

            if self.state.model is None or self.state.dataframe is None:
                raise ValueError("Model or dataset is missing from state.")

            # Move model to device
            model = self.state.model.to(device)

            # 2. Extract Data
            df = self.state.dataframe
            target_col = self.state.target_column
            problem_type = self.state.problem_type

            X = df.drop(columns=[target_col]).values.astype(np.float32)
            y = df[target_col].values
            if len(X) < 2:
                raise ValueError("Training requires at least two rows to create a validation split.")
            first_module = next(iter(model.children()), None)
            expects_sequence_input = isinstance(first_module, (nn.Conv1d, nn.LazyConv1d))

            def model_input(array: np.ndarray) -> np.ndarray:
                return array[:, np.newaxis, :] if expects_sequence_input else array

            if problem_type == "classification":
                # Label Encoding
                unique_labels = np.unique(y)
                num_classes = len(unique_labels)
                self.log_message.emit(f"Detected {num_classes} classes: {unique_labels.tolist()}")

                last_linear = None
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module

                if last_linear and last_linear.out_features != num_classes:
                    error_msg = (f"Model output mismatch: The last layer has {last_linear.out_features} neurons, "
                                 f"but your data has {num_classes} unique labels. "
                                 f"Please adjust your last layer to have {num_classes} neurons.")
                    self.log_message.emit(f"❌ {error_msg}")
                    raise ValueError(error_msg)

                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y).astype(np.int64)
                self.log_message.emit("Labels encoded to range [0, C-1] successfully.")
            else:
                y = y.astype(np.float32).reshape(-1, 1)

                last_linear = None
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module

                if last_linear and last_linear.out_features != 1:
                    error_msg = (f"Model output mismatch: The last layer has {last_linear.out_features} neurons, "
                                 f"but regression requires exactly 1 neuron for the final output. "
                                 f"Please go back to the Model Builder and adjust your last layer to have 1 neuron.")
                    self.log_message.emit(f"❌ {error_msg}")
                    raise ValueError(error_msg)

            # Hyperparams
            hp = self.state.hyperparams
            lr = hp.get("lr", 0.001)
            epochs = hp.get("epochs", 50)
            batch_size = hp.get("batch_size", 32)

            criterion = build_loss(self.state.loss_fn_name)
            self.log_message.emit(
                f"Loss: {self.state.loss_fn_name}  |  "
                f"Optimizer: {self.state.optimizer_name}  |  lr={lr}"
            )

            # 3. Handle splits
            split_cfg = self.state.split_config
            final_X_val, final_y_val = None, None

            if split_cfg["method"] == "percentage":
                ratio = split_cfg.get("ratio", 0.8)
                self.log_message.emit(f"Data split: {ratio*100:.0f}% Train / {(1-ratio)*100:.0f}% Val")

                indices = np.random.permutation(len(X))
                split_idx = int(ratio * len(X))
                split_idx = min(max(split_idx, 1), len(X) - 1)

                X_train, X_val = X[indices[:split_idx]], X[indices[split_idx:]]
                y_train, y_val = y[indices[:split_idx]], y[indices[split_idx:]]

                # Resampling
                resample = split_cfg.get("resample", "none")
                if resample != "none" and problem_type == "classification":
                    self.log_message.emit(f"Applying {resample.upper()} resampling to training set...")
                    try:
                        if resample == "smote":
                            from imblearn.over_sampling import SMOTE
                            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
                        elif resample == "undersample":
                            from imblearn.under_sampling import RandomUnderSampler
                            X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
                        self.log_message.emit(f"Resampling successful. New training size: {len(X_train)}")
                    except ImportError:
                        self.log_message.emit("⚠️ imbalanced-learn is not installed. Skipping resampling. (pip install imbalanced-learn)")

                optimizer = build_optimizer(
                    self.state.optimizer_name, model.parameters(), lr=lr
                )
                self._train_loop(
                    model, device, criterion, optimizer,
                    model_input(X_train), y_train, model_input(X_val), y_val,
                    epochs, batch_size, fold_msg=""
                )
                final_X_val, final_y_val = model_input(X_val), y_val

            elif split_cfg["method"] == "kfold":
                k = split_cfg.get("k", 5)
                self.log_message.emit(f"Data split: {k}-Fold Cross Validation")
                splitter = get_kfold_splitter(k=k)

                from copy import deepcopy
                initial_weights = deepcopy(model.state_dict())

                for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
                    if not self._is_running:
                        break

                    self.log_message.emit(f"--- Starting Fold {fold + 1}/{k} ---")
                    model.load_state_dict(initial_weights) # Reset per fold
                    optimizer = build_optimizer(
                        self.state.optimizer_name, model.parameters(), lr=lr
                    )

                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    # Resampling
                    resample = split_cfg.get("resample", "none")
                    if resample != "none" and problem_type == "classification":
                        try:
                            if resample == "smote":
                                from imblearn.over_sampling import SMOTE
                                X_train, y_train = SMOTE().fit_resample(X_train, y_train)
                            elif resample == "undersample":
                                from imblearn.under_sampling import RandomUnderSampler
                                X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
                        except ImportError:
                            pass # Already logged warning in fold 1

                    self._train_loop(
                        model, device, criterion, optimizer,
                        model_input(X_train), y_train, model_input(X_val), y_val,
                        epochs, batch_size, fold_msg=f"[Fold {fold+1}] "
                    )
                    # For metrics, just take the last fold's validation set
                    final_X_val, final_y_val = model_input(X[val_idx]), y[val_idx]

            else:
                raise ValueError(f"Unknown split method: {split_cfg['method']}")

            if self._is_running:
                # Calculate advanced evaluation metrics on the final validation set
                if final_X_val is not None:
                    self._evaluate_model(model, device, final_X_val, final_y_val, problem_type)

                # Save final model state
                self.state.model = model.cpu() # Return to CPU for safe keeping
                self.training_finished.emit(True, "Training completed successfully.")
            else:
                self.training_finished.emit(True, "Training stopped by user.")

        except Exception as e:
            self.training_finished.emit(False, str(e))

    def _train_loop(self, model, device, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size, fold_msg=""):
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            if not self._is_running:
                break

            # Training Phase
            model.train()
            total_train_loss = 0.0
            total_batches = len(train_loader)

            for batch_idx, (data, target) in enumerate(train_loader):
                if not self._is_running:
                    break

                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * data.size(0)
                self.batch_progress.emit(batch_idx + 1, total_batches)

            # Validation Phase
            model.eval()
            total_val_loss = 0.0

            all_preds = []
            all_targets = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    total_val_loss += loss.item() * data.size(0)

                    if self.state.problem_type == "classification":
                        # If output is (N, 1) or (N,), treat as binary logits. Else argmax.
                        if output.ndim == 1 or output.shape[1] == 1:
                            preds = (torch.sigmoid(output) >= 0.5).long().view(-1)
                        else:
                            preds = torch.argmax(output, dim=1)

                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())

            avg_train_loss = total_train_loss / len(train_dataset)
            avg_val_loss = total_val_loss / len(val_dataset)

            metrics = {}
            if self.state.problem_type == "classification" and len(all_targets) > 0:
                acc = accuracy_score(all_targets, all_preds)
                # Determine average method for F1 (binary if 2 classes max, else macro)
                unique_classes = len(np.unique(all_targets))
                avg_method = 'binary' if unique_classes <= 2 else 'macro'
                try:
                    f1 = f1_score(all_targets, all_preds, average=avg_method, zero_division=0)
                except ValueError:
                    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

                metrics["val_acc"] = acc
                metrics["val_f1"] = f1
                self.log_message.emit(
                    f"{fold_msg}Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} "
                    f"- Val Loss: {avg_val_loss:.4f} - Val Acc: {acc:.4f} - Val F1: {f1:.4f}"
                )
            else:
                self.log_message.emit(f"{fold_msg}Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

            self.epoch_finished.emit(epoch + 1, avg_train_loss, avg_val_loss, metrics)

    def _evaluate_model(self, model, device, X_val, y_val, problem_type):
        self.log_message.emit("Computing advanced evaluation metrics...")
        model.eval()
        with torch.no_grad():
            t_X = torch.tensor(X_val).to(device)
            outputs = model(t_X).cpu()

        y_true = y_val
        metrics = {}

        try:
            if problem_type == "classification":
                if outputs.ndim == 1 or outputs.shape[1] == 1:
                    scores = torch.sigmoid(outputs).numpy().reshape(-1)
                    y_pred = (scores >= 0.5).astype(int)
                    probs = np.column_stack([1.0 - scores, scores])
                else:
                    probs = torch.softmax(outputs, dim=1).numpy()
                    y_pred = np.argmax(probs, axis=1)

                metrics["Accuracy"] = accuracy_score(y_true, y_pred)
                # use average='weighted' to handle multi-class seamlessly
                metrics["Precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                metrics["Recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                metrics["F1 Score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

                try:
                    # ROC-AUC needs probabilities. For binary, pass probs of class 1. For multi-class, pass all probs.
                    if probs.shape[1] == 2:
                        metrics["ROC-AUC"] = roc_auc_score(y_true, probs[:, 1])
                    else:
                        metrics["ROC-AUC"] = roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")
                except Exception as e:
                    self.log_message.emit(f"Could not compute ROC-AUC: {e}")

            else: # regression
                y_pred = outputs.numpy().flatten()
                y_true_flat = y_true.flatten()

                metrics["MSE"] = mean_squared_error(y_true_flat, y_pred)
                metrics["RMSE"] = np.sqrt(metrics["MSE"])
                metrics["MAE"] = mean_absolute_error(y_true_flat, y_pred)
                metrics["R2 Score"] = r2_score(y_true_flat, y_pred)

            self.evaluation_finished.emit(metrics)
            self.log_message.emit("Metrics computed successfully.")
        except Exception as e:
            self.log_message.emit(f"Error computing metrics: {e}")

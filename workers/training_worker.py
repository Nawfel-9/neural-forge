"""
training_worker.py
==================
Multithreaded training loop using PyQt6 QThread and PyTorch.
Handles both percentage split and K-Fold cross validation.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from PyQt6.QtCore import QThread, pyqtSignal

from utils.project_state import ProjectState
from backend.data_handler import split_data_percentage, get_kfold_splitter


class TrainingWorker(QThread):
    """
    Runs the PyTorch training loop in a background thread.
    Emits signals to update the UI (progress, loss curves, logs).
    """

    # Signals
    epoch_finished = pyqtSignal(int, float, float)  # epoch, train_loss, val_loss
    batch_progress = pyqtSignal(int, int)  # current_batch, total_batches
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
            
            if problem_type == "classification":
                # Label Encoding
                unique_labels = np.unique(y)
                num_classes = len(unique_labels)
                self.log_message.emit(f"Detected {num_classes} classes: {unique_labels.tolist()}")
                
                # Check model output neurons vs classes
                # We can find the last linear layer's output features
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

            # Hyperparams
            hp = self.state.hyperparams
            lr = hp.get("lr", 0.001)
            epochs = hp.get("epochs", 50)
            batch_size = hp.get("batch_size", 32)

            criterion = nn.CrossEntropyLoss() if problem_type == "classification" else nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # 3. Handle splits
            split_cfg = self.state.split_config
            
            if split_cfg["method"] == "percentage":
                ratio = split_cfg.get("ratio", 0.8)
                self.log_message.emit(f"Data split: {ratio*100:.0f}% Train / {(1-ratio)*100:.0f}% Val")
                
                indices = np.random.permutation(len(X))
                split_idx = int(ratio * len(X))
                
                X_train, X_val = X[indices[:split_idx]], X[indices[split_idx:]]
                y_train, y_val = y[indices[:split_idx]], y[indices[split_idx:]]
                
                self._train_loop(
                    model, device, criterion, optimizer,
                    X_train, y_train, X_val, y_val,
                    epochs, batch_size, fold_msg=""
                )
                
            elif split_cfg["method"] == "kfold":
                k = split_cfg.get("k", 5)
                self.log_message.emit(f"Data split: {k}-Fold Cross Validation")
                splitter = get_kfold_splitter(k=k)
                
                # For basic visual tracking, we'll train on each fold but emit continuous epochs
                # Or just reset model (for true k-fold evaluation). Here we reset model for each fold.
                from copy import deepcopy
                initial_weights = deepcopy(model.state_dict())
                
                for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
                    if not self._is_running:
                        break
                    
                    self.log_message.emit(f"--- Starting Fold {fold + 1}/{k} ---")
                    model.load_state_dict(initial_weights) # Reset per fold
                    
                    self._train_loop(
                        model, device, criterion, optimizer,
                        X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                        epochs, batch_size, fold_msg=f"[Fold {fold+1}] "
                    )
                    
            if self._is_running:
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
                
                # Update progress
                self.batch_progress.emit(batch_idx + 1, total_batches)

            # Validation Phase
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    total_val_loss += loss.item() * data.size(0)

            avg_train_loss = total_train_loss / len(train_dataset)
            avg_val_loss = total_val_loss / len(val_dataset)
            
            self.log_message.emit(f"{fold_msg}Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            self.epoch_finished.emit(epoch + 1, avg_train_loss, avg_val_loss)

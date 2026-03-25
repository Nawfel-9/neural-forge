import sys
import pytest
import pandas as pd
import torch
import torch.nn as nn
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop

from utils.project_state import ProjectState
from workers.training_worker import TrainingWorker
from ui.window_training import TrainingWindow

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

@pytest.fixture
def dummy_state():
    state = ProjectState()
    # Create simple dataset
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "feat2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1]
    })
    state.dataframe = df
    state.target_column = "target"
    state.problem_type = "classification"
    state.split_config = {"method": "percentage", "ratio": 0.5}
    
    # Create simple model
    model = nn.Sequential(nn.Linear(2, 2))
    state.model = model
    state.device = "cpu"
    state.hyperparams = {"lr": 0.01, "epochs": 2, "batch_size": 2}
    return state

def test_training_worker_success(qapp, dummy_state):
    worker = TrainingWorker(dummy_state)
    
    # Track signals
    epochs_emitted = []
    def on_epoch(e, t_loss, v_loss):
        epochs_emitted.append(e)
        
    finished_emitted = []
    def on_finished(success, msg):
        finished_emitted.append(success)
        
    worker.epoch_finished.connect(on_epoch)
    worker.training_finished.connect(on_finished)
    
    # Use an event loop to wait for the thread to finish
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    
    worker.start()
    loop.exec()
    
    assert len(finished_emitted) == 1
    assert finished_emitted[0] is True
    assert len(epochs_emitted) == 2
    assert epochs_emitted == [1, 2]

def test_training_window_ui(qapp, dummy_state):
    win = TrainingWindow(dummy_state)
    assert win.spin_lr.value() == 0.01
    assert win.spin_epochs.value() == 2
    assert win.spin_bs.value() == 2
    
    # Test setting device and parsing
    win.combo_device.setCurrentIndex(0) # CPU
    assert win.combo_device.currentData() == "cpu"

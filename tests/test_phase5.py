import os
import pytest
import torch
import torch.nn as nn

from backend.exporter import export_to_onnx

def test_exporter_success(tmp_path):
    """Test that a simple PyTorch model can be successfully exported to ONNX format."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    dummy_input = torch.randn(1, 10)
    
    export_path = tmp_path / "test_model.onnx"
    
    success, msg = export_to_onnx(model, dummy_input, str(export_path))
    
    assert success is True
    assert "successfully exported" in msg
    assert os.path.exists(str(export_path))

def test_exporter_failure(tmp_path):
    """Test that the exporter gracefully catches and reports errors (e.g. invalid arguments)."""
    model = nn.Sequential(nn.Linear(10, 5))
    dummy_input = torch.randn(1, 5) # Shape mismatch!
    
    export_path = tmp_path / "test_fail.onnx"
    
    success, msg = export_to_onnx(model, dummy_input, str(export_path))
    
    assert success is False
    assert "failed" in msg.lower()

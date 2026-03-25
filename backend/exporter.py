"""
exporter.py
===========
ONNX export for trained PyTorch models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

def export_to_onnx(model: nn.Module, dummy_tensor: torch.Tensor, filepath: str) -> tuple[bool, str]:
    """
    Export a PyTorch model to ONNX format.
    
    Parameters
    ----------
    model : nn.Module
        The trained PyTorch model.
    dummy_tensor : torch.Tensor
        A tensor matching the input shape of the model.
    filepath : str
        The destination path for the .onnx file.
        
    Returns
    -------
    tuple[bool, str]
        (success_boolean, status_message)
    """
    try:
        model.eval()
        # We ensure both the model and the dummy tensor are on CPU
        model_cpu = model.to('cpu')
        dummy_tensor_cpu = dummy_tensor.to('cpu')
        
        torch.onnx.export(
            model_cpu,
            dummy_tensor_cpu,
            filepath,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        return True, "Model successfully exported to ONNX format."
    except Exception as exc:
        return False, f"Export failed: {type(exc).__name__} - {str(exc)}"

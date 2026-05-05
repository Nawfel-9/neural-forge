"""
test_phase5.py
==============
Automated tests for Phase 5: ONNX export.

Run with:
    python -m pytest tests/test_phase5.py -v

Test class
----------
TestOnnxExporter — covers success, failure, file integrity, and edge cases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.exporter import export_to_onnx


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simple_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxExporter:
    def test_success(self, tmp_path: Path):
        """A valid model + matching dummy tensor produces a file."""
        model = _simple_model()
        dummy = torch.randn(1, 10)
        path = str(tmp_path / "model.onnx")

        success, msg = export_to_onnx(model, dummy, path)

        assert success is True
        assert "successfully exported" in msg.lower()
        assert os.path.exists(path)

    def test_output_file_is_non_empty(self, tmp_path: Path):
        """The exported ONNX file must contain actual bytes."""
        model = _simple_model()
        dummy = torch.randn(1, 10)
        path = str(tmp_path / "model.onnx")

        export_to_onnx(model, dummy, path)

        assert os.path.getsize(path) > 0

    def test_wrong_input_shape_returns_false(self, tmp_path: Path):
        """Shape mismatch must return (False, message) — not raise an exception."""
        model = nn.Sequential(nn.Linear(10, 5))
        dummy = torch.randn(1, 5)  # wrong: model expects 10 features
        path = str(tmp_path / "bad.onnx")

        success, msg = export_to_onnx(model, dummy, path)

        assert success is False
        assert "failed" in msg.lower()

    def test_failure_message_contains_error_type(self, tmp_path: Path):
        """The failure message should name the exception type."""
        model = nn.Sequential(nn.Linear(10, 5))
        dummy = torch.randn(1, 5)
        path = str(tmp_path / "bad.onnx")

        success, msg = export_to_onnx(model, dummy, path)

        assert success is False
        # The format is "Export failed: ExceptionType - details"
        assert "Export failed:" in msg

    def test_export_does_not_mutate_model_device(self, tmp_path: Path):
        """Exporting must not permanently move the model away from its device."""
        model = _simple_model()  # starts on CPU
        dummy = torch.randn(1, 10)
        path = str(tmp_path / "model.onnx")

        export_to_onnx(model, dummy, path)

        # All parameters should still be on CPU after export
        for param in model.parameters():
            assert param.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available on this machine",
    )
    def test_model_on_cuda_exports_cleanly(self, tmp_path: Path):
        """A model on GPU should be transparently moved to CPU for export."""
        model = _simple_model().cuda()
        dummy = torch.randn(1, 10).cuda()
        path = str(tmp_path / "cuda_model.onnx")

        success, msg = export_to_onnx(model, dummy, path)

        assert success is True
        assert os.path.exists(path)

    def test_exported_onnx_is_loadable(self, tmp_path: Path):
        """The exported file should be parseable by the onnx library (if installed)."""
        pytest.importorskip("onnx")
        import onnx  # noqa: PLC0415

        model = _simple_model()
        dummy = torch.randn(1, 10)
        path = str(tmp_path / "model.onnx")

        export_to_onnx(model, dummy, path)
        loaded = onnx.load(path)

        # onnx.checker will raise if the graph is malformed
        onnx.checker.check_model(loaded)

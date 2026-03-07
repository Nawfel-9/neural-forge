"""
test_phase3.py
==============
Automated tests for Phase 3: Blueprint → nn.Sequential + ghost run.

Run with:
    python -m pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.model_builder import (
    build_model,
    ghost_run,
    build_and_validate,
    _make_activation,
)
from utils.validators import validate_blueprint


# ─────────────────────────────────────────────────────────────────────────────
# 1. Activation factory
# ─────────────────────────────────────────────────────────────────────────────

class TestActivationFactory:
    def test_relu(self):
        act = _make_activation("ReLU")
        assert isinstance(act, nn.ReLU)

    def test_sigmoid(self):
        act = _make_activation("Sigmoid")
        assert isinstance(act, nn.Sigmoid)

    def test_tanh(self):
        act = _make_activation("Tanh")
        assert isinstance(act, nn.Tanh)

    def test_leaky_relu(self):
        act = _make_activation("LeakyReLU")
        assert isinstance(act, nn.LeakyReLU)

    def test_softmax(self):
        act = _make_activation("Softmax")
        assert isinstance(act, nn.Softmax)

    def test_none(self):
        assert _make_activation("None") is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. build_model — Linear-only blueprints
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildLinear:
    def test_single_linear(self):
        bp = [{"type": "Linear", "neurons": 10, "activation": "None"}]
        model = build_model(bp)
        assert isinstance(model, nn.Sequential)
        assert isinstance(model[0], nn.LazyLinear)

    def test_multi_linear(self):
        bp = [
            {"type": "Linear", "neurons": 64, "activation": "ReLU"},
            {"type": "Linear", "neurons": 32, "activation": "Sigmoid"},
            {"type": "Linear", "neurons": 1, "activation": "None"},
        ]
        model = build_model(bp)
        # LazyLinear(64), ReLU, Linear(64,32), Sigmoid, Linear(32,1)
        modules = list(model.children())
        assert isinstance(modules[0], nn.LazyLinear)
        assert isinstance(modules[1], nn.ReLU)
        assert isinstance(modules[2], nn.Linear)
        assert modules[2].in_features == 64
        assert modules[2].out_features == 32
        assert isinstance(modules[3], nn.Sigmoid)
        assert isinstance(modules[4], nn.Linear)
        assert modules[4].in_features == 32
        assert modules[4].out_features == 1

    def test_linear_with_dropout(self):
        bp = [
            {"type": "Linear", "neurons": 64, "activation": "ReLU"},
            {"type": "Dropout", "rate": 0.5},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        model = build_model(bp)
        modules = list(model.children())
        assert isinstance(modules[2], nn.Dropout)
        assert modules[2].p == 0.5

    def test_linear_with_batchnorm(self):
        bp = [
            {"type": "Linear", "neurons": 128, "activation": "ReLU"},
            {"type": "BatchNorm1d"},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        model = build_model(bp)
        modules = list(model.children())
        assert isinstance(modules[2], nn.BatchNorm1d)
        assert modules[2].num_features == 128


# ─────────────────────────────────────────────────────────────────────────────
# 3. build_model — Conv1d blueprints
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildConv:
    def test_conv1d_basic(self):
        bp = [
            {"type": "Conv1d", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "Flatten"},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        model = build_model(bp)
        modules = list(model.children())
        assert isinstance(modules[0], nn.LazyConv1d)
        assert isinstance(modules[1], nn.Flatten)
        assert isinstance(modules[2], nn.LazyLinear)

    def test_conv1d_maxpool(self):
        bp = [
            {"type": "Conv1d", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0},
            {"type": "MaxPool1d", "kernel_size": 2, "stride": 2},
            {"type": "Flatten"},
            {"type": "Linear", "neurons": 5, "activation": "None"},
        ]
        model = build_model(bp)
        modules = list(model.children())
        assert isinstance(modules[1], nn.MaxPool1d)

    def test_conv1d_avgpool(self):
        bp = [
            {"type": "Conv1d", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 0},
            {"type": "AvgPool1d", "kernel_size": 2, "stride": 2},
            {"type": "Flatten"},
            {"type": "Linear", "neurons": 3, "activation": "None"},
        ]
        model = build_model(bp)
        modules = list(model.children())
        assert isinstance(modules[1], nn.AvgPool1d)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ghost run — Linear
# ─────────────────────────────────────────────────────────────────────────────

class TestGhostRunLinear:
    def test_simple_forward_pass(self):
        bp = [
            {"type": "Linear", "neurons": 32, "activation": "ReLU"},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        model = build_model(bp)
        success, output, msg = ghost_run(model, input_features=5, batch_size=4)
        assert success is True
        assert output is not None
        assert output.shape == (4, 10)
        assert "passed" in msg.lower()

    def test_deep_network(self):
        bp = [
            {"type": "Linear", "neurons": 128, "activation": "ReLU"},
            {"type": "BatchNorm1d"},
            {"type": "Dropout", "rate": 0.3},
            {"type": "Linear", "neurons": 64, "activation": "ReLU"},
            {"type": "Dropout", "rate": 0.2},
            {"type": "Linear", "neurons": 1, "activation": "None"},
        ]
        model = build_model(bp)
        success, output, msg = ghost_run(model, input_features=20)
        assert success is True
        assert output.shape == (2, 1)

    def test_all_activations(self):
        """Each activation should produce a valid output."""
        for act in ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "None"]:
            bp = [{"type": "Linear", "neurons": 5, "activation": act}]
            model = build_model(bp)
            success, _, _ = ghost_run(model, input_features=3)
            assert success is True, f"Failed with activation={act}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Ghost run — Conv1d
# ─────────────────────────────────────────────────────────────────────────────

class TestGhostRunConv:
    def test_conv_flatten_linear(self):
        """Conv1d → Flatten → Linear should work with 2D tabular data reshaped."""
        bp = [
            {"type": "Conv1d", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "Flatten"},
            {"type": "Linear", "neurons": 5, "activation": "None"},
        ]
        model = build_model(bp)
        # For Conv1d: input must be (batch, channels, length)
        # With tabular data reshaped: (batch, 1, n_features)
        dummy = torch.randn(2, 1, 10)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        assert output.shape[0] == 2
        assert output.shape[1] == 5

    def test_conv_pool_flatten_linear(self):
        bp = [
            {"type": "Conv1d", "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 0},
            {"type": "MaxPool1d", "kernel_size": 2, "stride": 2},
            {"type": "Flatten"},
            {"type": "Linear", "neurons": 3, "activation": "None"},
        ]
        model = build_model(bp)
        dummy = torch.randn(4, 1, 20)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        assert output.shape[0] == 4
        assert output.shape[1] == 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ghost run — failure cases
# ─────────────────────────────────────────────────────────────────────────────

class TestGhostRunFailures:
    def test_negative_features(self):
        """Ghost run should fail gracefully with negative input features."""
        bp = [{"type": "Linear", "neurons": 10, "activation": "None"}]
        model = build_model(bp)
        success, _, msg = ghost_run(model, input_features=-1)
        assert success is False
        assert "failed" in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 7. build_and_validate convenience
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAndValidate:
    def test_success(self):
        bp = [
            {"type": "Linear", "neurons": 16, "activation": "ReLU"},
            {"type": "Linear", "neurons": 1, "activation": "None"},
        ]
        model, dummy, success, msg = build_and_validate(bp, input_features=5)
        assert success is True
        assert model is not None
        assert dummy is not None
        assert isinstance(model, nn.Sequential)

    def test_bad_blueprint(self):
        bp = [{"type": "UnknownLayer"}]
        model, dummy, success, msg = build_and_validate(bp, input_features=5)
        assert success is False
        assert model is None

    def test_unknown_layer_type(self):
        bp = [{"type": "INVALID_TYPE"}]
        _, _, success, msg = build_and_validate(bp, input_features=5)
        assert success is False


# ─────────────────────────────────────────────────────────────────────────────
# 8. Validator — new layer types
# ─────────────────────────────────────────────────────────────────────────────

class TestValidatorNewTypes:
    def test_conv1d_valid(self):
        bp = [
            {"type": "Conv1d", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 0},
            {"type": "Flatten"},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        valid, msg = validate_blueprint(bp)
        assert valid is True

    def test_conv1d_invalid_channels(self):
        bp = [
            {"type": "Conv1d", "out_channels": 0, "kernel_size": 3},
            {"type": "Flatten"},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        valid, _ = validate_blueprint(bp)
        assert valid is False

    def test_maxpool_valid(self):
        bp = [
            {"type": "MaxPool1d", "kernel_size": 2, "stride": 2},
            {"type": "Linear", "neurons": 5, "activation": "None"},
        ]
        valid, _ = validate_blueprint(bp)
        assert valid is True

    def test_avgpool_invalid_kernel(self):
        bp = [
            {"type": "AvgPool1d", "kernel_size": 0},
            {"type": "Linear", "neurons": 5, "activation": "None"},
        ]
        valid, _ = validate_blueprint(bp)
        assert valid is False

    def test_flatten_valid(self):
        bp = [
            {"type": "Flatten"},
            {"type": "Linear", "neurons": 10, "activation": "None"},
        ]
        valid, _ = validate_blueprint(bp)
        assert valid is True

    def test_flatten_as_last_layer_invalid(self):
        bp = [
            {"type": "Linear", "neurons": 10, "activation": "None"},
            {"type": "Flatten"},
        ]
        valid, _ = validate_blueprint(bp)
        assert valid is False  # Last must be Linear

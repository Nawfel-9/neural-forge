"""
test_phase4.py
==============
Automated tests for Phase 4: Multithreading, Hardware Selection,
and the Loss/Optimizer selection feature.

Run with:
    python -m pytest tests/test_phase4.py -v

Test classes
------------
TestTrainingConfig       — pure-logic tests for backend/training_config.py
TestProjectStateDefaults — new fields added to ProjectState
TestTrainingWorker       — integration tests via QThread
TestTrainingWindowUI     — widget-level tests for window_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

nn = pytest.importorskip("torch.nn")
qt_core = pytest.importorskip("PyQt6.QtCore")
qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
QEventLoop = qt_core.QEventLoop
QApplication = qt_widgets.QApplication

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training_config import (
    DEFAULT_LOSS,
    DEFAULT_OPTIMIZER,
    build_loss,
    build_optimizer,
    get_all_optimizers,
    get_losses_for,
)
from utils.project_state import ProjectState
from workers.training_worker import TrainingWorker


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _make_state(
    problem_type: str = "classification",
    loss_fn_name: str | None = None,
    optimizer_name: str = "Adam",
    epochs: int = 2,
) -> ProjectState:
    """Return a ready-to-train ProjectState with a small synthetic dataset."""
    state = ProjectState()
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "feat2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    state.dataframe = df
    state.target_column = "target"
    state.problem_type = problem_type
    state.split_config = {"method": "percentage", "ratio": 0.5}
    state.model = nn.Sequential(nn.Linear(2, 2))
    state.device = "cpu"
    state.hyperparams = {"lr": 0.01, "epochs": epochs, "batch_size": 2}
    state.loss_fn_name = loss_fn_name or DEFAULT_LOSS[problem_type]
    state.optimizer_name = optimizer_name
    return state


def _run_worker(state: ProjectState, qapp) -> tuple[list[bool], list[int]]:
    """Run a TrainingWorker synchronously and return (finished_flags, epochs)."""
    worker = TrainingWorker(state)

    finished_flags: list[bool] = []
    epochs_emitted: list[int] = []

    worker.epoch_finished.connect(lambda e, *_: epochs_emitted.append(e))
    worker.training_finished.connect(lambda ok, _: finished_flags.append(ok))

    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    worker.start()
    loop.exec()

    return finished_flags, epochs_emitted


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training Config — pure logic (no Qt needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingConfig:
    # ── get_losses_for ──────────────────────────────────────────────────────

    def test_classification_losses_count(self):
        losses = get_losses_for("classification")
        assert len(losses) == 3

    def test_classification_losses_names(self):
        losses = get_losses_for("classification")
        assert "CrossEntropyLoss" in losses
        assert "BCEWithLogitsLoss" in losses
        assert "NLLLoss" in losses

    def test_regression_losses_count(self):
        losses = get_losses_for("regression")
        assert len(losses) == 3

    def test_regression_losses_names(self):
        losses = get_losses_for("regression")
        assert "MSELoss" in losses
        assert "L1Loss" in losses
        assert "SmoothL1Loss" in losses

    def test_classification_and_regression_are_disjoint(self):
        cls_set = set(get_losses_for("classification"))
        reg_set = set(get_losses_for("regression"))
        assert cls_set.isdisjoint(reg_set)

    def test_get_losses_unknown_problem_type(self):
        with pytest.raises(ValueError, match="Unknown problem type"):
            get_losses_for("unknown_type")

    # ── get_all_optimizers ──────────────────────────────────────────────────

    def test_optimizer_count(self):
        opts = get_all_optimizers()
        assert len(opts) == 4

    def test_optimizer_names(self):
        opts = get_all_optimizers()
        for name in ("Adam", "AdamW", "SGD", "RMSprop"):
            assert name in opts

    # ── build_loss ──────────────────────────────────────────────────────────

    def test_build_crossentropy(self):
        loss = build_loss("CrossEntropyLoss")
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_build_mseloss(self):
        loss = build_loss("MSELoss")
        assert isinstance(loss, nn.MSELoss)

    def test_build_l1loss(self):
        loss = build_loss("L1Loss")
        assert isinstance(loss, nn.L1Loss)

    def test_build_smoothl1(self):
        loss = build_loss("SmoothL1Loss")
        assert isinstance(loss, nn.SmoothL1Loss)

    def test_build_bce_with_logits(self):
        loss = build_loss("BCEWithLogitsLoss")
        assert isinstance(loss, nn.BCEWithLogitsLoss)

    def test_build_nllloss(self):
        loss = build_loss("NLLLoss")
        assert isinstance(loss, nn.NLLLoss)

    def test_build_loss_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss function"):
            build_loss("FakeLoss")

    # ── build_optimizer ─────────────────────────────────────────────────────

    def test_build_adam(self):
        model = nn.Linear(2, 2)
        opt = build_optimizer("Adam", model.parameters(), lr=0.001)
        import torch.optim as optim
        assert isinstance(opt, optim.Adam)

    def test_build_adamw(self):
        model = nn.Linear(2, 2)
        opt = build_optimizer("AdamW", model.parameters(), lr=0.001)
        import torch.optim as optim
        assert isinstance(opt, optim.AdamW)

    def test_build_sgd(self):
        model = nn.Linear(2, 2)
        opt = build_optimizer("SGD", model.parameters(), lr=0.01)
        import torch.optim as optim
        assert isinstance(opt, optim.SGD)

    def test_build_rmsprop(self):
        model = nn.Linear(2, 2)
        opt = build_optimizer("RMSprop", model.parameters(), lr=0.001)
        import torch.optim as optim
        assert isinstance(opt, optim.RMSprop)

    def test_build_optimizer_unknown_raises(self):
        model = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer("FakeOptimizer", model.parameters())

    def test_build_optimizer_lr_is_applied(self):
        """Verify the lr parameter is actually passed to the optimizer."""
        model = nn.Linear(2, 2)
        opt = build_optimizer("Adam", model.parameters(), lr=0.123)
        assert abs(opt.param_groups[0]["lr"] - 0.123) < 1e-9

    # ── Defaults ────────────────────────────────────────────────────────────

    def test_default_classification_loss(self):
        assert DEFAULT_LOSS["classification"] == "CrossEntropyLoss"

    def test_default_regression_loss(self):
        assert DEFAULT_LOSS["regression"] == "MSELoss"

    def test_default_optimizer(self):
        assert DEFAULT_OPTIMIZER == "Adam"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ProjectState — new fields
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectStateDefaults:
    def test_default_loss_fn_name(self):
        state = ProjectState()
        assert state.loss_fn_name == "CrossEntropyLoss"

    def test_default_optimizer_name(self):
        state = ProjectState()
        assert state.optimizer_name == "Adam"

    def test_loss_fn_name_is_mutable(self):
        state = ProjectState()
        state.loss_fn_name = "MSELoss"
        assert state.loss_fn_name == "MSELoss"

    def test_optimizer_name_is_mutable(self):
        state = ProjectState()
        state.optimizer_name = "AdamW"
        assert state.optimizer_name == "AdamW"


# ─────────────────────────────────────────────────────────────────────────────
# 3. TrainingWorker — integration
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingWorker:
    def test_worker_default_classification(self, qapp):
        state = _make_state("classification")
        finished, epochs = _run_worker(state, qapp)
        assert finished == [True]
        assert epochs == [1, 2]

    def test_worker_with_adamw(self, qapp):
        state = _make_state("classification", optimizer_name="AdamW")
        finished, epochs = _run_worker(state, qapp)
        assert finished == [True]
        assert len(epochs) == 2

    def test_worker_with_sgd(self, qapp):
        state = _make_state("classification", optimizer_name="SGD")
        finished, epochs = _run_worker(state, qapp)
        assert finished == [True]

    def test_worker_with_rmsprop(self, qapp):
        state = _make_state("classification", optimizer_name="RMSprop")
        finished, epochs = _run_worker(state, qapp)
        assert finished == [True]

    def test_worker_stop_flag(self, qapp):
        """Stopping early must emit training_finished(True, ...) not False."""
        state = _make_state("classification", epochs=10)
        worker = TrainingWorker(state)

        finished_flags: list[bool] = []
        worker.training_finished.connect(lambda ok, _: finished_flags.append(ok))

        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        worker.start()
        # Request stop immediately
        worker.stop()
        loop.exec()

        assert len(finished_flags) == 1
        # Stopped by user → still counts as True (not an error)
        assert finished_flags[0] is True

    def test_worker_missing_model_emits_error(self, qapp):
        """Worker should emit training_finished(False, ...) if model is None."""
        state = ProjectState()
        state.dataframe = pd.DataFrame({"a": [1], "b": [0]})
        state.target_column = "b"
        state.problem_type = "classification"
        # Intentionally leave state.model = None

        worker = TrainingWorker(state)
        finished_flags: list[bool] = []
        worker.training_finished.connect(lambda ok, _: finished_flags.append(ok))

        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        worker.start()
        loop.exec()

        assert finished_flags == [False]


# ─────────────────────────────────────────────────────────────────────────────
# 4. TrainingWindow — UI widget tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingWindowUI:
    def _make_window(self, problem_type: str = "classification"):
        from ui.window_training import TrainingWindow
        state = _make_state(problem_type)
        return TrainingWindow(state)

    def test_loss_combo_classification_count(self, qapp):
        win = self._make_window("classification")
        assert win.combo_loss.count() == 3

    def test_loss_combo_classification_items(self, qapp):
        win = self._make_window("classification")
        items = [win.combo_loss.itemText(i) for i in range(win.combo_loss.count())]
        assert "CrossEntropyLoss" in items
        assert "BCEWithLogitsLoss" in items
        assert "NLLLoss" in items

    def test_loss_combo_regression_count(self, qapp):
        win = self._make_window("regression")
        assert win.combo_loss.count() == 3

    def test_loss_combo_regression_items(self, qapp):
        win = self._make_window("regression")
        items = [win.combo_loss.itemText(i) for i in range(win.combo_loss.count())]
        assert "MSELoss" in items
        assert "L1Loss" in items
        assert "SmoothL1Loss" in items

    def test_optimizer_combo_count(self, qapp):
        win = self._make_window()
        assert win.combo_optimizer.count() == 4

    def test_optimizer_combo_default(self, qapp):
        win = self._make_window()
        assert win.combo_optimizer.currentText() == "Adam"

    def test_loss_combo_default_classification(self, qapp):
        win = self._make_window("classification")
        assert win.combo_loss.currentText() == "CrossEntropyLoss"

    def test_loss_combo_default_regression(self, qapp):
        win = self._make_window("regression")
        assert win.combo_loss.currentText() == "MSELoss"

    def test_refresh_ui_repopulates_loss_for_regression(self, qapp):
        """Simulates user going Back → changing problem type → coming forward again."""
        from ui.window_training import TrainingWindow
        state = _make_state("classification")
        win = TrainingWindow(state)

        # Simulate problem type change after the window was already built
        state.problem_type = "regression"
        state.loss_fn_name = DEFAULT_LOSS["regression"]
        win.refresh_ui()

        items = [win.combo_loss.itemText(i) for i in range(win.combo_loss.count())]
        assert "MSELoss" in items
        assert "CrossEntropyLoss" not in items

    def test_hyperparams_displayed(self, qapp):
        win = self._make_window()
        assert win.spin_lr.value() == pytest.approx(0.01)
        assert win.spin_epochs.value() == 2
        assert win.spin_bs.value() == 2

    def test_device_combo_has_cpu(self, qapp):
        win = self._make_window()
        items = [win.combo_device.itemText(i) for i in range(win.combo_device.count())]
        assert any("CPU" in item for item in items)

from __future__ import annotations

import sys

import pytest

qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
QApplication = qt_widgets.QApplication

from main import DevProjectWindow
from utils.project_state import ProjectState


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_continue_to_training_enabled_when_required_files_exist(tmp_path, qapp):
    for filename in ("model.py", "dataset.py", "config.yaml"):
        (tmp_path / filename).write_text("", encoding="utf-8")

    called = []
    state = ProjectState(dev_project_path=str(tmp_path))
    window = DevProjectWindow(state, on_train=lambda: called.append(True))

    window.refresh_status()

    assert window.btn_training.isEnabled()
    window.btn_training.click()
    assert called == [True]


def test_continue_to_training_disabled_when_required_file_missing(tmp_path, qapp):
    (tmp_path / "model.py").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    state = ProjectState(dev_project_path=str(tmp_path))
    window = DevProjectWindow(state, on_train=lambda: None)

    window.refresh_status()

    assert not window.btn_training.isEnabled()

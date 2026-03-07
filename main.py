"""
main.py
=======
Application entry point.

Offers two workflow paths from a Home Screen:
  Path 1 — No-Code Pipeline: Window 1 (Data) → Window 2 (Model) → Window 3 (Training)
  Path 2 — Developer Mode:   Import existing PyTorch project → shared Training window

Currently wired:
  Path 1: Window 1 ↔ Window 2 (Window 3 stubbed for Phase 4+)
  Path 2: Home → Guide dialog → folder picker (code editor stubbed)
"""

from __future__ import annotations

import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QDialog)
from PyQt6.QtCore import Qt

from ui.styles import DARK_QSS, apply_dark_palette
from ui.window_data import DataWindow
from ui.window_model import ModelBuilderWindow
from ui.window_project_guide import ProjectGuideDialog   # ← NEW
from utils.project_state import ProjectState


# --- Home Window Component ---
class HomeWindow(QWidget):
    """Entry point for the application."""
    def __init__(self, on_no_code, on_import_code):
        super().__init__()
        self.setWindowTitle("Neural Forge — Select Workspace")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)

        title = QLabel("Welcome to Neural Forge")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        btn_layout = QHBoxLayout()

        self.btn_no_code = QPushButton("No-Code Pipeline\n(Beginner Friendly)")
        self.btn_no_code.setFixedSize(220, 150)
        self.btn_no_code.clicked.connect(on_no_code)

        self.btn_code = QPushButton("Import Project\n(Developer Mode)")
        self.btn_code.setFixedSize(220, 150)
        self.btn_code.clicked.connect(on_import_code)

        btn_layout.addWidget(self.btn_no_code)
        btn_layout.addWidget(self.btn_code)
        layout.addLayout(btn_layout)

        subtitle = QLabel("Build, train, and monitor neural networks — no code required.")
        subtitle.setStyleSheet("color: #aaaaaa; font-style: italic;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)


# --- Controller ---
class PipelineController:
    """
    Manages transitions between the Home Screen and the different
    workflow modules (No-Code vs Code-Centric).
    """

    def __init__(self, state: ProjectState) -> None:
        self.state = state
        self._home_win: HomeWindow | None = None
        self._win_data: DataWindow | None = None
        self._win_model: ModelBuilderWindow | None = None

    def start(self) -> None:
        self._home_win = HomeWindow(
            on_no_code=self._start_no_code_pipeline,
            on_import_code=self._open_code_editor,
        )
        self._home_win.show()

    # ── Path 1: No-Code Pipeline ──────────────────────────────────────────
    def _start_no_code_pipeline(self) -> None:
        if self._home_win:
            self._home_win.hide()

        self._win_data = DataWindow(
            project_state=self.state,
            on_next=self._open_model_window,
        )
        self._win_data.show()

    def _open_model_window(self) -> None:
        if self._win_data:
            self._win_data.hide()

        self._win_model = ModelBuilderWindow(
            project_state=self.state,
            on_back=self._back_to_data,
        )
        self._win_model.show()

    def _back_to_data(self) -> None:
        if self._win_model:
            self._win_model.close()
            self._win_model = None
        if self._win_data:
            self._win_data.show()

    # ── Path 2: Import Project ────────────────────────────────────────────
    def _open_code_editor(self) -> None:
        """
        1. Show the Project Structure Guide (unless suppressed).
        2. If the user confirms, open a folder-picker dialog.
        3. Hand the selected path to the CodeEditorWindow (placeholder).
        """
        # Step 1 — show guide if not suppressed
        if ProjectGuideDialog.should_show():
            dlg = ProjectGuideDialog(parent=self._home_win)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return   # user cancelled

        # Step 2 — folder picker
        project_dir = QFileDialog.getExistingDirectory(
            self._home_win,
            "Select your project folder",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not project_dir:
            return   # user cancelled folder picker

        # Step 3 — hand off to CodeEditorWindow (implement when ready)
        print(f"[Developer Mode] Project folder selected: {project_dir}")
        # self._home_win.hide()
        # self._win_code_editor = CodeEditorWindow(state=self.state, project_dir=project_dir)
        # self._win_code_editor.show()


def main() -> None:
    app = QApplication(sys.argv)

    apply_dark_palette(app)
    app.setStyleSheet(DARK_QSS)

    state = ProjectState()

    controller = PipelineController(state)
    controller.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

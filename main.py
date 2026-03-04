"""
main.py
=======
Application entry point.

Launches the 3-window pipeline:
  Window 1  (Data)  →  Window 2  (Model)  →  Window 3  (Training)

Currently wired: Window 1 ↔ Window 2.
Window 3 is stubbed for Phase 4+.
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from ui.styles import DARK_QSS, apply_dark_palette
from ui.window_data import DataWindow
from ui.window_model import ModelBuilderWindow
from utils.project_state import ProjectState


class PipelineController:
    """
    Manages window transitions: Data → Model → Training.

    Each window receives the shared ProjectState and callbacks for
    forward/backward navigation.
    """

    def __init__(self, state: ProjectState) -> None:
        self.state = state
        self._win_data: DataWindow | None = None
        self._win_model: ModelBuilderWindow | None = None

    def start(self) -> None:
        """Show Window 1 (Data)."""
        self._win_data = DataWindow(
            project_state=self.state,
            on_next=self._open_model_window,
        )
        self._win_data.show()

    # ── Transitions ─────────────────────────────────────────────────────────
    def _open_model_window(self) -> None:
        """Called when Window 1 clicks 'Next →'."""
        if self._win_data:
            self._win_data.hide()

        self._win_model = ModelBuilderWindow(
            project_state=self.state,
            on_back=self._back_to_data,
        )
        self._win_model.show()

    def _back_to_data(self) -> None:
        """Called when Window 2 clicks '← Back'."""
        if self._win_model:
            self._win_model.close()
            self._win_model = None

        if self._win_data:
            self._win_data.show()


def main() -> None:
    app = QApplication(sys.argv)

    # Apply dark theme
    apply_dark_palette(app)
    app.setStyleSheet(DARK_QSS)

    # Shared state
    state = ProjectState()

    # Launch pipeline
    controller = PipelineController(state)
    controller.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

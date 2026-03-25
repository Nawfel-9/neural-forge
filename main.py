import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QDialog, QStackedWidget)
from PyQt6.QtCore import Qt

from ui.styles import DARK_QSS, apply_dark_palette
from ui.window_data import DataWindow
from ui.window_model import ModelBuilderWindow
from ui.window_training import TrainingWindow
from ui.window_project_guide import ProjectGuideDialog
from utils.project_state import ProjectState

class HomeWindow(QWidget):
    """Entry point for the application."""
    def __init__(self, on_no_code, on_import_code):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)

        title = QLabel("Welcome to Neural Forge")
        title.setProperty("class", "heading")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffffff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(20)

        btn_layout = QHBoxLayout()

        self.btn_no_code = QPushButton("No-Code Pipeline\n(Beginner Friendly)")
        self.btn_no_code.setFixedSize(260, 180)
        self.btn_no_code.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.btn_no_code.clicked.connect(on_no_code)

        self.btn_code = QPushButton("Import Project\n(Developer Mode)")
        self.btn_code.setFixedSize(260, 180)
        self.btn_code.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.btn_code.clicked.connect(on_import_code)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_no_code)
        btn_layout.addSpacing(40)
        btn_layout.addWidget(self.btn_code)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addSpacing(20)

        subtitle = QLabel("Design, train, and deploy neural networks visually.")
        subtitle.setStyleSheet("color: #8b949e; font-style: italic;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

class PipelineController:
    """Manages window stack and state transitions."""

    def __init__(self, state: ProjectState) -> None:
        self.state = state
        self.stack = QStackedWidget()
        self.stack.setWindowTitle("Neural Forge")
        self.stack.resize(1000, 750)

        # ── Initialize Windows ──
        self.home_win = HomeWindow(
            on_no_code=self._start_no_code_pipeline,
            on_import_code=self._open_code_editor
        )
        self.data_win = DataWindow(self.state, on_next=self._open_model_window)
        self.model_win = ModelBuilderWindow(self.state, on_back=self._back_to_data, on_next=self._open_training_window)
        self.train_win = TrainingWindow(self.state, on_back=self._back_to_model)

        # Add to stack
        self.stack.addWidget(self.home_win)
        self.stack.addWidget(self.data_win)
        self.stack.addWidget(self.model_win)
        self.stack.addWidget(self.train_win)

    def start(self) -> None:
        self.stack.setCurrentWidget(self.home_win)
        self.stack.show()

    def _start_no_code_pipeline(self) -> None:
        self.stack.setCurrentWidget(self.data_win)

    def _open_model_window(self) -> None:
        self.model_win.refresh_data_info()
        self.stack.setCurrentWidget(self.model_win)

    def _back_to_data(self) -> None:
        self.stack.setCurrentWidget(self.data_win)

    def _open_training_window(self) -> None:
        self.train_win.refresh_ui()
        self.stack.setCurrentWidget(self.train_win)

    def _back_to_model(self) -> None:
        self.stack.setCurrentWidget(self.model_win)

    def _open_code_editor(self) -> None:
        if ProjectGuideDialog.should_show():
            dlg = ProjectGuideDialog(parent=self.stack)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

        project_dir = QFileDialog.getExistingDirectory(
            self.stack,
            "Select your project folder",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if project_dir:
            print(f"[Developer Mode] Project folder selected: {project_dir}")

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

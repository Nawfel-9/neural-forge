import sys
import torch
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QFrame, QFileDialog,
                             QGroupBox, QDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap

from ui.styles import apply_theme_palette, get_qss
from ui.window_data import DataWindow
from ui.window_model import ModelBuilderWindow
from ui.window_training import TrainingWindow
from ui.window_export import ExportWindow
from ui.window_assistant import AssistantWindow
from ui.window_project_guide import ProjectGuideDialog
from ui.custom_toggle import PremiumToggle
from utils.project_state import ProjectState

class HomeDashboard(QWidget):
    """Welcome screen for the new dashboard."""
    def __init__(self, no_code_callback, dev_mode_callback):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(18)

        logo_label = QLabel()
        pixmap = QPixmap("assets/logo.png")
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        title = QLabel("NEURAL FORGE")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 32pt; font-weight: 800; letter-spacing: 4px; color: #00A3FF;")
        layout.addWidget(title)

        subtitle = QLabel("Premium Data Science & Deep Learning Platform")
        subtitle.setStyleSheet("color: #64748B; font-size: 14pt; font-weight: 500; margin-bottom: 30px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        path_row = QHBoxLayout()
        path_row.setSpacing(18)
        path_row.addStretch()
        path_row.addWidget(self._build_path_card(
            "No-Code Pipeline",
            "Load a dataset, build layers visually, train, evaluate, and export without writing code.",
            "Start No-Code",
            no_code_callback,
            primary=True,
        ))
        path_row.addWidget(self._build_path_card(
            "Developer Mode",
            "Import an existing PyTorch project and prepare it for the shared training workflow.",
            "Import Project",
            dev_mode_callback,
            primary=False,
        ))
        path_row.addStretch()
        layout.addLayout(path_row)

    def _build_path_card(self, title: str, body: str, button_text: str, callback, primary: bool) -> QFrame:
        card = QFrame()
        card.setObjectName("HomeActionCard")
        card.setFixedSize(340, 190)
        card.setStyleSheet("""
            QFrame#HomeActionCard {
                background-color: rgba(15, 23, 42, 0.42);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
            }
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 20, 22, 20)
        card_layout.setSpacing(12)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 16pt; font-weight: 800;")
        card_layout.addWidget(title_lbl)

        body_lbl = QLabel(body)
        body_lbl.setWordWrap(True)
        body_lbl.setStyleSheet("color: #64748B; font-size: 10.5pt; line-height: 1.4;")
        card_layout.addWidget(body_lbl)
        card_layout.addStretch()

        button = QPushButton(button_text)
        button.setMinimumHeight(42)
        if primary:
            button.setProperty("class", "primary")
        button.clicked.connect(callback)
        card_layout.addWidget(button)
        return card


class DevProjectWindow(QWidget):
    """Developer Mode landing page after a project folder is imported."""

    REQUIRED_FILES = ("model.py", "dataset.py", "config.yaml")
    OPTIONAL_FILES = ("loss.py", "metrics.py", "checkpoints", "logs")

    def __init__(self, project_state: ProjectState, on_import=None, on_no_code=None, on_train=None, parent=None):
        super().__init__(parent)
        self.state = project_state
        self._on_import_callback = on_import
        self._on_no_code_callback = on_no_code
        self._on_train_callback = on_train  # ← add this
        self._file_rows: dict[str, QLabel] = {}
        self._build_ui()
        self.refresh_status()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(24)

        header = QVBoxLayout()
        title = QLabel("Developer Mode")
        title.setProperty("class", "PageTitle")
        subtitle = QLabel("Import a PyTorch project folder that follows Neural Forge's project conventions.")
        subtitle.setProperty("class", "PageSubtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        status_group = QGroupBox("Imported Project")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(12)

        self.lbl_project_path = QLabel("No project imported yet.")
        self.lbl_project_path.setWordWrap(True)
        self.lbl_project_path.setStyleSheet("font-weight: 700;")
        status_layout.addWidget(self.lbl_project_path)

        self.lbl_project_status = QLabel("")
        self.lbl_project_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_project_status)

        root.addWidget(status_group)

        structure_group = QGroupBox("Project Structure")
        structure_layout = QVBoxLayout(structure_group)
        structure_layout.setSpacing(8)

        structure_layout.addWidget(self._section_label("Required"))
        for filename in self.REQUIRED_FILES:
            structure_layout.addLayout(self._build_file_row(filename, required=True))

        structure_layout.addWidget(self._section_label("Optional"))
        for filename in self.OPTIONAL_FILES:
            structure_layout.addLayout(self._build_file_row(filename, required=False))

        root.addWidget(structure_group)

        btn_row = QHBoxLayout()
        btn_import = QPushButton("Import PyTorch Project")
        btn_import.setProperty("class", "primary")
        btn_import.setMinimumHeight(44)
        btn_import.clicked.connect(self._on_import_callback)
        btn_row.addWidget(btn_import)

        # replace the existing btn_training lines with:
        self.btn_training = QPushButton("Continue to Training")
        self.btn_training.setMinimumHeight(44)
        self.btn_training.setEnabled(False)
        self.btn_training.setToolTip("Developer training integration is not implemented yet.")
        self.btn_training.clicked.connect(
            lambda: self._on_train_callback() if self._on_train_callback else None
        )
        btn_row.addWidget(self.btn_training)

        btn_row.addStretch()

        btn_no_code = QPushButton("Use No-Code Pipeline")
        btn_no_code.setMinimumHeight(44)
        btn_no_code.clicked.connect(self._on_no_code_callback)
        btn_row.addWidget(btn_no_code)

        root.addLayout(btn_row)
        root.addStretch()

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setStyleSheet("color: #0EA5E9; font-size: 8pt; font-weight: 800; letter-spacing: 1px;")
        return label

    def _build_file_row(self, filename: str, required: bool) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        name = QLabel(filename)
        name.setMinimumWidth(150)
        name.setStyleSheet("font-family: Consolas, monospace; font-weight: 700;")
        row.addWidget(name)

        badge = QLabel("Required" if required else "Optional")
        badge.setFixedWidth(80)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            "border-radius: 4px; padding: 3px 6px; font-size: 8pt; font-weight: 800; "
            f"background-color: {'rgba(239, 68, 68, 0.16)' if required else 'rgba(16, 185, 129, 0.16)'}; "
            f"color: {'#EF4444' if required else '#10B981'};"
        )
        row.addWidget(badge)

        status = QLabel("Not checked")
        status.setStyleSheet("color: #64748B; font-weight: 600;")
        row.addWidget(status, stretch=1)
        self._file_rows[filename] = status
        return row

    def set_project_path(self, path: str) -> None:
        self.state.dev_project_path = path
        self.refresh_status()

    def refresh_status(self) -> None:
        project_path = getattr(self.state, "dev_project_path", "")
        if not project_path:
            self.lbl_project_path.setText("No project imported yet.")
            self.lbl_project_status.setText("Choose a folder to enable Developer Mode project checks.")
            self.lbl_project_status.setStyleSheet("color: #64748B;")
            for status in self._file_rows.values():
                status.setText("Not checked")
                status.setStyleSheet("color: #64748B; font-weight: 600;")
            return

        root = Path(project_path)
        self.lbl_project_path.setText(str(root))

        missing_required = []
        for filename in self.REQUIRED_FILES + self.OPTIONAL_FILES:
            path = root / filename
            exists = path.exists()
            status = self._file_rows[filename]
            if exists:
                status.setText("Found")
                status.setStyleSheet("color: #10B981; font-weight: 800;")
            else:
                status.setText("Missing")
                status.setStyleSheet("color: #EF4444; font-weight: 800;")
                if filename in self.REQUIRED_FILES:
                    missing_required.append(filename)

        # at the end of refresh_status, replace the existing btn_training enable logic:
        if missing_required:
            self.btn_training.setEnabled(False)
            self.btn_training.setToolTip("Fix missing required files first.")
        else:
            self.btn_training.setEnabled(True)
            self.btn_training.setToolTip("")

class NeuralForgeApp(QMainWindow):
    """Main application window with persistent sidebar and content stack."""
    def __init__(self):
        super().__init__()
        self.state = ProjectState()
        self.is_dark_mode = True

        self.setWindowTitle("Neural Forge — Enterprise Edition")
        self.setWindowIcon(QIcon("assets/logo.png"))
        self.resize(1200, 800)

        self._init_ui()
        self._apply_current_theme()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Sidebar ──
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 20, 0, 20)
        sidebar_layout.setSpacing(8)

        # Sidebar Header
        sidebar_header = QLabel(" NEURAL FORGE")
        sidebar_header.setStyleSheet("font-weight: 800; font-size: 14pt; letter-spacing: 1px; padding-left: 16px; margin-bottom: 20px;")
        sidebar_layout.addWidget(sidebar_header)

        # Navigation Buttons
        self.nav_buttons = []
        nav_items = [
            ("Home", 0),
            ("Data Lab", 1),
            ("Model Builder", 2),
            ("Train & Evaluate", 3),
            ("Export", 4),
            ("Developer Mode", 5),
            ("AI Assistant", 6),
        ]

        for text, index in nav_items:
            btn = QPushButton(f"  {text}")
            btn.setProperty("class", "SidebarButton")
            btn.clicked.connect(lambda checked, idx=index: self._switch_tab(idx))
            self.nav_buttons.append(btn)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        # Theme Toggle
        self.btn_theme = PremiumToggle(is_dark_mode=self.is_dark_mode)
        self.btn_theme.toggled.connect(self._on_theme_toggled)

        # Center the toggle in the layout
        toggle_layout = QHBoxLayout()
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.btn_theme)
        toggle_layout.addStretch()
        sidebar_layout.addLayout(toggle_layout)

        main_layout.addWidget(self.sidebar)

        # ── Content Stack ──
        self.stack = QStackedWidget()

        # Initialize dashboards
        self.home_dash = HomeDashboard(
            no_code_callback=self._start_no_code_pipeline,
            dev_mode_callback=self._open_dev_mode,
        )
        self.data_dash = DataWindow(self.state, on_next=lambda: self._switch_tab(2))
        self.model_dash = ModelBuilderWindow(self.state, on_back=lambda: self._switch_tab(1), on_next=lambda: self._switch_tab(3))
        self.train_dash = TrainingWindow(self.state, on_back=lambda: self._switch_tab(2))
        self.export_dash = ExportWindow(self.state)
        self.dev_dash = DevProjectWindow(
            self.state,
            on_import=self._open_dev_mode,
            on_no_code=self._start_no_code_pipeline,
            on_train=self._go_to_training,  # ← add this
        )
        self.assistant_dash = AssistantWindow(self.state)

        self.stack.addWidget(self.home_dash)   # 0
        self.stack.addWidget(self.data_dash)   # 1
        self.stack.addWidget(self.model_dash)  # 2
        self.stack.addWidget(self.train_dash)  # 3
        self.stack.addWidget(self.export_dash) # 4
        self.stack.addWidget(self.dev_dash)    # 5
        self.stack.addWidget(self.assistant_dash) # 6

        main_layout.addWidget(self.stack, stretch=1)

        # Set initial state
        self._switch_tab(0)

    def _start_no_code_pipeline(self):
        self.state.training_mode = "nocode"
        self._switch_tab(1)

    def _go_to_training(self):
        self.state.training_mode = "dev"
        self._switch_tab(3)

    def _open_dev_mode(self):
        if ProjectGuideDialog.should_show():
            guide = ProjectGuideDialog(parent=self)
            if guide.exec() != QDialog.DialogCode.Accepted:
                return

        path = QFileDialog.getExistingDirectory(
            self,
            "Import PyTorch Project",
            str(Path.home()),
        )
        if not path:
            return

        self.dev_dash.set_project_path(path)
        self._switch_tab(5)

    def _switch_tab(self, index: int):
        if index == 5 and not getattr(self.state, "dev_project_path", ""):
            self._open_dev_mode()
            return

        # Specific refresh logic before switching
        if index == 2:
            self.model_dash.refresh_data_info()
        elif index == 3:
            self.train_dash.refresh_ui()
        elif index == 4:
            self.export_dash.refresh_status()
        elif index == 5:
            self.dev_dash.refresh_status()
        elif index == 6:
            self.assistant_dash.refresh_status()

        self.stack.setCurrentIndex(index)

        # Update active state of sidebar buttons
        for i, btn in enumerate(self.nav_buttons):
            if i == index:
                btn.setProperty("active", "true")
            else:
                btn.setProperty("active", "false")
            # Force style re-evaluation
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_theme_toggled(self, is_dark: bool):
        self.is_dark_mode = is_dark
        self._apply_current_theme()

    def _apply_current_theme(self):
        apply_theme_palette(QApplication.instance(), self.is_dark_mode)
        self.setStyleSheet(get_qss(self.is_dark_mode))

def main() -> None:
    app = QApplication(sys.argv)
    window = NeuralForgeApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

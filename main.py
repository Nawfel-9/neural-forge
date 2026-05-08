import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QSpacerItem, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap

from ui.styles import apply_theme_palette, get_qss
from ui.window_data import DataWindow
from ui.window_model import ModelBuilderWindow
from ui.window_training import TrainingWindow
from ui.window_export import ExportWindow
from ui.custom_toggle import PremiumToggle
from utils.project_state import ProjectState

class HomeDashboard(QWidget):
    """Welcome screen for the new dashboard."""
    def __init__(self, start_callback):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        logo_label = QLabel()
        pixmap = QPixmap("assets/logo.png")
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
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

        btn_start = QPushButton("Initialize Project Environment")
        btn_start.setProperty("class", "primary")
        btn_start.setFixedSize(300, 50)
        btn_start.setStyleSheet("font-size: 12pt; font-weight: 700;")
        btn_start.clicked.connect(start_callback)
        layout.addWidget(btn_start, alignment=Qt.AlignmentFlag.AlignCenter)


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
            ("Export", 4)
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
        self.home_dash = HomeDashboard(start_callback=lambda: self._switch_tab(1))
        self.data_dash = DataWindow(self.state, on_next=lambda: self._switch_tab(2))
        self.model_dash = ModelBuilderWindow(self.state, on_back=lambda: self._switch_tab(1), on_next=lambda: self._switch_tab(3))
        self.train_dash = TrainingWindow(self.state, on_back=lambda: self._switch_tab(2))
        self.export_dash = ExportWindow(self.state)
        
        self.stack.addWidget(self.home_dash)   # 0
        self.stack.addWidget(self.data_dash)   # 1
        self.stack.addWidget(self.model_dash)  # 2
        self.stack.addWidget(self.train_dash)  # 3
        self.stack.addWidget(self.export_dash) # 4

        main_layout.addWidget(self.stack, stretch=1)

        # Set initial state
        self._switch_tab(0)

    def _switch_tab(self, index: int):
        # Specific refresh logic before switching
        if index == 2:
            self.model_dash.refresh_data_info()
        elif index == 3:
            self.train_dash.refresh_ui()
        elif index == 4:
            self.export_dash.refresh_status()

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

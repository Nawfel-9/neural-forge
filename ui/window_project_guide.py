"""
ui/window_project_guide.py

Developer Mode onboarding dialog — shown the first time a user clicks
"Import Project".  Explains the required project-folder naming conventions
and lets the user suppress future appearances with "Don't show again".
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFrame, QScrollArea, QWidget, QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt, QSettings, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QColor, QFont, QIcon


# ── Reusable "file-row" component ─────────────────────────────────────────────

class FileRow(QFrame):
    """
    A single row showing:   [icon]  filename   description
    """
    _ICONS = {
        "py":   "🐍",
        "yaml": "⚙️",
        "json": "📋",
        "dir":  "📁",
    }

    def __init__(self, filename: str, ext: str, description: str,
                 required: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("fileRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(14)

        # Icon
        icon_lbl = QLabel(self._ICONS.get(ext, "📄"))
        icon_lbl.setFixedWidth(24)
        icon_lbl.setStyleSheet("font-size: 16px;")
        layout.addWidget(icon_lbl)

        # Filename (monospaced)
        name_lbl = QLabel(filename)
        name_lbl.setFont(QFont("Consolas", 10))
        name_lbl.setFixedWidth(160)
        name_lbl.setStyleSheet("color: #e2b96f; font-weight: bold;")
        layout.addWidget(name_lbl)

        # Badge
        badge_text = "REQUIRED" if required else "OPTIONAL"
        badge_color = "#c0392b" if required else "#27ae60"
        badge = QLabel(badge_text)
        badge.setFixedWidth(70)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"background: {badge_color}; color: white; font-size: 9px; "
            f"font-weight: bold; border-radius: 3px; padding: 2px 4px;"
        )
        layout.addWidget(badge)

        # Description
        desc_lbl = QLabel(description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #b0b8c8; font-size: 11px;")
        layout.addWidget(desc_lbl, stretch=1)


# ── Section header ─────────────────────────────────────────────────────────────

class SectionHeader(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            "color: #7fb3d3; font-size: 11px; font-weight: bold; "
            "letter-spacing: 1.5px; text-transform: uppercase; "
            "padding: 12px 0 4px 0;"
        )


# ── Main dialog ────────────────────────────────────────────────────────────────

class ProjectGuideDialog(QDialog):
    """
    Onboarding dialog that explains the Developer Mode project structure.

    Usage
    -----
        if ProjectGuideDialog.should_show():
            dlg = ProjectGuideDialog(parent=self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                # user clicked "Got it — Import Project"
                open_file_chooser()
    """

    _SETTINGS_KEY = "developer_mode/skip_guide"

    # ------------------------------------------------------------------
    @staticmethod
    def should_show() -> bool:
        """Return True unless the user has previously checked 'Don't show again'."""
        settings = QSettings("NeuralForge", "NeuralForge")
        return not settings.value(ProjectGuideDialog._SETTINGS_KEY, False, type=bool)

    # ------------------------------------------------------------------
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Developer Mode — Project Structure Guide")
        self.setMinimumSize(720, 580)
        self.setModal(True)
        self.setObjectName("guideDialog")

        self._apply_styles()
        self._build_ui()
        self._animate_in()

    # ------------------------------------------------------------------
    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog#guideDialog {
                background: #12151c;
                border: 1px solid #2a2f3d;
                border-radius: 10px;
            }

            /* File rows */
            QFrame#fileRow {
                background: #1a1f2e;
                border: 1px solid #252b3b;
                border-radius: 6px;
                margin: 2px 0;
            }
            QFrame#fileRow:hover {
                border-color: #3d5a80;
                background: #1e2538;
            }

            /* Scroll area */
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #1a1f2e;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #3d5a80;
                border-radius: 3px;
                min-height: 20px;
            }

            /* Buttons */
            QPushButton#btnPrimary {
                background: #3d5a80;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 28px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton#btnPrimary:hover {
                background: #4a6fa5;
            }
            QPushButton#btnPrimary:pressed {
                background: #2d4a6e;
            }
            QPushButton#btnSecondary {
                background: transparent;
                color: #7a8499;
                border: 1px solid #2a2f3d;
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 13px;
            }
            QPushButton#btnSecondary:hover {
                color: #aab4c8;
                border-color: #3d4558;
            }

            /* Checkbox */
            QCheckBox {
                color: #7a8499;
                font-size: 12px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #3d4558;
                background: #1a1f2e;
            }
            QCheckBox::indicator:checked {
                background: #3d5a80;
                border-color: #4a6fa5;
                image: none;
            }

            /* Info box */
            QFrame#infoBox {
                background: #0d1b2a;
                border: 1px solid #1e3a5f;
                border-left: 3px solid #3d5a80;
                border-radius: 6px;
                padding: 4px;
            }
        """)

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────
        header_layout = QHBoxLayout()

        badge = QLabel("DEV MODE")
        badge.setStyleSheet(
            "background: #1e3a5f; color: #7fb3d3; font-size: 10px; "
            "font-weight: bold; letter-spacing: 1px; border-radius: 4px; "
            "padding: 4px 8px;"
        )
        badge.setFixedHeight(24)
        header_layout.addWidget(badge)
        header_layout.addStretch()

        root.addLayout(header_layout)
        root.addSpacing(12)

        title = QLabel("Project Structure Guide")
        title.setStyleSheet(
            "color: #e8ecf4; font-size: 22px; font-weight: bold;"
        )
        root.addWidget(title)

        subtitle = QLabel(
            "Before importing, make sure your project follows the naming conventions below.\n"
            "The platform auto-discovers your files by name — no manual linking needed."
        )
        subtitle.setStyleSheet("color: #7a8499; font-size: 12px; line-height: 1.5;")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)
        root.addSpacing(16)

        # ── Scrollable content ───────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(2)

        # Section: Core Python files
        content_layout.addWidget(SectionHeader("Core Python Files"))
        for fname, ext, desc, req in [
            ("model.py",   "py",  "Your nn.Module architecture class. The trainer imports this to instantiate the model.", True),
            ("dataset.py", "py",  "DataLoader logic — augmentations, normalization, resizing and split definitions.", True),
            ("loss.py",    "py",  "Custom loss functions (e.g. Focal Loss, IoU Loss). A default MSE is used if absent.", False),
            ("metrics.py", "py",  "Evaluation logic: mAP, IoU, Accuracy etc. Results are parsed for live graph plotting.", False),
        ]:
            content_layout.addWidget(FileRow(fname, ext, desc, req))

        # Section: Config
        content_layout.addWidget(SectionHeader("Configuration Bridge"))
        content_layout.addWidget(FileRow(
            "config.yaml", "yaml",
            "The sync file. The UI writes hyperparameters here; your scripts read from it at runtime. "
            "Do NOT hardcode LR, batch size, or optimizer — always read from this file.",
            required=True,
        ))

        # Section: Output folder
        content_layout.addWidget(SectionHeader("Expected Output Folder"))
        content_layout.addWidget(FileRow(
            "checkpoints/", "dir",
            "The trainer auto-saves .pth weight files here. The UI monitors this folder for "
            "checkpoint events and displays them in the run history panel.",
            required=False,
        ))
        content_layout.addWidget(FileRow(
            "logs/",        "dir",
            "Optional — place your custom log files here. The terminal panel will tail them live.",
            required=False,
        ))

        # Info callout
        content_layout.addSpacing(12)
        info_box = QFrame()
        info_box.setObjectName("infoBox")
        info_layout = QVBoxLayout(info_box)
        info_layout.setContentsMargins(14, 10, 14, 10)

        info_title = QLabel("💡  How config.yaml works")
        info_title.setStyleSheet("color: #7fb3d3; font-weight: bold; font-size: 12px;")
        info_layout.addWidget(info_title)

        info_text = QLabel(
            "The UI Config Page exports values under standard keys:\n"
            "  learning_rate, batch_size, optimizer, epochs, image_size\n\n"
            "In your scripts, load them with:\n"
            "  import yaml\n"
            "  cfg = yaml.safe_load(open('config.yaml'))\n"
            "  lr = cfg['learning_rate']"
        )
        info_text.setFont(QFont("Consolas", 10))
        info_text.setStyleSheet("color: #8fa8c8; line-height: 1.6;")
        info_layout.addWidget(info_text)
        content_layout.addWidget(info_box)
        content_layout.addStretch()

        scroll.setWidget(content_widget)
        root.addWidget(scroll, stretch=1)
        root.addSpacing(20)

        # ── Bottom bar ───────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        self._dont_show_cb = QCheckBox("Don't show this again")
        bottom.addWidget(self._dont_show_cb)
        bottom.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("btnSecondary")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)

        btn_ok = QPushButton("Got it — Import Project  →")
        btn_ok.setObjectName("btnPrimary")
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.clicked.connect(self._accept)
        btn_ok.setDefault(True)
        bottom.addWidget(btn_ok)

        root.addLayout(bottom)

    # ------------------------------------------------------------------
    def _accept(self):
        if self._dont_show_cb.isChecked():
            settings = QSettings("NeuralForge", "NeuralForge")
            settings.setValue(self._SETTINGS_KEY, True)
        self.accept()

    # ------------------------------------------------------------------
    def _animate_in(self):
        """Subtle fade-in by animating window opacity via a QPropertyAnimation."""
        self.setWindowOpacity(0.0)
        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(220)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()

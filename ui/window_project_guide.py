"""
ui/window_project_guide.py

Developer Mode onboarding dialog — shown the first time a user clicks
"Import Project".  Explains the required project-folder naming conventions
and lets the user suppress future appearances with "Don't show again".
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFrame, QScrollArea, QWidget,
)
from PyQt6.QtCore import Qt, QSettings, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont


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
                 required: bool = True, is_dark: bool = True, parent=None):
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
        # Use amber/gold for filenames to stand out
        name_color = "#e2b96f" if is_dark else "#B45309"
        name_lbl.setStyleSheet(f"color: {name_color}; font-weight: bold;")
        layout.addWidget(name_lbl)

        # Badge
        badge_text = "REQUIRED" if required else "OPTIONAL"
        if required:
            badge_color = "#c0392b" if is_dark else "#DC2626"
        else:
            badge_color = "#27ae60" if is_dark else "#16A34A"
            
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
        desc_color = "#b0b8c8" if is_dark else "#64748B"
        desc_lbl.setStyleSheet(f"color: {desc_color}; font-size: 11px;")
        layout.addWidget(desc_lbl, stretch=1)


# ── Section header ─────────────────────────────────────────────────────────────

class SectionHeader(QLabel):
    def __init__(self, text: str, is_dark: bool = True, parent=None):
        super().__init__(text, parent)
        color = "#7fb3d3" if is_dark else "#0284C7"
        self.setStyleSheet(
            f"color: {color}; font-size: 11px; font-weight: bold; "
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
    def __init__(self, is_dark: bool = True, parent=None):
        super().__init__(parent)
        self.is_dark = is_dark
        self.setWindowTitle("Developer Mode — Project Structure Guide")
        self.setMinimumSize(720, 580)
        self.setModal(True)
        self.setObjectName("guideDialog")

        self._apply_styles()
        self._build_ui()
        self._animate_in()

    # ------------------------------------------------------------------
    def _apply_styles(self):
        # Theme colors
        bg_dialog = "#12151c" if self.is_dark else "#FFFFFF"
        border_dialog = "#2a2f3d" if self.is_dark else "rgba(0, 0, 0, 0.08)"
        bg_row = "#1a1f2e" if self.is_dark else "#F8FAFC"
        border_row = "#252b3b" if self.is_dark else "rgba(0, 0, 0, 0.05)"
        hover_row = "#1e2538" if self.is_dark else "#F1F5F9"
        accent = "#3d5a80" if self.is_dark else "#0284C7"
        accent_hover = "#4a6fa5" if self.is_dark else "#0369A1"
        accent_pressed = "#2d4a6e" if self.is_dark else "#075985"
        text_secondary = "#7a8499" if self.is_dark else "#64748B"
        border_input = "#3d4558" if self.is_dark else "rgba(0, 0, 0, 0.12)"
        bg_info = "#0d1b2a" if self.is_dark else "#F0F9FF"
        border_info = "#1e3a5f" if self.is_dark else "#BAE6FD"

        self.setStyleSheet(f"""
            QDialog#guideDialog {{
                background: {bg_dialog};
                border: 1px solid {border_dialog};
                border-radius: 10px;
            }}

            /* File rows */
            QFrame#fileRow {{
                background: {bg_row};
                border: 1px solid {border_row};
                border-radius: 6px;
                margin: 2px 0;
            }}
            QFrame#fileRow:hover {{
                border-color: {accent};
                background: {hover_row};
            }}

            /* Scroll area */
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: {bg_row};
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {accent};
                border-radius: 3px;
                min-height: 20px;
            }}

            /* Buttons */
            QPushButton#btnPrimary {{
                background: {accent};
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 28px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton#btnPrimary:hover {{
                background: {accent_hover};
            }}
            QPushButton#btnPrimary:pressed {{
                background: {accent_pressed};
            }}
            QPushButton#btnSecondary {{
                background: transparent;
                color: {text_secondary};
                border: 1px solid {border_dialog};
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 13px;
            }}
            QPushButton#btnSecondary:hover {{
                color: {"#aab4c8" if self.is_dark else "#1E293B"};
                border-color: {border_input};
            }}

            /* Checkbox */
            QCheckBox {{
                color: {text_secondary};
                font-size: 12px;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid {border_input};
                background: {bg_row};
            }}
            QCheckBox::indicator:checked {{
                background: {accent};
                border-color: {accent_hover};
                image: none;
            }}

            /* Info box */
            QFrame#infoBox {{
                background: {bg_info};
                border: 1px solid {border_info};
                border-left: 3px solid {accent};
                border-radius: 6px;
                padding: 4px;
            }}
        """)

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────
        header_layout = QHBoxLayout()

        badge_bg = "#1e3a5f" if self.is_dark else "#E0F2FE"
        badge_fg = "#7fb3d3" if self.is_dark else "#0284C7"
        badge = QLabel("DEV MODE")
        badge.setStyleSheet(
            f"background: {badge_bg}; color: {badge_fg}; font-size: 10px; "
            "font-weight: bold; letter-spacing: 1px; border-radius: 4px; "
            "padding: 4px 8px;"
        )
        badge.setFixedHeight(24)
        header_layout.addWidget(badge)
        header_layout.addStretch()

        root.addLayout(header_layout)
        root.addSpacing(12)

        title_color = "#e8ecf4" if self.is_dark else "#0F172A"
        title = QLabel("Project Structure Guide")
        title.setStyleSheet(
            f"color: {title_color}; font-size: 22px; font-weight: bold;"
        )
        root.addWidget(title)

        subtitle_color = "#7a8499" if self.is_dark else "#64748B"
        subtitle = QLabel(
            "Before importing, make sure your project follows the naming conventions below.\n"
            "The platform auto-discovers your files by name — no manual linking needed."
        )
        subtitle.setStyleSheet(f"color: {subtitle_color}; font-size: 12px; line-height: 1.5;")
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
        content_layout.addWidget(SectionHeader("Core Python Files", is_dark=self.is_dark))
        for fname, ext, desc, req in [
            ("model.py",   "py",  "Your nn.Module architecture class. The trainer imports this to instantiate the model.", True),
            ("dataset.py", "py",  "DataLoader logic — augmentations, normalization, resizing and split definitions.", True),
            ("loss.py",    "py",  "Custom loss functions (e.g. Focal Loss, IoU Loss). A default MSE is used if absent.", False),
            ("metrics.py", "py",  "Evaluation logic: mAP, IoU, Accuracy etc. Results are parsed for live graph plotting.", False),
        ]:
            content_layout.addWidget(FileRow(fname, ext, desc, req, is_dark=self.is_dark))

        # Section: Config
        content_layout.addWidget(SectionHeader("Configuration Bridge", is_dark=self.is_dark))
        content_layout.addWidget(FileRow(
            "config.yaml", "yaml",
            "The sync file. The UI writes hyperparameters here; your scripts read from it at runtime. "
            "Do NOT hardcode LR, batch size, or optimizer — always read from this file.",
            required=True, is_dark=self.is_dark
        ))

        # Section: Output folder
        content_layout.addWidget(SectionHeader("Expected Output Folder", is_dark=self.is_dark))
        content_layout.addWidget(FileRow(
            "checkpoints/", "dir",
            "The trainer auto-saves .pth weight files here. The UI monitors this folder for "
            "checkpoint events and displays them in the run history panel.",
            required=False, is_dark=self.is_dark
        ))
        content_layout.addWidget(FileRow(
            "checkpoints/", "dir",
            "The trainer auto-saves .pth weight files here. The UI monitors this folder for "
            "checkpoint events and displays them in the run history panel.",
            required=False, is_dark=self.is_dark
        ))
        content_layout.addWidget(FileRow(
            "logs/",        "dir",
            "Optional — reserve this folder for custom run logs in future Developer Mode execution.",
            required=False, is_dark=self.is_dark
        ))

        # Info callout
        content_layout.addSpacing(12)
        info_box = QFrame()
        info_box.setObjectName("infoBox")
        info_layout = QVBoxLayout(info_box)
        info_layout.setContentsMargins(14, 10, 14, 10)

        info_title_color = "#7fb3d3" if self.is_dark else "#0284C7"
        info_title = QLabel("💡  How config.yaml works")
        info_title.setStyleSheet(f"color: {info_title_color}; font-weight: bold; font-size: 12px;")
        info_layout.addWidget(info_title)

        info_text_color = "#8fa8c8" if self.is_dark else "#334155"
        info_text = QLabel(
            "The UI Config Page exports values under standard keys:\n"
            "  learning_rate, batch_size, optimizer, epochs, image_size\n\n"
            "In your scripts, load them with:\n"
            "  import yaml\n"
            "  cfg = yaml.safe_load(open('config.yaml'))\n"
            "  lr = cfg['learning_rate']"
        )
        info_text.setFont(QFont("Consolas", 10))
        info_text.setStyleSheet(f"color: {info_text_color}; line-height: 1.6;")
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

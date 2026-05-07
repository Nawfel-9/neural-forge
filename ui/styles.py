"""
styles.py
=========
"Glassmorphism Expert" Theme for Neural Forge.
Inspired by high-end SaaS and professional creative suites.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


# ─── Colour tokens (SaaS Premium Palette) ──────────────────────────────────
BG_SPACE     = "#0B0F17"  # Deep Onyx
BG_GLASS     = "rgba(30, 41, 59, 0.7)"  # Slate Glass
BG_INPUT     = "rgba(15, 23, 42, 0.6)"  # Deep Input
BORDER       = "rgba(255, 255, 255, 0.12)"
BORDER_FOCUS = "#38BDF8"  # Sky Blue
TEXT_MAIN    = "#F1F5F9"  # Slate 100
TEXT_MUTED   = "#94A3B8"  # Slate 400
ACCENT       = "#0EA5E9"  # Sky 500
ACCENT_GLOW  = "rgba(14, 165, 233, 0.15)"
SUCCESS      = "#10B981"  # Emerald 500
SUCCESS_GLOW = "rgba(16, 185, 129, 0.15)"
RADIUS       = "16px"
RADIUS_SM    = "8px"


def apply_dark_palette(app: QApplication) -> None:
    """Set a deep space QPalette on the application."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG_SPACE))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_MAIN))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG_SPACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_GLASS))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_GLASS))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_MAIN))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_MAIN))
    palette.setColor(QPalette.ColorRole.Button, QColor(BG_GLASS))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_MAIN))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(BG_SPACE))

    # Disabled colours
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(TEXT_MUTED))
    app.setPalette(palette)


# ─── QSS stylesheet ─────────────────────────────────────────────────────────
DARK_QSS = f"""
/* ── Global ────────────────────────────────────────────────────────────── */
QWidget {{
    background-color: transparent;
    color: {TEXT_MAIN};
    font-family: "Outfit", "Inter", "Segoe UI", sans-serif;
    font-size: 10pt;
    outline: none;
}}

QMainWindow, QDialog, QStackedWidget {{
    background-color: {BG_SPACE};
}}

/* ── Tab Widget (Modern Segmented Control) ──────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    background-color: {BG_GLASS};
    top: -1px;
}}

QTabBar {{
    background: transparent;
    qproperty-drawBase: 0;
}}

QTabBar::tab {{
    background: {BG_INPUT};
    color: {TEXT_MUTED};
    padding: 10px 20px;
    margin-right: 4px;
    margin-bottom: 8px;
    font-weight: 700;
    font-size: 9pt;
    border-radius: {RADIUS_SM};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border: 1px solid {BORDER};
}}

QTabBar::tab:hover {{
    color: {TEXT_MAIN};
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.2);
}}

QTabBar::tab:selected {{
    color: {TEXT_MAIN};
    background: {ACCENT};
    border: 1px solid {ACCENT};
}}

/* ── Group Box (Elegant Cards) ─────────────────────────────────────────── */
QGroupBox {{
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    margin-top: 28px;
    padding-top: 20px;
    padding-bottom: 12px;
    padding-left: 12px;
    padding-right: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    top: 6px;
    padding: 2px 10px;
    color: {ACCENT};
    font-weight: 800;
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 1px;
    background-color: {BG_SPACE};
    border-radius: 4px;
    border: 1px solid {BORDER};
}}

/* ── Buttons (Premium SaaS) ────────────────────────────────────────────── */
QPushButton {{
    background-color: {BG_INPUT};
    color: {TEXT_MAIN};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    padding: 10px 20px;
    font-weight: 700;
    min-height: 32px;
}}

QPushButton:hover {{
    background-color: rgba(255, 255, 255, 0.08);
    border-color: {BORDER_FOCUS};
}}

QPushButton:pressed {{
    background-color: rgba(255, 255, 255, 0.12);
}}

QPushButton[class="primary"] {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT}, stop:1 #0284C7);
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.1);
}}

QPushButton[class="primary"]:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #38BDF8, stop:1 {ACCENT});
    border-color: rgba(255, 255, 255, 0.3);
}}

/* ── Inputs (High Contrast & Clear) ───────────────────────────────────── */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QListWidget, QTextEdit {{
    background-color: {BG_INPUT};
    color: {TEXT_MAIN};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    padding: 8px 12px;
    min-height: 38px;
    selection-background-color: {ACCENT};
}}

QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QLineEdit:hover, QListWidget:hover {{
    border-color: rgba(255, 255, 255, 0.2);
}}

QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus, QListWidget:focus {{
    border-color: {BORDER_FOCUS};
    background-color: rgba(15, 23, 42, 0.8);
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {TEXT_MUTED};
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_SPACE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    selection-background-color: {ACCENT};
    color: {TEXT_MAIN};
    padding: 4px;
}}

/* ── Table View ────────────────────────────────────────────────────────── */
QTableView {{
    background-color: rgba(0, 0, 0, 0.2);
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    gridline-color: rgba(255, 255, 255, 0.03);
    selection-background-color: {ACCENT_GLOW};
    selection-color: {TEXT_MAIN};
}}

QHeaderView::section {{
    background-color: {BG_INPUT};
    color: {TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 12px;
    font-weight: 800;
    text-transform: uppercase;
    font-size: 7.5pt;
    letter-spacing: 1px;
}}

/* ── ScrollBar ──────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    border: none;
    background: transparent;
    width: 8px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER};
    min-height: 20px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── Progress Bar ───────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT}, stop:1 {SUCCESS});
    border-radius: 3px;
}}

/* ── Checkbox ──────────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {BORDER};
    background: {BG_INPUT};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QCheckBox::indicator:hover {{
    border-color: {BORDER_FOCUS};
}}
"""


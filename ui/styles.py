"""
styles.py
=========
Dark-theme QSS stylesheet and QPalette configuration for the entire app.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


# ─── Colour tokens ──────────────────────────────────────────────────────────
BG_DARKEST   = "#0d1117"
BG_DARK      = "#161b22"
BG_CARD      = "#1c2129"
BG_INPUT     = "#22272e"
BORDER       = "#30363d"
TEXT_PRIMARY  = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
ACCENT       = "#58a6ff"
ACCENT_HOVER = "#79c0ff"
DANGER       = "#f85149"
DANGER_HOVER = "#ff7b72"
SUCCESS      = "#3fb950"
WARNING      = "#d29922"


def apply_dark_palette(app: QApplication) -> None:
    """Set a dark QPalette on the application."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG_DARKEST))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG_INPUT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_DARK))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(BG_DARK))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(BG_DARKEST))

    # Disabled colours
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor(TEXT_SECONDARY),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(TEXT_SECONDARY),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(TEXT_SECONDARY),
    )
    app.setPalette(palette)


# ─── QSS stylesheet ─────────────────────────────────────────────────────────
DARK_QSS = f"""
/* ── Global ────────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {BG_DARKEST};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10pt;
}}

/* ── Scroll area ───────────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background: {BG_DARKEST};
}}
QScrollArea > QWidget > QWidget {{
    background: {BG_DARKEST};
}}

/* ── Cards (layer rows, panels) ────────────────────────────────────────── */
QFrame[frameShape="1"] {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px;
}}

/* ── Labels ────────────────────────────────────────────────────────────── */
QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QLabel[class="heading"] {{
    font-size: 18px;
    font-weight: 700;
}}
QLabel[class="subheading"] {{
    font-size: 14px;
    font-weight: 600;
    color: {TEXT_SECONDARY};
}}

/* ── Buttons ───────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {BG_CARD};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {ACCENT};
    color: {BG_DARKEST};
}}
QPushButton:disabled {{
    background-color: {BG_INPUT};
    color: {TEXT_SECONDARY};
    border-color: {BORDER};
}}

QPushButton[class="primary"] {{
    background-color: {ACCENT};
    color: {BG_DARKEST};
    border: none;
}}
QPushButton[class="primary"]:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton[class="danger"] {{
    background-color: transparent;
    color: {DANGER};
    border: 1px solid {DANGER};
}}
QPushButton[class="danger"]:hover {{
    background-color: {DANGER};
    color: {BG_DARKEST};
}}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 24px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: {BG_DARKEST};
}}

/* ── Scroll bars ───────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG_DARKEST};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    min-height: 30px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_SECONDARY};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── Table view ────────────────────────────────────────────────────────── */
QTableView {{
    background-color: {BG_DARK};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QHeaderView::section {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    padding: 6px;
    font-weight: 600;
}}

/* ── Group box ─────────────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 18px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: {ACCENT};
}}

/* ── Checkbox ──────────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    color: {TEXT_PRIMARY};
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

/* ── Tooltips ──────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px;
}}
"""

"""
styles.py
=========
"Premium SaaS" Theme for Neural Forge.
Supports dynamic Light and Dark mode toggling.
"""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

# ─── Dark Mode Tokens ("Deep Night") ───
DARK_BG_SPACE     = "#0B0F17"
DARK_BG_SIDEBAR   = "#111827"
DARK_BG_CARD      = "rgba(30, 41, 59, 0.45)"
DARK_BG_INPUT     = "rgba(15, 23, 42, 0.6)"
DARK_BORDER       = "rgba(255, 255, 255, 0.06)"
DARK_BORDER_FOCUS = "#38BDF8"
DARK_TEXT_MAIN    = "#F1F5F9"
DARK_TEXT_MUTED   = "#64748B"
DARK_ACCENT       = "#0EA5E9"
DARK_SUCCESS      = "#10B981"
DARK_DANGER       = "#EF4444"

# ─── Light Mode Tokens ("Clean Glass") ───
LIGHT_BG_SPACE     = "#F8FAFC"
LIGHT_BG_SIDEBAR   = "#FFFFFF"
LIGHT_BG_CARD      = "#FFFFFF"
LIGHT_BG_INPUT     = "#F1F5F9"
LIGHT_BORDER       = "rgba(0, 0, 0, 0.08)"
LIGHT_BORDER_FOCUS = "#0EA5E9"
LIGHT_TEXT_MAIN    = "#0F172A"
LIGHT_TEXT_MUTED   = "#64748B"
LIGHT_ACCENT       = "#0284C7"
LIGHT_SUCCESS      = "#059669"
LIGHT_DANGER       = "#DC2626"

RADIUS       = "12px"
RADIUS_SM    = "6px"

def apply_theme_palette(app: QApplication, is_dark: bool = True) -> None:
    """Set the QPalette on the application based on the theme."""
    palette = QPalette()

    bg_space = DARK_BG_SPACE if is_dark else LIGHT_BG_SPACE
    text_main = DARK_TEXT_MAIN if is_dark else LIGHT_TEXT_MAIN
    accent = DARK_ACCENT if is_dark else LIGHT_ACCENT
    text_muted = DARK_TEXT_MUTED if is_dark else LIGHT_TEXT_MUTED

    palette.setColor(QPalette.ColorRole.Window, QColor(bg_space))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(text_main))
    palette.setColor(QPalette.ColorRole.Base, QColor(bg_space))

    # QColor constructor cannot parse "rgba()" strings, returning black.
    # Use hex colors for AlternateBase to fix the pitch-black row bug.
    alt_base = "#121A28" if is_dark else "#F1F5F9"
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(alt_base))

    # Tooltips can still use the glass color if we want, but since QColor
    # fails on rgba(), let's use a solid hex for ToolTipBase too.
    tt_base = "#1E293B" if is_dark else "#FFFFFF"
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(tt_base))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(text_main))
    palette.setColor(QPalette.ColorRole.Text, QColor(text_main))
    btn_bg = "#1E293B" if is_dark else "#E2E8F0"
    palette.setColor(QPalette.ColorRole.Button, QColor(btn_bg))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(text_main))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(accent))
    palette.setColor(QPalette.ColorRole.Link, QColor(accent))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(bg_space))

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(text_muted))
    app.setPalette(palette)


def get_qss(is_dark: bool = True) -> str:
    bg_space     = DARK_BG_SPACE if is_dark else LIGHT_BG_SPACE
    bg_sidebar   = DARK_BG_SIDEBAR if is_dark else LIGHT_BG_SIDEBAR
    bg_card      = DARK_BG_CARD if is_dark else LIGHT_BG_CARD
    bg_input     = DARK_BG_INPUT if is_dark else LIGHT_BG_INPUT
    border       = DARK_BORDER if is_dark else LIGHT_BORDER
    border_focus = DARK_BORDER_FOCUS if is_dark else LIGHT_BORDER_FOCUS
    text_main    = DARK_TEXT_MAIN if is_dark else LIGHT_TEXT_MAIN
    text_muted   = DARK_TEXT_MUTED if is_dark else LIGHT_TEXT_MUTED
    accent       = DARK_ACCENT if is_dark else LIGHT_ACCENT
    success      = DARK_SUCCESS if is_dark else LIGHT_SUCCESS
    danger       = DARK_DANGER if is_dark else LIGHT_DANGER

    hover_overlay = "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.03)"
    pressed_overlay = "rgba(255, 255, 255, 0.1)" if is_dark else "rgba(0, 0, 0, 0.06)"

    return f"""
/* ── Global ────────────────────────────────────────────────────────────── */
QWidget {{
    background-color: transparent;
    color: {text_main};
    font-family: "Outfit", "Inter", "Segoe UI", sans-serif;
    font-size: 10pt;
    outline: none;
}}

QMainWindow, QDialog, QStackedWidget {{
    background-color: {bg_space};
}}

/* ── Sidebar & Navigation ──────────────────────────────────────────────── */
QWidget#Sidebar {{
    background-color: {bg_sidebar};
    border-right: 1px solid {border};
}}

QPushButton.SidebarButton {{
    text-align: left;
    padding: 12px 16px;
    background: transparent;
    border: none;
    border-radius: {RADIUS_SM};
    color: {text_muted};
    font-weight: 600;
    font-size: 11pt;
}}

QPushButton.SidebarButton:hover {{
    background: {hover_overlay};
    color: {text_main};
}}

QPushButton.SidebarButton[active="true"] {{
    background: {bg_input};
    color: {accent};
    font-weight: 800;
    border-left: 4px solid {accent};
    border-radius: 0px;
    border-top-right-radius: {RADIUS_SM};
    border-bottom-right-radius: {RADIUS_SM};
}}

QPushButton.ThemeToggle {{
    text-align: center;
    padding: 10px 16px;
    background: {bg_input};
    border: 1px solid {border};
    border-radius: {RADIUS};
    color: {text_main};
    font-weight: 700;
    font-size: 10pt;
    margin: 0px 12px;
}}

QPushButton.ThemeToggle:hover {{
    background: {hover_overlay};
    border-color: {border_focus};
    color: {accent};
}}

/* ── Typography ────────────────────────────────────────────────────────── */
QLabel.PageTitle {{
    font-size: 24pt;
    font-weight: 800;
    letter-spacing: -0.5px;
}}
QLabel.PageSubtitle {{
    font-size: 11pt;
    color: {text_muted};
}}

QLabel.HeroTitle {{
    font-size: 42pt;
    font-weight: 900;
    letter-spacing: -1px;
    color: {accent};
}}

QLabel.HeroSubtitle {{
    font-size: 14pt;
    color: {text_muted};
    font-weight: 500;
}}

/* ── Glass Cards ───────────────────────────────────────────────────────── */
QFrame.GlassCard {{
    background-color: {bg_card};
    border: 1px solid {border};
    border-radius: {RADIUS};
}}

QFrame.GlassCard:hover {{
    background-color: {hover_overlay};
    border-color: {border_focus};
}}

/* ── Group Box (Dashboard Cards) ───────────────────────────────────────── */
QGroupBox {{
    background-color: {bg_card};
    border: 1px solid {border};
    border-radius: {RADIUS};
    margin-top: 20px;
    padding-top: 18px;
    padding-bottom: 12px;
    padding-left: 16px;
    padding-right: 16px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    top: 0px;
    padding: 2px 8px;
    color: {accent};
    font-weight: 800;
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 1px;
    background-color: {bg_space};
    border-radius: 4px;
    border: 1px solid {border};
}}

/* ── Buttons ───────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {bg_input};
    color: {text_main};
    border: 1px solid {border};
    border-radius: {RADIUS_SM};
    padding: 8px 16px;
    font-weight: 600;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: {hover_overlay};
    border-color: {border_focus};
}}

QPushButton:pressed {{
    background-color: {pressed_overlay};
}}

QPushButton[class="primary"] {{
    background-color: {accent};
    color: #FFFFFF;
    border: none;
}}

QPushButton[class="primary"]:hover {{
    background-color: {border_focus};
}}

QPushButton[class="primary"]:disabled {{
    background-color: {text_muted};
    color: rgba(255, 255, 255, 0.4);
}}

QPushButton[class="danger"] {{
    background-color: transparent;
    color: {text_muted};
    border: 1px solid transparent;
    font-size: 18pt;
    font-weight: 300;
    padding: 0px;
    margin: 0px;
}}

QPushButton[class="danger"]:hover {{
    background-color: rgba(220, 38, 38, 0.08);
    color: {danger};
    border: 1px solid rgba(220, 38, 38, 0.3);
    border-radius: {RADIUS_SM};
}}

/* ── Inputs ────────────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QListWidget, QTextEdit {{
    background-color: {bg_input};
    color: {text_main};
    border: 1px solid {border};
    border-radius: {RADIUS_SM};
    padding: 6px 8px;
    min-height: 20px;
    selection-background-color: {accent};
    selection-color: #ffffff;
}}

QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QLineEdit:hover, QListWidget:hover {{
    border-color: {border_focus};
}}

QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus, QListWidget:focus {{
    border-color: {border_focus};
    background-color: {bg_space};
}}


QComboBox QAbstractItemView {{
    background-color: {bg_space};
    border: 1px solid {border};
    border-radius: {RADIUS_SM};
    selection-background-color: {accent};
    color: {text_main};
    padding: 4px;
}}

/* ── Table View ────────────────────────────────────────────────────────── */
QTableView {{
    background-color: transparent;
    border: 1px solid {border};
    border-radius: {RADIUS_SM};
    gridline-color: {border};
    selection-background-color: {accent};
    selection-color: #ffffff;
}}

QHeaderView::section {{
    background-color: {bg_input};
    color: {text_muted};
    border: none;
    border-bottom: 1px solid {border};
    border-right: 1px solid {border};
    padding: 8px 12px;
    font-weight: 700;
    font-size: 8pt;
}}

/* ── ScrollBar ──────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    border: none;
    background: transparent;
    width: 8px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {border};
    min-height: 20px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {text_muted};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── Progress Bar ───────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {bg_input};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: right;
    color: {text_muted};
    padding-right: 4px;
    font-size: 8pt;
}}

QProgressBar::chunk {{
    background-color: {success};
    border-radius: 4px;
}}

/* ── Checkbox ──────────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {border};
    background: {bg_input};
}}

QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}

QCheckBox::indicator:hover {{
    border-color: {border_focus};
}}

/* ── Specific Overrides ────────────────────────────────────────────────── */
QFrame#layerRow, QFrame#inputLayerRow {{
    background-color: {bg_card};
    border: 1px solid {border};
    border-radius: {RADIUS_SM};
}}

QFrame#layerRow:hover {{
    border-color: {border_focus};
    background-color: {hover_overlay};
}}

QTextEdit.CodeConsole {{
    background-color: {bg_input};
    color: {text_main};
    font-family: "Consolas", "Monaco", "Courier New", monospace;
    border: 1px solid {border};
    border-radius: {RADIUS_SM};
    font-size: 9pt;
}}

QLabel.StatText {{
    color: {text_muted};
    font-family: "Consolas", "Monaco", "Courier New", monospace;
    font-size: 9pt;
}}
"""

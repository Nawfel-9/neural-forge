"""
ui/window_project_validation.py

Project Validation Window — Developer Mode (Classification)
------------------------------------------------------------
Performs static AST analysis on the imported project folder.
No user code is imported or executed at this stage.

Checks performed
────────────────
  model.py      → exists  +  defines build_model(cfg)
  dataset.py    → exists  +  defines build_dataloaders(cfg)
  loss.py       → exists (optional) + defines build_criterion(cfg)
  metrics.py    → exists (optional) + defines compute_metrics(outputs, targets, cfg)
  config.yaml   → existence noted; absence is a WARNING, not a BLOCK
                  (will be created by the Config Page)

Result
──────
  All REQUIRED checks pass  →  "Continue to Config" button enabled
  Any REQUIRED check fails  →  button disabled, user must fix and re-scan
"""

import ast
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QPropertyAnimation,
                          QEasingCurve, QTimer)
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGraphicsOpacityEffect, QSizePolicy,
)


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

class CheckStatus(Enum):
    PENDING  = auto()
    RUNNING  = auto()
    OK       = auto()
    WARNING  = auto()
    ERROR    = auto()


@dataclass
class CheckResult:
    label:    str
    status:   CheckStatus = CheckStatus.PENDING
    detail:   str = ""
    required: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# AST helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ast_function_names(path: Path) -> Optional[List[str]]:
    """Return all top-level function names in *path*, or None on parse error."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree   = ast.parse(source, filename=str(path))
        return [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    except SyntaxError as exc:
        return None   # caller will report the error


def _check_file(
    folder: Path,
    filename: str,
    required_fn: Optional[str],
    required: bool,
) -> CheckResult:
    label = filename

    filepath = folder / filename
    if not filepath.exists():
        if required:
            return CheckResult(label, CheckStatus.ERROR,
                               f"{filename} not found in project folder.",
                               required=required)
        else:
            return CheckResult(label, CheckStatus.WARNING,
                               f"{filename} not found — platform default will be used.",
                               required=required)

    if required_fn is None:
        # Just existence check (e.g. config.yaml)
        return CheckResult(label, CheckStatus.OK,
                           f"{filename} found.", required=required)

    names = _ast_function_names(filepath)
    if names is None:
        return CheckResult(label, CheckStatus.ERROR,
                           f"{filename} has a syntax error — could not parse.",
                           required=required)

    if required_fn not in names:
        status = CheckStatus.ERROR if required else CheckStatus.WARNING
        detail = (
            f"'{required_fn}' not found in {filename}. "
            f"Found: {', '.join(names[:6]) or 'no functions'}."
        )
        return CheckResult(label, status, detail, required=required)

    return CheckResult(label, CheckStatus.OK,
                       f"'{required_fn}' detected.", required=required)


# ══════════════════════════════════════════════════════════════════════════════
# Background worker
# ══════════════════════════════════════════════════════════════════════════════

class ValidationWorker(QThread):
    """Runs all checks off the main thread, emits one result at a time."""

    result_ready = pyqtSignal(int, object)   # (index, CheckResult)
    finished_all = pyqtSignal(bool)          # True = all required passed

    _CHECKS = [
        # (filename,     required_fn,          required)
        ("model.py",    "build_model",         True),
        ("dataset.py",  "build_dataloaders",   True),
        ("loss.py",     "build_criterion",     False),
        ("metrics.py",  "compute_metrics",     False),
        ("config.yaml", None,                  False),
    ]

    def __init__(self, project_dir: str):
        super().__init__()
        self._dir = Path(project_dir)

    def run(self):
        all_ok = True
        for i, (fname, fn_name, required) in enumerate(self._CHECKS):
            self.msleep(260)   # brief pause so the UI animates visibly
            result = _check_file(self._dir, fname, fn_name, required)
            if required and result.status == CheckStatus.ERROR:
                all_ok = False
            self.result_ready.emit(i, result)
        self.finished_all.emit(all_ok)


# ══════════════════════════════════════════════════════════════════════════════
# Check row widget
# ══════════════════════════════════════════════════════════════════════════════

_STATUS_CONFIG = {
    CheckStatus.PENDING: ("#3a3f50", "···",  "#5a6070"),
    CheckStatus.RUNNING: ("#1a2540", "···",  "#4a7fc1"),
    CheckStatus.OK:      ("#0d2818", "✓",    "#2ecc71"),
    CheckStatus.WARNING: ("#2a1f08", "⚠",    "#f39c12"),
    CheckStatus.ERROR:   ("#2a0d0d", "✕",    "#e74c3c"),
}

class CheckRowWidget(QFrame):
    def __init__(self, filename: str, required: bool, parent=None):
        super().__init__(parent)
        self.setObjectName("checkRow")
        self.setFixedHeight(62)
        self._status = CheckStatus.PENDING

        # Fade-in effect
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(16)

        # Status indicator circle
        self._indicator = QLabel("···")
        self._indicator.setFixedSize(28, 28)
        self._indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._indicator.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        layout.addWidget(self._indicator)

        # Filename + badge
        name_col = QVBoxLayout()
        name_col.setSpacing(2)

        name_row = QHBoxLayout()
        self._name_lbl = QLabel(filename)
        self._name_lbl.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self._name_lbl.setStyleSheet("color: #dce4f0;")
        name_row.addWidget(self._name_lbl)

        badge_text  = "REQUIRED" if required else "OPTIONAL"
        badge_color = "#3d1a1a" if required else "#1a2e1a"
        badge_fg    = "#e74c3c" if required else "#2ecc71"
        badge = QLabel(badge_text)
        badge.setStyleSheet(
            f"background:{badge_color}; color:{badge_fg}; font-size:9px; "
            f"font-weight:bold; border-radius:3px; padding:2px 6px; "
            f"border: 1px solid {badge_fg}40;"
        )
        badge.setFixedHeight(18)
        name_row.addWidget(badge)
        name_row.addStretch()
        name_col.addLayout(name_row)

        self._detail_lbl = QLabel("Waiting…")
        self._detail_lbl.setStyleSheet("color: #5a6070; font-size: 11px;")
        name_col.addWidget(self._detail_lbl)

        layout.addLayout(name_col, stretch=1)
        self._set_style(CheckStatus.PENDING)

    # ------------------------------------------------------------------
    def reveal(self, delay_ms: int = 0):
        """Fade this row in after delay_ms milliseconds."""
        QTimer.singleShot(delay_ms, self._do_reveal)

    def _do_reveal(self):
        anim = QPropertyAnimation(self._opacity, b"opacity", self)
        anim.setDuration(300)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    # ------------------------------------------------------------------
    def update_result(self, result: CheckResult):
        self._status = result.status
        self._detail_lbl.setText(result.detail)
        self._set_style(result.status)

    def set_running(self):
        self._status = CheckStatus.RUNNING
        self._detail_lbl.setText("Scanning…")
        self._set_style(CheckStatus.RUNNING)

    def _set_style(self, status: CheckStatus):
        bg, symbol, color = _STATUS_CONFIG[status]
        self._indicator.setText(symbol)
        self._indicator.setStyleSheet(
            f"color: {color}; background: {color}18; "
            f"border: 1px solid {color}50; border-radius: 14px;"
        )
        self.setStyleSheet(
            f"QFrame#checkRow {{ background: {bg}; border: 1px solid {color}30; "
            f"border-radius: 8px; margin: 3px 0; }}"
        )
        if status == CheckStatus.OK:
            self._detail_lbl.setStyleSheet("color: #2ecc71aa; font-size: 11px;")
        elif status == CheckStatus.ERROR:
            self._detail_lbl.setStyleSheet("color: #e74c3ccc; font-size: 11px;")
        elif status == CheckStatus.WARNING:
            self._detail_lbl.setStyleSheet("color: #f39c12bb; font-size: 11px;")
        else:
            self._detail_lbl.setStyleSheet("color: #5a6070; font-size: 11px;")


# ══════════════════════════════════════════════════════════════════════════════
# Summary banner
# ══════════════════════════════════════════════════════════════════════════════

class SummaryBanner(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("summaryBanner")
        self.setFixedHeight(52)
        self.hide()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)

        self._icon  = QLabel()
        self._icon.setFont(QFont("Segoe UI Emoji", 16))
        layout.addWidget(self._icon)

        self._text  = QLabel()
        self._text.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        layout.addWidget(self._text)
        layout.addStretch()

    def show_result(self, success: bool):
        if success:
            self._icon.setText("🟢")
            self._text.setText("All required checks passed — ready to configure.")
            self.setStyleSheet(
                "QFrame#summaryBanner { background: #0a2318; border: 1px solid #2ecc7160; "
                "border-radius: 8px; } QLabel { color: #2ecc71; }"
            )
        else:
            self._icon.setText("🔴")
            self._text.setText("One or more required files are missing or invalid.")
            self.setStyleSheet(
                "QFrame#summaryBanner { background: #200a0a; border: 1px solid #e74c3c60; "
                "border-radius: 8px; } QLabel { color: #e74c3c; }"
            )

        eff = QGraphicsOpacityEffect(self)
        eff.setOpacity(0.0)
        self.setGraphicsEffect(eff)
        self.show()

        anim = QPropertyAnimation(eff, b"opacity", self)
        anim.setDuration(400)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


# ══════════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════════

class ProjectValidationWindow(QWidget):
    """
    Shown immediately after the user picks a project folder.

    Signals
    -------
    Callbacks passed in constructor:
        on_back()          — user wants to pick a different folder
        on_continue(path)  — validation passed, proceed to Config Page
    """

    def __init__(self, project_dir: str, on_back, on_continue, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VisionHub — Project Validation")
        self.setMinimumSize(680, 560)
        self._project_dir = project_dir
        self._on_back     = on_back
        self._on_continue = on_continue
        self._worker: Optional[ValidationWorker] = None
        self._rows: List[CheckRowWidget] = []
        self._validation_passed = False

        self._apply_styles()
        self._build_ui()
        self._start_scan()

    # ------------------------------------------------------------------
    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background: #0e1118;
                color: #dce4f0;
                font-family: 'Segoe UI', sans-serif;
            }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: #1a1f2e; width: 5px; border-radius: 2px;
            }
            QScrollBar::handle:vertical {
                background: #3d5a80; border-radius: 2px; min-height: 20px;
            }
            QPushButton#btnPrimary {
                background: #2a5298;
                color: #ffffff;
                border: none;
                border-radius: 7px;
                padding: 11px 32px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton#btnPrimary:hover  { background: #3464b8; }
            QPushButton#btnPrimary:pressed { background: #1e3e72; }
            QPushButton#btnPrimary:disabled {
                background: #1e2535;
                color: #3a4255;
            }
            QPushButton#btnSecondary {
                background: transparent;
                color: #5a6a80;
                border: 1px solid #252d3d;
                border-radius: 7px;
                padding: 11px 24px;
                font-size: 13px;
            }
            QPushButton#btnSecondary:hover {
                color: #8a9ab0;
                border-color: #3a4560;
            }
            QPushButton#btnRescan {
                background: transparent;
                color: #4a7fc1;
                border: 1px solid #2a4a70;
                border-radius: 7px;
                padding: 8px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton#btnRescan:hover { background: #0d1e35; }
        """)

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 30, 36, 26)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────
        top = QHBoxLayout()
        tag = QLabel("STATIC ANALYSIS")
        tag.setStyleSheet(
            "background: #0d1e35; color: #4a7fc1; font-size: 10px; "
            "font-weight: bold; letter-spacing: 1.5px; border-radius: 4px; "
            "padding: 4px 10px; border: 1px solid #1e3a5f;"
        )
        top.addWidget(tag)
        top.addStretch()

        self._rescan_btn = QPushButton("↺  Re-scan")
        self._rescan_btn.setObjectName("btnRescan")
        self._rescan_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rescan_btn.clicked.connect(self._start_scan)
        self._rescan_btn.setEnabled(False)
        top.addWidget(self._rescan_btn)
        root.addLayout(top)
        root.addSpacing(14)

        # ── Title ────────────────────────────────────────────────────
        title = QLabel("Validating Project Structure")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #e8ecf4;")
        root.addWidget(title)

        # Project path label
        short_path = self._project_dir
        if len(short_path) > 72:
            short_path = "…" + short_path[-70:]
        path_lbl = QLabel(f"📂  {short_path}")
        path_lbl.setFont(QFont("Consolas", 10))
        path_lbl.setStyleSheet("color: #4a5a70; margin-top: 2px;")
        path_lbl.setWordWrap(True)
        root.addWidget(path_lbl)
        root.addSpacing(22)

        # ── Check rows (scrollable) ──────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        rows_widget = QWidget()
        rows_layout = QVBoxLayout(rows_widget)
        rows_layout.setContentsMargins(0, 0, 8, 0)
        rows_layout.setSpacing(0)

        _FILES = [
            ("model.py",    True),
            ("dataset.py",  True),
            ("loss.py",     False),
            ("metrics.py",  False),
            ("config.yaml", False),
        ]
        for i, (fname, required) in enumerate(_FILES):
            row = CheckRowWidget(fname, required)
            row.reveal(delay_ms=i * 80)
            rows_layout.addWidget(row)
            self._rows.append(row)

        rows_layout.addStretch()
        scroll.setWidget(rows_widget)
        root.addWidget(scroll, stretch=1)
        root.addSpacing(16)

        # ── Summary banner ───────────────────────────────────────────
        self._banner = SummaryBanner()
        root.addWidget(self._banner)
        root.addSpacing(16)

        # ── Bottom buttons ───────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        btn_back = QPushButton("← Choose Different Folder")
        btn_back.setObjectName("btnSecondary")
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self._on_back)
        bottom.addWidget(btn_back)

        bottom.addStretch()

        self._continue_btn = QPushButton("Continue to Config  →")
        self._continue_btn.setObjectName("btnPrimary")
        self._continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._continue_btn.setEnabled(False)
        self._continue_btn.clicked.connect(
            lambda: self._on_continue(self._project_dir)
        )
        bottom.addWidget(self._continue_btn)

        root.addLayout(bottom)

    # ------------------------------------------------------------------
    def _start_scan(self):
        """Reset all rows and launch the worker thread."""
        self._validation_passed = False
        self._continue_btn.setEnabled(False)
        self._rescan_btn.setEnabled(False)
        self._banner.hide()

        for row in self._rows:
            row.update_result(CheckResult("", CheckStatus.PENDING))

        self._worker = ValidationWorker(self._project_dir)
        self._worker.result_ready.connect(self._on_result)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.start()

        # Mark first row as running immediately
        if self._rows:
            self._rows[0].set_running()

    # ------------------------------------------------------------------
    def _on_result(self, index: int, result: CheckResult):
        self._rows[index].update_result(result)
        # Pre-mark the next row as running
        if index + 1 < len(self._rows):
            self._rows[index + 1].set_running()

    def _on_finished(self, success: bool):
        self._validation_passed = success
        self._continue_btn.setEnabled(success)
        self._rescan_btn.setEnabled(True)
        self._banner.show_result(success)

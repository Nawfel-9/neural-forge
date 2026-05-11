"""
window_assistant.py
===================
AI Assistant chat tab for NVIDIA's OpenAI-compatible API.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QColor

from backend.assistant_client import build_project_context, is_assistant_configured, get_assistant_settings
from utils.project_state import ProjectState
from workers.assistant_worker import AssistantWorker



class MessageBubble(QFrame):
    """Premium message bubble with shadow and entrance animation."""
    def __init__(self, role: str, text: str, is_dark: bool = True, parent=None):
        super().__init__(parent)
        self.role = role
        self.is_dark = is_dark
        self._build_ui(text)
        self.apply_theme(is_dark)
        self._animate_entry()

    def _build_ui(self, text: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        
        self.label = QLabel(text)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setWordWrap(True)
        self.label.setMinimumWidth(300)
        # Span up to 90% of a typical desktop layout width
        self.label.setMaximumWidth(1200)
        layout.addWidget(self.label)

        # Subtle shadow for depth
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 45))
        self.setGraphicsEffect(shadow)

    def apply_theme(self, is_dark: bool):
        self.is_dark = is_dark
        if self.role == "user":
            self.setStyleSheet("""
                MessageBubble {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0EA5E9, stop:1 #2563EB);
                    color: white;
                    border-radius: 16px;
                    border-bottom-right-radius: 2px;
                }
                QLabel { color: white; font-weight: 600; }
            """)
        else:
            bg = "rgba(30, 41, 59, 0.7)" if is_dark else "#F1F5F9"
            border = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.06)"
            text = "#F1F5F9" if is_dark else "#0F172A"
            self.setStyleSheet(f"""
                MessageBubble {{
                    background-color: {bg};
                    border: 1px solid {border};
                    border-radius: 16px;
                    border-top-left-radius: 2px;
                }}
                QLabel {{ color: {text}; }}
            """)

    def _animate_entry(self):
        self.setWindowOpacity(0)
        self.anim = QPropertyAnimation(self, b"pos")
        self.anim.setDuration(450)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # We'll set the start position in the layout logic or just fade in
        self.fade = QPropertyAnimation(self, b"windowOpacity")
        self.fade.setDuration(300)
        self.fade.setStartValue(0.0)
        self.fade.setEndValue(1.0)
        self.fade.start()

    def setText(self, text: str):
        self.label.setText(text)


class AssistantWindow(QWidget):
    """Chat interface that can answer using current Neural Forge project context."""

    def __init__(self, project_state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = project_state
        self._messages: list[dict[str, str]] = []
        self._worker: AssistantWorker | None = None
        self._active_assistant_label: QLabel | None = None
        self._active_response = ""
        self._reasoning_preview = ""
        self._wait_seconds = 0
        self._wait_timer = QTimer(self)
        self._wait_timer.timeout.connect(self._update_waiting_message)
        self._build_ui()
        self.refresh_status()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(18)

        header = QVBoxLayout()
        
        # Title + Status Row
        title_row = QHBoxLayout()
        title = QLabel("AI Assistant")
        title.setProperty("class", "PageTitle")
        title_row.addWidget(title)
        
        title_row.addStretch()
        
        # Professional Badge-style Status
        self.lbl_status = QLabel("")
        title_row.addWidget(self.lbl_status)
        
        header.addLayout(title_row)
        
        subtitle = QLabel("Ask for engineering guidance about your dataset, model, training setup, export, or Developer Mode project.")
        subtitle.setProperty("class", "PageSubtitle")
        header.addWidget(subtitle)
        root.addLayout(header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.messages_host = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_host)
        self.messages_layout.setContentsMargins(4, 4, 4, 4)
        self.messages_layout.setSpacing(12)
        self.messages_layout.addStretch()
        self.scroll_area.setWidget(self.messages_host)
        root.addWidget(self.scroll_area, stretch=1)

        self._append_message(
            "assistant",
            "Hi. I can help inspect the current Neural Forge project state, explain training choices, "
            "debug export readiness, or reason about Developer Mode structure.",
        )

        composer = QGroupBox("Message")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setSpacing(10)

        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("Ask about the current project, model design, metrics, or export path...")
        self.input_box.setFixedHeight(92)
        self.input_box.installEventFilter(self)
        composer_layout.addWidget(self.input_box)

        actions = QHBoxLayout()
        self.lbl_hint = QLabel("Ctrl+Enter sends")
        self.lbl_hint.setStyleSheet("color: #64748B; font-size: 9pt;")
        actions.addWidget(self.lbl_hint)
        actions.addStretch()

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_chat)
        actions.addWidget(self.btn_clear)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_response)
        actions.addWidget(self.btn_stop)

        self.btn_send = QPushButton("Send")
        self.btn_send.setProperty("class", "primary")
        self.btn_send.clicked.connect(self._send_message)
        actions.addWidget(self.btn_send)

        composer_layout.addLayout(actions)
        root.addWidget(composer)

    def eventFilter(self, source, event) -> bool:
        if source is self.input_box and event.type() == QEvent.Type.KeyPress:
            is_enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            has_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if is_enter and has_ctrl:
                self._send_message()
                return True
        return super().eventFilter(source, event)

    def apply_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        self.refresh_status()
        
        # Update existing bubbles
        for i in range(self.messages_layout.count()):
            item = self.messages_layout.itemAt(i)
            if item and item.layout():
                for j in range(item.layout().count()):
                    w = item.layout().itemAt(j).widget()
                    if isinstance(w, MessageBubble):
                        w.apply_theme(is_dark)

    def refresh_status(self) -> None:
        is_dark = getattr(self, "is_dark", True)
        if is_assistant_configured():
            settings = get_assistant_settings()
            model_name = settings.model.split("/")[-1]
            self.lbl_status.setText(f"● NVIDIA API: Connected (Model: {model_name})")
            
            bg = "rgba(16, 185, 129, 0.15)" if is_dark else "rgba(16, 185, 129, 0.1)"
            text = "#10B981" if is_dark else "#059669"
            border = "rgba(16, 185, 129, 0.3)" if is_dark else "rgba(16, 185, 129, 0.2)"
            
            self.lbl_status.setStyleSheet(f"""
                QLabel {{
                    background-color: {bg};
                    color: {text};
                    border: 1px solid {border};
                    border-radius: 14px;
                    padding: 4px 12px;
                    font-size: 9.5pt;
                    font-weight: 700;
                }}
            """)
            self.btn_send.setEnabled(self._worker is None)
        else:
            self.lbl_status.setText("● NVIDIA API: Disconnected")
            
            bg = "rgba(239, 68, 68, 0.15)" if is_dark else "rgba(239, 68, 68, 0.1)"
            text = "#EF4444" if is_dark else "#DC2626"
            border = "rgba(239, 68, 68, 0.3)" if is_dark else "rgba(239, 68, 68, 0.2)"
            
            self.lbl_status.setStyleSheet(f"""
                QLabel {{
                    background-color: {bg};
                    color: {text};
                    border: 1px solid {border};
                    border-radius: 14px;
                    padding: 4px 12px;
                    font-size: 9.5pt;
                    font-weight: 700;
                }}
            """)
            self.btn_send.setEnabled(False)

    def _append_message(self, role: str, text: str) -> MessageBubble:
        row = QHBoxLayout()
        row.setSpacing(12)
        row.setContentsMargins(0, 4, 0, 4)

        is_dark = getattr(self, "is_dark", True)
        bubble = MessageBubble(role, text, is_dark=is_dark)

        if role == "user":
            row.addStretch(1) # Smaller stretch to allow bubble to grow
            row.addWidget(bubble, 9) # Give bubble more weight
        else:
            row.addWidget(bubble, 9)
            row.addStretch(1)

        self.messages_layout.insertLayout(self.messages_layout.count() - 1, row)
        self._scroll_to_bottom()
        return bubble

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(
            0,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )

    def _send_message(self) -> None:
        if self._worker is not None:
            return

        prompt = self.input_box.toPlainText().strip()
        if not prompt:
            return

        self.input_box.clear()
        self._append_message("user", prompt)
        self._messages.append({"role": "user", "content": prompt})

        self._active_response = ""
        self._reasoning_preview = ""
        self._active_assistant_label = self._append_message("assistant", "Connecting to NVIDIA...")
        self._start_wait_timer()
        self.btn_send.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_hint.setText("Assistant is responding...")

        self._worker = AssistantWorker(
            messages=self._messages,
            project_context=build_project_context(self.state),
        )
        self._worker.token_received.connect(self._on_token_received)
        self._worker.reasoning_received.connect(self._on_reasoning_received)
        self._worker.response_finished.connect(self._on_response_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.start()

    def _start_wait_timer(self) -> None:
        self._wait_seconds = 0
        self._wait_timer.start(1000)

    def _update_waiting_message(self) -> None:
        if self._active_assistant_label is None or self._active_response:
            return
        self._wait_seconds += 1
        if self._reasoning_preview:
            return
        self._active_assistant_label.setText(
            f"Waiting for NVIDIA response... {self._wait_seconds}s"
        )

    def _on_reasoning_received(self, token: str) -> None:
        if self._active_assistant_label is None or self._active_response:
            return
        self._reasoning_preview = (self._reasoning_preview + token).strip()
        preview = self._reasoning_preview[-420:]
        self._active_assistant_label.setText("Thinking...\n\n" + preview)
        self._scroll_to_bottom()

    def _on_token_received(self, token: str) -> None:
        if self._active_assistant_label is None:
            return
        self._wait_timer.stop()
        self._active_response += token
        self._active_assistant_label.setText(self._active_response)
        self._scroll_to_bottom()

    def _on_response_finished(self, response: str) -> None:
        final_text = response.strip()
        if not final_text:
            final_text = "The assistant returned no visible text."
        if self._active_assistant_label is not None:
            self._active_assistant_label.setText(final_text)
        self._messages.append({"role": "assistant", "content": final_text})

    def _on_error(self, message: str) -> None:
        self._wait_timer.stop()
        if self._active_assistant_label is not None:
            self._active_assistant_label.setText(message)
            self._active_assistant_label.setStyleSheet("""
                MessageBubble {
                    background-color: rgba(239, 68, 68, 0.15);
                    border: 1px solid rgba(239, 68, 68, 0.3);
                    border-radius: 12px;
                }
                QLabel { color: #EF4444; font-weight: 600; }
            """)

    def _cleanup_worker(self) -> None:
        self._worker = None
        self._active_assistant_label = None
        self._active_response = ""
        self._reasoning_preview = ""
        self._wait_timer.stop()
        self._wait_seconds = 0
        self.btn_clear.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_hint.setText("Ctrl+Enter sends")
        self.refresh_status()

    def _stop_response(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()
        self.btn_stop.setEnabled(False)
        self.lbl_hint.setText("Stopping after the current network chunk...")

    def _clear_chat(self) -> None:
        if self._worker is not None:
            return
        self._messages.clear()
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.layout() is not None:
                self._delete_layout(item.layout())
            elif item.widget() is not None:
                item.widget().deleteLater()
        self._append_message("assistant", "Chat cleared. What should we inspect next?")

    def _delete_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                self._delete_layout(item.layout())
        layout.deleteLater()

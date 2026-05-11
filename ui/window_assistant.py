"""
window_assistant.py
===================
AI Assistant chat tab for NVIDIA's OpenAI-compatible API.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend.assistant_client import build_project_context, is_assistant_configured
from utils.project_state import ProjectState
from workers.assistant_worker import AssistantWorker


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
        title = QLabel("AI Assistant")
        title.setProperty("class", "PageTitle")
        subtitle = QLabel("Ask for engineering guidance about your dataset, model, training setup, export, or Developer Mode project.")
        subtitle.setProperty("class", "PageSubtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        status_group = QGroupBox("Connection")
        status_layout = QHBoxLayout(status_group)
        status_layout.setSpacing(12)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_status, stretch=1)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_status)
        status_layout.addWidget(btn_refresh)

        root.addWidget(status_group)

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

    def refresh_status(self) -> None:
        if is_assistant_configured():
            self.lbl_status.setText("NVIDIA API key detected. The assistant will use the configured model from .env.")
            self.lbl_status.setStyleSheet("color: #10B981; font-weight: 700;")
            self.btn_send.setEnabled(self._worker is None)
        else:
            self.lbl_status.setText("NVIDIA_API_KEY is missing. Add it to .env, then refresh this tab.")
            self.lbl_status.setStyleSheet("color: #EF4444; font-weight: 700;")
            self.btn_send.setEnabled(False)

    def _append_message(self, role: str, text: str) -> QLabel:
        row = QHBoxLayout()
        row.setSpacing(8)

        bubble = QLabel(text)
        bubble.setTextFormat(Qt.TextFormat.PlainText)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bubble.setWordWrap(True)
        bubble.setMinimumWidth(220)
        bubble.setMaximumWidth(760)
        bubble.setContentsMargins(12, 10, 12, 10)

        if role == "user":
            bubble.setStyleSheet(
                "background-color: #0EA5E9; color: #FFFFFF; border-radius: 8px; "
                "padding: 10px 12px; font-weight: 600;"
            )
            row.addStretch()
            row.addWidget(bubble)
        else:
            bubble.setStyleSheet(
                "background-color: rgba(15, 23, 42, 0.55); border: 1px solid rgba(255, 255, 255, 0.08); "
                "border-radius: 8px; padding: 10px 12px;"
            )
            row.addWidget(bubble)
            row.addStretch()

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
            self._active_assistant_label.setStyleSheet(
                "background-color: rgba(239, 68, 68, 0.12); color: #EF4444; "
                "border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 10px 12px;"
            )

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

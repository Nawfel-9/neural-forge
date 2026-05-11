"""
assistant_worker.py
===================
Background streaming worker for the AI Assistant tab.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from backend.assistant_client import stream_chat_response


class AssistantWorker(QThread):
    """Streams chat completion tokens without blocking the Qt event loop."""

    token_received = pyqtSignal(str)
    reasoning_received = pyqtSignal(str)
    response_finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, messages: list[dict[str, str]], project_context: str):
        super().__init__()
        self.messages = list(messages)
        self.project_context = project_context
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            parts: list[str] = []
            for event in stream_chat_response(self.messages, self.project_context):
                if self._stop_requested:
                    break
                if event.kind == "reasoning":
                    self.reasoning_received.emit(event.text)
                    continue
                parts.append(event.text)
                self.token_received.emit(event.text)
            self.response_finished.emit("".join(parts))
        except Exception as exc:
            self.error.emit(str(exc))

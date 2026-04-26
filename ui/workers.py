"""Background QThread workers."""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal


class ChatWorker(QThread):
    """Runs agent.chat_stream() in a background thread, emitting each token."""

    token = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, agent: Any, message: str, parent=None) -> None:
        super().__init__(parent)
        self._agent = agent
        self._message = message

    def run(self) -> None:
        try:
            for tok in self._agent.chat_stream(self._message):
                if self.isInterruptionRequested():
                    break
                self.token.emit(tok)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class ApiWorker(QThread):
    """Runs any callable in a background thread and emits the result."""

    result = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, fn: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.result.emit(self._fn())
        except Exception as exc:
            self.error_occurred.emit(str(exc))

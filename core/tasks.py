from __future__ import annotations

from PySide6.QtCore import QObject, Signal, QRunnable, Slot
from typing import Callable, Generator, Tuple, Any


class TaskSignals(QObject):
    """
    Signal bundle for a background task that yields (progress:int, message:str).
    """
    progress = Signal(int)
    message = Signal(str)
    finished = Signal()
    error = Signal(str)


class Task(QRunnable):
    """
    Run a generator-producing factory on a worker thread.

    The factory must return a generator that yields (progress:int, message:str).
    Any exception is caught and emitted via `error` signal.
    """
    def __init__(self, work_gen_factory: Callable[[], Generator[Tuple[int, str], None, None]]):
        super().__init__()
        self.signals = TaskSignals()
        self._factory = work_gen_factory

    @Slot()
    def run(self) -> None:
        try:
            gen = self._factory()
            for p, msg in gen:
                self.signals.progress.emit(int(p))
                if msg:
                    self.signals.message.emit(str(msg))
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))

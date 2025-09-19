from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class FuncWorker(QThread):
    """
    Run a plain Python function in a Qt thread and emit results.

    Signals
    -------
    done(object): emitted with the function's return value on success
    failed(str): emitted with the error message on exception
    """
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, func, *args, **kw):
        super().__init__()
        self.func, self.args, self.kw = func, args, kw
        self.result = None

    def run(self) -> None:  # noqa: D401 (Qt override)
        try:
            self.result = self.func(*self.args, **self.kw)
        except Exception as e:  # keep it simple—propagate the message
            self.failed.emit(str(e))
        else:
            self.done.emit(self.result)

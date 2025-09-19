from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "gcamp", level: int = logging.INFO, propagate: bool = False) -> logging.Logger:
    """
    Create/retrieve a logger with a single stream handler.

    * Prevents duplicate handlers if called multiple times.
    * Sets a concise, informative formatter.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Avoid duplicate handlers when re-importing in dev/GUI sessions
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)

    return logger
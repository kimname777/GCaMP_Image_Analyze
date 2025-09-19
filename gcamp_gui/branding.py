from __future__ import annotations

from pathlib import Path
import sys


def resource_path(rel: str) -> Path:
    """
    Resolve resources both in dev and in a frozen build (one-dir).

    Dev: project root (two levels up from this file)
    Frozen: directory containing the executable
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.argv[0]).resolve().parent
    else:
        base = Path(__file__).resolve().parent.parent
    return (base / rel)

from __future__ import annotations
"""
report.py
---------
Lightweight text report exporter.

Writes a simple Markdown-ish text file summarizing the number of cells/frames
and any available tuning metrics. Extend as needed (e.g., add SNR, reliability).
"""

from pathlib import Path
from core.state import AppState


def export_report(outdir: Path, state: AppState):
    """
    Write a minimal analysis report next to exported figures/CSVs.

    Parameters
    ----------
    outdir : Path
        Destination directory (created by the caller).
    state : AppState
        Application state containing traces and tuning results.
    """
    lines = ["# GCaMP Analyzer Report\n\n"]
    if state.dff is not None:
        lines.append(f"Cells: {state.dff.shape[0]}, Frames: {state.dff.shape[1]}\n")
    if getattr(state, "tuning_results", None):
        for k, v in state.tuning_results.items():
            lines.append(f"{k}: {v}\n")
    (outdir / "report.txt").write_text("".join(lines), encoding="utf-8")

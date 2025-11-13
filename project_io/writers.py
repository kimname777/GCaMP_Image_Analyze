from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import csv

from core.state import AppState


def _save_matrix(path: Path, mat: np.ndarray, indices: Optional[Sequence[int]] = None) -> None:
    """Write a 2D array as CSV with ROI index in the first column.

    Parameters
    ----------
    path:
        Output CSV path.
    mat:
        2D array of shape (N, T) where N is the number of ROIs.
    indices:
        Optional sequence of ROI indices to export. If None, all rows are
        exported. Indices outside [0, N) are ignored.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D array (N, T), got shape {mat.shape}")

    n_rois, T = mat.shape

    # Determine which ROI indices to write
    if indices is None:
        roi_ids = np.arange(n_rois, dtype=int)
    else:
        roi_ids = np.array([int(i) for i in indices], dtype=int)
        # Keep only valid, unique indices in ascending order
        roi_ids = np.unique(roi_ids[(roi_ids >= 0) & (roi_ids < n_rois)])

    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        # Header: roi, t0, t1, ..., t{T-1}
        wr.writerow(["roi"] + [f"t{i}" for i in range(T)])
        for ridx in roi_ids:
            row = mat[int(ridx)]
            wr.writerow([int(ridx)] + [float(x) for x in row])


def save_csv_pack(outdir: Path | str, state: AppState, selected_indices: Optional[Sequence[int]] = None) -> None:
    """Export common arrays (ΔF/F, spikes) as CSV files for quick inspection.

    If ``selected_indices`` is provided, only those ROI indices are exported
    (rows with matching ROI index). Otherwise, all ROIs are written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if state.dff is not None:
        _save_matrix(outdir / "dff.csv", state.dff, selected_indices)
    if state.spikes is not None:
        _save_matrix(outdir / "spikes.csv", state.spikes, selected_indices)


def save_nwb(path: Path | str, state: AppState, fps: float = 30.0) -> None:
    """Save traces into an NWB file (if ``pynwb`` is installed).

    If NWB writing fails (e.g. missing dependency), this falls back to
    saving a compressed NPZ file with the same base name.
    """
    path = Path(path)
    try:
        from pynwb import NWBFile, NWBHDF5IO, TimeSeries  # type: ignore
        from datetime import datetime

        nwb = NWBFile(
            session_description="GCaMP session",
            identifier="uid-001",
            session_start_time=datetime.now(),
        )

        if state.dff is not None:
            ts = TimeSeries(name="dff", data=state.dff, rate=float(fps), unit="a.u.")
            nwb.add_acquisition(ts)
        if state.spikes is not None:
            ts2 = TimeSeries(name="spikes", data=state.spikes, rate=float(fps), unit="a.u.")
            nwb.add_acquisition(ts2)

        with NWBHDF5IO(str(path), "w") as io:
            io.write(nwb)
    except Exception:
        # Fallback: NPZ dump next to the intended NWB path
        data = {
            "dff": state.dff if state.dff is not None else None,
            "spikes": state.spikes if state.spikes is not None else None,
            "fps": float(fps),
        }
        np.savez_compressed(str(path).replace(".nwb", ".npz"), **data)

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import csv

from core.state import AppState


def _save_matrix(path: Path, mat: np.ndarray) -> None:
    """
    Write a 2D array as CSV with a simple header t0..t{T-1}.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([f"t{i}" for i in range(mat.shape[1])])
        for row in mat:
            wr.writerow([float(x) for x in row])


def save_csv_pack(outdir: Path | str, state: AppState) -> None:
    """
    Export common arrays (ΔF/F, spikes) as CSV files for quick inspection.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if state.dff is not None:
        _save_matrix(outdir / "dff.csv", state.dff)
    if state.spikes is not None:
        _save_matrix(outdir / "spikes.csv", state.spikes)


def save_nwb(path: Path | str, state: AppState, fps: float = 30.0) -> None:
    """
    Save traces into an NWB file (if `pynwb` is installed). Otherwise, fallback to NPZ.

    NWB contents
    ------------
    acquisitions:
      - TimeSeries 'dff'    (unit=a.u., rate=fps)
      - TimeSeries 'spikes' (unit=a.u., rate=fps)
    """
    path = Path(path)
    try:
        from pynwb import NWBFile, NWBHDF5IO, TimeSeries  # type: ignore
        from datetime import datetime

        nwb = NWBFile(session_description="GCaMP session", identifier="uid-001", session_start_time=datetime.now())

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

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
    # Note: plotting is handled by save_plots / save_results_bundle


def save_plots(outdir: Path | str, state: AppState, selected_indices: Optional[Sequence[int]] = None) -> None:
    """Save diagnostic plots (traces, correlation matrix, raster) as PNGs.

    This is a best-effort helper that uses matplotlib when available.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    # Determine which ROIs to plot
    if selected_indices is None:
        idxs = list(range(state.dff.shape[0])) if getattr(state, "dff", None) is not None else []
    else:
        idxs = [int(i) for i in selected_indices if getattr(state, "dff", None) is not None and 0 <= int(i) < state.dff.shape[0]]

    # Traces (stacked)
    try:
        if getattr(state, "dff", None) is not None and len(idxs) > 0:
            dff = state.dff
            t = np.arange(dff.shape[1])
            step = 1.2 * (np.nanstd(dff[idxs, :]) + 1e-6)
            fig, ax = plt.subplots(figsize=(8, max(3, 0.2 * len(idxs))))
            offset = 0.0
            for i in idxs:
                ax.plot(t, dff[i] + offset, linewidth=1.0)
                offset += step
            ax.set_xlabel("Frame")
            ax.set_ylabel("ΔF/F (offset by cell)")
            ax.set_title("Stacked ΔF/F traces")
            fig.tight_layout()
            fig.savefig(outdir / "traces.png", dpi=150)
            plt.close(fig)
    except Exception:
        pass

    # Correlation matrix
    try:
        if getattr(state, "dff", None) is not None and state.dff.shape[0] >= 2:
            C = np.corrcoef(state.dff)
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(C, vmin=-1, vmax=1, origin="lower", cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_title("Pairwise correlation matrix")
            fig.tight_layout()
            fig.savefig(outdir / "corr_matrix.png", dpi=150)
            plt.close(fig)
    except Exception:
        pass

    # Raster (spikes or thresholded dff)
    try:
        S = getattr(state, "spikes", None)
        if S is None and getattr(state, "dff", None) is not None:
            X = state.dff
            thr = np.quantile(X, 0.95, axis=1, keepdims=True)
            S = (X > thr).astype(np.float32)
        if S is not None and S.size:
            n, T = S.shape
            fig, ax = plt.subplots(figsize=(8, max(3, 0.2 * n)))
            for k in range(n):
                t_on = np.flatnonzero(S[k] > 0)
                if t_on.size:
                    ax.vlines(t_on, k - 0.45, k + 0.45, linewidth=1)
            ax.set_xlabel("time (frame)")
            ax.set_ylabel("neuron #")
            ax.set_title("Spike raster (estimated)")
            fig.tight_layout()
            fig.savefig(outdir / "raster.png", dpi=150)
            plt.close(fig)
    except Exception:
        pass

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
        # If NWB writing fails, do not write any NPZ/NPY artifacts (per user request).
        # Silently skip NWB export.
        pass

def save_results_bundle(outdir: Path | str, state: AppState, selected_indices: Optional[Sequence[int]] = None) -> None:
    """Save analysis outputs into an organized folder structure.

        Structure created under ``outdir``:
            - csv/       (dff.csv, spikes.csv)
            - images/    (overlay.tif, traces.png, corr_matrix.png, raster.png)
            - report/    (report.txt or richer export)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_dir = outdir / "csv"
    img_dir = outdir / "images"
    rpt_dir = outdir / "report"
    for d in (csv_dir, img_dir, rpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    # CSVs
    try:
        save_csv_pack(csv_dir, state, selected_indices)
    except Exception:
        pass

    # NWB/NPZ
    try:
        save_nwb(outdir / "session.nwb", state)
    except Exception:
        pass

    # Note: per-request we do not write .npy files here; only CSVs and images

    # Images: overlay + plots
    try:
        from viz.overlay import save_overlay_tiff
        stack = getattr(state, "reg_stack", None) or getattr(state, "raw_stack", None)
        masks = getattr(state, "roi_masks", None)
        if stack is not None and masks is not None:
            try:
                save_overlay_tiff(stack, masks, str(img_dir / "overlay.tif"))
            except Exception:
                pass
    except Exception:
        pass

    try:
        save_plots(img_dir, state, selected_indices)
    except Exception:
        pass

    # Report
    try:
        from viz.report import export_report
        export_report(rpt_dir, state)
    except Exception:
        try:
            lines = [
                f"ROIs: {getattr(state, 'roi_masks', None).shape[0] if getattr(state, 'roi_masks', None) is not None else 0}\n",
                f"Frames: {getattr(state, 'raw_stack', None).shape[0] if getattr(state, 'raw_stack', None) is not None else 0}\n",
            ]
            (rpt_dir / "report.txt").write_text("".join(lines), encoding="utf-8")
        except Exception:
            pass

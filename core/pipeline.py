from __future__ import annotations

from typing import Optional, Union, Generator, Tuple
from pathlib import Path
import importlib
import importlib.util
import os
import numpy as np

from core.state import AppState
from core.config import AppConfig

# I/O & preprocessing
from project_io.loaders import load_stack
from project_io.stim import load_stim_csv
from project_io.writers import save_csv_pack, save_nwb
from preprocess.motion import motion_correct
from signals.traces import extract_traces
from signals.baselines import compute_baseline_moving_percentile  # reserved

# Analysis
from analysis.kinetics import compute_kinetics
from analysis.tuning import compute_orientation_tuning, pseudo_tuning
from analysis.reliability import split_half_reliability

# Reporting
from viz.report import export_report


# -------------------- Deconvolution backend selection -------------------- #
def _choose_deconv_backend(prefer: str = "mlspike") -> str:
    """
    Decide which deconvolution backend to use.

    Logic
    -----
    - "oasis": require env OASIS_ENABLE=1 and importable 'oasis' package.
    - "mlspike": if importable 'signals.deconv_mlspike', else fallback to "naive".
    - otherwise: "naive".
    """
    prefer = (prefer or "").lower()
    if prefer == "oasis":
        if os.environ.get("OASIS_ENABLE", "0").lower() in ("1", "true"):
            if importlib.util.find_spec("oasis") is not None:
                return "oasis"
        # Fallback to mlspike if OASIS can't be used
        prefer = "mlspike"

    if prefer == "mlspike":
        return "mlspike" if importlib.util.find_spec("signals.deconv_mlspike") else "naive"

    return "naive"


def _run_deconv(dff: np.ndarray, fps: float, backend: str) -> np.ndarray:
    """
    Execute deconvolution using a chosen backend.

    Returns
    -------
    spikes : (N, T) array
    """
    if backend == "oasis":
        mod = importlib.import_module("signals.deconv_oasis")  # must not import 'oasis' at module import time
        return mod.deconvolve_series(dff, fps=fps)  # type: ignore[attr-defined]

    if backend == "mlspike":
        mod = importlib.import_module("signals.deconv_mlspike")
        return mod.deconvolve_series(dff, fps=fps)  # type: ignore[attr-defined]

    # Last resort fallback: simple positive discrete derivative
    d = np.diff(np.concatenate([dff[:, :1], dff], axis=1), axis=1)
    return np.maximum(d, 0.0)


# ------------------------------- Pipeline -------------------------------- #
class Pipeline:
    """
    End-to-end calcium imaging pipeline:
      load → motion correction → segmentation → traces/ΔF/F → deconvolution
      → tuning/QC/kinetics → export
    """

    def __init__(self, state: AppState, config: AppConfig):
        self.state = state
        self.cfg = config
        self.stim_events: Optional[list[dict]] = None

    # --------------------------- I/O --------------------------- #
    def load_data(self, path: Union[str, Path]) -> None:
        """Load a movie stack and metadata; update FPS if provided by loader."""
        stack, meta = load_stack(path)
        self.state.raw_stack = stack  # (T, Y, X)
        self.cfg.fps = float(meta.get("fps", self.cfg.fps))

    def load_stim(self, path: Union[str, Path]) -> int:
        """Load stimulus CSV to enable event-based tuning analysis."""
        self.stim_events = load_stim_csv(path)
        return len(self.stim_events)

    # ----------------------- Main runner ----------------------- #
    def run_all(self) -> Generator[Tuple[int, str], None, None]:
        """Yield (progress, message) pairs as the pipeline advances."""
        def push(p: int, msg: str = ""):
            yield p, msg

        if self.state.raw_stack is None:
            raise RuntimeError("No movie loaded. Call `load_data()` first.")

        # 1) Motion correction
        yield from push(5, "Motion correction (rigid MVP)")
        self.state.reg_stack, mc_meta = motion_correct(
            self.state.raw_stack,
            self.cfg.preprocess.get("motion", {})
        )

        # 2) Segmentation (always via segmentation.api for consistent surface)
        yield from push(20, "Segmentation (Suite2p / threshold / fallback)")
        seg_backend = self.cfg.segmentation.get("backend", "suite2p")
        diam = self.cfg.segmentation.get("diameter", 14)
        tau = self.cfg.segmentation.get("tau", 1.0)

        seg_api = importlib.import_module("segmentation.api")
        masks = seg_api.segment_cells(self.state.reg_stack, backend=seg_backend, diameter=diam, tau=tau)

        # Fallback #1: threshold mode in Suite2p wrapper
        if masks is None or masks.size == 0 or masks.shape[0] == 0:
            yield from push(28, "No ROIs; fallback to threshold segmentation")
            s2p_wrap = importlib.import_module("segmentation.suite2p_wrapper")
            masks = s2p_wrap.segment_cells_threshold(self.state.reg_stack, diameter=diam)

        # Fallback #2: variance-based blobs (very conservative)
        if masks is None or masks.size == 0 or masks.shape[0] == 0:
            yield from push(32, "Threshold failed; variance-based fallback")
            mov = self.state.reg_stack.astype(np.float32)
            var_img = mov.var(axis=0)
            thr = np.percentile(var_img, 99.5)  # top 0.5%
            bw = var_img > thr

            from skimage.measure import label, regionprops  # lazy import
            lbl = label(bw)
            ms = []
            min_area = max(20, (diam ** 2) // 4)
            for r in regionprops(lbl):
                if r.area < min_area:
                    continue
                m = (lbl == r.label)
                ms.append(m)
            masks = np.stack(ms, axis=0) if ms else np.zeros((0, bw.shape[0], bw.shape[1]), bool)

        self.state.roi_masks = masks.astype(bool, copy=False)

        # 3) Neuropil / Traces / ΔF/F
        yield from push(40, "Neuropil + trace extraction + ΔF/F")
        F_raw, F_np, F0, dff, np_masks = extract_traces(
            self.state.reg_stack, self.state.roi_masks,
            r=self.cfg.preprocess.get("neuropil", {}).get("r", 0.7),
            inner=1, outer=4, baseline_fn=None
        )
        self.state.F_raw = F_raw
        self.state.F_np = F_np
        self.state.F0 = F0
        self.state.dff = dff.astype("float32", copy=False)
        # keep both for compatibility
        self.state.np_masks = np_masks.astype(bool, copy=False)
        self.state.neuropil_masks = self.state.np_masks

        if self.state.dff is None:
            raise RuntimeError("dFF is None; traces must be computed before deconvolution.")

        # 4) Deconvolution (robust backend selection + execution)
        deconv_backend = _choose_deconv_backend(self.cfg.deconvolution.get("backend", "mlspike"))
        try:
            spikes = _run_deconv(self.state.dff, fps=self.cfg.fps, backend=deconv_backend)
        except Exception:
            # Last-resort fallback: positive derivative
            d = np.diff(np.concatenate([self.state.dff[:, :1], self.state.dff], axis=1), axis=1)
            spikes = np.maximum(d, 0.0)
        self.state.spikes = spikes
        yield from push(65, f"Deconvolution ({deconv_backend}) done")

        # 5) Tuning (event-based if stim is available; otherwise pseudo)
        try:
            if self.stim_events:
                self.state.tuning_summary = compute_orientation_tuning(
                    self.state.dff, self.stim_events, fps=self.cfg.fps, window_s=2.0
                )
            else:
                self.state.tuning_summary = pseudo_tuning(self.state.dff, fps=self.cfg.fps)
        except Exception:
            self.state.tuning_summary = pseudo_tuning(self.state.dff, fps=self.cfg.fps)

        # 6) QC (split-half reliability)
        try:
            self.state.qc_metrics = {"reliability": split_half_reliability(self.state.dff)}
        except Exception:
            self.state.qc_metrics = {"reliability": float("nan")}

        # 7) Kinetics summary (peak / time-to-peak / half-decay)
        try:
            self.state.kinetics_summary = compute_kinetics(self.state.dff, fps=self.cfg.fps)
        except Exception:
            self.state.kinetics_summary = {"peak_mean": np.nan, "t_peak_mean_s": np.nan, "t_half_decay_mean_s": np.nan}

        # 8) Summary tick
        n_cells = 0 if self.state.roi_masks is None else int(self.state.roi_masks.shape[0])
        T = 0 if self.state.dff is None else int(self.state.dff.shape[1])
        yield from push(99, f"SUMMARY: stack={tuple(self.state.reg_stack.shape)}  ROIs={n_cells}  T={T}")
        yield from push(100, "Done")

    # --------------------------- Export --------------------------- #
    def export_all(self, outdir: Union[str, Path]) -> None:
        """Write CSV pack, PDF/HTML report, and NWB file to `outdir`."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_csv_pack(outdir, self.state)
        export_report(outdir, self.state)
        save_nwb(outdir / "session.nwb", self.state, fps=self.cfg.fps)

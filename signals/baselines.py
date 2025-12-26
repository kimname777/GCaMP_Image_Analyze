from __future__ import annotations

import numpy as np
from typing import Tuple


def moving_percentile(arr: np.ndarray, p: float = 10.0, w: int = 300, frame_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Moving percentile along the time axis.

    Parameters
    ----------
    arr : (N, T) or (T,)
        Signal(s). If 1D, it is treated as a single row.
    p : float
        Percentile in [0, 100].
    w : int
        Window length in frames (centered window).

    Returns
    -------
    out : same shape as `arr`
        Per-time moving percentile.
    """
    x = np.asarray(arr)
    if x.ndim == 1:
        x = x[None, :]
    N, T = x.shape
    w = int(max(3, w))
    half = w // 2
    out = np.empty_like(x, dtype=np.float32)
    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        if frame_mask is None:
            sel = slice(s, e)
            out[:, t] = np.percentile(x[:, sel], p, axis=1)
        else:
            mask = frame_mask[s:e]
            if np.any(mask):
                # select only frames that are not masked out
                vals = x[:, s:e][:, mask]
                # If after masking a ROI has no values (shouldn't happen), fallback to unmasked window
                if vals.shape[1] == 0:
                    out[:, t] = np.percentile(x[:, s:e], p, axis=1)
                else:
                    out[:, t] = np.percentile(vals, p, axis=1)
            else:
                # no valid frames in window: fallback to unmasked behavior
                out[:, t] = np.percentile(x[:, s:e], p, axis=1)
    return out if arr.ndim == 2 else out[0]


def compute_baseline_moving_percentile(
    F_corr: np.ndarray,
    p: float = 10.0,
    win_s: float = 90.0,
    fps: float = 30.0,
    dropout_frac: float = 0.2,
    roi_drop_frac: float = 0.2,
    invalid_mode: str = "hold",
    roi_thresh_mode: str = "relative",
    roi_percentile: float = 20.0,
) -> np.ndarray:
        # info_container parameter removed; do not expose internal diagnostic containers here.
    """
    Convenience wrapper: moving-percentile baseline with a time window.

    Parameters
    ----------
    F_corr : (N, T)
        Corrected fluorescence (e.g., F_raw - r * F_np).
    p : float
        Percentile (default 10th).
    win_s : float
        Window size in seconds.
    fps : float
        Sampling rate (frames per second).

    Returns
    -------
    F0 : (N, T)
        Time-varying baseline to use in ΔF/F.
    """
    w = max(3, int(round(float(win_s) * float(fps))))

    # Detect dropout frames using two complementary heuristics:
    # 1) Global median-based dropout (many ROIs low relative to global level)
    # 2) ROI-relative dropout: many ROIs are far below their own medians
    try:
        N, T = F_corr.shape
        # per-frame median (global view)
        frame_median = np.nanmedian(F_corr, axis=0)

        # robust global median across frames
        finite_frame_med = frame_median[np.isfinite(frame_median)]
        if finite_frame_med.size == 0:
            raise ValueError("no finite frame medians")
        global_med = float(np.nanmedian(finite_frame_med))
        global_thresh = dropout_frac * max(global_med, 1e-9)
        is_dropout_global = frame_median <= global_thresh

        # per-ROI threshold: either relative to ROI median or based on ROI percentile
        if str(roi_thresh_mode).lower() == "percentile":
            # threshold is the ROI-specific percentile of its own distribution
            roi_thresh = np.nanpercentile(F_corr, float(roi_percentile), axis=1)
            # scale by roi_drop_frac if requested (allows further tightening)
            if float(roi_drop_frac) != 1.0:
                roi_thresh = roi_thresh * float(roi_drop_frac)
        else:
            # relative mode: roi_drop_frac * median(ROI)
            roi_medians = np.nanmedian(F_corr, axis=1)
            # avoid zeros
            roi_medians_safe = np.maximum(roi_medians, 1e-9)
            roi_thresh = roi_medians_safe * float(roi_drop_frac)

        # compute per-ROI low mask (True where ROI value is below its threshold)
        low_mask = (F_corr < roi_thresh[:, None])  # (N, T) boolean
        frac_low = np.nanmean(low_mask.astype(float), axis=0)
        is_dropout_roi_rel = frac_low >= float(dropout_frac)

        # combine heuristics: mark frame as valid only if neither heuristic flags it
        is_dropout = np.logical_or(is_dropout_global, is_dropout_roi_rel)
        frame_mask = ~is_dropout
        # Build per-ROI invalid mask: ROI is invalid on frame if it is below its ROI
        # threshold OR the frame is globally flagged as dropout.
        roi_invalid_mask = low_mask.copy()
        if is_dropout_global is not None:
            roi_invalid_mask[:, is_dropout_global] = True
    except Exception:
        frame_mask = None
        roi_invalid_mask = None

    F0 = moving_percentile(F_corr, p=p, w=w, frame_mask=frame_mask).astype(np.float32)

    # If frame_mask is provided, we may want to avoid updating the baseline
    # on frames marked invalid (dropout). Two modes are supported:
    #  - "hold": F0[:, t] = F0[:, t-1] (carry-forward last valid baseline)
    #  - "interp": linear interpolation between neighboring valid baselines
    # If frame_mask is None (no invalid frames detected), return F0 unchanged.
    try:
        if frame_mask is None:
            return F0
    except NameError:
        # frame_mask variable may not be defined if earlier try/except failed
        return F0

    mode = (invalid_mode or "hold").lower()
    valid = np.asarray(frame_mask, dtype=bool)
    T = F0.shape[1]
    if valid.all():
        return F0

    # indices of valid frames
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        # no valid frames: return as-is
        return F0

    if mode == "interp":
        # linear interpolation per ROI over valid frames
        xi = valid_idx
        xo = np.arange(T)
        out = np.empty_like(F0)
        for i in range(F0.shape[0]):
            yi = F0[i, xi]
            # fill endpoints by nearest valid (np.interp does this)
            out[i] = np.interp(xo, xi, yi)
        return out

    # default: hold previous valid baseline
    out = F0.copy()
    # for frames before first valid, fill with first valid
    first = valid_idx[0]
    if first > 0:
        out[:, :first] = out[:, first:first+1]
    # forward-fill for invalid frames
    last_valid = first
    for t in range(first + 1, T):
        if valid[t]:
            last_valid = t
        else:
            out[:, t] = out[:, last_valid]

    return out

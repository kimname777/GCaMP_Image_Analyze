from __future__ import annotations

import numpy as np
from typing import Tuple


def moving_percentile(arr: np.ndarray, p: float = 10.0, w: int = 300) -> np.ndarray:
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
        out[:, t] = np.percentile(x[:, s:e], p, axis=1)
    return out if arr.ndim == 2 else out[0]


def compute_baseline_moving_percentile(
    F_corr: np.ndarray,
    p: float = 10.0,
    win_s: float = 90.0,
    fps: float = 30.0,
) -> np.ndarray:
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
    return moving_percentile(F_corr, p=p, w=w).astype(np.float32)

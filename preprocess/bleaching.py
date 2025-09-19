from __future__ import annotations

import numpy as np
from typing import Literal, Tuple

# Optional deps; we fail gracefully if missing.
try:
    from scipy.optimize import curve_fit  # type: ignore
except Exception:  # pragma: no cover
    curve_fit = None

try:
    import statsmodels.api as sm  # type: ignore
except Exception:  # pragma: no cover
    sm = None

Method = Literal["exponential", "polynomial", "loess", "percentile"]


def _exp_func(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Single-exponential decay model: a * exp(-b * t) + c."""
    return a * np.exp(-b * t) + c


def remove_bleach(
    trace: np.ndarray,
    fps: float = 30.0,
    method: Method = "exponential",
    poly_deg: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detrend a single 1D trace for photobleaching; return (detrended, trend).

    Parameters
    ----------
    trace : (T,) array-like
        Intensity or ΔF/F trace.
    fps : float
        Sampling rate (frames per second).
    method : {'exponential','polynomial','loess','percentile'}
        Trend estimator to remove from the signal.
    poly_deg : int
        Degree for polynomial trend.

    Notes
    -----
    * Output is re-centered by adding median(trend) so dynamic range is preserved.
    * If the chosen method is unavailable (missing dependency), it falls back in
      the order: exponential → polynomial → loess → percentile (rolling P10).
    """
    x = np.asarray(trace).astype(np.float32).ravel()
    T = x.size
    if T == 0:
        return x, x
    t = np.arange(T, dtype=np.float32) / float(max(1e-6, fps))

    # 1) Exponential decay fit (requires SciPy)
    if method == "exponential" and curve_fit is not None:
        try:
            p0 = [float(x.max() - x.min()), 0.01, float(np.median(x))]
            popt, _ = curve_fit(_exp_func, t, x, p0=p0, maxfev=8000)
            trend = _exp_func(t, *popt).astype(np.float32)
            return x - trend + np.median(trend), trend
        except Exception:
            pass  # fall through

    # 2) Polynomial regression
    if method in ("polynomial", "exponential"):  # allow fallthrough from exponential
        try:
            deg = int(max(1, poly_deg))
            X = np.vstack([t ** k for k in range(deg + 1)]).T
            coef, *_ = np.linalg.lstsq(X, x, rcond=None)
            trend = (X @ coef).astype(np.float32)
            return x - trend + np.median(trend), trend
        except Exception:
            pass  # fall through

    # 3) LOESS (requires statsmodels)
    if method in ("loess", "exponential", "polynomial") and sm is not None:
        try:
            lowess = sm.nonparametric.lowess(x, t, frac=0.05, return_sorted=False)
            trend = np.asarray(lowess, dtype=np.float32)
            return x - trend + np.median(trend), trend
        except Exception:
            pass  # fall through

    # 4) Percentile baseline (dependency-free fallback)
    k = max(3, int(round(90 * fps)))  # ~90 s window
    half = k // 2
    base = np.empty_like(x, dtype=np.float32)
    for i in range(T):
        s = max(0, i - half)
        e = min(T, i + half + 1)
        base[i] = float(np.percentile(x[s:e], 10))
    return x - base + np.median(base), base


def remove_bleach_stack(
    stack: np.ndarray,
    fps: float = 30.0,
    method: Method = "percentile",
    poly_deg: int = 2,
    axis_time: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply photobleaching correction per-pixel across a time stack.

    Parameters
    ----------
    stack : (T, Y, X) or any array with a time axis
    fps : float
        Sampling rate.
    method : Method
        Trend method (see `remove_bleach`).
    poly_deg : int
        Degree for polynomial trend.
    axis_time : int
        Index of time axis.

    Returns
    -------
    detrended : same shape as input
    trend : same shape as input
    """
    A = np.asarray(stack, dtype=np.float32)
    A = np.moveaxis(A, axis_time, 0)  # (T, ...)
    T = A.shape[0]
    flat = A.reshape(T, -1)
    out = np.empty_like(flat)
    trn = np.empty_like(flat)
    for k in range(flat.shape[1]):
        det, trend = remove_bleach(flat[:, k], fps=fps, method=method, poly_deg=poly_deg)
        out[:, k] = det
        trn[:, k] = trend
    out = out.reshape(A.shape)
    trn = trn.reshape(A.shape)
    return np.moveaxis(out, 0, axis_time), np.moveaxis(trn, 0, axis_time)

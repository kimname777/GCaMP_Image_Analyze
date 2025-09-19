from __future__ import annotations

"""
OASIS-based deconvolution with a robust built-in fallback.

- If `oasis` is not installed or disabled, fall back to a fast non-negative
  AR(1)-difference (discrete derivative through a leaky integrator).
- Matches Pipeline's expectation: `deconvolve_series(dff, fps, ...) -> (N, T)`.
"""

from typing import Optional, Union
import importlib.util
import os
import numpy as np


def _oasis_available() -> bool:
    """Check if the `oasis` package can be imported."""
    return importlib.util.find_spec("oasis") is not None


def _relu_ar1_diff(dff: np.ndarray, gamma: float = 0.95) -> np.ndarray:
    """
    Simple non-negative AR(1) 'inverse' filter:
        s[t] = ReLU( y[t] - gamma * y[t-1] ), broadcast across rows.
    """
    y = np.asarray(dff, dtype=np.float32)
    ypad = np.concatenate([y[:, :1], y], axis=1)
    s = ypad[:, 1:] - gamma * ypad[:, :-1]
    return np.maximum(s, 0.0).astype(np.float32, copy=False)


def deconvolve_series(
    dff: np.ndarray,
    fps: float,
    lam: float = 0.0,
    g: Union[float, None] = None,
) -> np.ndarray:
    """
    Deconvolve ΔF/F using OASIS (FOOPSI). Falls back to AR(1)-diff if unavailable.

    Parameters
    ----------
    dff : (N, T)
        Input traces.
    fps : float
        Sampling rate; only used for potential parameterization.
    lam : float
        L1 penalty (passed to OASIS if used).
    g : float | None
        AR(1) coefficient; None → let OASIS estimate it. Ignored by fallback.

    Returns
    -------
    spikes : (N, T) float32
        Non-negative 'spike' estimates.
    """
    Y = np.asarray(dff, dtype=np.float32)
    if Y.ndim != 2 or Y.size == 0:
        return np.zeros_like(Y, dtype=np.float32)

    # Disable switch via env var or missing package → fallback path
    if os.environ.get("GCAMP_DISABLE_OASIS", "0") in ("1", "true", "True") or not _oasis_available():
        # Reasonable gamma from a ~0.3 s decay as a crude default
        gamma = float(np.exp(-1.0 / max(1e-6, 0.3 * float(fps))))
        return _relu_ar1_diff(Y, gamma=gamma)

    # OASIS path
    try:
        from oasis.functions import constrained_foopsi  # type: ignore
    except Exception:
        # Package present but import failed → fallback
        gamma = float(np.exp(-1.0 / max(1e-6, 0.3 * float(fps))))
        return _relu_ar1_diff(Y, gamma=gamma)

    N, T = Y.shape
    S = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        yi = Y[i].astype(np.float64, copy=False)
        try:
            c, s, b, ghat, lamhat, z = constrained_foopsi(yi, g=g, sn=None, l=float(lam))
            si = np.asarray(s, dtype=np.float32)
            S[i, : si.shape[0]] = si
        except Exception:
            # If one trace fails, keep zeros for that cell
            pass
    return S

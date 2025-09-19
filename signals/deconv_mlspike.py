from __future__ import annotations

"""
MLSpike-style deconvolution with multi-tier fallbacks.

Priority
--------
1) Python port of MLSpike (if available) – very rare.
2) MATLAB MLSpike via MATLAB Engine for Python (if available).
3) Lightweight AR(1)-difference fallback (fast, dependency-free).

This module exposes `deconvolve_series(dff, fps, params=None)`, which is what
`core.pipeline._run_deconv()` expects when `backend == "mlspike"`.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


# --- Optional backends (checked lazily at runtime) ---
try:  # Python port (hypothetical)
    import mlspike as _py_mlspike  # type: ignore
    _HAVE_PY_MLSPIKE = True
except Exception:  # pragma: no cover
    _HAVE_PY_MLSPIKE = False
    _py_mlspike = None  # type: ignore

try:  # MATLAB Engine
    import matlab.engine  # type: ignore
    _HAVE_MATLAB = True
except Exception:  # pragma: no cover
    _HAVE_MATLAB = False


def _relu_ar1_diff(dff: np.ndarray, gamma: float) -> np.ndarray:
    """Non-negative AR(1) 'inverse' filter as a very fast fallback."""
    y = np.asarray(dff, dtype=np.float32)
    ypad = np.concatenate([y[:, :1], y], axis=1)
    s = ypad[:, 1:] - float(gamma) * ypad[:, :-1]
    return np.maximum(s, 0.0).astype(np.float32, copy=False)


@dataclass
class MLSpikeParams:
    """Minimal set of MLSpike-style parameters."""
    tau: float = 0.3         # calcium decay (s)
    a: float = 1.0           # amplitude scaling
    sigma: float = 0.02      # noise SD
    drift: float = 0.0       # slow drift
    saturation: Optional[float] = None  # nonlinearity (if supported)


def deconvolve_series(dff: np.ndarray, fps: float = 30.0, params: Optional[MLSpikeParams] = None) -> np.ndarray:
    """
    Deconvolve ΔF/F using MLSpike if available, else fall back to AR(1) difference.

    Parameters
    ----------
    dff : (N, T)
        Input traces.
    fps : float
        Sampling rate (Hz).
    params : MLSpikeParams | None
        Optional parameter bundle.

    Returns
    -------
    spikes : (N, T) float32
        Non-negative 'spike' estimates.
    """
    Y = np.asarray(dff, dtype=np.float32)
    if Y.ndim != 2 or Y.size == 0:
        return np.zeros_like(Y, dtype=np.float32)
    if params is None:
        params = MLSpikeParams()

    # 1) Python port path (rare in the wild; kept as best-effort)
    if _HAVE_PY_MLSPIKE:
        try:
            dt = 1.0 / float(max(1e-6, fps))
            out = np.zeros_like(Y, dtype=np.float32)
            for i in range(Y.shape[0]):
                yi = Y[i].astype(float, copy=False)
                # API below is hypothetical; adapt to your python-port if different
                par = _py_mlspike.Params(dt=dt, tau=params.tau, a=params.a,
                                         sigma=params.sigma, drift=params.drift,
                                         saturation=params.saturation)
                s = _py_mlspike.infer_spikes(yi, par)  # expected (T,)
                out[i, :len(s)] = np.asarray(s, dtype=np.float32)
            return out
        except Exception:
            pass  # fall through

    # 2) MATLAB Engine path
    if _HAVE_MATLAB:
        try:
            eng = matlab.engine.start_matlab()
            dt = 1.0 / float(max(1e-6, fps))
            out = np.zeros_like(Y, dtype=np.float32)
            for i in range(Y.shape[0]):
                yi = Y[i].astype(float).tolist()
                # We assume a small MATLAB helper `mlspike_wrapper.m` exists in the MATLAB path:
                #   function s = mlspike_wrapper(y, dt, tau, a, sigma, drift, saturation)
                s = eng.mlspike_wrapper(
                    matlab.double(yi), float(dt), float(params.tau), float(params.a),
                    float(params.sigma), float(params.drift),
                    float(params.saturation) if params.saturation is not None else float('nan')
                )
                s = np.asarray(s).ravel().astype(np.float32)
                out[i, :len(s)] = s
            eng.quit()
            return out
        except Exception:
            # engine unavailable or wrapper missing → fallback
            try:
                eng.quit()
            except Exception:
                pass

    # 3) Fallback: nonnegative AR(1) 'inverse' difference.
    gamma = float(np.exp(-1.0 / max(1e-6, params.tau * float(fps))))
    return _relu_ar1_diff(Y, gamma=gamma)

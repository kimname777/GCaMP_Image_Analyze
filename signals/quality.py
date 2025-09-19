from __future__ import annotations

import numpy as np
from typing import Literal, Tuple


def compute_snr(
    dff: np.ndarray,
    mode: Literal["global", "per_cell"] = "global"
) -> float | np.ndarray:
    """
    Quick-and-robust SNR estimate from ΔF/F.

    Definition
    ----------
    SNR ≈ (P95 - P5) / MAD, where MAD = median(|x - median(x)|).

    Parameters
    ----------
    dff : (N, T) or (T,)
        ΔF/F traces.
    mode : {'global', 'per_cell'}
        * 'global'   : scalar SNR across all cells/timepoints.
        * 'per_cell' : one SNR per row (cell).

    Returns
    -------
    snr : float or (N,)
    """
    X = np.asarray(dff, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]

    if mode == "per_cell":
        p95 = np.percentile(X, 95, axis=1)
        p05 = np.percentile(X, 5, axis=1)
        sig = p95 - p05
        mad = np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1) + 1e-6
        return (sig / mad).astype(np.float32)

    # global
    p95 = float(np.percentile(X, 95))
    p05 = float(np.percentile(X, 5))
    sig = p95 - p05
    mad = float(np.median(np.abs(X - np.median(X)))) + 1e-6
    return float(sig / mad)

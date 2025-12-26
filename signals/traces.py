from __future__ import annotations

"""
Trace extraction: ROI & neuropil signals, baseline, ΔF/F.

Design goals
------------
- No hard dependency on SciPy/skimage; uses preprocess.neuropil if present,
  else a small built-in dilation fallback.
- Vectorized matrix-form computations for speed and clarity.
- Returns shapes the pipeline expects:
    (F_raw, F_np, F0, dff, np_masks)
"""

import numpy as np
from typing import Callable, Tuple, Union, Optional
from signals.baselines import compute_baseline_moving_percentile


# --- Lightweight dilation fallback (only if preprocess.neuropil is unavailable) ---
def _binary_dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    """Naive 3x3 dilation (no external deps)."""
    out = mask.astype(bool, copy=True)
    for _ in range(int(max(0, iters))):
        m = out
        out = (
            m
            | np.roll(m, 1, 0) | np.roll(m, -1, 0)
            | np.roll(m, 1, 1) | np.roll(m, -1, 1)
            | np.roll(np.roll(m, 1, 0), 1, 1)
            | np.roll(np.roll(m, 1, 0), -1, 1)
            | np.roll(np.roll(m, -1, 0), 1, 1)
            | np.roll(np.roll(m, -1, 0), -1, 1)
        )
    return out


def _ring_mask_local(mask: np.ndarray, inner: int = 2, outer: int = 6) -> np.ndarray:
    """Neuropil ring around a single ROI (local fallback)."""
    inner = int(max(0, inner))
    outer = int(max(inner + 1, outer))
    inner_m = _binary_dilate(mask, inner)
    outer_m = _binary_dilate(mask, outer)
    return np.logical_and(outer_m ^ inner_m, ~mask)


def _build_neuropil_masks(roi_masks: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Try `preprocess.neuropil.ring_mask`; if missing, use local fallback."""
    try:
        from preprocess.neuropil import ring_mask as _ring
        return np.stack([_ring(roi_masks[i], inner=inner, outer=outer) for i in range(roi_masks.shape[0])], axis=0)
    except Exception:
        return np.stack([_ring_mask_local(roi_masks[i], inner=inner, outer=outer) for i in range(roi_masks.shape[0])], axis=0)


# --- Public API ---
def extract_traces(
    stack: np.ndarray,
    roi_masks: np.ndarray,
    r: float = 0.8,
    inner: int = 2,
    outer: int = 6,
    baseline_fn: Union[Callable[[np.ndarray], np.ndarray], None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROI/neuropil traces and ΔF/F.

    Parameters
    ----------
    stack : (T, Y, X)
        Registered movie.
    roi_masks : (N, Y, X) or (Y, X)
        ROI binary masks.
    r : float
        Neuropil contamination factor. If you want to estimate it from data,
        consider `preprocess.neuropil.estimate_r_by_regression`.
    inner, outer : int
        Neuropil ring radii (in dilation iterations).
    baseline_fn : callable | None
        Function mapping (N, T) → (N, T) baseline. If None, uses per-ROI P10.

    Returns
    -------
    F_raw, F_np, F0, dff, np_masks
        Shapes: (N, T), (N, T), (N, T), (N, T), (N, Y, X)
    """
    S = np.asarray(stack)
    if S.ndim != 3:
        raise ValueError(f"`stack` must be (T, Y, X); got {S.shape}")
    T, Y, X = S.shape

    R = np.asarray(roi_masks)
    if R.ndim == 2:
        R = R[None, ...]
    if R.ndim != 3:
        raise ValueError(f"`roi_masks` must be (N, Y, X) or (Y, X); got {roi_masks.shape}")
    if R.shape[1:] != (Y, X):
        raise ValueError("ROI mask dims do not match stack frames.")

    N = int(R.shape[0])
    if N == 0:
        return (np.zeros((0, T), np.float32),
                np.zeros((0, T), np.float32),
                np.zeros((0, T), np.float32),
                np.zeros((0, T), np.float32),
                np.zeros((0, Y, X), bool))

    Rb = R.astype(bool, copy=False)
    NP_masks = _build_neuropil_masks(Rb, inner=inner, outer=outer)

    # Flatten for fast matrix ops
    F_flat = S.reshape(T, -1).astype(np.float32)        # (T, P)
    R_flat = Rb.reshape(N, -1).astype(np.float32)       # (N, P)
    NP_flat = NP_masks.reshape(N, -1).astype(np.float32)

    # Avoid division by zero (empty masks)
    R_area = np.maximum(R_flat.sum(axis=1, keepdims=True), 1.0)   # (N,1)
    NP_area = np.maximum(NP_flat.sum(axis=1, keepdims=True), 1.0) # (N,1)

    # Mean over ROI / neuropil: (T,P)@(P,N) -> (T,N) -> (N,T)
    F_raw = (F_flat @ R_flat.T).T / R_area
    F_np = (F_flat @ NP_flat.T).T / NP_area

    # Neuropil correction
    F_corr = F_raw - float(r) * F_np

    # Baseline for ΔF/F
    if baseline_fn is not None:
        F0 = baseline_fn(F_corr).astype(np.float32)
    else:
        # Default: use a moving-percentile baseline (more robust than a
        # global P10 for signals with drifts or sparse transients). Fall
        # back to global P10 if moving baseline computation fails.
        try:
            # defaults: 10th percentile, 90s window, 30 fps (can be
            # overridden by passing a custom baseline_fn)
            F0 = compute_baseline_moving_percentile(F_corr, p=10.0, win_s=90.0, fps=30.0)
        except Exception:
            q10 = np.percentile(F_corr, 10, axis=1, keepdims=True).astype(np.float32)
            F0 = np.repeat(q10, T, axis=1)

    # Safety floor for F0 to avoid huge ratios when baseline is very small
    # Use a per-ROI floor: max(1e-6, 1% of median(F0))
    try:
        median_F0 = np.median(F0, axis=1, keepdims=True)
        eps = np.maximum(1e-6, 0.01 * np.maximum(median_F0, 1e-6)).astype(np.float32)
    except Exception:
        eps = np.float32(1e-6)

    dff = (F_corr - F0) / (F0 + eps)
    return (F_raw.astype(np.float32),
            F_np.astype(np.float32),
            F0.astype(np.float32),
            dff.astype(np.float32),
            NP_masks.astype(bool, copy=False))

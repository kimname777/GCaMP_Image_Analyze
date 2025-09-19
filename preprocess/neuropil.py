from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

# Optional; if missing we fall back to iterative binary dilation using convolution.
try:
    from scipy.ndimage import binary_dilation  # type: ignore
except Exception:  # pragma: no cover
    binary_dilation = None


def _binary_dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    """Small helper to dilate without SciPy (very simple 3x3 struct)."""
    if binary_dilation is not None:
        return binary_dilation(mask, iterations=int(max(0, iters)))
    # Naive dilation using 3x3 max filter rolled by neighborhood
    out = mask.copy()
    for _ in range(int(max(0, iters))):
        m = out
        out = (
            m
            | np.roll(m, 1, 0)
            | np.roll(m, -1, 0)
            | np.roll(m, 1, 1)
            | np.roll(m, -1, 1)
            | np.roll(np.roll(m, 1, 0), 1, 1)
            | np.roll(np.roll(m, 1, 0), -1, 1)
            | np.roll(np.roll(m, -1, 0), 1, 1)
            | np.roll(np.roll(m, -1, 0), -1, 1)
        )
    return out


def ring_mask(mask: np.ndarray, inner: int = 2, outer: int = 6) -> np.ndarray:
    """
    Build a neuropil ring around a single ROI mask.

    Parameters
    ----------
    mask : (H, W) bool
    inner : int
        Inner dilation iterations from ROI boundary.
    outer : int
        Outer dilation iterations from ROI boundary.

    Returns
    -------
    ring : (H, W) bool
        A ring that excludes ROI pixels.
    """
    inner = int(max(0, inner))
    outer = int(max(inner + 1, outer))
    inner_m = _binary_dilate(mask, inner)
    outer_m = _binary_dilate(mask, outer)
    ring = np.logical_and(outer_m ^ inner_m, ~mask)
    return ring


def build_neuropil_masks(roi_masks: np.ndarray, inner: int = 2, outer: int = 6) -> np.ndarray:
    """
    Build neuropil masks for a stack of ROI masks.

    Parameters
    ----------
    roi_masks : (N, H, W) bool
        ROI masks.
    inner, outer : int
        Dilation radii.

    Returns
    -------
    np_masks : (N, H, W) bool
        Neuropil ring for each ROI.
    """
    R = np.asarray(roi_masks).astype(bool)
    if R.ndim != 3:
        raise ValueError("`roi_masks` must be (N, H, W) boolean.")
    N = R.shape[0]
    out = np.zeros_like(R, dtype=bool)
    for i in range(N):
        out[i] = ring_mask(R[i], inner=inner, outer=outer)
    return out


def estimate_r_by_regression(F_raw: np.ndarray, F_np: np.ndarray) -> float:
    """
    Estimate global neuropil contamination factor r (F_np contribution).

    A robust slope estimator based on medians:
        r ≈ median( (y - median(y)) / (x - median(x)) ), clipped to [0.3, 0.95]
    """
    x = np.asarray(F_np, dtype=np.float32).ravel()
    y = np.asarray(F_raw, dtype=np.float32).ravel()
    x_c = x - np.median(x)
    y_c = y - np.median(y)
    denom = np.abs(x_c) + 1e-9
    ratio = y_c / denom
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return 0.8
    slope = float(np.median(ratio))
    return float(np.clip(slope, 0.3, 0.95))


def neuropil_traces_from_masks(
    stack: np.ndarray,
    np_masks: np.ndarray,
) -> np.ndarray:
    """
    Compute neuropil signal per ROI as the mean over its ring mask.

    Parameters
    ----------
    stack : (T, H, W) array
        Registered movie.
    np_masks : (N, H, W) bool
        Neuropil ring masks.

    Returns
    -------
    F_np : (N, T) float32
    """
    S = np.asarray(stack, dtype=np.float32)
    M = np.asarray(np_masks, dtype=bool)
    if S.ndim != 3 or M.ndim != 3 or S.shape[1:] != M.shape[1:]:
        raise ValueError("Shape mismatch: stack(T,H,W) and np_masks(N,H,W) expected.")
    T, H, W = S.shape
    N = M.shape[0]
    F_np = np.zeros((N, T), dtype=np.float32)
    area = M.reshape(N, -1).sum(axis=1).astype(np.float32)
    area[area < 1] = 1.0
    flat = S.reshape(T, -1)  # (T, H*W)
    masks_flat = M.reshape(N, -1)  # (N, H*W)
    for i in range(N):
        idx = masks_flat[i]
        if not np.any(idx):
            continue
        vals = flat[:, idx]  # (T, K)
        F_np[i] = vals.mean(axis=1)
    return F_np

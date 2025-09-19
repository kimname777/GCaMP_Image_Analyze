from __future__ import annotations

import numpy as np
from typing import Literal

# Optional deps with graceful fallback
try:
    from skimage.filters import gaussian  # type: ignore
except Exception:  # pragma: no cover
    gaussian = None

try:
    from scipy.ndimage import median_filter, uniform_filter  # type: ignore
except Exception:  # pragma: no cover
    median_filter = None
    uniform_filter = None

try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover
    pywt = None

try:
    from bm3d import bm3d  # type: ignore
except Exception:  # pragma: no cover
    bm3d = None

Method = Literal["gaussian", "median", "mean", "wavelet", "bm3d", "none"]


def denoise_stack(
    stack: np.ndarray,
    method: Method = "gaussian",
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Denoise a movie stack frame-by-frame.

    Parameters
    ----------
    stack : (T, Y, X) float/uint array
        Input movie. This function does not modify the input in-place.
    method : {'gaussian','median','mean','wavelet','bm3d','none'}
        Denoising algorithm; unavailable methods gracefully fall back to 'none'.
    sigma : float
        Smoothing strength. Interpreted per-method:
          - gaussian: std dev (pixels)
          - median/mean: neighborhood size ≈ 2*sigma+1
          - wavelet: affects universal threshold
          - bm3d: ignored (uses sigma_psd=0.05 internally)

    Returns
    -------
    out : (T, Y, X) array
        Denoised stack in the same dtype as input (whenever possible).
    """
    A = np.asarray(stack)
    if A.ndim != 3:
        raise ValueError("`stack` must be a 3D array (T, Y, X).")
    T, Y, X = A.shape
    dtype = A.dtype
    out = np.empty_like(A, dtype=np.float32)

    if method == "none":
        return A.copy()

    # --- Gaussian ---
    if method == "gaussian" and gaussian is not None:
        for t in range(T):
            out[t] = gaussian(A[t], sigma=float(sigma), preserve_range=True)
        return out.astype(dtype, copy=False)

    # --- Median ---
    if method == "median" and median_filter is not None:
        k = int(max(1, 2 * round(float(sigma)) + 1))
        for t in range(T):
            out[t] = median_filter(A[t], size=k)
        return out.astype(dtype, copy=False)

    # --- Mean (uniform) ---
    if method == "mean" and uniform_filter is not None:
        k = int(max(1, 2 * round(float(sigma)) + 1))
        for t in range(T):
            out[t] = uniform_filter(A[t], size=k)
        return out.astype(dtype, copy=False)

    # --- Wavelet denoise (soft-thresholding) ---
    if method == "wavelet" and pywt is not None:
        for t in range(T):
            coeffs2 = pywt.wavedec2(A[t].astype(np.float32), "db2", level=2)
            coeffs2 = list(coeffs2)
            # Estimate noise (HH subband)
            sigma_n = np.median(np.abs(coeffs2[-1][0])) / 0.6745 + 1e-6
            uthresh = sigma_n * np.sqrt(2 * np.log(A[t].size))
            coeffs2[1:] = [
                tuple(pywt.threshold(c, value=uthresh, mode="soft") for c in sub)
                for sub in coeffs2[1:]
            ]
            rec = pywt.waverec2(coeffs2, "db2")
            out[t] = rec[:Y, :X]
        return out.astype(dtype, copy=False)

    # --- BM3D (if available) ---
    if method == "bm3d" and bm3d is not None:
        # Normalize per-frame to help BM3D
        for t in range(T):
            frame = A[t].astype(np.float32)
            m = float(frame.max())
            if m <= 1e-8:
                out[t] = frame
                continue
            den = bm3d(frame / m, sigma_psd=0.05) * m
            out[t] = den
        return out.astype(dtype, copy=False)

    # Fallback: identity copy
    return A.copy()


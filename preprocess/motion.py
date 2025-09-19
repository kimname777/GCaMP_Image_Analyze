from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Literal, Optional

# Optional subpixel shifter; fallback to integer np.roll if not available.
try:
    from scipy.ndimage import shift as ndi_shift  # type: ignore
except Exception:  # pragma: no cover
    ndi_shift = None

Backend = Literal["identity", "rigid"]


def _phase_corr_shift(ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
    """
    Estimate (dy, dx) by phase correlation (FFT-based), subpixel via parabolic fit.

    Notes
    -----
    * Works on single-channel 2D arrays.
    * Robust to uniform intensity changes; assumes mostly translational motion.
    """
    f0 = np.fft.rfft2(ref)
    f1 = np.fft.rfft2(img)
    eps = 1e-9
    R = f0 * np.conj(f1)
    R /= (np.abs(R) + eps)
    c = np.fft.irfft2(R, s=ref.shape)
    # Peak location
    y0, x0 = np.unravel_index(np.argmax(c), c.shape)
    H, W = c.shape

    # Wrap to negative frequencies
    if y0 > H // 2:
        y0 = y0 - H
    if x0 > W // 2:
        x0 = x0 - W

    # Subpixel refinement by quadratic fit around the peak (if neighbors exist)
    def _subpixel_offset(arr: np.ndarray, coord: int, axis_len: int) -> float:
        # neighbors: -1, 0, +1 around peak along a dimension
        m1 = arr[(coord - 1) % axis_len]
        m0 = arr[coord % axis_len]
        p1 = arr[(coord + 1) % axis_len]
        denom = (m1 - 2 * m0 + p1)
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (m1 - p1) / denom

    # Extract 1D slices through the peak for subpixel estimation
    row = c[(y0 % H), :]
    col = c[:, (x0 % W)]
    dy_sub = _subpixel_offset(col, y0 % H, H)
    dx_sub = _subpixel_offset(row, x0 % W, W)

    return float(y0 + dy_sub), float(x0 + dx_sub)


def _apply_shift(img: np.ndarray, dy: float, dx: float, pad_mode: str = "reflect") -> np.ndarray:
    """Shift a 2D image. Uses subpixel interpolation if SciPy is available."""
    if ndi_shift is not None:
        return ndi_shift(img, shift=(dy, dx), order=1, mode=pad_mode)
    # Fallback: integer shift only (no interpolation)
    return np.roll(np.roll(img, int(round(dy)), axis=0), int(round(dx)), axis=1)


def motion_correct(
    stack: np.ndarray,
    cfg: Dict,
) -> Tuple[np.ndarray, Dict]:
    """
    Motion-correct a movie stack (rigid translation per frame).

    Parameters
    ----------
    stack : (T, Y, X) array
        Input movie (float or uint). Not modified in-place.
    cfg : dict
        {
          "backend": "rigid" | "identity",
          "reference": "median" | "mean" | "first",
          "max_shift": int,          # clip absolute shift (pixels), optional
          "pad_mode": "reflect" | "nearest" | "constant" | ... (SciPy modes)
        }

    Returns
    -------
    reg : (T, Y, X) array
        Motion-corrected stack (float32).
    meta : dict
        {
          "backend": str,
          "reference": str,
          "shifts": (T, 2) float array of (dy, dx),
          "drift_std": float  # std of shifts
        }

    Notes
    -----
    * If dependencies are missing, falls back to identity alignment.
    * Designed as a thin, predictable baseline; you can swap in NoRMCorre/Suite2p.
    """
    A = np.asarray(stack)
    if A.ndim != 3:
        raise ValueError("`stack` must be (T, Y, X).")
    T, Y, X = A.shape
    backend: Backend = cfg.get("backend", "rigid")
    if backend not in ("rigid", "identity"):
        backend = "rigid"

    ref_mode = (cfg.get("reference") or "median").lower()
    pad_mode = cfg.get("pad_mode", "reflect")
    max_shift = float(cfg.get("max_shift", 0.0))

    # Build reference
    if ref_mode == "first":
        ref = A[0].astype(np.float32)
    elif ref_mode == "mean":
        ref = A.mean(axis=0).astype(np.float32)
    else:
        ref = np.median(A, axis=0).astype(np.float32)

    # Identity path
    if backend == "identity":
        meta = {
            "backend": "identity",
            "reference": ref_mode,
            "shifts": np.zeros((T, 2), dtype=np.float32),
            "drift_std": 0.0,
        }
        return A.astype(np.float32, copy=True), meta

    # Rigid registration
    reg = np.empty_like(A, dtype=np.float32)
    shifts = np.zeros((T, 2), dtype=np.float32)
    reg[0] = A[0].astype(np.float32)
    for t in range(T):
        img = A[t].astype(np.float32)
        try:
            dy, dx = _phase_corr_shift(ref, img)
        except Exception:
            dy, dx = 0.0, 0.0
        if max_shift > 0.0:
            dy = float(np.clip(dy, -max_shift, max_shift))
            dx = float(np.clip(dx, -max_shift, max_shift))
        shifts[t] = (dy, dx)
        reg[t] = _apply_shift(img, dy, dx, pad_mode=pad_mode)

    meta = {
        "backend": "rigid",
        "reference": ref_mode,
        "shifts": shifts,
        "drift_std": float(np.std(shifts, axis=0).mean()),
    }
    return reg, meta

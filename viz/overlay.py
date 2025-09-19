from __future__ import annotations
"""
overlay.py
----------
Export an RGB TIFF with ROI contours overlaid on the mean image.

* Uses `tifffile` if available; otherwise falls back to `imageio.v3.imwrite`.
* Contour extraction lazily uses scikit-image if present; simple fallback otherwise.
"""

import os
import numpy as np

# Optional writers
try:
    import tifffile as tiff  # type: ignore
except Exception:  # pragma: no cover
    tiff = None
try:
    import imageio.v3 as iio  # type: ignore
except Exception:  # pragma: no cover
    iio = None


def _scale_to_u8(img: np.ndarray) -> np.ndarray:
    """Min–max scale to uint8 [0..255]."""
    img = img.astype(np.float32)
    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    if hi <= lo + 1e-12:
        return np.zeros(img.shape, np.uint8)
    return np.clip(255.0 * (img - lo) / (hi - lo), 0, 255).astype(np.uint8)


def _colors_n(n: int) -> np.ndarray:
    """Generate n distinct-ish colors (cycle a small palette if needed)."""
    base = np.array(
        [[255, 64, 64], [64, 255, 64], [64, 64, 255], [255, 255, 64], [64, 255, 255], [255, 64, 255]],
        dtype=np.float32,
    )
    if n <= base.shape[0]:
        return base[:n] / 255.0
    reps = int(np.ceil(n / base.shape[0]))
    return np.vstack([base for _ in range(reps)])[:n] / 255.0


def _fallback_contours(m: np.ndarray):
    """
    Cheap contour substitute returning boundary points as a polyline.
    """
    m = m.astype(bool)
    e = np.zeros_like(m, dtype=bool)
    e[:-1, :] |= m[:-1, :] ^ m[1:, :]
    e[:, :-1] |= m[:, :-1] ^ m[:, 1:]
    ys, xs = np.nonzero(e)
    if ys.size == 0:
        return []
    return [np.stack([ys, xs], axis=1).astype(float)]


def save_overlay_tiff(
    stack: np.ndarray,
    roi_masks: np.ndarray,
    out_path: str,
    alpha: float = 0.8,
    line: int = 1,
):
    """
    Save an RGB TIFF: mean image in grayscale with colored ROI contours.

    Parameters
    ----------
    stack : (T, H, W)
    roi_masks : (N, H, W)
    out_path : str
        Destination filepath ('.tif' recommended).
    alpha : float
        Blend factor for the overlay color.
    line : int
        Half line width in pixels (painted as a small square around the contour).
    """
    mean_img = stack.mean(axis=0)
    rgb = np.stack([_scale_to_u8(mean_img)] * 3, axis=-1).astype(np.float32) / 255.0

    colors = _colors_n(roi_masks.shape[0])
    Y, X = mean_img.shape

    # Lazy import: scikit-image for contours if not disabled
    fc = None
    if os.environ.get("GCAMP_DISABLE_SKIMAGE", "0") not in ("1", "true", "True"):
        try:
            from skimage.measure import find_contours as fc  # type: ignore
        except Exception:
            fc = None

    for i in range(roi_masks.shape[0]):
        m = roi_masks[i].astype(bool)
        if not m.any():
            continue
        if fc is not None:
            try:
                contours = fc(m.view(np.float32), 0.5)  # type: ignore
            except Exception:
                contours = _fallback_contours(m)
        else:
            contours = _fallback_contours(m)

        for cnt in contours:
            cnt = np.asarray(cnt, dtype=np.int32)
            for (y, x) in cnt:
                if 0 <= y < Y and 0 <= x < X:
                    y0, y1 = max(0, y - line + 1), min(Y, y + line)
                    x0, x1 = max(0, x - line + 1), min(X, x + line)
                    rgb[y0:y1, x0:x1, :] = (1.0 - alpha) * rgb[y0:y1, x0:x1, :] + alpha * colors[i]

    out_img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if tiff is not None:
        tiff.imwrite(out_path, out_img)
    elif iio is not None:
        iio.imwrite(out_path, out_img)
    else:
        raise RuntimeError("Neither tifffile nor imageio are available to write the overlay.")

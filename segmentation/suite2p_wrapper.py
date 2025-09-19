from __future__ import annotations

"""
Suite2p integration and a light-weight threshold fallback.

Notes
-----
- We avoid importing Suite2p at module import time; heavy ops happen in a child process.
- The threshold fallback uses scikit-image if present, but works without it.
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import tifffile as tiff
except Exception:  # pragma: no cover
    tiff = None


# -------- Threshold fallback segmentation -------- #
def _lazy_skimage():
    """Import scikit-image pieces only when needed (avoid C-ext at import time)."""
    try:
        from skimage.filters import threshold_otsu
        from skimage.measure import label, regionprops
        from skimage.morphology import opening, disk
        return threshold_otsu, label, regionprops, opening, disk
    except Exception:
        return None, None, None, None, None


def segment_cells_threshold(stack: np.ndarray, diameter: int = 14) -> np.ndarray:
    """
    Basic segmentation by thresholding the mean image (robust fallback).

    Parameters
    ----------
    stack : (T, Y, X)
    diameter : int
        Approximate cell diameter; used to set morphology scale and min area.
    """
    if stack is None or stack.ndim != 3:
        return np.zeros((0, 0, 0), dtype=bool)
    mean_img = stack.mean(axis=0).astype(np.float32)
    Y, X = mean_img.shape

    threshold_otsu, label, regionprops, opening, disk = _lazy_skimage()
    if threshold_otsu is None:
        thr = float(np.percentile(mean_img, 99.0))
        bw = mean_img > thr
    else:
        thr = threshold_otsu(mean_img)
        bw = opening(mean_img > thr, disk(max(1, diameter // 6)))

    if label is None or regionprops is None:
        # Minimal connected-component labeling without skimage, using scipy is overkill here.
        # Fallback: keep top-K bright blobs via percentile mask; not ideal, but safe.
        ys, xs = np.where(bw)
        if ys.size == 0:
            return np.zeros((0, Y, X), dtype=bool)
        m = np.zeros((Y, X), bool)
        m[ys, xs] = True
        return np.array([m], dtype=bool)

    lbl = label(bw)
    props = regionprops(lbl)
    masks = []
    min_area = max(20, (int(diameter) ** 2) // 4)
    for p in props:
        if p.area < min_area:
            continue
        m = (lbl == p.label)
        masks.append(m)
    if not masks:
        return np.zeros((0, Y, X), dtype=bool)
    return np.stack(masks, axis=0)


# -------- Suite2p (child process) -------- #
def _worker_cmd(tif_path: Path, out_npz: Path, diameter: int, tau: float) -> list[str]:
    """
    Build a command to run the Suite2p child worker, handling frozen builds.
    """
    if getattr(sys, "frozen", False):
        # PyInstaller: prefer a sibling Worker.exe if you ship one
        worker = Path(sys.argv[0]).with_name("Worker.exe")
        return [str(worker), str(tif_path), str(out_npz), str(int(diameter)), str(float(tau))]
    # Dev: run as a Python module
    return [
        sys.executable,
        "-m",
        "segmentation.s2p_child",
        str(tif_path),
        str(out_npz),
        str(int(diameter)),
        str(float(tau)),
    ]


def segment_cells_suite2p(stack: np.ndarray, diameter: int = 14, tau: float = 1.0) -> np.ndarray:
    """
    Run Suite2p ROI detection in a separate process and load masks.
    """
    if stack is None or stack.ndim != 3:
        return np.zeros((0, 0, 0), dtype=bool)
    if tiff is None:
        # no tifffile -> fallback
        return segment_cells_threshold(stack, diameter=int(diameter))

    tmpdir = Path(tempfile.mkdtemp(prefix="s2p_sub_"))
    try:
        tif_path = tmpdir / "movie.tif"
        tiff.imwrite(str(tif_path), stack.astype("float32"), imagej=True)
        out_npz = tmpdir / "masks.npz"

        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMBA_NUM_THREADS", "1")
        env.setdefault("NUMBA_THREADING_LAYER", "safe")

        cmd = _worker_cmd(tif_path, out_npz, int(diameter), float(tau))
        # Hide console window on Windows (nice for GUI)
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True, creationflags=creationflags)

        if ret.returncode != 0 or not out_npz.exists():
            return segment_cells_threshold(stack, diameter=int(diameter))
        return np.load(out_npz)["masks"].astype(bool)
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


# Backward-compat short alias used by some callers
def segment_cells(stack: np.ndarray, backend: str = "suite2p", diameter: int = 14, tau: float = 1.0) -> np.ndarray:
    """Default to Suite2p; keep for backward compatibility with legacy imports."""
    if backend == "suite2p":
        return segment_cells_suite2p(stack, diameter=diameter, tau=tau)
    return segment_cells_threshold(stack, diameter=diameter)

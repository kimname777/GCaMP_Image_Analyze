from __future__ import annotations

"""
Thin wrapper around CaImAn's CNMF-E source extraction.

Design goals
------------
- Import CaImAn only when this module is used (heavy dependency).
- Graceful fallback to threshold segmentation if CaImAn or tifffile is missing.
- Return binary ROI masks (N,Y,X) compatible with the rest of the pipeline.
"""

from pathlib import Path
from typing import Optional
import tempfile
import shutil
import numpy as np


def _have_caiman() -> bool:
    try:
        import caiman  # noqa: F401
        from caiman.source_extraction.cnmf import cnmf as _  # noqa: F401
        from caiman.source_extraction.cnmf import params as _  # noqa: F401
        return True
    except Exception:
        return False


def segment_cells_cnmf(
    stack: np.ndarray,
    gSig: int = 5,
    merge_thr: float = 0.8,
    k: int = 200,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Run CNMF-E via CaImAn and produce ROI masks.

    Parameters
    ----------
    stack : (T, Y, X)
        Input movie (prefer motion-corrected).
    gSig : int
        Gaussian width of a neuron (pixels).
    merge_thr : float
        Merge threshold for spatial components.
    k : int
        Initial number of components.
    fps : float
        Frame rate (Hz).

    Returns
    -------
    masks : (N, Y, X) bool
    """
    from .suite2p_wrapper import segment_cells_threshold  # light fallback

    if stack is None or stack.ndim != 3:
        return segment_cells_threshold(stack, diameter=max(2 * gSig + 2, 12))

    if not _have_caiman():
        return segment_cells_threshold(stack, diameter=max(2 * gSig + 2, 12))

    # Lazy imports (heavy)
    try:
        import tifffile as tiff
    except Exception:
        return segment_cells_threshold(stack, diameter=max(2 * gSig + 2, 12))

    from caiman.source_extraction.cnmf import cnmf as cnmf
    from caiman.source_extraction.cnmf import params as params

    T, Y, X = stack.shape
    tmpdir = Path(tempfile.mkdtemp(prefix="cnmf_tmp_"))
    try:
        # CaImAn prefers file input
        tif_path = tmpdir / "movie.tif"
        tiff.imwrite(str(tif_path), stack.astype(np.float32), imagej=True)

        p = params.CNMFParams(
            {
                "fnames": [str(tif_path)],
                "fr": float(fps),
                "gSig": (int(gSig), int(gSig)),
                "merge_thr": float(merge_thr),
                "p": 1,
                "nb": 1,
                "rf": 40,
                "stride": 20,
                "K": int(k),
                "method_init": "greedy_roi",
                "rolling_sum": True,
            }
        )
        cnm = cnmf.CNMF(n_processes=1, params=p).fit_file()
        # Convert sparse A to dense masks
        A = cnm.estimates.A.toarray().reshape((Y, X, -1), order="F").transpose(2, 0, 1)  # (N, Y, X)
        if A.size == 0:
            return np.zeros((0, Y, X), dtype=bool)
        # Adaptive threshold per component
        mx = A.max(axis=(1, 2), keepdims=True) + 1e-8
        masks = (A > (0.2 * mx)).astype(bool)

        # Post-process: remove tiny ROIs and relabel overlaps
        from .rois import remove_small_rois, relabel_unique
        masks = remove_small_rois(masks, min_area=max(20, (gSig * gSig) // 2))
        masks = relabel_unique(masks)
        return masks
    except Exception:
        return segment_cells_threshold(stack, diameter=max(2 * gSig + 2, 12))
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

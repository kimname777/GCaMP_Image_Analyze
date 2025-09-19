from __future__ import annotations

import os
import numpy as np
from typing import Literal

Backend = Literal["suite2p", "cnmf", "threshold"]


def _empty_like_TYX(stack: np.ndarray | None) -> np.ndarray:
    """
    Return an empty masks array with shape (0, Y, X) even if `stack` is invalid.
    """
    if stack is None or not hasattr(stack, "shape"):
        return np.zeros((0, 0, 0), dtype=bool)
    shp = tuple(getattr(stack, "shape", (0, 0, 0)))
    if len(shp) >= 2:
        Y, X = shp[-2], shp[-1]
    else:
        Y, X = 0, 0
    return np.zeros((0, Y, X), dtype=bool)


def segment_cells(
    stack: np.ndarray,
    backend: Backend = "suite2p",
    diameter: int = 14,
    tau: float = 1.0,
) -> np.ndarray:
    """
    Unified entry point for ROI segmentation.

    Parameters
    ----------
    stack : (T, Y, X) float/uint array
        Registered movie or raw stack (rigid motion is recommended beforehand).
    backend : {'suite2p','cnmf','threshold'}
        Segmentation strategy. If not available, falls back gracefully.
    diameter : int
        Rough cell diameter (pixels) used by several backends and the threshold fallback.
    tau : float
        Suite2p temporal parameter (decay in seconds). Ignored by other backends.

    Returns
    -------
    masks : (N, Y, X) boolean
        One binary mask per detected ROI.
    """
    # Input validation
    if stack is None or getattr(stack, "ndim", 0) != 3:
        return _empty_like_TYX(stack)

    backend = (backend or "suite2p").lower()

    # Honor environment flags to disable heavy deps
    disable_s2p = os.environ.get("GCAMP_DISABLE_S2P", "0") in ("1", "true", "True")
    disable_cnmf = os.environ.get("GCAMP_DISABLE_CAIMAN", "0") in ("1", "true", "True")

    # 1) Explicit "threshold" → light-weight fallback
    if backend == "threshold":
        from .suite2p_wrapper import segment_cells_threshold
        return segment_cells_threshold(stack, diameter=diameter)

    # 2) CNMF-E (CaImAn) if allowed and installed
    if backend == "cnmf" and not disable_cnmf:
        try:
            from .cnmf_wrapper import segment_cells_cnmf
            masks = segment_cells_cnmf(stack, gSig=max(3, diameter // 3))
            if isinstance(masks, np.ndarray) and masks.ndim == 3:
                return masks.astype(bool, copy=False)
        except Exception:
            pass  # fall through to threshold later

    # 3) Suite2p if allowed and installed
    if backend == "suite2p" and not disable_s2p:
        try:
            from .suite2p_wrapper import segment_cells_suite2p
            masks = segment_cells_suite2p(stack, diameter=diameter, tau=tau)
            if isinstance(masks, np.ndarray) and masks.ndim == 3:
                return masks.astype(bool, copy=False)
        except Exception:
            pass  # fall through to threshold

    # 4) Last resort: threshold segmentation
    from .suite2p_wrapper import segment_cells_threshold
    return segment_cells_threshold(stack, diameter=diameter)

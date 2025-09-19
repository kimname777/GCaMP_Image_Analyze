from __future__ import annotations

"""
ROI utility functions:
- IoU / greedy merge
- size filtering
- watershed-based split with safe fallbacks
- unique relabeling to resolve overlaps
"""

import os
import numpy as np
from typing import Tuple


def masks_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union for two boolean masks."""
    a = a.astype(bool, copy=False)
    b = b.astype(bool, copy=False)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum() + 1e-9
    return float(inter / union)


def merge_overlapping_rois(masks: np.ndarray, iou_thr: float = 0.3) -> np.ndarray:
    """
    Greedy IoU-based merge of overlapping ROIs.

    Parameters
    ----------
    masks : (N, Y, X) bool
    iou_thr : float
        Threshold above which two ROIs will be merged.
    """
    if masks is None or masks.size == 0:
        return np.zeros((0, 0, 0), dtype=bool)
    N = masks.shape[0]
    used = np.zeros(N, dtype=bool)
    merged = []
    for i in range(N):
        if used[i]:
            continue
        cur = masks[i].astype(bool).copy()
        used[i] = True
        for j in range(i + 1, N):
            if used[j]:
                continue
            if masks_iou(cur, masks[j]) >= float(iou_thr):
                cur |= masks[j]
                used[j] = True
        merged.append(cur)
    return np.stack(merged, axis=0) if merged else masks


def remove_small_rois(masks: np.ndarray, min_area: int = 30) -> np.ndarray:
    """Drop ROIs smaller than `min_area` pixels."""
    if masks is None or masks.size == 0:
        return np.zeros((0, 0, 0), dtype=bool)
    min_area = int(max(1, min_area))
    keep = [m for m in masks if int(m.sum()) >= min_area]
    if not keep:
        return np.zeros((0, masks.shape[1], masks.shape[2]), dtype=bool)
    return np.stack(keep, axis=0)


def _fallback_split_by_markers(mask: np.ndarray, min_distance: int = 3) -> np.ndarray:
    """
    Very simple split using distance transform markers if skimage is unavailable.

    Strategy
    --------
    - Compute distance transform inside the mask.
    - Use a binary threshold to produce multiple markers.
    - Assign each pixel to the nearest marker (Voronoi-like).
    """
    mask = mask.astype(bool)
    if not mask.any():
        return mask[None]

    try:
        from scipy.ndimage import distance_transform_edt, label as ndi_label
    except Exception:
        return mask[None]

    dist = distance_transform_edt(mask)
    markers_mask = dist > float(min_distance)
    markers, n = ndi_label(markers_mask)
    if n <= 1:
        return mask[None]

    # Marker centroids
    centers = []
    for k in range(1, n + 1):
        idx = np.argwhere(markers == k)
        if idx.size == 0:
            continue
        centers.append(idx.mean(axis=0))
    if not centers:
        return mask[None]
    centers = np.asarray(centers, dtype=np.float32)  # (M, 2) [y, x]

    ysxs = np.argwhere(mask)  # (P, 2)
    if ysxs.size == 0:
        return mask[None]
    diff = ysxs[:, None, :] - centers[None, :, :]  # (P, M, 2)
    d2 = np.sum(diff * diff, axis=2)  # (P, M)
    nearest = np.argmin(d2, axis=1) + 1  # 1..M

    labels = np.zeros_like(mask, dtype=np.int32)
    labels[ysxs[:, 0], ysxs[:, 1]] = nearest

    parts = [(labels == k) for k in range(1, int(labels.max()) + 1)]
    parts = [m for m in parts if m.sum() > 0]
    return np.stack(parts, axis=0) if parts else mask[None]


def split_roi_watershed(mask: np.ndarray, min_distance: int = 3) -> np.ndarray:
    """
    Split a single ROI into multiple parts using watershed.

    Fallbacks
    ---------
    - If scikit-image is not available or fails, use `_fallback_split_by_markers`.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return mask[None]

    try:
        from scipy.ndimage import distance_transform_edt, label as ndi_label
    except Exception:
        return mask[None]

    dist = distance_transform_edt(mask)
    markers_mask = dist > float(min_distance)
    markers, _ = ndi_label(markers_mask)

    # Respect environment flag to disable skimage
    if os.environ.get("GCAMP_DISABLE_SKIMAGE", "0") in ("1", "true", "True"):
        return _fallback_split_by_markers(mask, min_distance)

    try:
        from skimage.segmentation import watershed
        labels = watershed(-dist, markers, mask=mask)
    except Exception:
        return _fallback_split_by_markers(mask, min_distance)

    parts = [(labels == k) for k in range(1, int(labels.max()) + 1)]
    parts = [m for m in parts if m.sum() > 0]
    return np.stack(parts, axis=0) if parts else mask[None]


def relabel_unique(masks: np.ndarray) -> np.ndarray:
    """
    Resolve overlaps by assigning each overlapping pixel to the nearest ROI center.
    """
    if masks is None or masks.size == 0:
        return np.zeros((0, 0, 0), dtype=bool)
    N, Y, X = masks.shape
    centers = []
    for i in range(N):
        idx = np.argwhere(masks[i])
        centers.append(idx.mean(axis=0) if idx.size else np.array([np.nan, np.nan]))
    centers = np.asarray(centers, dtype=np.float32)

    out = masks.astype(bool).copy()
    stack = out.astype(np.uint8).sum(axis=0)
    ov = np.argwhere(stack > 1)
    for y, x in ov:
        d = np.sqrt((centers[:, 0] - y) ** 2 + (centers[:, 1] - x) ** 2)
        k = int(np.nanargmin(d))
        out[:, y, x] = False
        out[k, y, x] = True
    return out

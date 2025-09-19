import numpy as np
from typing import Dict

def _first_half_decay_time(trace: np.ndarray, peak_idx: int, peak_val: float, fps: float) -> float:
    """Return seconds from peak to first time trace <= half peak. NaN if never decays."""
    if not np.isfinite(peak_val) or peak_idx is None:
        return float("nan")
    half = 0.5 * peak_val
    post = trace[int(peak_idx):]
    if post.size == 0:
        return float("nan")
    below = np.where(post <= half)[0]
    if below.size == 0:
        return float("nan")
    return float(below[0]) / float(fps)

def compute_kinetics(dff: np.ndarray, fps: float = 30.0) -> Dict[str, float]:
    
    """
    Compute simple kinetics summaries from ΔF/F traces.
    Returns population means across ROIs:
      - peak_mean: mean of per-ROI max(ΔF/F)
      - t_peak_mean_s: mean latency-to-peak in seconds
      - t_half_decay_mean_s: mean half-decay time (peak -> first <= half-peak)
    """
    if dff is None or dff.ndim != 2 or dff.size == 0:
        return {"peak_mean": float("nan"), "t_peak_mean_s": float("nan"), "t_half_decay_mean_s": float("nan")}
    # Replace NaNs only for argmax by using -inf sentinel
    dff_for_arg = np.where(np.isnan(dff), -np.inf, dff)
    peak_vals = np.nanmax(dff, axis=1)
    peak_idx = np.argmax(dff_for_arg, axis=1)
    t_peak = peak_idx.astype(float) / float(fps)
    t_half = [
        _first_half_decay_time(dff[i], int(peak_idx[i]), float(peak_vals[i]), fps)
        for i in range(dff.shape[0])
    ]
    return {
        "peak_mean": float(np.nanmean(peak_vals)),
        "t_peak_mean_s": float(np.nanmean(t_peak)),
        "t_half_decay_mean_s": float(np.nanmean(t_half))
    }

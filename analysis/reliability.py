import numpy as np
from typing import Dict

def split_half_reliability(dff: np.ndarray) -> float:
    """
    Split-half reliability over time: correlate ROI means in first vs second half.
    Returns Pearson r. NaN if undefined.
    """
    if dff is None or dff.ndim != 2 or dff.size == 0:
        return float("nan")
    N, T = dff.shape
    if T < 4:
        return float("nan")
    h = T // 2
    v1 = np.nanmean(dff[:, :h], axis=1)
    v2 = np.nanmean(dff[:, h:], axis=1)
    s1 = np.nanstd(v1); s2 = np.nanstd(v2)
    if not np.isfinite(s1) or not np.isfinite(s2) or s1 < 1e-12 or s2 < 1e-12:
        return float("nan")
    r = np.corrcoef(v1, v2)[0, 1]
    return float(r)

def compute_reliability(dff: np.ndarray) -> Dict[str, float]:
    """Wrapper returning a dict for downstream JSON/report usage."""
    return {"split_half_r": split_half_reliability(dff)}

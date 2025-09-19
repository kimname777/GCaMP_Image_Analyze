import numpy as np
from typing import Tuple

def bootstrap_mean(x: np.ndarray, n: int = 1000, ci: float = 0.95, seed: int | None = None) -> Tuple[float, float]:
    """
    Nonparametric bootstrap CI for the mean.
    x: 1D array
    n: bootstrap samples
    ci: central confidence level
    """
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if seed is not None:
        rng = np.random.default_rng(seed)
        randint = rng.integers
    else:
        randint = np.random.randint
    vals = np.empty(n, dtype=float)
    for i in range(n):
        idx = randint(0, x.size, size=x.size)
        vals[i] = float(np.mean(x[idx]))
    vals.sort()
    lo_idx = int(np.floor((1 - ci) / 2 * n))
    hi_idx = int(np.ceil((1 + ci) / 2 * n)) - 1
    lo = float(vals[max(0, min(n-1, lo_idx))])
    hi = float(vals[max(0, min(n-1, hi_idx))])
    return lo, hi

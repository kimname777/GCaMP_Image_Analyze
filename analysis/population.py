import numpy as np
from typing import Tuple

def pca(dff: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA via SVD on mean-centered ΔF/F (cells x time).
    Returns U[:, :k], S[:k], Vt[:k].
    """
    if dff is None or dff.ndim != 2 or dff.size == 0:
        return np.empty((0, 0)), np.empty((0,)), np.empty((0, 0))
    X = dff - np.nanmean(dff, axis=1, keepdims=True)
    X = np.nan_to_num(X, copy=False)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = max(0, min(k, S.size))
    return U[:, :k], S[:k], Vt[:k]

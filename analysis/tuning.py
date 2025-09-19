import numpy as np
from typing import Dict, List, Optional, Tuple

def _trial_responses(dff: np.ndarray, stim_events: List[dict], fps: float, window_s: float = 1.0) -> Dict[float, List[np.ndarray]] | None:
    """
    Compute per-trial mean ΔF/F in a post-stimulus window for each event.
    Returns a dictionary: direction(deg) -> list of (N,) response vectors across trials.
    """
    if dff is None or dff.ndim != 2 or dff.size == 0 or not stim_events:
        return None
    N, T = dff.shape
    w = max(1, int(round(window_s * fps)))
    by_dir: Dict[float, List[np.ndarray]] = {}
    for ev in stim_events:
        onset_s = float(ev.get("onset", 0.0))
        direction = float(ev.get("direction", 0.0))
        onset_idx = max(0, int(round(onset_s * fps)))
        end_idx = min(T, onset_idx + w)
        seg = dff[:, onset_idx:end_idx]
        resp = np.nanmean(seg, axis=1)  # (N,)
        by_dir.setdefault(direction, []).append(resp)
    return by_dir

def _population_tuning(by_dir: Dict[float, List[np.ndarray]]) -> Tuple[List[float], np.ndarray]:
    """
    Average across trials within each direction, then across cells → population tuning curve.
    Returns (dirs_sorted, pop_mean_per_dir)
    """
    if not by_dir:
        return [], np.empty((0,), dtype=float)
    dirs = sorted(by_dir.keys())
    pop = []
    for d in dirs:
        trials = by_dir[d]
        if len(trials) == 0:
            pop.append(np.nan)
            continue
        trials_mat = np.stack(trials, axis=1)  # (N, n_trials)
        per_cell = np.nanmean(trials_mat, axis=1)  # (N,)
        pop.append(float(np.nanmean(per_cell)))
    return dirs, np.array(pop, dtype=float)

def _osi_dsi(dirs: List[float], pop: np.ndarray) -> Tuple[float, float]:
    """
    Compute OSI (2θ vector sum) and DSI (pref vs opposite) from a population tuning curve.
    """
    if len(dirs) == 0 or pop.size == 0 or not np.isfinite(pop).any():
        return float("nan"), float("nan")
    theta = np.deg2rad(np.array(dirs, dtype=float))
    w = np.nan_to_num(pop, nan=0.0, posinf=0.0, neginf=0.0)
    vx = float((w * np.cos(2 * theta)).sum())
    vy = float((w * np.sin(2 * theta)).sum())
    OSI = np.hypot(vx, vy) / (float(w.sum()) + 1e-6)
    # DSI: difference between preferred and closest opposite
    i_pref = int(np.nanargmax(w))
    pref_dir = float(dirs[i_pref])
    null_dir = (pref_dir + 180.0) % 360.0
    # find index with direction closest to opposite
    diffs = np.abs((np.array(dirs) - null_dir + 180.0) % 360.0 - 180.0)
    i_null = int(np.nanargmin(diffs))
    DSI = (w[i_pref] - w[i_null]) / (w[i_pref] + w[i_null] + 1e-6)
    return float(OSI), float(DSI)

def compute_orientation_tuning(dff: np.ndarray, stim_events: Optional[List[dict]], fps: float, window_s: float = 1.0) -> Dict:
    """
    Event-based orientation tuning. If stim_events is None/empty, returns a minimal summary with NaNs.
    Output:
      {
        "summary": {"OSI":..., "DSI":..., "n_dirs":..., "n_cells":...},
        "dirs": [deg,...],
        "population_mean": [mean_resp_per_dir,...]
      }
    """
    if not stim_events:
        return {
            "summary": {"OSI": float("nan"), "DSI": float("nan"), "n_dirs": 0, "n_cells": int(dff.shape[0] if dff is not None and dff.ndim==2 else 0)},
            "dirs": [],
            "population_mean": []
        }
    by_dir = _trial_responses(dff, stim_events, fps, window_s)
    dirs, pop = _population_tuning(by_dir if by_dir is not None else {})
    OSI, DSI = _osi_dsi(dirs, pop)
    return {
        "summary": {"OSI": OSI, "DSI": DSI, "n_dirs": len(dirs), "n_cells": int(dff.shape[0])},
        "dirs": dirs,
        "population_mean": pop.tolist(),
    }

def pseudo_tuning(dff: np.ndarray, fps: float, window_s: float = 1.0, n_dirs: int = 8) -> Dict:
    """
    Pseudo orientation tuning without events: split time into blocks and treat as pseudo-directions.
    """
    if dff is None or dff.ndim != 2 or dff.size == 0:
        return {
            "summary": {"OSI": float("nan"), "DSI": float("nan"), "n_dirs": 0, "n_cells": 0},
            "dirs": [],
            "population_mean": []
        }
    N, T = dff.shape
    blocks = max(1, n_dirs)
    block_len = T // blocks
    if block_len < 1:
        blocks = T
        block_len = 1
    dirs = np.linspace(0, 180, blocks, endpoint=False)  # orientations (not directions)
    pop = []
    for i in range(blocks):
        seg = dff[:, i*block_len:(i+1)*block_len]
        per_cell = np.nanmean(seg, axis=1) if seg.size > 0 else np.full((N,), np.nan)
        pop.append(float(np.nanmean(per_cell)))
    pop = np.array(pop, dtype=float)
    OSI, DSI = _osi_dsi(list(map(float, dirs)), pop)
    return {
        "summary": {"OSI": OSI, "DSI": DSI, "n_dirs": int(blocks), "n_cells": int(N)},
        "dirs": list(map(float, dirs)),
        "population_mean": pop.tolist(),
    }

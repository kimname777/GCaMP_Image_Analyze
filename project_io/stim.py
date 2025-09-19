from __future__ import annotations

from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


def _infer_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Heuristically pick a time column and a value/TTL column.

    Returns
    -------
    (time_col, value_col)
    """
    cols = [c.strip().lower() for c in df.columns]
    # candidate for time
    t_idx = None
    for i, c in enumerate(cols):
        if "time" in c or "sec" in c or c in ("t", "t(s)", "[sec]", "# time (sec)"):
            t_idx = i
            break
    if t_idx is None:
        t_idx = 0
    # candidate for value
    v_idx = 1 if df.shape[1] >= 2 else 0
    for i, c in enumerate(cols):
        if i == t_idx:
            continue
        if any(k in c for k in ("intensity", "mean", "value", "ttl", "stim")):
            v_idx = i
            break
    return df.columns[t_idx], df.columns[v_idx]


def _ttl_to_events(
    time_s: np.ndarray,
    value: np.ndarray,
    min_dur_s: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Convert a continuous TTL-like trace into onset/offset events using a robust threshold.
    """
    time_s = np.asarray(time_s, float)
    value = np.asarray(value, float)

    # Robust threshold at midpoint of [P5, P95]
    v0, v1 = np.nanpercentile(value, [5, 95])
    thr = (v0 + v1) / 2.0
    high = value > thr

    # Edge detection on binary signal
    d = np.diff(high.astype(int), prepend=int(high[0]))
    on_idx = np.where(d == 1)[0]
    off_idx = np.where(d == -1)[0]
    if high[0]:
        on_idx = np.insert(on_idx, 0, 0)
    if len(off_idx) < len(on_idx):
        off_idx = np.append(off_idx, len(high) - 1)

    # Minimum duration ~ 2*dt by default (at least > 0)
    if min_dur_s is None:
        dt = float(np.median(np.diff(time_s)))
        min_dur_s = max(2 * dt, 0.02)

    events: List[Dict[str, float]] = []
    for i, j in zip(on_idx, off_idx):
        t0 = float(time_s[i])
        t1 = float(time_s[min(j, len(time_s) - 1)])
        if (t1 - t0) >= float(min_dur_s):
            # Direction unknown in TTL → set None
            events.append({"onset": t0, "offset": t1, "direction": None})
    return events


def load_stim_csv_with_meta(path: Union[str, Path]) -> Tuple[List[Dict[str, float]], Dict[str, float | str | None]]:
    """
    Load stimulus CSV and return (events, meta).

    Supported formats
    -----------------
    A) Explicit event list with columns:
       onset*, offset* [, direction*]
    B) TTL-like continuous trace:
       [time(sec) column], [value/intensity/ttl column] → auto eventization.

    Returned events
    ---------------
    List of dicts with keys:
      'onset' (seconds), 'offset' (seconds), 'direction' (deg or None)
    """
    df = pd.read_csv(path)
    cols_low = [c.strip().lower() for c in df.columns]

    # A) Event list: must have onset & offset
    if any("onset" in c for c in cols_low) and any("offset" in c for c in cols_low):
        t_on = next(c for c in df.columns if "onset" in c.lower())
        t_off = next(c for c in df.columns if "offset" in c.lower())
        dir_col = next((c for c in df.columns if ("direction" in c.lower()) or ("deg" in c.lower())), None)
        events: List[Dict[str, float]] = []
        for _, r in df.iterrows():
            ev = {
                "onset": float(r[t_on]),
                "offset": float(r[t_off]),
                "direction": float(r[dir_col]) if (dir_col is not None and pd.notna(r[dir_col])) else None,
            }
            events.append(ev)
        return events, {"mode": "events", "sample_hz": None}

    # B) TTL-like continuous signal
    tcol, vcol = _infer_cols(df)
    events = _ttl_to_events(df[tcol].values, df[vcol].values)
    # Meta: sampling rate from median dt
    dt = float(np.median(np.diff(df[tcol].values)))
    fs = 1.0 / dt if dt > 0 else np.nan
    return events, {"mode": "ttl", "thr_mode": "mid(P5,P95)", "sample_hz": fs}


def load_stim_csv(path: Union[str, Path]) -> List[Dict[str, float]]:
    """
    Convenience wrapper used by the pipeline:
    return **only** the event list (no meta), with keys 'onset'/'offset'/'direction'.
    """
    events, _ = load_stim_csv_with_meta(path)
    return events

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import numpy as np


@dataclass
class AppState:
    """
    Central in-memory state shared across the pipeline and GUI.

    Shapes
    ------
    raw_stack, reg_stack : (T, Y, X)
    roi_masks, neuropil_masks/np_masks : (N, Y, X) boolean
    F_raw, F_np, F0, dff, spikes : (N, T)
    """
    # Raw & registered movies
    raw_stack: Optional[np.ndarray] = None  # (T, Y, X)
    reg_stack: Optional[np.ndarray] = None  # (T, Y, X)

    # Masks & signals
    roi_masks: Optional[np.ndarray] = None        # (N, Y, X) bool
    neuropil_masks: Optional[np.ndarray] = None   # (N, Y, X) bool
    # For compatibility with existing code paths; either may be used:
    np_masks: Optional[np.ndarray] = None         # alias of neuropil_masks (if needed)

    F_raw: Optional[np.ndarray] = None  # (N, T)
    F_np: Optional[np.ndarray] = None   # (N, T)
    F0: Optional[np.ndarray] = None     # (N, T)
    dff: Optional[np.ndarray] = None    # (N, T)
    spikes: Optional[np.ndarray] = None # (N, T)

    # Analysis summaries / QC
    tuning_summary: Dict[str, Any] = field(default_factory=dict)
    qc_metrics: Dict[str, Any] = field(default_factory=dict)
    kinetics_summary: Dict[str, Any] = field(default_factory=dict)

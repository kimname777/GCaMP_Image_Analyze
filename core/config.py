from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from pathlib import Path
import copy
import yaml


def _shallow_merge(dst: dict, src: dict) -> dict:
    """
    Shallow-merge dicts: update top-level keys only. If both sides have a dict,
    perform a one-level update (no deep recursion for simplicity/robustness).
    """
    out = copy.deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


@dataclass
class AppConfig:
    """
    Application configuration shared by the pipeline & GUI.

    Notes
    -----
    * Keep defaults minimal and robust.
    * Use shallow per-section merges when loading from YAML.
    """
    fps: float = 30.0

    preprocess: Dict[str, Any] = field(
        default_factory=lambda: {
            "motion": {"backend": "identity", "block": 64, "overlap": 16, "max_shift": 15},
            "neuropil": {"r": 0.8, "inner": 2, "outer": 6},
            "bleaching": {"method": "percentile", "p": 10, "window_s": 90},
        }
    )

    segmentation: Dict[str, Any] = field(
        default_factory=lambda: {
            "backend": "threshold",  # or "suite2p", "cnmf"
            "diameter": 14,
            "tau": 1.0,              # optional param for some backends
        }
    )

    deconvolution: Dict[str, Any] = field(
        default_factory=lambda: {
            "backend": "simple_ar1",  # or "oasis", "mlspike", "naive"
            "ar": 1,
            "gamma": 0.95,
        }
    )

    @staticmethod
    def load_default() -> "AppConfig":
        """Return a fresh config instance with default values."""
        return AppConfig()

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> "AppConfig":
        """
        Load config from YAML and shallow-merge with defaults.
        Unknown top-level keys are set as-is on the dataclass if they exist.
        """
        path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        base = AppConfig.load_default()
        for k, v in data.items():
            if hasattr(base, k):
                current = getattr(base, k)
                if isinstance(current, dict) and isinstance(v, dict):
                    setattr(base, k, _shallow_merge(current, v))
                else:
                    setattr(base, k, v)
        return base

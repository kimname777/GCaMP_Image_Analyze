from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Union, Iterable
import io
import zipfile
import yaml
import numpy as np

from core.state import AppState
from core.config import AppConfig


PROJECT_VERSION = "0.3"


def _npz_bytes(obj: Dict[str, Any]) -> bytes:
    """
    Serialize a dict of arrays to NPZ bytes using an in-memory buffer.
    """
    bio = io.BytesIO()
    np.savez_compressed(bio, **obj)
    return bio.getvalue()


def _npz_read(data: bytes) -> Dict[str, Any]:
    """
    Read NPZ bytes and return a dict of arrays.
    """
    bio = io.BytesIO(data)
    with np.load(bio, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _maybe_write_npz(z: zipfile.ZipFile, name: str, arr: Any) -> None:
    if arr is not None:
        z.writestr(name, _npz_bytes({name.split(".")[0]: arr}))


def save_project(
    path: Union[str, Path],
    state: AppState,
    cfg: AppConfig,
    include_raw: bool = False,
) -> None:
    """
    Save a lightweight project bundle as a single .zip:

    Contents
    --------
    project.yaml           : metadata (version/fps/shapes)
    raw_stack.npz          : optional raw movie (large; disabled by default)
    reg_stack.npz          : registered movie (if available)
    roi_masks.npz          : (N, Y, X) ROI masks
    np_masks.npz           : (N, Y, X) neuropil masks (if available)
    dff.npz, spikes.npz    : (N, T) traces
    """
    path = Path(path)
    meta = {
        "version": PROJECT_VERSION,
        "fps": float(cfg.fps),
        "shapes": {
            "raw_stack": list(state.raw_stack.shape) if (include_raw and state.raw_stack is not None) else None,
            "reg_stack": list(state.reg_stack.shape) if state.reg_stack is not None else None,
            "roi_masks": list(state.roi_masks.shape) if state.roi_masks is not None else None,
            "np_masks": list(state.np_masks.shape) if getattr(state, "np_masks", None) is not None else None,
            "dff": list(state.dff.shape) if state.dff is not None else None,
            "spikes": list(state.spikes.shape) if state.spikes is not None else None,
        },
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.yaml", yaml.safe_dump(meta, sort_keys=False))
        if include_raw and state.raw_stack is not None:
            _maybe_write_npz(z, "raw_stack.npz", state.raw_stack)
        _maybe_write_npz(z, "reg_stack.npz", state.reg_stack)
        _maybe_write_npz(z, "roi_masks.npz", state.roi_masks)
        _maybe_write_npz(z, "np_masks.npz", getattr(state, "np_masks", None))
        _maybe_write_npz(z, "dff.npz", state.dff)
        _maybe_write_npz(z, "spikes.npz", state.spikes)


def load_project(path: Union[str, Path]) -> Tuple[AppState, AppConfig]:
    """
    Load a project bundle saved by `save_project`.

    Returns
    -------
    state : AppState
    cfg   : AppConfig    (FPS restored from metadata)
    """
    path = Path(path)
    state = AppState()
    cfg = AppConfig.load_default()

    with zipfile.ZipFile(path, "r") as z:
        meta = yaml.safe_load(z.read("project.yaml"))
        if isinstance(meta, dict):
            cfg.fps = float(meta.get("fps", cfg.fps))

        def _maybe_read(name: str, key: str) -> None:
            if name in z.namelist():
                arrs = _npz_read(z.read(name))
                if key in arrs:
                    setattr(state, key, arrs[key])

        _maybe_read("raw_stack.npz", "raw_stack")
        _maybe_read("reg_stack.npz", "reg_stack")
        _maybe_read("roi_masks.npz", "roi_masks")
        _maybe_read("np_masks.npz", "np_masks")
        _maybe_read("dff.npz", "dff")
        _maybe_read("spikes.npz", "spikes")

    return state, cfg

from __future__ import annotations

"""
Helpers to run CaImAn in an **external** process/environment and load its results.

Use-cases
---------
- You ship a separate script (e.g. `caiman_cnmf_export.py`) that saves masks/traces.
- You want to run inside a dedicated conda env without importing CaImAn here.
"""

from pathlib import Path
from typing import Tuple
import subprocess
import json
import numpy as np


def run_caiman_external(
    input_path: str,
    fps: float,
    out_npz: str,
    env_name: str = "caiman",
    script_path: str = "caiman_cnmf_export.py",
) -> tuple[bool, str]:
    """
    Invoke an external CaImAn script inside a given conda environment.

    Parameters
    ----------
    input_path : str
        Path to npy/tif/avi/mp4 file (whatever your external script accepts).
    fps : float
        Frame rate to pass to the script.
    out_npz : str
        Output NPZ path created by the script (must include 'masks', optionally 'C','S','dims').
    env_name : str
        Conda environment name.
    script_path : str
        Path to the external Python script.

    Returns
    -------
    (ok, message)
        ok=True on success, message contains stdout/diagnostics on failure.
    """
    ip = str(Path(input_path).absolute())
    op = str(Path(out_npz).absolute())
    sp = str(Path(script_path).absolute())

    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        sp,
        "--input",
        ip,
        "--fps",
        str(float(fps)),
        "--out",
        op,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if proc.returncode != 0:
            return False, f"CaImAn external failed (code {proc.returncode}):\n{proc.stdout}"
        return True, proc.stdout
    except Exception as e:
        return False, f"Exception: {e}"


def load_caiman_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Load results produced by an external CaImAn script.

    Returns
    -------
    masks : (K, H, W) bool
    C     : (K, T) float32  (dFF or raw)
    S     : (K, T) float32  (deconvolved)
    dims  : (H, W)
    """
    npz = np.load(str(npz_path))
    masks = npz["masks"].astype(bool)
    C = npz["C"].astype(np.float32) if "C" in npz.files else np.zeros((masks.shape[0], 0), np.float32)
    S = npz["S"].astype(np.float32) if "S" in npz.files else np.zeros((masks.shape[0], 0), np.float32)
    if "dims" in npz.files:
        dims = tuple(int(x) for x in npz["dims"].tolist())
    else:
        dims = (int(masks.shape[1]), int(masks.shape[2]))
    return masks, C, S, dims

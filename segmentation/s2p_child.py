from __future__ import annotations

"""
Run Suite2p in a separate Python process and export ROI masks as NPZ.

This module is intentionally minimal—it's invoked with:
    python -m segmentation.s2p_child <in_tif> <out_npz> <diameter> <tau>

Environment
-----------
We limit BLAS/numba threads for reproducibility and to avoid oversubscription.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Be gentle with threads when called from a GUI
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "safe")


def run(in_tif: str, out_npz: str, diameter: int = 14, tau: float = 1.0) -> None:
    import suite2p  # noqa: F401
    from suite2p.run_s2p import run_s2p

    in_tif = str(Path(in_tif))
    out_npz = str(Path(out_npz))

    ops = suite2p.default_ops()
    ops.update(
        {
            "nplanes": 1,
            "nchannels": 1,
            "data_path": [str(Path(in_tif).parent)],
            "tiff_list": [[str(in_tif)]],
            "save_path0": str(Path(in_tif).parent),
            "do_registration": 0,  # we already motion-correct
            "roidetect": True,
            "diameter": int(diameter),
            "tau": float(tau),
            "functional_chan": 1,
            "fs": 30.0,  # informational
        }
    )
    run_s2p(ops)

    plane = Path(in_tif).parent / "suite2p" / "plane0"
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    iscell = np.load(plane / "iscell.npy")
    # If meanImg missing, compute from input
    try:
        import tifffile as tiff
        mean_img = np.load(plane / "meanImg.npy") if (plane / "meanImg.npy").exists() else tiff.imread(in_tif).mean(axis=0)
    except Exception:
        mean_img = None

    if mean_img is None:
        # last resort—get dims from first stat entry
        if len(stat) > 0 and "Ly" in stat[0] and "Lx" in stat[0]:
            Y, X = int(stat[0]["Ly"]), int(stat[0]["Lx"])
        else:
            raise RuntimeError("Cannot determine image size for Suite2p masks.")
    else:
        Y, X = int(mean_img.shape[0]), int(mean_img.shape[1])

    masks = []
    for d, flag in zip(stat, iscell[:, 0]):
        if int(flag) != 1:
            continue
        ypix = np.asarray(d["ypix"]).astype(int)
        xpix = np.asarray(d["xpix"]).astype(int)
        m = np.zeros((Y, X), dtype=bool)
        m[ypix, xpix] = True
        masks.append(m)

    ms = np.stack(masks, axis=0) if masks else np.zeros((0, Y, X), bool)
    np.savez_compressed(out_npz, masks=ms)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python -m segmentation.s2p_child <in_tif> <out_npz> <diameter> <tau>", file=sys.stderr)
        sys.exit(2)
    _, in_tif, out_npz, diameter, tau = sys.argv
    run(in_tif, out_npz, int(diameter), float(tau))

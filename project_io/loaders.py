from __future__ import annotations

import os
import numpy as np
from typing import Tuple, Dict, Any


def _to_tyxc(arr: np.ndarray) -> np.ndarray:
    """
    Normalize various array layouts to (T, Y, X) grayscale.

    Accepted inputs
    ---------------
    (Y, X)            -> (1, Y, X)
    (T, Y, X)         -> (T, Y, X)
    (Y, X, C=3|4)     -> RGB[A] to Gray -> (1, Y, X)
    (T, Y, X, C=3|4)  -> RGB[A] to Gray -> (T, Y, X)

    Notes
    -----
    * RGB→Gray uses standard luminosity weights.
    * dtype is not changed here (done by callers).
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a[None, ...]
    if a.ndim == 3:
        # Could be (T, Y, X) OR (Y, X, C)
        if a.shape[-1] in (3, 4):  # (Y, X, C)
            rgb = a[..., :3]
            g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
            return g[None, ...]
        # Assume already (T, Y, X)
        return a
    if a.ndim == 4 and a.shape[-1] in (3, 4):
        rgb = a[..., :3]
        g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        return g
    raise ValueError(f"Unsupported array shape: {a.shape}")


# ------------------------- TIFF ------------------------- #
def _read_tiff(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read multi-page TIFF as (T, Y, X) and try to recover FPS from ImageJ tags.

    Strategy
    --------
    1) Prefer `tifffile` (fast, robust).
    2) Fallback to `imageio.v3`.
    """
    meta: Dict[str, Any] = {"fps": None}
    arr = None

    # 1) tifffile first
    try:
        import tifffile as tiff  # type: ignore
        with tiff.TiffFile(path) as tf:
            arr = tf.asarray()
            # Attempt ImageJ metadata FPS inference
            fps = None
            ij = getattr(tf, "imagej_metadata", None) or {}
            # common keys: fps, FrameRate, frame_rate, FrameRateHz
            for k in ("fps", "FrameRate", "frame_rate", "FrameRateHz"):
                v = ij.get(k) if isinstance(ij, dict) else None
                if v is not None:
                    try:
                        fps = float(v)
                        break
                    except Exception:
                        pass
            if fps is None:
                # sometimes interval is provided instead
                for k in ("finterval", "Frame Interval", "FrameInterval"):
                    v = ij.get(k) if isinstance(ij, dict) else None
                    if v is not None:
                        try:
                            val = float(v)
                            fps = 1.0 / val if val > 0 else None
                            break
                        except Exception:
                            pass
            meta["fps"] = fps
    except Exception:
        arr = None

    # 2) imageio.v3 fallback
    if arr is None:
        try:
            import imageio.v3 as iio  # type: ignore
            arr = iio.imread(path)  # (T, Y, X) or (T, Y, X, C)
            try:
                imeta = iio.immeta(path)
                meta["fps"] = float(imeta.get("fps")) if "fps" in imeta else None
            except Exception:
                pass
        except Exception as e:  # pragma: no cover - I/O environment dependent
            raise RuntimeError(f"TIFF read failed: {e}")

    arr = _to_tyxc(arr).astype(np.float32, copy=False)
    return arr, meta


# -------------------------- NPY ------------------------- #
def _read_npy(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a NumPy array and normalize to (T, Y, X) float32."""
    meta: Dict[str, Any] = {"fps": None}
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        raise RuntimeError(f"NPY read failed: {e}")
    arr = _to_tyxc(arr).astype(np.float32, copy=False)
    return arr, meta


# ------------------------- Video ------------------------ #
def _read_video_gray(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read .avi/.mp4 into (T, Y, X) float32 grayscale.

    Strategy
    --------
    1) Prefer `imageio.v3` (ffmpeg-backed).
    2) Fallback to OpenCV if available.
    """
    meta: Dict[str, Any] = {"fps": None}
    stack = None

    # 1) imageio.v3
    try:
        import imageio.v3 as iio  # type: ignore
        arr = iio.imread(path)  # (T, H, W) or (T, H, W, C)
        if arr.ndim == 4 and arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            arr = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        elif arr.ndim == 3:
            arr = arr.astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported video array shape: {arr.shape}")
        stack = arr
        try:
            imeta = iio.immeta(path)
            meta["fps"] = float(imeta.get("fps")) if "fps" in imeta else None
        except Exception:
            pass
    except Exception:
        stack = None

    # 2) OpenCV fallback
    if stack is None:
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("cv2.VideoCapture failed to open.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    gray = frame.astype(np.float32)
                frames.append(gray)
            cap.release()
            if not frames:
                raise RuntimeError("No frames read from video.")
            stack = np.stack(frames, axis=0)
            meta["fps"] = float(fps) if fps and fps > 0 else None
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Video read failed: {e}")

    stack = _to_tyxc(stack).astype(np.float32, copy=False)
    return stack, meta


# ------------------------ Public API -------------------- #
def load_stack(path: str | os.PathLike) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a stack as (T, Y, X) float32 with optional FPS metadata.

    Supported formats
    -----------------
    .tif/.tiff : multi-page TIFF (tifffile → imageio fallback)
    .npy       : NumPy array
    .avi/.mp4  : Video (imageio → OpenCV fallback)

    Returns
    -------
    stack : (T, Y, X) float32
    meta  : {'fps': float|None}
    """
    p = str(path)
    ext = os.path.splitext(p)[1].lower()
    if ext in (".tif", ".tiff"):
        return _read_tiff(p)
    if ext == ".npy":
        return _read_npy(p)
    if ext in (".avi", ".mp4"):
        return _read_video_gray(p)
    raise ValueError(f"Unsupported file extension: {ext}")


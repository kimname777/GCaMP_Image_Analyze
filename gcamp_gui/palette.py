from __future__ import annotations

from typing import Tuple, List, Optional
from PySide6.QtGui import QColor
import colorsys

# Matplotlib is an optional heavy import; import lazily when needed.
_MPL_AVAILABLE = True
try:
    import matplotlib.cm as _cm  # type: ignore
except Exception:
    _MPL_AVAILABLE = False

# Shared palette used for ROI preview and trace plotting.
# Keep colors as 0-255 RGB tuples for Qt, and provide helpers for matplotlib and Qt.
PALETTE: List[Tuple[int, int, int]] = [
    (244, 67, 54), (33, 150, 243), (76, 175, 80), (255, 193, 7),
    (156, 39, 176), (0, 188, 212), (255, 87, 34), (121, 85, 72),
    (63, 81, 181), (139, 195, 74), (3, 169, 244), (255, 152, 0),
    (233, 30, 99), (0, 150, 136), (205, 220, 57), (158, 158, 158),
]


def qcolor(index: int, alpha: int = 255, total: Optional[int] = None, cmap_name: str = "tab20") -> QColor:
    """Return a QColor for the given palette index.

    If ``total`` is provided and larger than the discrete PALETTE, a matplotlib
    colormap (default: ``tab20``) is used to generate a distinct color scaled
    across [0, total-1]. Falls back to the discrete PALETTE if matplotlib is
    unavailable or ``total`` is small.
    """
    # If total is not provided or small, use the discrete PALETTE.
    if total is None or total <= len(PALETTE):
        r, g, b = PALETTE[index % len(PALETTE)]
    else:
        # Prefer a simple HSV wheel to guarantee distinct colors for arbitrary N.
        # Use a fixed saturation/value for good visibility.
        h = float(index) / float(total)
        s = 0.65
        v = 0.92
        rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
        r, g, b = int(rr * 255), int(gg * 255), int(bb * 255)
    return QColor(r, g, b, alpha)


def mpl_color(index: int, total: Optional[int] = None, cmap_name: str = "tab20") -> Tuple[float, float, float]:
    """Return an (r,g,b) tuple normalized to [0,1] for matplotlib.

    If ``total`` is provided and larger than the discrete palette, a matplotlib
    colormap is used to generate smoothly varying colors.
    """
    if total is None or total <= len(PALETTE):
        r, g, b = PALETTE[index % len(PALETTE)]
        return (r / 255.0, g / 255.0, b / 255.0)
    # Use HSV wheel for distinct matplotlib colors when total is provided.
    h = float(index) / float(total)
    s = 0.65
    v = 0.92
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
    return (float(rr), float(gg), float(bb))

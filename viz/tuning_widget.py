from __future__ import annotations
"""
TuningWidget
------------
Summarize direction/orientation tuning metrics in a tiny panel.

Expects a dict like:
{
  "summary": {"OSI": 0.45, "DSI": 0.32, "n_cells": 57},
  "n_dirs": 8,
  ...
}
"""

from typing import Union
import numpy as np
from PySide6.QtWidgets import QWidget, QFormLayout, QLabel


class TuningWidget(QWidget):
    """Minimal readout of OSI/DSI and counts."""
    def __init__(self):
        super().__init__()
        f = QFormLayout(self)
        self.l_osi, self.l_dsi, self.l_dirs, self.l_cells = QLabel("-"), QLabel("-"), QLabel("-"), QLabel("-")
        f.addRow("OSI:", self.l_osi)
        f.addRow("DSI:", self.l_dsi)
        f.addRow("# dirs:", self.l_dirs)
        f.addRow("# cells:", self.l_cells)
        self.debug_fn = None

    def update_results(self, res: Union[dict, None]):
        """Update labels from a result dictionary (see module docstring)."""
        if not res:
            self.l_osi.setText("n/a (no stimulus)")
            self.l_dsi.setText("n/a (no stimulus)")
            self.l_dirs.setText("0")
            self.l_cells.setText("0")
            return
        summary = res.get("summary", {})
        osi = summary.get("OSI", float("nan"))
        dsi = summary.get("DSI", float("nan"))
        self.l_osi.setText("n/a (no stimulus)" if not np.isfinite(osi) else f"{osi:.3f}")
        self.l_dsi.setText("n/a (no stimulus)" if not np.isfinite(dsi) else f"{dsi:.3f}")
        self.l_dirs.setText(str(res.get("n_dirs", 0)))
        self.l_cells.setText(str(summary.get("n_cells", 0)))

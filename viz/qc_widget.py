from __future__ import annotations
"""
QcWidget
--------
Tiny widget to display quick quality-control metrics (e.g., split-half reliability).
"""

from typing import Union
import numpy as np
from PySide6.QtWidgets import QWidget, QFormLayout, QLabel


class QcWidget(QWidget):
    """Minimal QC panel; extend with more metrics as needed."""
    def __init__(self):
        super().__init__()
        f = QFormLayout(self)
        self.l_r = QLabel("-")
        f.addRow("Split-half reliability:", self.l_r)
        self.debug_fn = None  # Optional callable for debugging

    def update_qc(self, qc: Union[dict, None]):
        """Update reliability value from a dict like {'reliability': 0.87}."""
        if not qc:
            self.l_r.setText("n/a")
            return
        r = qc.get("reliability", float("nan"))
        self.l_r.setText("n/a" if not np.isfinite(r) else f"{r:.3f}")

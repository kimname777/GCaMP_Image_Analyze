from __future__ import annotations
"""
TracesWidget
------------
Two-pane widget: a checkable ROI list on the left and a Matplotlib plot on the right.

Features
* Offset-stacked ΔF/F traces for checked ROIs (selection highlights in bold)
* Optional spike markers (drawn as vertical bars when `spikes` is provided)
* Signals:
    - selectionChanged(list[int])
    - requestRectSelect()  (ask parent to start rectangle selection in the image view)
"""

from typing import Optional, List
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QSplitter, QSizePolicy
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as Toolbar
from matplotlib.figure import Figure


class TracesWidget(QWidget):
    selectionChanged = Signal(list)   # list of checked ROI indices
    requestRectSelect = Signal()      # start rectangle selection in the image view

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._dff: Optional[np.ndarray] = None   # (N, T)
        self._spk: Optional[np.ndarray] = None   # (N, T)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # Left: ROI list + controls
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)
        self.list = QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.setMinimumWidth(180)
        lv.addWidget(QLabel("Cells"))
        lv.addWidget(self.list, 1)

        btn_row = QHBoxLayout()
        self.btn_all = QPushButton("All")
        self.btn_none = QPushButton("None")
        self.btn_select_rect = QPushButton("Select in View")
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        btn_row.addWidget(self.btn_select_rect)
        btn_row.addStretch(1)
        lv.addLayout(btn_row)

        splitter.addWidget(left)

        # Right: large Matplotlib canvas
        fig = Figure(figsize=(7.5, 4.8), tight_layout=True)
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = fig.add_subplot(111)
        self.toolbar = Toolbar(self.canvas, self)

        plot_box = QVBoxLayout()
        plot_box.setContentsMargins(0, 0, 0, 0)
        plot_box.addWidget(self.toolbar, 0)
        plot_box.addWidget(self.canvas, 1)
        plot_wrap = QWidget()
        plot_wrap.setLayout(plot_box)

        splitter.addWidget(plot_wrap)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Signals
        self.btn_all.clicked.connect(self.select_all)
        self.btn_none.clicked.connect(self.select_none)
        self.btn_select_rect.clicked.connect(self.requestRectSelect.emit)
        self.list.itemChanged.connect(self._emit_selection)
        self.list.itemSelectionChanged.connect(self.apply_selection_to_plot)

    # ---------------- API ----------------
    def reset(self):
        """Clear list and plot."""
        self._dff = None
        self._spk = None
        self.list.clear()
        self.ax.clear()
        self.canvas.draw_idle()

    def set_num_rois(self, n: int):
        """Rebuild the list to contain N checkable rows."""
        self.list.blockSignals(True)
        self.list.clear()
        for i in range(int(n)):
            it = QListWidgetItem(f"cell {i}")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self._emit_selection()

    def update_traces(self, dff: Optional[np.ndarray], spikes: Optional[np.ndarray]):
        """Refresh traces; rebuild the list if N changed."""
        self._dff = dff
        self._spk = spikes
        n = 0 if dff is None else dff.shape[0]
        if self.list.count() != n:
            self.set_num_rois(n)
        self.apply_selection_to_plot()

    def current_checked_indices(self) -> List[int]:
        """Return indices of all checked rows."""
        return self._current_checked_indices()

    def check_only(self, indices: List[int]):
        """Check a subset and uncheck everything else."""
        want = set(int(i) for i in (indices or []))
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if i in want else Qt.Unchecked)
        self.list.blockSignals(False)
        self._emit_selection()
        self.apply_selection_to_plot()

    # ---------------- plot ----------------
    def apply_selection_to_plot(self):
        """Draw offset-stacked ΔF/F for checked ROIs (with optional spike markers)."""
        self.ax.clear()
        if self._dff is None or self._dff.size == 0:
            self.canvas.draw_idle()
            return

        checked = self._current_checked_indices()
        if not checked:
            self.ax.set_title("No cells selected")
            self.canvas.draw_idle()
            return

        selected_rows = [idx.row() for idx in self.list.selectedIndexes()]
        t = np.arange(self._dff.shape[1])
        step = 1.2 * (np.nanstd(self._dff[checked, :]) + 1e-6)
        offset = 0.0

        for i in checked:
            y = self._dff[i] + offset
            lw = 2.0 if i in selected_rows else 1.0
            a = 1.0 if i in selected_rows else 0.85
            self.ax.plot(t, y, linewidth=lw, alpha=a, label=f"cell {i}")
            if self._spk is not None and self._spk.shape == self._dff.shape:
                s = (self._spk[i] > 0).astype(bool)
                if s.any():
                    self.ax.plot(t[s], y[s], marker="|", linestyle="None", markersize=8, alpha=0.9)
            offset += step

        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("ΔF/F (offset by cell)")
        if len(checked) <= 8:
            self.ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
        self.canvas.draw_idle()

    # ---------------- actions ----------------
    def select_all(self):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked)
        self.list.blockSignals(False)
        self._emit_selection()
        self.apply_selection_to_plot()

    def select_none(self):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Unchecked)
        self.list.blockSignals(False)
        self._emit_selection()
        self.apply_selection_to_plot()

    # ---------------- helpers ----------------
    def _current_checked_indices(self) -> List[int]:
        idxs = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                idxs.append(i)
        return idxs

    def _emit_selection(self):
        self.selectionChanged.emit(self._current_checked_indices())

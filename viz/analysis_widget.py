from __future__ import annotations
"""
AnalysisWidget
--------------
Interactive analysis views for ΔF/F and (optionally) spike trains.

Features
* Connectivity: dendrogram, triangular correlation map, pairwise correlation
* Distribution distance: EMD (Wasserstein) matrix (fallback to 1 - corr)
* Spiking: raster plot and van Rossum distance matrix
* Pairwise cross-correlation for two selected neurons
* Fixed colorbar axes (cax) to prevent shrinking when replotting
* Axis tick labels display actual ROI indices (+ label_base)

SciPy is optional. If missing, the affected views degrade gracefully.
"""

from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QSizePolicy, QComboBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as Toolbar
from matplotlib.figure import Figure

# Optional SciPy: clustering + EMD
try:
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
    from scipy.stats import wasserstein_distance
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


class AnalysisWidget(QWidget):
    """
    A dockable QWidget that renders multiple analysis plots using Matplotlib.

    Public API
    ----------
    reset()
    set_data(dff=None, spikes=None, masks=None)
    set_selected_indices(idx_list)
    set_label_base(base)
    refresh()
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.dff: Optional[np.ndarray] = None
        self.spikes: Optional[np.ndarray] = None
        self.masks: Optional[np.ndarray] = None
        self.selected_indices: List[int] = []
        self.label_base: int = 0  # 0 or 1

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Top bar: pair selector for cross-correlation ───────────────────────
        pair_bar = QHBoxLayout()
        pair_bar.addWidget(QLabel("Cross-corr pair:"))
        self.cmb_i = QComboBox()
        self.cmb_j = QComboBox()
        for c in (self.cmb_i, self.cmb_j):
            c.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            c.setMinimumWidth(90)
        self.btn_xcorr_pair = QPushButton("Plot XCorr")
        self.btn_xcorr_pair.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        pair_bar.addWidget(self.cmb_i)
        pair_bar.addWidget(QLabel("×"))
        pair_bar.addWidget(self.cmb_j)
        pair_bar.addWidget(self.btn_xcorr_pair)
        pair_bar.addStretch(1)
        root.addLayout(pair_bar)

        # ── Middle: grouped actions ────────────────────────────────────────────
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        g_conn = QGroupBox("Connectivity")
        gl_conn = QHBoxLayout(g_conn)
        self.btn_dend = QPushButton("Dendrogram")
        self.btn_tri = QPushButton("Triangular Corr")
        self.btn_corr = QPushButton("Pairwise Corr")
        self.btn_emd = QPushButton("EMD Matrix")
        for b in (self.btn_dend, self.btn_tri, self.btn_corr, self.btn_emd):
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            gl_conn.addWidget(b)
        gl_conn.addStretch(1)

        g_spk = QGroupBox("Spiking")
        gl_spk = QHBoxLayout(g_spk)
        self.btn_raster = QPushButton("Spike Raster")
        self.btn_vr = QPushButton("van Rossum Dist")
        self.btn_clear = QPushButton("Clear")
        for b in (self.btn_raster, self.btn_vr, self.btn_clear):
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            gl_spk.addWidget(b)
        gl_spk.addStretch(1)

        grid.addWidget(g_conn, 0, 0)
        grid.addWidget(g_spk, 1, 0)
        root.addLayout(grid)

        # ── Bottom: Matplotlib figure with a fixed colorbar axis ───────────────
        self.fig = Figure(figsize=(8.6, 5.2))
        self.ax = self.fig.add_subplot(111)
        # Manually pin axes to avoid layout jitter
        self.ax.set_position([0.10, 0.12, 0.78, 0.80])
        self.cax = self.fig.add_axes([0.90, 0.15, 0.03, 0.70])  # dedicated colorbar axis
        self.cax.set_visible(False)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(720, 420)
        self.toolbar = Toolbar(self.canvas, self)

        root.addWidget(self.toolbar, 0)

        # Center the canvas horizontally
        center = QHBoxLayout()
        center.addStretch(1)
        center.addWidget(self.canvas, 0, Qt.AlignCenter)
        center.addStretch(1)
        root.addLayout(center, 1)

        # Status line
        self.note = QLabel("Load data and run the pipeline first.")
        root.addWidget(self.note, 0)

        # Connections
        self.btn_xcorr_pair.clicked.connect(self._do_xcorr_pair)
        self.btn_dend.clicked.connect(self._do_dendrogram)
        self.btn_tri.clicked.connect(self._do_triangular_corr)
        self.btn_corr.clicked.connect(self._do_corr)
        self.btn_emd.clicked.connect(self._do_emd_heatmap)
        self.btn_raster.clicked.connect(self._do_raster)
        self.btn_vr.clicked.connect(self._do_van_rossum)
        self.btn_clear.clicked.connect(self._clear)

        self._clear_axes_and_note("Load data and run the pipeline first.")
        self.setStyleSheet(
            "QGroupBox{font-weight:600;margin-top:6px} "
            "QGroupBox::title{subcontrol-origin: margin; left:6px;}"
        )

    # ===== Public API ==========================================================
    def reset(self):
        """Clear data and reset UI state."""
        self.dff = None
        self.spikes = None
        self.masks = None
        self.selected_indices = []
        self._rebuild_pair_combos()
        self._clear_axes_and_note("Reset.")

    def set_data(self, dff: Optional[np.ndarray] = None, spikes: Optional[np.ndarray] = None,
                 masks: Optional[np.ndarray] = None):
        """Update ΔF/F, spikes, and masks (if any)."""
        self.dff = dff
        self.spikes = spikes
        self.masks = masks
        n = 0 if dff is None else int(dff.shape[0])
        self.selected_indices = list(range(n))
        self._rebuild_pair_combos()
        self.refresh()

    def set_selected_indices(self, idx_list: List[int]):
        """Set which ROIs are considered 'selected' for analysis."""
        self.selected_indices = list(map(int, idx_list or []))
        self._rebuild_pair_combos()
        self.refresh()

    def set_label_base(self, base: int = 0):
        """Choose whether tick labels start at 0 or 1."""
        self.label_base = int(base)

    def refresh(self):
        """Refresh status based on the currently loaded data."""
        if self.dff is None:
            self._clear_axes_and_note("No ΔF/F available.")
            return
        if not self.selected_indices:
            self._clear_axes_and_note("No cells selected.")
            return
        self.note.setText(f"Ready. Selected cells: {len(self.selected_indices)}")

    # ===== Internal helpers ====================================================
    def _rebuild_pair_combos(self):
        """Repopulate the two comboboxes used to pick a cross-corr pair."""
        for cmb in (self.cmb_i, self.cmb_j):
            cmb.blockSignals(True)
            cmb.clear()
            for gid in self.selected_indices:
                cmb.addItem(f"{gid + self.label_base}", gid)
            cmb.blockSignals(False)
        if len(self.selected_indices) >= 2:
            self.cmb_i.setCurrentIndex(0)
            self.cmb_j.setCurrentIndex(1)

    def _hide_colorbar(self):
        self.cax.cla()
        self.cax.set_visible(False)

    def _show_colorbar(self, mappable):
        self.cax.cla()
        self.cax.set_visible(True)
        self.fig.colorbar(mappable, cax=self.cax)

    def _clear_axes_and_note(self, text: str):
        self.ax.clear()
        self._hide_colorbar()
        self.ax.text(0.5, 0.5, text, ha="center", va="center", transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw_idle()
        self.note.setText(text)

    def _clear(self):
        self._clear_axes_and_note("Cleared.")

    def _sub_by_global(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Return (dff_sub, spikes_sub_or_None, global_ids_array)."""
        assert self.dff is not None
        gids = np.array([i for i in self.selected_indices if 0 <= i < self.dff.shape[0]], dtype=int)
        if gids.size == 0:
            gids = np.arange(self.dff.shape[0], dtype=int)
        dff = self.dff[gids, :]
        spk = None if self.spikes is None else self.spikes[gids, :]
        return dff, spk, gids

    def _apply_tick_labels(self, order: np.ndarray):
        """Apply tick labels corresponding to (possibly reordered) global ROI ids."""
        _, _, gids = self._sub_by_global()
        labels = (gids[order] + self.label_base).astype(int).tolist()
        n = len(labels)
        self.ax.set_xticks(range(n))
        self.ax.set_xticklabels(labels, rotation=90, fontsize=8)
        self.ax.set_yticks(range(n))
        self.ax.set_yticklabels(labels, fontsize=8)

    # ===== Plotters ============================================================
    @Slot()
    def _do_dendrogram(self):
        """Plot a correlation-distance dendrogram (fallback: bar plot proxy)."""
        if self.dff is None or not self.selected_indices:
            self._clear_axes_and_note("No cells selected.")
            return
        X, _, gids = self._sub_by_global()
        if X.shape[0] < 2:
            self._clear_axes_and_note("Need ≥ 2 cells.")
            return

        C = np.corrcoef(X)
        D = 1.0 - C
        tri = D[np.triu_indices_from(D, 1)]
        self.ax.clear()
        self._hide_colorbar()
        if _HAS_SCIPY and tri.size > 0:
            labels = [str(g + self.label_base) for g in gids.tolist()]
            Z = linkage(tri, method="average")
            dendrogram(Z, ax=self.ax, color_threshold=None, labels=labels)
            self.ax.set_title("Correlation-coefficient dendrogram")
            self.ax.set_xlabel("neuron")
            self.ax.set_ylabel("distance")
        else:
            order = np.argsort(np.sum(D, axis=1))
            self.ax.bar(np.arange(len(order)), np.sort(np.sum(D, axis=1)))
            self.ax.set_title("Dendrogram (fallback)")
            self.ax.set_xlabel("neuron")
            self.ax.set_ylabel("distance proxy")
        self.canvas.draw_idle()
        self.note.setText("Dendrogram ready.")

    @Slot()
    def _do_triangular_corr(self):
        """Triangular correlation map, optionally cluster-ordered."""
        if self.dff is None or not self.selected_indices:
            self._clear_axes_and_note("No cells selected.")
            return
        X, _, gids = self._sub_by_global()
        n, _ = X.shape
        if n < 2:
            self._clear_axes_and_note("Need ≥ 2 cells.")
            return

        C = np.corrcoef(X)
        try:
            if _HAS_SCIPY and n >= 3:
                D = 1.0 - C
                Z = linkage(D[np.triu_indices(n, 1)], method="average")
                order = leaves_list(Z)
            else:
                order = np.arange(n)
        except Exception:
            order = np.arange(n)
        C_ord = C[order][:, order]

        ys, xs, vals = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                ys.append(n - 1 - i)
                xs.append(j)
                vals.append(C_ord[i, j])
        ys = np.array(ys)
        xs = np.array(xs)
        vals = np.array(vals)

        self.ax.clear()
        sc = self.ax.scatter(
            xs, ys, c=vals, s=50 + 250 * np.abs(vals),
            cmap="viridis", vmin=-1, vmax=1, marker="s"
        )
        self._show_colorbar(sc)
        self.ax.set_title("Triangular correlation (cluster-ordered)")
        labels = (gids[order] + self.label_base).astype(int).tolist()
        n = len(labels)
        self.ax.set_xticks(range(n))
        self.ax.set_xticklabels(labels, rotation=90, fontsize=8)
        self.ax.set_yticks(range(n))
        self.ax.set_yticklabels(labels[::-1], fontsize=8)
        self.ax.set_xlim(-0.5, n - 0.5)
        self.ax.set_ylim(-0.5, n - 0.5)
        self.canvas.draw_idle()
        self.note.setText(f"{n} cells; marker size ∝ |corr|.")

    @Slot()
    def _do_corr(self):
        """Full pairwise correlation matrix for the current selection."""
        if self.dff is None or not self.selected_indices:
            self._clear_axes_and_note("No cells selected.")
            return
        X, _, _ = self._sub_by_global()
        if X.shape[0] < 2:
            self._clear_axes_and_note("Need ≥ 2 cells.")
            return

        C = np.corrcoef(X)
        self.ax.clear()
        im = self.ax.imshow(C, vmin=-1, vmax=1, origin="lower", aspect="auto", cmap="viridis")
        self._show_colorbar(im)
        self.ax.set_title("Pairwise correlation (selected)")
        self._apply_tick_labels(np.arange(C.shape[0]))
        self.canvas.draw_idle()
        self.note.setText(f"Corr matrix: shape {C.shape}")

    @Slot()
    def _do_emd_heatmap(self):
        """EMD/Wasserstein distance matrix (fallback: 1 − corr distance)."""
        if self.dff is None or not self.selected_indices:
            self._clear_axes_and_note("No cells selected.")
            return
        X, _, _ = self._sub_by_global()
        n, _ = X.shape

        if _HAS_SCIPY:
            bins = 32
            hist = []
            for i in range(n):
                h, edges = np.histogram(X[i], bins=bins, density=True)
                c = 0.5 * (edges[:-1] + edges[1:])
                h = h / (h.sum() + 1e-12)
                hist.append((c, h))
            D = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                ci, hi = hist[i]
                for j in range(i + 1, n):
                    cj, hj = hist[j]
                    g = np.linspace(min(ci.min(), cj.min()), max(ci.max(), cj.max()), bins)
                    hi2 = np.interp(g, ci, hi)
                    hi2 /= (hi2.sum() + 1e-12)
                    hj2 = np.interp(g, cj, hj)
                    hj2 /= (hj2.sum() + 1e-12)
                    D[i, j] = D[j, i] = wasserstein_distance(g, g, u_weights=hi2, v_weights=hj2)
        else:
            C = np.corrcoef(X)
            D = 1.0 - C
            D[D < 0] = 0.0

        self.ax.clear()
        im = self.ax.imshow(D, origin="lower", aspect="auto", cmap="inferno")
        self._show_colorbar(im)
        self.ax.set_title(
            "Earth Mover's Distance (EMD) matrix" if _HAS_SCIPY else "Distance matrix (1 − corr fallback)"
        )
        self._apply_tick_labels(np.arange(n))
        self.canvas.draw_idle()
        self.note.setText(f"EMD matrix shape {D.shape}")

    @Slot()
    def _do_raster(self):
        """Spike raster; if spikes are missing, threshold ΔF/F as a proxy."""
        if self.spikes is None and self.dff is None:
            self._clear_axes_and_note("No spikes/DFF.")
            return
        _, S, gids = self._sub_by_global()
        if S is None:
            X, _, gids = self._sub_by_global()
            thr = np.quantile(X, 0.95, axis=1, keepdims=True)
            S = (X > thr).astype(np.float32)

        n, T = S.shape
        self.ax.clear()
        self._hide_colorbar()
        for k in range(n):
            t_on = np.flatnonzero(S[k] > 0)
            if t_on.size:
                self.ax.vlines(t_on, k - 0.45, k + 0.45, linewidth=1)
        self.ax.set_xlabel("time (frame)")
        self.ax.set_yticks(range(n))
        self.ax.set_yticklabels((gids + self.label_base).astype(int).tolist())
        self.ax.set_ylabel("neuron #")
        self.ax.set_title("Estimated spike times (raster)")
        self.canvas.draw_idle()
        self.note.setText(f"Raster: {n} neurons.")

    @Slot()
    def _do_van_rossum(self, tau_frames: int = 10):
        """van Rossum distance matrix on spike trains (or thresholded ΔF/F)."""
        if self.spikes is None and self.dff is None:
            self._clear_axes_and_note("No spikes/DFF.")
            return
        _, S, _ = self._sub_by_global()
        if S is None:
            X, _, _ = self._sub_by_global()
            thr = np.quantile(X, 0.95, axis=1, keepdims=True)
            S = (X > thr).astype(np.float32)

        n, T = S.shape
        tau = max(1, int(tau_frames))
        k = np.exp(-np.arange(0, 8 * tau + 1) / float(tau))

        def filt(x):  # causal exponential filter
            return np.convolve(x, k, mode="full")[:T]

        F = np.stack([filt(S[i]) for i in range(n)], axis=0)
        D = np.zeros((n, n), dtype=np.float64)
        scale = 1.0 / float(tau)
        for i in range(n):
            for j in range(i + 1, n):
                d = F[i] - F[j]
                D[i, j] = D[j, i] = np.sqrt(scale * np.sum(d * d))

        self.ax.clear()
        im = self.ax.imshow(D, origin="lower", aspect="auto", cmap="viridis")
        self._show_colorbar(im)
        self.ax.set_title(f"van Rossum distance (τ = {tau} frames)")
        self._apply_tick_labels(np.arange(n))
        self.canvas.draw_idle()
        self.note.setText(f"van Rossum matrix shape {D.shape}")

    @Slot()
    def _do_xcorr_pair(self):
        """Cross-correlation between two chosen neurons."""
        if self.dff is None or len(self.selected_indices) < 2:
            self._clear_axes_and_note("Pick ≥ 2 cells.")
            return
        gi = self.cmb_i.currentData()
        gj = self.cmb_j.currentData()
        if gi is None or gj is None or gi == gj:
            self._clear_axes_and_note("Choose two different cells.")
            return

        X, _, gids = self._sub_by_global()
        try:
            i = int(np.where(gids == gi)[0][0])
            j = int(np.where(gids == gj)[0][0])
        except Exception:
            self._clear_axes_and_note("Pair not in selection.")
            return

        xi = X[i]
        xj = X[j]
        T = xi.size
        xi = (xi - xi.mean()) / (xi.std() + 1e-9)
        xj = (xj - xj.mean()) / (xj.std() + 1e-9)

        lags = np.arange(-T + 1, T)
        cc = np.correlate(xi, xj, mode="full") / T
        L = min(100, T - 1)
        mid = T - 1

        self.ax.clear()
        self._hide_colorbar()
        self.ax.plot(lags[mid - L: mid + L + 1], cc[mid - L: mid + L + 1], linewidth=1.8)
        self.ax.axvline(0, color="k", linewidth=1, alpha=0.4)
        self.ax.set_xlabel("lags (frames)")
        self.ax.set_ylabel("cross correlation")
        self.ax.set_title(f"XCorr: cell {gi + self.label_base} × cell {gj + self.label_base}")
        self.canvas.draw_idle()

        kmax = np.argmax(cc)
        lag = int(lags[kmax])
        self.note.setText(f"max r = {cc[kmax]:.3f} at lag {lag} frames")

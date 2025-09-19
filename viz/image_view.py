from __future__ import annotations
"""
ImageView
---------
Qt GraphicsView-based grayscale image viewer with ROI contours.

Highlights
* Converts arbitrary float images to 8-bit for display using min–max scaling.
* Renders ROI masks as contour overlays (lazy import of scikit-image).
* Falls back to a simple edge-based polyline if scikit-image is unavailable.
* Emits `roiClicked(int)` with the ROI index (−1 if background) on mouse press.
"""

from typing import Optional, List
import os
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QPolygonF
from PySide6.QtCore import Qt, QPointF, QTimer, Signal, QEvent

# Lazy import cache for skimage.measure.find_contours
_SK_FIND_CONTOURS = None


def _get_find_contours():
    """Try importing scikit-image only when needed. Return callable or None."""
    global _SK_FIND_CONTOURS
    if _SK_FIND_CONTOURS is not None:
        return _SK_FIND_CONTOURS
    if os.environ.get("GCAMP_DISABLE_SKIMAGE", "0") in ("1", "true", "True"):
        _SK_FIND_CONTOURS = None
        return None
    try:
        from skimage.measure import find_contours as _fc  # type: ignore
    except Exception:
        _fc = None
    _SK_FIND_CONTOURS = _fc
    return _fc


def _fallback_contours(m: np.ndarray):
    """
    Very simple marching-squares substitute:
    return a set of boundary points (single broken polyline).
    """
    m = m.astype(bool)
    e = np.zeros_like(m, dtype=bool)
    e[:-1, :] |= m[:-1, :] ^ m[1:, :]
    e[:, :-1] |= m[:, :-1] ^ m[:, 1:]
    ys, xs = np.nonzero(e)
    if ys.size == 0:
        return []
    return [np.stack([ys, xs], axis=1).astype(float)]


class ImageView(QWidget):
    """GraphicsView wrapper which shows a single grayscale image and ROI overlays."""
    roiClicked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        self.view = QGraphicsView()
        v.addWidget(self.view)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)

        self._cached_qpixmap: Optional[QPixmap] = None
        self._roi_items: List = []
        self._label_map: Optional[np.ndarray] = None
        self.rois: Optional[np.ndarray] = None

        # Capture mouse events on the viewport
        self.view.viewport().installEventFilter(self)

        # Debounced scene update
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._render_scene)

    # ---- Public API -----------------------------------------------------------
    def set_image(self, img: Optional[np.ndarray]):
        """
        Set the background image. If None, clears the scene.
        Accepts a 2D array, any dtype. Internally mapped to QImage Grayscale8.
        """
        if img is None:
            self._cached_qpixmap = None
            self.scene.clear()
            return
        if img.ndim != 2:
            raise ValueError("ImageView expects a 2D grayscale image.")
        f = img.astype(np.float32)
        lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
        if hi <= lo + 1e-12:
            f8 = np.zeros_like(f, dtype=np.uint8)
        else:
            f8 = np.clip(255.0 * (f - lo) / (hi - lo), 0, 255).astype(np.uint8)
        h, w = f8.shape
        # copy() so the underlying buffer outlives the QImage
        qimg = QImage(f8.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
        self._cached_qpixmap = QPixmap.fromImage(qimg)
        self._schedule_scene_update()

    def set_rois(self, masks: Optional[np.ndarray]):
        """
        Provide ROI masks as (N, H, W) or (H, W). Builds a label map for hit-testing.
        """
        if masks is None:
            self.rois = None
            self._label_map = None
        else:
            self.rois = masks if masks.ndim == 3 else masks[None, ...]
            Y, X = self.rois.shape[1:]
            lab = np.zeros((Y, X), dtype=np.int32)
            for i, m in enumerate(self.rois, start=1):
                lab[m.astype(bool)] = i
            self._label_map = lab
        self._schedule_scene_update()

    def clear(self):
        """Remove the background image and all overlays."""
        self._cached_qpixmap = None
        self.rois = None
        self._label_map = None
        self.scene.clear()

    # ---- Internals ------------------------------------------------------------
    def _schedule_scene_update(self):
        if not self._update_timer.isActive():
            self._update_timer.start(0)

    def _render_scene(self):
        """(Re)render the scene with current image and ROI overlays."""
        self.view.setUpdatesEnabled(False)
        self.scene.clear()
        if self._cached_qpixmap is not None:
            self.scene.addPixmap(self._cached_qpixmap)
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        # ROI contours
        if self.rois is not None:
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(1)
            fc = _get_find_contours()
            for m in self.rois:
                mm = m.astype(bool, copy=False)
                if fc is not None:
                    try:
                        contours = fc(mm.view(np.float32), 0.5)  # type: ignore
                    except Exception:
                        contours = _fallback_contours(mm)
                else:
                    contours = _fallback_contours(mm)
                for cnt in contours:
                    if len(cnt) < 2:
                        continue
                    poly = QPolygonF([QPointF(float(xy[1]), float(xy[0])) for xy in cnt])
                    self.scene.addPolygon(poly, pen)

        self.view.setUpdatesEnabled(True)
        self.view.viewport().update()

    # ---- Mouse hit-testing ----------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self.view.viewport() and event.type() == QEvent.MouseButtonPress:
            pos = event.position() if hasattr(event, "position") else event.posF()
            sp = self.view.mapToScene(int(pos.x()), int(pos.y()))
            x, y = int(sp.x()), int(sp.y())
            if self._label_map is not None:
                Y, X = self._label_map.shape
                if 0 <= x < X and 0 <= y < Y:
                    ridx = int(self._label_map[y, x]) - 1
                    self.roiClicked.emit(ridx if ridx >= 0 else -1)
        return super().eventFilter(obj, event)

# gcamp_gui/main_window.py
from __future__ import annotations
import traceback
from pathlib import Path
from typing import Optional, List

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QCoreApplication
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSplitter,
    QStatusBar, QPushButton, QFileDialog, QTabWidget, QFrame, QToolBox,
    QProgressBar, QSizePolicy
)

from core.state import AppState
from core.config import AppConfig


# ---------------- Progress helper ----------------
class TaskProgress:
    def __init__(self, status: QStatusBar):
        self.status = status
        self._box = QWidget()
        lay = QHBoxLayout(self._box); lay.setContentsMargins(4, 0, 4, 0); lay.setSpacing(8)
        self.label = QLabel("")
        self.bar = QProgressBar(); self.bar.setFixedWidth(240)
        self.bar.setRange(0, 100); self.bar.setValue(0)
        lay.addWidget(self.label); lay.addWidget(self.bar)
        self.active = False

    def start(self, text: str, total: int | None = 100):
        self.label.setText(text)
        if total is None: self.bar.setRange(0, 0)
        else: self.bar.setRange(0, 100); self.bar.setValue(0)
        self.status.addPermanentWidget(self._box, 0)
        self._box.show(); self.active = True
        QCoreApplication.processEvents()

    def set(self, pct: float | int, text: str | None = None):
        if not self.active: return
        if text is not None: self.label.setText(text)
        try:
            v = int(max(0, min(100, pct)))
            if self.bar.maximum() == 0: self.bar.setRange(0, 100)
            self.bar.setValue(v)
        except Exception: pass
        QCoreApplication.processEvents()

    def finish(self, text: str | None = None):
        if not self.active: return
        if text: self.label.setText(text)
        try: self.status.removeWidget(self._box)
        except Exception: pass
        self.active = False
        QCoreApplication.processEvents()


# ---------------- SafeImageView ----------------
class SafeImageView(QWidget):
    roiDrawn = Signal(object)  # 최종 ROI 마스크(Bool) 1개(추가분)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._frame_u8: Optional[np.ndarray] = None
        self._stack: Optional[np.ndarray] = None
        self._roi_masks: Optional[np.ndarray] = None     # (N,H,W)
        self._np_masks: Optional[np.ndarray]  = None     # (N,H,W)
        self._highlights: set[int] = set()
        self._show_only_selected = False
        self._show_rois = True
        self._show_neuropil = False

        # 드로잉 상태
        self._tool: Optional[str] = None   # None|'rect'|'poly'|'free'
        self._dragging = False
        # rect
        self._p0_img = None
        self._p1_img = None
        # polygon
        self._poly_pts: list[tuple[float, float]] = []
        self._poly_hover: Optional[tuple[float, float]] = None
        self._poly_close_eps = 6.0
        # freehand
        self._free_preview_pos: Optional[tuple[float, float]] = None
        self._free_brush_r = 6
        self._free_temp_mask: Optional[np.ndarray] = None  # 드래그 중 임시 마스크
        # 누적 Crop 마스크(여러 개 합쳐짐)
        self._crop_mask: Optional[np.ndarray] = None

        self._font = QFont(); self._font.setPointSize(9); self._font.setBold(True)
        self._palette = [
            (244, 67, 54), (33, 150, 243), (76, 175, 80), (255, 193, 7),
            (156, 39, 176), (0, 188, 212), (255, 87, 34), (121, 85, 72),
            (63, 81, 181), (139, 195, 74), (3, 169, 244), (255, 152, 0),
            (233, 30, 99), (0, 150, 136), (205, 220, 57), (158, 158, 158),
        ]

        self.lbl = QLabel("No image", self)
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setScaledContents(False)

        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.addWidget(self.lbl)

        # 마우스 이벤트가 부모로 오게
        self.setMouseTracking(True)
        self.lbl.setMouseTracking(True)
        self.lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    # ---------- public setters ----------
    def set_stack(self, stack: np.ndarray | None): self._stack = stack
    def set_frame_index(self, i: int):
        if self._stack is None or self._stack.shape[0] == 0: return
        i = int(max(0, min(i, self._stack.shape[0]-1))); self.set_image(self._stack[i])
    def set_image(self, img: Optional[np.ndarray]):
        if img is None:
            self._frame_u8 = None; self.lbl.setText("No image"); self.lbl.setPixmap(QPixmap()); return
        f = img.astype(np.float32, copy=False)
        lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
        f8 = np.zeros_like(f, dtype=np.uint8) if (not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12) \
             else np.clip(255.0*(f-lo)/(hi-lo + 1e-12), 0, 255).astype(np.uint8)
        self._frame_u8 = f8; self._redraw()

    def set_rois(self, masks: Optional[np.ndarray]):
        self._roi_masks = masks.astype(bool, copy=False) if isinstance(masks, np.ndarray) and masks.ndim == 3 else None
        self._show_only_selected = False
        self._highlights.clear()
        self._redraw()

    def set_neuropil_masks(self, masks: Optional[np.ndarray]):
        self._np_masks = masks.astype(bool, copy=False) if isinstance(masks, np.ndarray) and masks.ndim == 3 else None
        self._redraw()

    def set_highlights(self, indices: List[int]):
        if self._roi_masks is None: self._highlights = set()
        else:
            n = int(self._roi_masks.shape[0])
            self._highlights = set(int(i) for i in (indices or []) if 0 <= int(i) < n)
        self._redraw()

    def set_show_only_selected(self, on: bool): self._show_only_selected = bool(on); self._redraw()
    def set_show_rois(self, on: bool): self._show_rois = bool(on); self._redraw()
    def set_show_neuropil(self, on: bool): self._show_neuropil = bool(on); self._redraw()

    def set_draw_tool(self, tool: Optional[str]):
        """
        tool ∈ {'rect','poly','free', None}
        - 다른 툴로 '변경'되는 경우: 누적된 ROI(오렌지 마스크) 초기화.
        - 같은 툴 → 그대로 유지(같은 툴 내에서 계속 누적).
        - None → 드로잉 해제(진행 중 스케치도 제거), 누적은 유지.
        """
        if tool not in (None, 'rect', 'poly', 'free'):
            tool = None

        if (self._tool is not None) and (tool is not None) and (tool != self._tool):
            # 툴 전환 → 누적 ROI 초기화
            self.cancel_active_draw(keep_accumulated=False)
        else:
            # 툴 유지 또는 None → 진행 중만 정리
            self.cancel_active_draw(keep_accumulated=True if tool is not None else True)

        self._tool = tool
        self._redraw()

    def enable_draw_mode(self, on: bool):
        # 외부 호환용: on=True여도 현재 tool=None이면 강제로 툴을 만들지 않음
        if not on:
            self._tool = None
            self.cancel_active_draw(keep_accumulated=True)
        self._redraw()

    def cancel_active_draw(self, keep_accumulated: bool = True):
        self._dragging = False
        self._p0_img = self._p1_img = None
        self._poly_pts.clear(); self._poly_hover = None
        self._free_preview_pos = None
        self._free_temp_mask = None
        if not keep_accumulated:
            self._crop_mask = None
        self._redraw()

    def set_crop_mask(self, mask_bool: Optional[np.ndarray]):
        self._crop_mask = mask_bool.astype(bool) if isinstance(mask_bool, np.ndarray) else None; self._redraw()
    def clear_crop(self): self.cancel_active_draw(keep_accumulated=False)

    # ---------- events ----------
    def resizeEvent(self, ev): super().resizeEvent(ev); self._redraw()

    def mousePressEvent(self, ev):
        if self._frame_u8 is None or ev.button() != Qt.LeftButton or not self._tool:
            return super().mousePressEvent(ev)
        pt = ev.position() if hasattr(ev, "position") else ev.pos()
        iy, ix = self._map_widget_to_image(pt)
        if self._tool == 'rect':
            self._dragging = True; self._p0_img = (iy, ix); self._p1_img = (iy, ix)
        elif self._tool == 'poly':
            if self._poly_pts:
                sy, sx = self._poly_pts[0]
                if (iy - sy)**2 + (ix - sx)**2 <= self._poly_close_eps**2 and len(self._poly_pts) >= 3:
                    self._finalize_polygon(); return
            self._poly_pts.append((iy, ix))
        elif self._tool == 'free':
            self._dragging = True
            if self._free_temp_mask is None:
                H, W = self._frame_u8.shape
                self._free_temp_mask = np.zeros((H, W), dtype=bool)
            self._stamp_brush(self._free_temp_mask, iy, ix, self._free_brush_r)
        self._redraw()

    def mouseMoveEvent(self, ev):
        if self._frame_u8 is None:
            return super().mouseMoveEvent(ev)
        pt = ev.position() if hasattr(ev, "position") else ev.pos()
        iy, ix = self._map_widget_to_image(pt)
        if self._tool == 'rect':
            if self._dragging: self._p1_img = (iy, ix)
        elif self._tool == 'poly':
            self._poly_hover = (iy, ix)
        elif self._tool == 'free':
            self._free_preview_pos = (iy, ix)
            if self._dragging:
                if self._free_temp_mask is None:
                    H, W = self._frame_u8.shape
                    self._free_temp_mask = np.zeros((H, W), dtype=bool)
                self._stamp_brush(self._free_temp_mask, iy, ix, self._free_brush_r)
        self._redraw()

    def mouseReleaseEvent(self, ev):
        if self._frame_u8 is None or ev.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(ev)
        if self._tool == 'rect' and self._dragging:
            self._dragging = False
            if (self._p0_img is not None) and (self._p1_img is not None):
                y0, x0 = self._p0_img; y1, x1 = self._p1_img
                mask = self._mask_from_rect(y0, x0, y1, x1)
                self._accumulate_crop(mask); self.roiDrawn.emit(mask)
            self._p0_img = self._p1_img = None
        elif self._tool == 'free' and self._dragging:
            self._dragging = False
            if self._free_temp_mask is not None:
                self._accumulate_crop(self._free_temp_mask.copy())
                self.roiDrawn.emit(self._free_temp_mask.copy())
            self._free_temp_mask = None
        self._redraw()

    def wheelEvent(self, ev):
        if self._tool == 'free' and (ev.modifiers() & Qt.ControlModifier):
            delta = ev.angleDelta().y() / 120.0
            self._free_brush_r = int(np.clip(self._free_brush_r + delta, 1, 50))
            self._redraw()
        else:
            super().wheelEvent(ev)

    # ---------- coord helper ----------
    def _map_widget_to_image(self, pt):
        H, W = self._frame_u8.shape
        lw, lh = max(1, self.lbl.width()), max(1, self.lbl.height())
        s = min(lw/float(W), lh/float(H))
        disp_w, disp_h = int(W*s), int(H*s)
        off_x = (lw - disp_w)//2; off_y = (lh - disp_h)//2
        x = (float(pt.x()) - off_x) / s; y = (float(pt.y()) - off_y) / s
        return float(np.clip(y, 0, H-1)), float(np.clip(x, 0, W-1))

    # ---------- mask builders ----------
    def _mask_from_rect(self, y0, x0, y1, x1) -> np.ndarray:
        H, W = self._frame_u8.shape
        yy0, yy1 = sorted([int(np.floor(y0)), int(np.ceil(y1))])
        xx0, xx1 = sorted([int(np.floor(x0)), int(np.ceil(x1))])
        yy0, xx0 = max(0, yy0), max(0, xx0)
        yy1, xx1 = min(H, yy1), min(W, xx1)
        m = np.zeros((H, W), dtype=bool)
        if yy1 > yy0 and xx1 > xx0: m[yy0:yy1, xx0:xx1] = True
        return m

    def _mask_from_polygon(self, pts: list[tuple[float, float]]) -> np.ndarray:
        from matplotlib.path import Path
        H, W = self._frame_u8.shape
        poly = np.array([(x, y) for (y, x) in pts], dtype=float)  # Path는 (x,y)
        if poly.shape[0] < 3: return np.zeros((H, W), dtype=bool)
        path = Path(poly)
        yy, xx = np.mgrid[0:H, 0:W]
        inside = path.contains_points(np.vstack((xx.ravel(), yy.ravel())).T).reshape((H, W))
        return inside

    def _stamp_brush(self, dst: np.ndarray, y: float, x: float, r: int):
        H, W = dst.shape
        y0 = max(0, int(np.floor(y - r))); y1 = min(H, int(np.ceil(y + r)) + 1)
        x0 = max(0, int(np.floor(x - r))); x1 = min(W, int(np.ceil(x + r)) + 1)
        if y1 <= y0 or x1 <= x0: return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dst[y0:y1, x0:x1] |= ((yy - y)**2 + (xx - x)**2) <= (r**2)

    def _finalize_polygon(self):
        if len(self._poly_pts) >= 3:
            mask = self._mask_from_polygon(self._poly_pts)
            self._accumulate_crop(mask); self.roiDrawn.emit(mask)
        self._poly_pts.clear(); self._poly_hover = None

    def _accumulate_crop(self, mask: np.ndarray):
        if self._crop_mask is None: self._crop_mask = mask.astype(bool)
        else: self._crop_mask |= mask.astype(bool)

    # ---------- painting ----------
    def _palette_color(self, i: int) -> QColor:
        r,g,b = self._palette[i % len(self._palette)]
        return QColor(r, g, b)

    @staticmethod
    def _edge_from_mask(m: np.ndarray) -> np.ndarray:
        e = np.zeros_like(m, dtype=bool)
        e[:-1,:] |= m[:-1,:] ^ m[1:,:]
        e[:,:-1] |= m[:,:-1] ^ m[:,1:]
        return e

    def _draw_mask_points(self, p: QPainter, mask: np.ndarray, color: QColor):
        ys, xs = np.where(mask)
        if ys.size == 0: return
        p.setPen(QPen(color, 0))  # drawPoint는 펜 색을 사용
        for y, x in zip(ys, xs):
            p.drawPoint(int(x), int(y))

    def _redraw(self):
        if self._frame_u8 is None: return
        base = QPixmap.fromImage(QImage(
            self._frame_u8.tobytes(), self._frame_u8.shape[1], self._frame_u8.shape[0],
            self._frame_u8.shape[1], QImage.Format_Grayscale8).copy()
        )
        pm = self._paint(base)
        pm = pm.scaled(self.lbl.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.lbl.setPixmap(pm)

    def _paint(self, pixmap: QPixmap) -> QPixmap:
        pm = pixmap.copy(); p = QPainter(pm)
        try:
            p.setFont(self._font)

            # 0) 누적 Crop 마스크(오렌지 α≈0.3)
            if self._crop_mask is not None and self._crop_mask.any():
                self._draw_mask_points(p, self._crop_mask, QColor(255, 128, 0, 77))

            # 1) 뉴로필 링(항상 전체 ROI에 대해)
            if self._show_neuropil and (self._np_masks is not None) and (self._roi_masks is not None):
                N = int(min(self._np_masks.shape[0], self._roi_masks.shape[0]))
                for i in range(N):
                    ring = self._np_masks[i]
                    if not ring.any(): continue
                    edge = self._edge_from_mask(ring)
                    ys, xs = np.nonzero(edge)
                    if ys.size:
                        col = self._palette_color(i)
                        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 220), 0, Qt.DashLine))
                        for y, x in zip(ys, xs): p.drawPoint(int(x), int(y))

            # 2) ROI 윤곽 + 라벨
            if self._show_rois and (self._roi_masks is not None):
                N = self._roi_masks.shape[0]
                idxs = [int(i) for i in sorted(self._highlights)] if (self._show_only_selected and self._highlights) else range(N)
                for i in idxs:
                    m = self._roi_masks[i]
                    if not m.any(): continue
                    edge = self._edge_from_mask(m)
                    ys, xs = np.nonzero(edge)
                    col = self._palette_color(i)
                    if ys.size:
                        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 255), 0))
                        for y, x in zip(ys, xs): p.drawPoint(int(x), int(y))
                    coords = np.argwhere(m)
                    if coords.size:
                        cy, cx = coords.mean(axis=0); cy, cx = int(round(cy)), int(round(cx))
                        p.setPen(QPen(QColor(0, 0, 0, 220), 0)); p.drawText(cx+1, cy+1, str(i))
                        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 255), 0)); p.drawText(cx, cy, str(i))

            # 3) 진행 중 프리뷰(오렌지 채움, 실시간)
            orange = QColor(255, 128, 0, 77)
            if self._tool == 'rect' and (self._p0_img is not None) and (self._p1_img is not None):
                y0, x0 = self._p0_img; y1, x1 = self._p1_img
                self._draw_mask_points(p, self._mask_from_rect(y0, x0, y1, x1), orange)
                p.setPen(QPen(QColor(255,128,0,180), 0, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))

            elif self._tool == 'poly':
                if self._poly_pts:
                    pts = self._poly_pts + ([self._poly_hover] if self._poly_hover else [])
                    if len(pts) >= 3:
                        self._draw_mask_points(p, self._mask_from_polygon(pts), orange)
                    # 선분/점
                    p.setPen(QPen(QColor(255, 180, 0, 220), 0))
                    for k in range(len(self._poly_pts)-1):
                        y0, x0 = self._poly_pts[k]; y1, x1 = self._poly_pts[k+1]
                        p.drawLine(int(x0), int(y0), int(x1), int(y1))
                    if self._poly_hover and self._poly_pts:
                        y0, x0 = self._poly_pts[-1]; y1, x1 = self._poly_hover
                        p.drawLine(int(x0), int(y0), int(x1), int(y1))
                    # 점(시작점 강조)
                    for (yy, xx) in self._poly_pts:
                        p.drawEllipse(int(xx)-2, int(yy)-2, 4, 4)
                    sy, sx = self._poly_pts[0]
                    p.setPen(QPen(QColor(255, 90, 0, 255), 0))
                    p.drawEllipse(int(sx)-3, int(sy)-3, 6, 6)

            elif self._tool == 'free':
                # 브러시 프리뷰 링
                if self._free_preview_pos is not None:
                    y, x = self._free_preview_pos
                    p.setPen(QPen(QColor(0,0,0,90), 0))
                    p.setBrush(Qt.NoBrush)
                    r = self._free_brush_r
                    p.drawEllipse(int(x)-r, int(y)-r, r*2, r*2)
                # 드래그 중 임시 채움
                if self._free_temp_mask is not None and self._free_temp_mask.any():
                    self._draw_mask_points(p, self._free_temp_mask, orange)

        finally:
            p.end()
        return pm


# ---------------- MainWindow ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GCaMP Calcium Analysis")
        self.resize(1920, 1230)
        self.state = AppState()
        self.cfg = AppConfig()
        try:
            if not isinstance(self.cfg.segmentation, dict): self.cfg.segmentation = {}
            self.cfg.segmentation['backend'] = 'cnmf'  # 기본값 CNMF
        except Exception: pass

        self.status = QStatusBar(self); self.setStatusBar(self.status)
        self.progress = TaskProgress(self.status)
        self._log("Booting GUI...")

        root = QWidget(self); self.setCentralWidget(root)
        rv = QVBoxLayout(root); rv.setContentsMargins(6, 6, 6, 6); rv.setSpacing(6)
        self.splitter = QSplitter(Qt.Horizontal, self); rv.addWidget(self.splitter, 1)

        # ---------- left ----------
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(4,4,4,4); lv.setSpacing(6)
        header = QHBoxLayout()
        self.btn_open = QPushButton("Open Stack")
        self.file_label = QLabel("No file loaded"); self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        header.addWidget(self.btn_open, 0); header.addWidget(self.file_label, 1); header.addStretch(1)
        lv.addLayout(header)

        self.viewer = SafeImageView()

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedHeight(44); self.play_btn.setMinimumWidth(160)
        self.play_btn.setStyleSheet("font-size:18px; font-weight:600;")
        play_row = QHBoxLayout(); play_row.setContentsMargins(0,0,0,0)
        play_row.addStretch(1); play_row.addWidget(self.play_btn); play_row.addStretch(1)
        play_frame = QFrame(); play_frame.setLayout(play_row)

        lv.addWidget(self.viewer, 1)
        lv.addWidget(play_frame, 0)
        self.splitter.addWidget(left)

        # ---------- right ----------
        right = QWidget(); rv2 = QVBoxLayout(right); rv2.setContentsMargins(2,2,2,2); rv2.setSpacing(6)
        self.steps = QToolBox(); self.results = QTabWidget()
        rv2.addWidget(self.steps, 0); rv2.addWidget(self.results, 1)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 5); self.splitter.setStretchFactor(1, 4)

        self._build_panels()

        self._play_timer = QTimer(self); self._play_timer.setInterval(50)
        self._playing = False; self._play_idx = 0
        self.play_btn.clicked.connect(self._toggle_play)
        self.btn_open.clicked.connect(self.on_load_stack_clicked)

        self.viewer.roiDrawn.connect(self._on_viewer_box_drawn)

        self._original_stack: Optional[np.ndarray] = None
        self._preproc_params = dict(b=0.0, c=1.0, g=1.0, eq=False)
        self._crop_mode_active = False
        self._select_mode_active = False
        self._last_crop_mask: Optional[np.ndarray] = None
        self._active_tool: Optional[str] = None  

        self._log("Ready.")

    # ---------- Panels ----------
    def _build_panels(self):
        from viz.preprocess_widget import PreprocessWidget
        from viz.params_widget import ParamsWidget
        from viz.traces_widget import TracesWidget
        try:    from viz.analysis_widget import AnalysisWidget
        except Exception: AnalysisWidget = None
        try:    from viz.tuning_widget import TuningWidget
        except Exception: TuningWidget = None
        try:    from viz.qc_widget import QcWidget
        except Exception: QcWidget = None

        self.preproc_tab = PreprocessWidget()
        self.params_tab = ParamsWidget(self.cfg)
        self.steps.addItem(self.preproc_tab, "Preprocess")
        self.steps.addItem(self.params_tab, "Parameter Setting")

        # Preprocess wiring
        self.preproc_tab.paramsChanged.connect(self._on_preproc_params_changed)
        self.preproc_tab.applyClicked.connect(self._on_preproc_apply_clicked)
        self.preproc_tab.resetClicked.connect(self._on_preproc_reset_clicked)
        self.preproc_tab.cropModeToggled.connect(self._on_crop_mode_toggled)
        self.preproc_tab.applyCropClicked.connect(self._on_apply_crop_clicked)
        self.preproc_tab.clearCropClicked.connect(self._on_clear_crop_clicked)
        self.preproc_tab.toolSelected.connect(self._on_draw_tool_selected)

        self.traces_tab = TracesWidget()

        # Analysis 탭 상단에 Save/Clear
        self.analysis_tab = AnalysisWidget() if AnalysisWidget else QLabel("Analysis")
        self.analysis_page = QWidget()
        ap_v = QVBoxLayout(self.analysis_page); ap_v.setContentsMargins(6,6,6,6); ap_v.setSpacing(6)
        topbar = QFrame(); tb = QHBoxLayout(topbar); tb.setContentsMargins(0,0,0,6); tb.setSpacing(8)
        tb.addStretch(1)
        self.btn_save_results = QPushButton("Save Results")
        self.btn_clear_results = QPushButton("Clear Results")
        tb.addWidget(self.btn_save_results); tb.addWidget(self.btn_clear_results)
        ap_v.addWidget(topbar, 0); ap_v.addWidget(self.analysis_tab, 1)
        self.btn_save_results.clicked.connect(self._on_save_results_clicked)
        self.btn_clear_results.clicked.connect(self._on_clear_results_clicked)
        self._tweak_analysis_ui_hide_spiking_clear()

        self.tuning_tab = TuningWidget() if TuningWidget else QLabel("Tuning")
        self.qc_tab     = QcWidget() if QcWidget else QLabel("QC")

        self.results.addTab(self.traces_tab, "Traces")
        self.results.addTab(self.analysis_page, "Analysis")
        self.results.addTab(self.tuning_tab, "Tuning")
        self.results.addTab(self.qc_tab, "QC")
        self._disable_results_tabs()

        # params
        self.params_tab.applyClicked.connect(self._on_params_apply_clicked)
        self.params_tab.runPipelineClicked.connect(self.on_run_clicked)

        # traces ↔ viewer/analysis
        self.traces_tab.selectionChanged.connect(self._on_rois_selection_changed)
        self.traces_tab.selectionChanged.connect(self._on_selection_for_analysis)
        self.traces_tab.requestRectSelect.connect(self._begin_rect_select_in_view)

    def _tweak_analysis_ui_hide_spiking_clear(self):
        try:
            buttons = self.analysis_tab.findChildren(QPushButton)
            targets = {"clear", "clear spikes", "clear spiking"}
            for b in buttons:
                if (b.text() or "").strip().lower() in targets:
                    b.hide()
        except Exception as e:
            self._log(f"[analysis tweak warn] {e}")

    def _disable_results_tabs(self):
        for w in (self.traces_tab, self.analysis_tab, self.tuning_tab, self.qc_tab):
            if hasattr(w, "setEnabled"): w.setEnabled(False)

    def _enable_results_tabs(self):
        for w in (self.traces_tab, self.analysis_tab, self.tuning_tab, self.qc_tab):
            if hasattr(w, "setEnabled"): w.setEnabled(True)

    # ---------- Open stack ----------
    @Slot()
    def on_load_stack_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Stack", "",
            "Images/Video (*.tif *.tiff *.npy *.avi *.mp4);;All Files (*)"
        )
        if not path: return
        try:
            self._hard_reset_state()
            self.progress.start("Loading stack…", None)
            QCoreApplication.processEvents()

            self._log(f"Loading stack: {path}")
            from project_io.loaders import load_stack
            stack, meta = load_stack(path)

            self.state.raw_stack = stack.astype(np.float32, copy=False)
            self._original_stack = self.state.raw_stack.copy()
            self.cfg.fps = meta.get("fps", self.cfg.fps)

            self.viewer.set_stack(self.state.raw_stack)
            self.viewer.set_rois(None)
            self.viewer.set_neuropil_masks(None)
            self.viewer.set_image(self.state.raw_stack.mean(axis=0))
            self.file_label.setText(f"File: {Path(path).name}")
            self._log(f"Loaded. shape={stack.shape} meta={meta}")
        except Exception as e:
            self._err("Load error", e)
        finally:
            self.progress.finish("Ready")

    def _hard_reset_state(self):
        self.viewer.set_rois(None); self.viewer.clear_crop()
        self.viewer.set_neuropil_masks(None)
        self.file_label.setText("No file loaded")
        for k in ("raw_stack","reg_stack","roi_masks","np_masks","F_raw","F_np","F0","dff","spikes","tuning_summary","qc_metrics"):
            if hasattr(self.state, k): setattr(self.state, k, None)
        self._disable_results_tabs()
        if hasattr(self.traces_tab, "reset"): self.traces_tab.reset()
        if hasattr(self.analysis_tab, "reset"):
            try: self.analysis_tab.reset()
            except Exception: pass
        # 툴 상태/누적도 초기화
        self._active_tool = None
        self._last_crop_mask = None
        self.viewer.set_draw_tool(None)

    # ---------- Playback ----------
    def _get_movie_stack(self) -> Optional[np.ndarray]:
        S = getattr(self.state, "reg_stack", None)
        if isinstance(S, np.ndarray) and S.ndim == 3 and S.shape[0] > 0: return S
        S = getattr(self.state, "raw_stack", None)
        if isinstance(S, np.ndarray) and S.ndim == 3 and S.shape[0] > 0: return S
        return None

    def _toggle_play(self):
        if self._playing:
            self._playing = False; self._play_timer.stop()
            try: self._play_timer.timeout.disconnect(self._on_play_tick)
            except Exception: pass
            self.play_btn.setText("▶ Play"); return
        mov = self._get_movie_stack()
        if mov is None: self._log("No movie loaded for playback."); return
        self.viewer.set_stack(mov); self._play_idx = 0; self._playing = True
        fps = float(self.cfg.fps) if getattr(self.cfg, "fps", None) else 10.0
        self._play_timer.setInterval(max(15, int(1000/max(1.0, fps))))
        try: self._play_timer.timeout.disconnect(self._on_play_tick)
        except Exception: pass
        self._play_timer.timeout.connect(self._on_play_tick); self._play_timer.start()
        self.play_btn.setText("⏸ Pause")

    def _on_play_tick(self):
        mov = self._get_movie_stack()
        if mov is None or mov.shape[0] == 0 or self.viewer is None:
            self._toggle_play(); return
        if self._play_idx >= mov.shape[0]: self._play_idx = 0
        self.viewer.set_image(mov[self._play_idx]); self._play_idx += 1

    # ---------- Preprocess ----------
    def _on_preproc_params_changed(self, b, c, g, eq):
        self._preproc_params.update(dict(b=b, c=c, g=max(1e-6, g), eq=bool(eq)))
        if getattr(self.state, "raw_stack", None) is None: return
        img = self.state.raw_stack.mean(axis=0)
        img_p = self._apply_preproc_frame(img, self._preproc_params)
        self.viewer.set_image(img_p)

    def _on_preproc_apply_clicked(self):
        if getattr(self.state, "raw_stack", None) is None:
            self._log("No stack loaded."); return
        S = self.state.raw_stack; T = int(S.shape[0])
        self._log("Applying preprocess to entire stack...")
        self.progress.start("Applying preprocess…", 100)
        prm = self._preproc_params; P = np.empty_like(S, dtype=np.float32)
        for t in range(T):
            P[t] = self._apply_preproc_frame(S[t], prm)
            if t % max(1, T // 100) == 0:
                self.progress.set(100.0 * (t+1) / T, f"Applying preprocess… {t+1}/{T}")
        self.state.raw_stack = P
        for k in ("reg_stack","roi_masks","np_masks","F_raw","F_np","F0","dff","spikes","tuning_summary","qc_metrics"):
            if hasattr(self.state, k): setattr(self.state, k, None)
        self.viewer.set_stack(P); self.viewer.set_rois(None); self.viewer.set_neuropil_masks(None)
        self.viewer.set_image(P.mean(axis=0))
        self._disable_results_tabs()
        self.progress.finish("Preprocess applied."); self._log("Preprocess applied to all frames.")

    def _on_draw_tool_selected(self, name: Optional[str]):
        # name ∈ {'rect','poly','free', None}
        name = name if name in ("rect", "poly", "free") else None
        # 툴 변경 시(MainWindow 레벨)도 last_crop 초기화
        if (self._active_tool is not None) and (name is not None) and (name != self._active_tool):
            self._last_crop_mask = None
        self._active_tool = name

        self._crop_mode_active = bool(name)
        if hasattr(self.viewer, "set_draw_tool"): self.viewer.set_draw_tool(name)
        if hasattr(self.viewer, "enable_draw_mode"): self.viewer.enable_draw_mode(bool(name))

    def _on_preproc_reset_clicked(self):
        if self._original_stack is None: return
        self.state.raw_stack = self._original_stack.copy()
        for k in ("reg_stack","roi_masks","np_masks","F_raw","F_np","F0","dff","spikes","tuning_summary","qc_metrics"):
            if hasattr(self.state, k): setattr(self.state, k, None)
        self.viewer.set_stack(self.state.raw_stack); self.viewer.set_rois(None); self.viewer.set_neuropil_masks(None)
        self.viewer.set_image(self.state.raw_stack.mean(axis=0))
        self._disable_results_tabs(); self._log("Reset to original stack.")
        # ROI/툴 상태 리셋
        self._last_crop_mask = None
        self._active_tool = None
        self.viewer.set_draw_tool(None)

    def _on_crop_mode_toggled(self, on: bool):
        self._crop_mode_active = bool(on)
        self.viewer.enable_draw_mode(on)
        self._log("Crop mode " + ("ON" if on else "OFF"))

    def _on_clear_crop_clicked(self):
        self._last_crop_mask = None; self.viewer.clear_crop(); self._log("Crop preview cleared.")

    # ---------- Rect selection for Traces ----------
    def _begin_rect_select_in_view(self):
        if getattr(self.state, "roi_masks", None) is None:
            self._log("Run pipeline (or import ROIs) first."); return

        # selection 모드 진입: crop 모드는 강제로 OFF, 기존 crop 마스크도 초기화
        self._select_mode_active = True
        self._crop_mode_active = False
        self._last_crop_mask = None

        if hasattr(self.viewer, "set_draw_tool"):
            self.viewer.set_draw_tool("rect")
        if hasattr(self.viewer, "enable_draw_mode"):
            self.viewer.enable_draw_mode(True)

        self._log("Draw a rectangle on image to SELECT cells inside it…")

    def _on_viewer_box_drawn(self, mask_bool: np.ndarray):
        if mask_bool is None or mask_bool.dtype != bool:
            return

        if self._select_mode_active:
            self._select_mode_active = False

            if hasattr(self.viewer, "enable_draw_mode"):
                self.viewer.enable_draw_mode(False)

            if hasattr(self.viewer, "clear_crop"):
                self.viewer.clear_crop()
            self._last_crop_mask = None  

            masks = getattr(self.state, "roi_masks", None)
            if masks is None or masks.size == 0:
                self._log("No ROI masks."); return

            inside = [
                i for i in range(masks.shape[0])
                if np.logical_and(masks[i], mask_bool).any()
            ]
            if not inside:
                self._log("No cells found in the box.")
                return

            try:
                self.traces_tab.check_only(inside)
            except Exception:
                pass

            self.viewer.set_show_only_selected(True)
            self.viewer.set_highlights(inside)

            self._on_selection_for_analysis(inside)
            return

        if self._crop_mode_active:
            if self._last_crop_mask is None:
                self._last_crop_mask = mask_bool.copy()
            else:
                try:
                    self._last_crop_mask |= mask_bool
                except ValueError:
                    self._last_crop_mask = mask_bool.copy()
            self._log("Crop ROI captured (accumulated). Click 'Apply Crop'.")
            return

        if self._crop_mode_active:
            if self._last_crop_mask is None:
                self._last_crop_mask = mask_bool.copy()
            else:
                try:
                    self._last_crop_mask |= mask_bool
                except ValueError:
                    self._last_crop_mask = mask_bool.copy()
            self._log("Crop ROI captured (accumulated). Click 'Apply Crop'.")


    def _on_apply_crop_clicked(self):
        if self._last_crop_mask is None:
            self._log("No crop ROI captured."); return
        S = getattr(self.state, "raw_stack", None)
        if S is None: return
        if self._last_crop_mask.shape != S.shape[1:3]:
            self._log("Crop mask size does not match current image."); return

        self.progress.start("Applying mask to stack…", 100)
        try:
            M = self._last_crop_mask.astype(S.dtype)
            S2 = S * M[None, :, :]  # ROI 외부는 0(검정)
            self.state.raw_stack = S2
            for k in ("reg_stack","roi_masks","np_masks","F_raw","F_np","F0","dff","spikes","tuning_summary","qc_metrics"):
                if hasattr(self.state, k): setattr(self.state, k, None)
            self.viewer.clear_crop()
            self.viewer.set_stack(S2); self.viewer.set_image(S2.mean(axis=0))
            self._disable_results_tabs()
            self._log("Applied mask (outside ROI -> black).")
        finally:
            self.progress.finish("Ready")

    # ---------- Params ----------
    def _on_params_apply_clicked(self, d: dict):
        try:
            self.cfg.segmentation.update(d.get("segmentation", {}))
            self.cfg.segmentation['backend'] = 'cnmf'
            pre = d.get("preprocess", {})
            if pre: self.cfg.preprocess.update(pre)
            self.cfg.deconvolution.update(d.get("deconvolution", {}))
            self._log("Parameters updated (backend forced to cnmf).")
        except Exception as e:
            self._err("Params apply error", e)

    # ---------- Save / Clear Results ----------
    def _on_save_results_clicked(self):
        outdir = self._pick_directory("Choose folder to save results")
        if not outdir: return
        outdir = Path(outdir)
        try:
            self.progress.start("Saving results…", None)
            outdir.mkdir(parents=True, exist_ok=True)
            try:
                from project_io.writers import save_csv_pack, save_nwb
                from viz.report import export_report

                # If available, use the currently CHECKED cells in the Traces tab
                # to decide which ROIs to export. If anything goes wrong, fall
                # back to exporting all ROIs.
                selected_indices = None
                try:
                    if getattr(self, "traces_tab", None) is not None and hasattr(self.traces_tab, "current_checked_indices"):
                        selected_indices = self.traces_tab.current_checked_indices()
                except Exception:
                    selected_indices = None

                save_csv_pack(outdir, self.state, selected_indices)
                export_report(outdir, self.state)
                save_nwb(outdir / "session.nwb", self.state, fps=self.cfg.fps)
            except Exception as e:
                self._log(f"[warn] writers/report: {e}")
            if getattr(self.state, "roi_masks", None) is not None:
                np.save(outdir / "roi_masks.npy", self.state.roi_masks.astype(np.uint8))
            if getattr(self.state, "np_masks", None) is not None:
                np.save(outdir / "np_masks.npy", self.state.np_masks.astype(np.uint8))
            if getattr(self.state, "dff", None) is not None:
                np.save(outdir / "dff.npy", self.state.dff.astype(np.float32))
            if getattr(self.state, "spikes", None) is not None:
                np.save(outdir / "spikes.npy", self.state.spikes.astype(np.float32))
            self._log(f"Saved results to: {outdir}")
        except Exception as e:
            self._err("Save results error", e)
        finally:
            self.progress.finish("Saved.")

    def _on_clear_results_clicked(self):
        try:
            if hasattr(self.analysis_tab, "clear_results"): self.analysis_tab.clear_results()
            elif hasattr(self.analysis_tab, "clear"): self.analysis_tab.clear()
            elif hasattr(self.analysis_tab, "reset"): self.analysis_tab.reset()
            else:
                if hasattr(self.analysis_tab, "set_selected_indices"): self.analysis_tab.set_selected_indices([])
                if hasattr(self.analysis_tab, "refresh"): self.analysis_tab.refresh()
            self._log("Analysis results cleared.")
        except Exception as e:
            self._err("Clear Results error", e)

    # ---------- CaImAn (optional) ----------
    def _ensure_and_push_np_masks(self):
        masks = getattr(self.state, "roi_masks", None)
        npm   = getattr(self.state, "np_masks",  None)
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            N = masks.shape[0]
            need = (not isinstance(npm, np.ndarray)) or npm.ndim != 3 or (npm.shape[0] != N)
            if need:
                def _dilate(m: np.ndarray, it: int) -> np.ndarray:
                    it = int(max(1, it)); out = m.astype(bool, copy=False)
                    for _ in range(it):
                        p = np.pad(out, 1, False)
                        out = (p[1:-1,1:-1] | p[:-2,1:-1] | p[2:,1:-1] |
                               p[1:-1,:-2] | p[1:-1,2:] | p[:-2,:-2] | p[:-2,2:] | p[2:,:-2] | p[2:,2:])
                    return out
                inner = int(self.cfg.preprocess.get("neuropil", {}).get("inner", 1))
                outer = int(self.cfg.preprocess.get("neuropil", {}).get("outer", 4))
                inner = max(1, inner); outer = max(inner+1, outer)
                rings = []
                for i in range(N):
                    m = masks[i].astype(bool, copy=False)
                    rin = _dilate(m, inner); rout = _dilate(m, outer)
                    ring = np.logical_and(rout ^ rin, ~m)
                    rings.append(ring)
                self.state.np_masks = np.stack(rings, 0).astype(bool)
            if hasattr(self.viewer, "set_neuropil_masks"):
                self.viewer.set_neuropil_masks(self.state.np_masks)

    # ---------- Save dialog helper ----------
    def _pick_directory(self, title: str) -> str:
        dlg = QFileDialog(self, title)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        start_dir = getattr(self, "_last_savedir", "")
        if start_dir: dlg.setDirectory(start_dir)
        if dlg.exec():
            sel = dlg.selectedFiles()
            if sel:
                path = sel[0]
                self._last_savedir = path
                return path
        return ""

    # ---------- ROI selection sync ----------
    def _on_rois_selection_changed(self, indices: List[int]):
        if self.viewer is None: return
        n = 0 if getattr(self.state, "roi_masks", None) is None else int(self.state.roi_masks.shape[0])
        if not indices or len(indices) >= n:
            self.viewer.set_show_only_selected(False); self.viewer.set_highlights([])
        else:
            self.viewer.set_show_only_selected(True); self.viewer.set_highlights(indices or [])

    def _on_selection_for_analysis(self, indices: List[int]):
        if self.analysis_tab is None: return
        try:
            if hasattr(self.analysis_tab, "set_selected_indices"):
                self.analysis_tab.set_selected_indices(indices or [])
            else:
                setattr(self.analysis_tab, "selected_indices", indices or [])
                if hasattr(self.analysis_tab, "refresh"): self.analysis_tab.refresh()
        except Exception as e:
            self._log(f"[Analysis selection warn] {e}")

    # ---------- Pipeline ----------
    @Slot()
    def on_run_clicked(self):
        if getattr(self.state, "raw_stack", None) is None:
            self._log("Load data first."); return
        try:
            self._log("Running pipeline...")
            from core.pipeline import Pipeline
            pipe = Pipeline(self.state, self.cfg)
            self.progress.start("Pipeline…", 100)
            for p, msg in pipe.run_all():
                self._log(f"[{p:3d}%] {msg}"); self.progress.set(p, f"{msg} ({p}%)")
            self.viewer.set_rois(getattr(self.state, "roi_masks", None))
            if hasattr(self.viewer, "set_neuropil_masks"):
                self.viewer.set_neuropil_masks(getattr(self.state, "np_masks", None))
            self._enable_results_tabs()
            if hasattr(self.traces_tab, "update_traces"):
                self.traces_tab.update_traces(getattr(self.state, "dff", None),
                                              getattr(self.state, "spikes", None))
            self._push_to_analysis_tabs()
            self._log("Pipeline done.")
        except Exception as e:
            self._err("Pipeline error", e)
        finally:
            self.progress.finish("Ready")

    # ---------- analysis feed ----------
    def _push_to_analysis_tabs(self):
        if self.analysis_tab is None: return
        dff    = getattr(self.state, "dff", None)
        spikes = getattr(self.state, "spikes", None)
        masks  = getattr(self.state, "roi_masks", None)
        try:
            if hasattr(self.analysis_tab, "set_data"):
                self.analysis_tab.set_data(dff=dff, spikes=spikes, masks=masks)
            elif hasattr(self.analysis_tab, "update_data"):
                self.analysis_tab.update_data(dff, spikes, masks)
            else:
                setattr(self.analysis_tab, "dff", dff)
                setattr(self.analysis_tab, "spikes", spikes)
                setattr(self.analysis_tab, "roi_masks", masks)
            sel: List[int] = []
            if hasattr(self.traces_tab, "current_checked_indices"):
                try: sel = list(self.traces_tab.current_checked_indices() or [])
                except Exception: sel = []
            if hasattr(self.analysis_tab, "set_selected_indices"):
                self.analysis_tab.set_selected_indices(sel)
            else:
                setattr(self.analysis_tab, "selected_indices", sel)
            if hasattr(self.analysis_tab, "refresh"): self.analysis_tab.refresh()
        except Exception as e:
            self._log(f"[Analysis feed warn] {e}")

    # ---------- utils ----------
    def _apply_preproc_frame(self, img: np.ndarray, prm: dict) -> np.ndarray:
        f = img.astype(np.float32, copy=False)
        lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
        x = (f - lo) / (hi - lo + 1e-12)
        b = float(prm.get("b", 0.0)); c = float(prm.get("c", 1.0))
        g = float(prm.get("g", 1.0)); eq = bool(prm.get("eq", False))
        y = np.clip(0.5 + c * (x + b - 0.5), 0.0, 1.0)
        if g > 0 and abs(g - 1.0) > 1e-6: y = np.clip(y, 1e-6, 1.0) ** (1.0 / g)
        if eq:
            bins = 256
            hist, _ = np.histogram(y.ravel(), bins=bins, range=(0.0, 1.0))
            cdf = hist.cumsum().astype(np.float32); cdf /= max(1.0, cdf[-1])
            idx = np.clip((y * (bins - 1)).astype(int), 0, bins - 1)
            y = cdf[idx]
        out = y * (hi - lo) + lo
        return out.astype(np.float32)

    def _pick_directory(self, title: str) -> str:
        dlg = QFileDialog(self, title)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        start_dir = getattr(self, "_last_savedir", "")
        if start_dir: dlg.setDirectory(start_dir)
        if dlg.exec():
            sel = dlg.selectedFiles()
            if sel:
                path = sel[0]
                self._last_savedir = path
                return path
        return ""

    def _log(self, msg: str):
        print(msg, flush=True)
        try: self.status.showMessage(msg, 5000)
        except Exception: pass

    def _err(self, title: str, e: Exception):
        m = f"{title}: {e}\n{traceback.format_exc()}"
        print(m, flush=True)
        try: self.status.showMessage(f"{title}: {e}", 8000)
        except Exception: pass

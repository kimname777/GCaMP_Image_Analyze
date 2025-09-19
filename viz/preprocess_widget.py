from __future__ import annotations
"""
PreprocessWidget
----------------
UI for previewing brightness/contrast/gamma and for defining a crop ROI.

Signals
-------
paramsChanged(b, c, g, eq) : brightness (−0.5..0.5), contrast (≈0..3), gamma (≈0..3), histogram equalization toggle
applyClicked()             : apply B/C/G to the whole stack
resetClicked()             : return to original stack
applyCropClicked()         : apply the current crop ROI
clearCropClicked()         : clear the crop ROI
cropModeToggled(bool)      : enter/leave crop drawing mode
toolSelected(str)          : 'rect' | 'poly' | 'free' | ''
"""

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QSlider, QCheckBox, QButtonGroup, QSizePolicy
)


class PreprocessWidget(QWidget):
    paramsChanged = Signal(float, float, float, bool)  # b, c, g, eq
    applyClicked = Signal()
    resetClicked = Signal()
    applyCropClicked = Signal()
    clearCropClicked = Signal()
    cropModeToggled = Signal(bool)
    nextClicked = Signal()
    toolSelected = Signal(str)  # 'rect' | 'poly' | 'free' | ''

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(10)

        # ===== Brightness / Contrast / Gamma =====
        bcg_grp = QGroupBox("Brightness / Contrast / Gamma")
        bcg = QVBoxLayout(bcg_grp)
        bcg.setSpacing(6)

        row1 = QHBoxLayout()
        row1.setSpacing(10)

        def _mk_slider(minv, maxv, val, tick_int=10):
            s = QSlider(Qt.Horizontal)
            s.setMinimum(minv)
            s.setMaximum(maxv)
            s.setValue(val)
            s.setTickInterval(tick_int)
            s.setSingleStep(1)
            s.setPageStep(1)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            return s

        # Internally we expose normalized values via paramsChanged
        self.sld_b = _mk_slider(-50, 50, 0)     # → b in −0.5..0.5
        self.sld_c = _mk_slider(1, 300, 100)    # → c in 0.01..3.00
        self.sld_g = _mk_slider(5, 300, 100)    # → g in 0.05..3.00

        row1.addWidget(QLabel("B"))
        row1.addWidget(self.sld_b)
        row1.addSpacing(8)
        row1.addWidget(QLabel("C"))
        row1.addWidget(self.sld_c)
        row1.addSpacing(8)
        row1.addWidget(QLabel("G"))
        row1.addWidget(self.sld_g)

        self.chk_eq = QCheckBox("Histogram Equalization (preview only)")
        bcg.addLayout(row1)
        bcg.addWidget(self.chk_eq)

        # ===== Crop ROI =====
        crop_grp = QGroupBox("Crop ROI")
        crop = QVBoxLayout(crop_grp)
        crop.setSpacing(6)

        title = QLabel("Draw Crop ROI Tools")
        title.setStyleSheet("font-weight: 600;")
        crop.addWidget(title)

        tools = QHBoxLayout()
        tools.setSpacing(10)

        def _mk_tool(name: str) -> QPushButton:
            b = QPushButton(name)
            b.setCheckable(True)
            b.setMinimumWidth(96)
            b.setStyleSheet(
                "QPushButton { padding:6px 10px; }"
                "QPushButton:checked { background:#2b7cff; color:white; }"
            )
            return b

        self.btn_rect = _mk_tool("Rect")
        self.btn_poly = _mk_tool("Polygon")
        self.btn_free = _mk_tool("Freehand")

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(False)  # click again to unselect
        for b in (self.btn_rect, self.btn_poly, self.btn_free):
            self._tool_group.addButton(b)

        def _on_tool(btn: QPushButton, name: str):
            if btn.isChecked():
                # enforce radio behavior manually
                for other in (self.btn_rect, self.btn_poly, self.btn_free):
                    if other is not btn:
                        other.blockSignals(True)
                        other.setChecked(False)
                        other.blockSignals(False)
                self.toolSelected.emit(name)
                self.cropModeToggled.emit(True)
            else:
                self.toolSelected.emit("")
                self.cropModeToggled.emit(False)

        self.btn_rect.clicked.connect(lambda: _on_tool(self.btn_rect, "rect"))
        self.btn_poly.clicked.connect(lambda: _on_tool(self.btn_poly, "poly"))
        self.btn_free.clicked.connect(lambda: _on_tool(self.btn_free, "free"))

        tools.addStretch(1)
        tools.addWidget(self.btn_rect)
        tools.addWidget(self.btn_poly)
        tools.addWidget(self.btn_free)
        tools.addStretch(1)
        crop.addLayout(tools)

        act2 = QHBoxLayout()
        act2.setSpacing(8)
        self.btn_apply_crop = QPushButton("Apply Crop")
        self.btn_clear_crop = QPushButton("Clear Crop")
        act2.addWidget(self.btn_apply_crop)
        act2.addWidget(self.btn_clear_crop)
        act2.addStretch(1)
        crop.addLayout(act2)

        # ===== Apply / Reset to entire stack =====
        act1 = QHBoxLayout()
        act1.setSpacing(8)
        self.btn_apply_all = QPushButton("Apply to Stack")
        self.btn_reset = QPushButton("Reset to Original")
        act1.addWidget(self.btn_apply_all)
        act1.addWidget(self.btn_reset)
        act1.addStretch(1)

        # Assemble
        root.addWidget(bcg_grp)
        root.addWidget(crop_grp)
        root.addLayout(act1)

        # Signal wiring
        def _emit_params():
            b = self.sld_b.value() / 100.0
            c = self.sld_c.value() / 100.0
            g = self.sld_g.value() / 100.0
            eq = self.chk_eq.isChecked()
            self.paramsChanged.emit(b, c, g, eq)

        for s in (self.sld_b, self.sld_c, self.sld_g):
            s.valueChanged.connect(lambda _=None: _emit_params())
        self.chk_eq.toggled.connect(lambda _=None: _emit_params())

        self.btn_apply_all.clicked.connect(self.applyClicked)
        self.btn_reset.clicked.connect(self.resetClicked)
        self.btn_apply_crop.clicked.connect(self.applyCropClicked)
        self.btn_clear_crop.clicked.connect(self.clearCropClicked)

        _emit_params()

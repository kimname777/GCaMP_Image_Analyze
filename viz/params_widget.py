from __future__ import annotations
"""
ParamsWidget
------------
Compact control panel for segmentation and deconvolution parameters.

Signals
-------
applyClicked(dict)        : emit the full parameter dict
runPipelineClicked()      : ask the parent to run the pipeline
paramsChanged(...)        : optional live-preview hook (not required)

Notes
-----
Segmentation backends are aligned with `segmentation.api.segment_cells`:
    ['suite2p', 'cnmf', 'threshold']
"""

from typing import Dict

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QFrame, QSizePolicy
)


class ParamsWidget(QWidget):
    # Outgoing signals
    applyClicked = Signal(dict)
    runPipelineClicked = Signal()
    paramsChanged = Signal(float, float, float, bool)  # (preprocess preview hook)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ---------------- Segmentation ----------------
        seg_group = QGroupBox("Segmentation Parameters")
        seg_lay = QFormLayout(seg_group)
        seg_lay.setLabelAlignment(Qt.AlignRight)

        self.cmb_backend = QComboBox()
        self.cmb_backend.addItems(["suite2p", "cnmf", "threshold"])
        self.cmb_backend.setCurrentText("cnmf")

        self.sp_gsig = QSpinBox()
        self.sp_gsig.setRange(1, 99)
        self.sp_gsig.setValue(5)

        self.sp_gsiz = QSpinBox()
        self.sp_gsiz.setRange(3, 255)
        self.sp_gsiz.setValue(21)

        self.sp_snr = QDoubleSpinBox()
        self.sp_snr.setRange(0.0, 20.0)
        self.sp_snr.setDecimals(2)
        self.sp_snr.setValue(2.50)

        self.sp_rval = QDoubleSpinBox()
        self.sp_rval.setRange(0.0, 1.0)
        self.sp_rval.setDecimals(2)
        self.sp_rval.setSingleStep(0.05)
        self.sp_rval.setValue(0.85)

        seg_lay.addRow("Backend", self.cmb_backend)
        seg_lay.addRow("gSig", self.sp_gsig)
        seg_lay.addRow("gSiz", self.sp_gsiz)
        seg_lay.addRow("minSNR", self.sp_snr)
        seg_lay.addRow("rval", self.sp_rval)

        # ---------------- Deconvolution ----------------
        deconv_group = QGroupBox("Deconvolution Parameters")
        deconv_lay = QFormLayout(deconv_group)
        deconv_lay.setLabelAlignment(Qt.AlignRight)

        self.cmb_deconv = QComboBox()
        self.cmb_deconv.addItems(["mlspike", "oasis"])

        self.sp_lambda = QDoubleSpinBox()
        self.sp_lambda.setRange(0.0, 5.0)
        self.sp_lambda.setDecimals(3)
        self.sp_lambda.setValue(0.000)

        deconv_lay.addRow("Backend", self.cmb_deconv)
        deconv_lay.addRow("λ (L1 penalty)", self.sp_lambda)

        # ---------------- Buttons ----------------
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_apply = QPushButton("Apply Parameters")
        self.btn_apply.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_run)

        root.addWidget(seg_group)
        root.addWidget(deconv_group)
        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        hr.setFrameShadow(QFrame.Sunken)
        root.addWidget(hr)
        root.addLayout(btn_row)
        root.addStretch(1)

        # Connections
        self.btn_apply.clicked.connect(self._emit_apply)
        self.btn_run.clicked.connect(self.runPipelineClicked.emit)

    # ---- Parameter collection -------------------------------------------------
    def _collect_params(self) -> Dict:
        return {
            "segmentation": {
                "backend": self.cmb_backend.currentText(),
                "gsig": int(self.sp_gsig.value()),
                "gsiz": int(self.sp_gsiz.value()),
                "minsnr": float(self.sp_snr.value()),
                "rval": float(self.sp_rval.value()),
            },
            "deconvolution": {
                "backend": self.cmb_deconv.currentText(),
                "lambda": float(self.sp_lambda.value()),
            },
        }

    def _emit_apply(self):
        self.applyClicked.emit(self._collect_params())

    def set_defaults_from_cfg(self, cfg):
        """Populate fields from an AppConfig-like object if keys exist."""
        try:
            seg = getattr(cfg, "segmentation", {}) or {}
            self.cmb_backend.setCurrentText(seg.get("backend", "suite2p"))
            self.sp_gsig.setValue(int(seg.get("gsig", 5)))
            self.sp_gsiz.setValue(int(seg.get("gsiz", 21)))
            self.sp_snr.setValue(float(seg.get("minsnr", 2.5)))
            self.sp_rval.setValue(float(seg.get("rval", 0.85)))

            dec = getattr(cfg, "deconvolution", {}) or {}
            self.cmb_deconv.setCurrentText(dec.get("backend", "mlspike"))
            self.sp_lambda.setValue(float(dec.get("lambda", 0.0)))
        except Exception:
            # Be forgiving if cfg is partial
            pass

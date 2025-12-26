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
        self.cmb_backend.addItems(["cnmf", "suite2p", "threshold"])
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

        # --- Backend-specific controls (segmentation) ---
        # Suite2p: diameter, tau
        self.sp_diameter = QSpinBox()
        self.sp_diameter.setRange(1, 999)
        self.sp_diameter.setValue(14)

        self.sp_tau = QDoubleSpinBox()
        self.sp_tau.setRange(0.01, 10.0)
        self.sp_tau.setDecimals(2)
        self.sp_tau.setValue(1.0)

        # CNMF: merge_thr, k, fps (gSig re-used)
        self.sp_merge_thr = QDoubleSpinBox()
        self.sp_merge_thr.setRange(0.0, 1.0)
        self.sp_merge_thr.setDecimals(2)
        self.sp_merge_thr.setValue(0.80)

        self.sp_k = QSpinBox()
        self.sp_k.setRange(1, 2000)
        self.sp_k.setValue(200)

        self.sp_fps = QDoubleSpinBox()
        self.sp_fps.setRange(1.0, 1000.0)
        self.sp_fps.setDecimals(2)
        self.sp_fps.setValue(getattr(self.cfg, "fps", 30.0))

        # Add but default-hide backend specific rows; visibility toggled by _on_seg_backend_changed
        seg_lay.addRow("diameter", self.sp_diameter)
        seg_lay.addRow("tau (s)", self.sp_tau)
        seg_lay.addRow("merge_thr", self.sp_merge_thr)
        seg_lay.addRow("k", self.sp_k)
        seg_lay.addRow("fps", self.sp_fps)

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

        # Backend specific deconv controls
        # MLSpike params
        self.sp_ml_tau = QDoubleSpinBox()
        self.sp_ml_tau.setRange(0.01, 5.0)
        self.sp_ml_tau.setDecimals(3)
        self.sp_ml_tau.setValue(0.3)

        self.sp_ml_a = QDoubleSpinBox()
        self.sp_ml_a.setRange(0.0, 10.0)
        self.sp_ml_a.setDecimals(3)
        self.sp_ml_a.setValue(1.0)

        self.sp_ml_sigma = QDoubleSpinBox()
        self.sp_ml_sigma.setRange(0.0, 1.0)
        self.sp_ml_sigma.setDecimals(4)
        self.sp_ml_sigma.setValue(0.02)

        deconv_lay.addRow("tau (s)", self.sp_ml_tau)
        deconv_lay.addRow("a", self.sp_ml_a)
        deconv_lay.addRow("sigma", self.sp_ml_sigma)

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
        # Dynamic visibility
        self.cmb_backend.currentTextChanged.connect(self._on_seg_backend_changed)
        self.cmb_deconv.currentTextChanged.connect(self._on_deconv_backend_changed)

        # Initialize visibility state
        self._on_seg_backend_changed(self.cmb_backend.currentText())
        self._on_deconv_backend_changed(self.cmb_deconv.currentText())

    # ---- Parameter collection -------------------------------------------------
    def _collect_params(self) -> Dict:
        # Emit both the traditional flat keys (backwards compatible) and a
        # `backend_params` dict containing backend-specific options. The
        # pipeline can later consume cfg.segmentation['backend_params'] or
        # cfg.deconvolution['backend_params'] when wiring params to backends.
        seg_backend = self.cmb_backend.currentText()
        deconv_backend = self.cmb_deconv.currentText()

        seg_backend_params = {}
        if seg_backend == "suite2p":
            seg_backend_params = {"diameter": int(self.sp_diameter.value()), "tau": float(self.sp_tau.value())}
        elif seg_backend == "cnmf":
            seg_backend_params = {
                "gSig": int(self.sp_gsig.value()),
                "merge_thr": float(self.sp_merge_thr.value()),
                "k": int(self.sp_k.value()),
                "fps": float(self.sp_fps.value()),
            }
        else:  # threshold or unknown
            seg_backend_params = {"diameter": int(self.sp_diameter.value())}

        deconv_backend_params = {}
        if deconv_backend == "oasis":
            deconv_backend_params = {"lambda": float(self.sp_lambda.value())}
        elif deconv_backend == "mlspike":
            deconv_backend_params = {
                "tau": float(self.sp_ml_tau.value()),
                "a": float(self.sp_ml_a.value()),
                "sigma": float(self.sp_ml_sigma.value()),
            }

        return {
            "segmentation": {
                "backend": seg_backend,
                # keep legacy flat keys for backward compatibility
                "gsig": int(self.sp_gsig.value()),
                "gsiz": int(self.sp_gsiz.value()),
                "minsnr": float(self.sp_snr.value()),
                "rval": float(self.sp_rval.value()),
                # explicit backend-specific params
                "backend_params": seg_backend_params,
            },
            "deconvolution": {
                "backend": deconv_backend,
                "lambda": float(self.sp_lambda.value()),
                "backend_params": deconv_backend_params,
            },
        }

    # ---- Dynamic visibility handlers --------------------------------------
    def _on_seg_backend_changed(self, name: str):
        name = (name or "").lower()
        # Show/enable only relevant fields
        is_suite2p = name == "suite2p"
        is_cnmf = name == "cnmf"
        is_threshold = name == "threshold"

        self.sp_diameter.setEnabled(is_suite2p or is_threshold or is_cnmf)
        self.sp_tau.setEnabled(is_suite2p)

        # CNMF-specific
        self.sp_merge_thr.setEnabled(is_cnmf)
        self.sp_k.setEnabled(is_cnmf)
        self.sp_fps.setEnabled(is_cnmf)

    def _on_deconv_backend_changed(self, name: str):
        name = (name or "").lower()
        is_mlspike = name == "mlspike"
        is_oasis = name == "oasis"

        # For OASIS we surface λ; for MLSpike expose its params
        self.sp_lambda.setEnabled(is_oasis)
        self.sp_ml_tau.setEnabled(is_mlspike)
        self.sp_ml_a.setEnabled(is_mlspike)
        self.sp_ml_sigma.setEnabled(is_mlspike)

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

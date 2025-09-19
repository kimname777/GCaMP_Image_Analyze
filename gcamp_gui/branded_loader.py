from __future__ import annotations

from PySide6.QtWidgets import QDialog
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QFontMetrics


class BrandedLoader(QDialog):
    """
    A simple branded splash dialog:
      * Full-bleed background image (cover)
      * Subtle dark overlay
      * Title, subtitle, version text
      * Thin white progress bar at the bottom (animated)
    """
    def __init__(self, parent=None, image_path: str = "",
                 title: str = "GCaMP Image Analysis",
                 subtitle: str = "Made by Bon-lab",
                 version: str = "v1.0"):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        self._pix = QPixmap(image_path)
        self._title, self._subtitle, self._version = title, subtitle, version
        self._progress = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self.resize(1000, 380)  # adjust to taste
        self._timer.start(33)   # ~30 fps animation

    # API compatible with app.py's wrapper
    def start(self, text: str | None = None) -> None:
        if text:
            self._subtitle = text
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self.accept()  # close

    # --- internals ---
    def _tick(self) -> None:
        self._progress = (self._progress + 0.012) % 1.0
        self.update()

    def paintEvent(self, _):  # noqa: D401 (Qt override)
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)

        # 1) Background image (cover)
        if not self._pix.isNull():
            scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            sx = (scaled.width() - self.width()) // 2
            sy = (scaled.height() - self.height()) // 2
            p.drawPixmap(0, 0, scaled, sx, sy, self.width(), self.height())

        # 2) Dim overlay
        p.fillRect(self.rect(), QColor(0, 0, 0, 110))

        # 3) Texts (title / version / subtitle)
        margin = 24
        base_y = self.height() - 86

        title_font = QFont("Segoe UI", 36, QFont.Black)
        p.setFont(title_font)
        p.setPen(QColor(255, 255, 255))
        p.drawText(margin, base_y, self._title)

        fm = QFontMetrics(title_font)
        title_w = fm.horizontalAdvance(self._title)

        ver_font = QFont("Segoe UI", 16, QFont.DemiBold)
        p.setFont(ver_font)
        p.drawText(margin + title_w + 12, base_y, self._version)

        sub_font = QFont("Segoe UI", 14, QFont.DemiBold)
        p.setFont(sub_font)
        p.drawText(margin, base_y + 30, self._subtitle)

        # 4) Thin white bar + moving chunk
        bar_h = 8
        y = self.height() - bar_h
        p.fillRect(0, y, self.width(), bar_h, QColor(255, 255, 255))
        chunk_w = int(self.width() * 0.18)
        x = int((self.width() + chunk_w) * self._progress) - chunk_w
        p.fillRect(x, y, chunk_w, bar_h, QColor(255, 255, 255, 180))

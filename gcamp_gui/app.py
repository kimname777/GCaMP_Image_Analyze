from __future__ import annotations

import os
import sys
import platform
from pathlib import Path

from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPixmap


# --- Clean up noisy Qt plugin env vars (helps when packaging with PyInstaller) ---
for _v in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
           "QT_PLUGIN_PATHS", "QT_PLUGIN_PATH_OVERRIDE"):
    os.environ.pop(_v, None)

# --- Pick a sane default QPA platform per OS unless user already set one ---
os.environ.setdefault(
    "QT_QPA_PLATFORM",
    {"Windows": "windows", "Linux": "xcb", "Darwin": "cocoa"}.get(platform.system(), "xcb")
)


# --- Optional branded splash (falls back to vanilla QSplashScreen) ---
try:
    from gcamp_gui.branded_loader import BrandedLoader
    HAVE_BRANDED = True
except Exception:
    BrandedLoader = None  # type: ignore[assignment]
    HAVE_BRANDED = False


def _resource_path(rel: str) -> Path:
    """
    Resolve a resource path that works both in dev (source tree) and in a frozen build.

    Dev: project root
    Frozen (one-dir): directory of the executable
    """
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.argv[0]).resolve().parent
        candidates += [exe_dir, exe_dir / "_internal"]
    else:
        # gcamp_gui/ -> project root
        candidates += [Path(__file__).resolve().parent.parent]

    for base in candidates:
        p = base / rel
        if p.exists():
            return p

    # If we couldn't find it, return the first candidate path (caller may check exists()).
    return candidates[0] / rel


def _show_startup_splash(app: QApplication):
    """
    Show a startup splash and return an object with a .stop() method.
    We use a thin wrapper so caller can always call splash.stop() safely.
    """
    img_path = _resource_path("resources/splash.png")
    print(f"[splash] try: {img_path}  exists={img_path.exists()}", file=sys.stderr, flush=True)

    # 1) Branded splash (if image exists and class available)
    if HAVE_BRANDED and img_path.exists():
        splash = BrandedLoader(
            None, str(img_path),
            title="GCaMP Image Analysis",
            subtitle="Initializing…",
            version="v1.0",
        )
        # Center on primary screen
        g = app.primaryScreen().availableGeometry().center()
        splash.move(g.x() - splash.width() // 2, g.y() - splash.height() // 2)
        splash.start("Initializing…")
        app.processEvents()
        return splash

    # 2) Fallback: plain QSplashScreen (blank if the image is missing)
    pm = QPixmap(str(img_path)) if img_path.exists() else QPixmap()
    splash = QSplashScreen(pm)
    splash.show()
    app.processEvents()

    class _Wrap:
        def stop(self_inner):  # noqa: N802 (Qt-style)
            splash.finish(None)

    return _Wrap()


def main() -> None:
    app = QApplication(sys.argv)

    # 1) Splash while we build the main window
    splash = _show_startup_splash(app)

    # 2) Create and show main window
    from gcamp_gui.main_window import MainWindow
    win = MainWindow()
    win.show()

    # 3) Close splash slightly later for a clean transition
    QTimer.singleShot(120, splash.stop)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

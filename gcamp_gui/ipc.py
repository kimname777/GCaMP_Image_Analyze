from __future__ import annotations

import json
import subprocess
import sys
import pathlib


def _base_dir() -> pathlib.Path:
    """
    Return the base directory for bundled helpers.

    In frozen builds, PyInstaller exposes a temporary unpack dir via _MEIPASS.
    Fall back to the directory of this file in dev.
    """
    return pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(__file__).parent)).resolve()


def call_worker(payload: dict) -> dict:
    """
    Call an external worker binary with a JSON payload.

    Notes
    -----
    * This is a thin synchronous wrapper (subprocess.run).
    * The worker is expected to read JSON from stdin and write JSON to stdout.
    * Adjust the binary name if you ship a platform-specific worker.
    """
    worker = _base_dir() / ("Worker.exe" if sys.platform.startswith("win") else "worker")
    proc = subprocess.run(
        [str(worker)],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    out = proc.stdout.decode("utf-8", "ignore") or "{}"
    try:
        return json.loads(out)
    except Exception:
        return {
            "ok": False,
            "error": "Worker JSON parse failed",
            "raw": out,
            "stderr": proc.stderr.decode("utf-8", "ignore"),
        }

"""Start the program as a double-click does, off screen, and say who loaded what.

    python startup_check.py <folder for the settings and the log> [file to open]

Prints one line of JSON: the heavy modules the window's own thread imported (there should
be none: PyTorch takes seconds to load, and the window must not wait for it), the state
of the engine once it is loaded and, with a file, whether its picture came.
"""
import json
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HEAVY = ("torch", "av", "imagecodecs", "tifffile")
imported_by = {}


class Watch:
    """Notes which thread asks for each heavy module first. Finds nothing itself."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name in HEAVY and name not in imported_by:
            thread = threading.current_thread()
            imported_by[name] = "window" if thread is threading.main_thread() else thread.name
        return None


def main() -> int:
    folder, files = sys.argv[1], sys.argv[2:]
    sys.meta_path.insert(0, Watch())

    from PySide6.QtCore import QTimer

    from hdr_blaster.ui import app as ui_app

    # The log and the settings of this run are kept in the folder given, away from the
    # user's. (The settings by a file of their own, handed to main(): QSettings("org", "app")
    # goes to the registry whatever QSettings.setDefaultFormat says.)
    real_logging = ui_app.configure_logging
    ui_app.configure_logging = lambda path: real_logging(os.path.join(folder, "hdr_blaster.log"))
    app = ui_app.create_application(["hdr-blaster"])
    seen = {"window": None, "cuda": None, "picture": False, "failed": "", "closing": False}

    def look():
        """In the window's thread: it alone may touch the window."""
        window = seen["window"]
        if window is None:
            for widget in app.topLevelWidgets():
                if type(widget).__name__ == "MainWindow" and widget.isVisible():
                    seen["window"] = widget
                    widget.backend.ready.connect(ready)
                    widget.backend.preview_failed.connect(failed)
            return
        if files and not seen["picture"] and window.viewer.has_picture():
            seen["picture"] = True
        done = seen["failed"] or (seen["picture"] if files else seen["cuda"] is not None)
        if done and not seen["closing"]:
            seen["closing"] = True
            QTimer.singleShot(100, window.close)          # let the panel hear the engine first

    # These two are called by the background thread: they only take note.
    def ready(cuda):
        seen["cuda"] = bool(cuda)

    def failed(request, message):
        seen["failed"] = message or "failed"

    poll = QTimer()
    poll.timeout.connect(look)
    poll.start(5)
    QTimer.singleShot(120000, lambda: app.exit(3))
    code = ui_app.main(["hdr-blaster"] + files, settings_path=os.path.join(folder, "settings.ini"))

    import torch

    window = seen["window"]
    print(json.dumps({
        "exit": code,
        "window_shown": window is not None,
        "imported_by": imported_by,
        "engine_reported": seen["cuda"],
        "cuda": torch.cuda.is_available(),
        "deterministic": bool(torch.backends.cudnn.deterministic),
        "benchmark": bool(torch.backends.cudnn.benchmark),
        "gpu_switch_enabled": window.panel.use_gpu.isEnabled() if window is not None else None,
        "log_written": os.path.isfile(os.path.join(folder, "hdr_blaster.log")),
        "settings": window._settings.fileName() if window is not None else None,
        "picture": seen["picture"],
        "failed": seen["failed"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())

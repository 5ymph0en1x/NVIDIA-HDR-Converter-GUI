"""Where the bundled assets live (model weights, training/model.py, icon)."""
import os
import sys


def app_data_dir() -> str:
    """Base directory for bundled app data.

    A frozen one-file build extracts bundled data to ``sys._MEIPASS``; read the
    assets from there so the .exe is fully self-contained. From source, it is
    the project root (the folder that contains the ``hdr_blaster`` package).
    """
    if getattr(sys, 'frozen', False):  # PyInstaller bundle
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

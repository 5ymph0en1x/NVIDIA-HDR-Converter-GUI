"""NVIDIA HDR Converter GUI (HDR Blaster): double-click, or ``python NHC.py [files...]``.

Opens the window. The headless commands of the engine are available too:
    python NHC.py --convert-video recording.mp4
    python NHC.py --convert capture.jxr
    python NHC.py --version
"""
import sys

from hdr_blaster.__main__ import run

if __name__ == "__main__":
    sys.exit(run())

"""Does any control get less room than its content needs? Run as a script (so that a display
scale can be set through QT_SCALE_FACTOR before Qt starts) or imported by the tests.

    python ui_layout_check.py <video> [<capture>]      prints 'layout ok' or the offenders
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir("C:/Windows/Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def wait(condition, timeout=120.0) -> bool:
    from PySide6.QtCore import QCoreApplication

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def settle(seconds=0.3) -> None:
    wait(lambda: False, seconds)


def cramped(window) -> list:
    """Visible buttons, labels and inputs that are smaller than what they show."""
    from PySide6.QtWidgets import (QAbstractButton, QAbstractScrollArea, QComboBox, QDoubleSpinBox, QLabel,
                                   QSizePolicy, QWidget)

    offenders = []
    kinds = (QAbstractButton, QLabel, QComboBox, QDoubleSpinBox)
    for widget in window.findChildren(QWidget):
        if not isinstance(widget, kinds):
            continue
        if not widget.isVisible() or widget.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored:
            continue                                   # 'Ignored' labels elide their text on purpose
        if isinstance(widget, QLabel) and widget.wordWrap():
            continue
        need = widget.minimumSizeHint() if isinstance(widget, QLabel) else widget.sizeHint()
        if widget.width() < need.width() - 1 or widget.height() < need.height() - 1:
            text = widget.text() if hasattr(widget, "text") else type(widget).__name__
            offenders.append(f"{type(widget).__name__} '{text}': {widget.width()}x{widget.height()} "
                             f"for {need.width()}x{need.height()}")
        # and nothing may stick out of the window
        corner = widget.mapTo(window, widget.rect().bottomRight())
        scrolled, parent = False, widget.parentWidget()
        while parent is not None and not scrolled:
            scrolled = isinstance(parent, QAbstractScrollArea)     # reachable by scrolling: not cut off
            parent = parent.parentWidget()
        if corner.x() > window.width() or (corner.y() > window.height() and not scrolled):
            offenders.append(f"{type(widget).__name__} '{getattr(widget, 'text', lambda: '')()}' leaves the window")
    return offenders


def check(window, sizes) -> list:
    problems = []
    for width, height in sizes:
        window.resize(width, height)
        settle()
        for state in ("closed", "open"):
            window.toggle_advanced(state == "open")
            settle(0.35)
            problems += [f"{width}x{height}, panel {state}: {p}" for p in cramped(window)]
            bar_top = window.bar.mapTo(window, window.bar.rect().topLeft()).y()
            panel_bottom = window.panel.mapTo(window, window.panel.rect().bottomLeft()).y()
            if state == "open" and panel_bottom > bar_top:
                problems.append(f"{width}x{height}: the advanced panel covers the action bar")
        window.toggle_advanced(False)
        settle(0.35)
    return problems


def main() -> int:
    from hdr_blaster.ui.app import create_application

    app = create_application([])
    from hdr_blaster.ui.window import MainWindow

    window = MainWindow(os.path.join(tempfile.mkdtemp(), "settings.ini"))
    window.show()
    window.add_paths(sys.argv[1:])
    if not wait(lambda: window.viewer.has_picture(), 180):
        print("no preview appeared")
        return 2
    window.notify("holiday.jpg is not an HDR capture or recording. JXR screenshots and HDR10 videos can be converted.")
    minimum = (window.minimumWidth(), window.minimumHeight())
    problems = check(window, [minimum, (1280, 800), (1920, 1080)])
    window.close()
    del app
    if problems:
        print("\n".join(problems))
        return 1
    print("layout ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

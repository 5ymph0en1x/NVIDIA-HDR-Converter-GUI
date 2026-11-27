"""Check, on the real screen, that the picture area is presented in HDR.

    python tools/check_hdr_surface.py                 the surface alone, with a known picture
    python tools/check_hdr_surface.py CAPTURE.jxr     the whole window, with a capture

A window opens for a few seconds. What it presents is read back from its display buffer
(scRGB: 1.0 = 80 nits) and compared with what it was given. Prints "presented: ok" and
exits with 0 when everything matches. Needs a display with HDR switched on.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QImage


def wait(condition, timeout=60.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def check_surface() -> bool:
    from hdr_blaster.ui.app import create_application
    from hdr_blaster.ui.hdr_surface import HdrSurface, hdr_output_available

    app = create_application([])
    print("HDR output available:", hdr_output_available())
    surface = HdrSurface()
    surface.setTitle("HDR Blaster, checking the HDR surface")
    surface.resize(640, 360)
    failures = []
    surface.failed.connect(failures.append)
    surface.show()
    if not wait(lambda: surface.presents_hdr() or failures, 20):
        print("the surface did not start", failures)
        return False
    if failures:
        print("no HDR here:", failures[0])
        return False
    ratio = surface.devicePixelRatio()
    width, height = round(640 * ratio), round(360 * ratio)
    ramp = np.linspace(0.0, 12.5, width, dtype=np.float32)               # 0 to 1000 nits
    before = np.ones((height, width, 4), dtype=np.float16)
    before[..., :3] = ramp[None, :, None]
    after = np.ones((height, width, 4), dtype=np.float16)
    after[..., :3] = 3.0                                                 # 240 nits everywhere
    overlay = QImage(width, height, QImage.Format_RGBA8888_Premultiplied)
    overlay.fill(0)
    handle = QImage(8, 8, QImage.Format_RGBA8888_Premultiplied)
    handle.fill(0)
    surface.set_layout((0, 0, width, height), 480.0, 0.0, (1.0, 1.0, 1.0, 0.0), (0.0, 0.0, 0.0))
    surface.set_pictures(before, after)
    surface.set_overlay(overlay)
    surface.set_handle(handle, 8.0)
    surface.set_split(0.5)
    wait(lambda: False, 0.5)
    shown = surface.read_back()
    surface.close()
    if shown is None:
        print("nothing could be read back")
        return False
    shown = shown.astype(np.float32)
    row = shown[shown.shape[0] // 2, :, 0]
    left = slice(4, shown.shape[1] // 2 - 4)
    right = slice(shown.shape[1] // 2 + 4, shown.shape[1] - 4)
    expected = ramp[: shown.shape[1]] if shown.shape[1] == width else np.linspace(0.0, 12.5, shown.shape[1])
    error = float(np.abs(row[left] - expected[left]).max())
    print(f"surface {shown.shape[1]}x{shown.shape[0]}: left half follows the ramp to {error:.3f}; "
          f"brightest value presented on the left {row[left].max():.2f} = {row[left].max() * 80:.0f} nits; "
          f"right half {row[right].min():.3f} to {row[right].max():.3f} (given 3.000)")
    return error < 0.05 and abs(float(row[right].mean()) - 3.0) < 0.01 and float(row[left].max()) > 6.0


def check_window(capture: str) -> bool:
    import tempfile

    import torch

    from hdr_blaster.engine.constants import ENGINE_UNIT_NITS
    from hdr_blaster.io.jxr import OptimizedJXRLoader
    from hdr_blaster.ui.app import create_application
    from hdr_blaster.ui.window import MainWindow

    app = create_application([])
    window = MainWindow(os.path.join(tempfile.mkdtemp(), "settings.ini"))
    window.showMaximized()
    window.add_paths([capture])
    viewer = window.viewer
    if not wait(lambda: viewer.has_picture() and window._facts is not None, 120):
        print("no preview")
        return False
    wait(lambda: False, 2.0)
    print("the viewer shows HDR:", viewer.shows_hdr(), "| source label:", viewer._source_tag)
    if not viewer.shows_hdr():
        window.close()
        return False
    viewer.set_split(1.0)                                                 # the source over the whole picture
    wait(lambda: False, 0.3)
    shown = viewer._hdr.read_back()
    rect, ratio = viewer.picture_rect(), viewer.devicePixelRatioF()
    x0, y0 = round(rect.left() * ratio), round(rect.top() * ratio)
    w, h = round(rect.width() * ratio), round(rect.height() * ratio)
    viewer.set_split(0.0)                                                 # then the result
    wait(lambda: False, 0.3)
    result = viewer._hdr.read_back()
    white = float(window._facts["white_nits"])
    # With the queue open the surface starts where the queue ends: the picture must be
    # presented where the viewer says it is, and the queue must not be under the surface.
    window.toggle_queue(True)
    wait(lambda: False, 0.6)
    wait(lambda: viewer._made_for(viewer._after, viewer.picture_rect()), 60)
    viewer.set_split(1.0)
    wait(lambda: False, 0.5)
    beside = viewer._hdr.read_back()
    inset, box = viewer._inset_left, viewer._hdr_box.geometry()
    place = viewer.picture_rect()
    queue_right = window.queue_panel.x() + window.queue_panel.width()
    window.close()
    if shown is None or result is None:
        print("nothing could be read back")
        return False
    picture = shown[y0:y0 + h, x0:x0 + w, :3].astype(np.float32)
    linear, _, _ = OptimizedJXRLoader(torch.device("cpu")).load_jxr(capture)
    file_nits = linear[0].clamp(min=0).amax(0).numpy() * ENGINE_UNIT_NITS
    inner = picture[60:-110, 60:-60]                                      # clear of the labels and the corners
    shown_nits = inner.max(-1) * 80.0
    print(f"picture {w}x{h} at {x0},{y0} on a surface {shown.shape[1]}x{shown.shape[0]}; SDR white {white:.0f} nits")
    print(f"   the file peaks at {file_nits.max():.0f} nits; the source is presented up to {shown_nits.max():.0f} nits "
          f"({(shown_nits > white).mean():.1%} of it above SDR white, which an SDR window cuts)")
    after = result[y0:y0 + h, x0:x0 + w, :3].astype(np.float32)[60:-110, 60:-60].max(-1) * 80.0
    print(f"   the SDR result is presented up to {after.max():.0f} nits")
    dark = shown_nits < 0.5 * white
    agreement = float(np.median(after[dark] / np.maximum(shown_nits[dark], 1e-3)))
    print(f"   below half of SDR white, result / source = {agreement:.3f} (1.000: the same picture on both sides)")
    ok = (shown_nits.max() > 0.9 * min(file_nits.max(), 10000.0) and after.max() <= white * 1.01
          and abs(agreement - 1.0) < 0.03)
    if beside is None:
        print("nothing could be read back with the queue open")
        return False
    bx, by = round((place.left() - inset) * ratio), round(place.top() * ratio)
    bw, bh = round(place.width() * ratio), round(place.height() * ratio)
    inside = beside[by:by + bh, bx:bx + bw, :3].astype(np.float32)[60:-110, 60:-60].max(-1) * 80.0
    margin = beside[by + bh // 4:by + 3 * bh // 4, max(0, bx - 10):max(1, bx - 4), :3].astype(np.float32).max() * 80.0
    print(f"   queue open: the surface starts at {box.x()} (the queue ends at {queue_right}), is {beside.shape[1]} wide "
          f"for a box of {round(box.width() * ratio)}; picture {bw}x{bh} at {bx},{by}, presented up to "
          f"{inside.max():.0f} nits; just left of it {margin:.0f} nits")
    beside_ok = (inset > 0 and box.x() == inset == queue_right and abs(beside.shape[1] - box.width() * ratio) <= 2
                 and inside.max() > 0.8 * shown_nits.max() and margin < 0.1 * white)
    return ok and beside_ok


if __name__ == "__main__":
    ok = check_window(sys.argv[1]) if len(sys.argv) > 1 else check_surface()
    print("presented: ok" if ok else "presented: NOT as expected")
    sys.exit(0 if ok else 1)

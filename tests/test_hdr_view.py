"""The source shown in real HDR.

Reported by the author, with Windows' own viewer next to ours: the "HDR source" of the
window did not look like the capture. It could not: a window like ours is SDR, its white
is the brightest thing it can show, and everything brighter in the capture (the sun, the
sky around it) was cut to that white. Windows' viewer presents the capture in HDR.

So, on a screen with HDR switched on, the picture area is presented in HDR too (scRGB:
1.0 stands for 80 nits, and values go as far above as the picture does):

* on the left, the capture's own values, untouched: what the game showed;
* on the right, the SDR result at the luminance an SDR window gives it on this screen.

This file tests the pictures handed to that surface. The surface itself needs a real
screen: see ``test_the_surface_presents_what_it_is_given`` (run with HDR_BLASTER_ONSCREEN=1).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SAMPLES
from hdr_blaster import preview
from hdr_blaster.engine.constants import ENGINE_UNIT_NITS
from hdr_blaster.engine.pipeline import RenderSettings
from hdr_blaster.video.job import make_engine

ROOT = Path(__file__).resolve().parents[1]
WHITE = 480.0


@pytest.fixture(scope="module")
def video_engine(device):
    return make_engine(device)


def scene(device, nits, size=64):
    """Flat grey patches side by side, engine units."""
    row = [torch.full((1, 3, size, size), float(v), device=device) for v in nits]
    return torch.cat(row, dim=3) / ENGINE_UNIT_NITS


def source_of(linear, white=WHITE, size=None):
    height, width = linear.shape[-2:]
    return preview.PreviewSource(kind="image", path="patches", linear=linear, width=width, height=height,
                                 size=size or (height, width), white_nits=white)


def test_the_source_is_handed_over_at_its_own_luminance(device, video_engine):
    """1.0 is 80 nits. A tone of 40 nits is 0.5, SDR white on this screen (480 nits) is 6,
    and the 1000 nits of a sun are 12.5: nothing is cut."""
    source = source_of(scene(device, [40.0, 480.0, 1000.0, 4000.0]))
    after = preview.render_preview(video_engine, source, RenderSettings(method="faithful", edge_strength=0.0))
    before16, after16 = preview.hdr_pictures(source, after)
    assert before16.dtype == np.float16 and before16.shape == (64, 256, 4) and before16.flags["C_CONTIGUOUS"]
    assert after16.dtype == np.float16 and after16.shape == before16.shape
    centres = before16[32, 32::64, :3].astype(np.float32)
    assert centres[:, 0].tolist() == pytest.approx([0.5, 6.0, 12.5, 50.0], rel=1e-3)
    assert float(before16[..., 3].min()) == 1.0 == float(before16[..., 3].max())


def test_the_result_is_handed_over_at_the_luminance_of_an_sdr_window(device, video_engine):
    """The SDR result never passes the screen's SDR white (6 on a 480-nit screen), and what
    fits under white is the same on both sides of the divider."""
    source = source_of(scene(device, [40.0, 200.0, 1000.0]))
    after = preview.render_preview(video_engine, source, RenderSettings(method="faithful", edge_strength=0.0))
    before16, after16 = preview.hdr_pictures(source, after)
    result = after16[32, 32::64, 0].astype(np.float32)
    assert float(after16[..., :3].max()) == pytest.approx(WHITE / 80.0, rel=2e-3), "the peak lands on SDR white"
    assert result[0] == pytest.approx(0.5, rel=0.01) and result[1] == pytest.approx(2.5, rel=0.01)
    assert before16[32, 32, 0] == pytest.approx(result[0], rel=0.01), "40 nits on the left, 40 nits on the right"
    other = source_of(scene(device, [40.0, 200.0, 1000.0]), white=203.0)
    after = preview.render_preview(video_engine, other, RenderSettings(method="faithful", edge_strength=0.0))
    assert float(preview.hdr_pictures(other, after)[1][..., :3].max()) == pytest.approx(203.0 / 80.0, rel=2e-3)


def test_the_hdr_source_is_brought_to_its_place_in_one_resampling(device):
    lights = torch.full((1, 3, 64, 64), 2.0 / ENGINE_UNIT_NITS, device=device)
    lights[..., ::4] = 4000.0 / ENGINE_UNIT_NITS                     # one column in four at 4000 nits
    reduced = preview.for_display_hdr(lights * (ENGINE_UNIT_NITS / 80.0), (16, 16))
    assert reduced.shape == (1, 3, 16, 16)
    # In HDR nothing is cut, so light is simply averaged: a quarter of 4000 nits, plus the rest.
    assert float(reduced[..., 4:-4, 4:-4].mean()) * 80.0 == pytest.approx(0.25 * 4000.0 + 0.75 * 2.0, rel=0.02)
    same = lights * (ENGINE_UNIT_NITS / 80.0)
    assert preview.for_display_hdr(same, (64, 64)) is same
    larger = preview.for_display_hdr(same, (96, 96))
    assert larger.shape == (1, 3, 96, 96) and float(larger.min()) >= 0.0
    assert float(larger.max()) * 80.0 == pytest.approx(4000.0, rel=0.01), "no overshoot above the light itself"


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_a_capture_shown_at_its_size_is_handed_over_untouched(device, video_engine):
    """Shown at the size of the file, the left picture is the file's own numbers."""
    source = preview.load_image(SAMPLES[0], device, box=(2560, 1440), white_nits=WHITE)
    after = preview.render_preview(video_engine, source, RenderSettings(method="faithful", white_nits=WHITE))
    before16, after16 = preview.hdr_pictures(source, after)
    expected = (source.linear[0] * (ENGINE_UNIT_NITS / 80.0)).permute(1, 2, 0).half().cpu().numpy()
    assert np.array_equal(before16[..., :3], expected)
    assert float(before16[..., :3].max()) * 80.0 > 900.0, "the highlights are all there"
    assert float(after16[..., :3].max()) <= WHITE / 80.0 * 1.002


def test_a_video_frame_is_handed_over_in_hdr_too(device, video_engine, tmp_path):
    import video_clips as clips

    path = tmp_path / "game.mp4"
    pts = clips.make_clip(path, frames=12, audio_streams=0)
    source = preview.load_video_frame(path, pts[5] / 90000, device, max_side=4096, white_nits=400.0)
    after = preview.render_preview(video_engine, source, RenderSettings(method="faithful", edge_strength=0.0))
    before16, after16 = preview.hdr_pictures(source, after)
    assert before16.shape == (144, 256, 4) == after16.shape
    assert float(before16[..., :3].max()) * 80.0 == pytest.approx(source.peak_nits, rel=0.25), "its lights, uncut"
    assert float(before16[..., :3].max()) * 80.0 > 500.0
    assert float(after16[..., :3].max()) <= 400.0 / 80.0 * 1.002


def test_the_screen_asked_about_is_the_one_the_window_is_on(monkeypatch):
    """With two screens, one in HDR and one not: on the screen without HDR the window keeps
    its SDR view (scRGB values meant for an HDR screen would come out far too bright there)."""
    from hdr_blaster import display

    first, second = "\\\\.\\DISPLAY1", "\\\\.\\DISPLAY2"
    monkeypatch.setattr(display.sys, "platform", "win32")
    monkeypatch.setattr(display, "_windows_hdr_displays", lambda: [(second, 300.0)])
    assert display.sdr_white_nits(second) == (300.0, "display")
    assert display.sdr_white_nits(first) == (display.REFERENCE_WHITE_NITS, "reference"), "that screen is not in HDR"
    assert display.sdr_white_nits() == (300.0, "display"), "no screen named: the first one in HDR answers"
    assert display.sdr_white_nits("some other name") == (300.0, "display")
    monkeypatch.setattr(display, "_windows_hdr_displays", lambda: [])
    assert display.sdr_white_nits(second) == (display.REFERENCE_WHITE_NITS, "reference")
    assert display.display_of_window(0) == ""


def test_a_file_dropped_on_the_surface_reaches_the_window(tmp_path):
    """The surface lies over the picture: a file dragged there lands on it, not on the
    window. It must pass the file on, or dropping a second file would do nothing."""
    code = (
        "import os, sys; os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        "from PySide6.QtCore import QCoreApplication, QMimeData, QPoint, QPointF, Qt, QUrl\n"
        "from PySide6.QtGui import QDragEnterEvent, QDropEvent\n"
        "from hdr_blaster.ui.app import create_application\n"
        "from hdr_blaster.ui.hdr_surface import HdrSurface\n"
        "from hdr_blaster.ui.window import MainWindow\n"
        "app = create_application([])\n"
        "window = MainWindow(sys.argv[1])\n"
        "added = []\n"
        "window.add_paths = added.extend\n"
        "surface = HdrSurface()\n"
        "surface.drop_target = window\n"
        "data = QMimeData(); data.setUrls([QUrl.fromLocalFile(sys.argv[2])])\n"
        "enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)\n"
        "QCoreApplication.sendEvent(surface, enter)\n"
        "drop = QDropEvent(QPointF(5, 5), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)\n"
        "QCoreApplication.sendEvent(surface, drop)\n"
        "print('accepted', enter.isAccepted(), drop.isAccepted(), len(added))\n"
        "window.close()\n")
    target = tmp_path / "shot.jxr"
    target.write_bytes(b"x")
    done = subprocess.run([sys.executable, "-c", code, str(tmp_path / "settings.ini"), str(target)], cwd=ROOT,
                          capture_output=True, text=True, timeout=180)
    assert "accepted True True 1" in done.stdout, done.stdout + done.stderr


def test_without_a_real_screen_the_window_stays_as_it_was():
    """Off screen (tests, the frozen self-checks) there is no HDR surface: the SDR view is used."""
    code = ("import os; os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
            "from hdr_blaster.ui.app import create_application\n"
            "from hdr_blaster.ui import hdr_surface\n"
            "app = create_application([])\n"
            "print(hdr_surface.hdr_output_available())\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.stdout.strip().endswith("False"), done.stdout + done.stderr


@pytest.mark.skipif(os.environ.get("HDR_BLASTER_ONSCREEN") != "1",
                    reason="needs a real screen with HDR switched on: set HDR_BLASTER_ONSCREEN=1")
def test_the_surface_presents_what_it_is_given():
    """On the real screen: a window is opened for a moment, a picture going up to 1000 nits
    is given to the surface, and what it presented is read back from the display buffer."""
    # other test modules ask Qt for its off-screen platform: this check needs the real one
    real_screen = {name: value for name, value in os.environ.items() if name != "QT_QPA_PLATFORM"}
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "check_hdr_surface.py")], cwd=ROOT,
                          capture_output=True, text=True, timeout=180, env=real_screen)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "presented: ok" in done.stdout, done.stdout

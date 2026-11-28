"""The window, driven off screen: drop, preview, look, convert, cancel, close, layout."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import ui_layout_check as layout                    # sets the offscreen platform before Qt loads
import video_clips as clips
from conftest import SAMPLES
from ui_layout_check import settle, wait

ROOT = Path(__file__).resolve().parents[1]
needs_capture = pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")


@pytest.fixture(scope="module")
def qapp():
    from hdr_blaster.ui.app import create_application

    return create_application([])


@pytest.fixture
def window(qapp, tmp_path):
    from hdr_blaster.ui.window import MainWindow

    win = MainWindow(str(tmp_path / "settings.ini"))
    win.show()
    settle(0.05)
    yield win
    win.close()


@pytest.fixture
def capture(tmp_path):
    target = tmp_path / "files" / "shot.jxr"
    target.parent.mkdir()
    shutil.copy(SAMPLES[0], target)
    return target


@pytest.fixture
def video(tmp_path):
    target = tmp_path / "files" / "clip.mp4"
    target.parent.mkdir(exist_ok=True)
    clips.make_clip(target, frames=24, audio_streams=1)
    return target


def after_bytes(window) -> bytes:
    image = window.viewer._after
    return bytes(image.constBits())[: image.sizeInBytes()]


def assert_displayed_pixels(window, rgb):
    import numpy as np
    import torch
    from hdr_blaster import preview

    image = window.viewer._after
    shown = np.frombuffer(after_bytes(window), np.uint8).reshape(image.height(), image.bytesPerLine())
    shown = shown[:, :image.width() * 3].reshape(image.height(), image.width(), 3)
    tensor = torch.from_numpy(rgb.astype(np.float32) / np.iinfo(rgb.dtype).max).permute(2, 0, 1)[None]
    reference = preview.for_display(tensor.to(window.backend.engines.device_for(window.options)), shown.shape[:2])
    np.testing.assert_array_equal(shown, preview.to_rgb8(reference))


def idle(window) -> bool:
    return not window._converting


# ------------------------------------------------------------------------ empty

def test_the_window_opens_empty_and_invites_a_file(window):
    assert window.windowTitle() == "HDR Blaster"
    assert not window.viewer.has_picture()
    assert window.file_label.text() == "No file yet"
    assert not window.bar.convert.isEnabled() and window.bar.convert.text() == "Convert"
    assert not window.timeline_row.isVisible() and not window.queue_panel.isVisible()
    assert not window.banner.isVisible() and not window.panel.isVisible()
    assert window.bar.looks.look() == "Faithful"
    assert list(window.bar.looks.buttons) == ["Faithful", "Natural", "Punchy", "Golden"]


# --------------------------------------------------------------------- captures

@needs_capture
def test_dropping_a_capture_shows_before_and_after(window, capture):
    window.add_paths([str(capture)])
    assert wait(lambda: window.viewer.has_picture())
    assert window.file_label.text() == "shot.jxr"
    assert window.detail_label.text().startswith("2560 × 1440, JXR capture, peak ")
    assert "Auto chose" not in window.detail_label.text(), "Faithful chooses nothing"
    assert window.bar.convert.isEnabled()
    assert window.bar._full_summary == "JPEG, next to the source"
    assert not window.timeline_row.isVisible(), "no timeline for a still picture"
    assert window.viewer._note == "", "a capture's preview is not provisional"
    assert window.viewer._source_tag.startswith("HDR source, white at ") and window.viewer._source_tag.endswith(" nits")
    assert "peak " in window.detail_label.text()
    assert window.viewer._before.size() == window.viewer._after.size()


@needs_capture
def test_changing_the_look_updates_the_picture_and_the_last_one_wins(window, capture):
    window.add_paths([str(capture)])
    assert wait(lambda: window.viewer.has_picture())
    auto = after_bytes(window)
    for name in ("Punchy", "Natural", "Golden"):          # faster than any of them can be rendered
        window.bar.looks.buttons[name].click()
    assert wait(lambda: after_bytes(window) != auto)
    settle(1.0)
    golden = after_bytes(window)
    assert window.options.look == "Golden" and window.bar.looks.look() == "Golden"
    window._request_show()
    settle(1.0)
    assert after_bytes(window) == golden, "the picture on screen is the last look asked for"
    window.bar.looks.buttons["Natural"].click()
    assert wait(lambda: after_bytes(window) != golden)


@needs_capture
def test_converting_a_capture_writes_it_next_to_the_source(window, capture):
    window.add_paths([str(capture)])
    assert wait(lambda: window.viewer.has_picture())
    window.bar.convert.click()
    assert wait(lambda: idle(window) and window.queue.items[0].state == "done", 120)
    result = capture.with_name("shot_SDR.jpg")
    assert result.exists() and result.stat().st_size > 100_000
    assert window.queue.items[0].outputs == [str(result)]
    assert window.bar.reveal.isVisible() and not window.bar.cancel.isVisible()
    assert window.bar.convert.text() == "Convert again"
    assert window.bar._full_summary == "Saved shot_SDR.jpg"
    assert not window.banner.isVisible(), "a success adds no line to the window"
    assert sorted(p.name for p in capture.parent.iterdir()) == ["shot.jxr", "shot_SDR.jpg"]


# ----------------------------------------------------------------------- videos

def test_window_matches_decoded_still_for_each_export_format(window, tmp_path):
    from dataclasses import replace
    import imagecodecs
    import numpy as np
    from PIL import Image
    import tifffile

    path = tmp_path / "gradient.jxr"
    ramp = np.linspace(0, 3, 128 * 80, dtype=np.float32).reshape(80, 128)
    path.write_bytes(imagecodecs.jpegxr_encode(np.stack((ramp, ramp**2, ramp[::-1], np.ones_like(ramp)), -1)))
    window.add_paths([str(path)])
    for fmt in ("JPEG", "TIFF", "Both"):
        window._apply(replace(window.options, image_format=fmt, exposure=0.82, pre_gamma=1.35))
        tag = "TIFF 16-bit SDR" if fmt == "TIFF" else "JPEG SDR"
        assert wait(lambda: window.viewer._result_tag == tag and window.viewer._note == "")
        window.convert()
        assert wait(lambda: idle(window) and window.queue.current.state == "done")
        output = window.queue.current.outputs[0]
        if fmt == "TIFF":
            assert output.endswith("_tonemapped.tiff")
            with tifffile.TiffFile(output) as tiff:
                assert tiff.pages[0].tags[34675].value[36:40] == b"acsp"
                rgb = tiff.asarray()
            assert rgb.dtype == np.uint16
        else:
            with Image.open(output) as image:
                assert image.info["icc_profile"][36:40] == b"acsp"
                rgb = np.array(image)
        assert_displayed_pixels(window, rgb)


@needs_capture
def test_first_picture_refits_if_the_window_resized_during_loading(window, capture):
    window.resize(900, 620)
    window.add_paths([str(capture)])
    first_box = window._asked_box
    window.resize(1380, 930)
    window._refit.stop()
    window._fit_picture()  # the resize was delivered before the worker's first answer
    assert not window.viewer.has_picture() and window.viewer.picture_box() != first_box
    assert wait(lambda: window.viewer.has_picture()
                and window.viewer._made_for(window.viewer._after, window.viewer.picture_rect()))
    assert window._asked_box == window.viewer.picture_box()


def test_inspecting_the_encoded_video_then_saving_the_same_bytes(window, video):
    import av
    from dataclasses import replace

    window._apply(replace(window.options, video_fps=30, video_quality="compact"))
    window.add_paths([str(video)])
    assert wait(lambda: window._facts and window._facts["final"])
    window.encoded_button.click()
    assert wait(lambda: window.viewer._result_tag == "Encoded MP4 SDR", 120)
    proof = Path(window.backend._preview_dir.name) / "preview.mp4"
    encoded = proof.read_bytes()
    window.timeline.setValue(630)
    assert wait(lambda: window._facts["time_s"] > 0.1)
    with av.open(str(proof)) as container:
        frames = list(container.decode(video=0))
        held = max((f for f in frames if f.time <= window._facts["time_s"] + 1e-6), key=lambda f: f.pts)
        from hdr_blaster.video.encoded_preview import rgb_of

        assert_displayed_pixels(window, rgb_of(held))
    window.convert()
    assert wait(lambda: idle(window) and window.queue.current.state == "done")
    assert Path(window.queue.current.outputs[0]).read_bytes() == encoded
    assert not proof.exists()
    window._apply(replace(window.options, video_quality="high"))
    assert wait(lambda: window.viewer._result_tag == "SDR before H.264")

def test_a_video_gets_a_provisional_preview_then_the_exact_one(window, video):
    window.add_paths([str(video)])
    assert wait(lambda: window.viewer.has_picture())
    assert window.timeline_row.isVisible()
    assert window.detail_label.text().startswith("256 × 144, HDR10 video")
    assert wait(lambda: window._facts is not None and window._facts["final"], 120)
    assert window.viewer._note == ""
    assert "Auto chose" not in window.detail_label.text()
    first = after_bytes(window)
    window.timeline.setValue(800)
    assert wait(lambda: window._facts["time_s"] > 0.2 and after_bytes(window) != first)
    assert window._facts["final"], "scrubbing after the analysis stays exact"
    assert window.time_label.text().endswith("/ 0:00")


def test_a_video_is_sharpened_only_when_the_switch_is_on(window, video):
    """Sharpness and AI colour stay off a video unless asked for: what the preview shows is
    what the conversion will write."""
    assert window.panel.video_enhance.isChecked() is False
    window.add_paths([str(video)])
    assert wait(lambda: window._facts is not None and window._facts["final"], 120)
    plain = after_bytes(window)
    window.panel.sharpness.setValue(100)
    settle(0.6)
    assert window.options.sharpness == 100.0
    assert wait(lambda: window._facts["final"], 120) and after_bytes(window) == plain, \
        "the sharpness of captures does not touch a video"
    window.panel.video_enhance.click()
    assert window.options.video_enhance is True
    assert wait(lambda: after_bytes(window) != plain, 120), "switched on, the preview is sharpened"
    assert wait(lambda: window._facts["final"], 120)
    window.panel.video_enhance.click()
    assert wait(lambda: window._facts["final"] and after_bytes(window) == plain, 120)


def test_the_frame_rate_is_chosen_in_the_panel_and_keeps_the_analysis(window, video):
    box = window.panel.video_fps
    assert [box.itemText(i) for i in range(box.count())] == ["Original", "120 fps", "60 fps", "30 fps"]
    assert box.currentText() == "Original"
    window.add_paths([str(video)])
    assert wait(lambda: window._facts is not None and window._facts["final"], 120)
    analysis = window.backend.analysis
    box.setCurrentText("30 fps")
    assert window.options.video_fps == 30
    settle(0.6)
    assert window.backend.analysis is analysis and window._facts["final"], "no new analysis for a frame rate"
    stages = []
    window.backend.item_progress.connect(lambda index, stage, done, total: stages.append(stage))
    window.bar.convert.click()
    assert wait(lambda: idle(window), 180)
    assert window.queue.items[0].state == "done" and set(stages) == {"render"}
    written = len(clips.describe(video.with_name("clip_SDR.mp4"))["pts_s"])
    assert 0 < written < 24, "fewer frames than the recording has"


def test_motion_blur_is_offered_at_a_lower_frame_rate_and_shown_by_the_preview(window, video):
    from hdr_blaster.video.job import exposure_of, frames_to_keep

    panel = window.panel
    assert not panel.video_blur.isEnabled() and not panel.video_blur.isChecked()
    window.add_paths([str(video)])
    assert wait(lambda: window._facts is not None and window._facts["final"], 120)
    panel.video_fps.setCurrentText("30 fps")
    assert panel.video_blur.isEnabled()
    analysis = window.backend.analysis
    keep = frames_to_keep(analysis.times, 30.0, analysis.info.average_rate)
    cuts = set(analysis.cuts)
    first = next(i for i in range(3, len(keep)) if keep[i] and exposure_of(keep, cuts, i)[1] > 1)
    # put the timeline on that picture: a frame it is made of
    window.timeline.setValue(int(1000 * analysis.times[first] / window._facts["duration_s"]))
    assert wait(lambda: analysis.index_of(window.backend._source.pts) == first, 60)
    settle(0.6)
    assert wait(lambda: window._facts["final"], 120)
    sharp = after_bytes(window)
    panel.video_blur.click()
    assert window.options.video_blur is True
    assert wait(lambda: window._facts["final"] and after_bytes(window) != sharp, 120), "the preview blurs too"
    assert window.backend.analysis is analysis, "no new analysis for it"
    panel.video_fps.setCurrentText("Original")
    assert not panel.video_blur.isEnabled()
    assert wait(lambda: window._facts["final"] and after_bytes(window) == sharp, 120), \
        "at the original rate nothing is added up, whatever the switch says"


def test_converting_a_video_reuses_the_analysis(window, video):
    window.add_paths([str(video)])
    assert wait(lambda: window._facts is not None and window._facts.get("final"), 120)
    stages = []
    window.backend.item_progress.connect(lambda index, stage, done, total: stages.append(stage))
    window.bar.convert.click()
    assert window.bar.cancel.isVisible() and not window.bar.looks.isEnabled()
    assert wait(lambda: idle(window), 180)
    assert window.queue.items[0].state == "done"
    assert set(stages) == {"render"}, "the analysis made for the preview is not done again"
    assert video.with_name("clip_SDR.mp4").exists()
    assert len(clips.describe(video.with_name("clip_SDR.mp4"))["pts_s"]) == 24
    assert window.bar.looks.isEnabled()


def test_cancelling_a_conversion_leaves_nothing_behind(window, tmp_path):
    long_clip = tmp_path / "files" / "long.mp4"
    long_clip.parent.mkdir()
    clips.make_clip(long_clip, frames=150, audio_streams=0)
    window.add_paths([str(long_clip)])
    assert wait(lambda: window.viewer.has_picture())
    window.bar.convert.click()
    assert wait(lambda: window.bar.convert._progress and window.bar.convert._progress > 0.05, 120)
    window.bar.cancel.click()
    assert wait(lambda: idle(window), 120)
    assert window.queue.items[0].state == "cancelled"
    assert sorted(p.name for p in long_clip.parent.iterdir()) == ["long.mp4"]
    assert window.bar.convert.text() == "Convert" and window.bar.convert.isEnabled()
    assert not window.bar.reveal.isVisible()


# ----------------------------------------------------------------- refusals, queue

def test_an_unsupported_file_is_explained_in_the_window(window, tmp_path):
    photo = tmp_path / "holiday.jpg"
    photo.write_text("x")
    window.add_paths([str(photo)])
    assert window.banner.isVisible()
    assert "holiday.jpg is not an HDR capture or recording" in window.banner.text()
    assert window.queue.items == [] and not window.bar.convert.isEnabled()


def test_an_sdr_video_is_refused_with_its_reason(window, tmp_path):
    sdr = tmp_path / "sdr.mp4"
    clips.make_clip(sdr, frames=4, hdr=False, audio_streams=0)
    window.add_paths([str(sdr)])
    assert wait(lambda: window.banner.isVisible())
    assert window.banner.text().count("sdr.mp4") == 1 and "not an HDR10 video" in window.banner.text()
    assert window.queue.items == [], "a file that cannot be shown does not stay in the list"
    assert not window.viewer.has_picture() and not window.bar.convert.isEnabled()


@needs_capture
def test_a_folder_adds_its_supported_files_and_the_queue_appears(window, capture, video, tmp_path):
    (capture.parent / "notes.txt").write_text("x")
    window.add_paths([str(capture.parent)])
    assert [i.name for i in window.queue.items] == ["clip.mp4", "shot.jxr"]
    assert window.queue_panel.isVisible() and len(window.queue_panel.rows) == 2
    assert window.bar.convert.text() == "Convert 2 files"
    assert wait(lambda: window.viewer.has_picture())
    window.add_paths([str(capture)])
    assert len(window.queue.items) == 2 and "Already in the list" in window.banner.text()
    window.queue_panel.rows[1].click()
    assert wait(lambda: window._facts is not None and window._facts["path"] == str(capture))
    assert window.file_label.text() == "shot.jxr"
    window.bar.convert.click()
    assert wait(lambda: idle(window), 240)
    assert [i.state for i in window.queue.items] == ["done", "done"]
    assert window.bar._full_summary == "Saved 2 files" and not window.banner.isVisible()
    window.remove(0)
    assert [i.name for i in window.queue.items] == ["shot.jxr"]
    assert [row.name for row in window.queue_panel.rows] == ["shot.jxr"] and window.queue_button.count() == 1


# ------------------------------------------------------------ panel and settings

@needs_capture
def test_the_advanced_panel_slides_in_without_covering_the_picture_or_the_button(window, capture):
    window.add_paths([str(capture)])
    assert wait(lambda: window.viewer.has_picture())
    full_width = window.viewer.picture_rect().width()
    window.advanced_button.click()
    settle(0.4)
    assert window.panel.isVisible()
    picture = window.viewer.picture_rect()
    assert picture.width() < full_width
    assert window.viewer.mapTo(window, picture.toRect().topRight()).x() <= window.panel.mapTo(window, window.panel.rect().topLeft()).x()
    bar_top = window.bar.mapTo(window, window.bar.rect().topLeft()).y()
    assert window.panel.mapTo(window, window.panel.rect().bottomLeft()).y() <= bar_top
    before = after_bytes(window)
    window.panel.sharpness.setValue(80)
    assert wait(lambda: after_bytes(window) != before)
    assert window.options.sharpness == 80.0
    window.panel.operator.setCurrentText("Hable")
    assert window.options.method() == "hable"
    window.bar.looks.buttons["Punchy"].click()
    assert window.options.method() == "punchy", "choosing a look drops the advanced operator"
    assert window.panel.operator.currentText() == "Follow the look"
    window.advanced_button.click()
    settle(0.4)
    assert not window.panel.isVisible()
    assert window.viewer.picture_rect().width() == pytest.approx(full_width, abs=1)


@needs_capture
def test_a_click_in_the_advanced_panel_never_moves_the_divider(window, capture):
    """The panel lives in the picture area. A click on its ground, on a label or beside a
    control went through to the picture and threw the divider to the right edge."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLabel

    window.add_paths([str(capture)])
    assert wait(lambda: window.viewer.has_picture())
    window.advanced_button.click()
    settle(0.4)
    window.viewer.set_split(0.4)
    panel = window.panel
    labels = {label.text(): label for label in panel.findChildren(QLabel)}
    targets = {"the panel's ground": (panel, QPoint(panel.width() // 2, 10)),
               "a control's name": (labels["Exposure"], labels["Exposure"].rect().center()),
               "a section's name": (labels["Output"], labels["Output"].rect().center()),
               "a slider's value": (panel.exposure_value, panel.exposure_value.rect().center())}
    for what, (widget, point) in targets.items():
        QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, point)
        QTest.mouseDClick(widget, Qt.LeftButton, Qt.NoModifier, point)
        assert window.viewer.split == 0.4, f"a click on {what} moved the divider"
        assert not window.viewer._dragging, f"a click on {what} started a drag of the divider"
    # the picture itself still answers
    rect = window.viewer.picture_rect()
    QTest.mouseClick(window.viewer, Qt.LeftButton, Qt.NoModifier,
                     QPoint(round(rect.left() + rect.width() * 0.25), round(rect.center().y())))
    assert window.viewer.split == pytest.approx(0.25, abs=0.01)


def test_the_chosen_look_is_a_pill_that_follows_the_choice(window):
    from PySide6.QtCore import QRectF

    looks = window.bar.looks
    settle(0.3)
    assert looks._choice._shown == QRectF(looks.buttons["Faithful"].geometry())
    looks.buttons["Golden"].click()
    settle(0.4)                                             # the pill glides, then rests on the choice
    assert looks._choice._shown == QRectF(looks.buttons["Golden"].geometry())
    window.panel.operator.setCurrentText("Hable")           # an operator replaces the looks: no pill
    settle(0.2)
    assert looks.look() is None and looks._choice._shown is None


def test_a_switch_answers_on_its_whole_row_and_a_click_leaves_no_focus_ring(window):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton

    window.toggle_advanced(True)
    settle(0.4)
    switch = window.panel.show_histogram
    assert not switch.isChecked()
    QTest.mouseClick(switch, Qt.LeftButton, Qt.NoModifier, QPoint(switch.width() - 6, switch.height() // 2))
    assert switch.isChecked() and window.options.show_histogram, "the switch itself"
    QTest.mouseClick(switch, Qt.LeftButton, Qt.NoModifier, QPoint(4, switch.height() // 2))
    assert not switch.isChecked(), "its name"
    # A button shows its focus ring to the keyboard only: a click must not leave one behind.
    for button in window.findChildren(QAbstractButton):
        assert button.focusPolicy() == Qt.TabFocus, f"'{button.text() or button.accessibleName()}' keeps the focus after a click"


def test_settings_are_remembered_between_sessions(qapp, tmp_path):
    from hdr_blaster.ui.window import MainWindow

    path = str(tmp_path / "settings.ini")
    first = MainWindow(path)
    first.bar.looks.buttons["Golden"].click()
    first.panel.video_quality.setCurrentIndex(2)
    first.panel.show_histogram.setChecked(True)
    first.resize(1000, 640)
    first.show()
    settle(0.1)
    first.close()
    second = MainWindow(path)
    try:
        assert second.options.look == "Golden" and second.bar.looks.look() == "Golden"
        assert second.options.video_quality == "compact" and second.options.show_histogram
        assert second.panel.video_quality.currentIndex() == 2
        # the off-screen display is 800 pixels wide, so the width is clamped to it there
        assert second.height() == 640 and second.width() in (1000, second.minimumWidth())
    finally:
        second.close()


def test_closing_during_a_conversion_stops_cleanly(qapp, tmp_path):
    from hdr_blaster.ui.window import MainWindow

    long_clip = tmp_path / "files" / "long.mp4"
    long_clip.parent.mkdir()
    clips.make_clip(long_clip, frames=150, audio_streams=0)
    win = MainWindow(str(tmp_path / "settings.ini"))
    win.show()
    win.add_paths([str(long_clip)])
    assert wait(lambda: win.viewer.has_picture())
    win.bar.convert.click()
    assert wait(lambda: win.bar.convert._progress and win.bar.convert._progress > 0.05, 120)
    win.close()
    assert not win.thread.isRunning()
    assert sorted(p.name for p in long_clip.parent.iterdir()) == ["long.mp4"]


# ------------------------------------------------------------------------ layout

def test_nothing_is_cramped_from_the_smallest_window_to_full_hd(window, video):
    window.add_paths([str(video), str(video.parent)])
    assert wait(lambda: window.viewer.has_picture())
    window.notify("holiday.jpg is not an HDR capture or recording. JXR screenshots and HDR10 videos can be converted.")
    minimum = (window.minimumWidth(), window.minimumHeight())
    assert layout.check(window, [minimum, (1280, 800), (1920, 1080)]) == []


@pytest.mark.parametrize("scale", ["1.5", "2"])
def test_nothing_is_cramped_at_other_display_scales(video, scale):
    """Qt reads the scale when it starts, so each scale needs its own process."""
    done = subprocess.run([sys.executable, str(ROOT / "tests" / "ui_layout_check.py"), str(video)],
                          env={**os.environ, "QT_SCALE_FACTOR": scale, "QT_QPA_PLATFORM": "offscreen"},
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0 and "layout ok" in done.stdout, done.stdout + done.stderr[-2000:]


# ------------------------------------------------ found by the independent review

@needs_capture
def test_the_histogram_can_be_shown(window, capture):
    """Turning it on used to raise in the middle of showing a preview: no picture details,
    no timeline, and the setting was remembered for the next launch."""
    import numpy as np

    from hdr_blaster.ui.viewer import histogram_of

    curves = histogram_of(np.full((4, 6, 3), 255, dtype=np.uint8))
    assert curves.shape == (3, 64) and curves[:, -1].min() == 1.0 and curves[:, :-1].max() == 0.0
    window.add_paths([str(capture)])
    assert wait(lambda: window.viewer.has_picture())
    window.panel.show_histogram.setChecked(True)
    assert wait(lambda: window.viewer._histogram is not None)
    assert window.detail_label.text().startswith("2560 × 1440")
    window.panel.show_histogram.setChecked(False)
    assert wait(lambda: window.viewer._histogram is None)


def test_exposure_and_gamma_are_visible_and_usable(window):
    window.advanced_button.click()
    settle(0.4)
    assert window.panel.exposure_value.text() == "1.00" and window.panel.gamma_value.text() == "1.00"
    assert window.panel.exposure_value.width() >= window.panel.exposure_value.sizeHint().width()
    window.panel.exposure.setValue(150)
    assert window.options.exposure == 1.5 and window.panel.exposure_value.text() == "1.50"
    window.panel.gamma.setValue(80)
    assert window.options.pre_gamma == 0.8


def test_the_panel_never_covers_the_timeline_the_banner_or_the_bar(window, video):
    """With the panel open, adding a video or a second file moved the bar under the panel."""
    window.advanced_button.click()
    settle(0.4)
    window.add_paths([str(video), str(video.parent)])
    other = video.with_name("other.mp4")
    clips.make_clip(other, frames=6, audio_streams=0)
    window.add_paths([str(other)])
    assert wait(lambda: window.viewer.has_picture())
    window.notify("holiday.jpg is not an HDR capture or recording.")
    settle(0.3)
    assert window.panel.isVisible() and window.timeline_row.isVisible() and window.banner.isVisible()
    assert window.queue_button.count() == 2, "the settings stay open; the queue's button tells what waits"

    def box(widget):
        top_left = widget.mapTo(window, widget.rect().topLeft())
        return top_left.x(), top_left.y(), top_left.x() + widget.width(), top_left.y() + widget.height()

    px0, py0, px1, py1 = box(window.panel)
    vx0, vy0, vx1, vy1 = box(window.viewer)
    assert (py0, py1) == (vy0, vy1) and px1 == vx1, "the panel occupies the right of the picture area, nothing else"
    for widget in (window.timeline_row, window.banner, window.bar, window.time_label):
        _, y0, _, y1 = box(widget)
        assert y1 <= py0 or y0 >= py1, f"{type(widget).__name__} is under the panel"
    window.resize(1700, 900)
    window.toggle_queue(True)
    settle(0.5)
    assert window.panel.isVisible() and window.queue_panel.isVisible()
    qx0, qy0, qx1, qy1 = box(window.queue_panel)
    vx0, vy0, vx1, vy1 = box(window.viewer)
    assert (qx0, qy0, qy1) == (vx0, vy0, vy1), "and the queue the left of it"
    for widget in (window.timeline_row, window.banner, window.bar, window.time_label):
        _, y0, _, y1 = box(widget)
        assert y1 <= qy0 or y0 >= qy1, f"{type(widget).__name__} is under the queue"


def test_a_failure_to_start_does_not_leave_the_window_stuck(window, video, monkeypatch):
    window.add_paths([str(video)])
    assert wait(lambda: window._facts is not None and window._facts.get("final"), 120)

    def broken(options):
        raise RuntimeError("the graphics driver went away")

    monkeypatch.setattr(window.backend.engines, "get", broken)
    window.bar.convert.click()
    assert wait(lambda: idle(window), 30), "the window stayed in 'Starting…'"
    assert window.queue.items[0].state == "failed"
    assert "graphics driver went away" in window.banner.text()
    assert window.bar.convert.isEnabled() and window.bar.looks.isEnabled() and not window.bar.cancel.isVisible()


def test_progress_without_a_frame_count_does_not_claim_zero_percent(window):
    window._converting, window._job_total, window._job_index = True, 1, 1
    window._on_item_progress(0, "render", 40, 0)
    assert window.bar.convert.text() == "Converting…"
    window._on_item_progress(0, "render", 50, 40)
    assert window.bar.convert.text() == "Converting… 100 %"
    window._converting = False


def test_progress_text_always_fits_the_button(window):
    button = window.bar.convert
    window._converting, window._job_total, window._job_index = True, 15, 12
    for stage in ("method", "analyse", "render"):
        window._on_item_progress(0, stage, 100, 100)
        assert button.fontMetrics().horizontalAdvance(button.text()) <= button.width() - 16, button.text()
    window._converting = False


def test_coming_back_to_an_analysed_video_does_not_analyse_it_again(window, video, tmp_path):
    other = tmp_path / "files" / "other.mp4"
    clips.make_clip(other, frames=12, audio_streams=0)
    window.add_paths([str(video), str(other)])
    assert wait(lambda: window._facts is not None and window._facts.get("final"), 120)
    analysed = window.backend.analysis
    window.select(1)
    assert wait(lambda: window._facts is not None and window._facts["path"] == str(other) and window._facts["final"], 120)
    assert window.backend.analysis is not analysed, "the other video has its own analysis"
    window.select(1)                                         # the same video again: nothing to analyse
    passes = []
    window.backend.analysis_progress.connect(lambda *a: passes.append(a))
    assert wait(lambda: window._facts is not None and window._facts["path"] == str(other) and window._facts["final"], 120)
    settle(0.5)
    assert passes == [], "a video already analysed with these settings is not analysed again"


def test_a_conversion_keeps_its_analysis_for_the_preview(window, video):
    window.add_paths([str(video)])
    assert wait(lambda: window.viewer.has_picture())
    window.bar.convert.click()                               # before the background analysis is done
    assert wait(lambda: idle(window), 180)
    assert window.queue.items[0].state == "done"
    passes = []
    window.backend.analysis_progress.connect(lambda *a: passes.append(a))
    assert wait(lambda: window._facts is not None and window._facts.get("final"), 60)
    settle(0.5)
    assert passes == [], "the conversion's own analysis serves the preview: no further pass"


def test_an_advanced_operator_unlights_the_looks(window):
    window.panel.operator.setCurrentText("Hable")
    assert all(not b.isChecked() for b in window.bar.looks.buttons.values())
    window.panel.operator.setCurrentText("Follow the look")
    assert window.bar.looks.look() == "Faithful"


def test_long_names_are_elided_not_cut(window, tmp_path):
    long_name = tmp_path / ("a very long capture name " * 6 + ".mp4")
    clips.make_clip(long_name, frames=4, audio_streams=0)
    window.resize(window.minimumWidth(), window.minimumHeight())
    window.add_paths([str(long_name)])
    settle(0.3)
    assert "…" in window.file_label.text() and window.file_label.toolTip() == str(long_name)


# --------------------------------------------------------------- faithful, auto

@needs_capture
def test_by_default_the_result_is_the_source_where_the_scene_fits(window, capture):
    """The seam reported on a dark capture: either side of the divider must be one picture."""
    import numpy as np

    window.add_paths([str(capture)])
    assert wait(lambda: window.viewer.has_picture() and window._facts and window._facts["final"])

    def pixels(image):
        raw = np.frombuffer(bytes(image.constBits())[: image.sizeInBytes()], dtype=np.uint8)
        return raw.reshape(image.height(), image.bytesPerLine())[:, : image.width() * 3].astype(np.int16)

    before, after = pixels(window.viewer._before), pixels(window.viewer._after)
    dark = before < 128                                       # well under white on any screen
    assert float(np.abs(after - before)[dark].mean()) < 1.5, "the result's shadows are the source's"
    # The real JPEG includes dither and compression. Neither may systematically lift the shadows.
    delta = (after - before)[dark]
    assert np.median(delta) == 0 and np.median(np.abs(delta)) <= 1


@needs_capture
def test_window_names_neural_fallback(window, capture, monkeypatch):
    from hdr_blaster.engine import tonemap

    monkeypatch.setattr(tonemap, "_get_neural_tonemap", lambda device: None)
    window.panel.operator.setCurrentText("Neural (AI)")
    window.add_paths([str(capture)])
    assert wait(lambda: "Neural unavailable, using Perceptual" in window.detail_label.text())
    assert window._facts["method"] == "perceptual"


@needs_capture
def test_auto_select_still_names_the_operator_it_chose(window, capture):
    window.add_paths([str(capture)])
    assert wait(lambda: window.viewer.has_picture())
    window.panel.operator.setCurrentText("Auto-select")
    assert wait(lambda: "Auto chose " in window.detail_label.text())
    assert window.detail_label.text().split("Auto chose ")[1][0].isupper(), "a name, not an identifier"
    assert all(not b.isChecked() for b in window.bar.looks.buttons.values())


@needs_capture
def test_exposure_and_gamma_leave_the_left_picture_alone(window, capture):
    """Reported: the Advanced sliders changed the original too. Only the result may move."""
    window.add_paths([str(capture)])
    assert wait(lambda: window.viewer.has_picture() and window._facts and window._facts["final"])

    def before_bytes():
        image = window.viewer._before
        return bytes(image.constBits())[: image.sizeInBytes()]

    original, result = before_bytes(), after_bytes(window)
    window.panel.exposure.setValue(180)
    assert wait(lambda: after_bytes(window) != result), "the result follows Exposure"
    assert before_bytes() == original, "the source view does not"
    result = after_bytes(window)
    window.panel.gamma.setValue(140)
    assert wait(lambda: after_bytes(window) != result), "the result follows Gamma"
    assert before_bytes() == original


# ------------------------------------------------- the picture, pixel for pixel

def shown(window):
    image = window.viewer._before
    return image.width(), image.height()


@needs_capture
def test_the_picture_is_made_for_its_place_and_drawn_as_it_is(window, capture):
    """Reported: staircases on thin lines, on both sides of the divider. The capture was
    reduced to 1920 pixels at most and the window stretched it to its place. The picture
    now arrives at the exact size of its place, and the window draws it untouched."""
    import numpy as np
    from PySide6.QtGui import QImage

    from hdr_blaster.resample import fit_size

    window.add_paths([str(capture)])
    viewer = window.viewer
    assert wait(lambda: viewer.has_picture() and shown(window) == fit_size(*viewer.picture_box(), 2560, 1440))
    ratio = viewer.devicePixelRatioF()
    rect = viewer.picture_rect()
    assert (round(rect.width() * ratio), round(rect.height() * ratio)) == shown(window)
    assert rect.left() * ratio == pytest.approx(round(rect.left() * ratio), abs=1e-6), "on the screen's pixel grid"
    assert rect.top() * ratio == pytest.approx(round(rect.top() * ratio), abs=1e-6)
    assert viewer._pixmap(viewer._after, rect).size() == viewer._after.size(), "not rescaled by the window"
    # what is painted is the picture itself, pixel for pixel
    settle(0.3)                                             # let the cross-fade end
    viewer.set_split(0.0)
    painted = viewer.grab().toImage().convertToFormat(QImage.Format_RGB888)
    after = viewer._after

    def pixels(image):
        rows = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
        return rows[:, : image.width() * 3].reshape(image.height(), image.width(), 3)

    left, top = round(rect.left() * ratio), round(rect.top() * ratio)
    inner = (slice(60, after.height() - 100), slice(60, after.width() - 60))       # clear of the tags and corners
    on_screen = pixels(painted)[top:top + after.height(), left:left + after.width()]
    assert np.array_equal(on_screen[inner], pixels(after)[inner])


@needs_capture
def test_resizing_the_window_brings_the_picture_at_its_new_size(window, capture):
    from hdr_blaster.resample import fit_size

    window.add_paths([str(capture)])
    viewer = window.viewer
    assert wait(lambda: viewer.has_picture())
    first = shown(window)
    window.resize(1000, 640)
    assert wait(lambda: shown(window) != first and shown(window) == fit_size(*viewer.picture_box(), 2560, 1440), 10)
    assert viewer._after.size() == viewer._before.size()
    smaller = shown(window)
    window.advanced_button.click()                          # the panel takes room: the picture is made again
    assert wait(lambda: shown(window) != smaller and shown(window) == fit_size(*viewer.picture_box(), 2560, 1440), 10)
    assert shown(window)[0] < smaller[0]


def test_a_video_frame_is_shown_at_the_size_of_its_place_too(window, video):
    from hdr_blaster.resample import fit_size

    window.add_paths([str(video)])
    viewer = window.viewer
    assert wait(lambda: viewer.has_picture() and shown(window) == fit_size(*viewer.picture_box(), 256, 144))
    assert shown(window)[0] > 256, "a small video is enlarged once, by the resampler, not by the window"


# ------------------------------------------------ a success moves nothing; the icon

@needs_capture
def test_a_success_adds_no_line_and_moves_nothing(window, capture):
    """Reported: every conversion that succeeded inserted a line above the picture, which
    pushed the picture down and made it smaller. The news now takes the place of the
    output summary, next to 'Show in folder', and gives it back when something else is asked."""
    window.add_paths([str(capture)])
    viewer = window.viewer
    assert wait(lambda: viewer.has_picture())
    settle(0.5)
    assert window.bar._full_summary == "JPEG, next to the source"
    place, picture = viewer.geometry(), viewer.picture_rect()
    window.bar.convert.click()
    assert wait(lambda: idle(window) and window.queue.items[0].state == "done", 120)
    settle(0.5)
    assert not window.banner.isVisible()
    assert viewer.geometry() == place and viewer.picture_rect() == picture, "the picture did not move"
    assert window.bar._full_summary == "Saved shot_SDR.jpg" and window.bar.reveal.isVisible()
    assert window.bar.summary.property("role") == "done"
    window.bar.looks.buttons["Punchy"].click()               # another picture is asked for
    assert window.bar._full_summary == "JPEG, next to the source"
    assert window.bar.summary.property("role") == "muted"
    window.bar.convert.click()
    assert wait(lambda: idle(window) and window.bar._full_summary == "Saved shot_SDR-2.jpg", 120)
    assert not window.banner.isVisible() and viewer.geometry() == place


def test_a_failure_is_still_explained_in_a_line(window, tmp_path):
    photo = tmp_path / "holiday.jpg"
    photo.write_bytes(b"not a capture")
    window.add_paths([str(photo)])
    assert window.banner.isVisible() and "holiday.jpg" in window.banner.text()


def test_the_icon_file_is_a_real_windows_icon():
    """Reported: a blank white icon in the taskbar. The file named app.ico was a PNG picture
    under another name, which Windows cannot read as an icon."""
    data = (ROOT / "hdr_blaster" / "ui" / "icon.ico").read_bytes()
    assert data[:4] == bytes([0, 0, 1, 0]), "an ICO file starts with its own signature"
    count = int.from_bytes(data[4:6], "little")
    sizes = {data[6 + 16 * i] or 256 for i in range(count)}
    assert {16, 24, 32, 48, 256} <= sizes, "every size Windows asks for, from the title bar to the taskbar"
    spec = (ROOT / "NHC.spec").read_text(encoding="utf-8")
    assert 'icon="hdr_blaster/ui/icon.ico"' in spec, "the executable carries it"
    assert '("hdr_blaster/ui/icon.ico", "hdr_blaster/ui")' in spec, "and the window finds it once frozen"


def test_the_window_shows_the_icon(window, qapp):
    from PySide6.QtGui import QImage

    icon = window.windowIcon()
    assert not icon.isNull() and not qapp.windowIcon().isNull()
    assert {16, 32, 48, 256} <= {size.width() for size in icon.availableSizes()}
    image = icon.pixmap(48, 48).toImage().convertToFormat(QImage.Format_RGBA8888)
    corner, middle = image.pixelColor(0, 0), image.pixelColor(24, 6)
    assert corner.alpha() == 0, "round: its corners are transparent"
    assert middle.alpha() == 255 and middle.saturation() > 80, "and it is not a blank square"


# ------------------------------------------------------- the panel keeps its width

def beyond_the_panel(panel) -> list:
    """The visible controls of the panel that reach past its right edge."""
    from PySide6.QtWidgets import QWidget

    out = []
    for widget in panel.findChildren(QWidget):
        if widget.isVisible() and widget.width() > 0:
            right = widget.mapTo(panel, widget.rect().topRight()).x()
            if right > panel.width():
                name = getattr(widget, "text", lambda: "")() or type(widget).__name__
                out.append(f"{type(widget).__name__} '{name[:30]}' ends at {right} of {panel.width()}")
    return out


DEEP_FOLDER = (r"C:\Users\SYMPHO~1\AppData\Local\Temp\claude\762bb878c4e13d8f"
               r"\ae547df8-6d00-404f-9721-893571242893\scratchpad\out_old")


@pytest.mark.parametrize("folder", [r"D:\Pictures", DEEP_FOLDER, "D:\\" + "a_folder_with_a_very_long_name_" * 5,
                                    "//server/share/" + "screenshots/" * 12 + "sdr"],
                         ids=["short", "deep", "one long name", "network"])
def test_a_destination_never_pushes_the_panel_beyond_its_width(window, folder):
    """Reported with a screenshot: every control of the Advanced panel cut off at the right.
    The folder shown under "Save to" was a long path, which a label cannot break: it asked
    for its whole width, and the panel's content grew wider than the panel."""
    from dataclasses import replace

    window.resize(1280, 800)
    window.toggle_advanced(True)
    settle(0.4)
    panel = window.panel
    assert not beyond_the_panel(panel), "the panel fits to begin with"
    panel.set_options(replace(window.options, destination=folder))
    settle(0.3)
    assert panel.reset_folder.isVisible(), "a folder is chosen: the way back is offered"
    assert not beyond_the_panel(panel)
    assert panel.destination.toolTip() == folder, "the whole path can be read"
    assert panel.destination.text().replace("\u200b", "") == folder, "and it is the path that is shown"
    for button in (panel.choose_folder, panel.reset_folder):
        assert button.width() >= button.sizeHint().width() - 1, f"'{button.text()}' has room for its text"
    panel.set_options(replace(window.options, destination=""))
    settle(0.2)
    assert panel.destination.text() == "Next to the source file" and panel.destination.toolTip() == ""
    assert not beyond_the_panel(panel)

"""What SDR white stands for is asked of the screen the window is on, for the file as for the preview.

Two screens in HDR whose "SDR content brightness" differ, the window on the second: the
preview asked that screen, the conversion asked no screen in particular and was answered
by the first. A grey of 40 nits came out at 102 of 255 in the preview and at 82 in the
picture rendered for the file. Windows' answers are simulated here; nothing else is.
"""
import shutil

import pytest

import ui_layout_check as layout                    # sets the offscreen platform before Qt loads  # noqa: F401
import video_clips as clips
from conftest import SAMPLES
from hdr_blaster.session import Engines, Item, Options, convert_item
from ui_layout_check import settle, wait

FIRST, SECOND = "\\\\.\\DISPLAY1", "\\\\.\\DISPLAY2"
needs_capture = pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")


@pytest.fixture
def screens(monkeypatch):
    """Two screens in HDR, white at 480 and 300 nits, and the window on the second. A test
    changes what Windows answers by changing this."""
    from hdr_blaster import display
    from hdr_blaster.ui import window as window_module

    state = {"hdr": [(FIRST, 480.0), (SECOND, 300.0)], "window": SECOND}
    monkeypatch.setattr(display, "_windows_hdr_displays", lambda: list(state["hdr"]))
    monkeypatch.setattr(window_module, "display_of_window", lambda _window_id: state["window"])
    return state


@pytest.fixture
def asked(monkeypatch):
    """The white each conversion step was asked for: the render of a capture, the analysis
    and the render of a video."""
    from hdr_blaster.engine import pipeline
    from hdr_blaster.video import job

    seen = {"capture": [], "analysis": [], "video": []}
    render, analyse_video, render_video = pipeline.render, job.analyse_video, job.render_video

    def spy_render(*args, **kwargs):
        seen["capture"].append((args[4] if len(args) > 4 else kwargs["settings"]).white_nits)
        return render(*args, **kwargs)

    def spy_analysis(path, settings, *args, **kwargs):
        seen["analysis"].append(settings.render.white_nits)
        return analyse_video(path, settings, *args, **kwargs)

    def spy_video(analysis, dst, settings, *args, **kwargs):
        seen["video"].append(settings.render.white_nits)
        return render_video(analysis, dst, settings, *args, **kwargs)

    monkeypatch.setattr(pipeline, "render", spy_render)
    monkeypatch.setattr(job, "analyse_video", spy_analysis)
    monkeypatch.setattr(job, "render_video", spy_video)
    return seen


@pytest.fixture(scope="module")
def qapp():
    from hdr_blaster.ui.app import create_application

    return create_application([])


@pytest.fixture
def window(qapp, tmp_path, screens):
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


def idle(window) -> bool:
    return not window._converting


def white_shown(window):
    return window._facts["white_nits"] if window._facts else None


# ------------------------------------------------------------------ the settings

def test_the_settings_of_a_conversion_take_the_white_of_the_screen_named(screens):
    options = Options()
    assert options.render_settings(screen=SECOND).white_nits == 300.0
    assert options.render_settings("video", screen=SECOND).white_nits == 300.0
    assert options.video_settings(screen=SECOND).render.white_nits == 300.0
    assert options.render_settings(screen=FIRST).white_nits == 480.0
    assert options.render_settings().white_nits == 480.0, "no window (the command line): the first screen in HDR"
    assert options.picture_key(SECOND) != options.picture_key(FIRST), "another white is another picture"
    assert options.picture_key(SECOND) == Options(image_format="TIFF").picture_key(SECOND)
    screens["hdr"] = [(SECOND, 300.0)]
    assert options.render_settings(screen=FIRST).white_nits == 203.0, "a screen that is not in HDR: the reference white"


@needs_capture
def test_a_capture_is_converted_for_the_screen_named(tmp_path, screens):
    """The file made for the second screen is the one a machine with that screen alone makes."""
    import numpy as np
    from PIL import Image

    src = tmp_path / "shot.jxr"
    shutil.copy(SAMPLES[0], src)
    engine, device = Engines().get(Options())

    def converted(**how):
        written = convert_item(Item(str(src), "image"), Options(), engine, device, **how)
        return np.asarray(Image.open(written[0]), dtype=np.float32)

    def apart(a, b) -> float:
        """Mean difference in levels of 255. (Never zero between two files: each JPEG is
        dithered with its own noise, a level at most.)"""
        return float(np.abs(a - b).mean())

    on_second = converted(screen=SECOND)
    on_first = converted(screen=FIRST)
    screens["hdr"] = [(SECOND, 300.0)]
    alone_second = apart(converted(), on_second)
    screens["hdr"] = [(FIRST, 480.0)]
    alone_first = apart(converted(), on_first)
    assert max(alone_second, alone_first) < 2.0, "the same picture, to the dither and the JPEG coding"
    assert apart(on_second, on_first) > 5 * max(alone_second, alone_first), "two whites, two pictures"


# -------------------------------------------------------------------- the window

@needs_capture
def test_a_capture_is_converted_for_the_screen_the_window_is_on(window, capture, asked):
    window.add_paths([str(capture)])
    assert wait(lambda: white_shown(window) is not None)
    assert white_shown(window) == 300.0, "the preview is made for the window's screen"
    window.bar.convert.click()
    assert wait(lambda: idle(window), 240) and window.queue.items[0].state == "done"
    assert asked["capture"] and set(asked["capture"]) == {300.0}, "and so is the file: the same white, the same picture"


def test_a_video_is_analysed_and_converted_for_the_screen_the_window_is_on(window, tmp_path, asked):
    video = tmp_path / "clip.mp4"
    clips.make_clip(video, frames=24, audio_streams=0)
    window.add_paths([str(video)])
    assert wait(lambda: window._facts is not None and window._facts.get("final"), 120), "analysed in the background"
    assert white_shown(window) == 300.0 and asked["analysis"] == [300.0]
    window.bar.convert.click()
    assert wait(lambda: idle(window), 240) and window.queue.items[0].state == "done"
    assert asked["video"] == [300.0]
    assert asked["analysis"] == [300.0], "the analysis made for the preview serves the file: it is for the same white"


@needs_capture
def test_the_whole_queue_is_converted_for_the_screen_the_window_is_on(window, capture, tmp_path, asked):
    """The files that were never shown too: they have no preview to take their white from."""
    video = capture.with_name("clip.mp4")
    clips.make_clip(video, frames=12, audio_streams=0)
    window.add_paths([str(capture), str(video)])
    assert wait(lambda: white_shown(window) is not None)
    window.bar.convert.click()
    assert wait(lambda: idle(window), 240)
    assert [item.state for item in window.queue.items] == ["done", "done"]
    assert set(asked["capture"]) == {300.0} and asked["video"] == [300.0] and set(asked["analysis"]) == {300.0}


@needs_capture
def test_the_picture_is_made_again_when_the_window_changes_screen(window, capture, screens):
    """Shown on one screen, converted on the other: the file would not be the picture seen."""
    screens["window"] = FIRST
    window.add_paths([str(capture)])
    assert wait(lambda: white_shown(window) == 480.0)
    asked_before = window.backend.latest_show
    window.move(window.x() + 40, window.y() + 10)                  # within the same screen
    settle(0.6)
    assert window.backend.latest_show == asked_before, "nothing is made again while the white is the same"
    screens["window"] = SECOND
    window.move(window.x() + 40, window.y() + 10)                  # dragged onto the other screen
    assert wait(lambda: white_shown(window) == 300.0, 30), "the preview follows the screen"


@needs_capture
def test_the_picture_is_made_again_when_the_screens_white_was_changed_meanwhile(window, capture, screens):
    """'SDR content brightness' moved in Windows' settings while the program was open: seen
    when the window comes back to the front."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    window.add_paths([str(capture)])
    assert wait(lambda: white_shown(window) == 300.0)
    screens["hdr"] = [(FIRST, 480.0), (SECOND, 200.0)]
    QApplication.sendEvent(window, QEvent(QEvent.ActivationChange))
    assert wait(lambda: white_shown(window) == 200.0, 30)

"""The queue: a panel on the left with one row per file to convert.

Each row says what the file is (a capture, a video), shows a picture of it, and tells where
its conversion stands. The panel slides in as the Advanced panel does on the other side:
the picture gives way, it is never covered.
"""
import shutil
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
    win.resize(1280, 800)
    win.show()
    settle(0.05)
    yield win
    win.close()


@pytest.fixture
def videos(tmp_path):
    folder = tmp_path / "files"
    folder.mkdir()
    paths = []
    for name, frames in (("first.mp4", 12), ("second.mp4", 10), ("third.mp4", 8)):
        clips.make_clip(folder / name, frames=frames, audio_streams=0)
        paths.append(folder / name)
    return paths


def box(widget, window):
    top_left = widget.mapTo(window, widget.rect().topLeft())
    return top_left.x(), top_left.y(), top_left.x() + widget.width(), top_left.y() + widget.height()


def idle(window) -> bool:
    return not window._converting


# ------------------------------------------------------------------- the panel

def test_one_file_needs_no_queue_and_a_second_one_opens_it_on_the_left(window, videos):
    panel = window.queue_panel
    assert not panel.isVisible() and window.queue_button.count() == 0
    window.add_paths([str(videos[0])])
    assert wait(lambda: window.viewer.has_picture())
    assert not panel.isVisible() and window.queue_button.count() == 1, "a single file is simply shown"
    full = window.viewer.picture_rect()
    window.add_paths([str(videos[1])])
    settle(0.5)
    assert panel.isVisible() and window.queue_button.isChecked() and window.queue_button.count() == 2
    assert wait(lambda: window.viewer.has_picture() and window._facts and window._facts["path"] == str(videos[1]))
    settle(0.5)                                     # (the timeline of the video has taken its place by now)
    px0, py0, px1, py1 = box(panel, window)
    vx0, vy0, vx1, vy1 = box(window.viewer, window)
    assert (px0, py0, py1) == (vx0, vy0, vy1), "it takes the left of the picture area, top to bottom, nothing else"
    picture = window.viewer.picture_rect()
    assert window.viewer.mapTo(window, picture.toRect().topLeft()).x() >= px1, "the picture gives way to it"
    assert picture.width() < full.width()
    for widget in (window.bar, window.timeline_row):
        _, y0, _, y1 = box(widget, window)
        assert y1 <= py0 or y0 >= py1, f"{type(widget).__name__} is under the queue"
    assert wait(lambda: window.viewer._made_for(window.viewer._after, window.viewer.picture_rect()), 30), \
        "and is made again for the room it has left"


def test_each_row_says_what_the_file_is_and_the_chosen_one_is_marked(window, videos, tmp_path):
    capture = None
    if SAMPLES:
        capture = tmp_path / "files" / "shot.jxr"
        shutil.copy(SAMPLES[0], capture)
    window.add_paths([str(p) for p in videos[:2]] + ([str(capture)] if capture else []))
    settle(0.5)
    rows = window.queue_panel.rows
    assert [row.name for row in rows] == ["first.mp4", "second.mp4"] + (["shot.jxr"] if capture else [])
    assert [row.caption() for row in rows[:2]] == ["HDR10 video", "HDR10 video"]
    if capture:
        assert rows[2].caption() == "JXR capture" and rows[2].kind == "image"
    assert [row.selected for row in rows].count(True) == 1 and rows[0].selected, "the first file added is shown"
    assert window.queue_panel.summary.text() == f"{len(rows)} files"
    rows[1].click()
    assert wait(lambda: window._facts is not None and window._facts["path"] == str(videos[1]))
    assert window.file_label.text() == "second.mp4" and rows[1].selected and not rows[0].selected
    settle(0.4)
    pill = window.queue_panel.pill_place()
    assert pill is not None and abs(pill.top() - rows[1].y()) <= 1 and abs(pill.height() - rows[1].height()) <= 1, \
        "the mark has glided onto the chosen row"


def test_rows_are_kept_when_the_list_is_shown_again_and_follow_what_changes(window, videos):
    window.add_paths([str(p) for p in videos])
    settle(0.5)
    panel = window.queue_panel
    before = list(panel.rows)
    window._refresh()
    window._refresh()
    assert all(a is b for a, b in zip(before, panel.rows)), "a row is not made again each time the window refreshes"
    panel.rows[1].remove()
    settle(0.4)
    assert [row.name for row in panel.rows] == ["first.mp4", "third.mp4"] and len(window.queue.items) == 2
    assert panel.rows[0] is before[0] and panel.rows[1] is before[2]
    assert panel.rows[1].y() < before[2].height() * 2, "the row below has moved up into the gap"
    assert panel.summary.text() == "2 files"
    panel.rows[0].remove()
    panel.rows[0].remove()
    settle(0.3)
    assert panel.rows == [] and not window.viewer.has_picture() and window.queue_button.count() == 0


def test_each_row_gets_a_picture_of_its_file(window, videos):
    window.add_paths([str(p) for p in videos])
    panel = window.queue_panel
    assert wait(lambda: all(row.thumbnail is not None for row in panel.rows), 60), \
        "the file shown gives its picture at once, the others are read one by one behind"
    for row in panel.rows:
        assert not row.thumbnail.isNull() and row.thumbnail.width() >= 96
        assert max(row.thumbnail.width(), row.thumbnail.height()) <= 400, "a small picture, not the file's"
    first = panel.rows[0].thumbnail
    window._refresh()
    assert panel.rows[0].thumbnail is first, "and it is kept"


def test_the_queue_can_be_walked_with_the_keyboard(window, videos):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    window.add_paths([str(p) for p in videos])
    settle(0.5)
    panel = window.queue_panel
    assert panel.rows[0].toolTip() == str(videos[0]), "a row tells where its file is"
    panel.rows[0].click()
    assert window.focusWidget() is panel, "choosing a row hands the keys to the list"
    QTest.keyClick(panel, Qt.Key_Down)
    assert window.queue.selected == 1 and panel.rows[1].selected
    QTest.keyClick(panel, Qt.Key_Down)
    QTest.keyClick(panel, Qt.Key_Down)
    assert window.queue.selected == 2, "the last row is the end of the list"
    QTest.keyClick(panel, Qt.Key_Up)
    assert window.queue.selected == 1
    QTest.keyClick(panel, Qt.Key_Delete)
    assert [row.name for row in panel.rows] == ["first.mp4", "third.mp4"]
    window.viewer.set_split(0.5)
    QTest.keyClick(panel, Qt.Key_Right)
    assert window.viewer._split > 0.5, "the other keys still go to the picture"


def test_the_list_goes_to_the_file_at_work_once_and_leaves_the_scrolling_to_the_user(window, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    folder = tmp_path / "many"
    folder.mkdir()
    for index in range(12):
        clips.make_clip(folder / f"clip{index:02d}.mp4", frames=3, audio_streams=0)
    window.resize(1280, 620)
    window.add_paths([str(folder)])
    settle(0.6)
    panel = window.queue_panel
    bar = panel._scroll.verticalScrollBar()
    assert len(panel.rows) == 12 and bar.maximum() > 0, "more rows than the panel is tall: it scrolls"
    view = panel._scroll.viewport()

    def whole_in_view(row) -> bool:
        top = row.mapTo(view, row.rect().topLeft()).y()
        return top >= 0 and top + row.height() <= view.height()

    assert whole_in_view(panel.rows[0])
    bar.setValue(bar.maximum())
    panel.rows[-1].click()
    settle(0.4)
    window.resize(1280, 540)
    settle(0.4)
    assert whole_in_view(panel.rows[-1]), "the row of the file on screen stays whole in view when the panel gets shorter"
    bar.setValue(0)
    window.resize(1280, 620)
    settle(0.4)
    assert bar.value() == 0, "but not against the user, who has scrolled away from it"
    QTest.keyClick(panel, Qt.Key_Up)
    settle(0.4)
    assert panel.rows[-2].selected and whole_in_view(panel.rows[-2]), "a row chosen with the keys comes into view"
    bar.setValue(0)
    ring_right = panel.rows[0].mapTo(panel, panel.rows[0].rect().topRight()).x() - 24 + 13
    assert bar.isVisible() and ring_right <= bar.mapTo(panel, bar.rect().topLeft()).x(), \
        "the state of a row, and the cross that takes it out, stay clear of the scroll bar"
    window._converting, window._job_total = True, 12
    window._on_item_started(11)
    window._on_item_progress(11, "render", 1, 3)
    settle(0.2)
    assert bar.value() > 0, "the row whose turn has come is brought into view"
    bar.setValue(0)
    window._on_item_progress(11, "render", 2, 3)
    settle(0.2)
    assert bar.value() == 0, "and the list stays where the user put it afterwards"
    window._converting = False


def test_a_file_left_before_its_picture_came_still_gets_one(window, videos):
    """The file shown gives its preview to its row. Another row clicked at once, that preview
    never came, and the row stayed bare."""
    window.add_paths([str(p) for p in videos])
    window.select(2)
    assert wait(lambda: all(row.thumbnail is not None for row in window.queue_panel.rows), 60)
    window.select(0)
    window.select(1)
    settle(0.5)
    assert len(window.backend._thumbnails) == 0, "and nothing is read twice for a row that has its picture"


# ------------------------------------------------------------------ converting

def test_rows_tell_where_each_conversion_stands(window, videos):
    window.add_paths([str(p) for p in videos[:2]])
    settle(0.4)
    panel = window.queue_panel
    seen = {"working": set(), "captions": set(), "overall": [], "locked": []}

    def watch():
        for index, row in enumerate(panel.rows):
            if row.state == "working":
                seen["working"].add(index)
                seen["captions"].add(row.caption().split("…")[0])
                seen["locked"].append(row.removable)
        if panel.overall() is not None:
            seen["overall"].append(panel.overall())
        return idle(window)

    assert panel.overall() is None
    window.bar.convert.click()
    assert wait(watch, 240)
    settle(0.8)
    assert seen["working"] == {0, 1}, "each row was shown at work in its turn"
    assert {"Analysing", "Converting"} <= seen["captions"], f"with what was being done: {seen['captions']}"
    assert seen["overall"] and seen["overall"] == sorted(seen["overall"]) and max(seen["overall"]) <= 1.0, \
        "and the whole run moved forward, never back"
    assert not any(seen["locked"]), "a file cannot be taken out of the list while the list is converted"
    assert [row.state for row in panel.rows] == ["done", "done"]
    assert [row.caption() for row in panel.rows] == ["Saved first_SDR.mp4", "Saved second_SDR.mp4"]
    assert all(row.progress() == 1.0 and row.removable for row in panel.rows)
    assert panel.overall() is None and panel.summary.text() == "2 files, 2 done"
    panel.clear_button.click()
    settle(0.3)
    assert panel.rows == [] and window.queue.items == [], "Clear finished takes the converted files out"


def test_a_file_that_fails_says_why_on_its_row(window, videos):
    window.add_paths([str(p) for p in videos[:2]])
    settle(0.4)
    window._on_item_started(0)
    window._on_item_finished(0, "failed", "first.mp4: the disk is full", [])
    window._on_item_finished(1, "cancelled", "", [])
    rows = window.queue_panel.rows
    assert rows[0].state == "failed" and rows[0].caption() == "the disk is full", "what went wrong, without the name again"
    assert "the disk is full" in rows[0].toolTip()
    assert rows[1].state == "cancelled" and rows[1].caption() == "Stopped"
    assert window.queue_panel.summary.text() == "2 files, 1 failed"


def test_a_capture_at_work_claims_no_percentage(window, videos):
    """A capture is converted in one step: its row said 'Converting… 0 %' until it was done."""
    window.add_paths([str(p) for p in videos[:2]])
    settle(0.4)
    window._converting, window._job_total = True, 2
    window._on_item_started(0)
    row = window.queue_panel.rows[0]
    assert row.caption() == "Starting…"
    window._on_item_progress(0, "render", 0, 1)
    assert row.caption() == "Converting…" and window.bar.convert.text() == "1 of 2, converting…"
    window._on_item_progress(0, "render", 3, 12)
    assert row.caption() == "Converting… 25 %" and row.progress() == 0.25
    window._converting = False


def test_a_video_that_is_not_hdr_leaves_the_queue_and_the_window_says_why(window, videos):
    """Found when its picture is read for its row: it does not wait to be chosen or converted
    to say so, under a row that called it an HDR10 video."""
    plain = videos[0].with_name("plain.mp4")
    other = videos[0].with_name("plain too.mp4")
    clips.make_clip(plain, frames=6, hdr=False, audio_streams=0)
    clips.make_clip(other, frames=6, hdr=False, audio_streams=0)
    window.add_paths([str(videos[0]), str(plain), str(videos[1])])
    assert wait(lambda: [row.name for row in window.queue_panel.rows] == ["first.mp4", "second.mp4"], 60)
    assert "plain.mp4" in window.banner.text() and "not an HDR10 video" in window.banner.text()
    assert window.queue_button.count() == 2 and window._facts is None or window._facts["path"] == str(videos[0])
    window.add_paths([str(plain), str(other)])
    assert wait(lambda: len(window.queue_panel.rows) == 2 and window.viewer.has_picture(), 60), \
        "the first of them is refused as it is shown, the second as its picture is read"
    assert "2 files cannot be converted" in window.banner.text()
    assert "plain.mp4" in window.banner.text() and "plain too.mp4" in window.banner.text()
    assert [row.name for row in window.queue_panel.rows] == ["first.mp4", "second.mp4"]
    junk = videos[0].with_name("junk.jxr")
    junk.write_bytes(b"not a capture at all")
    window.add_paths([str(videos[2]), str(junk)])
    assert wait(lambda: [row.name for row in window.queue_panel.rows] == ["first.mp4", "second.mp4", "third.mp4"]
                and "junk.jxr" in window.banner.text(), 60), "a capture that is not one leaves the same way"


def test_pictures_are_read_while_a_video_is_analysed_but_not_while_the_queue_is_converted(window, videos):
    """A long recording shown first kept every other row bare until its analysis was over."""
    backend = window.backend
    seen = []
    backend.thumbnail_ready.connect(lambda path, _rgb: seen.append(path))
    backend._converting = True                      # as during a conversion
    window.add_paths([str(p) for p in videos])
    assert wait(lambda: window.viewer.has_picture())
    settle(1.0)
    assert seen == [] and len(backend._thumbnails) == 2, "held back for the whole conversion"
    backend._converting = False
    backend._busy = True                            # as during a background analysis
    window._thumbnail.emit(str(videos[1]), "video", window.options.to_dict(), "")
    assert wait(lambda: len(seen) >= 2, 60), "read between two frames of the analysis"
    backend._busy = False


# ------------------------------------------------------------- opening, closing

def test_closing_the_queue_is_remembered_and_the_button_brings_it_back(window, videos):
    window.add_paths([str(p) for p in videos[:2]])
    settle(0.5)
    panel = window.queue_panel
    assert panel.isVisible()
    panel.close_button.click()
    settle(0.5)
    assert not panel.isVisible() and not window.queue_button.isChecked()
    assert window.viewer._inset_left == 0, "the picture has its room back"
    window.add_paths([str(videos[2])])
    settle(0.5)
    assert not panel.isVisible() and window.queue_button.count() == 3, "closed by hand: it stays closed, the count tells"
    window.queue_button.click()
    settle(0.5)
    assert panel.isVisible() and len(panel.rows) == 3


def test_both_panels_fit_a_wide_window_and_take_turns_in_a_narrow_one(window, videos):
    window.resize(1700, 900)
    window.add_paths([str(p) for p in videos[:2]])
    assert wait(lambda: window.viewer.has_picture())
    window.advanced_button.click()
    settle(0.6)
    assert window.queue_panel.isVisible() and window.panel.isVisible()
    picture = window.viewer.picture_rect()
    left = window.viewer.mapTo(window, picture.toRect().topLeft()).x()
    right = window.viewer.mapTo(window, picture.toRect().topRight()).x()
    assert left >= box(window.queue_panel, window)[2] and right <= box(window.panel, window)[0], \
        "the picture sits between the two"
    window.resize(window.minimumWidth(), 700)
    settle(0.6)
    assert window.queue_panel.isVisible() != window.panel.isVisible(), "too narrow for both: one gives way"
    was_queue = window.queue_panel.isVisible()
    (window.advanced_button if was_queue else window.queue_button).click()
    settle(0.6)
    assert window.queue_panel.isVisible() != was_queue and window.panel.isVisible() == was_queue, \
        "opening one closes the other"


@pytest.mark.parametrize("size", [(880, 560), (1280, 800), (1920, 1080)], ids=str)
def test_nothing_in_the_queue_is_cramped_or_leaves_it(size, window, videos):
    window.resize(*size)
    window.add_paths([str(p) for p in videos])
    assert wait(lambda: window.viewer.has_picture())
    window.toggle_queue(True)
    settle(0.6)
    assert layout.cramped(window) == []
    panel = window.queue_panel
    for row in panel.rows:
        assert row.width() <= panel.width() and row.x() >= 0
        assert row.mapTo(panel, row.rect().topRight()).x() <= panel.width()
    window._on_item_started(0)
    window._on_item_progress(0, "render", 3, 12)
    settle(0.2)
    assert layout.cramped(window) == []

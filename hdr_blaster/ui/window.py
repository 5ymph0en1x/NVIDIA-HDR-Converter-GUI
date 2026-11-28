"""The window: a picture, four looks, one button. Everything else is out of the way."""
import os
import subprocess
import sys
from dataclasses import replace

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, QRect, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QImage
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget

from ..display import display_of_window, sdr_white_nits
from ..session import Options, Queue, collect, describe_output, display_name
from .advanced import PANEL_WIDTH, AdvancedPanel
from .controls import ActionBar, ElidedLabel
from .queue_panel import QUEUE_WIDTH, QueueButton, QueuePanel
from .viewer import CompareView, histogram_of
from .workers import Backend

STAGES = {"method": "Choosing the operator", "analyse": "Analysing", "render": "Converting"}
FILE_FILTER = "HDR captures and recordings (*.jxr *.mp4 *.mkv *.mov);;All files (*)"
PICTURE_ROOM = 360          # what the picture keeps between the two panels; less, and one of them closes
# The part of a file's conversion each pass stands for, by the pass it began with: a video
# already analysed for its preview is only rendered, one whose operator is chosen by the
# program is read three times. Rough shares of the time, for the line of the whole run.
PASSES = {"method": {"method": (0.0, 0.25), "analyse": (0.25, 0.6), "render": (0.6, 1.0)},
          "analyse": {"analyse": (0.0, 0.45), "render": (0.45, 1.0)},
          "render": {"render": (0.0, 1.0)}}


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class MainWindow(QMainWindow):
    _show = Signal(int, str, str, float, object, object, bool, str)
    _analyse = Signal(int, str, object, str)
    _encode = Signal(int, str, object, str)
    _convert = Signal(object, object, str)
    _thumbnail = Signal(str, str, object, str)

    def __init__(self, settings_path: str = ""):
        super().__init__()
        self.setWindowTitle("HDR Blaster")
        self.setMinimumSize(880, 560)
        self.resize(1280, 800)
        from .app import app_icon

        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self._settings = (QSettings(settings_path, QSettings.IniFormat) if settings_path
                          else QSettings("HDRBlaster", "HDRBlaster"))
        self.options = Options.from_dict({k: self._settings.value(f"options/{k}")
                                          for k in Options().to_dict() if self._settings.contains(f"options/{k}")})
        self.queue = Queue()
        self._facts = None                         # what the backend last told us about the shown file
        self._analysed_for = None                  # (path, picture key) of the finished analysis
        self._converting = False
        self._encoding_preview = False
        self._saved_paths = set()
        self._run_options = {}
        self._job_index = self._job_total = 0
        self._panel_open = False
        self._first_pass = None                    # the pass the file being converted began with
        # The window moved, or came back to the front: once it has settled, see whether SDR
        # white still stands for what the picture on screen was made for.
        self._screen_check = QTimer(self)
        self._screen_check.setSingleShot(True)
        self._screen_check.setInterval(200)
        self._screen_check.timeout.connect(self._follow_screen)
        self._left_out = []                        # files just taken out because they cannot be converted
        self._queue_open = False
        self._queue_dismissed = False              # closed by hand: it does not open by itself again

        # ----- layout
        centre = QWidget()
        self.setCentralWidget(centre)
        column = QVBoxLayout(centre)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        header = QWidget()
        header.setObjectName("header")
        header.setAttribute(Qt.WA_StyledBackground, True)
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 9, 12, 9)
        self.queue_button = QueueButton()
        self.queue_button.clicked.connect(lambda: self.toggle_queue(by_hand=True))
        row.addWidget(self.queue_button)
        row.addSpacing(12)
        self.file_label = ElidedLabel(Qt.ElideMiddle)
        self.file_label.set_full_text("No file yet")
        self.detail_label = QLabel("")
        self.detail_label.setProperty("role", "muted")
        row.addWidget(self.file_label, 3)
        row.addWidget(self.detail_label, 0)
        row.addSpacing(8)
        self.open_button = QPushButton("Add files…")
        self.open_button.setProperty("role", "ghost")
        self.open_button.setCursor(Qt.PointingHandCursor)
        self.open_button.setFocusPolicy(Qt.TabFocus)      # a click leaves no focus ring behind
        self.open_button.clicked.connect(self.browse)
        row.addWidget(self.open_button)
        row.addSpacing(6)
        self.advanced_button = QPushButton("Advanced")
        self.advanced_button.setProperty("role", "ghost")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setCursor(Qt.PointingHandCursor)
        self.advanced_button.setFocusPolicy(Qt.TabFocus)
        self.advanced_button.clicked.connect(self.toggle_advanced)
        row.addWidget(self.advanced_button)
        column.addWidget(header)

        self.banner = QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.hide()
        column.addWidget(self.banner)
        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self.banner.hide)

        self.viewer = CompareView()
        from .hdr_surface import hdr_output_available

        if hdr_output_available():
            # A screen with HDR switched on: the source is shown as the game showed it, not
            # cut at the window's white.
            self.viewer.use_hdr_surface()
            self.viewer.hdr_lost.connect(self._request_show)
        self.viewer.browse_requested.connect(self.browse)
        column.addWidget(self.viewer, 1)

        self.timeline_row = QWidget()
        self.timeline_row.setObjectName("header")
        self.timeline_row.setAttribute(Qt.WA_StyledBackground, True)
        line = QHBoxLayout(self.timeline_row)
        line.setContentsMargins(18, 8, 18, 8)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.valueChanged.connect(self._on_timeline)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setProperty("role", "muted")
        self.time_label.setMinimumWidth(self.time_label.fontMetrics().horizontalAdvance("00:00 / 00:00") + 8)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.encoded_button = QPushButton("Preview encoded MP4")
        self.encoded_button.setFocusPolicy(Qt.TabFocus)
        self.encoded_button.setToolTip("Encode the complete recording for inspection. Convert saves that same MP4.")
        self.encoded_button.clicked.connect(self._toggle_encoded)
        line.addWidget(self.encoded_button)
        line.addWidget(self.timeline, 1)
        line.addWidget(self.time_label)
        self.timeline_row.hide()
        column.addWidget(self.timeline_row)
        self._scrub = QTimer(self)
        self._scrub.setSingleShot(True)
        self._scrub.setInterval(60)
        self._scrub.timeout.connect(lambda: self._request_show())

        self.bar = ActionBar()
        self.bar.looks.set_look(None if self.options.operator else self.options.look)
        self.bar.looks.look_chosen.connect(self._on_look)
        self.bar.convert_clicked.connect(self.convert)
        self.bar.cancel_clicked.connect(self.cancel)
        self.bar.reveal_clicked.connect(self.reveal)
        column.addWidget(self.bar)

        self.panel = AdvancedPanel(self.viewer)
        self.panel.set_options(self.options)
        self.panel.changed.connect(self._on_advanced)
        self.panel.close_requested.connect(lambda: self.toggle_advanced(False))
        self.panel.hide()
        self._slide = QPropertyAnimation(self.panel, b"geometry", self)
        self._slide.setDuration(180)
        self._slide.setEasingCurve(QEasingCurve.OutCubic)
        self._slide.finished.connect(self._after_slide)
        self._slide.valueChanged.connect(self._follow_panel)
        self.viewer.resized.connect(self._place_panel)

        self.queue_panel = QueuePanel(self.viewer)
        self.queue_panel.selected.connect(self.select)
        self.queue_panel.removed.connect(self.remove)
        self.queue_panel.add_requested.connect(self.browse)
        self.queue_panel.clear_requested.connect(self.clear_finished)
        self.queue_panel.close_requested.connect(lambda: self.toggle_queue(False, by_hand=True))
        self.queue_panel.hide()
        self._slide_queue = QPropertyAnimation(self.queue_panel, b"geometry", self)
        self._slide_queue.setDuration(180)
        self._slide_queue.setEasingCurve(QEasingCurve.OutCubic)
        self._slide_queue.finished.connect(self._after_slide)
        self._slide_queue.valueChanged.connect(self._follow_panel)
        # The room for the picture changed (window resized, panel opened): once it has
        # settled, ask for the picture at its new size. It is never stretched to fit.
        self._asked_box = None
        self._refit = QTimer(self)
        self._refit.setSingleShot(True)
        self._refit.setInterval(120)
        self._refit.timeout.connect(self._fit_picture)
        self.viewer.box_changed.connect(self._refit.start)

        # ----- the background thread
        self.thread = QThread(self)
        self.backend = Backend()
        self.backend.moveToThread(self.thread)
        self._show.connect(self.backend.show)
        self._analyse.connect(self.backend.analyse)
        self._encode.connect(self.backend.encode_preview)
        self.backend.encoded_ready.connect(self._encoded_ready)
        self.backend.encoded_progress.connect(self._encoded_progress)
        self._convert.connect(self.backend.convert)
        self._thumbnail.connect(self.backend.thumbnail)
        self.backend.thumbnail_ready.connect(self._on_thumbnail)
        self.backend.thumbnail_refused.connect(self._on_thumbnail_refused)
        self.backend.preview_ready.connect(self._on_preview)
        self.backend.preview_failed.connect(self._on_preview_failed)
        self.backend.analysis_progress.connect(self._on_analysis_progress)
        self.backend.analysis_ready.connect(self._on_analysis_ready)
        self.backend.item_started.connect(self._on_item_started)
        self.backend.item_progress.connect(self._on_item_progress)
        self.backend.item_finished.connect(self._on_item_finished)
        self.backend.queue_finished.connect(self._on_queue_finished)
        # That thread waits for the engine (PyTorch) as it starts; the window does not: the
        # engine takes seconds to load, and the window has nothing to compute until a file comes.
        self.backend.ready.connect(self.panel.set_cuda)
        self.thread.started.connect(self.backend.load)
        self.thread.start()

        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self._refresh()
        self.viewer.setFocus()

    # ----------------------------------------------------------------- files
    def browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add HDR captures or recordings", "", FILE_FILTER)
        if paths:
            self.add_paths(paths)

    def add_paths(self, paths) -> None:
        items, rejected = collect(paths)
        self._left_out = []                        # what these files turn out to be is a new matter
        first_new = len(self.queue.items)
        added = self.queue.add(items)
        if rejected:
            names = ", ".join(rejected[:3]) + ("…" if len(rejected) > 3 else "")
            verb = "is" if len(rejected) == 1 else "are"
            self.notify(f"{names} {verb} not an HDR capture or recording. JXR screenshots and HDR10 videos can be converted.")
        elif not added and items:
            self.notify("Already in the list.", kind="info")
        if added:
            self.select(first_new)
            # the file shown gives its own picture to its row; the others are read behind
            for item in self.queue.items[first_new + 1:]:
                self._ask_thumbnail(item)
            # Several files: the queue shows itself, unless the user closed it, or the settings
            # are open and there is no room for both (its button then tells how many wait).
            if len(self.queue.items) > 1 and not self._queue_open and not self._queue_dismissed \
                    and (not self._panel_open or self._room_for_both()):
                self.toggle_queue(True)
        else:
            self._refresh()

    def select(self, index: int) -> None:
        if not 0 <= index < len(self.queue.items):
            return
        # A file left before its preview came (another row clicked at once) has given no
        # picture to its row: it is read behind, as for the files that were never shown.
        left = self.queue.current
        if left is not None and left is not self.queue.items[index] and self.queue_panel.wants_thumbnail(left.path):
            self._ask_thumbnail(left)
        self.queue.selected = index
        self.timeline.blockSignals(True)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self._facts = None
        self._refresh()
        self._restart_preview()

    def remove(self, index: int) -> None:
        if self._converting or not 0 <= index < len(self.queue.items):
            return
        was_current = index == self.queue.selected
        self.queue.remove(index)
        if not self.queue.items:
            self._emptied()
        elif was_current:
            self.select(self.queue.selected)
            return
        self._refresh()

    def _emptied(self) -> None:
        self._cancel_encoded()
        self.backend.latest_show += 1
        self.backend.analysis_token += 1
        self.backend.forget_thumbnails()
        self.viewer.clear()
        self._facts = None

    def clear_finished(self) -> None:
        """Take the converted files out of the queue."""
        if self._converting:
            return
        current, at = self.queue.current, self.queue.selected
        self.queue.items = [item for item in self.queue.items if item.state != "done"]
        if not self.queue.items:
            self.queue.selected = -1
            self._emptied()
        elif current in self.queue.items:
            self.queue.selected = self.queue.items.index(current)
        else:
            self.select(min(at, len(self.queue.items) - 1))
            return
        self._refresh()

    # --------------------------------------------------------------- preview
    def _fit_picture(self) -> None:
        """The picture on screen was made for another size than its place now has: ask again."""
        if self.queue.current is not None and self.viewer.has_picture() \
                and self.viewer.picture_box() != self._asked_box:
            self._request_show()

    def _time(self) -> float:
        duration = (self._facts or {}).get("duration_s", 0.0)
        return duration * self.timeline.value() / 1000.0

    def _request_show(self) -> None:
        item = self.queue.current
        if item is None:
            return
        self.backend.latest_show += 1
        self._asked_box = self.viewer.picture_box()
        self._show.emit(self.backend.latest_show, item.path, item.kind, self._time(),
                        self.options.to_dict(), self._asked_box, self.viewer.hdr_ready(), self._screen())

    def _screen(self) -> str:
        """The screen this window is on. The preview, the analysis of a video and the
        conversion all ask it what SDR white stands for: the file is the picture previewed
        only if it is made for the same white."""
        return display_of_window(int(self.winId()))

    def _follow_screen(self) -> None:
        """The window is on another screen, or 'SDR content brightness' was changed in
        Windows while the program was open: the picture on screen was made for a white
        that is no longer this screen's. It is made again, so that what is converted next
        is what is seen."""
        if self._facts is None or self.queue.current is None:
            return
        white, _ = sdr_white_nits(self._screen())
        if white != self._facts["white_nits"]:
            if self._converting:
                self._request_show()               # (the run in progress keeps the white it started with)
            else:
                self._restart_preview()

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._screen_check.start()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.ActivationChange:
            self._screen_check.start()              # back from Windows' settings, perhaps

    def _cancel_encoded(self):
        self.backend.encode_token += 1
        self._encoding_preview = False
        self.encoded_button.setText("Preview encoded MP4")

    def _toggle_encoded(self):
        item = self.queue.current
        if item is None or item.kind != "video":
            return
        if self._encoding_preview:
            self._cancel_encoded()
            self._request_show()
            return
        self.backend.encode_token += 1
        self._encoding_preview = True
        self.encoded_button.setText("Cancel MP4 preview")
        self._encode.emit(self.backend.encode_token, item.path, self.options.to_dict(), self._screen())

    def _encoded_progress(self, token, percent):
        if self._encoding_preview and token == self.backend.encode_token:
            self.encoded_button.setText(f"Cancel MP4 preview ({percent}%)")

    def _encoded_ready(self, token, error):
        if token != self.backend.encode_token:
            return
        self._cancel_encoded()
        if error:
            self.notify(error)
        self._request_show()

    def _restart_preview(self) -> None:
        """The file or the picture settings changed: show it again, and analyse it if it is a video."""
        item = self.queue.current
        if item is None:
            return
        self._cancel_encoded()
        self.backend.analysis_token += 1
        self._analysed_for = None
        self.viewer.set_note("updating" if self.viewer.has_picture() else "loading")
        self._request_show()
        if item.kind == "video" and not self._converting:
            self._analyse.emit(self.backend.analysis_token, item.path, self.options.to_dict(), self._screen())

    def _on_preview(self, request: int, before, after, facts) -> None:
        item = self.queue.current
        if request != self.backend.latest_show or item is None or facts["path"] != item.path:
            return                                         # an answer to a question we no longer ask
        self._facts = facts
        self.viewer.set_pictures(before, after, fade=False, hdr=facts.get("hdr"),
                                 white_nits=facts["white_nits"])
        self.viewer.set_result_tag(facts["result_tag"])
        self.viewer.set_note("" if facts["final"] else "preview")
        # In HDR the source is shown whole. In an SDR window it is cut at the window's white:
        # the label says at how many nits.
        self.viewer.set_source_tag("HDR source" if self.viewer.shows_hdr()
                                   else f"HDR source, white at {round(facts['white_nits'])} nits")
        self.viewer.set_histogram(histogram_of(after) if self.options.show_histogram else None)
        if self.queue_panel.wants_thumbnail(item.path):
            self._on_thumbnail(item.path, before)   # its row gets the picture just made, for nothing more
        self._refresh()
        # The window may have reached its maximized size while the first render was
        # still running. The resize notification then saw an empty viewer.
        self._fit_picture()

    def _ask_thumbnail(self, item) -> None:
        self._thumbnail.emit(item.path, item.kind, self.options.to_dict(), self._screen())

    def _on_thumbnail_refused(self, path: str, message: str) -> None:
        """A file read for its row turned out not to be one this program converts (a video
        that is not HDR10, a capture that is not HDR): as a file that cannot be shown, it
        leaves the list, and the window says why. Not during a conversion: the conversion
        says it then, on the row."""
        index = next((i for i, item in enumerate(self.queue.items) if item.path == path), -1)
        if index < 0 or index == self.queue.selected or self._converting:
            return
        self._leave_out(self.queue.items[index].name, message)
        self.queue.remove(index)
        self._refresh()

    def _leave_out(self, name: str, message: str) -> None:
        """Say why a file left the list. Several in a row (a folder with videos that are not
        HDR) are named together: each one's message would only replace the one before."""
        if not self.banner.isVisible():
            self._left_out = []
        self._left_out.append(name)
        if len(self._left_out) == 1:
            self.notify(message if name in message else f"{name}: {message}", keep=True)
        else:
            names = ", ".join(self._left_out[:3]) + ("…" if len(self._left_out) > 3 else "")
            self.notify(f"{len(self._left_out)} files cannot be converted and were left out: {names}. "
                        "JXR screenshots and HDR10 videos can be converted.", keep=True)

    def _on_thumbnail(self, path: str, rgb) -> None:
        """A picture of a file for its row in the queue: ``rgb`` is (h, w, 3) bytes, any size."""
        if not self.queue_panel.wants_thumbnail(path):
            return
        height, width = rgb.shape[:2]
        image = QImage(rgb.tobytes(), width, height, 3 * width, QImage.Format_RGB888)
        wide, tall = self.backend.THUMBNAIL
        if width > wide or height > tall:
            image = image.scaled(wide, tall, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.queue_panel.set_thumbnail(path, image.copy())

    def _on_preview_failed(self, request: int, message: str) -> None:
        item = self.queue.current
        if request != self.backend.latest_show or item is None:
            return
        # A file that cannot even be shown does not stay in the list: say why, and move on.
        self._leave_out(item.name, message)
        self.backend.analysis_token += 1
        self.queue.remove(self.queue.selected)
        self._facts = None
        self.viewer.clear()
        if self.queue.items:
            self.select(self.queue.selected)
        else:
            self._refresh()

    def _on_analysis_progress(self, token: int, done: int, total: int) -> None:
        if token == self.backend.analysis_token and not self._converting and total:
            self.viewer.set_note(f"preview, analysing {100 * done // total} %")

    def _on_analysis_ready(self, token: int) -> None:
        item = self.queue.current
        if token != self.backend.analysis_token or item is None:
            return
        self._analysed_for = (item.path, self.options.picture_key(self._screen()))
        self._request_show()                               # the same frame with stabilized parameters

    def _on_timeline(self, value: int) -> None:
        if self._facts:
            self.time_label.setText(f"{clock(self._time())} / {clock(self._facts.get('duration_s', 0.0))}")
        self._scrub.start()

    # --------------------------------------------------------------- options
    def _on_look(self, name: str) -> None:
        self._apply(replace(self.options, look=name, operator=""))
        self.panel.set_options(self.options)

    def _on_advanced(self, options: Options) -> None:
        self._apply(options)

    def _apply(self, options: Options) -> None:
        picture_changed = options.picture_key() != self.options.picture_key()
        item = self.queue.current
        if item is not None and item.kind == "image":
            picture_changed |= options.image_format != self.options.image_format
        elif item is not None and item.kind == "video":
            picture_changed |= (options.video_height, options.video_fps, options.video_quality) != (
                self.options.video_height, self.options.video_fps, self.options.video_quality)
        histogram_changed = options.show_histogram != self.options.show_histogram
        self.options = options
        for name, value in options.to_dict().items():
            self._settings.setValue(f"options/{name}", value)
        if picture_changed and not self._converting:
            self._restart_preview()
        elif histogram_changed:
            self._request_show()
        self._refresh()

    # ------------------------------------------------------------- converting
    def convert(self) -> None:
        if self._converting:
            return
        jobs = [(i, item.path, item.kind) for i, item in enumerate(self.queue.items)
                if item.state in ("waiting", "cancelled", "failed")]
        if not jobs:
            current = self.queue.current
            if current is None:
                return
            jobs = [(self.queue.selected, current.path, current.kind)]     # convert it again, on request
        self._converting = True
        self._job_index, self._job_total = 0, len(jobs)
        self._saved_paths.clear()
        self.backend.stop_convert = False
        self.banner.hide()
        self.bar.convert.set_working("Starting…", 0.0)
        self.queue_panel.set_overall(0.0)
        self._refresh()
        # the screen is named once, here: the whole run is made for the white of the
        # screen the window is on as the conversion is asked for
        screen = self._screen()
        self._run_options = dict(self.options.to_dict(), white_nits=sdr_white_nits(screen)[0])
        self._convert.emit(jobs, self._run_options, screen)

    def cancel(self) -> None:
        if self._converting:
            self.backend.stop_convert = True
            self._cancel_encoded()
            self.bar.convert.set_working("Stopping…", 0.0)

    def _on_item_started(self, index: int) -> None:
        self._job_index += 1
        self._first_pass = None
        self.queue.items[index].state = "working"
        self._refresh()
        self.queue_panel.set_progress(index, "Starting…", None)

    def _on_item_progress(self, index: int, stage: str, done: int, total: int) -> None:
        if not self._converting or self.backend.stop_convert:
            return
        # Without a frame count there is no percentage to give; with an estimated one the
        # count may run past it.
        fraction = min(1.0, done / total) if total else 0.0
        prefix = f"{self._job_index} of {self._job_total}, " if self._job_total > 1 else ""
        verb = STAGES.get(stage, "Converting")
        # A capture is one step: nothing to count, it is being converted and then it is done.
        counted = total > 1
        percent = f" {int(100 * fraction)} %" if counted else ""
        self.bar.convert.set_working(f"{prefix}{verb.lower() if prefix else verb}…{percent}", fraction)
        self.queue_panel.set_progress(index, f"{verb}…{percent}", fraction if counted else None)
        # the row follows the pass it names; the line of the whole run never goes back
        if self._first_pass is None:
            self._first_pass = stage if stage in PASSES else "render"
        start, end = PASSES[self._first_pass].get(stage, (0.0, 1.0))
        whole = (self._job_index - 1 + start + (end - start) * fraction) / max(1, self._job_total)
        self.queue_panel.set_overall(max(whole, self.queue_panel.overall() or 0.0))

    def _on_item_finished(self, index: int, state: str, message: str, outputs) -> None:
        item = self.queue.items[index]
        item.state, item.message, item.outputs = state, message, list(outputs)
        if state == "done":
            item.output_settings = self._run_options
            self._saved_paths.add(item.path)
        elif state == "failed":
            self.notify(message if item.name in message else f"{item.name}: {message}")
        self._refresh()

    def _on_queue_finished(self) -> None:
        self._converting = False
        self.bar.convert.set_idle()
        self.queue_panel.set_overall(None)
        self._refresh()
        item = self.queue.current
        if item is not None and item.kind == "video":
            self._restart_preview()                        # display the exported bitstream

    def reveal(self) -> None:
        current = self.queue.current
        if current is None or not current.outputs:
            return
        target = current.outputs[0]
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(target)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(target)))

    # ------------------------------------------------------------------ view
    def notify(self, text: str, kind: str = "error", keep: bool = False) -> None:
        if not keep:
            self._left_out = []                    # another matter is on the banner now
        self.banner.setProperty("kind", kind)
        self.banner.style().unpolish(self.banner)
        self.banner.style().polish(self.banner)
        self.banner.setText(text)
        self.banner.show()
        self._banner_timer.start(6000 if kind == "info" else 12000)

    def _refresh(self) -> None:
        item = self.queue.current
        if item is None:
            self.file_label.set_full_text("No file yet")
            self.detail_label.setText("")
        else:
            self.file_label.set_full_text(item.name, tooltip=item.path)
            facts = self._facts if (self._facts and self._facts.get("path") == item.path) else None
            if facts:
                kind = "HDR10 video" if item.kind == "video" else "JXR capture"
                detail = f"{facts['width']} × {facts['height']}, {kind}, peak {round(facts['peak_nits'])} nits"
                if facts.get("method") and self.options.method() is None:
                    detail += f", Auto chose {display_name(facts['method'])}"
                elif self.options.method() == "neural" and facts.get("neural_fallback"):
                    detail += ", Neural unavailable, using Perceptual"
                self.detail_label.setText(detail)
            else:
                self.detail_label.setText("")
        is_video = item is not None and item.kind == "video" and bool(self._facts)
        self.timeline_row.setVisible(is_video)
        self.encoded_button.setEnabled(not self._converting)
        self.encoded_button.setVisible(is_video and (self._facts or {}).get("result_tag") != "Encoded MP4 SDR")
        if is_video:
            self.time_label.setText(f"{clock(self._time())} / {clock(self._facts.get('duration_s', 0.0))}")

        self.queue_panel.set_items(self.queue.items, self.queue.selected, converting=self._converting)
        self.queue_button.set_count(len(self.queue.items))
        self.bar.looks.set_look(None if self.options.operator else self.options.look)
        self.bar.looks.setEnabled(not self._converting)
        self.panel.setEnabled(not self._converting)
        saved = False
        if item and item.outputs and self._facts and self._facts.get("path") == item.path:
            settings = dict(self.options.to_dict(), white_nits=self._facts["white_nits"])
            saved = item.output_settings == settings and not self._converting
        if saved:
            count = len(self._saved_paths) if item.path in self._saved_paths else 1
            text = f"Saved {count} files" if count > 1 else f"Saved {os.path.basename(item.outputs[0])}"
            self.bar.set_summary(text, done=True, tooltip=os.linesep.join(item.outputs))
        else:
            self.bar.set_summary(describe_output(item.kind, self.options) if item else "")
        self.bar.cancel.setVisible(self._converting)
        self.bar.reveal.setVisible(saved)
        if not self._converting:
            waiting = [i for i in self.queue.items if i.state in ("waiting", "cancelled", "failed")]
            if len(waiting) > 1:
                self.bar.convert.set_idle(f"Convert {len(waiting)} files")
            elif item is not None and not waiting:
                self.bar.convert.set_idle("Convert again")
            else:
                self.bar.convert.set_idle("Convert")
            self.bar.convert.setEnabled(item is not None)

    # ------------------------------------------------------------ the panels
    def toggle_advanced(self, show=None) -> None:
        """Slide the advanced panel in or out. Without an argument, the other way round."""
        if not isinstance(show, bool):
            show = not self._panel_open
        self._panel_open = show
        self.advanced_button.setChecked(show)
        if show and self._queue_open and not self._room_for_both():
            self.toggle_queue(False)                # too narrow for both: the one asked for stays
        self._slide_it(self._slide, self.panel, self._panel_places(), show)

    def toggle_queue(self, show=None, by_hand: bool = False) -> None:
        """Slide the queue in or out, on the left. ``by_hand``: the user did it. A queue the
        user closed stays closed when more files come; its button still says how many."""
        if not isinstance(show, bool):
            show = not self._queue_open
        self._queue_open = show
        if by_hand:
            self._queue_dismissed = not show
        self.queue_button.setChecked(show)
        if show and self._panel_open and not self._room_for_both():
            self.toggle_advanced(False)
        self._slide_it(self._slide_queue, self.queue_panel, self._queue_places(), show)

    def _slide_it(self, slide, panel, places, show: bool) -> None:
        hidden, shown = places
        slide.stop()
        if show:
            if not panel.isVisible():
                panel.setGeometry(hidden)
            panel.show()
            panel.raise_()
        slide.setStartValue(panel.geometry())
        slide.setEndValue(shown if show else hidden)
        slide.start()

    def _room_for_both(self) -> bool:
        return self.viewer.width() - PANEL_WIDTH - QUEUE_WIDTH >= PICTURE_ROOM

    def _panel_places(self):
        """(hidden, shown) rectangles of the panel, inside the picture area. Being a child of
        the viewer, the panel can never cover the timeline, the banner or the action bar."""
        width, height = self.viewer.width(), self.viewer.height()
        return QRect(width, 0, PANEL_WIDTH, height), QRect(width - PANEL_WIDTH, 0, PANEL_WIDTH, height)

    def _queue_places(self):
        """The same for the queue, on the left edge."""
        height = self.viewer.height()
        return QRect(-QUEUE_WIDTH, 0, QUEUE_WIDTH, height), QRect(0, 0, QUEUE_WIDTH, height)

    def _place_panel(self) -> None:
        """The picture area changed size: keep each open panel on its edge. Too narrow for
        both, the settings give way to the files."""
        if self._panel_open and self._queue_open and not self._room_for_both():
            self.toggle_advanced(False)
        for is_open, slide, panel, places in ((self._panel_open, self._slide, self.panel, self._panel_places()),
                                              (self._queue_open, self._slide_queue, self.queue_panel,
                                               self._queue_places())):
            if is_open and slide.state() != QPropertyAnimation.Running:
                panel.setGeometry(places[1])
            elif is_open:
                slide.setEndValue(places[1])
        self._follow_panel()

    def _after_slide(self) -> None:
        if not self._panel_open and self._slide.state() != QPropertyAnimation.Running:
            self.panel.hide()
        if not self._queue_open and self._slide_queue.state() != QPropertyAnimation.Running:
            self.queue_panel.hide()
        self._place_panel()

    def _follow_panel(self, *_args) -> None:
        """The picture gives way to the panels instead of being covered by them."""
        covered = self.viewer.width() - self.panel.x() if self.panel.isVisible() else 0
        self.viewer.set_inset_right(covered)
        queue = self.queue_panel
        self.viewer.set_inset_left(queue.x() + queue.width() if queue.isVisible() else 0)

    # ---------------------------------------------------------- drag and drop
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.viewer.set_drop_hover(True)

    def dragLeaveEvent(self, event) -> None:
        self.viewer.set_drop_hover(False)

    def dropEvent(self, event) -> None:
        self.viewer.set_drop_hover(False)
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.add_paths(paths)

    # ------------------------------------------------------------------ close
    def closeEvent(self, event) -> None:
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.sync()
        self.backend.closing = True
        self.backend.stop_convert = True
        self.backend.analysis_token += 1
        self.thread.quit()
        if self.thread.wait(30000):
            self.backend.release_encoded_preview()
        super().closeEvent(event)

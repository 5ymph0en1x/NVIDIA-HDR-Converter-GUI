"""The window: a picture, four looks, one button. Everything else is out of the way."""
import os
import subprocess
import sys
from dataclasses import replace

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget

from ..display import display_of_window
from ..session import Options, Queue, collect, describe_output, display_name
from .advanced import PANEL_WIDTH, AdvancedPanel
from .controls import ActionBar, ElidedLabel
from .queue_strip import QueueStrip
from .viewer import CompareView, histogram_of
from .workers import Backend

STAGES = {"method": "Choosing the operator", "analyse": "Analysing", "render": "Converting"}
FILE_FILTER = "HDR captures and recordings (*.jxr *.mp4 *.mkv *.mov);;All files (*)"


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class MainWindow(QMainWindow):
    _show = Signal(int, str, str, float, object, object, bool, str)
    _analyse = Signal(int, str, object)
    _convert = Signal(object, object)

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
        self._last_outputs = []
        self._saved = None                     # (text, file, settings) of the last success
        self._job_index = self._job_total = self._run_saved = 0
        self._panel_open = False

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
        row.setContentsMargins(18, 9, 12, 9)
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

        self.strip = QueueStrip()
        self.strip.selected.connect(self.select)
        self.strip.removed.connect(self.remove)
        self.strip.hide()
        column.addWidget(self.strip)

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
        self._convert.connect(self.backend.convert)
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
        else:
            self._refresh()

    def select(self, index: int) -> None:
        if not 0 <= index < len(self.queue.items):
            return
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
            self.backend.latest_show += 1
            self.backend.analysis_token += 1
            self.viewer.clear()
            self._facts = None
        elif was_current:
            self.select(self.queue.selected)
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
                        self.options.to_dict(), self._asked_box, self.viewer.hdr_ready(),
                        display_of_window(int(self.winId())))

    def _restart_preview(self) -> None:
        """The file or the picture settings changed: show it again, and analyse it if it is a video."""
        item = self.queue.current
        if item is None:
            return
        self.backend.analysis_token += 1
        self._analysed_for = None
        self.viewer.set_note("loading" if not self.viewer.has_picture() else self.viewer._note)
        self._request_show()
        if item.kind == "video" and not self._converting:
            self._analyse.emit(self.backend.analysis_token, item.path, self.options.to_dict())

    def _on_preview(self, request: int, before, after, facts) -> None:
        item = self.queue.current
        if request != self.backend.latest_show or item is None or facts["path"] != item.path:
            return                                         # an answer to a question we no longer ask
        first = self._facts is None or self._facts.get("path") != facts["path"]
        self._facts = facts
        self.viewer.set_pictures(before, after, fade=not first, hdr=facts.get("hdr"),
                                 white_nits=facts["white_nits"])
        self.viewer.set_note("" if facts["final"] else "preview")
        # In HDR the source is shown whole. In an SDR window it is cut at the window's white:
        # the label says at how many nits.
        self.viewer.set_source_tag("HDR source" if self.viewer.shows_hdr()
                                   else f"HDR source, white at {round(facts['white_nits'])} nits")
        self.viewer.set_histogram(histogram_of(after) if self.options.show_histogram else None)
        self._refresh()

    def _on_preview_failed(self, request: int, message: str) -> None:
        item = self.queue.current
        if request != self.backend.latest_show or item is None:
            return
        # A file that cannot even be shown does not stay in the list: say why, and move on.
        self.notify(message if item.name in message else f"{item.name}: {message}")
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
        self._analysed_for = (item.path, self.options.picture_key())
        self._request_show()                               # the same frame, now exact

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
        self._job_index, self._job_total, self._run_saved = 0, len(jobs), 0
        self._last_outputs = []
        self._saved = None
        self.backend.stop_convert = False
        self.banner.hide()
        self.bar.convert.set_working("Starting…", 0.0)
        self._refresh()
        self._convert.emit(jobs, self.options.to_dict())

    def cancel(self) -> None:
        if self._converting:
            self.backend.stop_convert = True
            self.bar.convert.set_working("Stopping…", 0.0)

    def _on_item_started(self, index: int) -> None:
        self._job_index += 1
        self.queue.items[index].state = "working"
        self._refresh()

    def _on_item_progress(self, index: int, stage: str, done: int, total: int) -> None:
        if not self._converting or self.backend.stop_convert:
            return
        # Without a frame count there is no percentage to give; with an estimated one the
        # count may run past it.
        fraction = min(1.0, done / total) if total else 0.0
        prefix = f"{self._job_index} of {self._job_total}, " if self._job_total > 1 else ""
        verb = STAGES.get(stage, "Converting")
        percent = f" {int(100 * fraction)} %" if total else ""
        self.bar.convert.set_working(f"{prefix}{verb.lower() if prefix else verb}…{percent}", fraction)

    def _on_item_finished(self, index: int, state: str, message: str, outputs) -> None:
        item = self.queue.items[index]
        item.state, item.message, item.outputs = state, message, list(outputs)
        if state == "done":
            self._last_outputs = list(outputs)
            self._run_saved += 1
        elif state == "failed":
            self.notify(message if item.name in message else f"{item.name}: {message}")
        self._refresh()

    def _on_queue_finished(self) -> None:
        self._converting = False
        self.bar.convert.set_idle()
        if self._last_outputs:
            # The news of a success takes no line of its own (a line above the picture pushed
            # it down at every conversion): it is given where the output is described, next to
            # 'Show in folder', for as long as the same file and settings are on screen.
            name = os.path.basename(self._last_outputs[0])
            current = self.queue.current
            self._saved = (f"Saved {name}" if self._run_saved == 1 else f"Saved {self._run_saved} files",
                           current.path if current else None, self.options.to_dict())
        self._refresh()
        item = self.queue.current
        if item is not None and item.kind == "video":
            self._restart_preview()                        # exact at once if this video was just analysed

    def reveal(self) -> None:
        if not self._last_outputs:
            return
        target = self._last_outputs[0]
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(target)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(target)))

    # ------------------------------------------------------------------ view
    def notify(self, text: str, kind: str = "error") -> None:
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
                self.detail_label.setText(detail)
            else:
                self.detail_label.setText("")
        is_video = item is not None and item.kind == "video" and bool(self._facts)
        self.timeline_row.setVisible(is_video)
        if is_video:
            self.time_label.setText(f"{clock(self._time())} / {clock(self._facts.get('duration_s', 0.0))}")

        self.strip.set_items(self.queue.items, self.queue.selected)
        self.bar.looks.set_look(None if self.options.operator else self.options.look)
        self.bar.looks.setEnabled(not self._converting)
        self.panel.setEnabled(not self._converting)
        saved = self._saved
        if saved and not self._converting and item is not None and saved[1] == item.path \
                and saved[2] == self.options.to_dict():
            self.bar.set_summary(saved[0], done=True, tooltip=os.linesep.join(self._last_outputs))
        else:
            self.bar.set_summary(describe_output(item.kind, self.options) if item else "")
        self.bar.cancel.setVisible(self._converting)
        self.bar.reveal.setVisible(bool(self._last_outputs) and not self._converting)
        if not self._converting:
            waiting = [i for i in self.queue.items if i.state in ("waiting", "cancelled", "failed")]
            if len(waiting) > 1:
                self.bar.convert.set_idle(f"Convert {len(waiting)} files")
            elif item is not None and not waiting:
                self.bar.convert.set_idle("Convert again")
            else:
                self.bar.convert.set_idle("Convert")
            self.bar.convert.setEnabled(item is not None)

    # -------------------------------------------------------- advanced panel
    def toggle_advanced(self, show=None) -> None:
        """Slide the advanced panel in or out. Without an argument, the other way round."""
        if not isinstance(show, bool):
            show = not self._panel_open
        self._panel_open = show
        self.advanced_button.setChecked(show)
        hidden, shown = self._panel_places()
        self._slide.stop()
        if show:
            if not self.panel.isVisible():
                self.panel.setGeometry(hidden)
            self.panel.show()
            self.panel.raise_()
        self._slide.setStartValue(self.panel.geometry())
        self._slide.setEndValue(shown if show else hidden)
        self._slide.start()

    def _panel_places(self):
        """(hidden, shown) rectangles of the panel, inside the picture area. Being a child of
        the viewer, the panel can never cover the timeline, the banner or the action bar."""
        width, height = self.viewer.width(), self.viewer.height()
        return QRect(width, 0, PANEL_WIDTH, height), QRect(width - PANEL_WIDTH, 0, PANEL_WIDTH, height)

    def _place_panel(self) -> None:
        """The picture area changed size: keep the open panel on its right edge."""
        if self._panel_open and self._slide.state() != QPropertyAnimation.Running:
            self.panel.setGeometry(self._panel_places()[1])
        elif self._panel_open:
            self._slide.setEndValue(self._panel_places()[1])
        self._follow_panel()

    def _after_slide(self) -> None:
        if not self._panel_open:
            self.panel.hide()
        self._place_panel()

    def _follow_panel(self, *_args) -> None:
        """The picture gives way to the panel instead of being covered by it."""
        covered = self.viewer.width() - self.panel.x() if self.panel.isVisible() else 0
        self.viewer.set_inset_right(covered)

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
        self.thread.wait(30000)
        super().closeEvent(event)

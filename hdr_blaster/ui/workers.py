"""The one background thread that does every computation for the window.

One thread, because the engine and the GPU are used by one task at a time. Three kinds
of work arrive from the window:

* **show**: load a file (or a frame of a video) and render its preview. Quick. The latest
  request wins; older ones still waiting are dropped.
* **analyse**: the first pass over a video, so that its preview becomes exact and the
  conversion can skip that pass. Long. Abandoned as soon as the file or the picture
  settings change.
* **convert**: the queue. Long. Stopped by ``stop_convert``.

Long tasks let the thread's event loop run between frames, so a preview request is served
within a frame or two even in the middle of a conversion. They never nest: a long task
that arrives while another runs waits for it to unwind.
"""
import logging
from collections import deque

from PySide6.QtCore import QCoreApplication, QObject, Signal, Slot

from .. import preview
from ..display import sdr_white_nits
from ..session import Engines, Item, Options, convert_item
from ..video.job import Cancelled, analyse_video, exposure_of, frames_to_keep
from ..video.reader import UnsupportedVideo


class Backend(QObject):
    preview_ready = Signal(int, object, object, object)     # request, before rgb8, after rgb8, facts
    preview_failed = Signal(int, str)
    analysis_progress = Signal(int, int, int)                # token, frames done, total
    analysis_ready = Signal(int)                             # token
    item_started = Signal(int)
    item_progress = Signal(int, str, int, int)               # index, stage, done, total
    item_finished = Signal(int, str, str, object)            # index, state, message, files written
    queue_finished = Signal()

    def __init__(self):
        super().__init__()
        self.engines = Engines()
        # Written by the window's thread, read here: plain integers and a flag.
        self.latest_show = 0
        self.analysis_token = 0
        self.stop_convert = False
        self.closing = False
        self._source = None
        self._source_key = None
        self._light = None                 # motion blur: the light of the frames a picture stands for
        self._light_key = None
        self.analysis = None
        self._pending = deque()
        self._busy = False

    # ---------------------------------------------------------------- preview
    @Slot(int, str, str, float, object, object, bool, str)
    def show(self, request: int, path: str, kind: str, time_s: float, options: dict, box, hdr: bool = False,
             screen: str = "") -> None:
        """``box``: the room the window has for the picture, (width, height) in screen pixels.
        The two pictures sent back fit it exactly: the window draws them as they are.
        ``hdr``: the window has an HDR surface; send it the two pictures in scRGB as well.
        ``screen``: the name of the screen the window is on."""
        if request < self.latest_show or self.closing:
            return                                        # a newer request is already waiting
        try:
            chosen = Options.from_dict(options)
            engine, device = self.engines.get(chosen)
            white, white_source = sdr_white_nits(screen)   # what this screen's white stands for, now
            # Gamma and Exposure are not part of the key: they never change the source. Nor is
            # the size shown: the source is kept at its real size, whatever the window's.
            key = (path, kind, round(time_s, 4), device.type, white)
            if key != self._source_key:
                if kind == "video":
                    self._source = preview.load_video_frame(path, time_s, device, white_nits=white)
                else:
                    self._source = preview.load_image(path, device, white_nits=white)
                self._source_key = key
            self._source.pre_gamma, self._source.exposure = chosen.pre_gamma, chosen.exposure
            preview.fit(self._source, box=(int(box[0]), int(box[1])))
            source, params, final, light = self._source, None, kind == "image", None
            analysis = self.analysis
            video = chosen.video_settings() if kind == "video" else None
            if video is not None and analysis is not None and analysis.src == path and analysis.matches(video):
                index = analysis.index_of(source.pts)
                keep = frames_to_keep(analysis.times, video.frame_rate, analysis.info.average_rate)
                if video.motion_blur and not all(keep):
                    # The file will show, at this moment, the picture of the kept frame before
                    # it, made of the light of the frames that picture stands for.
                    index, count = exposure_of(keep, set(analysis.cuts), index)
                    key = (path, index, count, device.type)
                    if key != self._light_key:
                        self._light = preview.blended_light(path, analysis.times[index], count, device,
                                                            analysis.info)
                        self._light_key = key
                    light = self._light
                params = analysis.params[index]
                final = True
            after = preview.render_preview(engine, source, chosen.render_settings(kind), params, light=light)
            facts = {"kind": kind, "path": path, "width": source.width, "height": source.height,
                     "time_s": source.time_s, "duration_s": source.duration_s, "final": final,
                     "method": source.method if final else None,
                     "white_nits": source.white_nits, "white_source": white_source,
                     "peak_nits": source.peak_nits,
                     # for an HDR surface, and only while this screen really is in HDR
                     "hdr": preview.hdr_pictures(source, after) if hdr and white_source == "display" else None}
            self.preview_ready.emit(request, preview.to_rgb8(source.before), preview.to_rgb8(after), facts)
        except Exception as exc:
            logging.exception("preview failed")
            self._source_key = None
            self.preview_failed.emit(request, str(exc))

    # --------------------------------------------------------- long tasks, queued
    @Slot(int, str, object)
    def analyse(self, token: int, path: str, options: dict) -> None:
        self._pending.append(("analyse", token, path, options))
        self._drain()

    @Slot(object, object)
    def convert(self, jobs: list, options: dict) -> None:
        self._pending.append(("convert", jobs, options))
        self._drain()

    def _drain(self) -> None:
        if self._busy:
            return                                        # the running task will pick it up
        self._busy = True
        try:
            while self._pending and not self.closing:
                task = self._pending.popleft()
                if task[0] == "analyse":
                    self._run_analysis(*task[1:])
                else:
                    self._run_queue(*task[1:])
        finally:
            self._busy = False

    @staticmethod
    def _breathe() -> None:
        """Let preview requests (and newly queued tasks) through between two frames."""
        QCoreApplication.processEvents()

    def _run_analysis(self, token: int, path: str, options: dict) -> None:
        if token != self.analysis_token:
            return
        chosen = Options.from_dict(options)
        if self.analysis is not None and self.analysis.src == path \
                and self.analysis.matches(chosen.video_settings()):
            self.analysis_ready.emit(token)               # already known: nothing to do again
            return

        def progress(stage, done, total):
            if done % 8 == 0 or done == total:
                self.analysis_progress.emit(token, done, total)
                self._breathe()

        def cancelled():
            return token != self.analysis_token or self.closing or any(t[0] == "convert" for t in self._pending)

        try:
            engine, device = self.engines.get(chosen)
            self.analysis = analyse_video(path, chosen.video_settings(), device, progress, cancelled, engine)
            self.analysis_ready.emit(token)
        except Cancelled:
            pass
        except Exception:
            logging.exception("background analysis failed")   # the conversion will report it properly

    def _run_queue(self, jobs: list, options: dict) -> None:
        """Convert the jobs in order. Whatever happens, ``queue_finished`` is emitted: the
        window must never be left waiting."""
        try:
            chosen = Options.from_dict(options)
            for index, path, kind in jobs:
                if self.stop_convert or self.closing:
                    break
                self.item_started.emit(index)

                def progress(stage, done, total, index=index):
                    if done % 4 == 0 or done == total:
                        self.item_progress.emit(index, stage, done, total)
                        self._breathe()

                def cancelled():
                    return self.stop_convert or self.closing

                try:
                    engine, device = self.engines.get(chosen)
                    analysis = self.analysis
                    if kind == "video":
                        settings = chosen.video_settings()
                        if analysis is None or analysis.src != path or not analysis.matches(settings):
                            # Analysed here rather than inside convert_item, so that the result is
                            # kept: the preview of this video is exact from now on.
                            analysis = analyse_video(path, settings, device, progress, cancelled, engine)
                            self.analysis = analysis
                    written = convert_item(Item(path=path, kind=kind), chosen, engine, device, progress,
                                           cancelled=cancelled, analysis=analysis)
                    self.item_finished.emit(index, "done", "", written)
                except Cancelled:
                    self.item_finished.emit(index, "cancelled", "", [])
                    break
                except UnsupportedVideo as exc:
                    self.item_finished.emit(index, "failed", str(exc), [])
                except Exception as exc:
                    logging.exception("conversion failed")
                    self.item_finished.emit(index, "failed", str(exc), [])
        finally:
            self.queue_finished.emit()

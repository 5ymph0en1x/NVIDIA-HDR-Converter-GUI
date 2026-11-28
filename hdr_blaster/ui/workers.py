"""The one background thread that does every computation for the window.

One thread, because the engine and the GPU are used by one task at a time. Three kinds
of work arrive from the window:

* **show**: load a file (or a frame of a video) and render its preview. Quick. The latest
  request wins; older ones still waiting are dropped.
* **analyse**: the first pass over a video, so its preview uses stabilized parameters and the
  conversion can skip that pass. Long. Abandoned as soon as the file or the picture
  settings change.
* **convert**: the queue. Long. Stopped by ``stop_convert``.

Each of the three is told which screen the window is on: what SDR white stands for is
asked of that screen, by all three alike. The file is the picture that was previewed
only if both were made for the same white.

Long tasks let the thread's event loop run between frames, so a preview request is served
within a frame or two even in the middle of a conversion. They never nest: a long task
that arrives while another runs waits for it to unwind.

The engine is not loaded when the window opens: importing PyTorch takes seconds, and the
window is on screen meanwhile (``start_loading_engine``, ``load_engine``). So nothing
that computes is imported at the top of this module, which the window imports: each task
waits for the engine, then imports what it uses.
"""
import logging
import os
import threading
import tempfile
from collections import deque

from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal, Slot

from ..display import sdr_white_nits
from ..session import Engines, Item, Options, convert_item, output_base

# The engine is imported by a thread started by Python, never by the thread of the Backend
# itself. PyTorch's bindings remember the Python thread state of the thread that imports
# them, and a slot of a QThread runs under a state made for that one call: imported there,
# PyTorch later attaches to a state that no longer exists, and the program dies on the
# first model it loads ("Fatal Python error: _PyThreadState_Attach"). Found by starting the
# program with a file; tests/test_startup.py does it again.
_engine_thread = None
_engine_error = None
_engine_lock = threading.Lock()


def _import_engine() -> None:
    global _engine_error
    try:
        from .. import configure_torch

        configure_torch()
        from .. import preview  # noqa: F401
        from ..video import job  # noqa: F401
    except Exception as exc:
        logging.exception("the engine could not be loaded")
        _engine_error = exc


def start_loading_engine() -> None:
    """Begin importing the engine, and return at once."""
    global _engine_thread
    with _engine_lock:
        if _engine_thread is None:
            _engine_thread = threading.Thread(target=_import_engine, name="hdr-blaster-engine", daemon=True)
            _engine_thread.start()


def load_engine() -> None:
    """Return once the engine is imported and set up. Raises what stopped it, each time."""
    start_loading_engine()
    _engine_thread.join()
    if _engine_error is not None:
        raise _engine_error


def _file_key(path):
    stat = os.stat(path)
    return path, stat.st_size, stat.st_mtime_ns


def _capture_key(file_key, options, settings, device):
    return (file_key, settings, options.pre_gamma, options.exposure,
            options.image_format, device.type, options.use_fp16)


class Backend(QObject):
    ready = Signal(bool)                                     # the engine is loaded; a CUDA card can be used
    preview_ready = Signal(int, object, object, object)     # request, before rgb8, after rgb8, facts
    preview_failed = Signal(int, str)
    analysis_progress = Signal(int, int, int)                # token, frames done, total
    encoded_ready = Signal(int, str)                     # request token, error if any
    encoded_progress = Signal(int, int)                  # request token, percent
    analysis_ready = Signal(int)                             # token
    item_started = Signal(int)
    item_progress = Signal(int, str, int, int)               # index, stage, done, total
    item_finished = Signal(int, str, str, object)            # index, state, message, files written
    queue_finished = Signal()
    thumbnail_ready = Signal(str, object)                    # path, rgb8 picture of the file, small
    thumbnail_refused = Signal(str, str)                     # path, why this file cannot be converted

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
        self._capture_key = self._capture = None
        self._light = None                 # motion blur: the light of the frames a picture stands for
        self._light_key = None
        self.analysis = None
        self._analysis_input = None
        self.encode_token = 0
        self._preview_dir = None
        self._encoded = {}                         # source path -> (settings, output file identity)
        self._pending = deque()
        self._busy = False
        self._thumbnails = deque()                 # files of the queue whose picture is still to be read
        self._converting = False                   # the queue is being converted: no picture is read meanwhile
        self._thumbnail_due = False                # the next picture is already asked of this thread's loop

    # ----------------------------------------------------------------- engine
    @Slot()
    def load(self) -> None:
        """Wait for the engine, as this thread starts, and tell the window what it found.
        Whatever the window asks for in the meantime waits its turn behind this."""
        try:
            load_engine()
        except Exception:
            return        # logged; the first file shown fails the same way, and says why
        import torch

        self.ready.emit(torch.cuda.is_available())

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
            load_engine()
            from .. import preview
            from ..video.job import exposure_of, frames_to_keep, output_size

            chosen = Options.from_dict(options)
            engine, device = self.engines.get(chosen)
            white, white_source = sdr_white_nits(screen)   # what this screen's white stands for, now
            # Gamma and Exposure are not part of the key: they never change the source. Nor is
            # the size shown: the source is kept at its real size, whatever the window's.
            file_key = _file_key(path)
            key = (file_key, kind, round(time_s, 4), device.type, white)
            if key != self._source_key:
                self._capture_key = self._capture = None
                if kind == "video":
                    self._source = preview.load_video_frame(path, time_s, device, white_nits=white)
                else:
                    self._source = preview.load_image(path, device, white_nits=white)
                self._source_key = key
            self._source.pre_gamma, self._source.exposure = chosen.pre_gamma, chosen.exposure
            preview.fit(self._source, box=(int(box[0]), int(box[1])))
            source, params, final, light = self._source, None, kind == "image", None
            video = chosen.video_settings(screen, white) if kind == "video" else None
            analysis = self._matching_analysis(file_key, chosen, video) if video else None
            encoded = self._encoded_path(file_key, video, chosen) if video is not None else None
            if video is not None and not encoded and analysis is not None:
                index = analysis.index_of(source.pts)
                keep = frames_to_keep(analysis.times, video.frame_rate, analysis.info.average_rate)
                if not keep[index] or video.motion_blur and not all(keep):
                    # The file will show, at this moment, the picture of the kept frame before
                    # it, made of the light of the frames that picture stands for.
                    index, count = exposure_of(keep, set(analysis.cuts), index)
                    count = count if video.motion_blur else 1
                    key = (file_key, index, count, device.type)
                    if key != self._light_key:
                        self._light = preview.blended_light(path, analysis.times[index], count, device,
                                                            analysis.info)
                        self._light_key = key
                    light = self._light
                params = analysis.params[index]
                final = True
            if kind == "image":
                from ..engine.pipeline import prepare_capture

                settings = chosen.render_settings(kind, screen, white)
                capture_key = _capture_key(file_key, chosen, settings, device)
                if self._capture_key != capture_key:
                    self._capture = prepare_capture(engine, source.linear, source.capture_loader, settings,
                                                    chosen.pre_gamma, chosen.exposure, chosen.image_format)
                    self._capture_key = capture_key
                source.method = self._capture.method
                after = preview.for_display(self._capture.still.pixels(device), source.size)
                result_tag = "TIFF 16-bit SDR" if chosen.image_format == "TIFF" else "JPEG SDR"
            elif encoded:
                from ..video.encoded_preview import frame_at

                after = preview.for_display(frame_at(encoded, source.time_s, device), source.size)
                source.method = analysis.method if analysis else chosen.method()
                result_tag, final = "Encoded MP4 SDR", True
            else:
                after = preview.render_preview(engine, source, video.render, params, light=light,
                                               output_size=output_size(source, video.out_height))
                result_tag = "SDR before H.264"

            facts = {"kind": kind, "path": path, "result_tag": result_tag, "width": source.width, "height": source.height,
                     "time_s": source.time_s, "duration_s": source.duration_s, "final": final,
                     "method": source.method if final else None,
                     "neural_fallback": chosen.method() == "neural" and source.method == "perceptual",
                     "white_nits": source.white_nits, "white_source": white_source,
                     "peak_nits": source.peak_nits,
                     # for an HDR surface, and only while this screen really is in HDR
                     "hdr": preview.hdr_pictures(source, after) if hdr and white_source == "display" else None}
            self.preview_ready.emit(request, preview.to_rgb8(source.before), preview.to_rgb8(after), facts)
        except Exception as exc:
            logging.exception("preview failed")
            self._source_key = None
            self.preview_failed.emit(request, str(exc))

    # ------------------------------------------------------- pictures of the queue
    THUMBNAIL = (192, 108)                         # the room a row has for it, twice over for sharp screens

    @Slot(str, str, object, str)
    def thumbnail(self, path: str, kind: str, options: dict, screen: str = "") -> None:
        """A small picture of a file waiting in the queue: the capture, or a frame from the
        start of the recording, as the source view shows it. Asked for many files at once,
        they are read one at a time, each after whatever the window asked meanwhile: a
        preview never waits behind a row of them. They are read between the frames of a
        background analysis, as previews are (a long recording shown first would keep the
        other rows bare for a minute), but never during a conversion."""
        if not any(waiting[0] == path for waiting in self._thumbnails):
            self._thumbnails.append((path, kind, options, screen))
        self._schedule_thumbnail()

    def _schedule_thumbnail(self) -> None:
        """Read the next picture once what already waits for this thread has been served."""
        if self._thumbnails and not self._thumbnail_due:
            self._thumbnail_due = True
            QTimer.singleShot(0, self._next_thumbnail)

    def forget_thumbnails(self) -> None:
        """(window's thread) The files left the queue: their pictures are not wanted any more."""
        self._thumbnails.clear()

    def _next_thumbnail(self) -> None:
        self._thumbnail_due = False
        if self._converting or self.closing:
            return                                 # the conversion starts them again when it ends
        try:
            path, kind, options, screen = self._thumbnails.popleft()
        except IndexError:
            return                                 # none left (the window may empty the list at any time)
        try:
            load_engine()
            from .. import preview
            from ..io.jxr_info import UnsupportedCapture
            from ..video.reader import UnsupportedVideo
        except Exception:
            return                                 # the preview has said why
        try:
            _, device = self.engines.get(Options.from_dict(options))
            white, _ = sdr_white_nits(screen)
            if kind == "video":
                source = preview.load_video_frame(path, 1.0, device, white_nits=white, box=self.THUMBNAIL)
            else:
                source = preview.load_image(path, device, white_nits=white, box=self.THUMBNAIL)
            self.thumbnail_ready.emit(path, preview.to_rgb8(source.before))
        except (UnsupportedVideo, UnsupportedCapture) as exc:
            self.thumbnail_refused.emit(path, str(exc))               # not a file this program converts
        except Exception:
            logging.debug("no picture for the queue", exc_info=True)   # the row keeps its plain frame
        self._schedule_thumbnail()

    # --------------------------------------------------------- long tasks, queued
    @Slot(int, str, object, str)
    def analyse(self, token: int, path: str, options: dict, screen: str = "") -> None:
        """``screen``: the screen the window is on, as for ``show``."""
        self._pending.append(("analyse", token, path, options, screen))
        self._drain()

    def _matching_analysis(self, file_key, chosen, settings):
        if (self._analysis_input == (file_key, chosen.use_gpu, chosen.use_fp16)
                and self.analysis is not None and self.analysis.matches(settings)):
            return self.analysis
        return None

    def _analyse_video(self, file_key, chosen, settings, device, progress, cancelled, engine):
        from ..video.job import analyse_video

        analysis = analyse_video(file_key[0], settings, device, progress, cancelled, engine)
        self.analysis = analysis
        self._analysis_input = (file_key, chosen.use_gpu, chosen.use_fp16)
        return analysis

    def _encoded_path(self, file_key, settings, chosen):
        entry = self._encoded.get(file_key[0])
        if entry is not None and entry[0] == (file_key, settings, chosen.use_gpu, chosen.use_fp16):
            output = entry[1]
            try:
                if _file_key(output[0]) == output:
                    return output[0]
            except FileNotFoundError:
                pass
            self._encoded.pop(file_key[0], None)
        return None

    def _remember_video(self, file_key, settings, chosen, output):
        self._encoded[file_key[0]] = ((file_key, settings, chosen.use_gpu, chosen.use_fp16), _file_key(output))

    def release_encoded_preview(self):
        if self._preview_dir is not None:
            self._preview_dir.cleanup()
            self._preview_dir = None

    @Slot(int, str, object, str)
    def encode_preview(self, token, path, options, screen):
        self._pending.append(("encode", token, path, options, screen))
        self._drain()

    @Slot(object, object, str)
    def convert(self, jobs: list, options: dict, screen: str = "") -> None:
        """``screen``: the screen the window was on when the conversion was asked for. The
        whole run is made for that screen's white, the files that were never shown too."""
        self._pending.append(("convert", jobs, options, screen))
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
                elif task[0] == "encode":
                    self._run_encode(*task[1:])
                else:
                    self._run_queue(*task[1:])
        finally:
            self._busy = False
            self._schedule_thumbnail()

    @staticmethod
    def _breathe() -> None:
        """Let preview requests (and newly queued tasks) through between two frames."""
        QCoreApplication.processEvents()

    def _run_analysis(self, token: int, path: str, options: dict, screen: str = "") -> None:
        if token != self.analysis_token:
            return
        try:
            load_engine()
        except Exception:
            return                                        # the preview has said why
        from ..video.job import Cancelled

        chosen = Options.from_dict(options)
        settings = chosen.video_settings(screen)
        try:
            file_key = _file_key(path)
        except OSError:
            return
        if self._matching_analysis(file_key, chosen, settings) is not None:
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
            self._analyse_video(file_key, chosen, settings, device, progress, cancelled, engine)
            self.analysis_ready.emit(token)
        except Cancelled:
            pass
        except Exception:
            logging.exception("background analysis failed")   # the conversion will report it properly

    def _run_encode(self, token, path, options, screen):
        from ..video.job import Cancelled, render_video

        def cancelled():
            return token != self.encode_token or self.closing

        def progress(stage, done, total):
            if done % 4 == 0 or done == total:
                self.encoded_progress.emit(token, 100 * done // max(1, total))
                self._breathe()

        if cancelled():
            return
        try:
            chosen = Options.from_dict(options)
            engine, device = self.engines.get(chosen)
            settings = chosen.video_settings(screen)
            file_key = _file_key(path)
            if not self._encoded_path(file_key, settings, chosen):
                self.release_encoded_preview()
                folder = tempfile.TemporaryDirectory(prefix="hdr-blaster-preview-")
                try:
                    if self._matching_analysis(file_key, chosen, settings) is None:
                        self._analyse_video(file_key, chosen, settings, device, progress, cancelled, engine)
                    output = os.path.join(folder.name, "preview.mp4")
                    render_video(self.analysis, output, settings, device, progress, cancelled, engine)
                    if cancelled() or file_key != _file_key(path):
                        raise Cancelled()
                    self._remember_video(file_key, settings, chosen, output)
                    self._preview_dir = folder
                finally:
                    if self._preview_dir is not folder:
                        folder.cleanup()
            self.encoded_ready.emit(token, "")
        except Cancelled:
            pass
        except Exception as exc:
            logging.exception("encoded preview failed")
            self.encoded_ready.emit(token, str(exc))

    def _run_queue(self, jobs: list, options: dict, screen: str = "") -> None:
        """Convert the jobs in order. Whatever happens, ``queue_finished`` is emitted: the
        window must never be left waiting."""
        self._converting = True
        try:
            load_engine()
            from ..video.job import Cancelled
            from ..video.reader import UnsupportedVideo

            chosen = Options.from_dict(options)
            white = options.get("white_nits")
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
                    analysis = None
                    if kind == "video":
                        settings = chosen.video_settings(screen, white)
                        file_key = _file_key(path)
                        analysis = self._matching_analysis(file_key, chosen, settings)
                        encoded = self._encoded_path(file_key, settings, chosen)
                        if encoded:
                            from ..video.encoded_preview import publish

                            output = output_base(path, kind, chosen) + ".mp4"
                            publish(encoded, output, progress, cancelled)
                            self._remember_video(file_key, settings, chosen, output)
                            if self._preview_dir and os.path.dirname(encoded) == self._preview_dir.name:
                                self.release_encoded_preview()
                            self.item_finished.emit(index, "done", "", [output])
                            continue
                        if analysis is None:
                            analysis = self._analyse_video(file_key, chosen, settings, device, progress, cancelled, engine)
                    prepared = None
                    if kind == "image" and self._capture_key == _capture_key(
                            _file_key(path), chosen, chosen.render_settings(screen=screen, white_nits=white), device):
                        prepared = self._capture
                    written = convert_item(Item(path=path, kind=kind), chosen, engine, device, progress,
                                           cancelled=cancelled, analysis=analysis, screen=screen, prepared=prepared, white_nits=white)
                    if kind == "video":
                        self._remember_video(file_key, settings, chosen, written[0])
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
            self._converting = False
            self.queue_finished.emit()

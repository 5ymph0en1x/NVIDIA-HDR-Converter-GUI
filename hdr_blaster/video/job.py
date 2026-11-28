"""Converting one HDR10 video to SDR, start to finish.

Two full-resolution passes over the file:

1. **analyse** - every frame is measured as a rendering of it would measure it; what
   was measured is kept. Shots are detected. The picture itself is only made when
   something is measured on it (sharpening, AI colour, the learned and Adaptive
   renderings); otherwise the scene's peak is all there is to measure.
2. **render**  - the measurements are stabilised over time, then every frame is
   rendered with them (nothing is measured), encoded and written, with the
   source's own timestamps and its audio copied next to it.

With the Auto-detect operator there is a short pass before those, to choose the
single operator the whole file is rendered with.

The stages of each pass run side by side (:mod:`.stages`): frames are decoded a few
ahead, by the graphics card when it can, and encoded a few behind. Every function
takes ``reference=True`` for the plain way instead (the processor decodes, one frame
at a time from end to end, every frame rendered in full when analysed): slower, the
same result, and what the tests hold the fast way against.
"""
import contextlib
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from bisect import bisect_left
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..engine.pipeline import RenderSettings, measure, render, resolve_method
from . import fused
from .colour import scrgb_to_engine, srgb_to_yuv420, yuv_to_scrgb
from ..engine.params import RenderParams
from .reader import (HardwareDecodingFailed, UnsupportedVideo, VideoInfo, check_supported, probe,
                     read_frames)
from .stages import Ahead, Behind
from .temporal import detect_cuts, frame_signature, stabilise
from .writer import QUALITY, VideoWriter

AUTO_SAMPLE_INTERVAL_S = 0.5      # Auto-detect looks at one frame out of every half second
SIZE_ATTEMPTS = 5                 # encodes we are willing to do to stay under the source's size
MIN_VIDEO_BIT_RATE = 20_000       # below this there is no picture left to speak of


class Cancelled(Exception):
    """The caller asked to stop. Nothing was written."""


class OutputTooLarge(Exception):
    """The result could not be made lighter than the source."""


@dataclass(frozen=True)
class VideoSettings:
    """What the user chose for a video. By default neither sharpening nor AI colour: a
    video is then analysed without rendering its frames, about twice as fast."""
    render: RenderSettings = field(default_factory=RenderSettings)
    quality: str = "high"
    out_height: Optional[int] = None      # e.g. 1080; None keeps the source size
    frame_rate: Optional[float] = None    # e.g. 60; None keeps every frame of the source
    motion_blur: bool = False             # at a lower frame rate: add up the light of the dropped frames
    pre_gamma: float = 1.0
    exposure: float = 1.0
    window_s: float = 1.0                 # time scale of the stabilisation; 0 = none (per-frame)


@dataclass
class VideoReport:
    dst: str
    frames: int
    method: str
    encoder: str
    seconds: float
    src_bytes: int
    dst_bytes: int
    cuts: List[int]
    width: int
    height: int

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def make_engine(device, use_fp16: bool = False):
    """The engine objects a conversion needs, built once."""
    from ..engine.processor import HDRColorProcessor
    from ..engine.tonemap import AdvancedToneMapper
    from ..io.jxr import OptimizedJXRLoader

    loader = OptimizedJXRLoader(device)
    return SimpleNamespace(loader=loader, tone_mapper=AdvancedToneMapper(device, loader.metadata),
                           processor=HDRColorProcessor(device, loader, use_fp16=use_fp16))


def default_output_path(src) -> str:
    """``<name>_SDR.mp4`` next to the source; ``_SDR-2``, ``_SDR-3``... if that exists."""
    base = os.path.splitext(os.fspath(src))[0] + "_SDR"
    candidate, n = base + ".mp4", 1
    while os.path.exists(candidate):
        n += 1
        candidate = f"{base}-{n}.mp4"
    return candidate


def output_size(info: VideoInfo, out_height: Optional[int]) -> tuple:
    """(width, height) of the result: the source's, or scaled down to ``out_height``. Always even."""
    if not out_height or out_height >= info.height:
        return info.width, info.height
    height = max(2, int(out_height) // 2 * 2)
    width = max(2, int(round(info.width * height / info.height / 2.0)) * 2)
    return width, height


def resize_for_output(picture, size):
    """Apply the recording's export dimensions before encoding or fitting it to the viewer."""
    width, height = size
    if picture.shape[-2:] == (height, width):
        return picture
    return F.interpolate(picture, size=(height, width), mode="bicubic",
                         align_corners=False, antialias=True).clamp(0.0, 1.0)


def frames_to_keep(times: Sequence[float], frame_rate: Optional[float], source_rate=None) -> List[bool]:
    """Which frames a conversion at ``frame_rate`` writes, one answer per frame.

    Frames are dropped, never invented and never moved. An ideal clock ticks ``frame_rate``
    times a second from the first frame to the last; for each tick the frame nearest to it
    is kept, with the timestamp it has in the source, and a tick no frame is nearest to
    gets none.
    A 120 frames per second recording gives every other frame at 60 and every fourth at
    30, also when its timestamps are not quite regular (those of a capture never are): the
    cadence of what is left stays even. A rate the source does not exceed keeps everything.
    """
    if not frame_rate or not times or (source_rate and frame_rate >= float(source_rate) * 0.99):
        return [True] * len(times)
    start, nearest = times[0], {}                    # tick -> (distance to it, frame)
    end = (times[-1] - start) * frame_rate + 1e-6    # the clock does not tick after the last frame
    for index, t in enumerate(times):
        position = (t - start) * frame_rate
        tick = math.floor(position + 0.5)
        distance = abs(position - tick)
        if tick <= end and (tick not in nearest or distance < nearest[tick][0] - 1e-9):
            nearest[tick] = (distance, index)        # (a tie goes to the earlier frame)
    chosen = {index for _, index in nearest.values()}
    return [index in chosen for index in range(len(times))]


def exposure_of(keep: Sequence[bool], cuts, index: int) -> Tuple[int, int]:
    """With motion blur, the frames whose light makes the picture that stands for frame
    ``index``, as (first, how many): the kept frame at or before it, and the frames dropped
    after that one, up to the next kept frame or the next shot. A picture never holds the
    light of two shots."""
    first = index
    while first > 0 and not keep[first]:
        first -= 1
    count = 1
    while first + count < len(keep) and not keep[first + count] and (first + count) not in cuts:
        count += 1
    return first, count


def bit_rate_budget(info: VideoInfo, quality: str) -> int:
    """Ceiling for the quality-driven first attempt: a share of the source's video bit rate."""
    return int(max(info.bit_rate * QUALITY[quality]["cap"], 200_000))


def _container_bytes(info: VideoInfo) -> int:
    return 4096 + 16 * max(info.frames, 1)


def corrected_bit_rate(info: VideoInfo, audio_bytes: int, used: int, produced: int) -> int:
    """After a mean-bit-rate attempt that still came out too heavy: scale the rate by how far
    off the video payload was (encoders overshoot on short clips), aiming 10 % under."""
    overhead = audio_bytes + _container_bytes(info)
    wanted, got = info.size_bytes * 0.90 - overhead, produced - overhead
    if wanted <= 0 or got <= 0:
        return 0
    return int(used * min(wanted / got, 0.8))


def size_target_bit_rate(info: VideoInfo, audio_bytes: int, retry: int) -> int:
    """Mean video bit rate that makes the whole file land under the source's size.

    Used after an attempt came out heavier than the source: what is left of 90 % of the
    source's size once the copied audio and the container are paid for, spread over the
    duration; 20 % less at each further retry. 0 when nothing is left.
    """
    available = info.size_bytes * 0.90 - audio_bytes - _container_bytes(info)
    if available <= 0 or info.duration_s <= 0:
        return 0
    return int(available * 8 / info.duration_s * (0.8 ** retry))


def _off_gpu(params: RenderParams) -> RenderParams:
    """Recorded parameters are kept in main memory between the passes, large values in
    half precision: the neural grid is 98 kB per frame, which on the GPU and in float64
    would not survive a long recording."""
    stats = {}
    for key, value in params.stats.items():
        if torch.is_tensor(value):
            value = value.detach().to("cpu", torch.float16 if value.numel() > 64 else value.dtype)
        stats[key] = value
    return RenderParams(method=params.method, stats=stats, frozen=params.frozen)


def _on_device(params: RenderParams, device) -> RenderParams:
    stats = {}
    for key, value in params.stats.items():
        if torch.is_tensor(value):
            value = value.to(device, torch.float32 if value.dtype == torch.float16 else value.dtype)
        stats[key] = value
    return RenderParams(method=params.method, stats=stats, frozen=params.frozen)


def _preflight(dst: str, info: VideoInfo, settings: VideoSettings, size: tuple) -> None:
    """Everything that can refuse the OUTPUT is tried before the long analysis pass."""
    folder = os.path.dirname(os.path.abspath(dst))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"the output folder does not exist: {folder}")
    writer = VideoWriter(dst, info, quality=settings.quality, size=size, frame_rate=settings.frame_rate)
    try:
        writer.start()
    except Exception as exc:
        raise UnsupportedVideo(
            f"{os.path.basename(info.path)} cannot be written as MP4 with its audio copied as it is "
            f"({exc}). Its audio format is probably one MP4 does not hold.") from exc
    finally:
        writer.abort()


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise Cancelled()


def _to_engine(frame, info: VideoInfo, settings: VideoSettings, chain=None) -> torch.Tensor:
    """The frame in the engine's working space. ``chain``: the fused stages, when they are used."""
    light = (yuv_to_scrgb if chain is None else chain.yuv_to_scrgb)(frame.y, frame.u, frame.v, info.fmt)
    return scrgb_to_engine(light, settings.pre_gamma, settings.exposure)


def _chain(device, reference: bool):
    """The fused stages of a frame (see :mod:`fused`): the same bits in far fewer operations.
    None for the reference way, which is what they are held against, and on the processor."""
    return None if reference else fused.chain_for(device)


def _frames(src, device, reference: bool, on_packet=None, wanted=None):
    """The frames of ``src``, for a ``with`` block: decoded ahead in a thread of their own,
    or (``reference``) by the processor, one at a time, in this thread."""
    if reference:
        return contextlib.closing(read_frames(src, device, on_packet=on_packet, hardware=False, wanted=wanted))
    return Ahead(lambda: read_frames(src, device, on_packet=on_packet, wanted=wanted))


def _decoding_again(run):
    """Run one pass over the file. If the graphics card stops decoding part of the way, run
    it once more from the start: the reader has taken note and uses the processor."""
    try:
        return run()
    except HardwareDecodingFailed as exc:
        logging.warning(f"{exc}; starting this pass again with the processor decoding")
        return run()


def _choose_method(src, info, settings, engine, device, progress, cancelled, reference=False) -> str:
    """One operator for the whole file: the one Auto-detect picks most often."""
    votes, next_sample = Counter(), 0.0
    with _frames(src, device, reference) as frames:
        for frame in frames:
            _check(cancelled)
            if frame.time_s >= next_sample:
                votes[resolve_method(engine.tone_mapper, _to_engine(frame, info, settings), None)] += 1
                next_sample = frame.time_s + AUTO_SAMPLE_INTERVAL_S
            if progress:
                progress("method", frame.index + 1, info.frames)
    if not votes:
        raise ValueError("the video has no frame")
    method = votes.most_common(1)[0][0]
    logging.info(f"Auto-detect over the whole video: {dict(votes)} -> {method}")
    return method


@dataclass
class VideoAnalysis:
    """Everything the first pass learned about a video: enough to render any of its frames
    exactly as the converted file will show it, and to convert without analysing again."""
    src: str
    info: VideoInfo
    method: str                    # the single operator the file is rendered with
    params: List[RenderParams]     # one per frame, stabilised and frozen, in main memory
    times: List[float]
    pts: List[int]
    cuts: List[int]
    key: tuple                     # the settings the analysis depends on

    def matches(self, settings: "VideoSettings") -> bool:
        return self.key == analysis_key(settings)

    def index_of(self, pts: int) -> int:
        """The frame with this timestamp (or the nearest one after it)."""
        return min(bisect_left(self.pts, pts), len(self.pts) - 1)


def analysis_key(settings: VideoSettings) -> tuple:
    """What an analysis depends on. Quality, output size and frame rate are not part of it:
    one analysis serves any encode."""
    r = settings.render
    return (r.method, r.color_strength, r.edge_strength, r.use_enhancement, r.white_nits,
            settings.pre_gamma, settings.exposure, settings.window_s)


def _check_destination(src: str, dst: str, info: VideoInfo, settings: VideoSettings) -> tuple:
    """Refuse what must be refused about the output before any long work. Returns its size."""
    if os.path.exists(dst):
        if os.path.samefile(src, dst):
            raise ValueError("the output cannot be the source file itself")
        raise FileExistsError(f"{dst} already exists; it will not be overwritten")
    size = output_size(info, settings.out_height)
    _preflight(dst, info, settings, None if size == (info.width, info.height) else size)
    return size


def analyse_video(src, settings: VideoSettings, device,
                  progress: Optional[Callable[[str, int, int], None]] = None,
                  cancelled: Optional[Callable[[], bool]] = None,
                  engine=None, info: Optional[VideoInfo] = None, reference: bool = False) -> VideoAnalysis:
    """First pass: choose the operator (Auto-detect), measure every frame, find the shots,
    stabilise. Nothing is written. Raises :class:`Cancelled` when ``cancelled()`` turns true."""
    src = os.fspath(src)
    if info is None:
        info = probe(src)
        check_supported(info)
    engine = engine or make_engine(device)
    with torch.inference_mode():
        method = settings.render.method
        if method is None:
            method = _decoding_again(lambda: _choose_method(src, info, settings, engine, device,
                                                            progress, cancelled, reference))
        else:
            method = engine.tone_mapper.available_method(method)
        render_settings = replace(settings.render, method=method)
        chain = _chain(device, reference)

        def measure_every_frame():
            recorded, times, pts, signatures = [], [], [], []
            with _frames(src, device, reference) as frames:
                for frame in frames:
                    _check(cancelled)
                    linear = _to_engine(frame, info, settings, chain)
                    if reference:
                        params = render(engine.tone_mapper, engine.loader, engine.processor, linear,
                                        render_settings, release_cache=False)[1]
                    else:
                        params = measure(engine.tone_mapper, engine.loader, engine.processor, linear,
                                         render_settings, release_cache=False)
                    recorded.append(_off_gpu(params))
                    times.append(frame.time_s)
                    pts.append(frame.pts)
                    signatures.append(frame_signature(frame.y, info.fmt.bit_depth))
                    if progress:
                        progress("analyse", frame.index + 1, info.frames)
            return recorded, times, pts, signatures

        recorded, times, pts, signatures = _decoding_again(measure_every_frame)
        if not recorded:
            raise ValueError("the video has no frame")
        cuts = detect_cuts(torch.stack(signatures))
        stable = stabilise(recorded, times, cuts, window_s=settings.window_s)
    return VideoAnalysis(src=src, info=info, method=method, params=stable, times=times, pts=pts,
                         cuts=cuts, key=analysis_key(settings))


def render_video(analysis: VideoAnalysis, dst, settings: VideoSettings, device,
                 progress: Optional[Callable[[str, int, int], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None,
                 engine=None, _checked_size: Optional[tuple] = None,
                 _started: Optional[float] = None, reference: bool = False) -> VideoReport:
    """Second pass: render every frame with the analysis' parameters, encode, write.

    The analysis must have been made with the same picture settings (``analysis.matches``);
    quality, output size and frame rate are free. At a lower frame rate the frames written
    are the very frames a conversion at the source's rate writes, fewer of them.

    With ``motion_blur``, at a lower frame rate, each of them is rendered from the light of
    all the frames it stands for, added up as a longer exposure adds it: the scene's linear
    light is averaged (:func:`exposure_of` says over which frames), then rendered once with
    the parameters of the frame that is kept."""
    src, dst = analysis.src, os.fspath(dst)
    started = _started if _started is not None else time.perf_counter()
    if not analysis.matches(settings):
        raise ValueError("this analysis was made with other picture settings: analyse again")
    info, stable = analysis.info, analysis.params
    size = _checked_size or _check_destination(src, dst, info, settings)
    engine = engine or make_engine(device)
    render_settings = replace(settings.render, method=analysis.method)
    keep = frames_to_keep(analysis.times, settings.frame_rate, info.average_rate)
    written = sum(keep)
    blur = bool(settings.motion_blur) and written < len(keep)
    cuts = set(analysis.cuts)
    chain = _chain(device, reference)
    # The file is the same whichever way the pictures reach the encoder, so the faster one
    # is taken, as measured (RTX 4090, 1440p, frames a second of the source): every frame
    # written, 173 through memory for 157 on the card; fewer frames written, 295 on the
    # card for 253 through memory at 60 a second, 357 for 290 at 30.
    on_card = not reference and written < len(keep)
    to_light = yuv_to_scrgb if chain is None else chain.yuv_to_scrgb
    to_planes = srgb_to_yuv420 if chain is None else chain.srgb_to_yuv420

    def kept(index: int) -> bool:
        return index >= len(keep) or keep[index]      # a frame past the analysis is looked at: it is an error

    def encode_every_frame(budget: int, average: bool) -> VideoWriter:
        """One encode of the whole file. Returns its writer, closed; leaves no file on failure."""
        writer = VideoWriter(dst, info, quality=settings.quality,
                             size=None if size == (info.width, info.height) else size,
                             max_bit_rate=budget, average=average, frame_rate=settings.frame_rate,
                             sessions=1 if reference else 2,      # the two hardware encoders at once, or in turns
                             device=device if on_card else None)  # pictures left on the card, or through memory
        encoder = None
        try:
            # Dropped frames are only counted, unless their light is wanted.
            with _frames(src, device, reference, on_packet=writer.copy_packet,
                         wanted=None if blur or written == len(keep) else kept) as frames:
                encoder = None if reference else Behind(writer.encode)

                def write(index: int, pts: int, light: torch.Tensor) -> None:
                    """Render the scene's light with the parameters of frame ``index``; encode."""
                    linear = scrgb_to_engine(light, settings.pre_gamma, settings.exposure)
                    srgb, _ = render(engine.tone_mapper, engine.loader, engine.processor, linear,
                                     render_settings, params=_on_device(stable[index], device),
                                     release_cache=False, fast=chain)
                    srgb = resize_for_output(srgb, size)
                    picture = writer.frame(pts, *to_planes(srgb))
                    if encoder is None:
                        writer.encode(picture)
                    else:
                        encoder.put(picture)

                def finish(exposure) -> None:
                    index, pts, light, frames_in_it = exposure
                    write(index, pts, light / frames_in_it if frames_in_it > 1 else light)

                exposure, gathering = None, False     # motion blur: [frame, timestamp, light so far, frames in it]
                for frame in frames:
                    _check(cancelled)
                    if frame.index >= len(stable):
                        raise RuntimeError("the video changed since it was analysed")
                    if keep[frame.index]:
                        if exposure is not None:
                            finish(exposure)           # the picture before this one has all its light
                        light = to_light(frame.y, frame.u, frame.v, info.fmt)
                        if blur:
                            exposure, gathering = [frame.index, frame.pts, light, 1], True
                        else:
                            exposure = None
                            write(frame.index, frame.pts, light)
                    elif gathering:
                        if frame.index in cuts:
                            gathering = False          # another shot begins: not this picture's light
                        else:
                            exposure[2] = exposure[2] + to_light(frame.y, frame.u, frame.v, info.fmt)
                            exposure[3] += 1
                    if progress:
                        progress("render", frame.index + 1, info.frames)
                if exposure is not None:
                    finish(exposure)
                if encoder is not None:
                    encoder, finishing = None, encoder
                    finishing.close()              # waits for the last frames; raises what they raised
            if writer.frames_written != written:
                raise RuntimeError("the video changed since it was analysed")
            writer.close()
        except BaseException:
            if encoder is not None:
                encoder.abort()                    # before the file it writes to is taken away
            writer.abort()
            raise
        return writer

    with torch.inference_mode():
        budget, average, writer = bit_rate_budget(info, settings.quality), False, None
        for attempt in range(SIZE_ATTEMPTS):
            writer = _decoding_again(lambda: encode_every_frame(budget, average))
            produced = os.path.getsize(dst)
            if produced <= info.size_bytes:
                break
            # Heavier than the source: never acceptable. Encode again, this time aiming at
            # a mean bit rate worked out from the source's size.
            os.remove(dst)
            logging.info(f"Result was {produced} bytes for a {info.size_bytes}-byte source; encoding again")
            if average:
                budget = corrected_bit_rate(info, writer.audio_bytes, budget, produced)
            else:
                budget, average = size_target_bit_rate(info, writer.audio_bytes, 0), True
            if budget < MIN_VIDEO_BIT_RATE:
                break
        if not os.path.exists(dst):
            raise OutputTooLarge(
                f"could not make the result lighter than the source ({info.size_bytes} bytes); "
                "try a lower quality or a smaller output size")

    return VideoReport(dst=dst, frames=written, method=analysis.method, encoder=writer.encoder_name,
                       seconds=time.perf_counter() - started, src_bytes=info.size_bytes,
                       dst_bytes=os.path.getsize(dst), cuts=analysis.cuts, width=size[0], height=size[1])


def convert_video(src, dst, settings: VideoSettings, device,
                  progress: Optional[Callable[[str, int, int], None]] = None,
                  cancelled: Optional[Callable[[], bool]] = None,
                  engine=None, reference: bool = False) -> VideoReport:
    """Convert ``src`` (HDR10) to ``dst`` (SDR MP4): analyse, then render. The source is only read.

    ``progress(stage, done, total)`` is called for every frame of every pass
    (``stage`` is "method", "analyse" or "render"; ``total`` is 0 when the
    container does not give a frame count). ``cancelled()`` is polled for every
    frame; when it returns true the conversion stops with :class:`Cancelled`
    and leaves no file.
    """
    src, dst = os.fspath(src), os.fspath(dst)
    started = time.perf_counter()
    info = probe(src)
    check_supported(info)
    size = _check_destination(src, dst, info, settings)
    engine = engine or make_engine(device)
    analysis = analyse_video(src, settings, device, progress, cancelled, engine, info=info, reference=reference)
    return render_video(analysis, dst, settings, device, progress, cancelled, engine,
                        _checked_size=size, _started=started, reference=reference)

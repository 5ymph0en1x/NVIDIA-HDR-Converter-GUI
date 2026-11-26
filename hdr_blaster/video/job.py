"""Converting one HDR10 video to SDR, start to finish.

Two full-resolution passes over the file:

1. **analyse** - every frame is rendered once in recording mode; what the engine
   measured on it is kept, the picture is thrown away. Shots are detected.
2. **render**  - the measurements are stabilised over time, then every frame is
   rendered again with them (nothing is measured), encoded and written, with the
   source's own timestamps and its audio copied next to it.

With the Auto-detect operator there is a short pass before those, to choose the
single operator the whole file is rendered with.
"""
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from bisect import bisect_left
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from ..engine.pipeline import RenderSettings, render, resolve_method
from .colour import srgb_to_yuv420, yuv_to_engine
from ..engine.params import RenderParams
from .reader import UnsupportedVideo, VideoInfo, check_supported, probe, read_frames
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
    """What the user chose for a video. Defaults match the photo defaults."""
    render: RenderSettings = field(default_factory=lambda: RenderSettings(edge_strength=25.0))
    quality: str = "high"
    out_height: Optional[int] = None      # e.g. 1080; None keeps the source size
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
    writer = VideoWriter(dst, info, quality=settings.quality, size=size)
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


def _to_engine(frame, info: VideoInfo, settings: VideoSettings) -> torch.Tensor:
    return yuv_to_engine(frame.y, frame.u, frame.v, info.fmt,
                         pre_gamma=settings.pre_gamma, exposure=settings.exposure)


def _choose_method(src, info, settings, engine, device, progress, cancelled) -> str:
    """One operator for the whole file: the one Auto-detect picks most often."""
    votes, next_sample = Counter(), 0.0
    for frame in read_frames(src, device):
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
    """What an analysis depends on. Quality and output size are not part of it: one
    analysis serves any encode."""
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
                  engine=None, info: Optional[VideoInfo] = None) -> VideoAnalysis:
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
            method = _choose_method(src, info, settings, engine, device, progress, cancelled)
        render_settings = replace(settings.render, method=method)

        recorded, times, pts, signatures = [], [], [], []
        for frame in read_frames(src, device):
            _check(cancelled)
            linear = _to_engine(frame, info, settings)
            _, params = render(engine.tone_mapper, engine.loader, engine.processor, linear, render_settings)
            recorded.append(_off_gpu(params))
            times.append(frame.time_s)
            pts.append(frame.pts)
            signatures.append(frame_signature(frame.y, info.fmt.bit_depth))
            if progress:
                progress("analyse", frame.index + 1, info.frames)
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
                 _started: Optional[float] = None) -> VideoReport:
    """Second pass: render every frame with the analysis' parameters, encode, write.

    The analysis must have been made with the same picture settings (``analysis.matches``);
    quality and output size are free."""
    src, dst = analysis.src, os.fspath(dst)
    started = _started if _started is not None else time.perf_counter()
    if not analysis.matches(settings):
        raise ValueError("this analysis was made with other picture settings: analyse again")
    info, stable = analysis.info, analysis.params
    size = _checked_size or _check_destination(src, dst, info, settings)
    engine = engine or make_engine(device)
    render_settings = replace(settings.render, method=analysis.method)

    with torch.inference_mode():
        budget, average, writer = bit_rate_budget(info, settings.quality), False, None
        for attempt in range(SIZE_ATTEMPTS):
            writer = VideoWriter(dst, info, quality=settings.quality,
                                 size=None if size == (info.width, info.height) else size,
                                 max_bit_rate=budget, average=average)
            try:
                for frame in read_frames(src, device, on_packet=writer.copy_packet):
                    _check(cancelled)
                    if frame.index >= len(stable):
                        raise RuntimeError("the video changed since it was analysed")
                    linear = _to_engine(frame, info, settings)
                    srgb, _ = render(engine.tone_mapper, engine.loader, engine.processor, linear,
                                     render_settings, params=_on_device(stable[frame.index], device))
                    if size != (info.width, info.height):
                        srgb = F.interpolate(srgb, size=(size[1], size[0]), mode="bicubic",
                                             align_corners=False, antialias=True).clamp(0.0, 1.0)
                    writer.write(frame.pts, *srgb_to_yuv420(srgb))
                    if progress:
                        progress("render", frame.index + 1, info.frames)
                if writer.frames_written != len(stable):
                    raise RuntimeError("the video changed since it was analysed")
                writer.close()
            except BaseException:
                writer.abort()
                raise
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

    return VideoReport(dst=dst, frames=len(stable), method=analysis.method, encoder=writer.encoder_name,
                       seconds=time.perf_counter() - started, src_bytes=info.size_bytes,
                       dst_bytes=os.path.getsize(dst), cuts=analysis.cuts, width=size[0], height=size[1])


def convert_video(src, dst, settings: VideoSettings, device,
                  progress: Optional[Callable[[str, int, int], None]] = None,
                  cancelled: Optional[Callable[[], bool]] = None,
                  engine=None) -> VideoReport:
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
    analysis = analyse_video(src, settings, device, progress, cancelled, engine, info=info)
    return render_video(analysis, dst, settings, device, progress, cancelled, engine,
                        _checked_size=size, _started=started)

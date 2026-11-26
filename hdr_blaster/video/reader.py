"""Reading HDR10 video with PyAV: what the file is, and its frames as GPU planes."""
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterator, Optional

import av
import numpy as np
import torch

from .colour import FrameFormat

# H.273 code points we care about
TRC_PQ = 16
PRIMARIES_BT2020 = 9
MATRIX_BT2020_NCL = 9
RANGE_FULL = 2

_TRC_NAMES = {1: "BT.709 (SDR)", 2: "unspecified", 4: "gamma 2.2 (SDR)", 6: "BT.601 (SDR)",
              13: "sRGB (SDR)", 14: "BT.2020 10-bit (SDR)", 16: "PQ (HDR10)", 18: "HLG"}
_PRIMARIES_NAMES = {1: "BT.709", 2: "unspecified", 5: "BT.601", 6: "BT.601", 9: "BT.2020", 12: "Display P3"}


class UnsupportedVideo(Exception):
    """The file cannot be converted. The message is written for the user."""


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    frames: int                   # an estimate when the container does not say; 0 if unknown
    duration_s: float
    time_base: Fraction
    average_rate: Optional[Fraction]
    bit_rate: int                 # of the video stream, bits per second (estimated if not stored)
    size_bytes: int
    codec: str
    pix_fmt: str
    fmt: FrameFormat
    transfer: int
    primaries: int
    matrix: int
    audio_streams: int
    rotation: int = 0             # degrees, from the display matrix (phone videos)
    sample_aspect_ratio: Optional[Fraction] = None     # non-square pixels when not 1


@dataclass
class Frame:
    index: int
    pts: int                      # in VideoInfo.time_base units
    time_s: float
    y: torch.Tensor               # (H, W) integer code values, on the requested device
    u: torch.Tensor               # (H/2, W/2)
    v: torch.Tensor


def probe(path) -> VideoInfo:
    path = os.fspath(path)
    try:
        container = av.open(path)
    except (av.FFmpegError, OSError) as exc:
        raise UnsupportedVideo(f"Could not open {os.path.basename(path)}: {exc}") from exc
    with container:
        if not container.streams.video:
            raise UnsupportedVideo(f"{os.path.basename(path)} has no video stream.")
        stream = container.streams.video[0]
        cc = stream.codec_context
        size = os.path.getsize(path)
        # Decode one frame: some files carry their colour tags only in the video
        # bitstream, which the decoder fills in then; the rotation is per frame too.
        rotation = 0
        try:
            first = next(container.decode(stream))
            rotation = int(getattr(first, "rotation", 0) or 0)
        except (StopIteration, av.FFmpegError):
            pass
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = (container.duration or 0) / 1_000_000
        bit_rate = stream.bit_rate or cc.bit_rate or 0
        if not bit_rate and duration > 0:
            bit_rate = int(size * 8 / duration)
        try:
            bit_depth = av.VideoFormat(cc.pix_fmt).components[0].bits
        except Exception:
            bit_depth = 8
        frames = stream.frames or 0
        if not frames and duration > 0 and stream.average_rate:
            # Matroska stores no frame count: estimate it, so that progress can be shown.
            frames = max(1, round(duration * float(stream.average_rate)))
        return VideoInfo(
            path=path, width=cc.width, height=cc.height, frames=frames,
            duration_s=duration, time_base=stream.time_base, average_rate=stream.average_rate,
            bit_rate=int(bit_rate), size_bytes=size, codec=cc.name, pix_fmt=cc.pix_fmt or "",
            fmt=FrameFormat(width=cc.width, height=cc.height, bit_depth=bit_depth,
                            full_range=cc.color_range == RANGE_FULL),
            transfer=cc.color_trc, primaries=cc.color_primaries, matrix=cc.colorspace,
            audio_streams=len(container.streams.audio),
            rotation=rotation,
            sample_aspect_ratio=stream.sample_aspect_ratio or cc.sample_aspect_ratio,
        )


def check_supported(info: VideoInfo) -> None:
    """Raise :class:`UnsupportedVideo` unless the file is an HDR10 recording we can convert."""
    name = os.path.basename(info.path)
    if info.transfer != TRC_PQ:
        kind = _TRC_NAMES.get(info.transfer, f"transfer {info.transfer}")
        raise UnsupportedVideo(
            f"{name} is not an HDR10 video (its transfer curve is {kind}). "
            "Only HDR10 recordings (PQ) can be converted.")
    if info.primaries != PRIMARIES_BT2020 or info.matrix != MATRIX_BT2020_NCL:
        kind = _PRIMARIES_NAMES.get(info.primaries, f"primaries {info.primaries}")
        raise UnsupportedVideo(
            f"{name} is PQ but not BT.2020 (colour primaries: {kind}, matrix {info.matrix}). "
            "Only standard HDR10 is supported.")
    if info.pix_fmt != "yuv420p10le":
        raise UnsupportedVideo(
            f"{name} uses pixel format {info.pix_fmt or 'unknown'}; only 10-bit 4:2:0 is supported.")
    if info.width % 2 or info.height % 2:
        raise UnsupportedVideo(f"{name} has odd dimensions ({info.width}x{info.height}).")


def planes_from_frame(frame, device):
    """(y, u, v) integer planes of a decoded 4:2:0 frame, on ``device``."""
    height, width = frame.height, frame.width
    # Signed 16-bit view: code values are at most 12 bits, and torch supports
    # every operation on int16 (not on uint16).
    flat = torch.from_numpy(frame.to_ndarray().view(np.int16)).reshape(-1)
    luma, chroma = height * width, (height // 2) * (width // 2)
    planes = flat.to(device, non_blocking=True)
    return (planes[:luma].view(height, width),
            planes[luma:luma + chroma].view(height // 2, width // 2),
            planes[luma + chroma:luma + 2 * chroma].view(height // 2, width // 2))


def read_frames(path, device, on_packet: Optional[Callable] = None) -> Iterator[Frame]:
    """Decode every video frame in display order, planes moved to ``device``.

    Packets of the audio streams are handed to ``on_packet`` as they are read, so
    that a writer can copy them next to the frames, in step with the video.
    """
    with av.open(os.fspath(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        wanted = [stream] + (list(container.streams.audio) if on_packet else [])
        index = 0
        for packet in container.demux(*wanted):
            if packet.stream.type != "video":
                if packet.dts is not None:
                    on_packet(packet)
                continue
            for frame in packet.decode():
                if frame.pts is None:
                    raise UnsupportedVideo(
                        f"{os.path.basename(os.fspath(path))} has frames without timestamps.")
                y, u, v = planes_from_frame(frame, device)
                yield Frame(index=index, pts=frame.pts, time_s=float(frame.pts * time_base), y=y, u=u, v=v)
                index += 1


def grab_frame(path, time_s: float, device) -> Frame:
    """The first frame shown at ``time_s`` or after it (the last frame when the time is past
    the end). Seeks to the previous key frame and decodes forward. ``index`` is -1: the
    position in the file is not known from a seek; match ``pts`` against an analysis."""
    with av.open(os.fspath(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        target = max(float(time_s), 0.0)
        container.seek(int(target / time_base), stream=stream, backward=True, any_frame=False)
        chosen = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            chosen = frame
            if float(frame.pts * time_base) >= target - 1e-6:
                break
        if chosen is None:
            raise UnsupportedVideo(f"{os.path.basename(os.fspath(path))} has no frame to show.")
        y, u, v = planes_from_frame(chosen, device)
        return Frame(index=-1, pts=chosen.pts, time_s=float(chosen.pts * time_base), y=y, u=u, v=v)

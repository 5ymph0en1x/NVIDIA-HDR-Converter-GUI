"""Reading HDR10 video with PyAV: what the file is, and its frames as GPU planes.

Frames are decoded by the graphics card when it can (NVDEC: several times faster than the
processor on AV1 and HEVC) and by the processor otherwise. Both give the same code values:
decoding these formats is exact by specification, and the tests compare the two.
"""
import logging
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


class HardwareDecodingFailed(Exception):
    """The graphics card stopped decoding part of the way through a file. The format is
    marked as one it does not decode: reading the file again uses the processor."""


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
    y: Optional[torch.Tensor]     # (H, W) integer code values, on the requested device
    u: Optional[torch.Tensor]     # (H/2, W/2)
    v: Optional[torch.Tensor]     # all three None for a frame that was decoded but not asked for


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


# FFmpeg's NVDEC decoders, by the name of the decoder a stream opens with by default.
_NVDEC = {"av1": "av1_cuvid", "libdav1d": "av1_cuvid", "libaom-av1": "av1_cuvid", "hevc": "hevc_cuvid"}
_SOFTWARE_THREADS = 32             # FFmpeg's own default leaves a third of the speed unused
_hardware_checked = {}             # (decoder, width, height, pixel format) -> does NVDEC decode it?


def _open_hardware(stream):
    """An opened NVDEC decoder for this stream, fed with its packets by hand; None when
    FFmpeg has none for the format."""
    source = stream.codec_context
    name = _NVDEC.get(source.name)
    if name is None or name not in av.codecs_available:
        return None
    decoder = av.CodecContext.create(name, "r")
    if source.extradata:
        decoder.extradata = source.extradata
    decoder.width, decoder.height, decoder.pix_fmt = source.width, source.height, source.pix_fmt
    decoder.open()
    return decoder


def _kind(stream) -> tuple:
    source = stream.codec_context
    return source.name, source.width, source.height, source.pix_fmt


def _first_frame(path, hardware: bool):
    """(timestamp, y, u, v) of the first frame shown, in main memory; None when the
    graphics card has no decoder for the format."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        decoder = _open_hardware(stream) if hardware else None
        if hardware and decoder is None:
            return None
        for packet in container.demux(stream):
            for frame in (decoder.decode(packet) if hardware else packet.decode()):
                return (frame.pts,) + tuple(_Planes(frame.height, frame.width, "cpu")(frame))
    raise ValueError("no frame")


def hardware_decoding(path) -> bool:
    """True when the graphics card decodes this file, and decodes it as the processor does.

    Tried for real, once per kind of stream: the first frame is decoded both ways and must
    come out with the same timestamp and the same code values. No NVIDIA card, an old
    driver, a format the card does not know or reads differently all end here, and the
    processor decodes instead."""
    path = os.fspath(path)
    with av.open(path) as container:
        key = _kind(container.streams.video[0])
    if key not in _hardware_checked:
        try:
            fast = _first_frame(path, hardware=True)
            usable = fast is not None
            if usable:
                plain = _first_frame(path, hardware=False)
                usable = fast[0] == plain[0] and fast[0] is not None and all(
                    a.shape == b.shape and torch.equal(a, b) for a, b in zip(fast[1:], plain[1:]))
                if not usable:
                    logging.warning(f"NVDEC does not decode {key[0]} {key[1]}x{key[2]} as the processor does; "
                                    "decoding on the processor")
        except Exception as exc:
            usable = False
            logging.info(f"NVDEC not available for {key[0]}, decoding on the processor: {exc}")
        _hardware_checked[key] = usable
    return _hardware_checked[key]


class _Planes:
    """Decoded frames -> integer code values on the device, through one reusable buffer
    (page-locked for a GPU, so that the upload is a single fast copy)."""

    def __init__(self, height: int, width: int, device):
        self.height, self.width, self.device = height, width, torch.device(device)
        self._buffer = torch.empty(height * width * 3 // 2, dtype=torch.int16)
        if self.device.type == "cuda":
            self._buffer = self._buffer.pin_memory()
        self._view = self._buffer.numpy()

    def __call__(self, frame):
        height, width = self.height, self.width
        half = (height // 2, width // 2)
        name = frame.format.name
        if name == "yuv420p10le":
            shapes = ((height, width), half, half)
        elif name == "p010le":                       # NVDEC: the two chroma planes interleaved
            shapes = ((height, width), (height // 2, width))
        else:
            raise UnsupportedVideo(f"the decoder returned frames in {name}; only 10-bit 4:2:0 is supported.")
        offset = 0
        for plane, (rows, columns) in zip(frame.planes, shapes):
            stored = np.frombuffer(plane, dtype=np.int16).reshape(rows, plane.line_size // 2)
            self._view[offset:offset + rows * columns].reshape(rows, columns)[:] = stored[:, :columns]
            offset += rows * columns
        # A GPU gets its own copy by the upload; in main memory the buffer is reused for the
        # next frame, so this one must leave it.
        data = self._buffer.to(self.device) if self.device.type == "cuda" else self._buffer.clone()
        luma, chroma = height * width, half[0] * half[1]
        if name == "p010le":
            # Ten bits at the top of sixteen. Through 32 bits, because the 16 are unsigned.
            data = ((data.to(torch.int32) & 0xFFFF) >> 6).to(torch.int16)
            both = data[luma:].view(half[0], half[1], 2)
            return data[:luma].view(height, width), both[..., 0], both[..., 1]
        return (data[:luma].view(height, width),
                data[luma:luma + chroma].view(half), data[luma + chroma:].view(half))


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


def read_frames(path, device, on_packet: Optional[Callable] = None,
                hardware: Optional[bool] = None,
                wanted: Optional[Callable[[int], bool]] = None) -> Iterator[Frame]:
    """Decode every video frame in display order, planes moved to ``device``.

    ``wanted(index)`` false: the frame is still decoded (the next ones depend on it) and
    still comes out, in its place, but without its planes, which are not moved anywhere.

    Packets of the audio streams are handed to ``on_packet`` as they are read, so
    that a writer can copy them next to the frames, in step with the video.

    ``hardware``: decode on the graphics card (NVDEC). By default it is used whenever the
    frames go to a GPU and the card decodes the file; ``False`` keeps the processor.
    """
    path = os.fspath(path)
    if hardware is None:
        hardware = torch.device(device).type == "cuda" and hardware_decoding(path)
    with av.open(path) as container:
        stream = container.streams.video[0]
        decoder = _open_hardware(stream) if hardware else None
        if hardware and decoder is None:
            raise UnsupportedVideo(f"{os.path.basename(path)} cannot be decoded by the graphics card.")
        if decoder is None:
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = min(os.cpu_count() or 1, _SOFTWARE_THREADS)
        logging.info(f"Decoding {os.path.basename(path)} with "
                     f"{'the graphics card (NVDEC)' if decoder is not None else 'the processor'}")
        time_base = stream.time_base
        streams = [stream] + (list(container.streams.audio) if on_packet else [])
        index, planes = 0, None
        for packet in container.demux(*streams):
            if packet.stream.type != "video":
                if packet.dts is not None:
                    on_packet(packet)
                continue
            try:
                frames = packet.decode() if decoder is None else decoder.decode(packet)
            except av.EOFError:                      # the decoder was already emptied
                break
            except av.FFmpegError as exc:
                if decoder is None:
                    raise
                _hardware_checked[_kind(stream)] = False
                raise HardwareDecodingFailed(
                    f"the graphics card stopped decoding {os.path.basename(path)} at frame {index}: {exc}") from exc
            for frame in frames:
                if frame.pts is None:
                    raise UnsupportedVideo(f"{os.path.basename(path)} has frames without timestamps.")
                if wanted is not None and not wanted(index):
                    y = u = v = None
                else:
                    if planes is None:
                        planes = _Planes(frame.height, frame.width, device)
                    y, u, v = planes(frame)
                yield Frame(index=index, pts=frame.pts, time_s=float(frame.pts * time_base), y=y, u=u, v=v)
                index += 1


def grab_frame(path, time_s: float, device) -> Frame:
    """The first frame shown at ``time_s`` or after it (the last frame when the time is past
    the end). Seeks to the previous key frame and decodes forward. ``index`` is -1: the
    position in the file is not known from a seek; match ``pts`` against an analysis."""
    return grab_frames(path, time_s, 1, device)[0]


def grab_frames(path, time_s: float, count: int, device) -> list:
    """Like :func:`grab_frame`, with the frames that follow: ``count`` of them in all, fewer
    when the file ends first."""
    with av.open(os.fspath(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        target = max(float(time_s), 0.0)
        container.seek(int(target / time_base), stream=stream, backward=True, any_frame=False)
        found, last = [], None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            last = frame
            if found or float(frame.pts * time_base) >= target - 1e-6:
                found.append(frame)
                if len(found) >= max(1, int(count)):
                    break
        if last is None:
            raise UnsupportedVideo(f"{os.path.basename(os.fspath(path))} has no frame to show.")
        out = []
        for frame in found or [last]:
            y, u, v = planes_from_frame(frame, device)
            out.append(Frame(index=-1, pts=frame.pts, time_s=float(frame.pts * time_base), y=y, u=u, v=v))
        return out

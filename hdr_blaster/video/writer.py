"""Writing the SDR result with PyAV: H.264 in MP4, BT.709, source timestamps, audio copied."""
import functools
import logging
import os
import threading
from fractions import Fraction
from typing import Optional

import av
import torch

from .reader import VideoInfo

# quality: constant-quality level, and the share of the source's video bit rate we may use.
# The cap is what keeps the result lighter than the source whatever the content.
QUALITY = {
    "high": {"level": 18, "cap": 0.90},
    "balanced": {"level": 22, "cap": 0.55},
    "compact": {"level": 27, "cap": 0.30},
}

# H.273 code points of the output
_BT709 = 1
_RANGE_LIMITED = 1


@functools.lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True when the NVIDIA hardware H.264 encoder can actually be opened on this machine."""
    try:
        cc = av.CodecContext.create("h264_nvenc", "w")
        cc.width, cc.height, cc.pix_fmt = 256, 144, "yuv420p"
        cc.time_base = Fraction(1, 30)
        cc.open()
        return True
    except Exception as exc:  # no NVIDIA GPU, old driver, encoder missing from the build
        logging.info(f"NVENC not available, using the software encoder: {exc}")
        return False


def pick_encoder() -> str:
    return "h264_nvenc" if nvenc_available() else "libx264"


def encoder_options(encoder: str, quality: str, max_bit_rate: int, average: bool = False) -> dict:
    """Encoder settings. Normally quality-driven with a ceiling (``max_bit_rate``). With
    ``average`` the encoder aims at ``max_bit_rate`` as the mean rate of the whole file
    instead: less efficient, but the file size becomes predictable."""
    if average:
        rate = {"maxrate": str(int(max_bit_rate * 1.5)), "bufsize": str(max_bit_rate)}
        if encoder == "h264_nvenc":
            return {"preset": "p6", "tune": "hq", "rc": "vbr", "profile": "high",
                    "spatial-aq": "1", "temporal-aq": "1", **rate}
        return {"preset": "medium", "profile": "high", **rate}
    level = QUALITY[quality]["level"]
    rate = {"maxrate": str(max_bit_rate), "bufsize": str(2 * max_bit_rate)}
    if encoder == "h264_nvenc":
        return {"preset": "p6", "tune": "hq", "rc": "vbr", "cq": str(level), "profile": "high",
                "spatial-aq": "1", "temporal-aq": "1", **rate}
    return {"preset": "medium", "crf": str(level), "profile": "high", **rate}


class VideoWriter:
    """Writes to ``<dst>.partial`` and renames on :meth:`close`; :meth:`abort` leaves nothing."""

    def __init__(self, dst, info: VideoInfo, quality: str = "high",
                 size: Optional[tuple] = None, encoder: Optional[str] = None,
                 max_bit_rate: Optional[int] = None, average: bool = False,
                 frame_rate: Optional[float] = None):
        if quality not in QUALITY:
            raise ValueError(f"unknown quality '{quality}' (expected one of {sorted(QUALITY)})")
        self.dst = os.fspath(dst)
        self.tmp = self.dst + ".partial"
        self.info = info
        self.width, self.height = size or (info.width, info.height)
        if self.width % 2 or self.height % 2:
            raise ValueError(f"output dimensions must be even, got {self.width}x{self.height}")
        self.encoder_name = encoder or pick_encoder()
        self.max_bit_rate = int(max_bit_rate or max(info.bit_rate * QUALITY[quality]["cap"], 200_000))
        self.average = average
        self.frames_written = 0
        self.audio_bytes = 0
        self._closed = False

        self._source = av.open(info.path)          # only for the audio stream templates
        self._out = av.open(self.tmp, "w", format="mp4", options={"movflags": "faststart"})
        try:
            rate = Fraction(info.average_rate or Fraction(60, 1))
            if frame_rate and frame_rate < rate:      # fewer frames are written than the source has
                rate = Fraction(frame_rate).limit_denominator(1001)
            self._video = self._out.add_stream(
                self.encoder_name, rate=rate,
                options=encoder_options(self.encoder_name, quality, self.max_bit_rate, average))
            self._video.width, self._video.height, self._video.pix_fmt = self.width, self.height, "yuv420p"
            cc = self._video.codec_context
            # Quality-driven by default (the ceiling is maxrate); a mean rate when asked.
            cc.bit_rate = self.max_bit_rate if average else 0
            cc.time_base = info.time_base
            self._video.time_base = info.time_base
            cc.color_range = _RANGE_LIMITED
            cc.colorspace = cc.color_trc = cc.color_primaries = _BT709
            # How the picture is to be shown travels with it: a phone video recorded
            # upright must not come out sideways, nor anamorphic pixels squashed.
            if info.sample_aspect_ratio and info.sample_aspect_ratio != 1 and size is None:
                self._video.sample_aspect_ratio = info.sample_aspect_ratio
            if info.rotation:
                self._video.set_display_rotation(info.rotation)
            self._audio = {s.index: self._out.add_stream_from_template(s)
                           for s in self._source.streams.audio}
        except Exception:
            self.abort()
            raise
        # The file is written from two threads when the stages run side by side: audio
        # packets as the reader meets them, video packets as the encoder produces them.
        self._mux = threading.Lock()

    def frame(self, pts: int, y: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        """One picture ready for the encoder: 8-bit planes (any device) copied straight into
        a frame FFmpeg owns, timestamp in the source's time base."""
        if tuple(y.shape) != (self.height, self.width):
            raise ValueError(f"frame is {y.shape[1]}x{y.shape[0]}, writer expects {self.width}x{self.height}")
        picture = av.VideoFrame(self.width, self.height, "yuv420p")
        for plane, tensor in zip(picture.planes, (y, u, v)):
            rows, columns = tensor.shape
            stored = torch.frombuffer(plane, dtype=torch.uint8).view(rows, plane.line_size)
            (stored if plane.line_size == columns else stored[:, :columns]).copy_(tensor)
        picture.pts = pts
        picture.time_base = self.info.time_base
        return picture

    def encode(self, picture) -> None:
        """Encode a picture made by :meth:`frame` and write what the encoder gives back."""
        for packet in self._video.encode(picture):
            with self._mux:
                self._out.mux(packet)
        self.frames_written += 1

    def write(self, pts: int, y: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> None:
        """One frame: 8-bit planes (any device), timestamp in the source's time base."""
        self.encode(self.frame(pts, y, u, v))

    def copy_packet(self, packet) -> None:
        """Copy one audio packet read from the source, untouched."""
        target = self._audio.get(packet.stream.index)
        if target is None:
            return
        self.audio_bytes += packet.size
        packet.stream = target
        with self._mux:
            self._out.mux(packet)

    def close(self) -> str:
        """Finish the file and move it to its final name. Returns that name."""
        if self._closed:
            return self.dst
        for packet in self._video.encode(None):
            with self._mux:
                self._out.mux(packet)
        self._out.close()
        self._source.close()
        os.replace(self.tmp, self.dst)      # if this fails, abort() still removes the partial file
        self._closed = True
        return self.dst

    def start(self) -> None:
        """Open the encoder and write the file header now, instead of at the first frame.
        Anything the container refuses (an audio codec MP4 cannot hold...) fails here."""
        self._out.start_encoding()

    def abort(self) -> None:
        """Stop and delete the partial file. The destination is never created."""
        if self._closed:
            return
        self._closed = True
        for handle in (getattr(self, "_out", None), getattr(self, "_source", None)):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        try:
            os.remove(self.tmp)
        except FileNotFoundError:
            pass

"""Writing the SDR result with PyAV: H.264 in MP4, BT.709, source timestamps, audio copied.

With the hardware encoder (NVENC) the video is encoded in segments of
:data:`SEGMENT_FRAMES` pictures by two encoders that take turns: the first encodes the
segments 0, 2, 4..., the second 1, 3, 5... A segment is what one encoder already made of
250 pictures: a group that begins with a key frame and refers to nothing outside itself.
So the file has its key frames where a single encoder put them, and since a segment owes
nothing to its neighbours, the two encoders can work at the same time: a card with two
encoder chips then uses both.

The file does not depend on whether they do. Each encoder always meets the same pictures
in the same order, its segments following one another as one stream whose every 250th
picture is a key frame, so taking turns one after the other or both at once gives the
same bytes (``sessions``). What an encoder gives back is told apart by counting: a
picture in, a packet out, in segment order.

Two encoders kept for the whole file, and not a new one for each segment: opening or
closing one stops the whole program for 0.3 s (measured), which would cost more than the
second chip brings. Nor are they emptied between their segments: an encoder started
again after being emptied encodes its key frame much worse (47.8 dB for 54.1, measured).

A second encoder is not always to be had: an older driver grants few of them, and
other programs may hold them (a recording in progress). When it cannot be opened, at
the second segment, the first encoder goes on alone: one stream for the whole video,
with the same key frames, and a line in the log. A card with a single encoder chip is
another matter and needs nothing: it grants two encoders like any other and shares its
chip between them; the file is the same, only not made faster.

The software encoder (libx264) uses every core by itself: one encoder, the whole stream.

Pictures computed on the graphics card are handed to the hardware encoder there
(``device``): it reads them where they are, instead of each being brought back to main
memory and sent to the card again (1.3 ms a picture at 2560x1440, and what kept the
second encoder from being of any use). The encoder gives the very same packets either
way; that is checked on each machine, for each picture size, before it is relied on
(:func:`hand_over_on_card`), and the hand-over through memory remains the reference.
"""
import functools
import logging
import os
import queue
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

# Pictures per segment: the distance between two key frames of the encoders used here.
SEGMENT_FRAMES = 250
# A segment starts on a key frame the others cannot see past (an IDR picture): the first
# picture of each is asked to be one, and this makes the hardware encoder obey to the letter.
_KEY_FRAMES = {"forced-idr": "1"}
_QUEUE_BYTES = 400_000_000        # pictures that may wait for one encoder, in memory
_END = object()
_TICK = 0.05                      # seconds between two looks at "did something fail meanwhile?"


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
                    "spatial-aq": "1", "temporal-aq": "1", **_KEY_FRAMES, **rate}
        return {"preset": "medium", "profile": "high", **rate}
    level = QUALITY[quality]["level"]
    rate = {"maxrate": str(max_bit_rate), "bufsize": str(2 * max_bit_rate)}
    if encoder == "h264_nvenc":
        return {"preset": "p6", "tune": "hq", "rc": "vbr", "cq": str(level), "profile": "high",
                "spatial-aq": "1", "temporal-aq": "1", **_KEY_FRAMES, **rate}
    return {"preset": "medium", "crf": str(level), "profile": "high", **rate}


_FLUSH = object()


# ----------------------------------------------------- pictures left on the card
def _card():
    """The card as PyTorch uses it, for the encoder: the two must share one context of the
    card to hand memory to each other. (Its "primary" context cannot be asked for by name:
    PyTorch holds it with other settings than FFmpeg would give it.) Made in a thread
    that computes on the card."""
    from av.codec.hwaccel import HWAccel

    return HWAccel("cuda", options={"current_ctx": "1"})


def _block(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The three planes as the encoder reads them on the card: one block of memory, the
    luma rows, then right behind them the chroma rows, blue and red samples in turn (NV12).
    One block, because the encoder is given where the luma starts and looks for the chroma
    after it: two separate tensors gave pictures with someone else's memory as colour."""
    height, width = y.shape
    block = torch.empty(height * 3 // 2, width, dtype=torch.uint8, device=y.device)
    block[:height] = y
    chroma = block[height:].view(height // 2, width // 2, 2)
    chroma[..., 0] = u
    chroma[..., 1] = v
    return block


def _card_frame(block: torch.Tensor, width: int, height: int):
    torch.cuda.synchronize(block.device)           # written, before the encoder reads it
    return av.VideoFrame.from_dlpack((block[:height], block[height:]), format="nv12", width=width, height=height,
                                     current_ctx=True)


def _memory_frame(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, width: int, height: int):
    picture = av.VideoFrame(width, height, "yuv420p")
    for plane, tensor in zip(picture.planes, (y, u, v)):
        rows, columns = tensor.shape
        stored = torch.frombuffer(plane, dtype=torch.uint8).view(rows, plane.line_size)
        (stored if plane.line_size == columns else stored[:, :columns]).copy_(tensor)
    return picture


_TRIAL = {"preset": "p6", "tune": "hq", "rc": "vbr", "cq": "18", "profile": "high", "spatial-aq": "1",
          "temporal-aq": "1", "forced-idr": "1", "maxrate": "20000000", "bufsize": "40000000"}


def _packets_of(width: int, height: int, on_card: bool) -> bytes:
    """A few difficult pictures of this size through the hardware encoder, handed over on
    the card or through memory: everything it gives back, end to end."""
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(20261127)
    luma = torch.randint(0, 256, (height, width), dtype=torch.uint8, device=device, generator=g)
    colour = torch.randint(0, 256, (2, height // 2, width // 2), dtype=torch.uint8, device=device, generator=g)
    encoder = av.CodecContext.create("h264_nvenc", "w", hwaccel=_card()) if on_card \
        else av.CodecContext.create("h264_nvenc", "w")
    encoder.width, encoder.height, encoder.pix_fmt = width, height, "cuda" if on_card else "yuv420p"
    encoder.time_base, encoder.framerate = Fraction(1, 90000), Fraction(60, 1)
    encoder.options = dict(_TRIAL)
    encoder.open()
    out = []
    for n in range(6):
        y = torch.roll(luma, shifts=(3 * n, 7 * n), dims=(0, 1)) if n % 2 else (luma >> n)
        u, v = torch.roll(colour[0], n, 1), torch.roll(colour[1], n, 0)
        picture = _card_frame(_block(y, u, v), width, height) if on_card else _memory_frame(y, u, v, width, height)
        picture.pts, picture.time_base = n * 1500, Fraction(1, 90000)
        out += [bytes(packet) for packet in encoder.encode(picture)]
    out += [bytes(packet) for packet in encoder.encode(None)]
    return b"".join(out)


_card_checked = {}


def hand_over_on_card(width: int, height: int) -> bool:
    """True when the hardware encoder, given pictures of this size where the card computed
    them, gives the very packets it gives for the same pictures handed over through
    memory. Tried for real, once per size. Anything else (no such way with this driver, a
    card that reads its own memory otherwise) ends here, and pictures go through memory."""
    key = (int(width), int(height))
    if width % 32 or height % 8:
        # Not even tried: with rows that are not a round number of bytes the encoder refuses
        # the picture (1918, 854 wide) or, worse, misreads the card's memory and leaves the
        # card in error for the rest of the program (1916 wide: "misaligned address").
        return False
    if key not in _card_checked:
        try:
            usable = _packets_of(width, height, True) == _packets_of(width, height, False)
            reason = "the encoder does not give the same packets that way"
        except Exception as exc:
            usable, reason = False, f"{type(exc).__name__}: {str(exc)[:160]}"
        _card_checked[key] = usable
        if usable:
            logging.info(f"Pictures of {width}x{height} are handed to the encoder on the card: the same packets.")
        else:
            logging.info(f"Pictures of {width}x{height} go to the encoder through main memory ({reason}).")
    return _card_checked[key]


class _Segment:
    """Up to SEGMENT_FRAMES pictures in a row, and the packets their encoder gave."""

    def __init__(self, index: int):
        self.index = index
        self.frames = 0
        self.ended = False             # every picture of it has been given to its encoder
        self.received = 0              # packets its encoder has given back so far
        self.packets = []

    @property
    def done(self) -> bool:
        return self.ended and self.received == self.frames


class VideoWriter:
    """Writes to ``<dst>.partial`` and renames on :meth:`close`; :meth:`abort` leaves nothing.

    ``sessions``: 2 lets the two hardware encoders work at the same time, each in a thread
    of its own; 1 has them take turns in the caller's thread. The file is the same.

    ``device``: where the pictures given to :meth:`frame` are computed. On a graphics card,
    with the hardware encoder, they are handed over there; ``None`` (the reference way)
    brings them through main memory. The file is the same. With a card, the writer must
    be made, and :meth:`frame` called, by a thread that computes on it."""

    def __init__(self, dst, info: VideoInfo, quality: str = "high",
                 size: Optional[tuple] = None, encoder: Optional[str] = None,
                 max_bit_rate: Optional[int] = None, average: bool = False,
                 frame_rate: Optional[float] = None, sessions: int = 1, device=None):
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
        # Two encoders taking turns with the hardware encoder, one for everything otherwise.
        self._turns = 2 if self.encoder_name == "h264_nvenc" else 1
        self.sessions = min(max(1, int(sessions)), self._turns)
        self._device = torch.device(device) if device is not None else None
        self.on_card = bool(self.encoder_name == "h264_nvenc" and self._device is not None
                            and self._device.type == "cuda" and hand_over_on_card(self.width, self.height))
        self._accel = _card() if self.on_card else None
        self._made = 0                             # pictures made by frame(), when they stay on the card

        rate = Fraction(info.average_rate or Fraction(60, 1))
        if frame_rate and frame_rate < rate:      # fewer frames are written than the source has
            rate = Fraction(frame_rate).limit_denominator(1001)
        # A segment is written once it is complete, which its encoder shows only when it
        # starts on its next one: the video reaches the file up to three segments behind
        # the sound read with it. The muxer puts the two in order of
        # time as long as neither is this far ahead of the other; beyond, it would write
        # as things come, and the file would depend on the moment. 10 s is its own value.
        behind = max(10.0, 3.5 * SEGMENT_FRAMES / float(rate) + 2.0)
        self._source = av.open(info.path)          # only for the audio stream templates
        self._out = av.open(self.tmp, "w", format="mp4", options={
            "movflags": "faststart", "max_interleave_delta": str(int(behind * 1_000_000))})
        try:
            self._options = encoder_options(self.encoder_name, quality, self.max_bit_rate, average)
            if self.on_card:
                self._video = self._out.add_stream(self.encoder_name, rate=rate, options=self._options,
                                                   hwaccel=self._accel)
            else:
                self._video = self._out.add_stream(self.encoder_name, rate=rate, options=self._options)
            self._video.width, self._video.height = self.width, self.height
            self._video.pix_fmt = "cuda" if self.on_card else "yuv420p"
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
        # The file is written from several threads when the stages run side by side: audio
        # packets as the reader meets them, video packets as the encoders finish segments.
        self._mux = threading.Lock()
        self._order = threading.Lock()             # one thread at a time decides what is written next
        self._started = False
        self._segments = []                        # begun and not yet written, in order
        self._current = None
        self._encoders = {}                        # turn -> encoder, opened at its first segment
        self._alone = False                        # no second encoder could be opened: the first does everything
        self._filling = {0: [], 1: []}             # turn -> its segments still short of packets, in order
        self._times = []                           # when each picture given so far is shown, in order
        self._packets_written = 0
        self._back = None                          # see _decoding_time
        self._last_dts = None
        self._failure = None
        self._workers = []
        if self.on_card:
            # Opened now, by this thread: the encoder joins the card's context it finds here.
            try:
                self._out.start_encoding()
                self._started = True
            except Exception:
                self.abort()
                raise
        if self.sessions > 1:
            depth = max(8, min(96, _QUEUE_BYTES // (self.width * self.height * 3 // 2)))
            for number in range(self._turns):
                waiting = queue.Queue(maxsize=depth)
                thread = threading.Thread(target=self._run, args=(waiting,), daemon=True,
                                          name=f"hdr-blaster-encoder-{number + 1}")
                thread.start()
                self._workers.append((thread, waiting))

    def frame(self, pts: int, y: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        """One picture ready for the encoder, timestamp in the source's time base: 8-bit
        planes (any device) left on the card for the encoder to read there, or copied
        straight into a frame FFmpeg owns in main memory."""
        if tuple(y.shape) != (self.height, self.width):
            raise ValueError(f"frame is {y.shape[1]}x{y.shape[0]}, writer expects {self.width}x{self.height}")
        if self.on_card:
            self._made += 1
            if self._made == SEGMENT_FRAMES + 1 and self._turns > 1:
                # The second encoder's first picture. An encoder that reads the card is opened
                # by a thread that computes on it: this one, not the thread it will work in.
                self._open_second()
            y, u, v = (plane.to(self._device) for plane in (y, u, v))
            picture = _card_frame(_block(y, u, v), self.width, self.height)
        else:
            picture = _memory_frame(y, u, v, self.width, self.height)
        picture.pts = pts
        picture.time_base = self.info.time_base
        return picture

    # ------------------------------------------------------------ segments
    def encode(self, picture) -> None:
        """Encode a picture made by :meth:`frame`, after the ones given before it. Called
        by one thread; what an encoder raised on an earlier picture is raised here."""
        self._raise()
        if not self._started:
            with self._mux:
                self._out.start_encoding()        # the stream's own encoder is open: the others are made like it
            self._started = True
        segment = self._current
        if segment is None or (segment.frames == SEGMENT_FRAMES and self._turns > 1):
            if segment is not None:
                self._hand(segment, _END)
            segment = self._current = _Segment(0 if segment is None else segment.index + 1)
            with self._order:
                self._segments.append(segment)
            if self._turns > 1:
                picture.pict_type = av.video.frame.PictureType.I      # with "forced-idr": a key frame nothing sees past
                if segment.index == 1:
                    self._open_second()            # (done already for pictures left on the card)
        if self._turns > 1:
            self._times.append(picture.pts)
        segment.frames += 1
        self._hand(segment, picture)
        self.frames_written += 1

    def _open_second(self) -> None:
        """Open the second encoder, before its first picture. If the card or its driver does
        not grant one, the first encoder goes on alone: decided here, once, for every
        segment to come, so that each still goes to one encoder and no other."""
        if self._alone or 1 in self._encoders:
            return
        try:
            self._encoders[1] = self._new_encoder()
        except Exception as exc:
            self._alone = True
            logging.warning(f"A second encoder could not be opened ({type(exc).__name__}: {str(exc)[:160]}): "
                            "one encoder encodes the whole video.")

    def _hand(self, segment: _Segment, item, turn: Optional[int] = None) -> None:
        if turn is None:
            turn = 0 if self._alone else segment.index % self._turns
        if not self._workers:
            self._work(turn, segment, item)
            return
        waiting = self._workers[turn][1]
        while True:
            self._raise()
            try:
                waiting.put((turn, segment, item), timeout=_TICK)
                return
            except queue.Full:
                pass

    def _run(self, waiting: queue.Queue) -> None:
        """An encoder's thread: the segments that fall to it, one after the other."""
        while True:
            item = waiting.get()
            if item is _END:
                return
            if self._failure is not None:
                continue                           # keep emptying: nobody must block on this queue
            try:
                self._work(*item)
            except BaseException as exc:
                self._failure = exc

    def _work(self, turn: int, segment: _Segment, item) -> None:
        encoder = self._encoders.get(turn)
        if encoder is None:
            if item is _FLUSH or item is _END:
                if item is _END:
                    segment.ended = True
                return                             # this encoder never had a picture
            if turn:
                raise RuntimeError("a picture was handed to a second encoder that was never opened")
            encoder = self._encoders[0] = self._video.codec_context       # the stream's own
        filling = self._filling[turn]
        if item is _END:
            segment.ended = True
            packets = []
        elif item is _FLUSH:                       # the end of the video: what the encoder still holds
            packets = encoder.encode(None)
        else:
            if not filling or filling[-1] is not segment:
                filling.append(segment)
            packets = encoder.encode(item)
        # A picture in, a packet out, in the order of the segments: the oldest one still
        # short of packets takes them (it holds SEGMENT_FRAMES of them unless it is the last).
        whole = SEGMENT_FRAMES if self._turns > 1 else float("inf")     # one encoder: one segment, however long
        for packet in packets:
            while filling and filling[0].received == (filling[0].frames if filling[0].ended else whole):
                filling.pop(0)
            if not filling:
                raise RuntimeError("the encoder gave more packets than it was given pictures")
            filling[0].packets.append(packet)
            filling[0].received += 1
        self._write_finished()

    def _new_encoder(self):
        """The second encoder, set up as the stream's own."""
        own = self._video.codec_context
        cc = av.CodecContext.create(self.encoder_name, "w", hwaccel=self._accel) if self.on_card \
            else av.CodecContext.create(self.encoder_name, "w")
        cc.width, cc.height, cc.pix_fmt = self.width, self.height, "cuda" if self.on_card else "yuv420p"
        cc.framerate, cc.time_base = own.framerate, own.time_base
        cc.bit_rate = self.max_bit_rate if self.average else 0
        cc.options = dict(self._options)
        cc.color_range, cc.colorspace = own.color_range, own.colorspace
        cc.color_trc, cc.color_primaries = own.color_trc, own.color_primaries
        if own.sample_aspect_ratio:
            cc.sample_aspect_ratio = own.sample_aspect_ratio
        cc.flags = own.flags                       # among them: the stream's header kept apart from the pictures
        cc.open()
        if cc.extradata != own.extradata:
            raise RuntimeError("the second encoder describes its stream otherwise than the first: "
                               "their segments cannot be written as one video")
        return cc

    def _write_finished(self) -> None:
        """Write every finished segment that comes next in order."""
        with self._order:
            while self._segments and self._segments[0].done:
                segment = self._segments.pop(0)
                packets, segment.packets = segment.packets, []
                if not packets:
                    continue
                if self._turns > 1 and self._back is None:
                    # How many places back the encoder looks: its first packets are timed
                    # before the first picture, and the one timed AT the first picture tells.
                    first = self._times[0]
                    self._back = next((n for n, packet in enumerate(packets) if packet.dts >= first), len(packets))
                for number, packet in enumerate(packets):
                    if self._turns > 1:
                        packet.dts = self._decoding_time(packet)
                    # What the container would refuse, said in words rather than "invalid argument".
                    if self._last_dts is not None and packet.dts <= self._last_dts or packet.pts < packet.dts:
                        raise RuntimeError(
                            f"segment {segment.index}, packet {number}: shown at {packet.pts}, decoded at {packet.dts} "
                            f"(key frame: {packet.is_keyframe}), after a packet decoded at {self._last_dts}")
                    self._last_dts = packet.dts
                    packet.stream = self._video
                    with self._mux:
                        self._out.mux(packet)

    def _decoding_time(self, packet) -> int:
        """When the next packet of the file is to be decoded.

        An encoder gives its n-th packet the time of the picture it was given a few places
        earlier (3 with these settings: pictures are not stored in the order they are
        shown), and times of its own, before the first picture, to its first packets. That
        is right for one encoder fed the whole video. Here each of the two is fed every
        other segment, and its "few places earlier" reaches back into its previous segment,
        two segments away in the file. So the same rule is applied here to the video as a
        whole: the times a single encoder would have given."""
        place = self._packets_written
        self._packets_written += 1
        if place < self._back:
            return packet.dts                      # the very first packets: as their encoder timed them
        return self._times[place - self._back]

    def _raise(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _stop_workers(self) -> None:
        workers, self._workers = self._workers, []
        for _, waiting in workers:
            waiting.put(_END)                      # the thread always empties its queue: this returns
        for thread, _ in workers:
            thread.join()

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
        if self._current is not None and not self._current.ended:
            self._hand(self._current, _END)
        for turn in range(self._turns):
            self._hand(_Segment(turn), _FLUSH, turn)   # addressed to that encoder; never written itself
        self._stop_workers()
        self._raise()
        self._write_finished()
        if self._segments:
            state = ", ".join(f"segment {s.index}: {s.received} packets for {s.frames} pictures"
                              for s in self._segments[:4])
            raise RuntimeError(f"a segment of the video was not encoded to its end ({state})")
        self._out.close()
        self._source.close()
        os.replace(self.tmp, self.dst)      # if this fails, abort() still removes the partial file
        self._closed = True
        return self.dst

    def start(self) -> None:
        """Open the encoder and write the file header now, instead of at the first frame.
        Anything the container refuses (an audio codec MP4 cannot hold...) fails here."""
        with self._mux:
            self._out.start_encoding()
        self._started = True

    def abort(self) -> None:
        """Stop and delete the partial file. The destination is never created."""
        if self._closed:
            return
        self._closed = True
        if getattr(self, "_workers", None):
            self._failure = self._failure or RuntimeError("the conversion was stopped")
            self._stop_workers()
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

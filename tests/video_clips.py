"""Small generated clips for the video tests, and a way to describe a file as a player sees it."""
from fractions import Fraction

import av
import numpy as np

from video_fixtures import nits709_to_hdr10

TIME_BASE = Fraction(1, 90000)
STEPS = (1500, 3000, 1200, 2700)          # 13 to 33 ms between frames, never regular


def scene_nits(width, height, t, base=60.0):
    """A moving colour gradient with a bright spot, linear BT.709 nits, (H, W, 3)."""
    yy, xx = np.mgrid[0:height, 0:width]
    u = xx / max(width - 1, 1)
    v = yy / max(height - 1, 1)
    rgb = np.stack([base * (0.2 + u), base * (0.2 + v), base * (0.6 - 0.4 * u)], axis=-1)
    cx = width * (0.2 + 0.6 * t)
    spot = np.exp(-(((xx - cx) / (0.08 * width)) ** 2 + ((yy - height / 2) / (0.12 * height)) ** 2))
    return rgb + 900.0 * spot[..., None]


def make_clip(path, frames=20, size=(256, 144), hdr=True, audio_streams=1, levels=None,
              full_range=False, scene=scene_nits, texture=0.25, rotation=0,
              sample_aspect_ratio=None):
    """Write a short test clip and return the list of frame timestamps (in 1/90000 s).

    HDR: HEVC Main 10, PQ, BT.2020, 4:2:0, irregular timestamps (like an NVIDIA recording).
    ``texture`` adds fine detail (a fixed pattern plus a little grain per frame) so that the
    clip costs bits the way real footage does; 0 gives a perfectly smooth picture.
    SDR (``hdr=False``): H.264 8-bit BT.709. ``levels`` scales the brightness of each frame.
    """
    width, height = size
    out = av.open(str(path), "w")
    if hdr:
        video = out.add_stream("libx265", rate=Fraction(60, 1),
                               options={"x265-params": "log-level=error", "preset": "ultrafast", "crf": "12"})
        video.pix_fmt = "yuv420p10le"
        tags = dict(color_range=2 if full_range else 1, colorspace=9, color_trc=16, color_primaries=9)
    else:
        video = out.add_stream("libx264", rate=Fraction(60, 1), options={"preset": "ultrafast"})
        video.pix_fmt = "yuv420p"
        tags = dict(color_range=1, colorspace=1, color_trc=1, color_primaries=1)
    video.width, video.height = width, height
    if rotation:
        video.set_display_rotation(rotation)
    if sample_aspect_ratio:
        video.sample_aspect_ratio = Fraction(*sample_aspect_ratio)
    video.time_base = TIME_BASE
    video.codec_context.time_base = TIME_BASE
    for name, value in tags.items():
        setattr(video.codec_context, name, value)
    audio = []
    for _ in range(audio_streams):
        stream = out.add_stream("aac", rate=48000)
        stream.layout = "stereo"
        audio.append(stream)

    rng = np.random.default_rng(0)
    pattern = rng.random((height, width, 1))
    pts_list, pts = [], 0
    for i in range(frames):
        level = 1.0 if levels is None else float(levels[i])
        nits = scene(width, height, i / max(frames - 1, 1)) * level
        if texture:
            grain = rng.random((height, width, 1))
            nits = nits * (1.0 + texture * (pattern - 0.5) + 0.2 * texture * (grain - 0.5))
        if hdr:
            y, u, v = nits709_to_hdr10(nits, full_range=full_range)
            packed = np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)])
            frame = av.VideoFrame.from_ndarray(packed.reshape(height * 3 // 2, width), format="yuv420p10le")
        else:
            rgb = np.clip(nits / 200.0, 0.0, 1.0) ** (1 / 2.2)
            frame = av.VideoFrame.from_ndarray(np.rint(rgb * 255).astype(np.uint8), format="rgb24")
            frame = frame.reformat(format="yuv420p")
        frame.pts, frame.time_base = pts, TIME_BASE
        pts_list.append(pts)
        pts += STEPS[i % len(STEPS)]
        for packet in video.encode(frame):
            out.mux(packet)
    for packet in video.encode(None):
        out.mux(packet)

    samples = int(48000 * pts / 90000)
    t = np.arange(samples) / 48000.0
    for k, stream in enumerate(audio):
        tone = (0.2 * np.sin(2 * np.pi * (440.0 + 220.0 * k) * t)).astype(np.float32)
        for start in range(0, samples, 1024):
            chunk = tone[start:start + 1024]
            frame = av.AudioFrame.from_ndarray(np.stack([chunk, chunk]), format="fltp", layout="stereo")
            frame.sample_rate, frame.pts = 48000, start
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    out.close()
    return pts_list


def describe(path):
    """What a player would learn about a file: codec, tags, timestamps, audio packets."""
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        cc = v.codec_context
        info = dict(codec=cc.name, pix_fmt=cc.pix_fmt, width=cc.width, height=cc.height,
                    range=cc.color_range, matrix=cc.colorspace, trc=cc.color_trc,
                    primaries=cc.color_primaries, audio_streams=len(c.streams.audio))
        order = {s.index: k for k, s in enumerate(c.streams.audio)}
        info["audio_packets"] = [0] * len(order)
        info["audio_bytes"] = [0] * len(order)
        if order:
            for packet in c.demux(*c.streams.audio):
                if packet.dts is not None:
                    info["audio_packets"][order[packet.stream.index]] += 1
                    info["audio_bytes"][order[packet.stream.index]] += packet.size
            c.seek(0)
        frames = list(c.decode(video=0))
        info["rotation"] = int(getattr(frames[0], "rotation", 0) or 0) % 360
        sar = v.sample_aspect_ratio or Fraction(1, 1)
        info["sample_aspect_ratio"] = (sar.numerator, sar.denominator)
        info["pts_s"] = [float(f.pts * v.time_base) for f in frames]
        info["mean_luma"] = [float(f.to_ndarray()[:cc.height].mean()) for f in frames]
    return info

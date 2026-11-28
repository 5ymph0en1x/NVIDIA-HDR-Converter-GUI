"""Inspect the finished MP4 and publish the same encode when the user saves it."""
import os
import tempfile

import av
import torch

from .job import Cancelled


# swscale: bilinear chroma (2), read and written at full resolution (0x4000, 0x2000), exact
# rounding (0x40000). Its default conversion truncates instead: on a converted recording
# every pixel was shown 1 to 1.5 levels darker than the file holds, greys included.
_EXACT_RGB = 2 | 0x2000 | 0x4000 | 0x40000


def rgb_of(frame):
    """A decoded frame as (H, W, 3) uint8, by the matrix and the range the file signals:
    each sample rounded to the nearest level, the chroma interpolated between its sites."""
    return frame.reformat(format="rgb24", interpolation=_EXACT_RGB).to_ndarray()


def frame_at(path, time_s, device):
    """The decoded presentation frame covering this timestamp, including held frames."""
    with av.open(os.fspath(path)) as container:
        stream = container.streams.video[0]
        tick = max(0, round(time_s / float(stream.time_base)))
        container.seek(tick, stream=stream, backward=True, any_frame=False)
        selected = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if frame.pts > tick and selected is not None:
                break
            selected = frame
            if frame.pts >= tick:
                break
        if selected is None:
            raise ValueError("The encoded MP4 has no frame at this time.")
        rgb = rgb_of(selected)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device)


def publish(src, dst, progress, cancelled):
    """Copy across volumes, then atomically publish without replacing another export."""
    size = os.path.getsize(src)
    target = tempfile.NamedTemporaryFile(dir=os.path.dirname(dst), suffix=".partial", delete=False)
    temp = target.name
    try:
        with target, open(src, "rb") as source:
            while chunk := source.read(4 * 1024 * 1024):
                target.write(chunk)
                progress("render", target.tell(), size)
                if cancelled():
                    raise Cancelled()
        if os.name == "nt":
            os.rename(temp, dst)  # fails if the destination already exists
        else:
            os.link(temp, dst)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

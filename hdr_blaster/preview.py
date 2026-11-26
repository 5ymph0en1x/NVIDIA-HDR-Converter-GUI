"""What the interface shows before converting: the HDR source on one side, the SDR result
on the other, both at the size of their place in the window.

Nothing here knows about windows. A source is loaded once (a capture, or one frame of a
video) and kept at its real size. It is rendered at that size, as the conversion renders
it, as often as the settings change; the two pictures shown are then brought to the size
of the window in ONE resampling (see ``for_display``). Shown at the size of the file,
they are the file's pixels and the converted picture's pixels, untouched. For a video,
rendering with the parameters of a finished analysis gives exactly the frame the
converted file will contain.

The first version reduced the HDR picture before rendering it, by averaging groups of
pixels, and let the window stretch the result to its place: thin lines came out as
staircases, on both sides of the divider, and a light brighter than white swallowed its
neighbours (averaged with the dark around it, it was still above white).

The HDR source is shown truthfully, not as a careless SDR program would show it: its
light is decoded from what the file says (scRGB for a float JXR; PQ, BT.2020 and the
stated range for a video) into absolute nits, and placed so that the window's white
stands for the luminance the screen really gives it. Everything up to that white then
has, in the window, the luminance it has in the game; only what is brighter is clipped.
"""
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import numpy as np
import torch

from .engine.colour import adjust_scene
from .engine.params import RenderParams
from .engine.pipeline import RenderSettings, render
from .resample import fit_size, resize


@dataclass
class PreviewSource:
    kind: str                      # "image" or "video"
    path: str
    linear: torch.Tensor           # (1, 3, H, W), the engine's working space, at the file's size
    width: int                     # size of the file itself
    height: int
    size: Tuple[int, int] = (0, 0)  # (height, width) of the pictures shown; (0, 0): the file's
    method: Optional[str] = None   # the operator of the last render (what Auto chose)
    time_s: float = 0.0            # video: time of the frame
    pts: int = 0                   # video: its timestamp, to find it in an analysis
    duration_s: float = 0.0
    white_nits: float = 203.0      # what the window's white stands for
    peak_nits: float = 0.0         # brightest luminance in the picture
    # The user's Gamma and Exposure: they shape the RESULT only (see render_preview). The
    # source view and `linear` always hold the picture as recorded.
    pre_gamma: float = 1.0
    exposure: float = 1.0
    _view: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if tuple(self.size) == (0, 0):
            self.size = tuple(self.linear.shape[-2:])

    @property
    def before(self) -> torch.Tensor:
        """(1, 3, h, w) sRGB: the HDR source as the window shows it (see reference_view),
        at the size shown. Made once for a size and a white."""
        key = (tuple(self.size), float(self.white_nits))
        if self._view.get("key") != key:
            self._view["key"] = key
            self._view["picture"] = for_display(reference_view(self.linear, self.white_nits), self.size)
        return self._view["picture"]


def shown_size(width: int, height: int, max_side: int = 1600,
               box: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """(height, width) of the pictures shown for a file of this size. With the ``box`` of
    the window (width, height in screen pixels): the largest picture that fits it, enlarged
    if the window is larger than the file. Without: the file's size, reduced so that its
    longer side is at most ``max_side``."""
    if box is not None:
        w, h = fit_size(int(box[0]), int(box[1]), width, height)
        return h, w
    scale = min(1.0, max_side / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def fit(source: "PreviewSource", max_side: int = 1600, box: Optional[Tuple[int, int]] = None) -> "PreviewSource":
    """Give a loaded source another display size (the window was resized): nothing is read again."""
    source.size = shown_size(source.width, source.height, max_side, box)
    return source


def for_display(srgb: torch.Tensor, size) -> torch.Tensor:
    """A finished sRGB picture at the size it is shown: one Spline36 resampling, none at
    all when it is shown at its own size.

    Always the FINISHED picture, never the HDR one: what is brighter than white has been
    brought under it first, as on the screen at full size. Reduced, the picture is
    resampled as light adds up (linear light): a thin bright line keeps its brightness
    instead of fading. Enlarged, it is resampled on its encoded values, where a kernel's
    overshoot is least visible."""
    size = (int(size[0]), int(size[1]))
    if tuple(srgb.shape[-2:]) == size:
        return srgb
    srgb = srgb.float()
    if size[0] <= srgb.shape[-2] and size[1] <= srgb.shape[-1]:
        return _srgb(resize(_srgb_decode(srgb), *size))
    return resize(srgb.clamp(0.0, 1.0), *size)


SCRGB_WHITE_NITS = 80.0            # what 1.0 stands for on an HDR (scRGB) surface


def for_display_hdr(scrgb: torch.Tensor, size) -> torch.Tensor:
    """An HDR picture (linear, 1.0 = 80 nits) at the size it is shown, for a surface that
    presents HDR: one Spline36 resampling, none when it is shown at its own size.

    Nothing is cut on such a surface, so a reduced picture is light simply averaged
    (linear light). Enlarged, it is resampled on its PQ codes, the scale on which a
    kernel's overshoot is least visible; what lies outside BT.709 is then brought to its edge."""
    from .engine.renderings import pq_decode, pq_encode

    size = (int(size[0]), int(size[1]))
    if tuple(scrgb.shape[-2:]) == size:
        return scrgb
    scrgb = scrgb.float()
    if size[0] <= scrgb.shape[-2] and size[1] <= scrgb.shape[-1]:
        return resize(scrgb, *size)
    return pq_decode(resize(pq_encode(scrgb.clamp(min=0.0) * SCRGB_WHITE_NITS), *size)) / SCRGB_WHITE_NITS


def to_rgba16f(image: torch.Tensor) -> np.ndarray:
    """(1, 3, h, w) -> (h, w, 4) half floats, opaque, contiguous: a texture for an HDR surface."""
    height, width = image.shape[-2:]
    out = torch.ones((height, width, 4), dtype=torch.float16, device=image.device)
    out[..., :3] = image[0].detach().permute(1, 2, 0).to(torch.float16)
    return np.ascontiguousarray(out.cpu().numpy())


def hdr_pictures(source: "PreviewSource", after: torch.Tensor):
    """The two pictures for a surface that presents HDR, as (before, after) half-float
    textures in scRGB (1.0 = 80 nits):

    * before: the capture's own values, as recorded, at the size shown. Nothing is cut:
      a sun at 1000 nits is 12.5. This is what the game showed, and what Windows' own
      viewer shows.
    * after: the SDR result (``after``, sRGB, from :func:`render_preview`) at the luminance
      an SDR window gives it on this screen: its white is ``source.white_nits``.
    """
    from .engine.constants import ENGINE_UNIT_NITS

    before = for_display_hdr(source.linear.float() * (ENGINE_UNIT_NITS / SCRGB_WHITE_NITS), source.size)
    result = _srgb_decode(after.float()) * (float(source.white_nits) / SCRGB_WHITE_NITS)
    return to_rgba16f(before), to_rgba16f(result)


def _srgb(linear01: torch.Tensor) -> torch.Tensor:
    x = linear01.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def _srgb_decode(encoded: torch.Tensor) -> torch.Tensor:
    x = encoded.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def reference_view(linear: torch.Tensor, white_nits: float) -> torch.Tensor:
    """The HDR picture as an SDR window can truthfully show it: absolute luminance, with
    the window's white standing for ``white_nits``. Tones up to that level come out at
    their real luminance; brighter ones are clipped. ``linear`` is in the engine's
    working space."""
    from .engine.constants import ENGINE_UNIT_NITS

    return _srgb(linear.float() * (ENGINE_UNIT_NITS / float(white_nits)))


def _peak_nits(linear: torch.Tensor) -> float:
    from .engine.constants import ENGINE_UNIT_NITS

    luminance = 0.2126 * linear[:, 0] + 0.7152 * linear[:, 1] + 0.0722 * linear[:, 2]
    return float(luminance.max()) * ENGINE_UNIT_NITS


def load_image(path, device, max_side: int = 1600, pre_gamma: float = 1.0,
               exposure: float = 1.0, white_nits: float = 203.0,
               box: Optional[Tuple[int, int]] = None) -> PreviewSource:
    """A JXR capture, read as its pixel format says (see ``io.jxr_info``). ``pre_gamma``
    and ``exposure`` are only remembered for the result: the source is never adjusted."""
    from .io.jxr import OptimizedJXRLoader
    from .io.jxr_info import check_hdr_capture, read_jxr_info

    check_hdr_capture(read_jxr_info(path))
    loader = OptimizedJXRLoader(device)
    linear, error, _ = loader.load_jxr(str(path))
    if linear is None:
        raise ValueError(f"Could not read this capture: {error}")
    height, width = linear.shape[-2:]
    return PreviewSource(kind="image", path=str(path), linear=linear, white_nits=white_nits,
                         peak_nits=_peak_nits(linear), width=width, height=height,
                         size=shown_size(width, height, max_side, box),
                         pre_gamma=pre_gamma, exposure=exposure)


def load_video_frame(path, time_s: float, device, max_side: int = 1600, pre_gamma: float = 1.0,
                     exposure: float = 1.0, info=None, white_nits: float = 203.0,
                     box: Optional[Tuple[int, int]] = None) -> PreviewSource:
    """One frame of an HDR10 video, decoded as its tags say: transfer curve, primaries,
    matrix, range and bit depth."""
    from .video.colour import yuv_to_engine
    from .video.reader import check_supported, grab_frame, probe

    if info is None:
        info = probe(path)
        check_supported(info)
    frame = grab_frame(path, time_s, device)
    linear = yuv_to_engine(frame.y, frame.u, frame.v, info.fmt)
    height, width = linear.shape[-2:]
    return PreviewSource(kind="video", path=str(path), linear=linear, white_nits=white_nits,
                         peak_nits=_peak_nits(linear), width=info.width, height=info.height,
                         size=shown_size(width, height, max_side, box),
                         time_s=frame.time_s, pts=frame.pts, duration_s=info.duration_s,
                         pre_gamma=pre_gamma, exposure=exposure)


def blended_light(path, time_s: float, count: int, device, info=None) -> torch.Tensor:
    """The light of ``count`` frames in a row, from the one shown at ``time_s`` on, added up
    as the conversion adds it for motion blur. In the engine's working space, as recorded:
    hand it to :func:`render_preview` as ``light``."""
    from .video.colour import scrgb_to_engine, yuv_to_scrgb
    from .video.reader import grab_frames, probe

    info = info or probe(path)
    total, frames = None, grab_frames(path, time_s, count, device)
    for frame in frames:
        light = yuv_to_scrgb(frame.y, frame.u, frame.v, info.fmt)
        total = light if total is None else total + light
    return scrgb_to_engine(total / len(frames) if len(frames) > 1 else total)


def render_preview(engine, source: PreviewSource, settings: RenderSettings,
                   params: Optional[RenderParams] = None, light: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The 'after' picture, sRGB, at the size shown: (1, 3, *source.size).

    The picture is rendered at its REAL size, exactly as the conversion renders it (the
    operator Auto-select picks, the scene's peak, the sharpening), and only then brought
    to the size of the window.

    * A capture: the picture the conversion writes.
    * A video frame with the frozen ``params`` of a finished analysis: the frame of the
      converted file.
    * A video frame without them is measured by itself: provisional.

    ``light`` replaces the frame's own light in the result (motion blur: see
    :func:`blended_light`); the source view keeps showing the frame itself.
    """
    # One white for both halves of the comparison: the one the source view is made with.
    settings = replace(settings, white_nits=source.white_nits)
    with torch.inference_mode():
        tm, loader, processor = engine.tone_mapper, engine.loader, engine.processor
        # Gamma and Exposure shape the result, as the conversion applies them while reading.
        linear = adjust_scene(source.linear if light is None else light, source.pre_gamma, source.exposure)
        if params is not None:
            from .video.job import _on_device

            params = _on_device(params, linear.device)
            settings = replace(settings, method=params.method)
        out, used = render(tm, loader, processor, linear, settings, params=params)
        source.method = used.method
        return for_display(out, source.size)


def to_rgb8(image: torch.Tensor) -> np.ndarray:
    """(1, 3, h, w) in [0, 1] -> (h, w, 3) uint8, contiguous, ready for a QImage."""
    array = (image[0].detach().float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return np.ascontiguousarray(array.permute(1, 2, 0).cpu().numpy())

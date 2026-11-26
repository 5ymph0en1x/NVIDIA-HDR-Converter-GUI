"""What the user is doing, without any window: the files queued, the options chosen, and
how one queued file is converted. The interface is a thin layer over this module.
"""
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .display import sdr_white_nits
from .engine.pipeline import RenderSettings, export_image, render
from .video.job import Cancelled, VideoAnalysis, VideoSettings, analyse_video, make_engine, render_video

# The four looks offered up front. Everything else lives in the advanced panel.
# Faithful shows the scene at its own luminance; the three others grade it above its shadows.
LOOKS: Dict[str, str] = {"Faithful": "faithful", "Natural": "natural", "Punchy": "punchy", "Golden": "golden"}
DEFAULT_LOOK = "Faithful"

# Advanced: a precise operator, which then replaces the look.
OPERATORS: Dict[str, str] = {
    "Follow the look": "", "Auto-select": "auto", "Perceptual": "perceptual", "Mantiuk06": "mantiuk06", "Drago03": "drago03",
    "Hable": "hable", "ACES": "aces", "Reinhard": "reinhard", "Adaptive": "adaptive", "Neural (AI)": "neural",
}

# How an operator is called in the interface (the engine's identifiers are not shown).
_NAMES = {**{v: k for k, v in LOOKS.items() if v}, **{v: k for k, v in OPERATORS.items() if v}}


def display_name(method: Optional[str]) -> str:
    if method is None:
        return "Auto"
    return _NAMES.get(method, method.replace("_", " ").capitalize())


# Allowed ranges of the numeric options (the panel's sliders use the same).
RANGES = {"pre_gamma": (0.1, 3.0), "exposure": (0.1, 3.0), "sharpness": (0.0, 100.0), "ai_color": (0.0, 100.0)}
_OWN_RESULT = re.compile(r"_SDR(-\d+)?$")          # what output_base() names its files

IMAGE_EXTENSIONS = (".jxr",)
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov")
IMAGE_FORMATS = ("JPEG", "TIFF", "Both")
VIDEO_QUALITIES = ("high", "balanced", "compact")
VIDEO_HEIGHTS = (0, 1080, 720)                 # 0 keeps the size of the source
VIDEO_FRAME_RATES = (0, 120, 60, 30)           # 0 keeps every frame of the source


def reference_white() -> float:
    """What SDR white stands for on this screen, in nits: the anchor of the Faithful
    rendering and of the HDR source view, so that the two agree."""
    return float(sdr_white_nits()[0])


@dataclass
class Options:
    look: str = DEFAULT_LOOK
    operator: str = ""                         # "" follows the look
    pre_gamma: float = 1.0
    exposure: float = 1.0
    sharpness: float = 25.0                    # 0-100
    ai_color: float = 0.0                      # 0-100
    image_format: str = "JPEG"
    video_quality: str = "high"
    video_height: int = 0
    video_fps: int = 0                         # frames per second at most; 0 keeps them all
    video_blur: bool = False                   # at a lower frame rate: motion blur from the dropped frames
    video_enhance: bool = False                # Sharpness and AI colour on videos too
    destination: str = ""                      # "" writes next to the source
    use_gpu: bool = True
    use_fp16: bool = False
    show_histogram: bool = False

    def method(self) -> Optional[str]:
        """The engine's operator name, or None for Auto-select (the engine picks an operator
        from the image's statistics)."""
        chosen = self.operator or LOOKS.get(self.look, LOOKS[DEFAULT_LOOK])
        return None if chosen == "auto" else chosen

    def render_settings(self, kind: str = "image") -> RenderSettings:
        """What to render a capture with, or (``kind="video"``) a frame of a recording.

        A video takes neither Sharpness nor AI colour unless ``video_enhance`` is on:
        nothing is then measured on its pictures, and its analysis does not have to
        render them (about twice as fast)."""
        enhance = kind != "video" or self.video_enhance
        return RenderSettings(method=self.method(), color_strength=self.ai_color if enhance else 0.0,
                              edge_strength=self.sharpness if enhance else 0.0, use_enhancement=True,
                              white_nits=reference_white())

    def video_settings(self) -> VideoSettings:
        return VideoSettings(render=self.render_settings("video"), quality=self.video_quality,
                             out_height=self.video_height or None,
                             frame_rate=float(self.video_fps) if self.video_fps else None,
                             motion_blur=bool(self.video_blur and self.video_fps),
                             pre_gamma=self.pre_gamma, exposure=self.exposure)

    def picture_key(self) -> tuple:
        """Changes exactly when the converted picture would change (not when only the file
        format, the quality or the destination do)."""
        # (the frame rate alone changes no picture; with motion blur it decides whose light adds up)
        return (self.method(), self.pre_gamma, self.exposure, self.sharpness, self.ai_color,
                self.video_enhance, self.video_fps if self.video_blur else 0,
                self.use_gpu, self.use_fp16, reference_white())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Options":
        """Tolerant of what an older or damaged settings file may hold."""
        options, known = cls(), {f.name: f for f in fields(cls)}
        for name, value in (data or {}).items():
            if name not in known:
                continue
            default = getattr(options, name)
            if value is None:
                continue
            try:
                if isinstance(default, bool):
                    value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
                else:
                    value = type(default)(value)
            except (TypeError, ValueError):
                continue
            if isinstance(value, float):
                if not math.isfinite(value):
                    continue
                low, high = RANGES.get(name, (value, value))
                value = min(high, max(low, value))
            setattr(options, name, value)
        if options.look not in LOOKS:                 # includes "Auto", the first look of older versions
            options.look = DEFAULT_LOOK
        if options.operator not in OPERATORS.values():
            options.operator = ""
        if options.image_format not in IMAGE_FORMATS:
            options.image_format = "JPEG"
        if options.video_quality not in VIDEO_QUALITIES:
            options.video_quality = "high"
        if options.video_height not in VIDEO_HEIGHTS:
            options.video_height = 0
        if options.video_fps not in VIDEO_FRAME_RATES:
            options.video_fps = 0
        return options


@dataclass
class Item:
    path: str
    kind: str                                  # "image" or "video"
    state: str = "waiting"                     # waiting / working / done / failed / cancelled
    message: str = ""
    outputs: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def kind_of(path) -> Optional[str]:
    extension = os.path.splitext(os.fspath(path))[1].lower()
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    return None


def collect(paths) -> Tuple[List[Item], List[str]]:
    """Files and folders dropped on the window -> (items to queue, names that were refused).

    A folder contributes the supported files directly inside it, except the results of
    earlier conversions (``*_SDR``); what it contains besides is ignored silently. A file
    dropped by itself that is not supported is reported, and so is a folder that holds
    nothing to convert.
    """
    items, rejected = [], []
    for path in paths:
        path = os.fspath(path)
        if os.path.isdir(path):
            found = 0
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                if _OWN_RESULT.search(os.path.splitext(name)[0]):
                    continue
                if os.path.isfile(full) and kind_of(full):
                    items.append(Item(path=full, kind=kind_of(full)))
                    found += 1
            if not found:
                rejected.append(os.path.basename(os.path.normpath(path)) or path)
        elif os.path.isfile(path) and kind_of(path):
            items.append(Item(path=path, kind=kind_of(path)))
        else:
            rejected.append(os.path.basename(path) or path)
    return items, rejected


class Queue:
    def __init__(self):
        self.items: List[Item] = []
        self.selected = -1

    def add(self, items: List[Item]) -> int:
        """Append the items not already queued. Returns how many were added."""
        known = {os.path.normcase(os.path.abspath(i.path)) for i in self.items}
        added = 0
        for item in items:
            key = os.path.normcase(os.path.abspath(item.path))
            if key not in known:
                known.add(key)
                self.items.append(item)
                added += 1
        if self.selected < 0 and self.items:
            self.selected = 0
        return added

    def remove(self, index: int) -> None:
        del self.items[index]
        if not self.items:
            self.selected = -1
        elif self.selected >= len(self.items) or self.selected > index:
            self.selected = max(0, self.selected - 1)

    @property
    def current(self) -> Optional[Item]:
        return self.items[self.selected] if 0 <= self.selected < len(self.items) else None

    def waiting(self) -> List[Item]:
        return [i for i in self.items if i.state == "waiting"]


def _unique(base: str, suffixes) -> str:
    """``base`` or ``base-2``, ``base-3``... such that none of ``base + suffix`` exists."""
    candidate, n = base, 1
    while any(os.path.exists(candidate + s) for s in suffixes):
        n += 1
        candidate = f"{base}-{n}"
    return candidate


def output_base(src, kind: str, options: Options) -> str:
    """Where the result goes, without extension: ``<name>_SDR`` next to the source or in the
    chosen folder, numbered so that nothing existing is ever overwritten."""
    folder = options.destination or os.path.dirname(os.path.abspath(src))
    base = os.path.join(folder, os.path.splitext(os.path.basename(src))[0] + "_SDR")
    if kind == "video":
        return _unique(base, (".mp4", ".mp4.partial"))
    return _unique(base, (".jpg", "_tonemapped.tiff", "_raw.tiff"))


def describe_output(kind: str, options: Options) -> str:
    """One short line for the interface: what will be written and where."""
    where = "next to the source" if not options.destination else os.path.basename(options.destination) or options.destination
    if kind == "video":
        size = "original size" if not options.video_height else f"{options.video_height}p"
        rate = f", {options.video_fps} fps at most" if options.video_fps else ""
        rate += ", motion blur" if options.video_fps and options.video_blur else ""
        return f"MP4, {options.video_quality} quality, {size}{rate}, {where}"
    what = {"JPEG": "JPEG", "TIFF": "TIFF", "Both": "JPEG and TIFF"}[options.image_format]
    return f"{what}, {where}"


class Engines:
    """The engine objects, built once per device and precision and reused."""

    def __init__(self):
        self._cache = {}

    def device_for(self, options: Options) -> torch.device:
        return torch.device("cuda" if options.use_gpu and torch.cuda.is_available() else "cpu")

    def get(self, options: Options):
        device = self.device_for(options)
        fp16 = bool(options.use_fp16 and device.type == "cuda")
        key = (device.type, fp16)
        if key not in self._cache:
            self._cache[key] = make_engine(device, use_fp16=fp16)
        return self._cache[key], device


def convert_item(item: Item, options: Options, engine, device,
                 progress: Optional[Callable[[str, int, int], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None,
                 analysis: Optional[VideoAnalysis] = None) -> List[str]:
    """Convert one queued file. Returns the files written. Raises :class:`Cancelled` when
    asked to stop (nothing is left behind), any other exception on failure."""
    base = output_base(item.path, item.kind, options)
    if options.destination and not os.path.isdir(options.destination):
        raise FileNotFoundError(f"the destination folder does not exist: {options.destination}")

    if item.kind == "video":
        settings = options.video_settings()
        if analysis is None or analysis.src != item.path or not analysis.matches(settings):
            analysis = analyse_video(item.path, settings, device, progress, cancelled, engine)
        report = render_video(analysis, base + ".mp4", settings, device, progress, cancelled, engine)
        return [report.dst]

    from .io.jxr_info import check_hdr_capture, read_jxr_info

    check_hdr_capture(read_jxr_info(item.path))            # an ordinary JXR picture is not scRGB
    loader = engine.loader
    loader.selected_pre_gamma, loader.selected_auto_exposure = options.pre_gamma, options.exposure
    if progress:
        progress("render", 0, 1)
    linear, error, _ = loader.load_jxr(item.path)
    if linear is None:
        raise ValueError(f"Could not read this capture: {error}")
    if cancelled and cancelled():
        raise Cancelled()
    with torch.inference_mode():
        enhanced, _ = render(engine.tone_mapper, loader, engine.processor, linear, options.render_settings())
    if cancelled and cancelled():
        raise Cancelled()
    written = export_image(engine.processor, linear, enhanced, base, options.image_format, include_raw=True)
    if progress:
        progress("render", 1, 1)
    return written

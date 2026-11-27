"""The one conversion sequence: HDR linear tensor -> finished SDR tensor -> files.

The GUI (single file, folder, preview) and the CLI all go through these
functions, so there is a single place where the order of operations is decided.
"""
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import torch

from .constants import REFERENCE_WHITE_NITS
from .renderings import canonical
from .params import RenderParams


@dataclass(frozen=True)
class RenderSettings:
    """What the user chose. ``method=None`` means Auto-detect."""
    method: Optional[str] = None
    color_strength: float = 0.0     # AI colour (3D LUT), 0-100
    edge_strength: float = 0.0      # sharpness, 0-100
    use_enhancement: bool = True
    # What SDR white stands for, in nits: every rendering keeps the scene's luminance
    # against it, so the result agrees with the HDR source view made with the same white.
    white_nits: float = REFERENCE_WHITE_NITS


def resolve_method(tone_mapper, linear: torch.Tensor, requested: Optional[str]) -> str:
    """The user's pick (under its current name), or the Auto-select oracle's choice for
    this image."""
    if requested is not None:
        return canonical(requested)
    return tone_mapper.select_optimal_tonemap(tone_mapper.analyze_image_statistics(linear))


def tonemap_srgb(tone_mapper, loader, linear: torch.Tensor, method: str,
                 params: Optional[RenderParams] = None,
                 white_nits: float = REFERENCE_WHITE_NITS) -> torch.Tensor:
    """Tone-map a scene-linear image and encode it as sRGB, without enhancement."""
    tonemapped = tone_mapper.apply_tone_mapping(linear.clone(), method=method, params=params,
                                                white_nits=white_nits)
    return loader.linear_to_srgb(tonemapped)


def render(tone_mapper, loader, processor, linear: torch.Tensor, settings: RenderSettings,
           params: Optional[RenderParams] = None, release_cache: bool = True, fast=None
           ) -> Tuple[torch.Tensor, RenderParams]:
    """Full conversion of one image. Returns ``(sRGB tensor, parameters used)``.

    Without ``params`` the image is measured as it is rendered (a photo): the
    returned parameters hold every value that was measured. With frozen
    ``params`` nothing is measured: the recorded values are applied as they are
    (a video frame), and ``settings.method`` is ignored in favour of the
    recorded method.

    ``fast``: the fused stages of a video (:class:`hdr_blaster.video.fused.Chain`). A
    frozen Faithful frame that is neither sharpened nor recoloured is then rendered by
    one operation on the card instead of some ninety, to the very same bits; anything
    else is rendered here as usual.
    """
    if params is None:
        params = RenderParams(method=resolve_method(tone_mapper, linear, settings.method))
    elif not params.frozen:
        raise ValueError(
            "render() replays frozen parameters only: call .freeze() on recorded "
            "parameters first (un-frozen ones would be measured again and overwritten)")
    if (fast is not None and params.frozen and params.method == "faithful"
            and not getattr(processor, "use_fp16", False)
            and not processor.measures(settings.color_strength, settings.edge_strength, settings.use_enhancement)):
        peak = params.measure("scene.peak", lambda: None)          # frozen: the recorded value, or it is missing

        def plain(scene, level, white_nits):
            """The same frame the usual way: what the fused stage is held against."""
            recorded = RenderParams(method="faithful", stats={"scene.peak": level}, frozen=True)
            return _finish(tone_mapper, loader, processor, scene, replace(settings, white_nits=white_nits),
                           recorded, release_cache=False)

        picture = fast.render(plain, linear, peak, settings.white_nits)
        if release_cache:
            processor.clear_gpu_memory()
        return picture, params
    return _finish(tone_mapper, loader, processor, linear, settings, params, release_cache), params


def _finish(tone_mapper, loader, processor, linear: torch.Tensor, settings: RenderSettings,
            params: RenderParams, release_cache: bool) -> torch.Tensor:
    """Tone mapping, sRGB encoding, enhancement: the picture, with ``params`` filled or replayed."""
    srgb = tonemap_srgb(tone_mapper, loader, linear, params.method, params, settings.white_nits)
    enhanced = processor.process_image(
        srgb.clone(),
        color_strength=settings.color_strength,
        edge_strength=settings.edge_strength,
        use_enhancement=settings.use_enhancement,
        params=params,
        keep_levels=tone_mapper.keeps_scene_levels(params.method),
        release_cache=release_cache,
    )
    return enhanced


def measure(tone_mapper, loader, processor, linear: torch.Tensor, settings: RenderSettings,
            release_cache: bool = True) -> RenderParams:
    """What :func:`render` would measure on this image, value for value, without making the
    picture when nothing is measured on the picture.

    That is the case of a look or a classical operator with neither sharpening nor AI
    colour: all it measures is the scene's peak. Everything else is rendered in full, as
    before. ``settings.method`` must be chosen already (a video's is)."""
    if settings.method is not None and not processor.measures(
            settings.color_strength, settings.edge_strength, settings.use_enhancement):
        params = RenderParams(method=canonical(settings.method))
        if tone_mapper.measure_only(linear, params.method, params, settings.white_nits):
            return params
    return render(tone_mapper, loader, processor, linear, settings, release_cache=release_cache)[1]


def analyze(tone_mapper, loader, processor, linear: torch.Tensor,
            settings: RenderSettings) -> RenderParams:
    """Measure an image and return its frozen parameters, ready to be replayed.

    Some values are measured on the tone-mapped image (shadow lift, colour LUT,
    sharpening), so the analysis runs the whole chain; pass a reduced image to
    make it cheap.
    """
    return render(tone_mapper, loader, processor, linear, settings)[1].freeze()


def export_image(processor, linear: torch.Tensor, enhanced: torch.Tensor, out_base: str,
                 fmt: str, include_raw: bool) -> List[str]:
    """Write the result next to ``out_base`` (no extension). ``fmt`` is JPEG, TIFF or Both.

    ``include_raw`` also writes the untouched HDR data as a float TIFF (GUI behaviour).
    Returns the paths written, in order.
    """
    if fmt not in ("JPEG", "TIFF", "Both"):
        raise ValueError(f"Unsupported output format: {fmt}")
    written = []
    if fmt in ("JPEG", "Both"):
        path = f"{out_base}.jpg"
        processor.save_tensor_as_image(enhanced, path, "JPEG")
        written.append(path)
    if fmt in ("TIFF", "Both"):
        if include_raw:
            path = f"{out_base}_raw.tiff"
            processor.save_tensor_as_image(linear, path, "TIFF", is_hdr_data=True)
            written.append(path)
        path = f"{out_base}_tonemapped.tiff"
        processor.save_tensor_as_image(enhanced, path, "TIFF", is_hdr_data=False)
        written.append(path)
    return written

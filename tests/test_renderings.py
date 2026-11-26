"""Every rendering is an aesthetic variation of the SAME picture.

Reported twice by the author, on dark captures: next to the HDR source, the result's dark
zones were glaringly different (lifted up to three times with the classical operators,
crushed to black with Punchy), with a colour cast, and the lift made the noise and the
compression blocks hidden in the dark visible ("aliasing").

So, for every look and every operator:

* the dark tones are the scene's own, untouched (below a quarter of mid-grey);
* mid-grey stays mid-grey: nothing re-exposes the picture;
* the scene's peak reaches white: no dull highlights;
* a brighter tone is never rendered darker; nothing is amplified in the dark.

What distinguishes them lives above the shadows: how highlights roll off, contrast,
saturation, tint.
"""
import pytest
import torch

from conftest import SAMPLES
from hdr_blaster import preview
from hdr_blaster.engine.colour import _srgb_decode
from hdr_blaster.engine.constants import ENGINE_UNIT_NITS, SUPPORTED_TONE_MAPS
from hdr_blaster.engine.params import MissingRenderParam, RenderParams
from hdr_blaster.engine.pipeline import RenderSettings, analyze, render
from hdr_blaster.engine.renderings import GREY, canonical, shadow_weight

LOOKS = ("natural", "punchy", "golden")
OPERATORS = ("perceptual", "reinhard", "hable", "aces", "drago03", "mantiuk06", "adaptive")
ALL = ("faithful",) + LOOKS + OPERATORS
WHITE = 480.0
_LUMA = (0.2126, 0.7152, 0.0722)


def luminance(x):
    return sum(w * x[:, i] for i, w in enumerate(_LUMA))


def patches(device, colours_nits, size=96):
    """A row of flat patches; a number is a grey, a triple a colour. Engine units."""
    row = []
    for colour in colours_nits:
        rgb = (colour,) * 3 if isinstance(colour, (int, float)) else colour
        row.append(torch.tensor(rgb, dtype=torch.float32, device=device).view(1, 3, 1, 1).expand(1, 3, size, size))
    return torch.cat(row, dim=3) / ENGINE_UNIT_NITS


def show(engine, linear, method, edge=0.0, white=WHITE):
    settings = RenderSettings(method=method, edge_strength=edge, white_nits=white)
    return render(engine.tone_mapper, engine.loader, engine.processor, linear, settings)


def centres(image, count, size=96):
    """The middle pixel of each patch: (count, 3)."""
    return torch.stack([image[0, :, size // 2, size // 2 + i * size] for i in range(count)])


# ------------------------------------------------------------------ the rules

def test_the_names():
    assert set(ALL) <= SUPPORTED_TONE_MAPS
    assert canonical("agx_punchy") == "punchy" and canonical("agx") == "natural" and canonical("agx_golden") == "golden"
    assert canonical("uncharted2") == "hable" and canonical("filmic") == "hable"
    assert canonical("punchy") == "punchy" and canonical(None) is None


def test_the_shadow_weight_is_zero_below_a_quarter_of_grey_and_one_from_grey_up(device):
    levels = torch.tensor([0.0, 0.001, GREY / 4, GREY / 2, GREY, 1.0, 5.0], device=device)
    weight = shadow_weight(levels)
    assert weight[:3].tolist() == [0.0, 0.0, 0.0] and weight[4:].tolist() == [1.0, 1.0, 1.0]
    assert 0.3 < float(weight[3]) < 0.7


@pytest.mark.parametrize("rendering", ALL)
def test_dark_tones_are_the_scenes_own(engine, device, rendering):
    method = rendering
    """Greys and colours below a quarter of mid-grey, next to a strong highlight: the
    result is the HDR source view, to the last level."""
    dark = [0.001 * WHITE, 0.005 * WHITE, 0.02 * WHITE, 0.04 * WHITE,
            (0.03 * WHITE, 0.01 * WHITE, 0.02 * WHITE), (0.004 * WHITE, 0.02 * WHITE, 0.035 * WHITE)]
    linear = patches(device, dark + [1500.0])
    out, _ = show(engine, linear, method)
    expected = preview.reference_view(linear, WHITE)
    got, wanted = centres(out, len(dark)), centres(expected, len(dark))
    assert torch.allclose(got, wanted, atol=0.5 / 255), f"{(got - wanted).abs().max() * 255:.2f} levels off"


@pytest.mark.parametrize("white", [203.0, 480.0])
@pytest.mark.parametrize("rendering", ALL)
def test_mid_grey_stays_mid_grey(engine, device, rendering, white):
    method = rendering
    out, _ = show(engine, patches(device, [GREY * white, 0.01 * white, 1200.0]), method, white=white)
    shown = float(luminance(_srgb_decode(centres(out, 1)[None, 0].view(1, 3, 1, 1))))
    assert shown == pytest.approx(GREY, abs=0.006), f"mid-grey shown at {shown:.3f}"


@pytest.mark.parametrize("rendering", ALL)
def test_the_scenes_peak_reaches_white(engine, device, rendering):
    method = rendering
    """No dull highlights: the brightest thing in the scene is (nearly) the brightest the
    file can hold, and nothing exceeds it."""
    out, _ = show(engine, patches(device, [GREY * WHITE, 300.0, 1000.0, 2000.0]), method)
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0 and torch.isfinite(out).all()
    peak = centres(out, 4)[3]
    assert float(peak.max()) >= 0.94, f"the peak is rendered at {float(peak.max()):.3f}"


@pytest.mark.parametrize("rendering", ALL)
def test_a_brighter_tone_is_never_rendered_darker(engine, device, rendering):
    method = rendering
    levels = [0.2, 0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2000]
    out, _ = show(engine, patches(device, levels, size=64), method)
    shown = luminance(_srgb_decode(out))[0, 32, 32::64]
    assert bool((shown[1:] >= shown[:-1] - 1e-5).all()), [round(float(v), 4) for v in shown]


@pytest.mark.parametrize("rendering", ALL)
def test_a_dark_tone_does_not_depend_on_what_surrounds_it(engine, device, rendering):
    method = rendering
    alone, _ = show(engine, patches(device, [8.0, 8.0]), method)
    by_the_sun, _ = show(engine, patches(device, [8.0, 3000.0]), method)
    assert torch.allclose(centres(by_the_sun, 1), centres(alone, 1), atol=0.5 / 255)


@pytest.mark.parametrize("rendering", ALL)
def test_sharpening_moves_no_level(engine, device, rendering):
    method = rendering
    flat = patches(device, [70.0, 70.0])
    soft, _ = show(engine, flat, method, edge=0.0)
    sharp, _ = show(engine, flat, method, edge=100.0)
    assert torch.allclose(sharp, soft, atol=1e-6)


# ------------------------------------------------------------- real captures

@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
@pytest.mark.parametrize("rendering", ALL)
def test_a_captures_dark_zones_match_the_source(engine, linear, rendering):
    method = rendering
    """Every sample, default sharpness: from 0.3 to 15 nits the result carries the light
    the scene had (Auto-select used to show two to three times more)."""
    out, _ = show(engine, linear, method, edge=25.0)
    scene = linear.clamp(min=0.0).amax(1) * ENGINE_UNIT_NITS
    shown = _srgb_decode(out).amax(1) * WHITE
    for low, high in ((0.3, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 15.0)):
        band = (scene >= low) & (scene < high)
        if int(band.sum()) < 500:
            continue
        ratio = float((shown[band] / scene[band]).median())
        assert ratio == pytest.approx(1.0, abs=0.04), f"{low:g} to {high:g} nits shown x{ratio:.2f}"


def _roughness(image, where):
    """How uneven a picture is from one pixel to the next (noise, blocks, staircases), over
    the pixels ``where`` is true: (1, 1, H, W)."""
    grey = image.mean(1, keepdim=True)
    across = (grey[..., 2:] - 2 * grey[..., 1:-1] + grey[..., :-2]).abs()
    down = (grey[..., 2:, :] - 2 * grey[..., 1:-1, :] + grey[..., :-2, :]).abs()
    return float(across[where[..., 1:-1]].mean() + down[where[..., 1:-1, :]].mean())


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
@pytest.mark.parametrize("rendering", ALL)
def test_nothing_hidden_in_the_dark_is_amplified(engine, linear, rendering):
    method = rendering
    """The 'aliasing': lifting shadows made the noise and the compression blocks of the
    capture visible. Wherever the scene is in the shadows (below a quarter of mid-grey,
    its neighbours too), the result must be as smooth as the source view. Every dark pixel
    of every sample is measured. (The first version of this test measured the darkest
    block of a capture, which in a sunlit scene is not dark at all.)"""
    source = preview.reference_view(linear, WHITE)
    level = linear.clamp(min=0.0).amax(1, keepdim=True) * (ENGINE_UNIT_NITS / WHITE)
    dark = torch.nn.functional.max_pool2d(level, 5, stride=1, padding=2) < GREY / 4
    if int(dark.sum()) < 20_000:
        pytest.skip("this capture has no dark zone to speak of")
    out, _ = show(engine, linear, method, edge=25.0)
    assert _roughness(out, dark) <= 1.10 * _roughness(source, dark) + 1e-4


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_the_renderings_are_all_different_pictures(engine, get_linear):
    """They are choices: none may be a copy of another."""
    linear = get_linear(SAMPLES[0])
    rendered = {method: show(engine, linear, method)[0] for method in ALL}
    names = list(rendered)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            difference = float((rendered[a] - rendered[b]).abs().mean()) * 255
            assert difference > 0.25, f"{a} and {b} are the same picture ({difference:.2f} levels apart)"


# ------------------------------------------------------------------ the looks

def test_each_look_has_its_character(engine, device):
    """Above the shadows: Natural is softer than the scene, Punchy has more contrast and
    colour, Golden is warm."""
    light = patches(device, [0.6 * WHITE, (0.30 * WHITE, 0.12 * WHITE, 0.10 * WHITE), GREY * WHITE, 1500.0])
    shown = {m: centres(_srgb_decode(show(engine, light, m)[0]), 3) for m in ("faithful",) + LOOKS}

    def lum(pixel):
        return float(sum(w * pixel[i] for i, w in enumerate(_LUMA)))

    def saturation(pixel):
        return float((pixel.max() - pixel.min()) / pixel.max())

    assert lum(shown["natural"][0]) < lum(shown["faithful"][0]) < lum(shown["punchy"][0]), "soft, true, punchy"
    assert saturation(shown["punchy"][1]) > saturation(shown["faithful"][1]) + 0.02
    assert saturation(shown["natural"][1]) <= saturation(shown["faithful"][1]) + 1e-3
    grey = shown["golden"][2]
    assert float(grey[0]) > float(grey[2]) * 1.05, "warm: more red than blue on a neutral grey"
    neutral = shown["punchy"][2]
    assert float((neutral.max() - neutral.min())) < 0.004


def test_one_stray_pixel_does_not_decide_the_roll_off(engine, device):
    """The peak is where the roll-off ends. A single over-bright pixel (the chroma of a
    4:2:0 video overshoots on sharp coloured edges) must not darken every highlight."""
    scene = patches(device, [GREY * WHITE, 300.0, 900.0], size=128)
    _, clean = show(engine, scene, "faithful")
    stray = scene.clone()
    stray[0, :, 5, 5] = 6000.0 / ENGINE_UNIT_NITS
    with_stray, measured = show(engine, stray, "faithful")
    assert float(measured.stats["scene.peak"]) == pytest.approx(float(clean.stats["scene.peak"]), rel=0.01)
    assert float(with_stray[0, :, 5, 5].min()) == pytest.approx(1.0, abs=1e-4), "the stray pixel is simply white"


# ------------------------------------------------------------ record / replay

@pytest.mark.parametrize("rendering", ALL)
def test_only_the_scenes_peak_is_measured(engine, device, rendering):
    method = rendering
    linear = patches(device, [5.0, 60.0, (900.0, 500.0, 200.0)])
    first, recorded = show(engine, linear, method)
    expected = {"scene.peak"}
    if method == "adaptive":
        expected |= {"adaptive.extreme_highlights", "adaptive.shadow_recovery"}
    assert set(recorded.stats) == expected
    settings = RenderSettings(method=method, white_nits=WHITE)
    frozen = analyze(engine.tone_mapper, engine.loader, engine.processor, linear, settings)
    again, _ = render(engine.tone_mapper, engine.loader, engine.processor, linear, settings, params=frozen)
    assert torch.equal(again, first)
    with pytest.raises(MissingRenderParam, match="scene.peak"):
        render(engine.tone_mapper, engine.loader, engine.processor, linear, settings,
               params=RenderParams(method=method).freeze())


def test_old_names_still_render(engine, device):
    linear = patches(device, [5.0, 60.0, 900.0])
    for old, new in (("agx_punchy", "punchy"), ("agx", "natural"), ("agx_golden", "golden"), ("uncharted2", "hable")):
        a, used = show(engine, linear, old)
        b, _ = show(engine, linear, new)
        assert torch.equal(a, b) and used.method == new

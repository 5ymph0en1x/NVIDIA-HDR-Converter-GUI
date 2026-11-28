"""Each operator is a picture of its own.

Reported by the author: "so little difference between the filters". Measured: Perceptual,
Reinhard and Mantiuk06 shared one curve, Drago's lay on top of it, and Hable's differed
from it by a level and a half. The common rule (tests/test_renderings.py) pins the
shadows, mid-grey and the peak; what an operator may still own is how it treats the
tones above the shadows:

* Reinhard    soft: nothing touched up to mid-grey, a gentle shoulder from there, pastel lights;
* Hable       film: more contrast in the mid-tones, muted colours, lights that fade;
* ACES        vivid: strong contrast, rich mid-tones, lights that burn to white;
* Drago03     flat: the whole range above the shadows compressed evenly, colours kept;
* Perceptual  luminous: a constant-hue gamut fit and a log-light shoulder in ICtCp;
* Mantiuk06   local: a soft roll-off of the large shapes, their detail reinforced.
"""
import math

import pytest
import torch

from conftest import SAMPLES
from hdr_blaster import preview
from hdr_blaster.engine import renderings as R
from hdr_blaster.engine.colour import _srgb_decode
from hdr_blaster.engine.constants import ENGINE_UNIT_NITS
from hdr_blaster.engine.pipeline import RenderSettings, render
from hdr_blaster.engine.renderings import GREY

WHITE = 480.0
SHOWN = ("faithful", "reinhard", "hable", "aces", "drago03", "perceptual", "mantiuk06")
_LUMA = (0.2126, 0.7152, 0.0722)


def patches(device, colours_nits, size=96):
    row = []
    for colour in colours_nits:
        rgb = (colour,) * 3 if isinstance(colour, (int, float)) else colour
        row.append(torch.tensor(rgb, dtype=torch.float32, device=device).view(1, 3, 1, 1).expand(1, 3, size, size))
    return torch.cat(row, dim=3) / ENGINE_UNIT_NITS


def shown(engine, linear, method, white=WHITE):
    """The rendering in LINEAR light relative to white, (1, 3, H, W)."""
    settings = RenderSettings(method=method, edge_strength=0.0, white_nits=white)
    out, _ = render(engine.tone_mapper, engine.loader, engine.processor, linear, settings)
    return _srgb_decode(out)


def centres(image, count, size=96):
    return torch.stack([image[0, :, size // 2, size // 2 + i * size] for i in range(count)])


def lum(pixel):
    return float(sum(w * pixel[i] for i, w in enumerate(_LUMA)))


def saturation(pixel):
    return float((pixel.max() - pixel.min()) / pixel.max().clamp(min=1e-6))


def contrast_at_grey(engine, device, method):
    """How fast the rendering rises around mid-grey, compared with the scene (1.0 = as the scene)."""
    low, high = 0.75 * GREY, 1.35 * GREY
    out = centres(shown(engine, patches(device, [low * WHITE, high * WHITE, 1000.0]), method), 2)
    return math.log(lum(out[1]) / lum(out[0])) / math.log(high / low)


# ------------------------------------------------ colour and exposure checks


def test_perceptual_gamut_fit_matches_absolute_primary_references(engine, device):
    # Calculated independently in binary64 from the BT.709/2020 chromaticities,
    # ST 2084 and BT.2100 matrices. At this white the red and blue intensities
    # fall below the shoulder; only their chroma needs to fit the display.
    values = [(1000., 0., 0.), (0., 0., 1000.), (332.0982, -24.91, -3.63), 1000.]
    linear = patches(device, values)
    out = centres(shown(engine, linear, "perceptual"), len(values)) * WHITE
    expected = out.new_tensor([(480., 146.413, 95.439), (8.809, 71.862, 480.)])
    assert torch.allclose(out[:2], expected, atol=0.15, rtol=0)
    source = linear[:, :, 48:49, 48::96] * ENGINE_UNIT_NITS
    a = R.to_ictcp(source.double())[0, 1:, 0, :3]
    b = R.to_ictcp(out.T.reshape(1, 3, 1, -1).double())[0, 1:, 0, :3]
    drift = torch.rad2deg(torch.atan2(a[0]*b[1]-a[1]*b[0], (a*b).sum(0)))
    assert float(drift.abs().max()) < 0.12


@pytest.mark.parametrize("white", [203., 480.])
def test_perceptual_neutral_ramp_never_reverses(device, white):
    ramp = torch.linspace(0, 10000 / white, 8192, device=device).view(1, 1, 1, -1)
    out = R.perceptual(ramp.expand(1, 3, 1, -1), ramp.max(), white)
    assert (out[..., 1:] >= out[..., :-1]).all()
    assert torch.equal(out[:, 0], out[:, 2])
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_classical_gamut_fit_preserves_signed_luminance_and_representable_shadows(device):
    from hdr_blaster.engine.colour import nonnegative_rgb

    colours = patches(device, [(80., -4., 15.), (-2., 30., 1.), (2., 6., 8.), (0., 0., 0.)])
    mapped = nonnegative_rgb(colours)
    assert float(mapped.min()) >= 0
    assert torch.allclose(R.luminance(mapped), R.luminance(colours), atol=1e-8, rtol=1e-6)
    assert torch.equal(mapped[..., 192:], colours[..., 192:])


def test_uniform_image_has_zero_contrast_at_every_size(engine, device):
    choices = []
    for size in (1, 4, 32, 257):
        scene = torch.full((1, 3, size, size), 0.81, device=device)
        stats = engine.tone_mapper.analyze_image_statistics(scene)
        assert all(math.isfinite(value) for value in stats.values())
        assert stats["contrast_score"] == stats["local_contrast"] == 0
        choices.append(engine.tone_mapper.select_optimal_tonemap(stats))
    assert len(set(choices)) == 1


@pytest.mark.parametrize("white,floor", [(203., 0.038), (480., 0.062)])
def test_cyberpunk_sun_and_sky_remain_distinct(engine, device, white, floor):
    # Native pixels of the sunrise at (664, 520) and (820, 288), plus its 99.99th percentile peak.
    linear = patches(device, [(998.75, 999.0625, 998.984375), (941.875, 797.265625, 635.625), 1185.078125])
    for method in ("perceptual", "natural", "punchy", "golden"):
        out = centres(shown(engine, linear, method, white), 2)
        assert lum(out[0]) / lum(out[1]) - 1 > floor, method


# ------------------------------------------------------------------ Reinhard

def test_the_reinhard_shoulder_leaves_the_knee_smoothly_and_lands_the_peak_on_white(device):
    x = torch.linspace(0.0, 4.0, 4001, dtype=torch.float64, device=device)
    peak = torch.tensor(4.0, dtype=torch.float64, device=device)
    y = R.from_knee(R.reinhard_shoulder, x, peak)
    below = x <= GREY
    assert torch.equal(y[below], x[below]), "up to mid-grey: the scene itself"
    i = int(below.sum())
    assert float((y[i + 1] - y[i]) / (x[i + 1] - x[i])) == pytest.approx(1.0, abs=0.01), "no corner at the knee"
    assert float(y[-1]) == pytest.approx(1.0, abs=1e-9) and bool((y[1:] >= y[:-1]).all())
    assert float(y.max()) <= 1.0 + 1e-9
    fits = R.from_knee(R.reinhard_shoulder, x.clamp(max=1.0), torch.tensor(0.7, dtype=torch.float64, device=device))
    assert torch.allclose(fits, x.clamp(max=1.0)), "a scene that fits under white needs no roll-off"


def test_reinhard_touches_nothing_up_to_mid_grey_and_is_soft_above(engine, device):
    tones = [0.05 * WHITE, 0.10 * WHITE, 0.17 * WHITE, (0.16 * WHITE, 0.07 * WHITE, 0.03 * WHITE)]
    linear = patches(device, tones + [0.9 * WHITE, 1000.0])
    out = shown(engine, linear, "reinhard")
    scene = (linear * ENGINE_UNIT_NITS / WHITE)
    assert torch.allclose(centres(out, 4), centres(scene, 4), atol=0.002), "the scene itself up to mid-grey"
    faithful = shown(engine, linear, "faithful")
    assert lum(centres(out, 5)[4]) < lum(centres(faithful, 5)[4]) - 0.12, "softer than the faithful roll-off"
    assert contrast_at_grey(engine, device, "reinhard") == pytest.approx(0.97, abs=0.05)


def test_reinhard_lets_bright_colours_go_pastel(engine, device):
    """Less colour than the scene's own, and than a rendering that keeps it (Drago). Faithful
    is no longer that rendering: since 3.1.2 it shows the weakest channel of a light as the
    source view does, and this colour comes out of both alike."""
    colour = (900.0, 380.0, 200.0)
    light = patches(device, [colour, GREY * WHITE, 1000.0])
    soft = saturation(centres(shown(engine, light, "reinhard"), 1)[0])
    assert soft < (1.0 - min(colour) / max(colour)) - 0.1
    assert soft < saturation(centres(shown(engine, light, "drago03"), 1)[0]) - 0.1


# --------------------------------------------------------------------- Hable

def test_hable_is_film_more_contrast_and_muted_colours(engine, device):
    assert contrast_at_grey(engine, device, "hable") > contrast_at_grey(engine, device, "reinhard") + 0.12
    assert contrast_at_grey(engine, device, "hable") > 1.05
    colour = patches(device, [(0.30 * WHITE, 0.12 * WHITE, 0.10 * WHITE), GREY * WHITE, 1000.0])
    film, true = centres(shown(engine, colour, "hable"), 2), centres(shown(engine, colour, "faithful"), 2)
    assert saturation(film[0]) < saturation(true[0]) - 0.03, "muted"
    assert float(film[1].max() - film[1].min()) < 0.004, "a grey stays grey"


# ---------------------------------------------------------------------- ACES

def test_aces_is_vivid_strong_contrast_rich_mid_tones_lights_that_burn_to_white(engine, device):
    assert contrast_at_grey(engine, device, "aces") > 1.2
    scene = patches(device, [(0.30 * WHITE, 0.12 * WHITE, 0.10 * WHITE), (1400.0, 500.0, 250.0), 1500.0])
    vivid, true = centres(shown(engine, scene, "aces"), 2), centres(shown(engine, scene, "faithful"), 2)
    assert saturation(vivid[0]) > saturation(true[0]) + 0.03, "richer mid-tones"
    assert saturation(vivid[1]) < (1.0 - 250.0 / 1400.0) - 0.15, "a bright colour burns towards white"
    assert lum(vivid[1]) > lum(true[1]) + 0.05, "further than Faithful lets it: it is brighter for it"


# --------------------------------------------------------------------- Drago

def test_drago_is_flat_and_keeps_colours(engine, device):
    assert contrast_at_grey(engine, device, "drago03") < 0.9
    scene = patches(device, [(900.0, 380.0, 200.0), GREY * WHITE, 1000.0])
    out = centres(shown(engine, scene, "drago03"), 1)[0]
    assert float(out[1] / out[0]) == pytest.approx(380.0 / 900.0, rel=0.02), "the colour's own proportions"
    assert float(out[2] / out[0]) == pytest.approx(200.0 / 900.0, rel=0.02)


# ---------------------------------------------------------------- Perceptual

def test_ictcp_is_the_one_of_bt2100(device):
    """Known values: a neutral has no chroma and its intensity is the PQ code of its
    luminance; the way back gives the colour again."""
    nits = torch.tensor([[100.0, 100.0, 100.0], [600.0, 40.0, 250.0], [3.0, 80.0, 900.0], [0.02, 0.5, 0.1]],
                        dtype=torch.float64, device=device).t().reshape(1, 3, 1, 4)
    ictcp = R.to_ictcp(nits)
    assert float(ictcp[0, 0, 0, 0]) == pytest.approx(float(R.pq_encode(torch.tensor(100.0, dtype=torch.float64))), abs=1e-6)
    assert float(ictcp[0, 1:, 0, 0].abs().max()) < 1e-5, "a neutral has no chroma"
    assert float(ictcp[0, 1:, 0, 1].abs().max()) > 0.05
    assert torch.allclose(R.from_ictcp(ictcp), nits, rtol=1e-6, atol=1e-6)


def test_perceptual_reserves_a_stop_for_neutral_highlight_contrast(engine, device):
    greys = patches(device, [20.0, 150.0, 400.0, 700.0, 1000.0])
    level = greys * ENGINE_UNIT_NITS / WHITE
    # Integrated unit-to-final log slope for a one-stop shoulder, half-stop transition.
    slope = 0.75 / (0.75 + math.log2(1000.0 / WHITE))
    u = torch.log2(level) + 1
    bend = u.clamp(0, 0.5)
    expected = torch.where(u <= 0, level, 2 ** (-1 + slope*u + (1-slope)*bend*(1-bend)))
    assert torch.allclose(shown(engine, greys, "perceptual"), expected, atol=0.004)


def test_perceptual_keeps_the_hue_and_the_brightness_of_a_coloured_light(engine, device):
    """A neon far above white. Rolled off on its brightest channel it stays saturated and
    turns dark; in ICtCp it keeps its hue and stays a light: its colour yields instead."""
    neon = (1500.0, 160.0, 900.0)
    scene = patches(device, [neon, GREY * WHITE, 1600.0])
    kept, faithful = shown(engine, scene, "perceptual"), shown(engine, scene, "faithful")
    assert float(kept.min()) >= 0.0 and float(kept.max()) <= 1.0

    def hue(pixel_nits):
        ictcp = R.to_ictcp(pixel_nits.double().view(1, 3, 1, 1))
        return math.degrees(math.atan2(float(ictcp[0, 2]), float(ictcp[0, 1])))

    source_hue = hue(torch.tensor(neon, device=device))
    kept_pixel, faithful_pixel = centres(kept, 1)[0], centres(faithful, 1)[0]
    assert abs(hue(kept_pixel * WHITE) - source_hue) < 4.0, "the same hue"
    assert lum(kept_pixel) > lum(faithful_pixel) * 1.25, "it stays a light"
    assert saturation(kept_pixel) < saturation(faithful_pixel), "its colour yields, not its brightness"


def test_perceptual_leaves_what_fits_alone(engine, device):
    tones = [8.0, 60.0, (120.0, 40.0, 20.0), (30.0, 200.0, 90.0)]
    scene = patches(device, tones + [1200.0])
    out = shown(engine, scene, "perceptual")
    assert torch.allclose(centres(out, 4), centres(scene * ENGINE_UNIT_NITS / WHITE, 4), atol=0.002)


# ----------------------------------------------------------------- Mantiuk06

def rippled(device, level_nits, size=384, period=12, depth=0.10):
    """A plateau with a fine ripple of +-10 % on it, next to a highlight. Engine units."""
    x = torch.arange(size, device=device, dtype=torch.float32)
    ripple = 1.0 + depth * torch.sin(2.0 * math.pi * x / period)
    plateau = (level_nits * ripple).view(1, 1, 1, size).expand(1, 3, size, size)
    light = torch.full((1, 3, size, 96), 1200.0, device=device)
    return torch.cat([plateau, light], dim=3) / ENGINE_UNIT_NITS


def ripple_depth(picture, size=384):
    row = picture[0, 1, size // 2, 96:size - 96]
    return float((row.max() - row.min()) / (row.max() + row.min()))


def test_mantiuk_reinforces_detail_above_the_shadows_only(engine, device):
    bright = rippled(device, 0.5 * WHITE)
    assert ripple_depth(shown(engine, bright, "mantiuk06")) > 1.3 * 0.10, "detail reinforced"
    assert ripple_depth(shown(engine, bright, "reinhard")) < 0.10, "a global curve flattens it"
    dark = rippled(device, 0.03 * WHITE)
    assert ripple_depth(shown(engine, dark, "mantiuk06")) == pytest.approx(0.10, abs=0.003), "nothing in the dark"


def test_mantiuk_draws_no_halo_along_an_edge(engine, device):
    """A dark wall next to a bright one: each stays flat right up to the edge."""
    wall = torch.cat([torch.full((1, 3, 256, 256), 0.25 * WHITE, device=device),
                      torch.full((1, 3, 256, 256), 1.6 * WHITE, device=device)], dim=3) / ENGINE_UNIT_NITS
    out = shown(engine, wall, "mantiuk06")[0, 1, 128]
    dark_side, bright_side = out[:250], out[262:]
    assert float(dark_side.max() / dark_side.min()) < 1.06
    assert float(bright_side.max() / bright_side.min()) < 1.06


# ------------------------------------------------------- pictures of their own

@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_every_operator_offered_is_a_picture_of_its_own(engine, get_linear):
    """On the captures, at the author's white: any two renderings offered differ visibly.
    By eight levels or more over the hundredth of the picture where they differ most (the
    lights: that is where they may), and by 1.3 levels on average above mid-grey. The
    proportion of protected shadows in a full frame should not set this threshold. Before, four
    operators were within half a level of one another."""
    from hdr_blaster.session import OPERATORS

    offered = [name for name in OPERATORS.values() if name not in ("", "auto", "neural")]
    assert set(offered) <= set(SHOWN) | {"adaptive"}
    names = ["faithful"] + offered
    mean = torch.zeros(len(names), len(names))
    top = torch.zeros(len(names), len(names))
    used = SAMPLES[:4]
    for sample in used:
        linear = get_linear(sample)
        above_grey = linear.amax(1).flatten()[::23] > (GREY * WHITE / ENGINE_UNIT_NITS)
        assert above_grey.any()
        pictures = [preview._srgb(shown(engine, linear, name)) for name in names]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                difference = (pictures[i] - pictures[j]).abs().mean(1).flatten()[::23] * 255.0
                mean[i, j] += float(difference[above_grey].mean()) / len(used)
                top[i, j] += float(torch.quantile(difference, 0.99)) / len(used)
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            assert float(top[i, j]) >= 8.0, f"{a} and {b}: only {float(top[i, j]):.1f} levels apart where they differ most"
            assert float(mean[i, j]) >= 1.3, f"{a} and {b}: only {float(mean[i, j]):.2f} levels apart on average"

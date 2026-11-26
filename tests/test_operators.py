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
* Perceptual  luminous: the roll-off of BT.2390 in ICtCp, hue and brightness kept;
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
    light = patches(device, [(900.0, 380.0, 200.0), GREY * WHITE, 1000.0])
    assert saturation(centres(shown(engine, light, "reinhard"), 1)[0]) < \
        saturation(centres(shown(engine, light, "faithful"), 1)[0]) - 0.1


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
    assert saturation(vivid[1]) < saturation(true[1]) - 0.15, "a bright colour burns towards white"


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


def test_perceptual_is_the_faithful_roll_off_on_a_neutral(engine, device):
    greys = patches(device, [20.0, 150.0, 400.0, 700.0, 1000.0])
    assert torch.allclose(shown(engine, greys, "perceptual"), shown(engine, greys, "faithful"), atol=0.004)


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
    lights: that is where they may), and by 1.3 levels or more on average. Before, four
    operators were within half a level of one another.

    One pair is allowed less on average: Perceptual is the Faithful roll-off done in
    another colour space. The two agree wherever nothing is brighter than white and part
    only on the lights, where they part clearly."""
    from hdr_blaster.session import OPERATORS

    offered = [name for name in OPERATORS.values() if name not in ("", "auto", "neural")]
    assert set(offered) <= set(SHOWN) | {"adaptive"}
    names = ["faithful"] + offered
    mean = torch.zeros(len(names), len(names))
    top = torch.zeros(len(names), len(names))
    used = SAMPLES[:4]
    for sample in used:
        linear = get_linear(sample)
        pictures = [preview._srgb(shown(engine, linear, name)) for name in names]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                difference = (pictures[i] - pictures[j]).abs().mean(1).flatten()[::23] * 255.0
                mean[i, j] += float(difference.mean()) / len(used)
                top[i, j] += float(torch.quantile(difference, 0.99)) / len(used)
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            assert float(top[i, j]) >= 8.0, f"{a} and {b}: only {float(top[i, j]):.1f} levels apart where they differ most"
            least = 0.7 if {a, b} == {"faithful", "perceptual"} else 1.3
            assert float(mean[i, j]) >= least, f"{a} and {b}: only {float(mean[i, j]):.2f} levels apart on average"

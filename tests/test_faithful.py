"""The Faithful rendering: the SDR result shows the scene at the luminance the HDR source
view shows it. Shadows and midtones are the same picture; only what is brighter than the
screen's white is compressed.

Found on a dark capture: with Auto the shadows of the result came out two to three times
brighter than the source next to it, because the classical operators end with a shadow
lift (a power below 1, i.e. a gamma change) and every operator chooses its own exposure
from the image's statistics.
"""
import pytest
import torch

from conftest import SAMPLES
from hdr_blaster import preview
from hdr_blaster.engine import renderings as R
from hdr_blaster.engine.colour import _srgb_decode
from hdr_blaster.engine.constants import ENGINE_UNIT_NITS, SUPPORTED_TONE_MAPS
from hdr_blaster.engine.params import MissingRenderParam, RenderParams
from hdr_blaster.engine.pipeline import RenderSettings, analyze, render
from hdr_blaster.engine.tonemap import faithful_curve

WHITE = 480.0
_LUMA = (0.2126, 0.7152, 0.0722)


def luminance(x):
    return sum(w * x[:, i] for i, w in enumerate(_LUMA))


def patches(device, colours_nits, size=64):
    """A row of flat patches, one per (r, g, b) in nits, in the engine's units."""
    row = [torch.tensor(c, dtype=torch.float32, device=device).view(1, 3, 1, 1).expand(1, 3, size, size)
           for c in colours_nits]
    return torch.cat(row, dim=3) / ENGINE_UNIT_NITS


def faithful(engine, linear, edge=0.0, white=WHITE):
    settings = RenderSettings(method="faithful", edge_strength=edge, white_nits=white)
    return render(engine.tone_mapper, engine.loader, engine.processor, linear, settings)


# ----------------------------------------------------------------- the curve

def test_the_curve_leaves_tones_alone_and_lands_the_peak_on_white(device):
    nits = torch.linspace(0.0, 1000.0, 4001, device=device)
    out = faithful_curve(nits, WHITE, 1000.0)
    low = nits <= 250.0
    assert torch.allclose(out[low], nits[low], rtol=1e-4, atol=1e-3), "untouched well below white"
    # single precision leaves about 0.02 nit of noise on the PQ scale: a hundredth of an 8-bit step
    assert bool((out[1:] >= out[:-1] - 0.05).all()), "never reverses the order of two tones"
    assert float(out.max()) <= WHITE + 0.01, "nothing is brighter than white"
    assert float(out[-1]) == pytest.approx(WHITE, abs=0.5), "the scene's peak is white"
    assert float(out[nits == 600.0]) < 600.0, "what exceeds white is brought down"
    above = out[~low]
    assert float((above[1:] - above[:-1]).max()) <= 0.30, "no step in the shoulder (slope at most 1)"


def test_a_scene_that_fits_under_white_is_not_touched_at_all(device):
    nits = torch.linspace(0.0, 400.0, 801, device=device)
    assert torch.allclose(faithful_curve(nits, WHITE, 400.0), nits, rtol=1e-5, atol=1e-4)
    assert torch.allclose(faithful_curve(nits, WHITE, WHITE), nits, rtol=1e-5, atol=1e-4)


def test_a_frame_brighter_than_the_recorded_peak_still_ends_at_white(device):
    """A video replays a peak smoothed over time: a frame can exceed it."""
    out = faithful_curve(torch.tensor([900.0, 1500.0, 9000.0], device=device), WHITE, 800.0)
    assert float(out.max()) <= WHITE + 0.01 and float(out[1]) == pytest.approx(WHITE, abs=0.5)


# ------------------------------------------------------- the rendered picture

def test_faithful_is_a_tone_map_like_the_others():
    assert "faithful" in SUPPORTED_TONE_MAPS


def test_the_result_is_the_source_view_below_the_shoulder(engine, device):
    colours = [(1, 1, 1), (10, 6, 3), (50, 50, 50), (20, 100, 40), (200, 150, 100), (0, 0, 0)]
    linear = patches(device, colours)
    out, params = faithful(engine, linear)
    expected = preview.reference_view(linear, WHITE)
    assert torch.allclose(out, expected, atol=0.5 / 255), "the same picture as the HDR source view"
    assert params.method == "faithful"


def test_only_the_highlights_are_compressed_and_keep_their_hue(engine, device):
    linear = patches(device, [(30, 30, 30), (1000, 500, 250), (700, 700, 700)])
    out, _ = faithful(engine, linear)
    grey = out[0, :, 32, 32]
    sun = _srgb_decode(out[0, :, 32, 96])
    cloud = _srgb_decode(out[0, :, 32, 160])
    assert torch.allclose(grey, preview.reference_view(linear, WHITE)[0, :, 32, 32], atol=0.5 / 255)
    assert float(sun[0]) == pytest.approx(1.0, abs=0.01), "the peak reaches white without clipping past it"
    assert float(sun[2]) == pytest.approx(250.0 / WHITE, abs=0.005), "its blue fitted under white: it is the scene's"
    assert hue(sun) == pytest.approx(hue(torch.tensor([1000.0, 500.0, 250.0])), abs=0.01), "the same hue"
    assert WHITE / 1000.0 < float(cloud[0]) < 1.0, "700 nits stays between its share of the peak and white"


# ------------------------------------------------- a coloured light above white
# Found on a sunrise (Cyberpunk 2077, 2026.11.25 - 11.16.07.87): the sun's glow, a fifth of
# the picture, came out as a flat peach surface. One gain set on the brightest channel kept
# the colour of every light whole, and gave up its luminance for it: 999 / 852 / 684 nits
# were shown as 255 / 238 / 216 under a white of 480 nits, where the source view shows white.
# Faithful and the graded looks now preserve the contrast between
# that glow and the solar disc itself, including after quantisation to an SDR file.

def hue(pixel):
    """Where the middle channel stands between the weakest and the brightest: 0 to 1."""
    low, mid, top = sorted(float(v) for v in pixel)
    return (mid - low) / (top - low)


def lights(engine, device, colours, white=WHITE):
    """The centre of each patch, in linear light relative to white: (count, 3)."""
    out, _ = faithful(engine, patches(device, colours), white=white)
    return torch.stack([_srgb_decode(out[0, :, 32, 32 + 64 * i]) for i in range(len(colours))])


def curve(nits, peak, white=WHITE):
    return float(R.faithful_curve(torch.tensor(float(nits)), white, peak)) / white


def test_the_legacy_3_1_2_shoulder_has_its_original_highlight_mapping(device):
    x = patches(device, [(999.0, 852.0, 684.0)]) * (ENGINE_UNIT_NITS / WHITE)
    glow = R.roll_off(x, WHITE, torch.tensor(1185.0 / WHITE, device=device))[0, :, 32, 32]
    assert float(glow[0]) == pytest.approx(curve(999.0, 1185.0), abs=0.004)
    assert float(glow[2]) == pytest.approx(curve(684.0, 1185.0), abs=0.004), "its weakest channel, through the curve too"
    assert float(glow.min()) >= curve(WHITE, 1185.0) - 0.004, "no darker than white itself is rendered"
    assert 1.0 - float(glow.min() / glow.max()) < 0.06, "it has burnt: the scene's glow has 0.32 of saturation"
    assert hue(glow) == pytest.approx(hue(torch.tensor([999.0, 852.0, 684.0])), abs=0.02), "the same hue"


@pytest.mark.parametrize("white", [203.0, 480.0])
def test_the_solar_disc_and_warm_sky_survive_sdr_quantisation(engine, device, white):
    # RGB measured in the original Cyberpunk sunrise at (664,520) and (820,288).
    disc, sky = (999.0, 999.0, 999.0), (942.0, 797.0, 636.0)
    out, _ = faithful(engine, patches(device, [disc, sky, (1185.0,) * 3]), white=white)
    disc8, sky8 = [out[0, :, 32, x].mul(255).round() for x in (32, 96)]
    assert float(disc8.max() - disc8.min()) == 0
    assert float(disc8.min()) >= 245
    assert float(disc8[1] - sky8[1]) >= 6
    assert float(sky8[0] - sky8[2]) >= 12


def test_neutral_highlights_retain_contrast_up_to_the_peak(device):
    peak = torch.tensor(5.0, device=device)
    ramp = torch.linspace(0.5, 5.0, 1001, device=device).view(1, 1, 1, -1).expand(1, 3, 1, -1)
    out = R.faithful_highlights(ramp, peak)
    assert bool((out[..., 1:] > out[..., :-1]).all())
    assert float(out[..., -1].min()) == pytest.approx(1.0)


def test_a_neon_keeps_its_colour(engine, device):
    """One channel far above white, the others under it: they are shown as they are, which
    is what the source view shows."""
    neon = (1000.0, 60.0, 800.0)
    linear = patches(device, [neon])
    shown = lights(engine, device, [neon])[0]
    assert float(shown[0]) == pytest.approx(1.0, abs=0.004)
    assert float(shown[1]) == pytest.approx(60.0 / WHITE, abs=0.002), "the channel that fits is the scene's"
    assert hue(shown) == pytest.approx(hue(torch.tensor(neon)), abs=0.01)
    assert 1.0 - float(shown.min() / shown.max()) > 0.85
    source = _srgb_decode(preview.reference_view(linear, WHITE)[0, :, 32, 32])
    assert torch.allclose(shown[:2], source[:2], atol=0.004)


def test_a_pure_colour_stays_pure(engine, device):
    shown = lights(engine, device, [(1000.0, 500.0, 0.0)])[0]
    assert float(shown[2]) == 0.0 and float(shown[1] / shown[0]) == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("white", [203.0, 480.0])
def test_the_hue_never_moves_and_nothing_passes_white(engine, device, white):
    colours = [(900.0, 700.0, 150.0), (150.0, 700.0, 900.0), (2000.0, 1900.0, 300.0), (300.0, 80.0, 1200.0),
               (640.0, 630.0, 20.0), (1500.0, 1400.0, 1300.0), (90.0, 400.0, 250.0)]
    shown = lights(engine, device, colours + [(2000.0, 2000.0, 2000.0)], white=white)
    assert float(shown.max()) <= 1.0 and float(shown.min()) >= 0.0
    for colour, pixel in zip(colours, shown):
        order = sorted(range(3), key=lambda i: colour[i])
        assert sorted(range(3), key=lambda i: float(pixel[i])) == order, f"{colour}: the channels changed places"
        assert hue(pixel) == pytest.approx(hue(torch.tensor(colour)), abs=0.02), f"{colour}"


def test_what_fits_is_the_scene_to_the_last_bit(device):
    """Under the knee nothing is computed: the picture is handed back as it came."""
    g = torch.Generator(device=device).manual_seed(11)
    scene = torch.rand((1, 3, 64, 64), device=device, generator=g) * 0.4
    scene[..., :8, :8] = 3.0
    out = R.faithful_highlights(scene, torch.tensor(3.0, device=device))
    assert torch.equal(out[..., 8:, :], scene[..., 8:, :]) and torch.equal(out[..., 8:], scene[..., 8:])
    fits = torch.rand((1, 3, 64, 64), device=device, generator=g) * 0.95
    assert torch.equal(R.faithful_highlights(fits, torch.tensor(0.95, device=device)), fits), "a scene under white"
    assert float(out[..., :8, :8].min()) == pytest.approx(1.0, abs=1e-3), "the peak is white"


def test_no_step_as_a_channel_passes_the_knee_or_another_channel(device):
    """The three channels of a light are not treated alike: the picture must not show where
    one takes the place of another, nor where one starts to be rolled off."""
    steps = torch.linspace(0.0, 2.4, 4801, device=device)
    peak = torch.tensor(2.5, device=device)
    for fixed in ((2.4, 0.3), (2.4, 1.6), (0.9, 0.2), (1.2, 1.1)):
        for place in range(3):                     # the moving channel, in each position
            others = [torch.full_like(steps, value) for value in fixed]
            scene = torch.stack(others[:place] + [steps] + others[place:]).view(1, 3, 1, -1)
            out = R.faithful_highlights(scene, peak)
            assert float((out[..., 1:] - out[..., :-1]).abs().max()) < 0.002, f"{fixed}, channel {place}"
            assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
@pytest.mark.parametrize("white", [203.0, 480.0])
def test_the_legacy_3_1_2_shoulder_keeps_broadband_highlights_bright(linear, white):
    """The original white-floor guarantee of the 3.1.2 graded shoulder on actual captures."""
    scene = linear.float().clamp(min=0.0) * (ENGINE_UNIT_NITS / white)
    above = scene.amin(dim=1) > 1.0
    if int(above.sum()) < 500:
        pytest.skip("nothing in this capture is above white on its three channels")
    peak = R.scene_peak(scene)
    floor = curve(white, float(peak) * white, white)
    weakest = R.roll_off(scene, white, peak).amin(dim=1)[above]
    assert float(weakest.min()) >= floor - 0.004, f"{float(weakest.min()) * white:.0f} nits where white gives {floor * white:.0f}"


def test_the_white_level_sets_the_exposure(engine, device):
    linear = patches(device, [(50, 50, 50), (100, 100, 100)])
    at_480, _ = faithful(engine, linear, white=480.0)
    at_203, _ = faithful(engine, linear, white=203.0)
    assert torch.allclose(at_480, preview.reference_view(linear, 480.0), atol=0.5 / 255)
    assert torch.allclose(at_203, preview.reference_view(linear, 203.0), atol=0.5 / 255)
    assert float(at_203.mean()) > float(at_480.mean())


def test_sharpening_leaves_the_levels_where_they_are(engine, device):
    """The classical sharpening rescales the whole picture to its brightest edge, which
    dims it (about 8 % of the light at the default strength). Not here."""
    flat = patches(device, [(40, 40, 40)], size=96)
    soft, _ = faithful(engine, flat, edge=0.0)
    sharp, _ = faithful(engine, flat, edge=100.0)
    assert torch.allclose(sharp, soft, atol=1e-6), "no edge, no change"
    stripes = patches(device, [(10, 10, 10), (80, 80, 80), (10, 10, 10), (80, 80, 80)], size=48)
    soft, _ = faithful(engine, stripes, edge=0.0)
    sharp, _ = faithful(engine, stripes, edge=50.0)
    inside = [slice(8, 40), slice(56, 88), slice(104, 136), slice(152, 184)]
    for columns in inside:
        assert torch.allclose(sharp[..., 8:40, columns], soft[..., 8:40, columns], atol=1e-6), "flat areas untouched"
    assert not torch.equal(sharp, soft), "the edges themselves are reinforced"


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_a_capture_keeps_its_shadows_and_midtones(engine, linear):
    """Every sample, default sharpness: between 0.3 and 100 nits the result carries the
    light the scene had, within 3 %, and the contrast is the scene's."""
    out, _ = faithful(engine, linear, edge=25.0)
    scene = luminance(linear).clamp_min(0) * ENGINE_UNIT_NITS
    shown = luminance(_srgb_decode(out)) * WHITE
    for low, high in ((0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, 100.0)):
        band = (scene >= low) & (scene < high)
        if int(band.sum()) < 500:
            continue
        ratio = float((shown[band] / scene[band]).median())
        assert ratio == pytest.approx(1.0, abs=0.03), f"{low:g} to {high:g} nits shown x{ratio:.2f}"
    band = (scene > 1.0) & (scene < 100.0)
    x, y = torch.log(scene[band][::3]), torch.log(shown[band][::3].clamp(min=1e-6))
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    assert slope == pytest.approx(1.0, abs=0.03), f"contrast x{slope:.2f}"
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0


# ------------------------------------------------------------ record / replay

def test_the_peak_is_measured_once_and_replayed(engine, device):
    linear = patches(device, [(30, 30, 30), (900, 600, 300)])
    settings = RenderSettings(method="faithful", edge_strength=25.0, white_nits=WHITE)
    first, recorded = render(engine.tone_mapper, engine.loader, engine.processor, linear, settings)
    assert float(recorded.stats["scene.peak"]) == pytest.approx(900.0 / WHITE, rel=1e-3)
    frozen = analyze(engine.tone_mapper, engine.loader, engine.processor, linear, settings)
    again, _ = render(engine.tone_mapper, engine.loader, engine.processor, linear, settings, params=frozen)
    assert torch.equal(again, first)
    empty = RenderParams(method="faithful").freeze()
    with pytest.raises(MissingRenderParam, match="scene.peak"):
        render(engine.tone_mapper, engine.loader, engine.processor, linear, settings, params=empty)


def test_the_peak_follows_the_brightest_moment_of_a_shot():
    from hdr_blaster.video.temporal import KEY_POLICY

    assert KEY_POLICY["scene.peak"] == "upper"

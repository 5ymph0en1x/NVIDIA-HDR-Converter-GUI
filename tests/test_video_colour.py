"""Video frame <-> engine working space."""
import numpy as np
import pytest
import torch

import video_fixtures as ref
from conftest import SAMPLES
from hdr_blaster.video.colour import (ENGINE_UNIT_NITS, FrameFormat, pq_eotf, srgb_to_yuv420,
                                      yuv_to_engine)


def to_engine(planes, device, **fmt):
    y, u, v = (torch.from_numpy(p).to(device) for p in planes)
    return yuv_to_engine(y, u, v, FrameFormat(width=y.shape[1], height=y.shape[0], **fmt))


def test_pq_eotf_matches_the_standard(device):
    code = torch.linspace(0.0, 1.0, 1024, dtype=torch.float32, device=device)
    expected = ref.pq_decode(code.cpu().numpy())
    actual = pq_eotf(code).cpu().numpy()
    assert np.allclose(actual, expected, rtol=2e-3, atol=1e-4)
    assert actual[0] == 0.0 and actual[-1] == pytest.approx(10000.0, rel=1e-4)


@pytest.mark.parametrize("full_range", [False, True], ids=["limited", "full"])
def test_grey_at_80_nits_is_one_tenth_in_the_engine(device, full_range):
    """scRGB 1.0 is 80 nits and the engine works in scRGB x 0.1."""
    out = to_engine(ref.flat_hdr10([80.0, 80.0, 80.0], full_range=full_range), device, full_range=full_range)
    assert out.shape == (1, 3, 8, 8) and out.dtype == torch.float32
    assert torch.allclose(out, torch.full_like(out, 80.0 / ENGINE_UNIT_NITS), rtol=0.02)


@pytest.mark.parametrize("nits", [0.5, 10.0, 203.0, 1000.0, 4000.0])
def test_limited_and_full_range_decode_to_the_same_light(device, nits):
    limited = to_engine(ref.flat_hdr10([nits] * 3, full_range=False), device, full_range=False)
    full = to_engine(ref.flat_hdr10([nits] * 3, full_range=True), device, full_range=True)
    expected = nits / ENGINE_UNIT_NITS
    assert torch.allclose(limited, torch.full_like(limited, expected), rtol=0.03)
    assert torch.allclose(full, torch.full_like(full, expected), rtol=0.03)


def test_pure_bt2020_red_is_clipped_to_the_bt709_gamut(device):
    out = to_engine(ref.flat_hdr10([200.0, 0.0, 0.0]), device)[0, :, 0, 0].tolist()
    assert out[0] == pytest.approx(200.0 * 1.660491 / ENGINE_UNIT_NITS, rel=0.03)
    assert out[1] == 0.0 and out[2] == 0.0


def test_black_is_exactly_zero(device):
    out = to_engine(ref.flat_hdr10([0.0, 0.0, 0.0]), device)
    assert float(out.abs().max()) == 0.0


def test_exposure_and_gamma_act_as_in_the_jxr_loader(device):
    """Loader order: scRGB ** (1/gamma), then x exposure, then x 0.1."""
    y, u, v = (torch.from_numpy(p).to(device) for p in ref.flat_hdr10([160.0] * 3))
    fmt = FrameFormat(width=8, height=8)
    plain = yuv_to_engine(y, u, v, fmt)
    assert torch.allclose(yuv_to_engine(y, u, v, fmt, exposure=2.0), plain * 2.0, rtol=1e-5)
    scrgb = plain / 0.1
    assert torch.allclose(yuv_to_engine(y, u, v, fmt, pre_gamma=2.0), scrgb ** 0.5 * 0.1, rtol=1e-4)


def test_chroma_is_sited_left(device):
    """A vertical colour edge must stay where it is, not shift by a fraction of a pixel."""
    h, w = 16, 32
    rgb = np.zeros((h, w, 3))
    rgb[:, : w // 2] = [150.0, 30.0, 30.0]
    rgb[:, w // 2:] = [30.0, 30.0, 150.0]
    out = to_engine(ref.nits709_to_hdr10(rgb), device)[0].cpu().numpy() * ENGINE_UNIT_NITS
    expected = np.moveaxis(rgb, -1, 0)
    flat = np.ones(w, dtype=bool)
    flat[w // 2 - 3: w // 2 + 3] = False                # the edge itself is blurred by 4:2:0
    assert np.allclose(out[:, :, flat], expected[:, :, flat], rtol=0.05, atol=1.0)
    left_error = np.abs(out[:, :, w // 2 - 3: w // 2] - expected[:, :, w // 2 - 3: w // 2]).mean()
    right_error = np.abs(out[:, :, w // 2: w // 2 + 3] - expected[:, :, w // 2: w // 2 + 3]).mean()
    assert left_error == pytest.approx(right_error, rel=0.35), "the edge blur should be symmetric"


# ------------------------------------------------------------------ SDR output

def decode_bt709_limited(y, u, v):
    """Reference decode of the encoder's output (nearest chroma), float64."""
    yp = (y.astype(np.float64) - 16.0) / 219.0
    cb = (np.repeat(np.repeat(u, 2, 0), 2, 1).astype(np.float64) - 128.0) / 224.0
    cr = (np.repeat(np.repeat(v, 2, 0), 2, 1).astype(np.float64) - 128.0) / 224.0
    r = yp + 1.5748 * cr
    b = yp + 1.8556 * cb
    g = (yp - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.stack([r, g, b])


def test_white_and_black_land_on_the_limited_range_ends(device):
    white = srgb_to_yuv420(torch.ones(1, 3, 4, 4, device=device))
    black = srgb_to_yuv420(torch.zeros(1, 3, 4, 4, device=device))
    assert [int(p.unique().item()) for p in white] == [235, 128, 128]
    assert [int(p.unique().item()) for p in black] == [16, 128, 128]
    assert all(p.dtype == torch.uint8 for p in white)
    assert white[0].shape == (4, 4) and white[1].shape == (2, 2)


@pytest.mark.parametrize("colour", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.2, 0.5, 0.8)])
def test_flat_colours_survive_the_round_trip(device, colour):
    srgb = torch.tensor(colour, device=device).view(1, 3, 1, 1).expand(1, 3, 8, 8)
    y, u, v = (p.cpu().numpy() for p in srgb_to_yuv420(srgb))
    back = decode_bt709_limited(y, u, v)
    assert np.allclose(back, np.array(colour).reshape(3, 1, 1), atol=1.5 / 255)


def test_odd_dimensions_are_rejected(device):
    with pytest.raises(ValueError, match="even"):
        srgb_to_yuv420(torch.zeros(1, 3, 5, 8, device=device))


# ---------------------------------------------------------- parity with photos

def _photo_and_video(engine, linear, device, which):
    from hdr_blaster.engine.pipeline import RenderSettings, render

    nits = linear[0].permute(1, 2, 0).double().cpu().numpy() * ENGINE_UNIT_NITS
    from_video = to_engine(ref.nits709_to_hdr10(nits), device)
    settings = RenderSettings(method=which)
    photo, photo_params = render(engine.tone_mapper, engine.loader, engine.processor, linear, settings)
    video, video_params = render(engine.tone_mapper, engine.loader, engine.processor, from_video, settings)
    mse = float(((photo - video) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(mse, 1e-12)), photo_params, video_params


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
@pytest.mark.parametrize("which", ["punchy", "perceptual"])
def test_a_capture_sent_through_hdr10_renders_like_the_capture(engine, get_linear, device, which):
    """The promise of the video path: same working space, hence same rendering as photos.

    Measured on a slightly blurred capture: HDR10 video is 4:2:0, so the pixel-sharp
    colour edges of a screenshot cannot survive ANY video encoding. That loss is
    checked separately below; here we check everything else.
    """
    linear = get_linear(SAMPLES[0])
    smooth = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(linear, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    psnr, _, _ = _photo_and_video(engine, smooth, device, which)
    # Punchy adds contrast above the shadows, and its roll-off follows a peak that 4:2:0
    # chroma moves a little (see the next test): measured 42.9 dB.
    assert psnr >= (41.5 if which == "punchy" else 45.0), f"PSNR {psnr:.1f} dB"


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
@pytest.mark.parametrize("which, key", [("faithful", "scene.peak"), ("perceptual", "scene.peak")])
def test_the_engine_measures_the_same_scene_through_hdr10(engine, get_linear, device, which, key):
    """On the sharp capture, what the engine measures (the peak of the scene) must agree:
    a scale or gamut error in the video path would show here first. The peak is taken on
    the brightest CHANNEL, which 4:2:0 chroma shifts on the pixel-sharp coloured edges of
    a screenshot (measured: 13 % on this capture); a scale error would be far larger."""
    psnr, photo_params, video_params = _photo_and_video(engine, get_linear(SAMPLES[0]), device, which)
    assert float(video_params.stats[key]) == pytest.approx(float(photo_params.stats[key]), rel=0.20)
    # Measured 30.7 and 31.3 dB, and 30.8 and 31.6 dB when the video frame is rendered with
    # the photo's own peak: the loss is the chroma of the sharp edges, not the measurement.
    assert psnr >= 29.5, f"PSNR {psnr:.1f} dB (4:2:0 chroma loss on sharp edges is expected, not more)"

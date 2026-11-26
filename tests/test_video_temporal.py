"""Temporal stabilisation of recorded render parameters."""
import pytest
import torch

from hdr_blaster.engine.params import RenderParams
from hdr_blaster.video import temporal
from hdr_blaster.video.temporal import (KEY_POLICY, detect_cuts, documented_keys, frame_signature,
                                        stabilise)

# The smoothing rules are tested with stand-in parameters, one per rule that no rendering
# uses any more (every rendering keeps the scene's exposure: no key, no log-average).
STAND_INS = {"perceptual.key": "log_level", "mantiuk.max_lum": "upper", "adaptive.lut_max": "upper"}


@pytest.fixture(autouse=True)
def stand_in_rules(monkeypatch):
    for key, rule in STAND_INS.items():
        monkeypatch.setitem(temporal.KEY_POLICY, key, rule)


def recorded(key, values, method="perceptual"):
    return [RenderParams(method=method, stats={key: torch.tensor(float(v))}) for v in values]


def regular(n, fps=60.0):
    return [i / fps for i in range(n)]


def series(params, key):
    return torch.stack([p.stats[key].double() for p in params])


# ------------------------------------------------------------------- the rules

def test_every_documented_parameter_has_a_rule():
    assert documented_keys(), "could not read the keys from engine/params.py"
    assert documented_keys() == set(KEY_POLICY) - set(STAND_INS)
    assert KEY_POLICY["scene.peak"] == "upper"


def test_results_are_frozen_and_inputs_untouched():
    torch.manual_seed(0)
    raw = recorded("perceptual.key", 0.07 + 0.01 * torch.rand(30))
    before = [p.stats["perceptual.key"].clone() for p in raw]
    out = stabilise(raw, regular(30), [0])
    assert all(p.frozen and p.method == "perceptual" for p in out)
    assert all(not p.frozen for p in raw)
    assert all(torch.equal(p.stats["perceptual.key"], b) for p, b in zip(raw, before))


def test_frame_to_frame_noise_is_removed():
    torch.manual_seed(0)
    noisy = 0.07 * torch.exp(0.05 * torch.randn(240))
    out = series(stabilise(recorded("perceptual.key", noisy), regular(240), [0]), "perceptual.key")
    jitter_before = (noisy[1:] - noisy[:-1]).abs().mean()
    jitter_after = (out[1:] - out[:-1]).abs().mean()
    assert float(jitter_after) < float(jitter_before) / 5.0
    assert float(out.mean()) == pytest.approx(0.07, rel=0.03)


def test_a_slow_change_is_followed():
    """A walk from a dark room into daylight over four seconds must still adapt."""
    ramp = torch.logspace(-1.5, -0.5, 240)
    out = series(stabilise(recorded("perceptual.key", ramp), regular(240), [0]), "perceptual.key")
    middle = slice(70, 170)
    assert torch.allclose(out[middle], ramp[middle].double(), rtol=0.05)


def test_a_cut_is_not_smoothed_across():
    values = [0.05] * 60 + [0.20] * 60
    out = series(stabilise(recorded("perceptual.key", values), regular(120), [0, 60]), "perceptual.key")
    assert torch.allclose(out[:60], torch.full((60,), 0.05, dtype=torch.float64), rtol=1e-4)
    assert torch.allclose(out[60:], torch.full((60,), 0.20, dtype=torch.float64), rtol=1e-4)
    blurred = series(stabilise(recorded("perceptual.key", values), regular(120), [0]), "perceptual.key")
    assert 0.05 < float(blurred[59]) < 0.20, "without the cut the step would be smoothed"


@pytest.mark.parametrize("key, upper", [("edge.in_max", True), ("mantiuk.max_lum", True),
                                        ("edge.out_min", False), ("edge.in_min", False)])
def test_bounds_stay_on_the_safe_side_of_every_frame(key, upper):
    torch.manual_seed(1)
    own = 0.5 + 0.4 * torch.rand(180)
    out = series(stabilise(recorded(key, own, method="mantiuk06"), regular(180), [0]), key)
    if upper:
        assert bool((out >= own.double() - 1e-9).all()), "an upper bound fell below a frame's own maximum"
    else:
        assert bool((out <= own.double() + 1e-9).all()), "a lower bound rose above a frame's own minimum"
    assert float((out[1:] - out[:-1]).abs().mean()) < float((own[1:] - own[:-1]).abs().mean()) / 5.0


def test_time_is_measured_in_seconds_not_frames():
    """Same content, twice the frame rate: the result at a given time is the same."""
    spike = lambda n: [0.1 if n // 2 - n // 60 <= i < n // 2 + n // 60 else 0.05 for i in range(n)]
    slow = series(stabilise(recorded("perceptual.key", spike(120)), regular(120, 60.0), [0]), "perceptual.key")
    fast = series(stabilise(recorded("perceptual.key", spike(240)), regular(240, 120.0), [0]), "perceptual.key")
    assert float(slow[60]) == pytest.approx(float(fast[120]), rel=0.05)


def test_irregular_timestamps_are_accepted():
    torch.manual_seed(2)
    steps = torch.tensor([0.009, 0.033, 0.017, 0.075] * 30)
    times = torch.cumsum(steps, 0).tolist()
    noisy = 0.07 * torch.exp(0.05 * torch.randn(120))
    out = series(stabilise(recorded("perceptual.key", noisy, method="perceptual"), times, [0]), "perceptual.key")
    assert torch.isfinite(out).all() and float(out.std()) < float(noisy.std()) / 2.0


def test_branch_choices_are_voted_per_shot():
    flags = [True] * 14 + [False] * 6 + [False] * 20
    raw = [RenderParams(method="adaptive", stats={"adaptive.extreme_highlights": f}) for f in flags]
    out = stabilise(raw, regular(40), [0, 20])
    assert [p.stats["adaptive.extreme_highlights"] for p in out] == [True] * 20 + [False] * 20


def test_a_key_recorded_by_only_some_frames_is_given_to_all():
    """With a voted branch, frames that took the other branch never measured its values."""
    raw = []
    for i in range(30):
        stats = {"adaptive.extreme_highlights": i % 3 != 0}
        if i % 3 != 0:
            stats["perceptual.key"] = torch.tensor(0.2)
        else:
            stats["adaptive.lut_max"] = torch.tensor(1.5)
        raw.append(RenderParams(method="adaptive", stats=stats))
    out = stabilise(raw, regular(30), [0])
    assert all(float(p.stats["perceptual.key"]) == pytest.approx(0.2) for p in out)
    assert all(float(p.stats["adaptive.lut_max"]) == pytest.approx(1.5) for p in out)


def test_multi_dimensional_parameters_keep_their_shape_dtype_and_device(device):
    torch.manual_seed(3)
    raw = [RenderParams(method="neural", stats={
        "neural.grid": torch.randn(1, 12, 2, 4, 4, device=device),
        "lut3d.weights": torch.randn(1, 3, device=device).half(),
    }) for _ in range(12)]
    out = stabilise(raw, regular(12), [0])
    grid, weights = out[5].stats["neural.grid"], out[5].stats["lut3d.weights"]
    assert grid.shape == (1, 12, 2, 4, 4) and grid.dtype == torch.float32 and grid.device.type == device.type
    assert weights.shape == (1, 3) and weights.dtype == torch.float16
    expected = torch.stack([p.stats["neural.grid"] for p in raw]).mean(0)
    assert torch.allclose(grid, expected, atol=1e-5)


def test_a_single_frame_video():
    out = stabilise(recorded("perceptual.key", [0.07]), [0.0], [0])
    assert float(out[0].stats["perceptual.key"]) == pytest.approx(0.07)


def test_misuse_is_reported():
    two = recorded("perceptual.key", [0.1, 0.1])
    with pytest.raises(ValueError, match="one time per frame"):
        stabilise(two, [0.0], [0])
    with pytest.raises(ValueError, match="one operator"):
        stabilise([two[0], RenderParams(method="hable")], [0.0, 0.1], [0])
    with pytest.raises(KeyError, match="mystery.key"):
        stabilise([RenderParams(method="agx", stats={"mystery.key": torch.tensor(1.0)})], [0.0], [0])
    with pytest.raises(ValueError, match="outside"):
        stabilise(two, [0.0, 0.1], [0, 5])
    assert stabilise([], [], []) == []


# ------------------------------------------------------------------------ cuts

def signatures(levels, noise=0.002, seed=0):
    torch.manual_seed(seed)
    base = torch.linspace(0.2, 0.6, 144)
    return torch.stack([base * level + noise * torch.randn(144) for level in levels])


def test_a_hard_cut_is_found():
    assert detect_cuts(signatures([1.0] * 30 + [0.3] * 30)) == [0, 30]


def test_steady_footage_has_one_shot():
    assert detect_cuts(signatures([1.0] * 120)) == [0]


def test_a_fade_is_not_a_cut():
    fade = torch.linspace(1.0, 0.3, 90).tolist()
    assert detect_cuts(signatures([1.0] * 15 + fade + [0.3] * 15)) == [0]


def test_two_cuts_close_together():
    assert detect_cuts(signatures([1.0] * 20 + [0.3] * 5 + [1.4] * 20)) == [0, 20, 25]


def test_a_flash_is_not_a_cut():
    """One very bright frame, then back to the same picture."""
    assert detect_cuts(signatures([1.0] * 20 + [2.5] + [1.0] * 20)) == [0]


def test_a_strobe_is_not_a_series_of_cuts():
    assert detect_cuts(signatures([1.0, 2.5] * 20)) == [0]


def test_a_cut_right_after_a_flash_is_still_found():
    assert detect_cuts(signatures([1.0] * 20 + [2.5] + [1.0] * 6 + [0.3] * 20)) == [0, 27]


def test_empty_and_single_frame():
    assert detect_cuts(torch.empty(0, 144)) == []
    assert detect_cuts(signatures([1.0])) == [0]


def test_signature_is_a_small_cpu_thumbnail_in_code_value(device):
    y = torch.full((144, 256), 512, dtype=torch.int16, device=device)
    signature = frame_signature(y, bit_depth=10)
    assert signature.shape == (144,) and signature.device.type == "cpu"
    assert torch.allclose(signature, torch.full((144,), 512 / 1023))


# ------------------------------------------------ found by the independent review

def test_a_key_that_appears_mid_shot_arrives_without_a_jump():
    """With a voted branch, the frames at the start of a shot may never have measured a
    value. They used to get the shot-wide mean while their neighbours got the local one:
    a one-frame exposure step of 2.6x."""
    values = torch.logspace(-1.7, -1.0, 180)                   # recorded from frame 120 on only
    raw = []
    for i in range(300):
        stats = {"adaptive.extreme_highlights": True}
        if i >= 120:
            stats["perceptual.key"] = values[i - 120].clone()
        raw.append(RenderParams(method="adaptive", stats=stats))
    out = series(stabilise(raw, regular(300), [0]), "perceptual.key")
    step = (out[1:] / out[:-1]).log().abs().max()
    assert float(step) < 0.05, f"largest frame-to-frame step is a factor of {float(step.exp()):.2f}"
    assert float(out[0]) == pytest.approx(float(out[60]), rel=1e-6)   # filled from the nearest window


def test_large_values_are_smoothed_like_small_ones():
    """The neural grid goes through a running sum instead of a stacked array: same result."""
    torch.manual_seed(4)
    steps = torch.tensor([0.009, 0.033, 0.017, 0.075] * 15)
    times = torch.cumsum(steps, 0).tolist()
    grids = [torch.randn(1, 12, 2, 4, 4).half() for _ in range(60)]
    grids[7] = None                                              # a frame that did not record it
    raw = [RenderParams(method="neural", stats={} if g is None else {"neural.grid": g}) for g in grids]
    out = stabilise(raw, times, [0], window_s=0.2)
    for i in (0, 7, 31, 59):
        near = [g.double() for g, t in zip(grids, times) if g is not None and abs(t - times[i]) <= 0.2]
        expected = torch.stack(near).mean(0)
        got = out[i].stats["neural.grid"]
        assert got.dtype == torch.float16 and got.shape == (1, 12, 2, 4, 4)
        assert torch.allclose(got.double(), expected, atol=2e-3)

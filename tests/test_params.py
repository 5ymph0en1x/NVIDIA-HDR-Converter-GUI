"""Analyse once, render many times: recorded parameters can be replayed."""
import pytest
import torch

from conftest import SAMPLES, assert_identical
from hdr_blaster.engine.params import MissingRenderParam, RenderParams, measure
from hdr_blaster.engine.pipeline import RenderSettings, analyze, render


def settings_for(method, edge=25.0, color=50.0):
    return RenderSettings(method=method, color_strength=color, edge_strength=edge)


def run(engine, linear, method, params=None, **kw):
    return render(engine.tone_mapper, engine.loader, engine.processor, linear,
                  settings_for(method, **kw), params=params)


def run_analyze(engine, linear, method, **kw):
    return analyze(engine.tone_mapper, engine.loader, engine.processor, linear,
                   settings_for(method, **kw))


# ------------------------------------------------------------------ unit level

def test_recording_computes_and_stores():
    params = RenderParams(method="natural")
    assert params.measure("k", lambda: 3) == 3
    assert params.stats == {"k": 3}


def test_replay_returns_the_stored_value_without_computing():
    params = RenderParams(method="natural", stats={"k": 3}, frozen=True)

    def must_not_run():
        raise AssertionError("measured during replay")

    assert params.measure("k", must_not_run) == 3


def test_replay_of_an_unrecorded_key_names_the_key():
    params = RenderParams(method="natural", frozen=True)
    with pytest.raises(MissingRenderParam, match="edge.in_min"):
        params.measure("edge.in_min", lambda: 0)


def test_freeze_gives_an_independent_frozen_copy():
    params = RenderParams(method="natural")
    params.measure("k", lambda: 1)
    frozen = params.freeze()
    params.measure("other", lambda: 2)
    assert frozen.frozen and frozen.stats == {"k": 1}
    assert not params.frozen


def test_without_params_measure_just_computes():
    assert measure(None, "k", lambda: 7) == 7


# ------------------------------------------------------------- on real captures

def test_replaying_on_the_same_image_changes_nothing(engine, linear, method):
    recorded, params = run(engine, linear, method)
    replayed, used = run(engine, linear, method, params=params.freeze())
    assert used.method == params.method
    assert_identical(replayed, recorded, "replayed")


def test_analyze_returns_frozen_params_that_render_identically(engine, linear, method):
    recorded, _ = run(engine, linear, method)
    params = run_analyze(engine, linear, method)
    assert params.frozen
    replayed, _ = run(engine, linear, method, params=params)
    assert_identical(replayed, recorded, "rendered from analyze()")


def _forbid_whole_image_measurements(monkeypatch, engine):
    """Make any whole-image reduction, or a run of either predictor network, fail loudly."""
    for name in ("median", "min", "max", "mean", "std"):
        original = getattr(torch.Tensor, name)

        def guarded(self, *args, _original=original, _name=name, **kwargs):
            if not args and not kwargs and self.numel() > 64:
                raise AssertionError(f"whole-image {_name}() taken during replay")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, guarded)

    def no_classifier(*a, **k):
        raise AssertionError("LUT classifier run during replay")

    monkeypatch.setattr(engine.processor.color_net.classifier, "forward", no_classifier)

    def no_grid_predictor(*a, **k):
        raise AssertionError("neural grid predicted during replay")

    monkeypatch.setattr(engine.tone_mapper.neural_model.coeffs, "forward", no_grid_predictor)


@pytest.mark.skipif(len(SAMPLES) < 2, reason="needs two captures")
def test_replay_on_another_image_measures_nothing(engine, get_linear, method, monkeypatch):
    """The video case: parameters analysed on frame A, applied to frame B."""
    image_a, image_b = get_linear(SAMPLES[0]), get_linear(SAMPLES[-1])
    params = run_analyze(engine, image_a, method)
    before = dict(params.stats)

    with monkeypatch.context() as patch:
        _forbid_whole_image_measurements(patch, engine)
        out, _ = run(engine, image_b, method, params=params)

    assert out.shape == image_b.shape
    assert torch.isfinite(out).all()
    assert params.stats.keys() == before.keys()
    assert all(params.stats[k] is before[k] for k in before)


@pytest.mark.skipif(len(SAMPLES) < 2, reason="needs two captures")
def test_replayed_exposure_is_the_analysed_one_not_the_images_own(engine, get_linear):
    image_a, image_b = get_linear(SAMPLES[0]), get_linear(SAMPLES[-1])
    image_b = image_b * 0.25                      # a much dimmer peak than the analysed picture's
    own, own_params = run(engine, image_b, "perceptual")
    params_a = run_analyze(engine, image_a, "perceptual")
    assert not torch.equal(params_a.stats["scene.peak"], own_params.stats["scene.peak"])
    replayed, _ = run(engine, image_b, "perceptual", params=params_a)
    assert not torch.equal(replayed, own)


def test_recorded_keys_per_method(engine, linear):
    """The documented keys are the ones actually recorded: a rendering measures the scene's
    peak and nothing else; the colour models add their weights, sharpening its scale."""
    common = {"scene.peak", "lut3d.weights", "edge.mag_max"}
    for method in ("faithful", "natural", "punchy", "golden", "perceptual", "reinhard", "hable", "aces",
                   "drago03", "mantiuk06"):
        params = run_analyze(engine, linear, method)
        assert set(params.stats) == common, method
    params = run_analyze(engine, linear, "adaptive")
    assert set(params.stats) == common | {"adaptive.extreme_highlights", "adaptive.shadow_recovery"}


def test_adaptive_records_its_branch_choices(engine, linear):
    params = run_analyze(engine, linear, "adaptive")
    assert isinstance(params.stats["adaptive.extreme_highlights"], bool)
    assert isinstance(params.stats["adaptive.shadow_recovery"], bool)


def test_neural_records_its_grid(engine, linear):
    params = run_analyze(engine, linear, "neural", edge=0.0, color=0.0)
    model = engine.tone_mapper.neural_model
    assert set(params.stats) == {"neural.grid"}
    assert params.stats["neural.grid"].shape == (
        1, 12, model.coeffs.grid_d, model.coeffs.grid_h, model.coeffs.grid_w)


def test_rendering_with_more_stages_than_analysed_fails_loudly(engine, linear):
    """Analysed without sharpening, rendered with it: an error, not a silently unsharpened image."""
    params = run_analyze(engine, linear, "hable", edge=0.0, color=0.0)
    with pytest.raises(MissingRenderParam, match="edge.mag_max"):
        run(engine, linear, "hable", params=params, edge=25.0, color=0.0)


def test_replay_errors_surface_instead_of_returning_the_unenhanced_image(engine, linear):
    """While recording, a failed enhancement falls back to the input (original behaviour).
    While replaying, that would silently ship a different-looking frame: it must raise."""
    params = run_analyze(engine, linear, "hable")
    broken = RenderParams(method=params.method, frozen=True,
                          stats={**params.stats, "lut3d.weights": params.stats["lut3d.weights"][:, :2]})
    with pytest.raises(Exception):
        run(engine, linear, "hable", params=broken)


def test_render_refuses_params_that_are_not_frozen(engine, linear):
    """Passing back the params of a previous render would re-measure and overwrite them."""
    _, recorded = run(engine, linear, "perceptual")
    before = recorded.stats["scene.peak"]
    with pytest.raises(ValueError, match="freeze"):
        run(engine, linear, "perceptual", params=recorded)
    assert recorded.stats["scene.peak"] is before

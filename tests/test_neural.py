"""The learned tone mapper, split into 'predict the grid' and 'apply the grid'.

``tests/legacy/model_legacy.py`` is the frozen original of ``training/model.py``.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from conftest import (LEGACY_NEURAL_CHECKPOINT, SHIPPED_NEURAL_CHECKPOINT, assert_identical,
                      use_neural_checkpoint)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "training" / "checkpoints" / "latest.pth"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def models(device):
    """(original model, current model), both loaded from the shipped checkpoint."""
    original = _load("model_legacy", ROOT / "tests" / "legacy" / "model_legacy.py")
    current = _load("model_current", ROOT / "training" / "model.py")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    args = ckpt.get("args", {}) or {}
    kwargs = dict(grid_d=args.get("grid_d", 8), grid_h=args.get("grid_hw", 16), grid_w=args.get("grid_hw", 16))
    pair = []
    for module in (original, current):
        model = module.NeuralToneMapper(**kwargs).to(device).eval()
        model.load_state_dict(ckpt["model"])
        pair.append(model)
    return pair


def test_checkpoint_layout_is_unchanged(models):
    original, current = models
    assert list(current.state_dict().keys()) == list(original.state_dict().keys())


def test_forward_matches_the_original_model(models, linear):
    original, current = models
    with torch.inference_mode():
        assert_identical(current(linear), original(linear), "forward")


def test_predict_then_apply_equals_forward(models, linear):
    _, current = models
    with torch.inference_mode():
        grid = current.predict_grid(linear)
        assert grid.shape == (1, 12, current.coeffs.grid_d, current.coeffs.grid_h, current.coeffs.grid_w)
        assert_identical(current.apply_grid(linear, grid), current(linear), "apply_grid(predict_grid)")


def test_apply_grid_does_not_run_the_predictor(models, linear, monkeypatch):
    _, current = models
    with torch.inference_mode():
        grid = current.predict_grid(linear)

        def no_predictor(*a, **k):
            raise AssertionError("coefficient predictor run while applying a recorded grid")

        monkeypatch.setattr(current.coeffs, "forward", no_predictor)
        out = current.apply_grid(linear, grid)
    assert out.shape == linear.shape


def test_training_step_still_works():
    """forward + backward in training mode, as train.py uses the model."""
    module = _load("model_current_train", ROOT / "training" / "model.py")
    torch.manual_seed(0)
    model = module.NeuralToneMapper().train()
    x = torch.rand(2, 3, 96, 96) * 10.0
    y = model(x)
    assert y.shape == x.shape and float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    (y - 0.5).abs().mean().backward()
    assert all(p.grad is not None for p in model.guide.parameters())


# ------------------------------------------------------ the checkpoint we ship

@pytest.fixture
def shipped_engine(device, legacy):
    """An engine using the checkpoint the application ships, instead of the oracle's."""
    from types import SimpleNamespace

    from hdr_blaster.engine.processor import HDRColorProcessor
    from hdr_blaster.engine.tonemap import AdvancedToneMapper
    from hdr_blaster.io.jxr import OptimizedJXRLoader

    use_neural_checkpoint(SHIPPED_NEURAL_CHECKPOINT)
    try:
        loader = OptimizedJXRLoader(device)
        yield SimpleNamespace(loader=loader, tone_mapper=AdvancedToneMapper(device, loader.metadata),
                              processor=HDRColorProcessor(device, loader, use_fp16=False))
    finally:
        use_neural_checkpoint(LEGACY_NEURAL_CHECKPOINT)


def test_shipped_checkpoint_loads_safely_and_states_its_unit():
    ckpt = torch.load(SHIPPED_NEURAL_CHECKPOINT, map_location="cpu", weights_only=True)
    assert ckpt["input_unit"] == "nits"
    assert ckpt["teacher"] == "punchy" and ckpt["anchor_white_nits"] == 203.0
    assert "model" in ckpt and "val_loss" in ckpt


def test_application_feeds_the_shipped_model_in_nits(shipped_engine):
    model = shipped_engine.tone_mapper.neural_model
    assert model.engine_input_scale == 800.0 and model.anchor_white_nits == 203.0
    assert shipped_engine.tone_mapper.keeps_scene_levels("neural")


@pytest.mark.parametrize("white", [203.0, 480.0])
def test_shipped_model_renders_the_look_it_was_taught_on_any_screen(shipped_engine, sample, white):
    """The current teacher anchors shadows and highlights, while the grid renders mid-tones."""
    from hdr_blaster.engine.pipeline import RenderSettings, render

    engine = shipped_engine
    linear, error, _ = engine.loader.load_jxr(str(sample))
    assert linear is not None, error
    learned, params = render(engine.tone_mapper, engine.loader, engine.processor, linear,
                             RenderSettings(method="neural", white_nits=white))
    taught, _ = render(engine.tone_mapper, engine.loader, engine.processor, linear,
                       RenderSettings(method="punchy", white_nits=white))
    assert set(params.stats) == {"neural.grid", "scene.peak"}
    assert torch.isfinite(learned).all() and float(learned.min()) >= 0.0 and float(learned.max()) <= 1.0
    psnr = -10.0 * torch.log10(((learned - taught) ** 2).mean())
    assert float(psnr) > 26.0, f"{float(psnr):.1f} dB"

    def luma(x):
        return 0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]

    assert float(luma(learned).mean()) == pytest.approx(float(luma(taught).mean()), abs=0.03), "same exposure"
    blacks = [float(luma(x).flatten()[::17].quantile(0.01)) for x in (learned, taught)]
    assert blacks[0] == pytest.approx(blacks[1], abs=0.02), "no veil over the blacks, no crushing either"
    # the rule every rendering obeys: the dark zones are the scene's own
    from hdr_blaster.engine.colour import _srgb_decode

    scene = engine.tone_mapper._relative(linear, white).amax(1) * white
    anchors = ((scene >= white) | (scene <= 0.045 * white)).unsqueeze(1).expand_as(learned)
    assert torch.allclose(learned[anchors], taught[anchors], atol=1e-6, rtol=0)
    shown = _srgb_decode(learned).amax(1) * white
    band = (scene >= 0.5) & (scene < 0.03 * white)
    if int(band.sum()) > 2000:
        ratio = float((shown[band] / scene[band]).median())
        assert 0.85 < ratio < 1.20, f"dark zones shown x{ratio:.2f}"

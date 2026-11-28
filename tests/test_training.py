"""The training code of the neural tone mapper (training/).

One convention everywhere: the model takes absolute luminance in nits, BT.709
primaries. Every loader converts to it; the application converts its own
working space to it for checkpoints trained that way.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SAMPLES

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"


def _load(name):
    """Import a training/ module the way the scripts do (training/ on sys.path)."""
    if str(TRAINING) not in sys.path:
        sys.path.insert(0, str(TRAINING))
    spec = importlib.util.spec_from_file_location(f"training_{name}", TRAINING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"training_{name}"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dataset():
    return _load("dataset")


@pytest.fixture(scope="module")
def losses():
    return _load("losses")


def pq_encode(nits):
    """Reference BT.2100 PQ inverse EOTF, nits -> code value in [0, 1]."""
    m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
    y = (np.asarray(nits, dtype=np.float64) / 10000.0) ** m1
    return ((c1 + c2 * y) / (1.0 + c3 * y)) ** m2


def write_png16(path, rgb01):
    import imagecodecs
    arr = np.clip(np.rint(np.asarray(rgb01) * 65535.0), 0, 65535).astype(np.uint16)
    path.write_bytes(imagecodecs.png_encode(arr))


def write_png8(path, rgb01):
    from PIL import Image
    Image.fromarray(np.clip(np.rint(np.asarray(rgb01) * 255.0), 0, 255).astype(np.uint8)).save(path)


def hdrtv_pair(tmp_path, hdr_rgb_nits, size=64):
    (tmp_path / "HDR").mkdir()
    (tmp_path / "SDR").mkdir()
    hdr = np.broadcast_to(pq_encode(hdr_rgb_nits), (size, size, 3))
    write_png16(tmp_path / "HDR" / "a.png", hdr)
    write_png8(tmp_path / "SDR" / "a.png", np.full((size, size, 3), 0.5))
    return tmp_path


# ------------------------------------------------------------------- loaders

def test_hdrtv4k_reads_16_bits_and_decodes_pq_to_nits(dataset, tmp_path):
    """Pillow opens a 16-bit RGB PNG as 8 bits: the PQ decode used to be skipped silently."""
    root = hdrtv_pair(tmp_path, [100.0, 100.0, 100.0])
    item = dataset.HDRTV4KDataset(str(root), patch_size=32)[0]
    assert item["hdr"].shape == (3, 32, 32)
    assert torch.allclose(item["hdr"], torch.full((3, 32, 32), 100.0), rtol=0.01)


def test_hdrtv4k_converts_bt2020_primaries_to_bt709(dataset, tmp_path):
    """A neutral grey stays neutral; a pure BT.2020 red lies outside BT.709."""
    root = hdrtv_pair(tmp_path, [200.0, 0.0, 0.0])
    r, g, b = dataset.HDRTV4KDataset(str(root), patch_size=32)[0]["hdr"][:, 0, 0].tolist()
    assert r == pytest.approx(200.0 * 1.660491, rel=0.01)
    assert (g, b) == pytest.approx((-24.910, -3.6302), rel=0.01)


def test_synthetic_hdr_is_in_nits(dataset, tmp_path):
    write_png8(tmp_path / "grey.png", np.full((64, 64, 3), 0.5))
    ds = dataset.SyntheticHDRDataset(str(tmp_path), patch_size=32, ev_range=(0.0, 0.0))
    item = ds[0]
    linear = ((128 / 255 + 0.055) / 1.055) ** 2.4
    assert torch.allclose(item["hdr"], torch.full((3, 32, 32), linear * dataset.SDR_WHITE_NITS), rtol=1e-3)
    assert torch.allclose(item["sdr"], torch.full((3, 32, 32), 128 / 255))


def test_synthetic_highlights_scale_with_exposure(dataset):
    """A clipped white is extrapolated to the same multiple of white at every exposure."""
    import random
    white = torch.ones(3, 4, 4)
    random.seed(0)
    dark = dataset.fake_hdr_from_srgb(white, ev_range=(-2.0, -2.0))
    bright = dataset.fake_hdr_from_srgb(white, ev_range=(1.0, 1.0))
    assert torch.allclose(bright / dark, torch.full_like(dark, 8.0), rtol=1e-4)
    peak = dataset.SDR_WHITE_NITS * dataset.HIGHLIGHT_HEADROOM
    assert torch.allclose(dataset.fake_hdr_from_srgb(white, ev_range=(0.0, 0.0)),
                          torch.full((3, 4, 4), peak), rtol=1e-3)


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_jxr_loader_agrees_with_the_application(dataset, engine):
    """Same capture, training loader vs application loader: the same nits."""
    nits = dataset.load_jxr_nits(SAMPLES[0])
    app, error, _ = engine.loader.load_jxr(str(SAMPLES[0]))
    assert app is not None, error
    from hdr_blaster.engine.neural import ENGINE_UNIT_NITS
    expected = (app[0] * ENGINE_UNIT_NITS).permute(1, 2, 0).cpu().numpy()
    assert nits.shape == expected.shape
    assert np.allclose(nits, expected, rtol=1e-4, atol=1e-3)


# -------------------------------------------------------------------- losses

def srgb_encode(x):
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def test_chroma_loss_is_zero_for_a_faithful_render(losses):
    """sRGB-encoding the scene exactly keeps its hue: the loss must not ask for more saturation."""
    torch.manual_seed(0)
    hdr = torch.rand(2, 3, 16, 16) * 0.9 + 0.05
    faithful = srgb_encode(hdr)
    assert float(losses.ChromaLoss()(faithful, hdr)) < 1e-5
    oversaturated = srgb_encode((hdr - hdr.mean(1, keepdim=True)) * 1.6 + hdr.mean(1, keepdim=True)).clamp(0, 1)
    assert float(losses.ChromaLoss()(oversaturated, hdr)) > 10 * float(losses.ChromaLoss()(faithful, hdr)) + 1e-4


def test_range_loss_sees_out_of_range_predictions(losses):
    inside = torch.full((1, 3, 8, 8), 0.5)
    assert float(losses.RangeLoss()(inside + 0.9)) > float(losses.RangeLoss()(inside))
    assert float(losses.RangeLoss()(inside - 0.9)) > float(losses.RangeLoss()(inside))


def test_range_loss_does_not_reward_values_outside_the_displayable_range(losses):
    """The 'use the whole range' reward must stop at [0, 1]. On un-clamped training
    output an unbounded reward pays the network for exploding (seen in a real run:
    the range term fell to -0.86 while the pixel error doubled)."""
    ramp = torch.linspace(0.0, 1.0, 4096).view(1, 1, 64, 64).expand(1, 3, 64, 64).clone()
    spiked = ramp.clone()
    spiked[0, 0, 10, 10] = 20.0            # one runaway pixel: tiny mean penalty, huge max - min
    assert float(losses.RangeLoss()(spiked)) > float(losses.RangeLoss()(ramp))
    assert float(losses.RangeLoss()(ramp)) == pytest.approx(-0.1, abs=1e-6)     # the best possible value


def test_losses_stay_finite_with_out_of_range_predictions(losses):
    torch.manual_seed(0)
    pred = torch.rand(2, 3, 16, 16) * 3.0 - 1.0          # un-clamped network output
    target = torch.rand(2, 3, 16, 16)
    hdr = torch.rand(2, 3, 16, 16) * 400.0
    total, parts = losses.CompositeLoss()(pred, target, hdr)
    assert torch.isfinite(total) and all(np.isfinite(v) for v in parts.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed precision is a GPU option")
def test_losses_are_computed_in_float32_under_mixed_precision(losses):
    torch.manual_seed(0)
    pred = (torch.rand(2, 3, 32, 32, device="cuda") * 1.2).half().requires_grad_()
    target = torch.rand(2, 3, 32, 32, device="cuda")
    hdr = torch.rand(2, 3, 32, 32, device="cuda") * 4000.0
    with torch.amp.autocast("cuda"):
        total, parts = losses.CompositeLoss()(pred, target, hdr)
    assert total.dtype == torch.float32
    assert torch.isfinite(total) and all(np.isfinite(v) for v in parts.values())
    total.backward()
    assert torch.isfinite(pred.grad).all()


# --------------------------------------------------------------------- model

def test_training_mode_keeps_gradients_for_out_of_range_pixels():
    """Clamping in training zeroes the gradient of saturated pixels: they could never recover."""
    model_module = _load("model")
    torch.manual_seed(0)
    model = model_module.NeuralToneMapper()
    with torch.no_grad():
        model.coeffs.fusion.bias.mul_(40.0)             # force a saturated output
    x = torch.full((1, 3, 64, 64), 500.0)
    out = model.train()(x)
    assert float(out.detach().max()) > 1.0, "training output must not be clamped"
    model.zero_grad()
    out.sum().backward()
    assert float(model.coeffs.fusion.bias.grad.abs().sum()) > 0.0
    with torch.no_grad():
        assert float(model.eval()(x).max()) <= 1.0, "inference output stays in [0, 1]"


# ----------------------------------------------------- application integration

def test_engine_scale_for_checkpoints():
    from hdr_blaster.engine.neural import ENGINE_UNIT_NITS, engine_input_scale
    assert ENGINE_UNIT_NITS == 800.0                     # the JXR loader yields scRGB x 0.1; scRGB 1.0 = 80 nits
    assert engine_input_scale({"model": {}}) == 1.0     # shipped checkpoint: fed as before, unchanged rendering
    assert engine_input_scale({"input_unit": "nits"}) == 800.0
    with pytest.raises(ValueError, match="furlongs"):
        engine_input_scale({"input_unit": "furlongs"})


# ------------------------------------------------------------------ train.py

def run_train(cwd, *args):
    return subprocess.run([sys.executable, str(TRAINING / "train.py"), *args, "--device", "cpu",
                           "--num-workers", "0", "--batch-size", "4", "--patch-size", "32"],
                          cwd=cwd, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A one-epoch run on sixteen tiny images."""
    work = tmp_path_factory.mktemp("train")
    images = work / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for i in range(16):
        write_png8(images / f"{i:02d}.png", rng.random((48, 48, 3)))
    done = run_train(work, "--synthetic-dir", str(images), "--out-dir", str(work / "run1"), "--epochs", "1")
    assert done.returncode == 0, done.stdout + done.stderr
    return work, images


def test_training_writes_a_checkpoint_that_states_its_input_unit(trained):
    work, _ = trained
    ckpt = torch.load(work / "run1" / "latest.pth", map_location="cpu", weights_only=True)
    assert ckpt["input_unit"] == "nits"
    assert ckpt["epoch"] == 0
    assert (work / "run1" / "best.pth").exists()
    assert np.isfinite(ckpt["val_loss"])


def test_fine_tuning_from_a_checkpoint_actually_trains(trained):
    """README phase 3 (`--resume latest.pth --epochs 10` after a 20-epoch run) used to run zero iterations."""
    work, images = trained
    done = run_train(work, "--synthetic-dir", str(images), "--out-dir", str(work / "run2"),
                     "--init-from", str(work / "run1" / "latest.pth"), "--epochs", "1", "--lr", "5e-5")
    assert done.returncode == 0, done.stdout + done.stderr
    before = torch.load(work / "run1" / "latest.pth", map_location="cpu", weights_only=True)
    after = torch.load(work / "run2" / "latest.pth", map_location="cpu", weights_only=True)
    assert after["epoch"] == 0
    assert any(not torch.equal(before["model"][k], after["model"][k]) for k in before["model"])


def test_resuming_a_finished_run_says_so(trained):
    work, images = trained
    done = run_train(work, "--synthetic-dir", str(images), "--out-dir", str(work / "run3"),
                     "--resume", str(work / "run1" / "latest.pth"), "--epochs", "1")
    assert done.returncode != 0
    assert "--init-from" in (done.stdout + done.stderr)


def test_a_checkpoint_taught_by_an_anchored_look_states_its_white():
    from hdr_blaster.engine.neural import anchor_white_nits

    assert anchor_white_nits({"input_unit": "nits"}) is None, "earlier checkpoints expose by content"
    assert anchor_white_nits({"input_unit": "nits", "anchor_white_nits": 203.0}) == 203.0
    with pytest.raises(ValueError, match="implausible"):
        anchor_white_nits({"anchor_white_nits": 0.0})


def test_the_application_shows_an_anchored_model_the_scene_at_its_own_white(device):
    """A model taught at a white of 203 nits, on a screen whose white is 406 nits, must be
    given the scene at half the luminance: this screen's white is then rendered like the
    teacher's white was."""
    from types import SimpleNamespace

    from hdr_blaster.engine.tonemap import AdvancedToneMapper

    seen = []

    def predict_grid(x):
        seen.append(float(x.mean()))
        return torch.zeros(1, 12, 2, 2, 2, device=device)

    mapper = AdvancedToneMapper(device)
    image = torch.full((1, 3, 8, 8), 100.0 / 800.0, device=device)            # 100 nits
    for anchor, white in ((203.0, 203.0), (203.0, 406.0), (None, 406.0)):
        mapper.neural_model = SimpleNamespace(engine_input_scale=800.0, anchor_white_nits=anchor,
                                              predict_grid=predict_grid, apply_grid=lambda x, grid: x * 0.0)
        mapper.apply_tone_mapping(image, method="neural", white_nits=white)
    assert seen == pytest.approx([100.0, 50.0, 100.0])
    assert mapper.keeps_scene_levels("neural") is False
    mapper.neural_model.anchor_white_nits = 203.0
    assert mapper.keeps_scene_levels("neural") is True

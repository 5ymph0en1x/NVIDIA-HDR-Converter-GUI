"""Shared fixtures for the differential tests.

The safety net of the refactor is a frozen copy of the original monolith
(``tests/legacy/nhc_legacy.py``). Every test here compares the new engine to
that oracle, bit for bit, on the JXR captures found in ``samples/``.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLES = sorted((ROOT / "samples").glob("*.jxr"))
SAMPLE_IDS = [p.stem[-22:].replace(" ", "_") for p in SAMPLES]

# Every name the original accepts, plus None for the Auto-detect oracle.
METHODS = [None, "agx", "agx_punchy", "agx_golden", "perceptual", "adaptive", "hable",
           "reinhard", "filmic", "aces", "uncharted2", "mantiuk06", "drago03", "neural"]
METHOD_IDS = [m or "auto" for m in METHODS]

# Tone mapping left the oracle on purpose (2026-11-25, at the author's request): every look
# and operator now keeps the scene's dark tones and mid-grey instead of re-exposing the
# picture (tests/test_renderings.py specifies them). What must still match the original
# bit for bit: reading a capture, the colour models, sharpening as the original did it,
# and writing the files (tests/test_differential.py).
ANCHORED_LOOKS = ("agx", "agx_punchy", "agx_golden")
ORACLE_METHODS = [m for m in METHODS if m not in ANCHORED_LOOKS]
ORACLE_METHOD_IDS = [m or "auto" for m in ORACLE_METHODS]

CROP = 768

# The neural weights the application shipped with when the oracle was frozen. The
# oracle always uses them, and so does the new engine during the tests: the
# differential tests compare CODE, and must not move when the shipped weights do.
LEGACY_NEURAL_CHECKPOINT = ROOT / "tests" / "legacy" / "neural_legacy.pth"
SHIPPED_NEURAL_CHECKPOINT = ROOT / "training" / "checkpoints" / "latest.pth"


def use_neural_checkpoint(path):
    """Point the new engine at a checkpoint (and forget any model already loaded)."""
    from hdr_blaster.engine import neural

    neural.NEURAL_TONEMAP_CHECKPOINT_PATH = str(path)
    neural._NEURAL_MODEL_CACHE.clear()


def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", default=False,
                     help="compare at full resolution instead of a 768x768 centre crop")


def pytest_collection_modifyitems(config, items):
    if SAMPLES:
        return
    skip = pytest.mark.skip(reason="no .jxr capture in samples/ : the safety net cannot run")
    for item in items:
        if "sample" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)


def pytest_generate_tests(metafunc):
    if "sample" in metafunc.fixturenames:
        metafunc.parametrize("sample", SAMPLES or [None], ids=SAMPLE_IDS or ["no-sample"])
    if "method" in metafunc.fixturenames:
        metafunc.parametrize("method", METHODS, ids=METHOD_IDS)
    if "oracle_method" in metafunc.fixturenames:
        metafunc.parametrize("oracle_method", ORACLE_METHODS, ids=ORACLE_METHOD_IDS)


@pytest.fixture(scope="session")
def legacy():
    """The frozen original application, imported as a module."""
    spec = importlib.util.spec_from_file_location(
        "nhc_legacy", ROOT / "tests" / "legacy" / "nhc_legacy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nhc_legacy"] = module
    spec.loader.exec_module(module)

    # The oracle imports training/model.py, which is live code. Point it at the
    # frozen original instead, so that a change to the model cannot move both
    # sides of a comparison at once.
    frozen = importlib.util.spec_from_file_location(
        "model_legacy", ROOT / "tests" / "legacy" / "model_legacy.py")
    model_legacy = importlib.util.module_from_spec(frozen)
    sys.modules["model_legacy"] = model_legacy
    frozen.loader.exec_module(model_legacy)
    assert module._NEURAL_TONEMAP_AVAILABLE, "the oracle could not import the neural model"
    module._NeuralToneMapperImpl = model_legacy.NeuralToneMapper
    module.NEURAL_TONEMAP_CHECKPOINT_PATH = str(LEGACY_NEURAL_CHECKPOINT)
    module._NEURAL_MODEL_CACHE.clear()
    use_neural_checkpoint(LEGACY_NEURAL_CHECKPOINT)
    return module


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def legacy_engine(legacy, device):
    """Oracle objects, built once: loader, tone mapper, colour processor."""
    loader = legacy.OptimizedJXRLoader(device)
    tone_mapper = legacy.AdvancedToneMapper(device, loader.metadata)
    processor = legacy.HDRColorProcessor(device, loader, use_fp16=False)
    return SimpleNamespace(loader=loader, tone_mapper=tone_mapper, processor=processor)


@pytest.fixture(scope="session")
def linear_cache():
    return {}


@pytest.fixture
def get_linear(engine, linear_cache, device, request):
    """Function: capture path -> scene-linear tensor, including the original signed scRGB channels."""
    full = request.config.getoption("--full")

    def load(sample):
        key = (sample, full)
        if key not in linear_cache:
            tensor, error, _ = engine.loader.load_jxr(str(sample))
            assert tensor is not None, error
            if not full:
                _, _, h, w = tensor.shape
                top, left = (h - CROP) // 2, (w - CROP) // 2
                tensor = tensor[:, :, top:top + CROP, left:left + CROP].contiguous()
            linear_cache[key] = tensor.cpu()
        return linear_cache[key].to(device)

    return load


@pytest.fixture
def linear(sample, get_linear):
    return get_linear(sample)


def make_engines(legacy, device, use_fp16=False):
    """A matched pair (oracle, new engine) on a given device and precision."""
    from hdr_blaster.engine.processor import HDRColorProcessor
    from hdr_blaster.engine.tonemap import AdvancedToneMapper
    from hdr_blaster.io.jxr import OptimizedJXRLoader

    pair = []
    for loader_cls, mapper_cls, processor_cls in (
            (legacy.OptimizedJXRLoader, legacy.AdvancedToneMapper, legacy.HDRColorProcessor),
            (OptimizedJXRLoader, AdvancedToneMapper, HDRColorProcessor)):
        loader = loader_cls(device)
        pair.append(SimpleNamespace(
            loader=loader,
            tone_mapper=mapper_cls(device, loader.metadata),
            processor=processor_cls(device, loader, use_fp16=use_fp16)))
    return pair


def legacy_chain(engine, linear, method, color_strength=0.0, edge_strength=0.0,
                 use_enhancement=True):
    """The original conversion sequence, exactly as ``_convert_single_file`` runs it."""
    tm = engine.tone_mapper
    linear = linear.clamp_min(0)  # The frozen loader supplied nonnegative coordinates.
    if method is None:
        method = tm.select_optimal_tonemap(tm.analyze_image_statistics(linear))
    tonemapped = tm.apply_tone_mapping(linear.clone(), method=method)
    srgb = engine.loader.linear_to_srgb(tonemapped)
    enhanced = engine.processor.process_image(
        srgb.clone(), color_strength=color_strength, edge_strength=edge_strength,
        use_enhancement=use_enhancement)
    return SimpleNamespace(method=method, tonemapped=tonemapped, srgb=srgb, enhanced=enhanced)


@pytest.fixture(scope="session")
def engine(device, legacy):
    """New-engine objects, built once. Depends on ``legacy`` only so that the
    oracle is imported first, as it would be in the original process."""
    from hdr_blaster.engine.processor import HDRColorProcessor
    from hdr_blaster.engine.tonemap import AdvancedToneMapper
    from hdr_blaster.io.jxr import OptimizedJXRLoader

    loader = OptimizedJXRLoader(device)
    tone_mapper = AdvancedToneMapper(device, loader.metadata)
    processor = HDRColorProcessor(device, loader, use_fp16=False)
    return SimpleNamespace(loader=loader, tone_mapper=tone_mapper, processor=processor)


def new_chain(engine, linear, method, color_strength=0.0, edge_strength=0.0,
              use_enhancement=True):
    """The same sequence, run by the new engine through its single pipeline."""
    from hdr_blaster.engine.pipeline import RenderSettings, render, resolve_method, tonemap_srgb

    used = resolve_method(engine.tone_mapper, linear, method)
    srgb = tonemap_srgb(engine.tone_mapper, engine.loader, linear, used)
    settings = RenderSettings(method=used, color_strength=color_strength,
                              edge_strength=edge_strength, use_enhancement=use_enhancement)
    enhanced, params = render(engine.tone_mapper, engine.loader, engine.processor, linear, settings)
    return SimpleNamespace(method=used, tonemapped=None, srgb=srgb, enhanced=enhanced, params=params)


def assert_identical(actual: torch.Tensor, expected: torch.Tensor, what: str):
    """Bit-exact comparison with a readable report when it fails."""
    assert actual.shape == expected.shape, f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert actual.dtype == expected.dtype, f"{what}: dtype {actual.dtype} != {expected.dtype}"
    if torch.equal(actual, expected):
        return
    diff = (actual.double() - expected.double()).abs()
    changed = int((diff > 0).sum())
    raise AssertionError(
        f"{what}: {changed} of {diff.numel()} values differ, max abs diff {diff.max().item():.3e}")

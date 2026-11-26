"""What the new engine still owes the original application, bit for bit.

Tone mapping is no longer compared: every look and operator was rebuilt to keep the
scene's dark tones and mid-grey (see tests/test_renderings.py and conftest.py). Everything
around it is unchanged and must stay so: reading a capture, the author's colour models
(the image-adaptive 3D LUTs behind the "AI colour" slider), the original sharpening, and
the files written.
"""
import pytest
import torch

from conftest import SAMPLES, assert_identical, legacy_chain, make_engines


def test_loader_matches_legacy(sample, legacy_engine, engine):
    expected, error, expected_meta = legacy_engine.loader.load_jxr(str(sample))
    assert expected is not None, error
    actual, error, actual_meta = engine.loader.load_jxr(str(sample))
    assert actual is not None, error
    assert actual_meta == expected_meta
    assert_identical(actual, expected, "loaded linear tensor")


@pytest.fixture
def picture(legacy_engine, linear):
    """A tone-mapped sRGB picture to feed both processors: the original's own render."""
    return legacy_chain(legacy_engine, linear, "hable").srgb


@pytest.mark.parametrize("colour, edge", [(50.0, 0.0), (100.0, 0.0), (0.0, 25.0), (50.0, 25.0)],
                         ids=["colour50", "colour100", "sharp25", "colour50+sharp25"])
def test_colour_models_and_original_sharpening_match_legacy(legacy_engine, engine, picture, colour, edge):
    """The author's colour models give the very same picture as in the original application,
    and so does the original sharpening (kept for a model taught before the renderings
    were anchored)."""
    expected = legacy_engine.processor.process_image(picture.clone(), color_strength=colour, edge_strength=edge,
                                                     use_enhancement=True)
    actual = engine.processor.process_image(picture.clone(), color_strength=colour, edge_strength=edge,
                                            use_enhancement=True)
    assert_identical(actual, expected, "enhanced")
    assert not torch.equal(expected, picture), "the oracle did not enhance"


def test_saved_jpeg_matches_legacy(legacy_engine, engine, picture, tmp_path):
    """Dithering draws random numbers: seed both sides the same way, compare the files."""
    old_file, new_file = tmp_path / "old.jpg", tmp_path / "new.jpg"
    torch.manual_seed(42)
    legacy_engine.processor.save_tensor_as_image(picture, str(old_file), "JPEG")
    torch.manual_seed(42)
    engine.processor.save_tensor_as_image(picture, str(new_file), "JPEG")
    assert new_file.read_bytes() == old_file.read_bytes()


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="half precision is a GPU option")
def test_half_precision_colour_models_match_legacy(legacy, get_linear):
    old, new = make_engines(legacy, torch.device("cuda"), use_fp16=True)
    picture = legacy_chain(old, get_linear(SAMPLES[0]), "hable").srgb
    expected = old.processor.process_image(picture.clone(), color_strength=50.0, edge_strength=25.0)
    actual = new.processor.process_image(picture.clone(), color_strength=50.0, edge_strength=25.0)
    assert_identical(actual, expected, "enhanced (fp16)")


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_cpu_colour_models_match_legacy(legacy, get_linear):
    old, new = make_engines(legacy, torch.device("cpu"))
    picture = legacy_chain(old, get_linear(SAMPLES[0]).cpu(), "hable").srgb
    expected = old.processor.process_image(picture.clone(), color_strength=50.0, edge_strength=25.0)
    actual = new.processor.process_image(picture.clone(), color_strength=50.0, edge_strength=25.0)
    assert_identical(actual, expected, "enhanced (cpu)")

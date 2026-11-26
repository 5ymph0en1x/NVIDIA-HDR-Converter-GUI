"""The oracle must be deterministic, or comparing against it proves nothing."""
from conftest import SAMPLES, assert_identical, legacy_chain


def test_the_safety_net_has_captures():
    """Without captures every comparison is skipped and the suite proves nothing."""
    assert SAMPLES, "put a few .jxr captures in samples/ : the differential tests need them"


def test_oracle_is_deterministic(legacy_engine, linear, method):
    first = legacy_chain(legacy_engine, linear, method, color_strength=50.0, edge_strength=25.0)
    second = legacy_chain(legacy_engine, linear, method, color_strength=50.0, edge_strength=25.0)
    assert first.method == second.method
    assert_identical(second.tonemapped, first.tonemapped, "tone-mapped")
    assert_identical(second.enhanced, first.enhanced, "enhanced")


def test_oracle_has_the_neural_model(legacy_engine):
    """A missing checkpoint would silently turn 'neural' into 'perceptual'."""
    model = legacy_engine.tone_mapper.neural_model
    assert model is not None
    assert type(model).__module__ == "model_legacy", "the oracle must use the frozen model"


def test_oracle_has_the_colour_lut(legacy_engine):
    """Missing weights would silently skip the AI colour stage."""
    assert legacy_engine.processor.color_net.enabled

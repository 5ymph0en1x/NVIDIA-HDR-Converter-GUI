"""The single conversion sequence and its file export."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import SAMPLES, assert_identical, legacy_chain

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fmt, include_raw, expected_suffixes", [
    ("JPEG", True, [".jpg"]),
    ("TIFF", True, ["_raw.tiff", "_tonemapped.tiff"]),
    ("Both", True, [".jpg", "_raw.tiff", "_tonemapped.tiff"]),
    ("TIFF", False, ["_tonemapped.tiff"]),
    ("Both", False, [".jpg", "_tonemapped.tiff"]),
])
def test_export_writes_the_expected_files(engine, tmp_path, fmt, include_raw, expected_suffixes):
    from hdr_blaster.engine.pipeline import export_image

    linear = torch.rand(1, 3, 32, 48) * 4.0
    enhanced = torch.rand(1, 3, 32, 48)
    base = str(tmp_path / "shot")
    written = export_image(engine.processor, linear, enhanced, base, fmt, include_raw)
    assert written == [base + s for s in expected_suffixes]
    assert all(Path(p).stat().st_size > 0 for p in written)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(Path(p).name for p in written)


def test_export_rejects_an_unknown_format(engine, tmp_path):
    from hdr_blaster.engine.pipeline import export_image

    image = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="PNG"):
        export_image(engine.processor, image, image, str(tmp_path / "x"), "PNG", True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_cli_finds_the_neural_model_in_a_fresh_process(tmp_path):
    """Nothing pre-imported: the application must locate training/model.py and the shipped
    checkpoint by itself (the weights differ from the oracle's, so no byte comparison here)."""
    base = tmp_path / "neural"
    done = subprocess.run(
        [sys.executable, str(ROOT / "NHC.py"), "--selftest", str(SAMPLES[0]), "--out", str(base),
         "--method", "neural"], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert "Loaded neural tone mapper" in output, output
    assert "falling back" not in output
    assert base.with_suffix(".jpg").stat().st_size > 100_000

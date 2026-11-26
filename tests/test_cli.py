"""The commands without the window: what a fresh clone answers to ``python NHC.py ...``."""
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from conftest import SAMPLES

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    return subprocess.run([sys.executable, str(ROOT / "NHC.py"), *map(str, args)], cwd=cwd or ROOT,
                          capture_output=True, text=True, timeout=600)


def test_the_version_is_3_0_0_everywhere():
    import hdr_blaster

    assert hdr_blaster.__version__ == "3.0.0"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == hdr_blaster.__version__
    done = run("--version")
    assert done.returncode == 0 and done.stdout.strip().endswith("3.0.0"), done.stdout + done.stderr
    assert "## What's new in 3.0.0" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert f'version = "{hdr_blaster.__version__}"' in (ROOT / "uv.lock").read_text(encoding="utf-8")


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_the_image_command_writes_next_to_the_source_and_never_over_a_file(tmp_path):
    """It used to write <name>.jpg, over whatever was there. Like the window, it writes
    <name>_SDR.jpg, numbered when that exists, with the default rendering (Faithful)."""
    capture = tmp_path / "shot.jxr"
    shutil.copy(SAMPLES[0], capture)
    mine = tmp_path / "shot.jpg"
    mine.write_bytes(b"a picture of mine")
    first = run("--convert", capture, cwd=tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "method=faithful" in first.stdout
    assert mine.read_bytes() == b"a picture of mine", "an existing file is never touched"
    assert (tmp_path / "shot_SDR.jpg").stat().st_size > 100_000
    second = run("--convert", capture, "--method", "auto", cwd=tmp_path)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "method=auto-detect" in second.stdout
    assert sorted(p.name for p in tmp_path.glob("*.jpg")) == ["shot.jpg", "shot_SDR-2.jpg", "shot_SDR.jpg"]


def test_the_video_command_renders_faithful_unless_told_otherwise(tmp_path):
    import video_clips as clips

    clip = tmp_path / "clip.mp4"
    clips.make_clip(clip, frames=8, audio_streams=0)
    done = run("--convert-video", clip, cwd=tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "method=faithful" in done.stdout and (tmp_path / "clip_SDR.mp4").exists()
    done = run("--convert-video", clip, "--method", "auto", cwd=tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "method=faithful" not in done.stdout and (tmp_path / "clip_SDR-2.mp4").exists()

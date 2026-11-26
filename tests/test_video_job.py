"""The whole conversion of a video, on small generated clips."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pytest
import torch

import video_clips as clips
from hdr_blaster.engine.constants import SUPPORTED_TONE_MAPS
from hdr_blaster.engine.pipeline import RenderSettings
from hdr_blaster.video import job
from hdr_blaster.video.job import (Cancelled, OutputTooLarge, VideoSettings, bit_rate_budget,
                                   convert_video, default_output_path, make_engine, output_size,
                                   corrected_bit_rate, size_target_bit_rate)
from hdr_blaster.video.reader import UnsupportedVideo, probe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def video_engine(device):
    return make_engine(device)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("job") / "game.mp4"
    clips.make_clip(path, frames=40)
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(method="punchy", **kwargs):
    return VideoSettings(render=RenderSettings(method=method, edge_strength=25.0), **kwargs)


# ------------------------------------------------------------- the happy path

def test_conversion_keeps_frames_timestamps_and_audio_and_is_not_heavier(clip, device, video_engine, tmp_path):
    before = digest(clip)
    dst = tmp_path / "out.mp4"
    stages = []
    report = convert_video(clip, dst, settings(), device, engine=video_engine,
                           progress=lambda stage, done, total: stages.append((stage, done, total)))

    source, out = clips.describe(clip), clips.describe(dst)
    assert report.frames == 40 and len(out["pts_s"]) == 40
    assert out["pts_s"] == pytest.approx(source["pts_s"], abs=1e-4)
    assert out["audio_bytes"] == source["audio_bytes"]
    assert (out["codec"], out["pix_fmt"], out["trc"], out["range"]) == ("h264", "yuv420p", 1, 1)
    assert report.dst_bytes == os.path.getsize(dst) <= report.src_bytes == os.path.getsize(clip)
    assert report.method == "punchy" and report.encoder in ("h264_nvenc", "libx264")
    assert report.cuts == [0] and (report.width, report.height) == (256, 144)
    assert digest(clip) == before, "the source must never be touched"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4"]
    assert [s for s, _, _ in stages if s == "analyse"] == ["analyse"] * 40
    assert stages[-1] == ("render", 40, 40)
    assert 40.0 < np.mean(out["mean_luma"]) < 200.0, "the picture should be a normally exposed image"


def test_auto_detect_uses_one_operator_for_the_whole_video(clip, device, video_engine, tmp_path):
    stages = set()
    report = convert_video(clip, tmp_path / "auto.mp4", settings(method=None), device, engine=video_engine,
                           progress=lambda stage, done, total: stages.add(stage))
    assert report.method in SUPPORTED_TONE_MAPS
    assert stages == {"method", "analyse", "render"}


def test_every_frame_is_measured_once_then_replayed_frozen(clip, device, video_engine, tmp_path, monkeypatch):
    calls = []
    real = job.render

    def spy(tone_mapper, loader, processor, linear, render_settings, params=None):
        calls.append("replay" if params is not None and params.frozen else "record")
        return real(tone_mapper, loader, processor, linear, render_settings, params=params)

    monkeypatch.setattr(job, "render", spy)
    convert_video(clip, tmp_path / "out.mp4", settings(method="perceptual"), device, engine=video_engine)
    # one recording pass, then replays only (a second encode may replay them again)
    assert calls[:40] == ["record"] * 40
    assert set(calls[40:]) == {"replay"} and len(calls) % 40 == 0


@pytest.mark.parametrize("operator", ["mantiuk06", "adaptive", "neural", "hable"])
def test_other_operators_convert_too(clip, device, video_engine, tmp_path, operator):
    report = convert_video(clip, tmp_path / "out.mp4", settings(method=operator), device, engine=video_engine)
    assert report.frames == 40 and report.method == operator


def test_downscaling_gives_even_dimensions(clip, device, video_engine, tmp_path):
    report = convert_video(clip, tmp_path / "small.mp4", settings(out_height=90), device, engine=video_engine)
    out = clips.describe(tmp_path / "small.mp4")
    assert (out["width"], out["height"]) == (160, 90) == (report.width, report.height)
    assert len(out["pts_s"]) == 40


# ------------------------------------------------------------------ stability

def flashing_scene(width, height, t):
    """A static scene with a large light that is on every other frame (muzzle flash, strobe)."""
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([40.0 + 60.0 * xx / width, 50.0 + 0.0 * xx, 70.0 - 30.0 * xx / width], axis=-1)
    frame = int(round(t * 39))
    if frame % 2:
        light = (xx > width * 0.55) & (yy > height * 0.2) & (yy < height * 0.8)
        base = base + 3000.0 * light[..., None]
    return base


def corner_series(path):
    """Mean luma of a corner the flash never touches, for each frame."""
    with av.open(str(path)) as c:
        return np.array([f.to_ndarray()[8:56, 8:72].mean() for f in c.decode(video=0)])


def test_a_flashing_light_does_not_make_the_rest_of_the_picture_pump(device, video_engine, tmp_path):
    """Measured per frame, the operator re-exposes the whole image at every flash. Stabilised,
    the part of the scene that does not change must stay put."""
    src = tmp_path / "flash.mp4"
    clips.make_clip(src, frames=40, audio_streams=0, scene=flashing_scene)
    # Drago03 compresses everything above the shadows according to the scene's peak: the
    # operator on which a peak measured per frame shows most. (Perceptual served here until
    # it became the BT.2390 roll-off, which leaves what fits under white alone.)
    convert_video(src, tmp_path / "per_frame.mp4", settings(method="drago03", window_s=0.0),
                  device, engine=video_engine)
    convert_video(src, tmp_path / "stable.mp4", settings(method="drago03"), device, engine=video_engine)
    per_frame = np.abs(np.diff(corner_series(tmp_path / "per_frame.mp4"))).mean()
    stable = np.abs(np.diff(corner_series(tmp_path / "stable.mp4"))).mean()
    assert per_frame > 3.0, f"the test scene should pump when measured per frame (got {per_frame:.2f})"
    assert stable < per_frame / 5.0, f"pumping {stable:.2f} vs {per_frame:.2f} per frame"
    assert stable < 1.0


def test_a_cut_is_detected_and_a_dark_shot_stays_dark(device, video_engine, tmp_path):
    """A bright shot, then one twelve times darker. The cut is found, and no rendering
    re-exposes the dark shot: it stays as dark as the recording shows it (the classical
    operators used to bring it back to the brightness of the first)."""
    src = tmp_path / "cut.mp4"
    levels = [1.0] * 20 + [0.08] * 20
    clips.make_clip(src, frames=40, audio_streams=0, levels=levels,
                    scene=lambda w, h, t: clips.scene_nits(w, h, 0.5))
    for method in ("punchy", "drago03"):
        report = convert_video(src, tmp_path / f"{method}.mp4", settings(method=method), device, engine=video_engine)
        assert report.cuts == [0, 20] and report.method == method
        luma = np.array(clips.describe(tmp_path / f"{method}.mp4")["mean_luma"])
        assert luma[25:].mean() < 0.5 * luma[:15].mean(), f"{method} re-exposed the dark shot"


# ------------------------------------------------------- refusals and cancelling

def test_cancelling_leaves_nothing_behind(clip, device, video_engine, tmp_path):
    for stage_to_stop in ("analyse", "render"):
        seen = []

        def progress(stage, done, total):
            seen.append(stage)

        def cancelled():
            return seen.count(stage_to_stop) >= 5

        with pytest.raises(Cancelled):
            convert_video(clip, tmp_path / "out.mp4", settings(), device, engine=video_engine,
                          progress=progress, cancelled=cancelled)
        assert list(tmp_path.iterdir()) == [], f"files left after cancelling during {stage_to_stop}"


def test_an_sdr_video_is_refused_before_anything_is_written(device, video_engine, tmp_path):
    src = tmp_path / "sdr.mp4"
    clips.make_clip(src, frames=4, hdr=False)
    with pytest.raises(UnsupportedVideo, match="not an HDR10 video"):
        convert_video(src, tmp_path / "out.mp4", settings(), device, engine=video_engine)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sdr.mp4"]


def test_an_existing_output_is_never_overwritten(clip, device, video_engine, tmp_path):
    dst = tmp_path / "out.mp4"
    dst.write_text("precious")
    with pytest.raises(FileExistsError):
        convert_video(clip, dst, settings(), device, engine=video_engine)
    assert dst.read_text() == "precious"
    with pytest.raises(ValueError, match="source file itself"):
        convert_video(clip, clip, settings(), device, engine=video_engine)


def test_full_range_and_limited_range_sources_look_the_same(device, video_engine, tmp_path):
    lumas = []
    for full in (False, True):
        src = tmp_path / f"range_{full}.mp4"
        clips.make_clip(src, frames=6, audio_streams=0, full_range=full)
        convert_video(src, tmp_path / f"out_{full}.mp4", settings(), device, engine=video_engine)
        lumas.append(np.mean(clips.describe(tmp_path / f"out_{full}.mp4")["mean_luma"]))
    assert lumas[0] == pytest.approx(lumas[1], rel=0.02)


# -------------------------------------------------------------------- helpers

def test_default_output_name_never_collides(tmp_path):
    src = tmp_path / "Cyberpunk.mp4"
    assert default_output_path(src) == str(tmp_path / "Cyberpunk_SDR.mp4")
    (tmp_path / "Cyberpunk_SDR.mp4").write_text("x")
    assert default_output_path(src) == str(tmp_path / "Cyberpunk_SDR-2.mp4")
    (tmp_path / "Cyberpunk_SDR-2.mp4").write_text("x")
    assert default_output_path(src) == str(tmp_path / "Cyberpunk_SDR-3.mp4")


def test_output_size(clip):
    info = probe(clip)                                    # 256 x 144
    assert output_size(info, None) == (256, 144)
    assert output_size(info, 1080) == (256, 144), "never upscale"
    assert output_size(info, 90) == (160, 90)
    assert output_size(info, 81) == (142, 80), "odd sizes are rounded to even"


def test_bit_rate_budgets(clip):
    info = probe(clip)
    first = bit_rate_budget(info, "high")
    assert first == pytest.approx(info.bit_rate * 0.9, rel=0.01) or first == 200_000
    assert bit_rate_budget(info, "compact") <= bit_rate_budget(info, "balanced") <= first
    # after an oversized attempt: what fits in 90 % of the source once audio is paid for
    target = size_target_bit_rate(info, audio_bytes=0, retry=0)
    assert 0 < target * info.duration_s / 8 < info.size_bytes * 0.90
    assert size_target_bit_rate(info, 0, 1) == pytest.approx(target * 0.8, rel=0.01)
    assert size_target_bit_rate(info, info.size_bytes, 0) == 0, "audio alone already fills the budget"
    # a mean-rate attempt that produced twice the allowed video payload: at most half the rate next
    again = corrected_bit_rate(info, 0, used=1_000_000, produced=info.size_bytes * 2)
    assert 0 < again < 500_000
    assert corrected_bit_rate(info, info.size_bytes, 1_000_000, info.size_bytes * 2) == 0


def test_a_source_too_small_to_beat_is_reported_and_leaves_nothing(clip, device, video_engine,
                                                                  tmp_path, monkeypatch):
    """When no encode can be lighter than the source, say so and leave no file."""
    from dataclasses import replace

    real_probe = job.probe
    monkeypatch.setattr(job, "probe", lambda path: replace(real_probe(path), size_bytes=3000))
    with pytest.raises(OutputTooLarge, match="lighter than the source"):
        convert_video(clip, tmp_path / "out.mp4", settings(), device, engine=video_engine)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------------- CLI

def test_command_line_converts_a_video(clip, tmp_path):
    out = tmp_path / "cli.mp4"
    done = subprocess.run(
        [sys.executable, str(ROOT / "NHC.py"), "--convert-video", str(clip), "--out", str(out),
         "--method", "punchy", "--quality", "balanced"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "40 frames" in done.stdout
    assert len(clips.describe(out)["pts_s"]) == 40


def test_command_line_reports_an_unsupported_video(tmp_path):
    src = tmp_path / "sdr.mp4"
    clips.make_clip(src, frames=4, hdr=False)
    done = subprocess.run([sys.executable, str(ROOT / "NHC.py"), "--convert-video", str(src)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert done.returncode == 2
    assert "not an HDR10 video" in done.stdout + done.stderr
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".mp4") == ["sdr.mp4"]


# ------------------------------------------------ found by the independent review

def test_a_missing_output_folder_is_reported_before_any_work(clip, device, video_engine, tmp_path):
    seen = []
    with pytest.raises(FileNotFoundError, match="nowhere"):
        convert_video(clip, tmp_path / "nowhere" / "out.mp4", settings(), device, engine=video_engine,
                      progress=lambda *a: seen.append(a))
    assert seen == [], "the whole video was analysed before the problem was reported"
    assert list(tmp_path.iterdir()) == []


def test_recorded_parameters_do_not_stay_on_the_gpu(clip, device, video_engine, tmp_path, monkeypatch):
    """The neural grid is 98 kB per frame: kept on the GPU for a whole recording it would
    exhaust it. Between the passes everything lives in main memory, large values in half precision."""
    seen = {}
    real = job.stabilise

    def spy(params, times, cuts, window_s=1.0):
        seen["devices"] = {v.device.type for p in params for v in p.stats.values() if torch.is_tensor(v)}
        seen["grid"] = params[0].stats["neural.grid"].dtype
        return real(params, times, cuts, window_s=window_s)

    monkeypatch.setattr(job, "stabilise", spy)
    report = convert_video(clip, tmp_path / "out.mp4", settings(method="neural"), device, engine=video_engine)
    assert seen["devices"] == {"cpu"} and seen["grid"] == torch.float16
    assert report.frames == 40


def test_an_oversized_first_encode_is_redone_and_ends_under_the_source(clip, device, video_engine,
                                                                        tmp_path, monkeypatch, caplog):
    """The retry path, for real. This short clip is mostly audio (15 kB of 26): the
    quality-driven encode comes out heavier than the source, so a mean-bit-rate encode follows."""
    import logging

    writers, real_writer = [], job.VideoWriter

    def counting_writer(*args, **kwargs):
        writers.append(kwargs.get("average", False))
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(job, "VideoWriter", counting_writer)
    with caplog.at_level(logging.INFO):
        report = convert_video(clip, tmp_path / "out.mp4", settings(), device, engine=video_engine)
    assert "encoding again" in caplog.text, "this clip is expected to overshoot on the first encode"
    assert writers[:2] == [False, False], "preflight, then the quality-driven attempt"
    assert len(writers) >= 3 and all(writers[2:]), "then mean-bit-rate attempts"
    assert report.dst_bytes <= report.src_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4"]


def test_rotation_and_pixel_shape_travel_with_the_picture(device, video_engine, tmp_path):
    src = tmp_path / "phone.mp4"
    clips.make_clip(src, frames=6, audio_streams=0, rotation=90, sample_aspect_ratio=(4, 3))
    assert clips.describe(src)["rotation"] == 90, "the test clip itself must be rotated"
    convert_video(src, tmp_path / "out.mp4", settings(), device, engine=video_engine)
    out = clips.describe(tmp_path / "out.mp4")
    assert out["rotation"] == 90
    assert out["sample_aspect_ratio"] == (4, 3)


def test_command_line_refuses_a_mistyped_operator(clip, tmp_path):
    done = subprocess.run(
        [sys.executable, str(ROOT / "NHC.py"), "--convert-video", str(clip), "--out", str(tmp_path / "x.mp4"),
         "--method", "punchi"], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert done.returncode == 2
    assert "punchi" in done.stderr and "punchy" in done.stderr
    assert not (tmp_path / "x.mp4").exists()


# ---------------------------------------------------------------- faithful video

def test_a_faithful_video_shows_each_frame_at_the_scenes_luminance(clip, device, video_engine, tmp_path):
    """Analysis then replay with the Faithful rendering: the frames the encoder receives
    are the source view of the same frames, wherever the scene is below white."""
    from hdr_blaster import preview
    from hdr_blaster.video.job import analyse_video, analysis_key

    chosen = VideoSettings(render=RenderSettings(method="faithful", edge_strength=0.0, white_nits=400.0))
    other = VideoSettings(render=RenderSettings(method="faithful", edge_strength=0.0, white_nits=203.0))
    assert analysis_key(chosen) != analysis_key(other), "another white needs another analysis"
    analysis = analyse_video(clip, chosen, device, engine=video_engine)
    assert analysis.method == "faithful" and analysis.matches(chosen) and not analysis.matches(other)
    assert all("scene.peak" in p.stats for p in analysis.params)
    source = preview.load_video_frame(clip, analysis.times[10], device, max_side=4096, white_nits=400.0)
    after = preview.render_preview(video_engine, source, chosen.render,
                                   analysis.params[analysis.index_of(source.pts)])
    below = (source.linear * 800.0).amax(1) < 200.0
    assert int(below.sum()) > 1000
    assert float((after - source.before).abs().amax(1)[below].max()) < 1.0 / 255
    report = convert_video(clip, tmp_path / "faithful.mp4", chosen, device, engine=video_engine)
    assert report.method == "faithful" and report.frames == 40

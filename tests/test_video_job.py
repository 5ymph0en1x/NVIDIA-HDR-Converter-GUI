"""The whole conversion of a video, on small generated clips."""
import hashlib
import os
import subprocess
import sys
import threading
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
    assert (out["codec"], out["pix_fmt"], out["trc"], out["range"]) == ("h264", "yuv420p", 13, 1)
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

    def spy(tone_mapper, loader, processor, linear, render_settings, params=None, **kwargs):
        calls.append("replay" if params is not None and params.frozen else "record")
        return real(tone_mapper, loader, processor, linear, render_settings, params=params, **kwargs)

    from hdr_blaster.engine import pipeline
    monkeypatch.setattr(job, "render", spy)
    monkeypatch.setattr(pipeline, "render", spy)       # the analysis reaches it through measure()
    convert_video(clip, tmp_path / "out.mp4", settings(method="perceptual"), device, engine=video_engine)
    # one recording pass, then replays only (a second encode may replay them again)
    assert calls[:40] == ["record"] * 40
    assert set(calls[40:]) == {"replay"} and len(calls) % 40 == 0


@pytest.mark.parametrize("operator", ["mantiuk06", "adaptive", "neural", "hable"])
def test_other_operators_convert_too(clip, device, video_engine, tmp_path, operator):
    report = convert_video(clip, tmp_path / "out.mp4", settings(method=operator), device, engine=video_engine)
    assert report.frames == 40 and report.method == operator


def test_missing_neural_model_reports_perceptual_in_the_encoded_video(clip, device, video_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(video_engine.tone_mapper, "neural_model", None)
    report = convert_video(clip, tmp_path / "fallback.mp4", settings(method="neural"), device, engine=video_engine)
    assert report.method == "perceptual" and report.frames == 40


def test_downscaling_gives_even_dimensions(clip, device, video_engine, tmp_path):
    report = convert_video(clip, tmp_path / "small.mp4", settings(out_height=90), device, engine=video_engine)
    out = clips.describe(tmp_path / "small.mp4")
    assert (out["width"], out["height"]) == (160, 90) == (report.width, report.height)
    assert len(out["pts_s"]) == 40


# ------------------------------------------------------- measuring without rendering

ALL_METHODS = ["faithful", "natural", "punchy", "golden", "reinhard", "hable", "aces", "drago03",
               "perceptual", "mantiuk06", "adaptive", "neural", "filmic", "agx_punchy"]


@pytest.mark.parametrize("rendering", ALL_METHODS)
@pytest.mark.parametrize("edge, colour", [(0.0, 0.0), (25.0, 0.0), (0.0, 40.0)])
def test_measuring_alone_finds_exactly_what_a_full_render_measures(clip, device, video_engine, rendering, edge, colour):
    """The light analysis must not be an approximation: same keys, same values, bit for bit."""
    from hdr_blaster.engine.pipeline import measure, render
    from hdr_blaster.video.colour import yuv_to_engine
    from hdr_blaster.video.reader import read_frames

    chosen = RenderSettings(method=rendering, edge_strength=edge, color_strength=colour, white_nits=300.0)
    info = probe(clip)
    engine = video_engine
    with torch.inference_mode():
        for frame in read_frames(clip, device):
            if frame.index % 13:
                continue
            linear = yuv_to_engine(frame.y, frame.u, frame.v, info.fmt)
            full = render(engine.tone_mapper, engine.loader, engine.processor, linear, chosen)[1]
            light = measure(engine.tone_mapper, engine.loader, engine.processor, linear, chosen)
            assert light.method == full.method and not light.frozen
            assert sorted(light.stats) == sorted(full.stats)
            for key, value in full.stats.items():
                if torch.is_tensor(value):
                    assert torch.equal(light.stats[key], value), f"{key} differs for {rendering}"
                else:
                    assert light.stats[key] == value, f"{key} differs for {rendering}"


def test_without_sharpening_or_ai_colour_the_analysis_renders_nothing(clip, device, video_engine, monkeypatch):
    from hdr_blaster.engine import pipeline
    from hdr_blaster.video.job import analyse_video

    rendered = []
    real = pipeline.render
    monkeypatch.setattr(pipeline, "render", lambda *a, **k: rendered.append(1) or real(*a, **k))
    plain = VideoSettings(render=RenderSettings(method="punchy", edge_strength=0.0))
    analysis = analyse_video(clip, plain, device, engine=video_engine)
    assert rendered == [] and all(sorted(p.stats) == ["scene.peak"] for p in analysis.params)
    sharp = VideoSettings(render=RenderSettings(method="punchy", edge_strength=25.0))
    analyse_video(clip, sharp, device, engine=video_engine)
    assert len(rendered) == 40, "with sharpening the edge maximum is measured on the rendered picture"


def test_a_video_never_gives_the_gpu_memory_back_between_frames(clip, device, video_engine, tmp_path, monkeypatch):
    """Emptying the cache after every frame cost about 7 ms per frame and per pass."""
    flushed = []
    monkeypatch.setattr(video_engine.processor, "clear_gpu_memory", lambda: flushed.append(1))
    convert_video(clip, tmp_path / "out.mp4", settings(), device, engine=video_engine)
    assert flushed == []


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


# ------------------------------------------------ the fast way against the plain way

def decoded(path):
    """Every picture of a file as a player decodes it, with its timestamp."""
    with av.open(str(path)) as container:
        return [(frame.pts, frame.to_ndarray()) for frame in container.decode(video=0)]


@pytest.mark.parametrize("rendering, edge, colour", [
    ("faithful", 0.0, 0.0),        # the default for a video: analysed without rendering
    ("punchy", 25.0, 0.0),         # sharpening: every frame rendered when analysed
    ("golden", 0.0, 40.0),         # AI colour
    ("neural", 0.0, 0.0),          # the learned rendering measures on the picture
    (None, 0.0, 0.0),              # Auto-detect: one more pass
])
def test_the_fast_way_writes_the_pictures_of_the_plain_way(clip, device, video_engine, tmp_path,
                                                          rendering, edge, colour):
    """Decoding on the graphics card, the stages side by side, the analysis that does not
    render: none of it may change one pixel. ``reference=True`` is the plain way."""
    from hdr_blaster.video.job import analyse_video

    chosen = VideoSettings(render=RenderSettings(method=rendering, edge_strength=edge,
                                                 color_strength=colour, white_nits=300.0))
    plain = analyse_video(clip, chosen, device, engine=video_engine, reference=True)
    fast = analyse_video(clip, chosen, device, engine=video_engine)
    assert (fast.method, fast.cuts, fast.pts, fast.times) == (plain.method, plain.cuts, plain.pts, plain.times)
    for a, b in zip(fast.params, plain.params):
        assert sorted(a.stats) == sorted(b.stats)
        for key, value in b.stats.items():
            same = torch.equal(a.stats[key], value) if torch.is_tensor(value) else a.stats[key] == value
            assert same, f"{key} differs"

    convert_video(clip, tmp_path / "plain.mp4", chosen, device, engine=video_engine, reference=True)
    convert_video(clip, tmp_path / "fast.mp4", chosen, device, engine=video_engine)
    one, other = decoded(tmp_path / "plain.mp4"), decoded(tmp_path / "fast.mp4")
    assert len(one) == len(other) == 40
    for (pts_a, picture_a), (pts_b, picture_b) in zip(one, other):
        assert pts_a == pts_b and np.array_equal(picture_a, picture_b)
    assert clips.describe(tmp_path / "fast.mp4")["audio_bytes"] == clips.describe(tmp_path / "plain.mp4")["audio_bytes"]


def test_a_failure_while_reading_stops_the_conversion_and_leaves_nothing(clip, device, video_engine,
                                                                        tmp_path, monkeypatch):
    real = job.read_frames

    def failing(path, device, on_packet=None, **kwargs):
        for frame in real(path, device, on_packet=on_packet, **kwargs):
            if on_packet is not None and frame.index == 17:
                raise OSError("the disk went away")
            yield frame

    monkeypatch.setattr(job, "read_frames", failing)
    with pytest.raises(OSError, match="the disk went away"):
        convert_video(clip, tmp_path / "out.mp4", settings(), device, engine=video_engine)
    assert list(tmp_path.iterdir()) == []
    assert not [t for t in threading.enumerate() if t.name.startswith("hdr-blaster")]


def test_a_failure_in_the_encoder_stops_the_conversion_and_leaves_nothing(clip, device, video_engine,
                                                                         tmp_path, monkeypatch):
    real = job.VideoWriter.encode

    def failing(self, picture):
        if self.frames_written == 11:
            raise RuntimeError("the encoder gave up")
        real(self, picture)

    monkeypatch.setattr(job.VideoWriter, "encode", failing)
    with pytest.raises(RuntimeError, match="the encoder gave up"):
        convert_video(clip, tmp_path / "out.mp4", settings(), device, engine=video_engine)
    assert list(tmp_path.iterdir()) == []
    assert not [t for t in threading.enumerate() if t.name.startswith("hdr-blaster")]


def test_cancelling_stops_the_threads_too(clip, device, video_engine, tmp_path):
    seen = []
    with pytest.raises(Cancelled):
        convert_video(clip, tmp_path / "out.mp4", settings(), device, engine=video_engine,
                      progress=lambda stage, done, total: seen.append(stage),
                      cancelled=lambda: seen.count("render") >= 9)
    assert list(tmp_path.iterdir()) == []
    assert not [t for t in threading.enumerate() if t.name.startswith("hdr-blaster")]


def test_a_video_is_neither_sharpened_nor_coloured_unless_asked():
    assert (VideoSettings().render.edge_strength, VideoSettings().render.color_strength) == (0.0, 0.0)


class _GivesUp:
    """A hardware decoder that fails after a number of packets, like a card that is reset."""

    def __init__(self, real, after):
        self.real, self.left = real, after

    def decode(self, packet):
        self.left -= 1
        if self.left < 0:
            raise av.FFmpegError(5, "the device was lost")
        return self.real.decode(packet)


@pytest.mark.parametrize("stage", ["analyse", "render"])
def test_if_the_card_stops_decoding_the_pass_starts_again_on_the_processor(clip, device, video_engine, tmp_path,
                                                                          monkeypatch, caplog, stage):
    import logging

    from hdr_blaster.video import reader

    if device.type != "cuda" or not reader.hardware_decoding(clip):
        pytest.skip("no graphics card that decodes this clip")
    convert_video(clip, tmp_path / "plain.mp4", settings(), device, engine=video_engine, reference=True)

    real, opened = reader._open_hardware, []

    def fragile(stream):
        opened.append(1)
        # the analysis is the first pass to open a decoder, the render the second
        fails = len(opened) == (1 if stage == "analyse" else 2)
        return _GivesUp(real(stream), 15) if fails else real(stream)

    monkeypatch.setattr(reader, "_hardware_checked", dict(reader._hardware_checked))
    monkeypatch.setattr(reader, "_open_hardware", fragile)
    stages = []
    with caplog.at_level(logging.WARNING):
        report = convert_video(clip, tmp_path / "out.mp4", settings(), device, engine=video_engine,
                               progress=lambda s, done, total: stages.append((s, done)))
    assert "starting this pass again" in caplog.text
    assert reader.hardware_decoding(clip) is False, "the format is remembered as one the card does not decode"
    assert report.frames == 40 and stages[-1] == ("render", 40)
    # (this short clip is encoded a second time anyway, to end lighter than its source)
    assert stages.count((stage, 1)) == (2 if stage == "analyse" else 3), "the pass was started once more"
    one, other = decoded(tmp_path / "plain.mp4"), decoded(tmp_path / "out.mp4")
    assert len(one) == len(other) == 40
    assert all(a[0] == b[0] and np.array_equal(a[1], b[1]) for a, b in zip(one, other))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4", "plain.mp4"]


def test_without_any_nvidia_card_the_fast_way_still_writes_the_same_pictures(clip, tmp_path, monkeypatch):
    """Everything on the processor: decoding, rendering, x264. The stages still run side by
    side, and still change nothing."""
    from hdr_blaster.video import reader, writer

    cpu = torch.device("cpu")
    engine = make_engine(cpu)
    asked = []
    monkeypatch.setattr(reader, "hardware_decoding", lambda path: asked.append(path) or True)
    monkeypatch.setattr(writer, "pick_encoder", lambda: "libx264")
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0))
    plain = convert_video(clip, tmp_path / "plain.mp4", chosen, cpu, engine=engine, reference=True)
    fast = convert_video(clip, tmp_path / "fast.mp4", chosen, cpu, engine=engine)
    assert asked == [], "the graphics card is not asked to decode for a conversion on the processor"
    assert plain.encoder == fast.encoder == "libx264"
    one, other = decoded(tmp_path / "plain.mp4"), decoded(tmp_path / "fast.mp4")
    assert len(one) == len(other) == 40
    assert all(a[0] == b[0] and np.array_equal(a[1], b[1]) for a, b in zip(one, other))


# ----------------------------------------------------------------- frame rate

def test_a_regular_recording_gives_every_other_frame_at_half_its_rate():
    from hdr_blaster.video.job import frames_to_keep

    times = [i / 120 for i in range(240)]
    assert frames_to_keep(times, None) == [True] * 240
    assert frames_to_keep(times, 60, 120) == [i % 2 == 0 for i in range(240)]
    assert frames_to_keep(times, 30, 120) == [i % 4 == 0 for i in range(240)]
    assert frames_to_keep(times, 120, 120) == [True] * 240, "the source's own rate keeps everything"
    assert frames_to_keep(times, 240, 120) == [True] * 240, "frames are never invented"
    ntsc = [i * 1001 / 60000 for i in range(120)]
    assert all(frames_to_keep(ntsc, 60, 60000 / 1001)), "59.94 asked for 60 loses nothing"
    assert frames_to_keep(ntsc, 30, 60000 / 1001) == [i % 2 == 0 for i in range(120)]
    assert frames_to_keep([], 60, 120) == [] and frames_to_keep([0.5], 30, 120) == [True]


def test_an_irregular_recording_gives_the_frame_nearest_to_each_tick_and_never_more_than_asked():
    from hdr_blaster.video.job import frames_to_keep

    rng = np.random.default_rng(3)
    times = (2.0 + np.cumsum(rng.uniform(0.004, 0.040, 2000))).tolist()      # 4 to 40 ms apart
    for rate in (30.0, 20.0):                                              # the recording runs at 45 on average
        keep = frames_to_keep(times, rate, 45.0)
        assert keep[0], "the first frame is always written"
        position = (np.array(times) - times[0]) * rate
        tick = np.floor(position + 0.5).astype(int)
        kept = np.flatnonzero(keep)
        ticks = sorted(set(tick[tick <= position[-1]].tolist()))             # none after the last frame
        assert tick[kept].tolist() == ticks, "one frame for every tick that has any, no other"
        for index in kept:
            rivals = np.flatnonzero(tick == tick[index])
            assert abs(position[index] - tick[index]) <= np.abs(position[rivals] - tick[index]).min() + 1e-9
        assert sum(keep) <= (times[-1] - times[0]) * rate + 1


def test_a_capture_with_jittering_timestamps_keeps_an_even_cadence():
    """An NVIDIA recording at 120 frames per second stamps its frames 5.7 to 10.3 ms apart.
    Halved, it must still give every other frame: 8 ms then 25 ms between two frames would
    stutter."""
    from hdr_blaster.video.job import frames_to_keep

    rng = np.random.default_rng(7)
    times = (np.arange(1200) / 120.0 + rng.uniform(-0.0023, 0.0023, 1200)).tolist()
    assert frames_to_keep(times, 60, 120) == [i % 2 == 0 for i in range(1200)]
    assert frames_to_keep(times, 30, 120) == [i % 4 == 0 for i in range(1200)]


def test_a_lower_frame_rate_writes_the_same_pictures_fewer_of_them(clip, device, video_engine, tmp_path, monkeypatch):
    """The frames written at a lower rate are the very frames of the full conversion, with
    their own timestamps: nothing is rendered differently, the others are left out."""
    from hdr_blaster.video.job import frames_to_keep

    handed, run = {}, ["full"]
    real = job.VideoWriter.frame

    def spy(self, pts, y, u, v):
        marks = hashlib.sha256(b"".join(p.cpu().numpy().tobytes() for p in (y, u, v))).hexdigest()
        handed.setdefault(run[0], {})[pts] = marks
        return real(self, pts, y, u, v)

    monkeypatch.setattr(job.VideoWriter, "frame", spy)
    full = convert_video(clip, tmp_path / "full.mp4", settings(), device, engine=video_engine)
    run[0] = "slow"
    slow = convert_video(clip, tmp_path / "slow.mp4", settings(frame_rate=20.0), device, engine=video_engine)
    run[0] = "plain"
    convert_video(clip, tmp_path / "plain.mp4", settings(frame_rate=20.0), device, engine=video_engine, reference=True)

    source = clips.describe(clip)
    keep = frames_to_keep(source["pts_s"], 20.0, probe(clip).average_rate)
    assert full.frames == 40 and slow.frames == sum(keep) and 10 < sum(keep) < 30
    out = clips.describe(tmp_path / "slow.mp4")
    assert out["pts_s"] == pytest.approx([t for t, k in zip(source["pts_s"], keep) if k], abs=1e-4)
    assert len(handed["slow"]) == sum(keep)
    assert handed["slow"] == {pts: handed["full"][pts] for pts in handed["slow"]}, "the same pictures"
    assert handed["plain"] == handed["slow"], "and the plain way keeps the same ones"
    assert out["audio_bytes"] == source["audio_bytes"], "the sound is whole"
    assert slow.dst_bytes <= slow.src_bytes
    # a rate the recording does not exceed changes nothing
    same = convert_video(clip, tmp_path / "same.mp4", settings(frame_rate=120.0), device, engine=video_engine)
    assert same.frames == 40


def test_one_analysis_serves_every_frame_rate(clip, device, video_engine, tmp_path):
    from hdr_blaster.video.job import analyse_video, analysis_key, render_video

    assert analysis_key(settings(frame_rate=30.0)) == analysis_key(settings())
    analysis = analyse_video(clip, settings(), device, engine=video_engine)
    report = render_video(analysis, tmp_path / "slow.mp4", settings(frame_rate=20.0), device, engine=video_engine)
    assert report.frames < 40 and len(clips.describe(tmp_path / "slow.mp4")["pts_s"]) == report.frames


def test_command_line_takes_a_frame_rate(clip, tmp_path):
    out = tmp_path / "cli.mp4"
    done = subprocess.run(
        [sys.executable, str(ROOT / "NHC.py"), "--convert-video", str(clip), "--out", str(out), "--fps", "20"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 10 < len(clips.describe(out)["pts_s"]) < 30


# ----------------------------------------------------------------- motion blur

def test_a_pictures_light_comes_from_the_frames_it_replaces_never_from_another_shot():
    from hdr_blaster.video.job import exposure_of

    keep = [True, False, True, False, False, True, False, False, False]
    assert exposure_of(keep, set(), 0) == (0, 2)
    assert exposure_of(keep, set(), 1) == (0, 2), "a dropped frame is shown by the picture before it"
    assert exposure_of(keep, set(), 2) == exposure_of(keep, set(), 4) == (2, 3)
    assert exposure_of(keep, set(), 5) == (5, 4), "up to the end of the file"
    assert exposure_of(keep, {4}, 2) == (2, 2), "the shot that begins at frame 4 keeps its light"
    assert exposure_of(keep, {3}, 4) == (2, 1)
    assert exposure_of(keep, {5}, 2) == (2, 3), "a cut on a kept frame changes nothing"
    assert exposure_of([True] * 4, set(), 2) == (2, 1)


def test_motion_blur_adds_up_the_scenes_light_and_renders_it_once(device, video_engine, tmp_path, monkeypatch):
    """Each picture written is rendered from the average, in the scene's linear light, of the
    kept frame and of the frames dropped after it, with the kept frame's parameters; the
    light of the next shot is left out."""
    from hdr_blaster.video.colour import scrgb_to_engine, yuv_to_scrgb
    from hdr_blaster.video.job import analyse_video, exposure_of, frames_to_keep, render_video
    from hdr_blaster.video.reader import read_frames

    src = tmp_path / "cut.mp4"
    clips.make_clip(src, frames=40, audio_streams=0, levels=[1.0] * 20 + [0.08] * 20)
    info = probe(src)
    analysis = analyse_video(src, settings(method="faithful"), device, engine=video_engine)
    assert analysis.cuts == [0, 20]

    def stopped_by_the_cut(rate):
        keep = frames_to_keep(analysis.times, rate, info.average_rate)
        return any(k and exposure_of(keep, set(), i) != exposure_of(keep, {20}, i) for i, k in enumerate(keep))

    rate = next(r for r in (20.0, 15.0, 12.0, 10.0, 8.0) if stopped_by_the_cut(r))
    keep = frames_to_keep(analysis.times, rate, info.average_rate)
    kept = [i for i, k in enumerate(keep) if k]
    rendered, real = [], job.render

    def spy(tone_mapper, loader, processor, linear, render_settings, params=None, **kwargs):
        if params is not None and params.frozen:
            rendered.append((linear.clone(), params.stats["scene.peak"].clone()))
        return real(tone_mapper, loader, processor, linear, render_settings, params=params, **kwargs)

    monkeypatch.setattr(job, "render", spy)
    chosen = settings(method="faithful", frame_rate=rate, motion_blur=True)
    report = render_video(analysis, tmp_path / "blur.mp4", chosen, device, engine=video_engine)
    assert report.frames == len(kept) and len(rendered) % len(kept) == 0, "one render per picture written"

    with torch.inference_mode():
        lights = [yuv_to_scrgb(f.y, f.u, f.v, info.fmt) for f in read_frames(src, device, hardware=False)]
        blended = 0
        for n, first in enumerate(kept):
            count = exposure_of(keep, {20}, first)[1]
            total = lights[first]
            for extra in range(1, count):
                total = total + lights[first + extra]
            expected = scrgb_to_engine(total / count if count > 1 else total)
            assert torch.equal(rendered[n][0], expected), f"picture {n} (frames {first} to {first + count - 1})"
            assert torch.equal(rendered[n][1].cpu(), analysis.params[first].stats["scene.peak"].cpu())
            blended += count > 1 and not torch.equal(expected, scrgb_to_engine(lights[first]))
        assert blended > 3, "the test clip must really blend something"
    out = clips.describe(tmp_path / "blur.mp4")
    assert out["pts_s"] == pytest.approx([analysis.times[i] for i in kept], abs=1e-4), "timestamps are the kept frames'"


def test_motion_blur_is_the_same_both_ways_and_does_nothing_at_the_original_rate(clip, device, video_engine,
                                                                               tmp_path, monkeypatch):
    handed, run = {}, ["blur"]
    real = job.VideoWriter.frame

    def spy(self, pts, y, u, v):
        marks = hashlib.sha256(b"".join(p.cpu().numpy().tobytes() for p in (y, u, v))).hexdigest()
        handed.setdefault(run[0], {})[pts] = marks
        return real(self, pts, y, u, v)

    monkeypatch.setattr(job.VideoWriter, "frame", spy)
    blurred = settings(frame_rate=15.0, motion_blur=True)
    convert_video(clip, tmp_path / "blur.mp4", blurred, device, engine=video_engine)
    run[0] = "plain way"
    convert_video(clip, tmp_path / "plain.mp4", blurred, device, engine=video_engine, reference=True)
    run[0] = "sharp"
    convert_video(clip, tmp_path / "sharp.mp4", settings(frame_rate=15.0), device, engine=video_engine)
    run[0] = "full"
    convert_video(clip, tmp_path / "full.mp4", settings(), device, engine=video_engine)
    run[0] = "full, blur asked"
    convert_video(clip, tmp_path / "full_blur.mp4", settings(motion_blur=True), device, engine=video_engine)
    assert handed["plain way"] == handed["blur"], "the fast way and the plain way blur alike"
    assert sorted(handed["blur"]) == sorted(handed["sharp"]), "the same frames are written, at the same times"
    assert handed["blur"] != handed["sharp"], "with other pictures"
    assert handed["full, blur asked"] == handed["full"], "at the recording's own rate there is nothing to add up"


def test_command_line_takes_motion_blur(clip, tmp_path):
    out = tmp_path / "cli.mp4"
    done = subprocess.run(
        [sys.executable, str(ROOT / "NHC.py"), "--convert-video", str(clip), "--out", str(out),
         "--fps", "15", "--blur"], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 5 < len(clips.describe(out)["pts_s"]) < 25

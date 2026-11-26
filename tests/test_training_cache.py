"""Frame caches, the teacher and GPU-side augmentation (training/cache.py, teacher.py)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import video_clips as clips

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import cache  # noqa: E402
import teacher as teacher_module  # noqa: E402


def write_radiance(path, rgb):
    """A flat (uncompressed) Radiance .hdr file from an (H, W, 3) float array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    brightest = rgb.max(axis=-1)
    exponent = np.where(brightest > 0, np.floor(np.log2(np.maximum(brightest, 1e-38))) + 1, 0)
    scale = np.where(brightest > 0, 256.0 / np.exp2(exponent), 0.0)
    rgbe = np.concatenate([np.clip(rgb * scale[..., None], 0, 255),
                           np.where(brightest > 0, exponent + 128, 0)[..., None]], axis=-1).astype(np.uint8)
    height, width = rgb.shape[:2]
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode("ascii")
    Path(path).write_bytes(header + rgbe.tobytes())


# ---------------------------------------------------------------------- caches

def test_frames_are_held_out_by_origin_never_frame_by_frame(tmp_path):
    writer = cache.CacheWriter(tmp_path / "c")
    for origin in range(40):
        for frame in range(3):
            writer.add(torch.full((3, cache.HEIGHT, cache.WIDTH), float(origin)), f"video_{origin}.mp4", f"{frame}")
    assert writer.close() == 120
    train, val = cache.FrameCache(tmp_path / "c", "train"), cache.FrameCache(tmp_path / "c", "val")
    assert len(train) + len(val) == 120 and len(val) > 0 and len(train) > len(val)
    assert set(train.origins).isdisjoint(val.origins)
    assert len(val) == 3 * len(val.origins)
    item = train[0]
    assert set(item) == {"hdr"} and item["hdr"].shape == (3, cache.HEIGHT, cache.WIDTH)
    assert item["hdr"].dtype == torch.float32
    assert float(item["hdr"].mean()) == float(train.origins[0].split("_")[1].split(".")[0])


def test_origins_can_be_excluded(tmp_path):
    writer = cache.CacheWriter(tmp_path / "c")
    for name in ["Desktop 2026.05.14.mp4", "Forza.mp4", "Desktop 2026.06.05.mp4", "Hogwarts.mp4"] * 5:
        writer.add(torch.zeros(3, cache.HEIGHT, cache.WIDTH), name)
    writer.close()
    kept = [o for split in ("train", "val")
            for o in cache.FrameCache(tmp_path / "c", split, exclude=["Desktop"]).origins]
    assert sorted(kept) == ["Forza.mp4", "Hogwarts.mp4"]
    total = sum(len(cache.FrameCache(tmp_path / "c", s, exclude=["Desktop"])) for s in ("train", "val"))
    assert total == 10


def test_wrong_frame_size_is_refused(tmp_path):
    writer = cache.CacheWriter(tmp_path / "c")
    with pytest.raises(ValueError, match="expected"):
        writer.add(torch.zeros(3, 100, 100), "x")
    writer.close()


def test_video_frames_are_stored_in_nits(tmp_path, device):
    source = tmp_path / "videos"
    source.mkdir()
    clips.make_clip(source / "game.mp4", frames=24, audio_streams=0, texture=0.0)
    clips.make_clip(source / "sdr.mp4", frames=4, hdr=False, audio_streams=0)      # must be skipped
    messages = []
    count = cache.from_videos(source, tmp_path / "c", every_s=0.05, device=device, log=messages.append)
    assert 3 <= count <= 24
    assert any("skipped sdr.mp4" in m for m in messages)
    frames = cache.FrameCache(tmp_path / "c", "train")
    frames = frames if len(frames) else cache.FrameCache(tmp_path / "c", "val")
    stored = frames[0]["hdr"]
    expected = torch.from_numpy(clips.scene_nits(256, 144, 0.0)).permute(2, 0, 1).float()
    expected = torch.nn.functional.interpolate(expected[None], size=(cache.HEIGHT, cache.WIDTH), mode="bilinear")[0]
    relative = ((stored - expected).abs() / (expected + 5.0)).mean()
    assert float(relative) < 0.10


def test_panoramas_are_read_in_rgb_order_and_normalised(tmp_path):
    source = tmp_path / "hdris"
    source.mkdir()
    pano = np.empty((128, 256, 3))
    pano[...] = [8.0, 2.0, 0.5]                                   # a strongly red-orange light
    pano[:, :8] = [4000.0, 3000.0, 2000.0]                        # and a very bright strip
    write_radiance(source / "sky.hdr", pano)
    assert np.allclose(cache.read_radiance(source / "sky.hdr")[64, 100], [8.0, 2.0, 0.5], rtol=0.02)
    count = cache.from_hdris(source, tmp_path / "c", crops=4)
    assert count == 4
    frames = cache.FrameCache(tmp_path / "c", "train")
    frames = frames if len(frames) else cache.FrameCache(tmp_path / "c", "val")
    view = frames[0]["hdr"]
    luminance = 0.2126 * view[0] + 0.7152 * view[1] + 0.0722 * view[2]
    assert float(luminance.median()) == pytest.approx(10.0, rel=0.02)
    red, green, blue = view[:, 100, 300].tolist()
    assert red > green > blue


def test_photos_become_plausible_hdr(tmp_path):
    from PIL import Image

    source = tmp_path / "photos"
    source.mkdir()
    Image.fromarray(np.full((300, 300, 3), 128, dtype=np.uint8)).save(source / "grey.png")
    assert cache.from_images(source, tmp_path / "c") == 1
    frames = cache.FrameCache(tmp_path / "c", "train")
    frames = frames if len(frames) else cache.FrameCache(tmp_path / "c", "val")
    linear = ((128 / 255 + 0.055) / 1.055) ** 2.4
    assert torch.allclose(frames[0]["hdr"], torch.full((3, cache.HEIGHT, cache.WIDTH), linear * 200.0), rtol=2e-3)


# --------------------------------------------------------------------- teacher

def test_the_teacher_is_the_applications_own_look_at_a_fixed_white(engine, device):
    """The teacher keeps the scene's exposure, like the looks of the application: two very
    different pictures in one batch are each the application's render of that picture, with
    SDR white standing for the teacher's white."""
    from hdr_blaster.engine.pipeline import tonemap_srgb

    torch.manual_seed(0)
    dark = torch.rand(1, 3, 72, 128, device=device) * 20.0
    bright = torch.rand(1, 3, 72, 128, device=device) * 900.0
    teach = teacher_module.Teacher("punchy", device)
    assert teach.white_nits == teacher_module.TEACHER_WHITE_NITS == 203.0
    out = teach(torch.cat([dark, bright]))
    for image, rendered in ((dark, out[0:1]), (bright, out[1:2])):
        expected = tonemap_srgb(engine.tone_mapper, engine.loader, image / 800.0, "punchy", white_nits=203.0)
        assert torch.allclose(rendered, expected, atol=1e-6)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert float(out[0].mean()) < 0.5 * float(out[1].mean()), "a dark picture is taught dark"
    brighter_white = teacher_module.Teacher("punchy", device, white_nits=480.0)(bright)
    assert float(brighter_white.mean()) < float(out[1].mean()), "a brighter white shows the same scene darker"


def test_unknown_teacher_is_refused(device):
    with pytest.raises(ValueError, match="perceptual"):
        teacher_module.Teacher("perceptual", device)


def test_augmentation_varies_exposure_and_peak_within_bounds(device):
    generator = torch.Generator(device=device).manual_seed(0)
    frames = torch.full((64, 3, 8, 16), 100.0, device=device)
    frames[:, :, 0, 0] = 50000.0                                   # a sun
    out = teacher_module.augment(frames, ev=2.0, generator=generator)
    level = out[:, 0, 4, 8]
    assert float(level.min()) >= 100.0 / 4.0 - 1e-3 and float(level.max()) <= 100.0 * 4.0 + 1e-3
    assert float(level.max() / level.min()) > 4.0, "exposures should differ between images"
    peaks = out.amax(dim=(1, 2, 3))
    assert float(peaks.min()) >= 600.0 - 1e-2 and float(peaks.max()) <= 4000.0 + 1e-2
    again = teacher_module.augment(frames, ev=2.0, generator=torch.Generator(device=device).manual_seed(0))
    assert torch.equal(out, again)
    assert torch.equal(teacher_module.augment(frames[:1], ev=0.0, peak_range=(1e9, 1e9 + 1)).abs(), frames[:1])


# -------------------------------------------------------------------- training

def tiny_cache(folder):
    torch.manual_seed(0)
    writer = cache.CacheWriter(folder)
    for origin in range(40):
        writer.add(torch.rand(3, cache.HEIGHT, cache.WIDTH) * (5.0 + 20.0 * origin), f"clip_{origin}.mp4")
    writer.close()


def run_train(cwd, *args):
    return subprocess.run([sys.executable, str(TRAINING / "train.py"), *args, "--device", "cpu",
                           "--num-workers", "0", "--batch-size", "4"],
                          cwd=cwd, capture_output=True, text=True, timeout=900)


def test_training_from_a_cache_with_a_teacher(tmp_path):
    tiny_cache(tmp_path / "cache")
    done = run_train(tmp_path, "--cache-dir", str(tmp_path / "cache"), "--teacher", "punchy",
                     "--out-dir", str(tmp_path / "run"), "--epochs", "1")
    assert done.returncode == 0, done.stdout + done.stderr
    ckpt = torch.load(tmp_path / "run" / "best.pth", map_location="cpu", weights_only=True)
    assert ckpt["teacher"] == "punchy" and ckpt["input_unit"] == "nits"
    assert ckpt["anchor_white_nits"] == 203.0, "the application needs it to feed the model"
    assert np.isfinite(ckpt["val_loss"])


def test_a_cache_without_a_teacher_is_refused_with_an_explanation(tmp_path):
    tiny_cache(tmp_path / "cache")
    done = run_train(tmp_path, "--cache-dir", str(tmp_path / "cache"), "--out-dir", str(tmp_path / "run"))
    assert done.returncode != 0
    assert "--teacher" in done.stdout + done.stderr
    assert not (tmp_path / "run" / "latest.pth").exists()

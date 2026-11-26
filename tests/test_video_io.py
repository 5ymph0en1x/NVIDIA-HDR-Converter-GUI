"""Reading HDR10 clips and writing SDR ones."""
import os

import pytest
import torch

import video_clips as clips
from hdr_blaster.video.reader import UnsupportedVideo, check_supported, probe, read_frames
from hdr_blaster.video.writer import QUALITY, VideoWriter, pick_encoder


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clips") / "hdr.mp4"
    pts = clips.make_clip(path, frames=20)
    return path, pts


def grey_planes(height, width, device):
    return (torch.full((height, width), 128, dtype=torch.uint8, device=device),
            torch.full((height // 2, width // 2), 128, dtype=torch.uint8, device=device),
            torch.full((height // 2, width // 2), 128, dtype=torch.uint8, device=device))


# --------------------------------------------------------------------- reading

def test_probe_describes_the_clip(clip):
    path, pts = clip
    info = probe(path)
    assert (info.width, info.height, info.frames) == (256, 144, 20)
    assert info.codec == "hevc" and info.pix_fmt == "yuv420p10le"
    assert info.fmt.bit_depth == 10 and info.fmt.full_range is False
    assert (info.transfer, info.primaries, info.matrix) == (16, 9, 9)
    assert info.audio_streams == 1
    assert info.size_bytes == os.path.getsize(path)
    assert info.bit_rate > 0
    assert info.duration_s == pytest.approx((pts[-1] + 2700) / 90000, abs=0.05)
    check_supported(info)


def test_full_range_is_detected(tmp_path):
    path = tmp_path / "full.mp4"
    clips.make_clip(path, frames=3, full_range=True, audio_streams=0)
    assert probe(path).fmt.full_range is True


def test_an_sdr_video_is_refused_with_a_message_for_the_user(tmp_path):
    path = tmp_path / "sdr.mp4"
    clips.make_clip(path, frames=3, hdr=False)
    with pytest.raises(UnsupportedVideo, match="not an HDR10 video") as error:
        check_supported(probe(path))
    assert "sdr.mp4" in str(error.value) and "BT.709" in str(error.value)


def test_a_file_that_is_not_a_video_is_refused(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("not a video")
    with pytest.raises(UnsupportedVideo, match="notes.mp4"):
        probe(path)


def test_frames_come_in_order_with_their_timestamps(clip, device):
    path, pts = clip
    frames = list(read_frames(path, device))
    assert [f.index for f in frames] == list(range(20))
    assert [f.pts for f in frames] == pts
    assert frames[3].time_s == pytest.approx(pts[3] / 90000)
    first = frames[0]
    assert first.y.shape == (144, 256) and first.u.shape == (72, 128) and first.v.shape == (72, 128)
    assert first.y.device.type == device.type
    assert 64 <= int(first.y.min()) and int(first.y.max()) <= 940          # limited range, 10 bit


def test_decoded_light_matches_what_was_encoded(clip, device):
    """Through a real HEVC encode: the scene comes back at the nits it was written with."""
    from hdr_blaster.video.colour import ENGINE_UNIT_NITS, yuv_to_engine

    path, _ = clip
    frame = next(iter(read_frames(path, device)))
    engine = yuv_to_engine(frame.y, frame.u, frame.v, probe(path).fmt)[0] * ENGINE_UNIT_NITS
    expected = torch.from_numpy(clips.scene_nits(256, 144, 0.0)).permute(2, 0, 1).float().to(device)
    relative = ((engine - expected).abs() / (expected + 5.0)).mean()
    assert float(relative) < 0.06


def test_audio_packets_are_handed_over(clip, device):
    path, _ = clip
    packets = []
    frames = sum(1 for _ in read_frames(path, device, on_packet=packets.append))
    assert frames == 20
    assert len(packets) == clips.describe(path)["audio_packets"][0] > 0


# --------------------------------------------------------------------- writing

def convert_as_grey(src, dst, device, **kwargs):
    writer = VideoWriter(dst, probe(src), **kwargs)
    for frame in read_frames(src, device, on_packet=writer.copy_packet):
        writer.write(frame.pts, *grey_planes(writer.height, writer.width, device))
    return writer


def test_written_file_is_sdr_h264_with_the_source_timestamps_and_audio(clip, device, tmp_path):
    src, _ = clip
    dst = tmp_path / "out.mp4"
    writer = convert_as_grey(src, dst, device)
    assert writer.close() == str(dst)
    assert not os.path.exists(str(dst) + ".partial")
    out, source = clips.describe(dst), clips.describe(src)
    assert (out["codec"], out["pix_fmt"]) == ("h264", "yuv420p")
    assert (out["width"], out["height"]) == (256, 144)
    assert (out["range"], out["matrix"], out["trc"], out["primaries"]) == (1, 1, 1, 1)
    assert out["pts_s"] == pytest.approx(source["pts_s"], abs=1e-4)
    assert out["audio_packets"] == source["audio_packets"]
    assert out["audio_bytes"] == source["audio_bytes"]


@pytest.mark.parametrize("streams", [0, 2])
def test_no_audio_and_several_audio_streams(device, tmp_path, streams):
    src = tmp_path / "in.mp4"
    clips.make_clip(src, frames=6, audio_streams=streams)
    dst = tmp_path / "out.mp4"
    convert_as_grey(src, dst, device).close()
    out, source = clips.describe(dst), clips.describe(src)
    assert out["audio_streams"] == streams
    assert out["audio_bytes"] == source["audio_bytes"]
    assert len(out["pts_s"]) == 6


def test_abort_leaves_no_file(clip, device, tmp_path):
    src, _ = clip
    dst = tmp_path / "out.mp4"
    writer = VideoWriter(dst, probe(src))
    for i in range(60):                      # enough for the encoder to start writing
        writer.write(i * 1500, *grey_planes(144, 256, device))
        if os.path.exists(str(dst) + ".partial"):
            break
    assert os.path.exists(str(dst) + ".partial"), "the encoder never wrote anything"
    writer.abort()
    assert list(tmp_path.iterdir()) == []


def test_the_software_encoder_works_too(clip, device, tmp_path):
    src, _ = clip
    dst = tmp_path / "soft.mp4"
    writer = convert_as_grey(src, dst, device, encoder="libx264")
    writer.close()
    assert writer.encoder_name == "libx264"
    out = clips.describe(dst)
    assert out["codec"] == "h264" and len(out["pts_s"]) == 20


def test_output_size_can_differ_from_the_source(clip, device, tmp_path):
    src, _ = clip
    dst = tmp_path / "small.mp4"
    convert_as_grey(src, dst, device, size=(192, 108)).close()
    out = clips.describe(dst)
    assert (out["width"], out["height"]) == (192, 108)


def test_bad_arguments_are_refused_before_any_file_is_created(clip, tmp_path):
    src, _ = clip
    with pytest.raises(ValueError, match="even"):
        VideoWriter(tmp_path / "a.mp4", probe(src), size=(193, 108))
    with pytest.raises(ValueError, match="ultra"):
        VideoWriter(tmp_path / "b.mp4", probe(src), quality="ultra")
    assert list(tmp_path.iterdir()) == []


def test_quality_levels_and_encoder_choice():
    assert set(QUALITY) == {"high", "balanced", "compact"}
    assert QUALITY["high"]["cap"] < 1.0, "the result must stay lighter than the source"
    assert QUALITY["high"]["cap"] > QUALITY["balanced"]["cap"] > QUALITY["compact"]["cap"]
    assert pick_encoder() in ("h264_nvenc", "libx264")


def test_a_failed_final_rename_leaves_no_partial_file(clip, device, tmp_path, monkeypatch):
    src, _ = clip
    dst = tmp_path / "out.mp4"
    writer = convert_as_grey(src, dst, device)
    real_replace = os.replace

    def locked(a, b):
        raise PermissionError("destination is locked")

    monkeypatch.setattr(os, "replace", locked)
    with pytest.raises(PermissionError):
        writer.close()
    monkeypatch.setattr(os, "replace", real_replace)
    writer.abort()
    assert list(tmp_path.iterdir()) == []

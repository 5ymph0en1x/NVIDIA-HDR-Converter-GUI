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
    assert (out["range"], out["matrix"], out["trc"], out["primaries"]) == (1, 1, 13, 1)
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


# ------------------------------------------------------ the graphics card as decoder

def test_the_graphics_card_decodes_the_same_code_values_as_the_processor(clip, device):
    """Decoding HEVC and AV1 is exact by specification: NVDEC must give what FFmpeg gives."""
    from hdr_blaster.video import reader

    path, _ = clip
    if device.type != "cuda" or not reader.hardware_decoding(path):
        pytest.skip("no graphics card that decodes this clip")
    packets = []
    plain = list(read_frames(path, device, hardware=False))
    fast = list(read_frames(path, device, on_packet=packets.append, hardware=True))
    assert len(fast) == len(plain) > 0 and packets, "audio packets are handed over the same way"
    for a, b in zip(fast, plain):
        assert (a.index, a.pts, a.time_s) == (b.index, b.pts, b.time_s)
        for one, other in ((a.y, b.y), (a.u, b.u), (a.v, b.v)):
            assert one.dtype == other.dtype and one.shape == other.shape and torch.equal(one, other)


def test_without_a_card_that_decodes_the_processor_does(clip, device, monkeypatch):
    from hdr_blaster.video import reader

    def refuse(stream):
        raise OSError("no NVDEC here")

    path, _ = clip
    monkeypatch.setattr(reader, "_open_hardware", refuse)
    monkeypatch.setattr(reader, "_hardware_checked", {})
    assert reader.hardware_decoding(path) is False
    assert len(list(read_frames(path, device))) == len(list(read_frames(path, device, hardware=False)))


def test_frames_read_into_main_memory_do_not_share_their_buffer(clip):
    """The reader reuses one buffer: a frame kept (the stages keep a few) must not change
    when the next one is read."""
    path, _ = clip
    frames = read_frames(path, torch.device("cpu"))
    first = next(frames)
    kept = first.y.clone()
    next(frames), next(frames)
    assert torch.equal(first.y, kept)


def test_a_card_that_decodes_differently_is_not_used(clip, device, monkeypatch, caplog):
    """The card is trusted only after its first frame came out as the processor decodes it."""
    import logging

    from hdr_blaster.video import reader

    path, _ = clip
    if device.type != "cuda" or not reader.hardware_decoding(path):
        pytest.skip("no graphics card that decodes this clip")
    real = reader._first_frame

    def one_code_off(path, hardware):
        frame = real(path, hardware)
        if hardware:
            y = frame[1].clone()
            y[5, 7] += 1
            frame = (frame[0], y) + frame[2:]
        return frame

    monkeypatch.setattr(reader, "_hardware_checked", {})
    monkeypatch.setattr(reader, "_first_frame", one_code_off)
    with caplog.at_level(logging.WARNING):
        assert reader.hardware_decoding(path) is False
    assert "as the processor does" in caplog.text
    assert len(list(read_frames(path, device))) == 20, "and the file is read all the same"


def test_frames_that_are_not_wanted_are_counted_but_not_carried(clip, device):
    path, _ = clip
    everything = list(read_frames(path, device))
    some = list(read_frames(path, device, wanted=lambda index: index % 3 == 0))
    assert [(f.index, f.pts) for f in some] == [(f.index, f.pts) for f in everything]
    for light, whole in zip(some, everything):
        if light.index % 3 == 0:
            assert torch.equal(light.y, whole.y) and torch.equal(light.u, whole.u) and torch.equal(light.v, whole.v)
        else:
            assert light.y is None and light.u is None and light.v is None


# ------------------------------------------------------- the inspected MP4 frame

def test_an_inspected_frame_is_decoded_to_the_rgb_its_samples_stand_for():
    """'Preview encoded MP4' shows what the file holds. The library's default conversion
    to RGB truncates: every pixel came out a level darker than its samples say (greys
    included), which is an error of the preview, not of the file."""
    import av
    import numpy as np

    from hdr_blaster.video.encoded_preview import rgb_of

    def frame_of(y, cb, cr):
        height, width = y.shape
        chroma = np.concatenate((np.full((height // 2, width // 2), cb, np.uint8).reshape(-1),
                                 np.full((height // 2, width // 2), cr, np.uint8).reshape(-1)))
        frame = av.VideoFrame.from_ndarray(np.vstack((y, chroma.reshape(height // 2, width))), format="yuv420p")
        frame.colorspace, frame.color_range = 1, 1            # BT.709, limited range: what the writer signals
        return frame

    def exact(y, cb, cr):                                     # ITU-R BT.709, limited range, in binary64
        yy, pb, pr = (y.astype(np.float64) - 16) / 219, (cb - 128) / 224, (cr - 128) / 224
        red, blue = yy + 1.5748 * pr, yy + 1.8556 * pb
        green = (yy - 0.2126 * red - 0.0722 * blue) / 0.7152
        return np.clip(np.stack((red, green, blue), -1), 0, 1) * 255

    ramp = np.repeat(np.arange(16, 236, dtype=np.uint8)[None, :], 64, 0)        # every grey of the range
    shown = rgb_of(frame_of(ramp, 128, 128)).astype(np.float64)
    off = shown - exact(ramp, 128, 128)
    assert shown.shape == (64, 220, 3) and np.abs(off).max() <= 0.5 + 1e-9, "each grey, rounded to the nearest level"
    assert abs(off.mean()) < 0.05, f"no bias: {off.mean():+.2f} level"
    for cb, cr in ((90, 200), (200, 90), (60, 60), (128, 240)):                 # flat colours: the matrix
        flat = np.full((64, 220), 120, np.uint8)
        off = rgb_of(frame_of(flat, cb, cr)).astype(np.float64) - exact(flat, cb, cr)
        assert np.abs(off).max() <= 1.0, f"Cb {cb}, Cr {cr}: {np.abs(off).max():.1f} levels off"

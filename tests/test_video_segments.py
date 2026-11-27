"""The video encoded in segments by two hardware encoders that take turns.

What must hold: the file is the same whether the two work at the same time or one after
the other; it is a sound H.264 stream (a key frame at the head of every segment, times in
order, every picture at its own time, the sound untouched); and when something fails,
nothing is left behind.
"""
import threading

import av
import pytest
import torch

import video_clips as clips
from hdr_blaster.engine.pipeline import RenderSettings
from hdr_blaster.video import job, writer
from hdr_blaster.video.job import VideoSettings, convert_video, make_engine
from hdr_blaster.video.reader import probe
from hdr_blaster.video.writer import VideoWriter

needs_nvenc = pytest.mark.skipif(not (torch.cuda.is_available() and writer.nvenc_available()),
                                 reason="two encoders take turns with the hardware encoder only")


@pytest.fixture(scope="module")
def card():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def video_engine(card):
    return make_engine(card)


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("segments") / "long.mp4"
    clips.make_clip(path, frames=96, size=(480, 272), audio_streams=1)
    return path


def packets_of(path):
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        rows = [(p.pts, p.dts, p.is_keyframe) for p in container.demux(video) if p.dts is not None]
    with av.open(str(path)) as container:
        sound = sum(p.size for s in container.streams.audio for p in container.demux(s) if p.size)
        streams = len(container.streams.audio)
    return rows, sound, streams


@needs_nvenc
@pytest.mark.parametrize("extra", [{}, dict(frame_rate=13.0), dict(frame_rate=13.0, motion_blur=True), dict(out_height=136)],
                         ids=["every frame", "fewer frames", "motion blur", "smaller"])
def test_both_encoders_at_once_or_one_at_a_time_the_same_file(extra, long_clip, card, video_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "SEGMENT_FRAMES", 11)                  # many short segments in a short clip
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0), **extra)
    turns = []
    real = VideoWriter._work

    def watched(self, turn, segment, item):
        if item is not writer._END and item is not writer._FLUSH:
            turns.append((segment.index, threading.current_thread().name))
        return real(self, turn, segment, item)

    monkeypatch.setattr(VideoWriter, "_work", watched)
    one = convert_video(long_clip, tmp_path / "one.mp4", chosen, card, engine=video_engine, reference=True)
    in_turns, turns[:] = list(turns), []
    two = convert_video(long_clip, tmp_path / "two.mp4", chosen, card, engine=video_engine)
    assert (tmp_path / "two.mp4").read_bytes() == (tmp_path / "one.mp4").read_bytes(), "the same file, byte for byte"
    assert one.frames == two.frames
    assert len({name for _, name in in_turns}) == 1, "the reference way: one thread, the encoders in turns"
    threads = {index % 2: name for index, name in turns}
    assert threads == {0: "hdr-blaster-encoder-1", 1: "hdr-blaster-encoder-2"}, "each encoder in a thread of its own"
    assert max(index for index, _ in turns) + 1 == -(-two.frames // 11)

    rows, sound, streams = packets_of(tmp_path / "two.mp4")
    source_rows, source_sound, _ = packets_of(long_clip)
    assert len(rows) == two.frames
    assert [n for n, row in enumerate(rows) if row[2]] == list(range(0, two.frames, 11)), "a key frame at each segment's head"
    assert all(b[1] > a[1] for a, b in zip(rows, rows[1:])), "decoding times in order"
    assert all(pts >= dts for pts, dts, _ in rows), "nothing shown before it is decoded"
    shown = sorted(pts for pts, _, _ in rows)
    if not extra.get("frame_rate"):
        assert shown == sorted(pts for pts, _, _ in source_rows), "every picture at its own time"
    assert len(set(shown)) == len(shown)
    assert streams == 1 and sound == source_sound, "the sound as it was"


@needs_nvenc
def test_a_video_shorter_than_a_segment_needs_one_encoder(long_clip, card, video_engine, tmp_path, monkeypatch):
    made = []
    real = VideoWriter._new_encoder
    monkeypatch.setattr(VideoWriter, "_new_encoder", lambda self: made.append(1) or real(self))
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0))
    report = convert_video(long_clip, tmp_path / "short.mp4", chosen, card, engine=video_engine)
    assert report.frames == 96 < writer.SEGMENT_FRAMES and made == [], "the second encoder is not even opened"
    rows, _, _ = packets_of(tmp_path / "short.mp4")
    assert [n for n, row in enumerate(rows) if row[2]] == [0]


@needs_nvenc
@pytest.mark.parametrize("extra", [{}, dict(frame_rate=13.0), dict(frame_rate=13.0, motion_blur=True)],
                         ids=["every frame", "fewer frames (pictures on the card)", "motion blur"])
def test_a_card_that_grants_a_single_encoder_converts_with_it_alone(extra, long_clip, card, video_engine, tmp_path,
                                                                   monkeypatch, caplog):
    """A second encoder is not always to be had: a driver that grants few of them, other
    programs holding them (a recording in progress). The conversion must not fail for it:
    the first encoder goes on alone, one stream for the whole video."""
    import logging

    monkeypatch.setattr(writer, "SEGMENT_FRAMES", 11)
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0), **extra)
    convert_video(long_clip, tmp_path / "two.mp4", chosen, card, engine=video_engine)

    asked, used = [], []
    real_work = VideoWriter._work

    def refused(self):
        asked.append(1)
        raise RuntimeError("OpenEncodeSessionEx failed: out of memory (10): no encoder left on this card")

    def watched(self, turn, segment, item):
        if item is not writer._END and item is not writer._FLUSH:
            used.append(turn)
        return real_work(self, turn, segment, item)

    monkeypatch.setattr(VideoWriter, "_new_encoder", refused)
    monkeypatch.setattr(VideoWriter, "_work", watched)
    caplog.set_level(logging.WARNING)
    before = threading.active_count()
    alone = convert_video(long_clip, tmp_path / "alone.mp4", chosen, card, engine=video_engine)
    assert set(used) == {0} and len(asked) >= 1, "every picture went to the first encoder"
    assert caplog.text.count("second encoder could not be opened") == len(asked), "said once for each file written"
    assert threading.active_count() <= before
    plain = convert_video(long_clip, tmp_path / "alone_plain.mp4", chosen, card, engine=video_engine, reference=True)
    assert (tmp_path / "alone.mp4").read_bytes() == (tmp_path / "alone_plain.mp4").read_bytes(), \
        "on such a card too, the fast way and the reference way write the same file"

    rows, sound, streams = packets_of(tmp_path / "alone.mp4")
    with_two, sound_two, _ = packets_of(tmp_path / "two.mp4")
    assert alone.frames == plain.frames == len(rows) == len(with_two)
    assert sorted(pts for pts, _, _ in rows) == sorted(pts for pts, _, _ in with_two), "the same pictures at the same times"
    assert [n for n, row in enumerate(rows) if row[2]] == list(range(0, len(rows), 11)), "key frames where they always are"
    assert all(b[1] > a[1] for a, b in zip(rows, rows[1:])) and all(pts >= dts for pts, dts, _ in rows)
    assert streams == 1 and sound == sound_two
    with av.open(str(tmp_path / "alone.mp4")) as container:
        assert sum(1 for _ in container.decode(container.streams.video[0])) == alone.frames, "and it decodes to the end"


@needs_nvenc
def test_an_encoder_that_stops_in_the_middle_stops_the_conversion_and_leaves_nothing(long_clip, card, video_engine,
                                                                                   tmp_path, monkeypatch):
    fails = "second encoder stops in the middle"
    monkeypatch.setattr(writer, "SEGMENT_FRAMES", 11)
    real = VideoWriter._new_encoder

    def broken(self):
        encoder, count = real(self), [0]

        class Stops:
            def encode(self, picture):
                count[0] += 1
                if count[0] > 5:
                    raise RuntimeError("the card stopped encoding")
                return encoder.encode(picture)

        return Stops()

    monkeypatch.setattr(VideoWriter, "_new_encoder", broken)
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0))
    before = threading.active_count()
    with pytest.raises(RuntimeError, match="stopped encoding"):
        convert_video(long_clip, tmp_path / "broken.mp4", chosen, card, engine=video_engine)
    assert list(tmp_path.iterdir()) == [], "no file, not even a partial one"
    assert threading.active_count() <= before, "and no thread left running"


def test_the_software_encoder_keeps_one_encoder_for_the_whole_stream(long_clip, tmp_path):
    info = probe(long_clip)
    out = VideoWriter(tmp_path / "soft.mp4", info, encoder="libx264", sessions=2)
    try:
        assert out.sessions == 1 and out._turns == 1 and out._workers == []
    finally:
        out.abort()
    assert list(tmp_path.iterdir()) == []


def test_the_software_encoder_is_not_cut_into_segments(long_clip, tmp_path, monkeypatch):
    """A video longer than a segment, encoded by the processor: one encoder, one stream, as before."""
    monkeypatch.setattr(writer, "SEGMENT_FRAMES", 11)
    monkeypatch.setattr(writer, "pick_encoder", lambda: "libx264")
    made = []
    monkeypatch.setattr(VideoWriter, "_new_encoder", lambda self: made.append(1))
    cpu = torch.device("cpu")
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0))
    report = convert_video(long_clip, tmp_path / "soft.mp4", chosen, cpu, engine=make_engine(cpu))
    rows, _, _ = packets_of(tmp_path / "soft.mp4")
    assert report.encoder == "libx264" and report.frames == len(rows) == 96 and made == []
    assert [n for n, row in enumerate(rows) if row[2]] == [0], "its own key frames, not one every segment"
    assert all(b[1] > a[1] for a, b in zip(rows, rows[1:])) and all(pts >= dts for pts, dts, _ in rows)


@needs_nvenc
def test_stopping_in_the_middle_ends_both_encoders(long_clip, card, video_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "SEGMENT_FRAMES", 11)
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0))
    seen = []

    def progress(stage, done, total):
        seen.append((stage, done))

    def cancelled():
        return any(stage == "render" and done >= 40 for stage, done in seen)

    before = threading.active_count()
    with pytest.raises(job.Cancelled):
        convert_video(long_clip, tmp_path / "stopped.mp4", chosen, card, progress, cancelled, engine=video_engine)
    assert list(tmp_path.iterdir()) == [] and threading.active_count() <= before

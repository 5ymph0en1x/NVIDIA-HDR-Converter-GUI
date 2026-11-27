"""Pictures handed to the hardware encoder without leaving the graphics card.

The encoder reads the picture where the card computed it, instead of the picture being
brought back to main memory and sent to the card again. It must change nothing: the same
packets, byte for byte, as the hand-over through memory, which stays the reference.
"""
import hashlib
import logging

import pytest
import torch

import video_clips as clips
from hdr_blaster.engine.pipeline import RenderSettings
from hdr_blaster.video import job, writer
from hdr_blaster.video.job import VideoSettings, convert_video, make_engine
from hdr_blaster.video.reader import probe
from hdr_blaster.video.writer import VideoWriter

needs_nvenc = pytest.mark.skipif(not (torch.cuda.is_available() and writer.nvenc_available()),
                                 reason="only the hardware encoder reads pictures on the card")


@pytest.fixture(scope="module")
def card():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def video_engine(card):
    return make_engine(card)


@pytest.fixture(scope="module")
def wide_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("card") / "wide.mp4"
    clips.make_clip(path, frames=30, size=(480, 272), audio_streams=1)
    return path


def pictures(width, height, card, count=12):
    g = torch.Generator(device=card).manual_seed(4)
    base = torch.randint(16, 236, (height, width), dtype=torch.uint8, device=card, generator=g)
    chroma = torch.randint(60, 200, (2, height // 2, width // 2), dtype=torch.uint8, device=card, generator=g)
    for i in range(count):
        yield torch.roll(base, shifts=(i, 3 * i), dims=(0, 1)), torch.roll(chroma[0], i, 1), torch.roll(chroma[1], i, 0)


def encoded(path, info, card, device, size):
    out = VideoWriter(path, info, size=size, max_bit_rate=8_000_000, device=device)
    on_card = out.on_card
    for i, (y, u, v) in enumerate(pictures(size[0], size[1], card)):
        out.write(i * 3000, y, u, v)
    out.close()
    return on_card, path.read_bytes()


@needs_nvenc
@pytest.mark.parametrize("size", [(480, 272), (1920, 1080), (2560, 1440), (1280, 720)], ids=str)
def test_the_encoder_gives_the_same_file_from_the_card_as_from_memory(size, wide_clip, card, tmp_path):
    info = probe(wide_clip)
    through_memory, plain = encoded(tmp_path / "memory.mp4", info, card, None, size)
    direct, fast = encoded(tmp_path / "card.mp4", info, card, card, size)
    assert through_memory is False and direct is True, "the second one really read the pictures on the card"
    assert fast == plain, "the same file, byte for byte"


@needs_nvenc
@pytest.mark.parametrize("size", [(1918, 1082), (642, 362), (1916, 1078), (854, 480), (496, 270)], ids=str)
def test_sizes_the_encoder_misreads_on_the_card_are_not_even_tried(size, wide_clip, card, tmp_path, monkeypatch):
    """With rows that are not a round number of bytes the encoder refuses the picture or,
    worse, misreads the card's memory and leaves the card in error for the whole program.
    Those sizes go through memory, and nothing is attempted on the card."""
    monkeypatch.setattr(writer, "_packets_of", lambda *args: pytest.fail("a trial was made on the card"))
    info = probe(wide_clip)
    direct, fast = encoded(tmp_path / "card.mp4", info, card, card, size)
    assert direct is False
    _, plain = encoded(tmp_path / "memory.mp4", info, card, None, size)
    assert fast == plain
    assert float(torch.arange(4, device=card).sum()) == 6.0, "the card still computes"


@needs_nvenc
def test_pictures_stay_on_the_card_for_a_conversion_and_the_file_is_the_reference_one(wide_clip, card, video_engine,
                                                                                    tmp_path, monkeypatch):
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0), frame_rate=13.0)
    made = {"card": 0, "memory": 0}
    real = VideoWriter.frame

    def counted(self, pts, y, u, v):
        picture = real(self, pts, y, u, v)
        made["card" if picture.format.name == "cuda" else "memory"] += 1
        return picture

    monkeypatch.setattr(VideoWriter, "frame", counted)
    convert_video(wide_clip, tmp_path / "plain.mp4", chosen, card, engine=video_engine, reference=True)
    through_memory = made["memory"]
    assert made["card"] == 0 and through_memory >= 10, "the reference way hands every picture over through memory"
    convert_video(wide_clip, tmp_path / "fast.mp4", chosen, card, engine=video_engine)
    assert made == {"card": through_memory, "memory": through_memory}, \
        "with fewer frames written than read, the fast way leaves every one on the card"
    assert (tmp_path / "fast.mp4").read_bytes() == (tmp_path / "plain.mp4").read_bytes()
    # every frame written: through memory is the faster way (measured), and the one taken
    every = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0))
    made.update(card=0, memory=0)
    convert_video(wide_clip, tmp_path / "every.mp4", every, card, engine=video_engine)
    assert made["card"] == 0 and made["memory"] >= 30


@needs_nvenc
@pytest.mark.parametrize("fault", ["other packets", "cannot be done"])
def test_a_card_that_encodes_otherwise_from_its_own_memory_goes_through_main_memory(fault, wide_clip, card, tmp_path,
                                                                                 monkeypatch, caplog):
    """Like every short cut here: used only once it has been seen to give the same thing."""
    writer._card_checked.clear()
    real = writer._packets_of

    def trial(width, height, on_card):
        if on_card and fault == "cannot be done":
            raise RuntimeError("this driver does not take pictures from the card")
        data = real(width, height, on_card)
        return data + b"x" if on_card else data

    monkeypatch.setattr(writer, "_packets_of", trial)
    caplog.set_level(logging.INFO)
    info = probe(wide_clip)
    out = VideoWriter(tmp_path / "wary.mp4", info, max_bit_rate=8_000_000, device=card)
    try:
        assert out.on_card is False
        assert "through main memory" in caplog.text
        for i, (y, u, v) in enumerate(pictures(info.width, info.height, card, 4)):
            assert out.frame(i * 3000, y, u, v).format.name == "yuv420p"
    finally:
        out.abort()
        writer._card_checked.clear()


def test_without_a_card_or_with_the_software_encoder_pictures_go_through_memory(wide_clip, tmp_path):
    info = probe(wide_clip)
    for kwargs in (dict(device=torch.device("cpu")), dict(device=None), dict(encoder="libx264", device="cuda")):
        out = VideoWriter(tmp_path / "soft.mp4", info, max_bit_rate=8_000_000, **kwargs)
        try:
            assert out.on_card is False
        finally:
            out.abort()

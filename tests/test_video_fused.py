"""The fused stages of a video frame: fewer operations on the graphics card, the very same bits.

The plain stages (PyTorch's own operations, one by one) are the reference. A fused stage
may only ever be used where it gives exactly what its plain twin gives, and it is the
program itself, not these tests alone, that makes sure of it on each machine.
"""
import hashlib
import logging

import pytest
import torch

import video_clips as clips
from hdr_blaster.engine.params import RenderParams
from hdr_blaster.engine.pipeline import RenderSettings, render
from hdr_blaster.video import colour, fused, job
from hdr_blaster.video.colour import FrameFormat, scrgb_to_engine
from hdr_blaster.video.job import VideoSettings, analyse_video, convert_video, make_engine

needs_card = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused stages are compiled for the graphics card")
WHITE = 300.0


@pytest.fixture(scope="module")
def card():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def video_engine(card):
    return make_engine(card)


@pytest.fixture
def chain(card):
    return fused.Chain(card)                 # a new one: nothing checked yet


def planes(height, width, card, seed=3, full_range=False, low=0, high=1023):
    """Planes of a 10-bit 4:2:0 frame holding any code value next to any other."""
    g = torch.Generator(device=card).manual_seed(seed)

    def noise(h, w):
        return torch.randint(low, high + 1, (h, w), device=card, generator=g).to(torch.int16)

    fmt = FrameFormat(width=width, height=height, full_range=full_range)
    return noise(height, width), noise(height // 2, width // 2), noise(height // 2, width // 2), fmt


def frozen(peak, card) -> RenderParams:
    return RenderParams(method="faithful", stats={"scene.peak": torch.tensor(peak, device=card)}, frozen=True)


def plain_picture(video_engine, linear, peak, card, white=WHITE):
    return render(video_engine.tone_mapper, video_engine.loader, video_engine.processor, linear,
                  RenderSettings(method="faithful", white_nits=white), params=frozen(peak, card), release_cache=False)[0]


# ------------------------------------------------------------- the same bits

@needs_card
@pytest.mark.parametrize("size", [(256, 256), (270, 480), (1080, 1920), (1082, 1918)], ids=str)
@pytest.mark.parametrize("full_range", [False, True], ids=["limited", "full range"])
def test_each_fused_stage_gives_the_bits_of_the_plain_stage(size, full_range, card, video_engine, chain):
    height, width = size
    with torch.inference_mode():
        for seed, (low, high) in enumerate(((0, 1023), (64, 940), (0, 80), (900, 1023))):
            y, u, v, fmt = planes(height, width, card, seed, full_range, low, high)
            light = colour.yuv_to_scrgb(y, u, v, fmt)
            assert fused._identical(light, chain._yuv_to_scrgb(y, u, v, fmt)), f"planes to scene light, codes {low}-{high}"
            linear = scrgb_to_engine(light)
            for peak in (0.3, 1.0, 1.7, 9.0, 33.0):
                picture = plain_picture(video_engine, linear, peak, card)
                assert fused._identical(picture, chain._faithful_srgb(linear, torch.tensor(peak, device=card), WHITE)), \
                    f"Faithful rendering, peak {peak} x white, codes {low}-{high}"
            assert fused._identical(colour.srgb_to_yuv420(picture), chain._srgb_to_yuv420(picture)), "8-bit planes"


@needs_card
def test_the_sign_of_a_zero_counts_and_so_does_the_type(card):
    zero = torch.zeros(2, 2, device=card)
    assert fused._identical(zero, zero.clone())
    assert not fused._identical(zero, -zero), "plus zero and minus zero are not the same bits"
    assert not fused._identical(zero, zero.double()) and not fused._identical(zero, zero[:1])
    assert not fused._identical((zero, zero), (zero, zero + 1))


@needs_card
@pytest.mark.parametrize("white", [80.0, 203.0, 480.0, 1000.0])
def test_the_faithful_rendering_is_the_same_for_any_white_and_any_shape_of_peak(white, card, video_engine, chain):
    with torch.inference_mode():
        y, u, v, fmt = planes(270, 480, card, seed=9)
        linear = scrgb_to_engine(colour.yuv_to_scrgb(y, u, v, fmt))
        for peak in (0.5, 2.0, 10000.0 / white):
            expected = plain_picture(video_engine, linear, peak, card, white)
            for shaped in (torch.tensor(peak, device=card), torch.tensor(peak, device=card).view(1, 1, 1, 1),
                           torch.tensor(peak, device=card, dtype=torch.float64)):
                assert fused._identical(expected, chain._faithful_srgb(linear, shaped, white))


# ------------------------------------------------- used only once checked

@needs_card
def test_a_stage_is_used_only_after_it_has_given_the_plain_bits(card, video_engine, chain, monkeypatch, caplog):
    with torch.inference_mode():
        y, u, v, fmt = planes(270, 480, card)
        picture = plain_picture(video_engine, scrgb_to_engine(colour.yuv_to_scrgb(y, u, v, fmt)), 2.0, card)
        expected = colour.srgb_to_yuv420(picture)
        calls, real = [], chain._srgb_to_yuv420
        monkeypatch.setattr(chain, "_srgb_to_yuv420", lambda srgb: calls.append(1) or real(srgb))
        caplog.set_level(logging.INFO)
        assert fused._identical(chain.srgb_to_yuv420(picture), expected)
        checked = len(calls)
        assert checked > 1, "tried on this picture and on difficult ones"
        assert "checked against the plain one" in caplog.text
        assert fused._identical(chain.srgb_to_yuv420(picture), expected) and len(calls) == checked + 1, "then used"
        # another size is another question: checked again
        other = picture[..., :200, :300].contiguous()
        assert fused._identical(chain.srgb_to_yuv420(other), colour.srgb_to_yuv420(other))
        assert len(calls) > checked + 2


@needs_card
@pytest.mark.parametrize("fault", ["one sample off", "fails"])
def test_a_stage_that_gives_other_bits_or_fails_is_never_used(fault, card, video_engine, chain, monkeypatch, caplog):
    """Another card, driver or PyTorch may round differently: the price is speed, never the picture."""
    with torch.inference_mode():
        y, u, v, fmt = planes(270, 480, card)
        expected = colour.yuv_to_scrgb(y, u, v, fmt)
        calls, real = [], chain._yuv_to_scrgb

        def wrong(*args):
            calls.append(1)
            if fault == "fails":
                raise RuntimeError("no compiler for this card")
            light = real(*args).clone()
            light[0, 1, 5, 7] = torch.nextafter(light[0, 1, 5, 7], light.new_tensor(1e9))   # the smallest step there is
            return light

        monkeypatch.setattr(chain, "_yuv_to_scrgb", wrong)
        caplog.set_level(logging.INFO)
        for _ in range(3):
            assert fused._identical(chain.yuv_to_scrgb(y, u, v, fmt), expected)
        assert len(calls) == 1, "tried once, then left alone"
        assert "is not used" in caplog.text and "plain stage goes on" in caplog.text


@needs_card
def test_what_the_fused_stages_were_not_made_for_goes_the_plain_way(card, video_engine, chain, monkeypatch):
    assert fused.chain_for("cpu") is None, "nothing is compiled for the processor"
    assert fused.chain_for(card) is fused.chain_for("cuda"), "one set of stages for a card"
    used = []
    for name in ("_yuv_to_scrgb", "_faithful_srgb", "_srgb_to_yuv420"):
        monkeypatch.setattr(chain, name, lambda *args, name=name: used.append(name))
    with torch.inference_mode():
        y, u, v, fmt = planes(144, 256, card)             # too few pixels: the colour matrix is computed otherwise
        light = colour.yuv_to_scrgb(y, u, v, fmt)
        assert fused._identical(chain.yuv_to_scrgb(y, u, v, fmt), light)
        tiny = torch.rand(1, 3, 32, 64, device=card)
        def plain(scene, peak, white):
            return plain_picture(video_engine, scene, float(peak), card, white)
        fallback = chain.render(plain, tiny, torch.tensor(1.0, device=card), WHITE)
        assert torch.equal(fallback, plain(tiny, 1.0, WHITE))
        odd = torch.rand(1, 3, 35, 64, device=card)
        with pytest.raises(ValueError):
            chain.srgb_to_yuv420(odd)
        on_processor = torch.rand(1, 3, 64, 64)
        assert torch.equal(chain.srgb_to_yuv420(on_processor)[0], colour.srgb_to_yuv420(on_processor)[0])
    assert used == []


# ------------------------------------------------------------ in the engine

@needs_card
def test_render_takes_the_fused_way_for_a_faithful_frame_and_only_for_it(card, video_engine, chain, monkeypatch):
    tm, loader, processor = video_engine.tone_mapper, video_engine.loader, video_engine.processor
    used, real = [], chain._faithful_srgb
    monkeypatch.setattr(chain, "_faithful_srgb", lambda *args: used.append(1) or real(*args))
    with torch.inference_mode():
        y, u, v, fmt = planes(270, 480, card)
        linear = scrgb_to_engine(colour.yuv_to_scrgb(y, u, v, fmt))
        plain_settings = RenderSettings(method="faithful", white_nits=WHITE)
        for peak in (2.0, 2.0, 0.7):
            expected = render(tm, loader, processor, linear, plain_settings, params=frozen(peak, card), release_cache=False)
            got = render(tm, loader, processor, linear, plain_settings, params=frozen(peak, card), release_cache=False,
                         fast=chain)
            assert fused._identical(expected[0].contiguous(), got[0].contiguous())
            assert got[1].method == "faithful" and got[1].frozen
        assert len(used) > 2, "checked, then used"
        del used[:]
        # sharpening, AI colour, another look, a picture that is measured: all the plain way
        punchy = RenderParams(method="punchy", stats={"scene.peak": torch.tensor(2.0, device=card)}, frozen=True)
        for settings, params in ((RenderSettings(method="faithful", edge_strength=25.0, white_nits=WHITE), None),
                                 (RenderSettings(method="punchy", white_nits=WHITE), punchy),
                                 (plain_settings, None)):
            first = render(tm, loader, processor, linear, settings, release_cache=False)[1].freeze() if params is None else params
            for given in ((first,) if params is not None else (first, None)):
                expected = render(tm, loader, processor, linear, settings, params=given, release_cache=False)[0]
                got = render(tm, loader, processor, linear, settings, params=given, release_cache=False, fast=chain)[0]
                assert fused._identical(expected.contiguous(), got.contiguous())
        assert len(used) == 1, "only the frozen Faithful frame of the last settings went the fused way"
        with pytest.raises(Exception, match="scene.peak"):
            render(tm, loader, processor, linear, plain_settings, release_cache=False, fast=chain,
                   params=RenderParams(method="faithful", stats={}, frozen=True))


@needs_card
def test_half_precision_keeps_the_plain_rendering(card, chain, monkeypatch):
    """With "Half precision" the picture goes through 16 bits on its way: not what the fused stage computes."""
    half = make_engine(card, use_fp16=True)
    used = []
    monkeypatch.setattr(chain, "_faithful_srgb", lambda *args: used.append(1))
    with torch.inference_mode():
        y, u, v, fmt = planes(270, 480, card)
        linear = scrgb_to_engine(colour.yuv_to_scrgb(y, u, v, fmt))
        settings = RenderSettings(method="faithful", white_nits=WHITE)
        expected = render(half.tone_mapper, half.loader, half.processor, linear, settings, params=frozen(2.0, card),
                          release_cache=False)[0]
        got = render(half.tone_mapper, half.loader, half.processor, linear, settings, params=frozen(2.0, card),
                     release_cache=False, fast=chain)[0]
    assert fused._identical(expected.contiguous(), got.contiguous()) and used == []


# -------------------------------------------------------- a whole conversion

@pytest.fixture(scope="module")
def wide_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("fused") / "wide.mp4"
    clips.make_clip(path, frames=16, size=(480, 272), audio_streams=1)      # enough pixels for every fused stage
    return path


def handed_to_the_encoder(monkeypatch):
    seen, real = {}, job.VideoWriter.frame

    def spy(self, pts, y, u, v):
        seen[pts] = hashlib.sha256(b"".join(p.cpu().numpy().tobytes() for p in (y, u, v))).hexdigest()
        return real(self, pts, y, u, v)

    monkeypatch.setattr(job.VideoWriter, "frame", spy)
    return seen


@needs_card
@pytest.mark.parametrize("rendering, extra", [("faithful", {}), ("faithful", dict(frame_rate=12.0, motion_blur=True)),
                                               ("faithful", dict(out_height=136)), ("golden", {}),
                                               ("faithful", dict(pre_gamma=1.2, exposure=0.8))],
                         ids=["faithful", "motion blur", "smaller", "golden", "gamma and exposure"])
def test_a_conversion_with_the_fused_stages_writes_the_pictures_of_the_plain_one(rendering, extra, wide_clip, card,
                                                                               video_engine, tmp_path, monkeypatch):
    chosen = VideoSettings(render=RenderSettings(method=rendering, white_nits=WHITE), **extra)
    fresh = fused.Chain(card)
    monkeypatch.setattr(fused, "chain_for", lambda device: fresh)
    used = {}
    for name in ("_yuv_to_scrgb", "_faithful_srgb", "_srgb_to_yuv420"):
        def counted(*args, name=name, real=getattr(fresh, name)):
            used[name] = used.get(name, 0) + 1
            return real(*args)
        monkeypatch.setattr(fresh, name, counted)

    plain = analyse_video(wide_clip, chosen, card, engine=video_engine, reference=True)
    assert used == {}, "the reference way never comes near the fused stages"
    fast = analyse_video(wide_clip, chosen, card, engine=video_engine)
    assert used.get("_yuv_to_scrgb", 0) >= 16, "the analysis reads the scene's light the fused way"
    assert (fast.method, fast.cuts, fast.pts) == (plain.method, plain.cuts, plain.pts)
    for a, b in zip(fast.params, plain.params):
        assert a.stats.keys() == b.stats.keys()
        assert all(torch.equal(a.stats[k], b.stats[k]) for k in a.stats), "the same measurements, to the bit"

    seen = handed_to_the_encoder(monkeypatch)
    convert_video(wide_clip, tmp_path / "plain.mp4", chosen, card, engine=video_engine, reference=True)
    reference, before = dict(seen), dict(used)
    seen.clear()
    convert_video(wide_clip, tmp_path / "fast.mp4", chosen, card, engine=video_engine)
    assert seen == reference and len(seen) >= 4, "every picture handed to the encoder is the same, byte for byte"
    assert used["_srgb_to_yuv420"] - before.get("_srgb_to_yuv420", 0) >= len(seen), "by the fused stages"
    assert (used.get("_faithful_srgb", 0) > before.get("_faithful_srgb", 0)) == (rendering == "faithful")
    assert all(fresh._verdicts.values()), f"every stage met was found identical: {fresh._verdicts}"
    assert (tmp_path / "fast.mp4").read_bytes() == (tmp_path / "plain.mp4").read_bytes(), "and so is the file"


@needs_card
def test_a_card_whose_fused_stages_disagree_converts_the_plain_way(wide_clip, card, video_engine, tmp_path, monkeypatch,
                                                                 caplog):
    """The whole conversion on a machine that rounds otherwise: the same file, with a line in the log."""
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=WHITE))
    broken = fused.Chain(card)
    for name in ("_yuv_to_scrgb", "_faithful_srgb"):
        def off(*args, real=getattr(broken, name)):
            out = real(*args).clone()
            out[0, 0, 3, 3] += 1e-3
            return out
        monkeypatch.setattr(broken, name, off)
    monkeypatch.setattr(broken, "_srgb_to_yuv420", lambda srgb: (_ for _ in ()).throw(RuntimeError("cannot compile")))
    convert_video(wide_clip, tmp_path / "plain.mp4", chosen, card, engine=video_engine, reference=True)
    monkeypatch.setattr(fused, "chain_for", lambda device: broken)
    caplog.set_level(logging.WARNING)
    convert_video(wide_clip, tmp_path / "wary.mp4", chosen, card, engine=video_engine)
    assert (tmp_path / "wary.mp4").read_bytes() == (tmp_path / "plain.mp4").read_bytes()
    assert not any(broken._verdicts.values()) and len(broken._verdicts) == 3
    assert caplog.text.count("is not used") == 3

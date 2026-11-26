"""What the interface previews, and the video analysis it shares with the conversion."""
import av
import numpy as np
import pytest
import torch

import video_clips as clips
from conftest import SAMPLES
from hdr_blaster import preview
from hdr_blaster.engine.pipeline import RenderSettings
from hdr_blaster.video import job
from hdr_blaster.video.job import (VideoSettings, analyse_video, convert_video, make_engine,
                                   render_video)
from hdr_blaster.video.reader import grab_frame


@pytest.fixture(scope="module")
def video_engine(device):
    return make_engine(device)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("preview") / "game.mp4"
    return path, clips.make_clip(path, frames=30, audio_streams=0)


def settings(method="punchy", **kwargs):
    return VideoSettings(render=RenderSettings(method=method, edge_strength=25.0), **kwargs)


def decoded(path, index):
    """Frame ``index`` of a file as float RGB (3, H, W) in [0, 1]."""
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                array = frame.to_ndarray(format="rgb24")
                return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    raise IndexError(index)


# ------------------------------------------------------------------ grab_frame

def test_grab_frame_returns_the_frame_shown_at_that_time(clip, device):
    path, pts = clip
    for wanted in (0, 7, 18, 29):
        frame = grab_frame(path, pts[wanted] / 90000, device)
        assert frame.pts == pts[wanted]
        assert frame.y.shape == (144, 256)
    between = grab_frame(path, (pts[7] + 100) / 90000, device)
    assert between.pts == pts[8], "a time between two frames gives the next one"


def test_grab_frame_past_the_end_gives_the_last_frame(clip, device):
    path, pts = clip
    assert grab_frame(path, 3600.0, device).pts == pts[-1]
    assert grab_frame(path, -5.0, device).pts == pts[0]


# ----------------------------------------------------------- analyse + render

def test_analyse_then_render_equals_convert(clip, device, video_engine, tmp_path):
    path, pts = clip
    convert_video(path, tmp_path / "one.mp4", settings(), device, engine=video_engine)
    analysis = analyse_video(path, settings(), device, engine=video_engine)
    report = render_video(analysis, tmp_path / "two.mp4", settings(), device, engine=video_engine)
    assert analysis.pts == pts and analysis.method == "punchy" and len(analysis.params) == 30
    assert all(p.frozen for p in analysis.params)
    one, two = clips.describe(tmp_path / "one.mp4"), clips.describe(tmp_path / "two.mp4")
    assert report.frames == 30 and two["pts_s"] == one["pts_s"]
    assert two["mean_luma"] == pytest.approx(one["mean_luma"], abs=0.6)


def test_one_analysis_serves_several_encodes_without_measuring_again(clip, device, video_engine,
                                                                    tmp_path, monkeypatch):
    path, _ = clip
    analysis = analyse_video(path, settings(), device, engine=video_engine)
    calls, real = [], job.render

    def spy(tone_mapper, loader, processor, linear, render_settings, params=None, **kwargs):
        calls.append(params is not None and params.frozen)
        return real(tone_mapper, loader, processor, linear, render_settings, params=params, **kwargs)

    monkeypatch.setattr(job, "render", spy)
    render_video(analysis, tmp_path / "high.mp4", settings(quality="high"), device, engine=video_engine)
    render_video(analysis, tmp_path / "small.mp4", settings(quality="compact", out_height=90), device,
                 engine=video_engine)
    assert calls and all(calls), "every render after the analysis replays recorded parameters"
    assert clips.describe(tmp_path / "small.mp4")["height"] == 90


def test_an_analysis_made_with_other_picture_settings_is_refused(clip, device, video_engine, tmp_path):
    path, _ = clip
    analysis = analyse_video(path, settings(method="punchy"), device, engine=video_engine)
    assert analysis.matches(settings(method="punchy", quality="compact"))
    assert not analysis.matches(settings(method="hable"))
    assert not analysis.matches(settings(exposure=1.5))
    with pytest.raises(ValueError, match="analyse again"):
        render_video(analysis, tmp_path / "x.mp4", settings(method="hable"), device, engine=video_engine)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------- preview

def test_video_preview_with_the_analysis_is_the_frame_of_the_converted_file(clip, device, video_engine, tmp_path):
    """The promise of the interface: what you see is what the file will contain."""
    path, pts = clip
    analysis = analyse_video(path, settings(), device, engine=video_engine)
    render_video(analysis, tmp_path / "out.mp4", settings(), device, engine=video_engine)
    for index in (3, 20):
        source = preview.load_video_frame(path, pts[index] / 90000, device, max_side=4096)
        assert source.pts == pts[index] and analysis.index_of(source.pts) == index
        shown = preview.render_preview(video_engine, source, settings().render,
                                       params=analysis.params[index])[0].cpu()
        written = decoded(tmp_path / "out.mp4", index)
        assert float((shown - written).abs().mean()) < 0.03


def test_video_source_view_shows_the_scene_at_its_real_luminance(clip, device, video_engine):
    """The left picture is the HDR frame decoded as its tags say, with the window's white
    standing for the given luminance: not the washed-out picture of a careless player."""
    path, pts = clip
    source = preview.load_video_frame(path, pts[10] / 90000, device, max_side=128, white_nits=400.0)
    assert source.kind == "video" and source.size == (72, 128) and source.before.shape[-2:] == (72, 128)
    assert source.linear.shape[-2:] == (144, 256), "the frame itself is kept at its real size"
    source.size = (144, 256)                                # shown pixel for pixel
    assert (source.width, source.height) == (256, 144) and source.duration_s > 0.4
    assert source.white_nits == 400.0 and 500.0 < source.peak_nits < 2000.0
    nits = source.linear * 800.0
    expected = preview._srgb(nits / 400.0)
    assert torch.allclose(source.before, expected, atol=1e-6)
    after = preview.render_preview(video_engine, source, settings().render)
    assert source.before.shape == after.shape


def test_reference_view_places_white_where_the_screen_does(device):
    grey = torch.full((1, 3, 4, 4), 100.0 / 800.0, device=device)          # 100 nits, engine units
    at_100 = preview.reference_view(grey, 100.0)
    at_400 = preview.reference_view(grey, 400.0)
    assert torch.allclose(at_100, torch.ones_like(at_100)), "100 nits is the window's white"
    quarter = 1.055 * 0.25 ** (1 / 2.4) - 0.055
    assert torch.allclose(at_400, torch.full_like(at_400, quarter), atol=1e-5), "a quarter of white, sRGB-encoded"
    bright = torch.full((1, 3, 4, 4), 1000.0 / 800.0, device=device)
    assert float(preview.reference_view(bright, 400.0).min()) == pytest.approx(1.0, abs=1e-6), "brighter than white is clipped"
    assert float(preview.reference_view(torch.zeros_like(grey), 400.0).max()) == 0.0


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_capture_source_view_follows_the_screens_white(device, video_engine):
    """With white taken as 80 nits a capture came out about six times too bright on a
    screen whose SDR white is 480 nits: every tone must scale with the white instead."""
    at_80 = preview.load_image(SAMPLES[0], device, max_side=640, white_nits=80.0)
    at_480 = preview.load_image(SAMPLES[0], device, max_side=640, white_nits=480.0)
    assert at_480.kind == "image" and at_480.before.shape[-2:] == (360, 640)
    assert at_480.linear.shape[-2:] == (1440, 2560), "the capture itself is kept at its real size"
    assert (at_480.width, at_480.height) == (2560, 1440)
    assert 800.0 < at_480.peak_nits < 1100.0, "this capture peaks near 1000 nits"
    assert torch.equal(at_80.linear, at_480.linear), "the picture itself does not depend on the screen"
    assert float(at_480.before.mean()) < float(at_80.before.mean()) - 0.15, "clearly darker, as in the game"
    dark = at_80.before.amax(1) < 0.5                       # tones that neither view clips
    ratio = preview._srgb_decode(at_480.before)[:, 1][dark] / preview._srgb_decode(at_80.before)[:, 1][dark].clamp(min=1e-6)
    assert float(ratio.median()) == pytest.approx(80.0 / 480.0, rel=0.02), "six times less light, exactly"
    at_480.size = (1440, 2560)                              # pixel for pixel
    nits = at_480.linear * 800.0
    clipped = (at_480.before.amax(1) >= 0.999).float().mean()
    above_white = (nits.amax(1) >= 480.0).float().mean()
    assert float(clipped) == pytest.approx(float(above_white), abs=0.001)
    at_480.size = (360, 640)
    after = preview.render_preview(video_engine, at_480, RenderSettings(method=None, edge_strength=25.0))
    assert after.shape == at_480.before.shape and 0.05 < float(after.mean()) < 0.8


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_image_preview_is_the_full_size_render_brought_to_the_display(device, video_engine):
    """The preview is the converted picture itself, resampled once for the window."""
    from hdr_blaster.engine.pipeline import render

    source = preview.load_image(SAMPLES[0], device, max_side=640)
    shown = preview.render_preview(video_engine, source, RenderSettings(method="punchy"))
    full, error, _ = video_engine.loader.load_jxr(str(SAMPLES[0]))
    assert full is not None, error
    exact, _ = render(video_engine.tone_mapper, video_engine.loader, video_engine.processor, full,
                      RenderSettings(method="punchy"))
    assert torch.equal(shown, preview.for_display(exact, shown.shape[-2:]))
    exact_small = torch.nn.functional.interpolate(exact, size=shown.shape[-2:], mode="area")
    assert float((shown - exact_small).abs().mean()) < 0.02


def test_to_rgb8_gives_a_contiguous_byte_image(device):
    image = torch.rand(1, 3, 5, 7, device=device) * 1.4 - 0.2
    array = preview.to_rgb8(image)
    assert array.shape == (5, 7, 3) and array.dtype == np.uint8 and array.flags["C_CONTIGUOUS"]
    assert array.min() >= 0 and array.max() == 255


# ------------------------------------------------ found by the independent review

@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
@pytest.mark.parametrize("max_side", [640, 1280])
def test_auto_preview_of_a_capture_is_the_picture_the_conversion_writes(device, video_engine, sample, max_side):
    """In Auto the operator used to be chosen on the reduced picture: on three captures out
    of four it was not the one the conversion chose, and the preview showed another image."""
    from hdr_blaster.engine.pipeline import render, resolve_method

    settings = RenderSettings(method=None, edge_strength=25.0)
    source = preview.load_image(sample, device, max_side=max_side)
    shown = preview.render_preview(video_engine, source, settings)
    full, error, _ = video_engine.loader.load_jxr(str(sample))
    assert full is not None, error
    assert source.method == resolve_method(video_engine.tone_mapper, full, None)
    exact, _ = render(video_engine.tone_mapper, video_engine.loader, video_engine.processor, full, settings)
    # reduce the converted picture the way light adds up: in linear light, not on sRGB values
    exact_small = preview._srgb(torch.nn.functional.interpolate(preview._srgb_decode(exact), size=shown.shape[-2:],
                                                                mode="area"))
    assert float((shown - exact_small).abs().mean()) < 0.015


def test_a_container_without_a_frame_count_still_reports_one(tmp_path):
    """MKV stores no frame count: progress stayed at 0 % for the whole conversion."""
    from hdr_blaster.video.reader import probe

    src = tmp_path / "clip.mp4"
    clips.make_clip(src, frames=30, audio_streams=0)
    mkv = tmp_path / "clip.mkv"
    with av.open(str(src)) as inp, av.open(str(mkv), "w") as out:
        stream = out.add_stream_from_template(inp.streams.video[0])
        for packet in inp.demux(inp.streams.video[0]):
            if packet.dts is not None:
                packet.stream = stream
                out.mux(packet)
    info = probe(mkv)
    assert 15 <= info.frames <= 60, "an estimate from duration and nominal frame rate"


# ------------------------------------------------------------- file metadata

@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_a_capture_says_it_is_floating_point_scrgb(sample):
    from hdr_blaster.io.jxr_info import check_hdr_capture, read_jxr_info

    info = read_jxr_info(sample)
    assert (info.width, info.height) == (2560, 1440)
    assert info.is_float and "float" in info.pixel_format and info.has_alpha
    assert not info.has_icc_profile and not info.has_xmp, "these captures carry no other colour metadata"
    check_hdr_capture(info)


def test_an_ordinary_jxr_picture_is_not_taken_for_an_hdr_capture(tmp_path, device):
    import imagecodecs
    import numpy as np

    from hdr_blaster.io.jxr_info import UnsupportedCapture, read_jxr_info

    path = tmp_path / "photo.jxr"
    path.write_bytes(imagecodecs.jpegxr_encode(np.full((16, 16, 3), 128, dtype=np.uint8)))
    info = read_jxr_info(path)
    assert not info.is_float and (info.width, info.height) == (16, 16)
    with pytest.raises(UnsupportedCapture, match="photo.jxr is not an HDR capture"):
        preview.load_image(path, device)
    text = tmp_path / "notes.jxr"
    text.write_text("not a picture at all")
    with pytest.raises(UnsupportedCapture, match="notes.jxr is not a JPEG XR file"):
        read_jxr_info(text)


def test_the_screens_white_is_a_plausible_luminance():
    from hdr_blaster import display

    nits, origin = display.sdr_white_nits()
    assert origin in ("display", "reference")
    assert 40.0 <= nits <= 2000.0
    if origin == "reference":
        assert nits == display.REFERENCE_WHITE_NITS == 203.0


# ------------------------------------------------------ source and result agree

@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_the_faithful_result_matches_the_source_view_next_to_it(device, video_engine):
    """What the divider shows: on either side of it, the same picture wherever the scene
    is below white. With Auto the result's shadows were two to three times brighter."""
    source = preview.load_image(SAMPLES[0], device, max_side=4096, white_nits=480.0)      # pixel for pixel
    # the settings carry another white on purpose: the preview must use the source's own
    after = preview.render_preview(video_engine, source,
                                   RenderSettings(method="faithful", edge_strength=25.0, white_nits=203.0))
    nits = (source.linear * 800.0).amax(1)
    below = nits < 250.0
    difference = (after - source.before).abs().amax(1)
    assert float(difference[below].median()) < 0.5 / 255
    assert float(difference[below].float().mean()) < 2.0 / 255, "only the sharpened edges differ"
    assert source.method == "faithful"
    auto = preview.render_preview(video_engine, source, RenderSettings(method=None, edge_strength=25.0))
    dark = nits < 10.0
    moved = float((auto - source.before)[:, :, dark[0]].abs().mean())
    assert moved < 1.0 / 255, "Auto-select keeps the dark zones too (it used to lift or crush them)"


# ------------------------------------------- Advanced never touches the source

@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_exposure_and_gamma_change_the_result_and_never_the_source_view(device, video_engine):
    """Reported: moving Exposure or Gamma in Advanced changed the left picture too. The
    left picture is the original; only the SDR result is the user's to shape."""
    from hdr_blaster.engine.colour import adjust_scene
    from hdr_blaster.io.jxr import OptimizedJXRLoader

    plain = preview.load_image(SAMPLES[0], device, max_side=320, white_nits=480.0)
    pushed = preview.load_image(SAMPLES[0], device, max_side=320, pre_gamma=1.4, exposure=1.8, white_nits=480.0)
    assert torch.equal(pushed.before, plain.before) and torch.equal(pushed.linear, plain.linear)
    assert pushed.peak_nits == plain.peak_nits
    assert (pushed.pre_gamma, pushed.exposure) == (1.4, 1.8)
    settings = RenderSettings(method="faithful", edge_strength=0.0)
    untouched = preview.render_preview(video_engine, plain, settings)
    shaped = preview.render_preview(video_engine, pushed, settings)
    assert float((shaped - untouched).abs().mean()) > 0.02, "the result follows the sliders"
    # ...and it is the picture the conversion writes: the loader applies the same arithmetic.
    loader = OptimizedJXRLoader(device)
    loader.selected_pre_gamma, loader.selected_auto_exposure = 1.4, 1.8
    converted, error, _ = loader.load_jxr(str(SAMPLES[0]))
    assert converted is not None, error
    assert torch.allclose(adjust_scene(plain.linear, 1.4, 1.8), converted, rtol=1e-4, atol=1e-6)


def test_exposure_and_gamma_never_touch_a_videos_source_view(clip, device):
    path, pts = clip
    plain = preview.load_video_frame(path, pts[5] / 90000, device, max_side=128, white_nits=400.0)
    pushed = preview.load_video_frame(path, pts[5] / 90000, device, max_side=128, pre_gamma=0.7, exposure=2.0,
                                      white_nits=400.0)
    assert torch.equal(pushed.before, plain.before) and (pushed.pre_gamma, pushed.exposure) == (0.7, 2.0)


# ------------------------------------------------- one resampling, from the real size

@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_shown_at_its_own_size_a_capture_is_shown_pixel_for_pixel(device, video_engine):
    """Reported: staircases on every thin line, on both sides of the divider. The window
    reduced the capture to 1920 pixels and enlarged it again. A picture shown at the size
    of the file must be the file's own pixels, and the converted picture's own pixels."""
    from hdr_blaster.engine.pipeline import render

    source = preview.load_image(SAMPLES[0], device, box=(2560, 1440), white_nits=480.0)
    assert source.size == (1440, 2560)
    assert torch.equal(source.before, preview.reference_view(source.linear, 480.0))
    settings = RenderSettings(method="punchy", edge_strength=25.0, white_nits=480.0)
    shown = preview.render_preview(video_engine, source, settings)
    exact, _ = render(video_engine.tone_mapper, video_engine.loader, video_engine.processor, source.linear, settings)
    assert torch.equal(shown, exact)


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_the_box_of_the_window_decides_the_size_shown(device, video_engine):
    source = preview.load_image(SAMPLES[0], device, box=(1500, 900))
    assert source.size == (844, 1500) and source.before.shape == (1, 3, 844, 1500)
    assert preview.render_preview(video_engine, source, RenderSettings(method="faithful")).shape == (1, 3, 844, 1500)
    preview.fit(source, box=(3000, 1500))                    # the window grew: same source, new size
    assert source.size == (1500, 2667) and source.before.shape == (1, 3, 1500, 2667)
    preview.fit(source, max_side=640)
    assert source.size == (360, 640)
    preview.fit(source, max_side=4096)
    assert source.size == (1440, 2560), "without a box the picture is never enlarged"


def test_highlights_are_clipped_before_the_picture_is_reduced(device, video_engine):
    """A thin light far brighter than white, next to the dark. Reducing the HDR picture
    first lets the light swallow its neighbours (it is still above white once averaged):
    lines come out thicker, edges harder and stepped. What the screen shows at full size
    is the clipped light; that is the picture to reduce."""
    linear = torch.full((1, 3, 64, 64), 2.0 / 800.0, device=device)        # 2 nits
    linear[..., ::4] = 4000.0 / 800.0                                       # one column in four at 4000 nits
    source = preview.PreviewSource(kind="image", path="lines", linear=linear, width=64, height=64,
                                   size=(16, 16), white_nits=400.0)
    light = preview._srgb_decode(source.before)[..., 4:-4, 4:-4]
    assert float(light.mean()) == pytest.approx(0.25 + 0.75 * 2.0 / 400.0, abs=0.01), "a quarter of white, not white"
    assert float(light.max()) < 0.4
    after = preview.render_preview(video_engine, source, RenderSettings(method="faithful", edge_strength=0.0))
    assert after.shape == source.before.shape
    assert float(preview._srgb_decode(after)[..., 4:-4, 4:-4].max()) < 0.4, "the result is reduced after rendering too"


def test_for_display_reduces_light_as_light_adds_up(device):
    checker = torch.zeros(1, 3, 64, 64, device=device)
    checker[..., ::2, ::2] = 1.0
    checker[..., 1::2, 1::2] = 1.0                           # half the pixels white: half the light
    small = preview.for_display(checker, (16, 16))[..., 3:-3, 3:-3]
    assert float(preview._srgb_decode(small).mean()) == pytest.approx(0.5, abs=0.02)
    assert float(small.min()) >= 0.0 and float(small.max()) <= 1.0
    assert preview.for_display(checker, (64, 64)) is checker
    large = preview.for_display(checker, (96, 96))
    assert large.shape == (1, 3, 96, 96) and float(large.min()) >= 0.0 and float(large.max()) <= 1.0


def test_the_preview_of_a_blurred_video_is_the_picture_the_conversion_writes(clip, device, video_engine,
                                                                            tmp_path, monkeypatch):
    """With motion blur the result is made of several frames. The preview adds up the same
    light the same way: what it shows is, value for value, what the file will hold."""
    from hdr_blaster.video.job import exposure_of, frames_to_keep

    path, _ = clip
    chosen = VideoSettings(render=RenderSettings(method="faithful", white_nits=300.0),
                           frame_rate=15.0, motion_blur=True)
    analysis = analyse_video(path, chosen, device, engine=video_engine)
    written, real = [], job.render

    def spy(tone_mapper, loader, processor, linear, render_settings, params=None, **kwargs):
        out = real(tone_mapper, loader, processor, linear, render_settings, params=params, **kwargs)
        if params is not None and params.frozen:
            written.append(out[0].clone())
        return out

    monkeypatch.setattr(job, "render", spy)
    render_video(analysis, tmp_path / "blur.mp4", chosen, device, engine=video_engine)
    monkeypatch.setattr(job, "render", real)

    keep = frames_to_keep(analysis.times, 15.0, analysis.info.average_rate)
    kept = [i for i, k in enumerate(keep) if k]
    cuts = set(analysis.cuts)
    n = next(n for n, i in enumerate(kept) if n > 1 and exposure_of(keep, cuts, i)[1] > 1)
    first, count = exposure_of(keep, cuts, kept[n])
    # the timeline stands on the last frame this picture replaces
    source = preview.load_video_frame(path, analysis.times[first + count - 1], device, max_side=4096,
                                      white_nits=300.0)
    assert exposure_of(keep, cuts, analysis.index_of(source.pts)) == (first, count)
    light = preview.blended_light(path, analysis.times[first], count, device, analysis.info)
    after = preview.render_preview(video_engine, source, chosen.render, analysis.params[first], light=light)
    assert torch.equal(after, written[n])
    alone = preview.render_preview(video_engine, source, chosen.render, analysis.params[first])
    assert not torch.equal(alone, after), "the frame by itself is another picture"

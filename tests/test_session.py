"""The session layer: options, queue, output names, converting one queued file."""
import os
import shutil

import pytest

import video_clips as clips
from conftest import SAMPLES
from hdr_blaster import session
from hdr_blaster.session import Engines, Item, Options, Queue, collect, convert_item, kind_of, output_base
from hdr_blaster.video.job import Cancelled, analyse_video


@pytest.fixture(scope="module")
def engines():
    return Engines()


# --------------------------------------------------------------------- options

def test_defaults_are_the_applications_usual_settings():
    options = Options()
    settings = options.render_settings()
    assert options.look == "Faithful" and settings.method == "faithful", "the scene as it was, by default"
    assert settings.white_nits == session.reference_white() >= 40.0
    assert (settings.edge_strength, settings.color_strength) == (25.0, 0.0)
    video = options.video_settings()
    assert (video.quality, video.out_height, video.pre_gamma, video.exposure) == ("high", None, 1.0, 1.0)


def test_the_four_looks_map_to_engine_operators():
    assert session.LOOKS == {"Faithful": "faithful", "Natural": "natural", "Punchy": "punchy",
                             "Golden": "golden"}
    assert Options(look="Punchy").method() == "punchy"
    assert Options(look="Punchy", operator="mantiuk06").method() == "mantiuk06", "an advanced operator wins"
    from hdr_blaster.engine.constants import SUPPORTED_TONE_MAPS
    assert set(filter(None, session.OPERATORS.values())) - {"auto"} <= SUPPORTED_TONE_MAPS


def test_picture_key_changes_only_when_the_picture_would():
    base = Options()
    assert base.picture_key() == Options(image_format="TIFF", video_quality="compact", destination="x").picture_key()
    assert base.picture_key() != Options(look="Golden").picture_key()
    assert base.picture_key() != Options(sharpness=40.0).picture_key()
    assert base.picture_key() != Options(exposure=1.2).picture_key()


def test_options_survive_saving_and_tolerate_damaged_settings():
    chosen = Options(look="Golden", sharpness=40.0, video_height=1080, use_fp16=True, destination="D:/out")
    assert Options.from_dict(chosen.to_dict()) == chosen
    strings = {k: str(v) for k, v in chosen.to_dict().items()}          # as a settings store returns them
    assert Options.from_dict(strings) == chosen
    damaged = Options.from_dict({"look": "Neon", "sharpness": "lots", "video_height": 333,
                                 "image_format": "BMP", "unknown": 1, "operator": "nope"})
    assert damaged == Options()
    assert Options.from_dict(None) == Options()


def test_output_description_is_plain_english():
    assert session.describe_output("image", Options()) == "JPEG, next to the source"
    assert session.describe_output("image", Options(image_format="Both")) == "JPEG and TIFF, next to the source"
    assert session.describe_output("video", Options(video_height=1080, destination="D:/exports")) == \
        "MP4, high quality, 1080p, exports"


# ---------------------------------------------------------------------- queue

def test_kinds_are_recognised_by_extension():
    assert kind_of("a/shot.JXR") == "image" and kind_of("clip.mp4") == "video" and kind_of("clip.MKV") == "video"
    assert kind_of("photo.jpg") is None and kind_of("notes.txt") is None


def test_dropping_files_and_a_mixed_folder(tmp_path):
    folder = tmp_path / "captures"
    (folder / "nested").mkdir(parents=True)
    for name in ("b.jxr", "a.mp4", "readme.txt", "photo.png"):
        (folder / name).write_text("x")
    (folder / "nested" / "deep.jxr").write_text("x")
    lone = tmp_path / "lone.jxr"
    lone.write_text("x")
    stray = tmp_path / "holiday.jpg"
    stray.write_text("x")
    items, rejected = collect([folder, lone, stray, tmp_path / "missing.mp4"])
    assert [(i.name, i.kind) for i in items] == [("a.mp4", "video"), ("b.jxr", "image"), ("lone.jxr", "image")]
    assert rejected == ["holiday.jpg", "missing.mp4"]


def test_queue_ignores_duplicates_and_keeps_a_selection(tmp_path):
    queue = Queue()
    assert queue.current is None
    a, b, c = (Item(str(tmp_path / n), "image") for n in ("a.jxr", "b.jxr", "c.jxr"))
    assert queue.add([a, b]) == 2 and queue.current is a
    assert queue.add([Item(str(tmp_path / "A.JXR"), "image"), c]) == (1 if os.name == "nt" else 2)
    queue.selected = len(queue.items) - 1
    queue.remove(0)
    assert queue.current is queue.items[-1]
    while queue.items:
        queue.remove(0)
    assert queue.current is None and queue.selected == -1


def test_output_names_never_collide(tmp_path):
    src = tmp_path / "Shot.jxr"
    assert output_base(src, "image", Options()) == str(tmp_path / "Shot_SDR")
    (tmp_path / "Shot_SDR_tonemapped.tiff").write_text("x")
    assert output_base(src, "image", Options()) == str(tmp_path / "Shot_SDR-2")
    video = tmp_path / "Clip.mp4"
    (tmp_path / "Clip_SDR.mp4.partial").write_text("x")
    assert output_base(video, "video", Options()) == str(tmp_path / "Clip_SDR-2")
    elsewhere = tmp_path / "out"
    elsewhere.mkdir()
    assert output_base(video, "video", Options(destination=str(elsewhere))) == str(elsewhere / "Clip_SDR")


# ----------------------------------------------------------------- converting

@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_a_capture_is_converted_next_to_itself(tmp_path, engines):
    src = tmp_path / "shot.jxr"
    shutil.copy(SAMPLES[0], src)
    options = Options(image_format="Both")
    engine, device = engines.get(options)
    seen = []
    written = convert_item(Item(str(src), "image"), options, engine, device,
                           progress=lambda *a: seen.append(a))
    assert [os.path.basename(p) for p in written] == ["shot_SDR.jpg", "shot_SDR_raw.tiff", "shot_SDR_tonemapped.tiff"]
    assert all(os.path.getsize(p) > 10_000 for p in written)
    assert seen[0] == ("render", 0, 1) and seen[-1] == ("render", 1, 1)
    again = convert_item(Item(str(src), "image"), Options(), engine, device)
    assert [os.path.basename(p) for p in again] == ["shot_SDR-2.jpg"], "nothing is overwritten"


@pytest.mark.skipif(not SAMPLES, reason="no .jxr capture in samples/")
def test_a_capture_converts_like_the_command_line(tmp_path, engines):
    """Same defaults, same pixels: the JPEG's picture equals the engine's direct render."""
    import numpy as np
    from PIL import Image

    from hdr_blaster.engine.pipeline import RenderSettings, render

    src = tmp_path / "shot.jxr"
    shutil.copy(SAMPLES[0], src)
    engine, device = engines.get(Options())
    written = convert_item(Item(str(src), "image"), Options(), engine, device)
    linear, _, _ = engine.loader.load_jxr(str(src))
    expected, _ = render(engine.tone_mapper, engine.loader, engine.processor, linear,
                         Options().render_settings())
    saved = np.asarray(Image.open(written[0]), dtype=np.float32) / 255.0
    reference = expected[0].permute(1, 2, 0).cpu().numpy()
    assert np.abs(saved - reference).mean() < 0.01


def test_a_video_is_converted_and_an_analysis_is_reused(tmp_path, engines, monkeypatch):
    src = tmp_path / "clip.mp4"
    clips.make_clip(src, frames=20)
    options = Options(look="Punchy")
    engine, device = engines.get(options)
    analysis = analyse_video(str(src), options.video_settings(), device, engine=engine)
    stages = []
    written = convert_item(Item(str(src), "video"), options, engine, device,
                           progress=lambda stage, done, total: stages.append(stage), analysis=analysis)
    assert [os.path.basename(p) for p in written] == ["clip_SDR.mp4"]
    assert set(stages) == {"render"}, "with a matching analysis the video is not analysed again"
    stages.clear()
    convert_item(Item(str(src), "video"), Options(look="Golden"), engine, device,
                 progress=lambda stage, done, total: stages.append(stage), analysis=analysis)
    assert "analyse" in stages, "another look needs its own analysis"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4", "clip_SDR-2.mp4", "clip_SDR.mp4"]


def test_cancelling_a_video_leaves_nothing(tmp_path, engines):
    src = tmp_path / "clip.mp4"
    clips.make_clip(src, frames=20)
    engine, device = engines.get(Options())
    calls = []
    with pytest.raises(Cancelled):
        convert_item(Item(str(src), "video"), Options(look="Punchy"), engine, device,
                     progress=lambda *a: calls.append(a), cancelled=lambda: len(calls) >= 5)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]


def test_a_missing_destination_is_reported_clearly(tmp_path, engines):
    src = tmp_path / "clip.mp4"
    clips.make_clip(src, frames=4)
    engine, device = engines.get(Options())
    with pytest.raises(FileNotFoundError, match="destination folder"):
        convert_item(Item(str(src), "video"), Options(destination=str(tmp_path / "nowhere")), engine, device)


def test_engines_are_built_once_per_device_and_precision():
    engines = Engines()
    first, device = engines.get(Options())
    assert engines.get(Options(look="Golden"))[0] is first
    cpu, cpu_device = engines.get(Options(use_gpu=False))
    assert cpu_device.type == "cpu" and (cpu is not first or device.type == "cpu")
    assert engines.get(Options(use_gpu=False, use_fp16=True))[0] is cpu, "half precision is a GPU option"


# ------------------------------------------------ found by the independent review

def test_hostile_settings_cannot_break_the_start(tmp_path):
    for bad in ({"sharpness": "nan"}, {"ai_color": "inf"}, {"exposure": 50}, {"pre_gamma": -3},
                {"destination": None}, {"sharpness": 1e9}, {"exposure": "nan"}):
        options = Options.from_dict(bad)
        assert 0.0 <= options.sharpness <= 100.0 and 0.0 <= options.ai_color <= 100.0
        assert 0.1 <= options.exposure <= 3.0 and 0.1 <= options.pre_gamma <= 3.0
        assert options.destination == ""


def test_a_folder_dropped_again_does_not_queue_its_own_results(tmp_path):
    folder = tmp_path / "captures"
    folder.mkdir()
    for name in ("a.mp4", "a_SDR.mp4", "b_SDR-2.mp4", "shot.jxr", "c_SDRish.mp4"):
        (folder / name).write_text("x")
    items, rejected = collect([folder])
    assert [i.name for i in items] == ["a.mp4", "c_SDRish.mp4", "shot.jxr"] and rejected == []
    items, _ = collect([folder / "a_SDR.mp4"])
    assert [i.name for i in items] == ["a_SDR.mp4"], "a file dropped by itself is taken as it is"


def test_an_empty_folder_is_reported(tmp_path):
    folder = tmp_path / "holiday"
    folder.mkdir()
    (folder / "photo.jpg").write_text("x")
    items, rejected = collect([folder])
    assert items == [] and rejected == ["holiday"]


def test_operators_have_names_for_people():
    assert session.display_name("punchy") == "Punchy" and session.display_name("hable") == "Hable"
    assert session.display_name("neural") == "Neural (AI)" and session.display_name(None) == "Auto"
    assert session.display_name("faithful") == "Faithful"


# ------------------------------------------------------------ the faithful look

def test_auto_select_is_an_advanced_operator_now():
    """Auto used to be the first look. It picks an operator from the image's statistics,
    which re-exposes the scene: it cannot promise the source's luminance. It lives on in
    Advanced; the engine still calls it ``None``."""
    assert session.OPERATORS["Auto-select"] == "auto"
    assert Options(operator="auto").method() is None
    assert Options(look="Punchy", operator="auto").render_settings().method is None


def test_a_settings_file_from_before_falls_back_to_faithful():
    assert Options.from_dict({"look": "Auto"}).look == "Faithful"
    assert Options.from_dict({"look": "Auto"}).method() == "faithful"
    assert Options.from_dict({"look": "Golden", "operator": "auto"}).method() is None


def test_the_screens_white_reaches_the_render_and_the_picture_key(monkeypatch):
    monkeypatch.setattr(session, "sdr_white_nits", lambda: (300.0, "display"))
    options = Options()
    assert options.render_settings().white_nits == 300.0
    assert options.video_settings().render.white_nits == 300.0
    before = options.picture_key()
    monkeypatch.setattr(session, "sdr_white_nits", lambda: (480.0, "display"))
    assert options.picture_key() != before, "another white is another picture"

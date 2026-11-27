"""Starting the program: the window first, the engine behind it."""
import json
import subprocess
import sys
from pathlib import Path

import ui_layout_check  # noqa: F401  (sets the offscreen platform before Qt loads)
import video_clips as clips

ROOT = Path(__file__).resolve().parents[1]


def start(tmp_path, *files) -> dict:
    """Start the program in a process of its own (PyTorch not loaded yet), off screen."""
    done = subprocess.run([sys.executable, str(ROOT / "tests" / "startup_check.py"), str(tmp_path), *map(str, files)],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_window_does_not_wait_for_the_engine(tmp_path):
    """Reported: the program takes long to start. Loading PyTorch takes seconds; the window's
    own thread must not be the one that does it. A thread of its own loads the engine,
    sets it up as the command line does, and the window is told what it found."""
    seen = start(tmp_path)
    assert seen["exit"] == 0 and seen["window_shown"]
    assert seen["imported_by"].get("torch") == "hdr-blaster-engine", seen["imported_by"]
    assert set(seen["imported_by"].values()) == {"hdr-blaster-engine"}, \
        f"neither the window's thread nor a slot of the background thread: {seen['imported_by']}"
    assert seen["engine_reported"] == seen["cuda"], "the window is told whether a graphics card can be used"
    assert seen["gpu_switch_enabled"] == seen["cuda"]
    assert seen["deterministic"] is True and seen["benchmark"] is False, "the engine is set up as before"
    assert seen["log_written"]


def test_starting_the_program_here_leaves_the_users_settings_alone(tmp_path):
    """The check above starts the real program. Its settings must be the ones of the test:
    an earlier version of the check read and wrote the user's own (the registry), and a
    script built the same way left its output folder there as the place to save to."""
    seen = start(tmp_path)
    assert Path(seen["settings"]) == tmp_path / "settings.ini", seen["settings"]
    assert (tmp_path / "settings.ini").is_file(), "the window's place and size were saved there"
    assert "window" in (tmp_path / "settings.ini").read_text(encoding="utf-8")


def test_a_file_given_at_start_is_shown(tmp_path):
    """Started with a file ("Open with"), the engine is used as soon as it is loaded. Loaded
    by a slot of the background thread, PyTorch died there on the first model it read
    (Fatal Python error: _PyThreadState_Attach): a thread started by Python loads it."""
    video = tmp_path / "clip.mp4"
    clips.make_clip(video, frames=12, audio_streams=0)
    seen = start(tmp_path, video)
    assert seen["failed"] == "" and seen["picture"], "the picture of the file comes"
    assert seen["exit"] == 0
    assert seen["imported_by"].get("torch") == "hdr-blaster-engine", seen["imported_by"]
    assert "window" not in seen["imported_by"].values(), seen["imported_by"]


def test_the_runtime_is_set_up_in_two_steps_that_make_the_whole():
    """The window sets up the log at once and leaves PyTorch to the background thread; the
    command line does both in one call, as it always did."""
    code = ("import sys, hdr_blaster, os\n"
            "hdr_blaster.configure_logging(os.devnull)\n"
            "assert 'torch' not in sys.modules, 'the log does not need PyTorch'\n"
            "hdr_blaster.configure_torch()\n"
            "import torch\n"
            "assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark\n"
            "state = torch.random.get_rng_state().clone()\n"
            "torch.rand(3)\n"
            "hdr_blaster.configure_runtime(os.devnull)\n"
            "assert not torch.equal(torch.random.get_rng_state(), state), 'set up once: the seed is not put back'\n"
            "print('ok')\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stdout + done.stderr


def test_the_graphics_card_switch_follows_what_the_engine_found():
    from hdr_blaster.session import Options
    from hdr_blaster.ui.advanced import AdvancedPanel
    from hdr_blaster.ui.app import create_application

    create_application([])
    panel = AdvancedPanel()
    panel.set_options(Options(use_gpu=True, use_fp16=True))
    assert panel.use_gpu.isEnabled() and panel.use_gpu.isChecked(), "until the engine has spoken, as the options say"
    heard = []
    panel.changed.connect(heard.append)
    panel.set_cuda(True)
    assert panel.use_gpu.isEnabled() and panel.use_gpu.isChecked() and panel.use_fp16.isEnabled()
    panel.set_cuda(False)
    assert not panel.use_gpu.isEnabled() and not panel.use_gpu.isChecked()
    assert "No CUDA graphics card" in panel.use_gpu.toolTip()
    assert not panel.use_fp16.isEnabled(), "half precision is a graphics card option"
    assert panel.options().use_gpu is True and not heard, "the user's choice is kept, and nothing is announced"
    panel.set_options(Options(use_gpu=True))
    assert not panel.use_gpu.isChecked(), "still none when the options are shown again"

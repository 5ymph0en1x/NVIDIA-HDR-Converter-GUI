"""Bundled assets must be found both from source and from a PyInstaller build."""
import os
import sys
from pathlib import Path

from hdr_blaster.paths import app_data_dir

ROOT = Path(__file__).resolve().parents[1]


def test_from_source_it_is_the_project_root():
    assert Path(app_data_dir()) == ROOT
    assert os.path.isdir(os.path.join(app_data_dir(), "models", "3dlut", "sRGB"))
    assert os.path.isfile(os.path.join(app_data_dir(), "training", "model.py"))


def test_frozen_build_reads_from_meipass(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert app_data_dir() == str(tmp_path)


def test_frozen_build_without_meipass_falls_back_to_the_exe_folder(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert app_data_dir() == os.path.dirname(sys.executable)

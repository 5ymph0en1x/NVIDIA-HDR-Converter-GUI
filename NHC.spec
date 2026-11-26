# -*- mode: python ; coding: utf-8 -*-
r"""
NHC.spec — build HDR Blaster (the Qt window, photos and videos) as a standalone
ONE-FILE .exe.

    .venv\Scripts\python.exe -m PyInstaller NHC.spec --noconfirm
    dist\HDR_Blaster.exe --convert capture.jxr            # headless self-test, image
    dist\HDR_Blaster.exe --convert-video recording.mp4    # headless self-test, video

CUDA trim (explained in BUILD.md): only DLLs that appear in no import table are
dropped. PyAV brings its own FFmpeg libraries; PySide6 is collected by its hook.
"""
import os
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ["imagecodecs._imagecodecs", "tifffile"]

# Packages with data files / native extensions PyInstaller's static analysis
# does not fully capture on its own.
for _pkg in ("imagecodecs", "tifffile", "av"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# --- Bundled app assets (read at runtime from sys._MEIPASS) ------------------
# Only the two small files we need from training/, NOT the datasets or its venv.
datas += [
    ("models", "models"),                                       # 3D LUT weights (~2.4 MB)
    ("training/model.py", "training"),                          # NeuralToneMapper source
    ("training/checkpoints/latest.pth", "training/checkpoints"),  # neural checkpoint (~1.8 MB)
    ("hdr_blaster/ui/icon.ico", "hdr_blaster/ui"),              # window and taskbar icon
    ("hdr_blaster/ui/icon.png", "hdr_blaster/ui"),
]

# Slim the bundle: none of these are used by the app at runtime.
# NB: do NOT exclude any torch.* submodule — torch/__init__.py imports many of
# them (distributions, fft, linalg, ...) at import time, so excluding breaks it.
excludes = ["matplotlib", "torchvision", "scipy", "pandas", "pytest", "IPython",
            "notebook", "sphinx", "setuptools", "pip", "wheel", "tkinter", "TKinterModernThemes",
            "PySide6.QtNetwork", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtSql", "PySide6.QtTest"]

a = Analysis(
    ["NHC.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["rt_hook_nhc.py"],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# --- Intelligent CUDA trim: drop torch/lib DLLs nothing links against --------
# NB: cufft64_12 / cusolver64_12 / cusparse64_12 are STATIC imports of
# torch_cuda.dll — never called by this app but required at load time, so they
# stay. Only DLLs in no PE import table (lazily loaded, never triggered by this
# pipeline) may be dropped. Verified by a probe run with the CUDA Toolkit
# scrubbed from PATH. ~336 MB.
_DROP = {
    "cufftw64_12.dll",            # FFTW-compat shim (0.2 MB)
    "cusolvermg64_12.dll",        # multi-GPU solvers (91 MB)
    "curand64_10.dll",            # RNG (56 MB)
    "cudnn_adv64_9.dll",          # cuDNN advanced / RNN / attn (102 MB)
    "nvrtc64_130_0.alt.dll",      # alternate NVRTC (87 MB)
}
_before = len(a.binaries)
a.binaries = [b for b in a.binaries
              if os.path.basename(b[0]).lower() not in _DROP]
print(f"[NHC.spec] dropped {_before - len(a.binaries)} unused CUDA DLLs "
      f"(~336 MB) from the bundle")

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="HDR_Blaster",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX corrupts CUDA DLLs -> keep disabled
    runtime_tmpdir=None,
    console=False,             # windowed release; the headless commands log to nhc_console.log
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="hdr_blaster/ui/icon.ico",
)

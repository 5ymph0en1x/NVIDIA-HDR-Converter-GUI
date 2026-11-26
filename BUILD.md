# Building HDR Blaster — standalone one-file .exe with bundled CUDA

HDR Blaster ships as a **single self-contained `HDR_Blaster.exe`** (the Qt window, photos and
videos). The end user needs **no Python, no CUDA Toolkit, no PyTorch install** — only a recent
**NVIDIA driver** (the driver provides `nvcuda.dll`; everything else is bundled).

## Build

```powershell
# one-off: the build tools (the venv is uv-managed and has no pip)
uv pip install --python .venv\Scripts\python.exe pyinstaller psutil

# build
.venv\Scripts\python.exe -m PyInstaller NHC.spec --noconfirm
# -> dist\HDR_Blaster.exe
```

Run PyInstaller through this environment's Python (`python -m PyInstaller`), not through
`.venv\Scripts\pyinstaller.exe`: that launcher remembers the folder the environment was
created in. If the project was copied from elsewhere, it silently builds with the OTHER
environment, and the result lacks whatever was installed since (found the hard way: an
executable without PyAV, whose video conversion failed at start).

Self-test the result (headless, no window):

```powershell
dist\HDR_Blaster.exe --convert "path\to\shot.jxr"             # writes shot_SDR.jpg, exits 0 on success
dist\HDR_Blaster.exe --convert-video "path\to\recording.mp4"   # writes recording_SDR.mp4
```

Verified on 2026-11-25 (Python 3.14.2, torch 2.11.0+cu130, PyAV 19.0.2, PySide6 6.12.0, RTX 4090):
the one-file executable is 1.77 GB; started without Python it converts a capture, converts a
796-frame 1440p recording with NVENC (84.4 MB to 75.3 MB), and its window opens and renders a
preview. Start-up takes 30 to 60 s: the one-file bundle unpacks itself first. PyAV carries its
own FFmpeg libraries; PySide6 is collected by its PyInstaller hook. The window writes its log to
`%LOCALAPPDATA%\HDR Blaster\hdr_blaster.log`.

## Intelligent CUDA bundling (the interesting part)

A naïve PyInstaller build of a PyTorch app drags in **~2.56 GB** of CUDA DLLs from
`torch\lib`. We cut **~336 MB** of that — verified, not guessed.

Two different mechanisms decide what can go:

- **Static PE imports must ship.** `torch_cuda.dll` *links against* `cufft64_12`,
  `cusolver64_12` and `cusparse64_12` (check its import table), so Windows must
  resolve them the moment `import torch` loads `torch\lib\*.dll` — even though the
  app never calls a single FFT/sparse/solver kernel. Dropping them raises
  `WinError 126 — Error loading "…\torch\lib\shm.dll" or one of its dependencies`
  on any machine without the CUDA Toolkit (`shm.dll` is simply the first DLL in
  torch's alphabetical load loop that pulls `torch_cuda.dll` in).
- **Lazily-loaded DLLs may go.** DLLs that appear in *no* import table are only
  loaded on first use; those this pipeline never triggers can be dropped.

The drop set was validated empirically by a probe (it moves candidate DLLs out of
`torch\lib` and re-runs the entire pipeline in a fresh process) — and, crucially, it
runs that test with the CUDA Toolkit **scrubbed from PATH**. A dev machine
silently resolves missing DLLs out of `C:\Program Files\NVIDIA GPU Computing
Toolkit\...\bin`, which is exactly how the original ~871 MB trim passed every local
test yet crashed with the `shm.dll` error on end-user machines. (The driver's
`nvcuda.dll` lives in System32, so the GPU still works in the scrubbed test.)
The verified drop set, applied in `NHC.spec`:

| Dropped (in no import table, never triggered) | Size |
|---|---|
| `cufftw64_12` (FFTW-compat shim) | 0.2 MB |
| `cusolverMg64_12` (multi-GPU solvers) | 91 MB |
| `curand64_10` (RNG) | 56 MB |
| `cudnn_adv64_9` (cuDNN RNN/attention) | 102 MB |
| `nvrtc64_130_0.alt` (alternate NVRTC) | 87 MB |
| **Total** | **~336 MB** |

**Kept**: everything on a static-import path — `cublas`, `cublasLt`, the core cuDNN
sub-libraries (`ops`, `cnn`, `engines_precompiled`, `engines_runtime_compiled`,
`heuristic`, `graph`, dispatcher), `cudart`, `nvrtc` (+builtins), `nvJitLink`,
`cupti`+`nvperf_host` (static imports of `torch_cpu`), **`cufft64_12` /
`cusolver64_12` / `cusparse64_12` (static imports of `torch_cuda`)**, plus
`torch_cuda` / `torch_cpu` / `c10`.

> Validate the drop set again after any PyTorch/CUDA upgrade — the safe set can change.
> To validate a built exe the way an end user will run it: strip CUDA/NVIDIA
> entries from `PATH`, unset `CUDA_PATH*`, then run `--selftest`.

## Bundled assets

`NHC.spec` bundles, and the app reads them from `sys._MEIPASS` when frozen:
`models\3dlut\sRGB\*` (3D LUT weights), `training\model.py` + `training\checkpoints\
latest.pth` (neural tone mapper), and the icon (`hdr_blaster\ui\icon.ico`, made from `app.ico`
by `tools\make_icon.py`). The heavy `training\datasets\` and
`training\venv\` are **not** bundled.

`rt_hook_nhc.py` runs before the app: it keeps `stdout`/`stderr` valid under
`--noconsole` (redirecting to `nhc_console.log` next to the exe) and makes the bundled
`torch\lib` discoverable.

## Sizes

| | |
|---|---|
| `torch\lib` CUDA DLLs, naïve | ~2.56 GB |
| dropped (verified unused) | ~336 MB |
| **`HDR_Blaster.exe`** (one-file, GPU) | **~1.77 GB** |
| the same build *without* the trim (est.) | ~2.06 GB → **‑22 %** |

Verified on Python 3.14.2 / torch 2.11.0+cu130 / RTX 4090, **with the CUDA Toolkit
scrubbed from the environment** (clean end-user machine simulation): the trimmed
one-file exe self-tests green (`--selftest` converts a JXR on CUDA) and the windowed
GUI initializes on CUDA and reaches its main event loop.

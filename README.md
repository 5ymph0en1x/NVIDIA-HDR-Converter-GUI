# NVIDIA HDR Converter GUI — HDR Blaster

**Version 3.0.3**

Convert NVIDIA HDR captures to SDR: JXR screenshots to JPEG or TIFF, and HDR10 recordings to
H.264 video, in one window. Version 3 of *NVIDIA HDR Converter GUI* is a new application,
called HDR Blaster; `NHC.py` still starts it.

<p align="center">
  <img width="1024" src="docs/screenshot.png">
</p>

## What's new in 3.0.3

- **The program opens in about a second.** It took thirty. The standalone build is now a
  folder (`HDR_Blaster.exe` next to its libraries, published as one zip) instead of a single
  file that unpacked 2.6 GB to the temporary folder at every launch, and left them there
  whenever it did not close cleanly. Unzip it once, anywhere, and keep the two together.
- **The window no longer waits for the engine**: it opens at once, and a file can be shown
  about three seconds after the start. The converted files are the same, byte for byte.
- **The Advanced panel keeps its width** when a folder is chosen under *Save to*: its
  controls were cut off at the right. A long folder is shown on several lines.

## What's new in 3.0.2

- **Videos convert about three times faster, and write the same pictures.** They are
  decoded by the graphics card (NVDEC), and decoding, rendering and encoding run side by
  side. A one-minute 1440p recording at 120 frames per second took 5.9 minutes; it takes about 1.8.
- **Frame rate**: Original, 120, 60 or 30 frames per second. Frames are dropped, never
  invented: the ones that stay are the very ones the full conversion writes, with their own
  timestamps. The same recording takes 1.2 minutes at 60 and 1 minute at 30.
- **Motion blur at a lower frame rate**, if you ask for it: each frame written holds the
  light of the frames it replaces, added up in the scene's linear light as a longer exposure
  adds it. Motion is smoother and softer, and nothing is invented. (Textures blur as a camera
  blurs them; a sharp edge that crosses the picture fast shows as two or four stepped copies
  rather than one streak: the recording holds no picture between two of its frames.)
- **Videos are no longer sharpened unless asked** (*Sharpness and AI colour on videos*, in
  Advanced): nothing is then measured on their pictures, and they are analysed twice as fast.
- From 3.0.1: refined controls (pills, a segmented choice of looks, switches), and a click
  in the Advanced panel no longer moves the divider.

## What's new in 3.0.0

- **A new window** (Qt): drop a file, look at the HDR source and the SDR result on either
  side of a divider, convert. Four looks up front, everything else in an Advanced panel.
- **Videos**: HDR10 recordings are converted to H.264 SDR, with their audio untouched.
- **Faithful by default**: the result shows the scene at the luminance the game gave it;
  only what is brighter than your screen's white is rolled off (ITU-R BT.2390).
- **Every look and operator keeps the picture's dark tones and mid-grey.** They are
  variations of the same picture, not re-exposures of it, and each classical operator has
  a character of its own.
- **The HDR source is shown in real HDR** on a display with HDR switched on, as Windows'
  own viewer shows it.
- **The preview is the conversion**: rendered at the file's size, shown pixel for pixel.
- **Commands without the window**: `--convert`, `--convert-video`, `--version`.
- The author's image-adaptive 3D LUT colour models are unchanged (the *AI colour* setting).

## Features

### Core functionality
- **JXR captures** (HDR screenshots) to JPEG and/or TIFF
- **HDR10 recordings** to H.264 SDR video, never heavier than the source
- **Video on the graphics card from end to end**: decoded by NVDEC, rendered, encoded by
  NVENC, the three side by side. Without an NVIDIA card the processor decodes and encodes
- **Looks and operators** that keep what the game decided (see *Renderings*)
- **Batch processing**: several files or a folder, in a queue
- **The HDR source shown in real HDR.** On a display with HDR switched on, the picture area
  is presented in HDR (an scRGB surface, like Windows' own viewer): on the left of the
  divider the capture as the game showed it, with its lights far above the window's white;
  on the right the SDR result at the luminance an SDR window gives it on that screen. On a
  display without HDR the source is shown in the SDR window, cut at its white, and labelled so
- **Live preview** with before/after comparison and an optional histogram. Both pictures
  are rendered at the file's real size and brought to the window in a single Spline36
  resampling (anti-ringing, linear light when reducing); shown at the file's size they are
  its pixels, untouched

### AI colour enhancement
- **Image-adaptive 3D LUT** pretrained on the MIT-Adobe FiveK dataset (Zeng et al., TPAMI 2020)
- **Per-image LUT prediction**: a lightweight CNN blends 3 basis 33×33×33 LUTs based on each input's content
- **Pure-PyTorch trilinear interpolation** via `F.grid_sample` — no CUDA extension to compile
- **Edge enhancement** with Sobel filters and adaptive strength control
- **Half-precision (FP16)** support for memory-efficient GPU processing
- **~2.4 MB pretrained weights**, included in `models/3dlut/` (downloaded on first launch if missing)

### Renderings
A game's HDR picture is finished work, so every rendering keeps what the game decided and
only handles what is brighter than SDR white. All of them obey the same rules: dark tones
are the scene's own, mid-grey stays mid-grey, the scene's peak lands on white. They differ
above the shadows.

- **Faithful** ⭐ — the default: the scene at its own luminance; only what exceeds your screen's
  white is rolled off (ITU-R BT.2390), hues kept.
- **Natural / Punchy / Golden** — Faithful, graded: softer with fading highlights; more
  contrast and colour; punchy in a warm light.
- The classical operators, each with the trait it is known for. (Held to the same shadows,
  mid-grey and peak, their published curves are the same curve to within a level or two;
  what tells them apart is how each treats the tones above the shadows.)
  - **Reinhard** — soft: nothing touched up to mid-grey, a gentle shoulder from there; bright
    colours go pastel.
  - **Hable** — film: more contrast in the mid-tones, muted colours, lights that fade.
  - **ACES** — vivid: strong contrast, rich mid-tones, lights that burn to white.
  - **Drago03** — flat: the whole range above the shadows compressed evenly, colours kept.
  - **Perceptual** — luminous: the BT.2390 roll-off in ICtCp. A coloured light keeps its hue
    and its brightness; its colour yields (Faithful keeps the colour and lets it darken).
  - **Mantiuk06** — local: a soft roll-off of the large shapes, their local contrast
    reinforced (grain is left alone).
- **Adaptive** — two of the operators, chosen from the picture and blended by brightness.
- **Auto-select** — one operator chosen from the picture's statistics (dynamic range,
  highlights, local contrast, saturation).
- **Neural (AI)** — HDRNet-style learned operator, taught by Punchy. See
  [`training/README.md`](training/README.md) to retrain or fine-tune.

### User interface
- **One window**: drop a file, look, convert
- **HDR source and SDR result side by side** under a divider you drag
- **Four looks** up front (Faithful, Natural, Punchy, Golden); everything else in an Advanced panel
- **The source view is the original**: Exposure and Gamma shape the result only
- **Queue** for several captures and recordings, progress and cancel on the one button
- **Exact video preview** once the recording has been analysed in the background
- **Frame rate of a video**: Original, 120, 60 or 30 frames per second, in Advanced, with
  motion blur made of the dropped frames if you switch it on
- **Videos are neither sharpened nor AI-coloured unless asked**: nothing is then measured on
  their pictures and the analysis is about twice as fast. *Sharpness and AI colour on videos*,
  in Advanced, applies both to videos as to captures
- **Nothing is ever overwritten**: results are written as `<name>_SDR`, numbered if needed

## Requirements

### System requirements
- **Windows 10 or 11** (built and tested there)
- **Python 3.14+**
- **NVIDIA GPU** (recommended) with CUDA support; the CPU path works but is slow
- **4GB+ RAM** (8GB+ recommended for large images)

### Python dependencies
```txt
torch==2.11.0
numpy==2.4.3
pillow==12.1.1
imagecodecs==2026.5.10
tifffile==2026.5.2
av==19.0.2
pyside6-essentials==6.12.0
```

**Install PyTorch first**, from the PyTorch index, to get the CUDA build:
`pip install torch --index-url https://download.pytorch.org/whl/cu130`.
The dependencies are declared in `pyproject.toml`; `requirements.txt` mirrors them.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/5ymph0en1x/NVIDIA-HDR-Converter-GUI.git
   cd NVIDIA-HDR-Converter-GUI
   ```

2. **Install dependencies:**
   ```bash
   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python NHC.py
   ```

A standalone build (no Python needed) can be made with PyInstaller: a folder holding
`HDR_Blaster.exe` and its libraries, whose window opens in about a second. See
[`BUILD.md`](BUILD.md).

## Usage

### The window

```bash
python NHC.py            # or: python -m hdr_blaster
```

1. Drop a JXR capture, an HDR10 recording, several files or a folder on the window.
2. Look at the result: the unconverted picture is on the left of the divider, the
   converted one on the right. Drag the divider. For a video, the timeline chooses
   the frame; once the video has been analysed in the background the preview is
   exactly the frame the converted file will contain.
3. Pick a look if you wish (Faithful, Natural, Punchy, Golden) and press **Convert**.
   The classical operators, *Auto-select* and the neural operator are in Advanced, under Operator.

The result is written next to the source as `<name>_SDR.jpg` or `<name>_SDR.mp4`,
never over an existing file. Everything else (precise operator, exposure, sharpness,
AI colour, formats, video quality and size, destination folder, GPU or CPU) is in the
**Advanced** panel.

### Without the window

```bash
python NHC.py --convert capture.jxr [--method punchy] [--format JPEG|TIFF|Both]
python NHC.py --convert-video recording.mp4 [--method punchy] [--quality high|balanced|compact] [--height 1080] [--fps 60] [--blur]
python NHC.py --version
```

Both commands render Faithful unless told otherwise (`--method auto` lets the engine pick
an operator) and write `<name>_SDR` next to the source, never over an existing file.

Videos: HDR10 (PQ, BT.2020, 10-bit) recordings are converted to H.264 SDR with the
source's own timestamps and its audio copied untouched. The result is never heavier
than the source. A video takes no sharpening and no AI colour unless `--edge` or `--color`
ask for them (the analysis then renders every frame, and takes about twice as long).
`--fps` writes at most that many frames per second: for each tick of that clock the nearest
frame of the recording is kept, with its own timestamp; the others are left out, or, with
`--blur`, their light is added to the frame that replaces them (never across two shots).

## Technical architecture

### Processing pipeline
1. **Decoding**: a JXR capture (floating-point scRGB) or a frame of an HDR10 recording
2. **Rendering** with the chosen look or operator (**Faithful** by default); a video is
   analysed first, so that its measured parameters are steady within each shot. Decoding,
   computing and encoding run side by side; the fast path writes the very pictures of the
   plain one (`reference=True` in `hdr_blaster/video/job.py`), which the tests compare
3. **AI colour retouch** (optional) via the image-adaptive 3D LUT
4. **Edge enhancement** (optional)
5. **sRGB conversion**, dithering, and JPEG/TIFF or H.264 encoding

### AI enhancement architecture
```
                          ┌─ basis LUT 0 (3×33×33×33) ─┐
Input (downsampled 256²) → Classifier CNN → 3 weights →┤  basis LUT 1                ├→ combined LUT → trilinear lookup → Output
                          └─ basis LUT 2                ┘
```
The classifier (~440 K params) predicts 3 blending weights from a 256×256 downsampled view of the input.
The 3 basis LUTs are linearly combined into a single image-adaptive 3D LUT, then applied to the full-resolution image via trilinear interpolation (pure PyTorch `F.grid_sample`).
Total inference cost: ~600 K params, sub-millisecond on a modern GPU.

### Layout of the code
- `NHC.py` — the launcher
- `hdr_blaster/engine/` — the renderings (`renderings.py`), tone-mapping dispatch and the
  Auto-select choice (`tonemap.py`), the 3D LUT colour models (`lut3d.py`), sharpening
- `hdr_blaster/io/` — JXR reading; `hdr_blaster/video/` — HDR10 decoding, analysis, encoding
- `hdr_blaster/preview.py`, `hdr_blaster/resample.py` — what the window shows
- `hdr_blaster/ui/` — the window, and the HDR surface (`hdr_surface.py`)
- `training/` — the neural tone mapper: model, training code, shipped checkpoint
- `models/3dlut/` — the 3D LUT weights
- `tests/` — the test suite; `tools/` — icon generation, on-screen HDR check

### Tests
```bash
pip install pytest
python -m pytest
```
The tests need real captures: put a few `.jxr` files in `samples/` first (without them
most tests are skipped, and one fails to say so). Reading a capture, the colour models, the original sharpening and the files
written are checked bit for bit against a frozen copy of the original application
(`tests/legacy/`). On a display with HDR switched on, `HDR_BLASTER_ONSCREEN=1` adds a check
that reads back what the HDR surface presents.

## Supported formats

### Input
- **JXR captures** (JPEG XR, floating-point scRGB) from the NVIDIA app or Windows
- **HDR10 recordings** (PQ, BT.2020, 10-bit 4:2:0) in MP4, MKV or MOV

### Output
- **JPEG** (4:4:4, dithered) and/or **TIFF** for captures
- **MP4** (H.264, BT.709) for recordings, never heavier than the source

## Logging
- The window writes its log to `%LOCALAPPDATA%\HDR Blaster\hdr_blaster.log`
- The commands without the window write `hdr_converter.log` in the current folder

## Known limitations
- **HDR input only**: SDR, HLG and Dolby Vision files are refused with an explanation
- **Windows**: built and tested on Windows with an NVIDIA GPU; the CPU path works but is slow
- **Real HDR display of the source** needs Windows with HDR switched on for the screen the
  window is on; elsewhere the source is shown cut at the window's white
- **Video speed**: a one-minute 1440p recording at 120 frames per second (7,142 frames) converts in about 1.8 minutes on an RTX 4090 (analysis 190 frames per second, rendering and
  encoding 110), 2.4 minutes with sharpening, 1.2 minutes at 60 frames per second and 1 minute at 30

## Troubleshooting
- **CUDA out of memory**: switch on *Half precision*, or switch off *Use the graphics card*, in the Advanced panel
- **A capture is refused**: only floating-point JXR captures (HDR screenshots) are accepted
- **A video is refused**: only HDR10 (PQ, BT.2020) recordings are accepted

## Acknowledgments
- **Hui Zeng et al.** for the [Image-Adaptive 3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) model (TPAMI 2020) — pretrained weights used under Apache 2.0
- **MIT-Adobe FiveK** dataset (Bychkovsky et al.) used to train the LUT model
- **Qt for Python (PySide6)** for the interface, **PyAV / FFmpeg** for video
- **imagecodecs** for JXR decoding support
- **NVIDIA** for HDR screenshot format documentation

## License
This project is open source. Please check the [license file](LICENSE) for details.

<div align="center">

# HDR Blaster

<p align="center"><img src="hdr_blaster/ui/icon.png" alt="HDR Blaster" width="120" /></p>
<p align="center"><img src="docs/screenshot.png" alt="HDR Blaster: the HDR source and the SDR result on either side of a divider" width="1000" /></p>

### Your game looked like this. Your screenshot should too.

*NVIDIA HDR captures and recordings, converted to SDR files that keep what the game decided: JXR to JPEG or TIFF, HDR10 to H.264, in one window.*

![Version](https://img.shields.io/badge/version-3.1.3-1f6feb?style=for-the-badge)
![Platform](https://img.shields.io/badge/Windows-10%20%7C%2011-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-GPL--3.0-2ea44f?style=for-the-badge)

![Input](https://img.shields.io/badge/in-JXR%20scRGB%20%C2%B7%20HDR10-e10098?style=for-the-badge)
![Output](https://img.shields.io/badge/out-JPEG%20%C2%B7%20TIFF%20%C2%B7%20H.264-fe7a16?style=for-the-badge)
![GPU](https://img.shields.io/badge/video-NVDEC%20%E2%86%92%20NVENC-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Tests](https://img.shields.io/badge/tests-1%2C435-5c2d91?style=for-the-badge)

*Formerly **NVIDIA HDR Converter GUI**. `NHC.py` still starts it.*

</div>

---

## Why this exists

You play in HDR. The light through a window is so right that you stop and press the capture key. What lands on your disk is a `.jxr`, or an HDR10 `.mp4`: files that hold a sun at a thousand nits and a cellar at half a nit, exactly as your screen showed them.

Then you try to show it to someone. Forums, chats and most screens speak SDR: one white, 255 levels, nothing above. So the capture has to be converted, and that is where pictures get hurt. The usual converter treats a capture like a raw photograph: it analyses it, re-exposes it, and hands back a new picture. That is how blacks end up lifted into a grey veil, how a night scene comes out as an evening, how a whole frame dims because one lamp was very bright.

But a game's HDR frame is not raw material. Somebody graded it. The shadows are where an artist put them.

**HDR Blaster starts from the opposite rule: the picture is finished work.** Everything that fits under your screen's white is written as bright as you saw it. Shadows and mid-tones are not touched. Only the light that SDR cannot hold is negotiated, and it is negotiated so that a sun stays brighter than the sky around it.

And because "trust me" is not a method, the window shows both sides: the capture in real HDR on the left, and on the right the file you are about to write. Not a rendering of it. The file.

---

## What makes it different

- 🎯 **Faithful by default.** A tone the game showed at 20 nits is written as 20 nits of your screen's white, whatever surrounds it. The only thing it measures on a picture is where its lights end. On the sample captures, the shadows of the result sit 0.00 level of 255 from the source's.
- 🪞 **The picture on the right is the file.** A capture is rendered once at its real size, dithered, encoded, and the *decoded file* is what you look at. **Convert** writes those very bytes. For a video, **Preview encoded MP4** shows the frames of the actual encode before you save it.
- 🌅 **Lights keep their shape.** Most roll-offs arrive flat at white, so a sun and its sky end up the same white. Here the curve keeps a positive slope up to the scene's peak: on a Cyberpunk sunrise the solar disc stays 9.6 % brighter than the sky next to it (22.4 % in the HDR source; 1.8 % with the standard roll-off).
- 🌈 **A neon stays a neon.** The hue of a light does not move as it is brought under white, and a saturated light keeps its dark channel dark instead of washing out to pastel.
- 🖥️ **It asks your screen what white means.** On an HDR display, Windows decides how bright SDR white is. HDR Blaster reads that setting for the screen the window is on, so the two halves of the divider agree, and so does the file.
- ⚡ **Video stays on the graphics card.** Decoded by NVDEC, rendered, encoded by NVENC, the three side by side. A one-minute 1440p recording at 120 fps converts in about 1.3 minutes on an RTX 4090. Audio is copied untouched, timestamps are the source's, and the result is never heavier than the source.
- 🧪 **Fast paths have to prove themselves.** Every accelerated stage is used only after it has produced, on *your* machine, exactly the bits of the plain one. If it does not, the plain one carries on, more slowly, and the log says so.
- 🔁 **Same picture, same file.** The same capture with the same settings gives the same file, byte for byte, today and next year. And nothing is ever overwritten: results are named `<name>_SDR`, numbered if that name is taken.

---

## Under the hood

For the curious, here is what happens between the capture and the file, and why each step was worth some care.

### 1. What "white" means

A JXR capture is floating-point **scRGB**: linear light, 1.0 = 80 nits, values far above 1 for lights and below 0 for colours outside BT.709. An SDR file has a single ceiling: white. The whole conversion hangs on one question: *how bright is that white?*

On an HDR display the answer is a slider in Windows ("SDR content brightness"), often several hundred nits. HDR Blaster reads it for the screen its window is on and anchors everything to it: a grey of 40 nits is written at the level your SDR window shows at 40 nits. That is why, wherever the scene fits under white, the left and the right of the divider are the same picture. Without an HDR screen to ask, the anchor is 203 nits, the reference white of ITU-R BT.2408; `--white` sets it on the command line.

### 2. The shoulder

A pixel whose brightest channel is half a stop below white, or lower, is returned as it came, to the bit. Above, a curve in log luminance leaves with slope one, bends smoothly, then holds a **constant positive slope** up to the scene's peak, which lands on white. (The peak is the level only one pixel in ten thousand exceeds: a stray pixel must not dim a whole picture.) A positive slope is what keeps two bright clouds distinct, and a sun distinct from both.

Colour is handled per pixel with two channels and a rule: the brightest and the weakest channel each go through the curve, and the middle one is placed between them where it stood in the scene. The hue cannot move. A glow that exceeds white on its three channels burns towards white, as it should; a neon, whose weakest channel is far below white, keeps it there.

The honest cost: lights make room for one another. What an SDR view would cut to plain white is written around 240 of 255 rather than 255. That is the price of seeing what was inside it.

### 3. Colours a file cannot hold

Games grade in wide gamuts, and scRGB stores what falls outside BT.709 as negative numbers. Cutting them at zero is the easy way and the wrong one: it adds light and shifts the hue. The values are kept signed from the decoder on, and brought into gamut **at constant luminance**; the Perceptual operator does it in ICtCp, along a line of constant hue. On an HDR screen, the source view shows them as recorded.

### 4. Video: measure first, then render

A video is read twice. The first pass measures each frame and finds the cuts; what it measured is then steadied *within each shot*, so the picture does not breathe when a light crosses the frame, and does change when the shot does. The second pass renders with those frozen parameters while decoding and encoding run alongside. If the card has two hardware encoders (an RTX 4090 does), both work, in segments of 250 frames, and the file is the same whether they work at once or in turn.

Each stage of a frame, some ninety tensor operations when written plainly, is fused into one kernel on the card. Fast, and dangerous: a fused multiply-add rounds differently from a multiply followed by an add. So every fused stage is held, on its first frame and on inputs built to be difficult, against the plain stage; only identical bits earn it the job. The hardware decoder is treated the same way: its first frame is compared with the processor's.

At a lower frame rate, frames are dropped, never invented: for each tick of the new clock the nearest frame is kept, with its own timestamp. Optional motion blur adds up the light of the dropped frames, in linear light, as a longer exposure would have, and never across a cut.

### 5. Looks, operators, and a small network

**Natural**, **Punchy** and **Golden** are Faithful, graded above the shadows: softer with fading highlights; more contrast and colour; punchy in a warm light. In the Advanced panel, the classical operators are held to the same rules (the scene's own shadows, mid-grey where it was, the peak on white), which leaves each one only the trait it is known for:

| Operator | Character |
|---|---|
| **Reinhard** | soft: untouched up to mid-grey, a gentle shoulder, bright colours go pastel |
| **Hable** | film: more contrast in the mid-tones, muted colours, lights that fade |
| **ACES filmic (Narkowicz)** | vivid: strong contrast, rich mid-tones, lights that burn to white |
| **Drago03** | flat: the whole range above the shadows compressed evenly |
| **Perceptual** | luminous: one stop reserved for bright structure, chroma fitted in ICtCp |
| **Guided local contrast** | local: large shapes rolled off, their relief reinforced |
| **Adaptive** / **Auto-select** | two operators blended by brightness / one chosen from the picture's statistics |
| **Neural (AI)** | a 150 K-parameter bilateral grid (HDRNet-style) taught by Punchy |

The network renders the tones in between; its deep shadows and its lights are bound to the look that taught it, so it cannot veil a black or flatten a sky. The training code and the recipe are in [`training/`](training/README.md). **AI colour** is a separate, optional retouch: an image-adaptive 3D LUT (Zeng et al., TPAMI 2020) whose small classifier blends three basis LUTs for each picture.

```
  .jxr capture                         HDR10 recording
       │                                     │  NVDEC
       ▼                                     ▼
  signed scRGB  ◄──── same engine ────►  signed scRGB  ──►  pass 1: measure, find the cuts
       │                                     │                      │ parameters, steady per shot
       ▼                                     ▼                      ▼
  look or operator, anchored on the screen's white  ◄───────────────┘
       │                                     │
       ▼                                     ▼
  sRGB, dither, JPEG / TIFF            sRGB, 8-bit planes, NVENC (H.264) + the source's audio
       │                                     │
       ▼                                     ▼
  decoded again: that is your preview   decoded again on demand: "Preview encoded MP4"
```

---

## War stories

A few lines of code, and what it took to find them.

- **The sunrise that came out peach.** A Cyberpunk sunrise: a fifth of the frame is glow, far above white. The first Faithful brought lights under white with one gain read on the brightest channel. It kept every colour whole and paid in luminance: 999 / 852 / 684 nits were written 255 / 238 / 216, a flat peach surface where the eye saw light. Giving the weakest channel its own trip through the curve fixed the glow. Then the sun itself turned out to have vanished into its sky, because a curve that arrives flat at white makes everything near the peak equal. Hence the shoulder that never goes flat.
- **The preview that was not the file.** Two faults, stacked. A capture opened in a maximised window could be rendered for the size the window had *while it was loading* (1170 × 658) and stretched to its place (2182 × 1227): grass and the outlines of wind turbines lost their detail. And the panel showed the picture before dither and JPEG compression, up to 15 levels of 255 away from what was written. The fix was not a better approximation of the file. It was to show the file.
- **The inspector that lied by one level.** The MP4 preview, built to show exactly what the file holds, showed every pixel one level darker, greys included. The file was right; the library's default conversion to RGB truncates instead of rounding. A tool made to remove a discrepancy had brought its own.
- **The byte that changed every second.** The executable and the source code, same capture, same pixels, two files one byte apart. The sRGB profile embedded in each picture was stamped with the time it was made. It now carries a fixed date, and a file depends on its picture and on nothing else.
- **Two screens, two whites.** With two HDR monitors set to different brightness, the preview asked the screen the window was on, and the conversion asked the first one. A grey of 40 nits shown at 102 was written at 82. One question is now asked once, of one screen.
- **Thirty seconds to open a window.** The single-file executable was an archive in disguise: at every launch it unpacked 2.6 GB of libraries to the temporary folder, and deleted them on the way out, when it remembered to. A folder with the libraries already unpacked opens its window in about a second.
- **The pupil who had learned yesterday's lesson.** After the highlight curve changed, the neural operator was still rendering the old one: 3.5 levels darker than its own teacher in the mid-tones. Nothing was broken, every test passed. It had simply been taught by a look that no longer existed, and had to go back to school.

None of this shows in a feature list. It is the difference between a file that looks plausible and a file that is the picture you saw.

---

## Get started

1. Download `HDR_Blaster_v3.1.3-portable-win-x64.zip` from **[Releases](https://github.com/5ymph0en1x/NVIDIA-HDR-Converter-GUI/releases)**.
2. Unzip it anywhere and start `HDR_Blaster_v3.1.3.exe`. Keep it next to its `_internal` folder: the two go together. No installer, no Python, no CUDA Toolkit; only a recent NVIDIA driver.
3. Drop a capture, a recording, several files or a whole folder on the window.
4. Look. The HDR source is on the left of the divider, the SDR result on the right; drag the divider. For a video, the timeline picks the frame.
5. Pick a look if you wish, and press **Convert**.

The result is written next to the source as `<name>_SDR.jpg` or `<name>_SDR.mp4`. With several files a queue opens on the left: one row per file, with its picture and where its conversion stands. Everything else lives in the **Advanced** panel: a precise operator, exposure and gamma, sharpness, AI colour, JPEG / TIFF / both, video quality, size and frame rate, a destination folder, GPU or CPU.

### Without the window

```bash
python NHC.py --convert capture.jxr [--method punchy] [--format JPEG|TIFF|Both] [--white 203]
python NHC.py --convert-video recording.mp4 [--method punchy] [--quality high|balanced|compact] [--height 1080] [--fps 60] [--blur]
python NHC.py --version
```

Both render Faithful unless told otherwise (`--method auto` lets the engine choose) and write `<name>_SDR` next to the source. A video is neither sharpened nor AI-coloured unless `--edge` or `--color` ask for it.

### From source

```bash
git clone https://github.com/5ymph0en1x/NVIDIA-HDR-Converter-GUI.git
cd NVIDIA-HDR-Converter-GUI
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python NHC.py
```

Python 3.14 or later. Install PyTorch first, from its own index, to get the CUDA build.

---

## What goes in, what comes out

| | Format | Notes |
|---|---|---|
| **In** | JXR capture | floating-point scRGB, from the NVIDIA app or Windows |
| **In** | HDR10 recording | PQ, BT.2020, 10-bit 4:2:0, in MP4, MKV or MOV |
| **Out** | JPEG | 4:4:4, dithered, sRGB profile |
| **Out** | TIFF | 16-bit, sRGB profile; the window adds the scene's linear light as a float TIFF |
| **Out** | MP4 | H.264, BT.709 primaries and matrix, sRGB transfer, the source's audio |

SDR, HLG and Dolby Vision files are refused, with an explanation rather than a wrong picture.

## Requirements and speed

- **Windows 10 or 11.** Built and tested there.
- **An NVIDIA card**, recommended. Without one everything runs on the processor, correctly and slowly.
- **A display with HDR switched on** to see the source in real HDR. Elsewhere it is shown cut at the window's white, and labelled so.

Measured on an RTX 4090, on a one-minute 1440p recording at 120 fps (7,142 frames):

| | Time |
|---|---|
| Whole conversion, every frame | 1.3 min |
| At 60 fps | 1.0 min |
| At 30 fps | 0.95 min |
| A setting changed on a 1440p capture, until the new file is on screen | 0.3 s |
| From double-click to window | 1 s |

## Build, tests, layout

A standalone folder is built with PyInstaller; [`BUILD.md`](BUILD.md) has the recipe and the traps.

```bash
pip install pytest
python -m pytest
```

1,435 tests. They want real captures: put a few `.jxr` files in `samples/` first. Reading a capture, the colour models and the files written are checked against a frozen copy of the original application (`tests/legacy/`).

```
NHC.py                  the launcher
hdr_blaster/engine/     the renderings, the operators, Auto-select, the 3D LUT, sharpening
hdr_blaster/io/         JXR reading, still-image writing
hdr_blaster/video/      HDR10 decoding, analysis, fused stages, encoding
hdr_blaster/ui/         the window, the queue, the HDR surface
training/               the neural tone mapper: model, training code, shipped checkpoint
models/3dlut/           the 3D LUT weights
```

If something goes wrong, the window writes its log to `%LOCALAPPDATA%\HDR Blaster\hdr_blaster.log`; the commands write `hdr_converter.log` in the current folder. Out of GPU memory? Switch on *Half precision*, or switch off *Use the graphics card*, in the Advanced panel.

What changed from one version to the next is told in the **[release notes](https://github.com/5ymph0en1x/NVIDIA-HDR-Converter-GUI/releases)**.

---

## License & credits

**Free and open source**, under the [GNU GPL v3](LICENSE).

- **Hui Zeng et al.** for the [Image-Adaptive 3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) (TPAMI 2020); pretrained weights used under Apache 2.0
- **MIT-Adobe FiveK** (Bychkovsky et al.), the dataset that model learned from
- **Gharbi et al.** for HDRNet, whose bilateral grid the learned operator borrows
- **[PyTorch](https://pytorch.org/)**, which does the arithmetic, on the card or off it
- **[Qt for Python](https://www.qt.io/qt-for-python)** for the window, **[PyAV](https://github.com/PyAV-Org/PyAV) / [FFmpeg](https://ffmpeg.org/)** for video
- **[imagecodecs](https://github.com/cgohlke/imagecodecs)** and **tifffile** for JXR and TIFF
- **ITU-R BT.2100 and BT.2408**, for PQ, ICtCp and a white everyone can agree on

---

<div align="center">

**For everyone who has stopped in the middle of a game because the light was beautiful.**

*If the picture you share is the picture you saw, the job is done.*

</div>

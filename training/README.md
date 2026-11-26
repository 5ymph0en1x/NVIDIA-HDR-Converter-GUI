# NeuralToneMapper

HDR linear → SDR sRGB neural tone mapper, designed for fidelity to the
original HDR scene (preserving local contrast, hue, and dynamic-range
structure) rather than photographic "pleasing-ness".

The model is a small HDRNet-style bilateral guided affine transform
(~200 K parameters, sub-millisecond at 4K on an RTX 4090).

## Layout

| File           | Purpose                                                       |
|----------------|---------------------------------------------------------------|
| `model.py`     | `NeuralToneMapper` (bilateral grid + guide map + slicing)     |
| `losses.py`    | `CompositeLoss` (pixel + struct + chroma + range)             |
| `dataset.py`   | `SyntheticHDRDataset`, `HDRTV4KDataset`, `UserPairsDataset`   |
| `train.py`     | Training entry point (mixed precision, cosine LR, validation) |
| `infer.py`     | Run a checkpoint on JXR files                                 |
| `requirements.txt` | Minimal training deps                                     |

## Units

The model takes **absolute luminance in nits**, BT.709 primaries. Every data
source converts to it (`dataset.py`), and each checkpoint records it
(`input_unit`). The application reads that field and scales its own working
space accordingly; the checkpoint shipped before the unit was recorded keeps
being fed as it always was.

A checkpoint taught by a look also records `anchor_white_nits`: the white its
teacher rendered against (203 nits). The looks keep the scene's exposure, so
the model learns a fixed rendering, not an exposure. The application shows it
each scene as a screen with that white would: luminance x 203 / the screen's
own SDR white.

## Training on real HDR frames (recommended)

There is no ready-made "correct SDR" for a game's HDR picture, and the public
HDR/SDR research sets are hard to obtain and restricted to academic use. The
route that works on your own machine:

1. **Build frame caches** (once). Every source is reduced to whole frames of
   512x288, in nits, in one memory-mapped file, so the GPU never waits for a decoder.

   ```powershell
   # your own HDR recordings from the NVIDIA app (only read, never modified)
   python prepare_cache.py videos "$env:USERPROFILE\Videos\NVIDIA" datasets\cache\games

   # 995 real-world HDR panoramas from Poly Haven, CC0 (about 6.4 GB to download)
   python fetch_polyhaven.py --out datasets\polyhaven --resolution 2k
   python prepare_cache.py hdris datasets\polyhaven datasets\cache\polyhaven

   # ordinary photos (DIV2K...), turned into plausible HDR
   python prepare_cache.py images datasets\DIV2K_train_HR datasets\cache\div2k
   ```

2. **Train with a teacher.** The SDR target is rendered on the GPU, for every
   batch, by a look of the application itself (`punchy`, `natural`,
   `golden`), from the very frame the network receives. The looks keep the
   scene's exposure (a grey at 18 % of white stays there), so the target of a dark
   frame is dark. Each frame is first shown at a random exposure and display peak:
   that is what covers screens brighter or dimmer than the teacher's white.

   ```powershell
   python train.py --cache-dir datasets\cache\games --cache-dir datasets\cache\polyhaven ^
                   --cache-dir datasets\cache\div2k --teacher punchy ^
                   --aug-ev 2.5 --out-dir checkpoints_anchored --batch-size 64 --epochs 100 --amp
   ```

Whole videos and whole panoramas are held out for validation, never single
frames: neighbouring frames of a recording are near-duplicates.

Licences: Poly Haven is CC0. DIV2K is for academic research; weights trained
with it inherit that restriction, so leave it out for a model you publish.

## Setup

```powershell
cd training
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use a Python that has PyTorch CUDA wheels available (currently 3.11 or
3.12; the project root `pyproject.toml` pins 3.14 for the app itself
but training can use an independent interpreter).

Smoke-test the architecture:

```powershell
python model.py
# Expected:
#   OK  shape=(2, 3, 512, 512)  range=[0.000, 1.000]
#   OK  loss=...  params=~200,000
```

## Training

### Phase 1 — Synthetic warm-up (no external dataset)

Point the trainer at any folder of well-exposed sRGB JPEGs (your own
photos, an existing image dump — anything). The dataset fakes HDR by
inverting an exposure curve, giving the network HDR-shaped inputs whose
canonical SDR target is the original JPEG.

```powershell
python train.py --synthetic-dir D:\photos --epochs 20 --amp
```

Expect ~30 min on a 4090 for 20 epochs over 5 K images. The result is
a baseline model that has learned plausible HDR→SDR mappings but
hasn't seen real high-dynamic-range data yet.

### Phase 2 — Real HDR-SDR pairs (HDRTV4K)

Download the [HDRTV4K dataset](https://github.com/cszhilu1998/HDRTVNet)
or equivalent and arrange it as:

    D:\HDRTV4K\
        HDR\  *.png  (16-bit PQ-encoded BT.2020)
        SDR\  *.png  ( 8-bit sRGB, same stems)

Start from the synthetic weights and train on real data:

```powershell
python train.py --init-from checkpoints\best.pth ^
                --out-dir checkpoints_hdrtv ^
                --hdrtv4k-dir D:\HDRTV4K ^
                --synthetic-dir D:\photos ^
                --epochs 50 --amp --batch-size 8
```

Expect 4–8 hours on a 4090 depending on dataset size.

### Phase 3 — Personal fine-tune (optional)

Build a small set of JXR captures alongside SDR JPEGs that you've hand-
retouched in Lightroom/DaVinci to match your taste:

    D:\my_pairs\
        hdr\  shot001.jxr  shot002.jxr ...
        sdr\  shot001.jpg  shot002.jpg ...   (same stems)

200 hand-curated pairs are enough to bias the model toward your specific
"gaming HDR dramatic" aesthetic.

```powershell
python train.py --init-from checkpoints_hdrtv\best.pth ^
                --out-dir checkpoints_user ^
                --user-pairs-dir D:\my_pairs ^
                --epochs 10 --lr 5e-5 --amp
```

`--init-from` starts a new run from trained weights, with a fresh optimizer
and learning-rate schedule. `--resume` continues an interrupted run exactly
where it stopped (same schedule, same epoch count) and refuses a run that is
already finished. A share of the data (`--val-fraction`, 5 % by default) is
held out; the epoch with the lowest held-out loss is kept as `best.pth`.

```powershell
python train.py --synthetic-dir D:\photos --epochs 20 --amp --resume checkpoints\latest.pth
```

## Architecture sketch

```
   HDR linear (B,3,H,W) unbounded
        |
        v  log1p / log(1+1e4)   (compress to [0,1])
        |
        +----------------+
        |                |
   resize 256x256    full-res
        |                |
   CoefficientPred   GuidePred
        |                |
   bilateral grid     guide(x,y)
   (B,12,gd,gh,gw)   (B,1,H,W)
        |                |
        +-> bilateral slice (F.grid_sample 3D)
                    |
                    v
            (B,12,H,W) per-pixel 3x4 affine
                    |
                    v  apply: out_c = W_ci * log_in_i + b_c
                    |
                    v  clamp [0,1]
            SDR (B,3,H,W) in [0,1]
```

The coefficient predictor is initialized to output identity affines, so
an untrained model passes through `log_compress(input)` almost as-is
(very dark output for HDR inputs, since most pixel values are in the
low-log range — but no NaNs and no surprises).

## Loss components

| Term       | Default weight | Intent                                                       |
|------------|----------------|--------------------------------------------------------------|
| `pixel`    | 1.0            | L1 in sRGB output space                                      |
| `struct`   | 2.0            | Per-pixel gradient alignment on log-luminance, weighted by HDR gradient strength — the fidelity term |
| `chroma`   | 0.5            | Preserves hue between scale-normalized HDR and SDR           |
| `range`    | 0.3            | Avoids clipping, encourages full [0,1] usage                 |

Override at the CLI with `--loss-weights '{"struct": 3.0}'` to push the
model harder on structural fidelity.

## Inference

```powershell
python infer.py --ckpt checkpoints\latest.pth ^
                --jxr-dir "C:\Users\me\Videos\NVIDIA" ^
                --out-dir out\
```

A/B compare the outputs against `NHC.py`'s classical tone mappers on
the same JXRs to validate the model is doing what you want.

## Integration into NHC.py

Once a checkpoint reproducibly beats the classical operators on your
content, add a `NeuralToneMapper` option to `AdvancedToneMapper`:

1. Load `checkpoints/latest.pth` into a `NeuralToneMapper` instance
   inside `OptimizedJXRLoader.__init__` (alongside the existing tone
   mappers).
2. Add `"neural"` to `SUPPORTED_TONE_MAPS` and the GUI dropdown.
3. In `apply_tone_mapping`, branch on `method == "neural"` and call the
   neural model on the HDR linear tensor instead of running a
   curve-based operator.

We'll wire this in once a trained checkpoint exists.

"""
Datasets for HDR -> SDR neural tone-mapping training.

Three sources, all yielding `{"hdr": (3, H, W) linear, in NITS, BT.709
                                     primaries, non-negative,
                              "sdr": (3, H, W) sRGB in [0, 1]}`:

1. SyntheticHDRDataset    -- generated from any folder of sRGB JPEGs by
                             inverting a tone-map. Zero external setup
                             required. Good for v1 sanity training.

2. HDRTV4KDataset         -- real HDR-SDR pairs. Expected layout:
                                root/HDR/   *.png   (16-bit PQ-encoded)
                                root/SDR/   *.png   (8-bit sRGB)
                             Filenames must match between HDR and SDR.

3. UserPairsDataset       -- the user's own JXR captures alongside
                             manually retouched SDR JPEGs:
                                root/hdr/   *.jxr
                                root/sdr/   *.jpg   (same basename)

ConcatDataset is supported — mix sources freely at training time.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import imagecodecs  # for JXR support in UserPairsDataset
    _HAS_IMAGECODECS = True
except ImportError:
    _HAS_IMAGECODECS = False


# ---------- units ------------------------------------------------------------
# One convention for every source and for the application: absolute luminance
# in nits (see NeuralToneMapper.INPUT_UNIT). Mixing sources that disagree on
# the unit teaches the network nothing useful about brightness.

SDR_WHITE_NITS = 200.0    # what sRGB white stands for in synthetic data, before the random EV
                          # (a usual HDR 'paper white' in games)
SYNTHETIC_EV_RANGE = (-3.0, 1.5)   # white from 25 to ~570 nits: covers dark and bright captures
HIGHLIGHT_HEADROOM = 6.0  # clipped whites are extrapolated up to this many times white
SCRGB_WHITE_NITS = 80.0   # scRGB (NVIDIA JXR captures): 1.0 = 80 nits
PQ_PEAK_NITS = 10000.0    # PQ code value 1.0

BT2020_TO_BT709 = np.array([
    [1.660491, -0.587641, -0.072850],
    [-0.124550, 1.132900, -0.008349],
    [-0.018151, -0.100579, 1.118730],
], dtype=np.float32)


def bt2020_to_bt709_np(rgb: np.ndarray) -> np.ndarray:
    """Linear BT.2020 -> linear BT.709. Out-of-gamut (negative) values are clipped,
    as the application's JXR loader does."""
    return np.clip(rgb @ BT2020_TO_BT709.T, 0.0, None).astype(np.float32)


def read_png(path) -> np.ndarray:
    """Read a PNG at its real bit depth. Pillow opens a 16-bit RGB PNG as 8 bits,
    which silently destroys PQ-encoded HDR frames."""
    if not _HAS_IMAGECODECS:
        raise ImportError("imagecodecs is required to read 16-bit PNG frames")
    arr = imagecodecs.png_decode(Path(path).read_bytes())
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3]


def load_jxr_nits(path) -> np.ndarray:
    """(H, W, 3) float32, linear BT.709 in nits, from an NVIDIA scRGB JXR capture.

    Same decode as the application's loader (RGBA float, alpha dropped,
    negatives clipped), expressed in nits."""
    if not _HAS_IMAGECODECS:
        raise ImportError("imagecodecs is required to read JXR captures")
    arr = imagecodecs.jpegxr_decode(Path(path).read_bytes())
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr.clip(min=0.0) * SCRGB_WHITE_NITS, dtype=np.float32)


def _resize_at_least(arr: np.ndarray, ps: int) -> np.ndarray:
    """Bicubic upscale of a float (H, W, 3) array so both sides reach ``ps``."""
    h, w = arr.shape[:2]
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(max(h, ps), max(w, ps)), mode="bicubic", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).clamp(min=0.0).numpy()


# ---------- color helpers ----------------------------------------------------

def srgb_to_linear_t(x: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB EOTF (gamma decode). Input [0, 1] -> linear [0, 1]."""
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def pq_to_linear_np(x: np.ndarray) -> np.ndarray:
    """
    BT.2100 PQ EOTF. Input in [0, 1] (decoded PQ values) -> linear nits/10000.
    """
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    x = np.clip(x, 0.0, 1.0)
    num = np.maximum(np.power(x, 1.0 / m2) - c1, 0.0)
    den = c2 - c3 * np.power(x, 1.0 / m2)
    return np.power(num / np.maximum(den, 1e-8), 1.0 / m1)  # in [0, 1]


def _random_crop(arr_h, arr_s, ps: int):
    """Take the same random patch from two equally-sized arrays."""
    h, w = arr_h.shape[-3], arr_h.shape[-2]
    if h < ps or w < ps:
        return None  # caller should resize or skip
    y = random.randint(0, h - ps)
    x = random.randint(0, w - ps)
    return arr_h[y : y + ps, x : x + ps], arr_s[y : y + ps, x : x + ps]


def _to_tensor_hwc(arr: np.ndarray) -> torch.Tensor:
    """(H, W, C) numpy -> (C, H, W) float tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).contiguous().float()


# ---------- 1. Synthetic dataset ---------------------------------------------

def fake_hdr_from_srgb(srgb: torch.Tensor, ev_range=SYNTHETIC_EV_RANGE) -> torch.Tensor:
    """
    Generate a plausible HDR linear image from a single sRGB JPEG by:
        1. sRGB -> linear gamma decode
        2. random exposure scaling (-EV to +EV)
        3. specular extrapolation for near-white regions
           (recovers some headroom that the JPEG clipped)

    NOTE: this is a TRAINING signal, not a real HDR recovery. The point
    is to give the network HDR-shaped data that maps back to the
    original sRGB.
    """
    linear = srgb_to_linear_t(srgb)
    ev = random.uniform(*ev_range)
    scale = 2.0 ** ev
    # Extrapolate saturated regions to mimic real HDR highlights: a clipped white
    # reaches HIGHLIGHT_HEADROOM times white. The extrapolation is applied BEFORE
    # the exposure scaling, so highlights stay proportional to the scene (it used
    # to be added after it, as a fixed amount whatever the exposure).
    near_white = (linear > 0.95).any(dim=0, keepdim=True).expand_as(linear).float()
    excess = (linear - 0.95).clamp(min=0.0)
    gain = math.log(HIGHLIGHT_HEADROOM) / 0.05
    hdr = (linear + near_white * (torch.exp(excess * gain) - 1.0)) * scale
    return hdr * SDR_WHITE_NITS


class SyntheticHDRDataset(Dataset):
    """
    Wraps a folder of sRGB images. Each item yields:
        hdr: synthetic HDR (linear, in nits)
        sdr: original sRGB image (target), in [0, 1]
    """

    EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, image_dir: str, patch_size: int = 512, ev_range=SYNTHETIC_EV_RANGE):
        self.root = Path(image_dir)
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in self.EXTS)
        if not self.paths:
            raise FileNotFoundError(f"No images under {image_dir}")
        self.ps = patch_size
        self.ev_range = ev_range

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w < self.ps or h < self.ps:
            new_w = max(w, self.ps)
            new_h = max(h, self.ps)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = img.size
        x = random.randint(0, w - self.ps)
        y = random.randint(0, h - self.ps)
        img = img.crop((x, y, x + self.ps, y + self.ps))

        sdr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        hdr = fake_hdr_from_srgb(sdr, ev_range=self.ev_range)
        return {"hdr": hdr, "sdr": sdr}


# ---------- 2. HDRTV4K-style real dataset ------------------------------------

class HDRTV4KDataset(Dataset):
    """
    Pre-extracted HDR/SDR frame pairs. Expected layout:

        root/HDR/  *.png  (16-bit PQ-encoded, BT.2020)
        root/SDR/  *.png  (8-bit sRGB pair, same stem)

    HDR frames are decoded from PQ to linear light in nits and converted from
    BT.2020 to BT.709 primaries, the gamut of the SDR target and of the captures.
    SDR frames are loaded as [0, 1] sRGB (the training target).
    """

    def __init__(self, root: str, patch_size: int = 512):
        self.root = Path(root)
        hdr_dir = self.root / "HDR"
        sdr_dir = self.root / "SDR"
        if not hdr_dir.is_dir() or not sdr_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {hdr_dir} and {sdr_dir} -- check HDRTV4K layout"
            )
        self.pairs = []
        for h in sorted(hdr_dir.glob("*.png")):
            s = sdr_dir / f"{h.stem}.png"
            if s.exists():
                self.pairs.append((h, s))
        if not self.pairs:
            raise FileNotFoundError(f"No matching HDR/SDR pairs in {root}")
        self.ps = patch_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        h_path, s_path = self.pairs[idx]
        hdr_arr = read_png(h_path)
        if hdr_arr.dtype != np.uint16:
            raise ValueError(
                f"{h_path.name}: expected a 16-bit PQ-encoded PNG, got {hdr_arr.dtype}")
        hdr_arr = pq_to_linear_np(hdr_arr.astype(np.float32) / 65535.0) * PQ_PEAK_NITS
        hdr_arr = bt2020_to_bt709_np(hdr_arr)
        sdr_arr = np.array(Image.open(s_path).convert("RGB")).astype(np.float32) / 255.0

        cropped = _random_crop(hdr_arr, sdr_arr, self.ps)
        if cropped is None:
            # Frame smaller than patch: resize once
            hdr_arr = _resize_at_least(hdr_arr, self.ps)
            sdr_arr = _resize_at_least(sdr_arr, self.ps).clip(0.0, 1.0)
            cropped = _random_crop(hdr_arr, sdr_arr, self.ps)

        hdr_p, sdr_p = cropped
        return {"hdr": _to_tensor_hwc(hdr_p), "sdr": _to_tensor_hwc(sdr_p)}


# ---------- 3. User pairs (JXR + retouched JPEG) -----------------------------

class UserPairsDataset(Dataset):
    """
    The user's own captures: linear HDR JXR + corresponding retouched SDR JPEG.

    Layout:
        root/hdr/  *.jxr
        root/sdr/  *.jpg   (same stem as the matching .jxr)

    Requires `imagecodecs` for JXR decoding. The JXR loader replicates the
    decode path used by `OptimizedJXRLoader.load_jxr` in NHC.py (RGBA
    float32, returns the RGB channels in scene-linear).
    """

    def __init__(self, root: str, patch_size: int = 512):
        if not _HAS_IMAGECODECS:
            raise ImportError("imagecodecs is required for UserPairsDataset")
        self.root = Path(root)
        hdr_dir = self.root / "hdr"
        sdr_dir = self.root / "sdr"
        self.pairs = []
        for h in sorted(hdr_dir.glob("*.jxr")):
            for ext in (".jpg", ".jpeg", ".png"):
                s = sdr_dir / f"{h.stem}{ext}"
                if s.exists():
                    self.pairs.append((h, s))
                    break
        if not self.pairs:
            raise FileNotFoundError(f"No matching .jxr/.jpg pairs under {root}")
        self.ps = patch_size

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_jxr(self, path: Path) -> np.ndarray:
        return load_jxr_nits(path)

    def __getitem__(self, idx: int) -> dict:
        h_path, s_path = self.pairs[idx]
        hdr_arr = self._load_jxr(h_path)
        sdr_arr = np.array(Image.open(s_path).convert("RGB")).astype(np.float32) / 255.0
        cropped = _random_crop(hdr_arr, sdr_arr, self.ps)
        if cropped is None:
            raise RuntimeError(
                f"Pair smaller than patch_size={self.ps}: {h_path.name}"
            )
        hdr_p, sdr_p = cropped
        return {"hdr": _to_tensor_hwc(hdr_p), "sdr": _to_tensor_hwc(sdr_p)}

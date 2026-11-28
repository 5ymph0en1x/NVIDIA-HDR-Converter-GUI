"""The SDR samples shared by the comparison window and the still-image writers."""
import struct
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
import tifffile
import torch


@lru_cache(maxsize=1)
def srgb_profile() -> bytes:
    """The sRGB profile every still carries. The library stamps it with the time it is made:
    the same capture then gave files one byte apart from one run to the next. Its date is
    set to a fixed one (the header holds no identifier computed over it)."""
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    profile[24:36] = struct.pack(">6H", 2000, 1, 1, 0, 0, 0)       # year, month, day, hour, minute, second
    return bytes(profile)


def quantize(picture, depth):
    """Round with triangular dither of one output code, independent of other renders."""
    maximum = (1 << depth) - 1
    image = picture.detach().to(device="cpu", dtype=torch.float32).contiguous()
    rng = torch.Generator(device="cpu").manual_seed(42)
    noise = torch.rand(image.shape, generator=rng) - torch.rand(image.shape, generator=rng)
    image = (image + noise * (1.0 / maximum)).clamp_(0, 1)
    rgb = image[0].permute(1, 2, 0).contiguous().numpy()
    return np.rint(rgb * maximum).astype(np.uint8 if depth == 8 else np.uint16)


def encode_jpeg(picture, quality):
    with BytesIO() as buf:
        Image.fromarray(quantize(picture, 8)).save(
            buf, "JPEG", quality=max(quality, 97), subsampling=0, optimize=True, icc_profile=srgb_profile())
        return buf.getvalue()


@dataclass
class SdrStill:
    jpeg: bytes | None
    tiff: np.ndarray | None

    @classmethod
    def prepare(cls, picture, fmt, quality=97):
        if fmt not in ("JPEG", "TIFF", "Both"):
            raise ValueError(f"Unsupported output format: {fmt}")
        return cls(encode_jpeg(picture, quality) if fmt in ("Both", "JPEG") else None,
                   quantize(picture, 16) if fmt in ("Both", "TIFF") else None)

    def pixels(self, device):
        if self.jpeg is not None:
            with Image.open(BytesIO(self.jpeg)) as image:
                rgb = np.array(image.convert("RGB"))
        else:
            rgb = self.tiff
        # Normalize on the CPU before transfer so the same samples reach each backend.
        rgb = rgb.astype(np.float32) / np.iinfo(rgb.dtype).max
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

    def write_jpeg(self, path):
        Path(path).write_bytes(self.jpeg)

    def write_tiff(self, path):
        tifffile.imwrite(path, self.tiff, photometric="rgb", compression="lzw", iccprofile=srgb_profile())

"""
Frame cache: training images decoded once, stored ready to be fed to the GPU.

Decoding 2K PNGs or HEVC frames inside the training loop keeps the GPU waiting. Here
every source is reduced once to WHOLE frames of 512x288, linear BT.709 in nits, half
precision, in one memory-mapped file per source. Whole frames, not crops of full-size
frames: at inference the network decides its tone curve from a small view of the entire
picture, so that is what it must see while learning.

A cache is a folder holding
    hdr.f16     raw float16, shape (count, 3, HEIGHT, WIDTH), nits
    meta.json   count, shape, and for each frame its origin and whether it is held out

Frames are held out for validation by ORIGIN (a whole video, a whole panorama), never
frame by frame: neighbouring frames of one video are near-duplicates.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))                 # the application package, for the video reader

WIDTH, HEIGHT = 512, 288
MAX_NITS = 60000.0                                # float16 tops out at 65504
VAL_SHARE = 10                                    # one origin in ten is held out


def is_held_out(origin: str) -> bool:
    return int(hashlib.sha1(origin.encode("utf-8")).hexdigest(), 16) % VAL_SHARE == 0


class CacheWriter:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._file = open(self.folder / "hdr.f16", "wb")
        self.items = []

    def add(self, nits: torch.Tensor, origin: str, detail: str = "") -> None:
        """``nits``: (3, HEIGHT, WIDTH) linear BT.709, any device."""
        if tuple(nits.shape) != (3, HEIGHT, WIDTH):
            raise ValueError(f"expected (3, {HEIGHT}, {WIDTH}), got {tuple(nits.shape)}")
        frame = torch.nan_to_num(nits.float(), nan=0.0, posinf=MAX_NITS).clamp(0.0, MAX_NITS)
        self._file.write(frame.to(torch.float16).cpu().numpy().tobytes())
        self.items.append({"origin": origin, "detail": detail, "val": is_held_out(origin)})

    def close(self) -> int:
        self._file.close()
        meta = {"count": len(self.items), "shape": [3, HEIGHT, WIDTH], "unit": "nits", "items": self.items}
        (self.folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return len(self.items)


class FrameCache(Dataset):
    """One cache folder, one split. Items are ``{"hdr": (3, H, W) float32 nits}``."""

    def __init__(self, folder, split: str = "train", exclude=()):
        """``exclude``: frames whose origin contains one of these texts are left out
        (for instance "Desktop": recordings of the Windows desktop are not game footage)."""
        self.folder = Path(folder)
        meta = json.loads((self.folder / "meta.json").read_text(encoding="utf-8"))
        self.shape = tuple(meta["shape"])
        self.count = meta["count"]
        wanted = split == "val"
        self.indices = [i for i, item in enumerate(meta["items"])
                        if item["val"] == wanted and not any(text in item["origin"] for text in exclude)]
        self.origins = sorted({meta["items"][i]["origin"] for i in self.indices})
        self._data = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        if self._data is None:                    # opened lazily: once per DataLoader worker
            self._data = np.memmap(self.folder / "hdr.f16", dtype=np.float16, mode="r",
                                   shape=(self.count, *self.shape))
        frame = np.array(self._data[self.indices[index]], dtype=np.float32)
        return {"hdr": torch.from_numpy(frame)}

    def load_all(self) -> torch.Tensor:
        """Every frame of this split at once, (N, 3, H, W) float16."""
        data = np.memmap(self.folder / "hdr.f16", dtype=np.float16, mode="r", shape=(self.count, *self.shape))
        return torch.from_numpy(np.ascontiguousarray(data[self.indices]))


class GpuLoader:
    """All the frames of some caches held in GPU memory, served in random batches.

    A frame is 0.9 MB in half precision: ten thousand of them fit in 9 GB. Nothing is
    read, decoded or copied during training, so the GPU is never waiting for data.
    Batches are ``{"hdr": (B, 3, H, W) float32 nits}``, like a DataLoader over the caches."""

    def __init__(self, caches, device, batch_size: int, shuffle: bool, drop_last: bool = False):
        self.data = torch.cat([c.load_all() for c in caches if len(c)]).to(device)
        self.batch_size, self.shuffle = batch_size, shuffle
        self.drop_last = drop_last and len(self.data) >= batch_size

    @staticmethod
    def bytes_needed(caches) -> int:
        return sum(len(c) * 3 * HEIGHT * WIDTH * 2 for c in caches)

    def __len__(self) -> int:
        full, rest = divmod(len(self.data), self.batch_size)
        return full if self.drop_last or not rest else full + 1

    def __iter__(self):
        count = len(self.data)
        order = torch.randperm(count, device=self.data.device) if self.shuffle \
            else torch.arange(count, device=self.data.device)
        for chunk in order.split(self.batch_size):
            if self.drop_last and len(chunk) < self.batch_size:
                continue
            yield {"hdr": self.data[chunk].float()}


# ------------------------------------------------------------------- sources

def from_videos(folder, out, every_s: float = 0.5, device=None, log=print) -> int:
    """HDR10 recordings: one frame every ``every_s`` seconds, skipping near-identical ones."""
    from hdr_blaster.engine.constants import ENGINE_UNIT_NITS
    from hdr_blaster.video.colour import yuv_to_engine
    import av

    from hdr_blaster.video.reader import UnsupportedVideo, check_supported, planes_from_frame, probe
    from hdr_blaster.video.temporal import frame_signature

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = CacheWriter(out)
    videos = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".mp4", ".mkv", ".mov"))
    for path in videos:
        try:
            info = probe(path)
            check_supported(info)
        except UnsupportedVideo as exc:
            log(f"  skipped {path.name}: {exc}")
            continue
        kept, next_time, last = 0, 0.0, None
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                # Every frame must be decoded, but only the kept ones are converted and
                # sent to the GPU: that conversion costs more than the decoding itself.
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    time_s = float(frame.pts * stream.time_base)
                    if time_s < next_time:
                        continue
                    next_time = time_s + every_s
                    y, u, v = planes_from_frame(frame, device)
                    signature = frame_signature(y, info.fmt.bit_depth)
                    if last is not None and float((signature - last).abs().mean()) < 0.004:
                        continue                      # a menu, a pause, a static shot
                    last = signature
                    nits = yuv_to_engine(y, u, v, info.fmt) * ENGINE_UNIT_NITS
                    small = F.interpolate(nits, size=(HEIGHT, WIDTH), mode="area")[0]
                    writer.add(small, origin=path.name, detail=f"{time_s:.2f}s")
                    kept += 1
        except Exception as exc:                      # a damaged file must not lose the others
            log(f"  stopped {path.name} after {kept} frames: {exc}")
        log(f"  {path.name}: {kept} frames")
    return writer.close()


def read_radiance(path) -> np.ndarray:
    """A Radiance .hdr or OpenEXR picture as (H, W, 3) float32 RGB, linear."""
    import av

    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
    return _planar_to_rgb(frame)


def _planar_to_rgb(frame) -> np.ndarray:
    """PyAV hands float pictures over as planar G, B, R: put them in R, G, B order."""
    planar = frame.reformat(format="gbrpf32le")
    height, width = planar.height, planar.width
    planes = [np.frombuffer(p, dtype=np.float32).reshape(height, -1)[:, :width] for p in planar.planes[:3]]
    green, blue, red = planes
    return np.stack([red, green, blue], axis=-1)


def from_hdris(folder, out, crops: int = 4, seed: int = 0, log=print) -> int:
    """Real-world HDR panoramas: a few 16:9 views of each, away from the poles.

    Panoramas have no absolute scale, so each view is stored with its median luminance at
    10 nits; the training loop then varies exposure and display peak at random.
    """
    rng = random.Random(seed)
    writer = CacheWriter(out)
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".hdr", ".exr"))
    for n, path in enumerate(files, start=1):
        try:
            pano = torch.from_numpy(np.ascontiguousarray(read_radiance(path))).permute(2, 0, 1)
        except Exception as exc:
            log(f"  skipped {path.name}: {exc}")
            continue
        _, height, width = pano.shape
        view_h = height * 9 // 16                     # 576 rows of a 1024-row panorama
        view_w = view_h * 16 // 9
        for _ in range(crops):
            top = rng.randint(height // 8, height - view_h - height // 8)
            left = rng.randrange(width)
            columns = [(left + c) % width for c in range(view_w)]      # wraps around the seam
            view = pano[:, top:top + view_h][:, :, columns].clamp(min=0.0)
            small = F.interpolate(view[None], size=(HEIGHT, WIDTH), mode="area")[0]
            luminance = 0.2126 * small[0] + 0.7152 * small[1] + 0.0722 * small[2]
            median = float(luminance.median())
            if not math.isfinite(median) or median <= 0.0:
                continue
            writer.add(small * (10.0 / median), origin=path.name)
        if n % 100 == 0:
            log(f"  {n}/{len(files)} panoramas")
    return writer.close()


def from_images(folder, out, log=print) -> int:
    """Ordinary photos turned into plausible HDR (see ``dataset.fake_hdr_from_srgb``),
    at exposure 0: sRGB white stands for ``SDR_WHITE_NITS``, clipped whites reach six times that."""
    from PIL import Image

    from dataset import SyntheticHDRDataset, fake_hdr_from_srgb

    writer = CacheWriter(out)
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in SyntheticHDRDataset.EXTS)
    for n, path in enumerate(files, start=1):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        if width * 9 > height * 16:                   # too wide: trim the sides
            new_w = height * 16 // 9
            box = ((width - new_w) // 2, 0, (width - new_w) // 2 + new_w, height)
        else:                                         # too tall: trim top and bottom
            new_h = width * 9 // 16
            box = (0, (height - new_h) // 2, width, (height - new_h) // 2 + new_h)
        image = image.crop(box).resize((WIDTH, HEIGHT), Image.LANCZOS)
        srgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        writer.add(fake_hdr_from_srgb(srgb, ev_range=(0.0, 0.0)), origin=path.name)
        if n % 200 == 0:
            log(f"  {n}/{len(files)} images")
    return writer.close()

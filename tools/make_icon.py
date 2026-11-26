"""Make the application's icon from the author's artwork.

    python tools/make_icon.py

``app.ico`` at the root of the project is the author's picture: a 1024-pixel PNG under an
icon's name, with its background (a transparency checkerboard and a dark band) drawn into
the pixels. Windows cannot read it as an icon, which is why the taskbar showed a blank
one. This script cuts the emblem out of it, the ring and what it holds, on a truly
transparent ground, and writes it at every size Windows asks for:

    hdr_blaster/ui/icon.ico    16 to 256 pixels, for the executable, the window, the taskbar
    hdr_blaster/ui/icon.png    256 pixels, for a Qt without an ICO reader

The artwork itself is not modified.
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARTWORK = ROOT / "app.ico"
OUT = ROOT / "hdr_blaster" / "ui"

CENTRE = (515.0, 471.0)          # of the ring, in the artwork's pixels
RADIUS = 266.0                   # just outside the ring
INK = (0x17, 0x16, 0x1B)         # the window's own background, behind the snake
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def emblem(side: int = 1024) -> Image.Image:
    """The ring and what it holds, RGBA, filling a square of ``side`` pixels."""
    art = np.asarray(Image.open(ARTWORK).convert("RGB")).astype(np.float32)
    height, width = art.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.hypot(x - CENTRE[0], y - CENTRE[1])
    # What was drawn as background (checkerboard, dark band) has no colour and is not bright;
    # the ring, the snake and their highlights have one or the other.
    colourful = art.max(-1) - art.min(-1)
    drawn = np.clip((colourful - 22.0) / 30.0, 0.0, 1.0)
    drawn = np.maximum(drawn, np.clip((art.max(-1) - 190.0) / 40.0, 0.0, 1.0))
    rgb = art * drawn[..., None] + np.array(INK, dtype=np.float32) * (1.0 - drawn[..., None])
    alpha = np.clip(RADIUS - distance + 0.5, 0.0, 1.0) * 255.0
    rgba = np.dstack([rgb, alpha]).round().astype(np.uint8)
    box = tuple(int(round(v)) for v in (CENTRE[0] - RADIUS - 2, CENTRE[1] - RADIUS - 2,
                                        CENTRE[0] + RADIUS + 2, CENTRE[1] + RADIUS + 2))
    return Image.fromarray(rgba, "RGBA").crop(box).resize((side, side), Image.LANCZOS)


def main() -> None:
    large = emblem()
    pictures = [large.resize((size, size), Image.LANCZOS) for size in SIZES]
    pictures[-1].save(OUT / "icon.png")
    pictures[-1].save(OUT / "icon.ico", format="ICO", sizes=[(s, s) for s in SIZES], append_images=pictures[:-1])
    print("written:", OUT / "icon.ico", OUT / "icon.png")


if __name__ == "__main__":
    main()

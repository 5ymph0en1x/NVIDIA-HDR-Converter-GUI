"""
Build the frame caches the trainer reads (see cache.py).

    # your own HDR recordings (NVIDIA app): one frame every half second
    python prepare_cache.py videos   "C:/Users/me/Videos/NVIDIA"  datasets/cache/games

    # Poly Haven panoramas fetched with fetch_polyhaven.py
    python prepare_cache.py hdris    datasets/polyhaven           datasets/cache/polyhaven

    # ordinary photos (DIV2K...), turned into plausible HDR
    python prepare_cache.py images   datasets/DIV2K_train_HR      datasets/cache/div2k

The sources are only read. Each command prints how many frames it stored and how many
are held out for validation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for entry in (str(HERE), str(HERE.parent)):       # explicit: also works with python -I
    if entry not in sys.path:
        sys.path.insert(0, entry)

import cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["videos", "hdris", "images"])
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--every", type=float, default=0.5, help="videos: seconds between kept frames")
    parser.add_argument("--crops", type=int, default=4, help="hdris: views taken from each panorama")
    args = parser.parse_args()

    if not Path(args.source).is_dir():
        print(f"source folder not found: {args.source}")
        return 2
    started = time.time()
    if args.kind == "videos":
        count = cache.from_videos(args.source, args.out, every_s=args.every)
    elif args.kind == "hdris":
        count = cache.from_hdris(args.source, args.out, crops=args.crops)
    else:
        count = cache.from_images(args.source, args.out)
    held_out = len(cache.FrameCache(args.out, "val"))
    size = (Path(args.out) / "hdr.f16").stat().st_size / 1e9
    print(f"{count} frames ({held_out} held out) in {args.out}: {size:.2f} GB, {time.time() - started:.0f} s")
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())

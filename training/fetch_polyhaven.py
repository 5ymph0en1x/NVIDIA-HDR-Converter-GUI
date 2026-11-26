"""
Download the Poly Haven HDRIs (https://polyhaven.com, CC0 licence) as training material.

Real-world HDR panoramas: outdoor, indoor, sun, night. They bring dynamic ranges and
lighting that game captures alone do not cover, under a licence that puts no condition
on the weights trained from them.

    python fetch_polyhaven.py --out datasets/polyhaven --resolution 2k

Files already present with the right size are skipped, so the command can be re-run to
resume. Only image files are written, each under the asset's own name. Total sizes at 2k:
6.4 GB as .hdr, 13.2 GB as .exr.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://api.polyhaven.com"
USER_AGENT = "hdr-tonemap-training/1.0"
SAFE_NAME = re.compile(r"^[a-z0-9_]+$")


def get_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_one(asset: str, out_dir: Path, resolution: str, fmt: str) -> tuple[str, str, int]:
    """Returns (asset, status, bytes). Status: 'downloaded', 'present' or an error message."""
    if not SAFE_NAME.match(asset):
        return asset, "skipped: unexpected asset name", 0
    target = out_dir / f"{asset}.{fmt}"
    try:
        entry = get_json(f"{API}/files/{asset}")["hdri"][resolution][fmt]
        url, size = entry["url"], int(entry["size"])
        if not url.startswith("https://dl.polyhaven.org/"):
            return asset, f"skipped: unexpected host in {url}", 0
        if target.exists() and target.stat().st_size == size:
            return asset, "present", size
        partial = target.with_suffix(f".{fmt}.part")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
        if partial.stat().st_size != size:
            partial.unlink(missing_ok=True)
            return asset, "error: size mismatch", 0
        partial.replace(target)
        return asset, "downloaded", size
    except Exception as exc:  # network errors: report and carry on with the others
        return asset, f"error: {exc}", 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="datasets/polyhaven")
    parser.add_argument("--resolution", default="2k", choices=["1k", "2k", "4k"])
    parser.add_argument("--format", dest="fmt", default="hdr", choices=["hdr", "exr"],
                        help="Radiance .hdr (half the size, about 1 %% precision) or OpenEXR")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="only the first N assets (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = sorted(get_json(f"{API}/assets?t=hdris"))
    if args.limit:
        assets = assets[:args.limit]
    print(f"{len(assets)} HDRIs to fetch at {args.resolution} ({args.fmt}) into {out_dir}", flush=True)

    started, done, total_bytes, errors = time.time(), 0, 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed(pool.submit(fetch_one, a, out_dir, args.resolution, args.fmt) for a in assets):
            asset, status, size = future.result()
            done += 1
            total_bytes += size
            if status.startswith(("error", "skipped")):
                errors.append(f"{asset}: {status}")
            if done % 50 == 0 or done == len(assets):
                print(f"  {done}/{len(assets)}  {total_bytes / 1e9:.2f} GB  "
                      f"{time.time() - started:.0f} s  errors: {len(errors)}", flush=True)
    for line in errors[:20]:
        print("  " + line)
    print(f"done: {len(assets) - len(errors)} files, {total_bytes / 1e9:.2f} GB, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

"""
Inference helper for NeuralToneMapper.

Loads a checkpoint and tone-maps one or more JXR files to JPEG, bypassing
the GUI. Use this for quick A/B comparison against the classical
operators in NHC.py.

Examples:
    python infer.py --ckpt checkpoints/latest.pth \\
                    --jxr "C:/Users/me/Videos/NVIDIA/foo.jxr" \\
                    --out out.jpg

    python infer.py --ckpt checkpoints/latest.pth \\
                    --jxr-dir "C:/Users/me/Videos/NVIDIA" --out-dir out/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import SCRGB_WHITE_NITS, load_jxr_nits
from model import NeuralToneMapper


def load_jxr(path: Path, input_unit) -> torch.Tensor:
    """Returns a (1, 3, H, W) HDR linear tensor in the unit the checkpoint expects.

    Checkpoints written before the unit was recorded were fed raw scRGB by
    this script; that is kept for them."""
    arr = load_jxr_nits(path)
    if input_unit is None:
        arr = arr / SCRGB_WHITE_NITS
    elif input_unit != NeuralToneMapper.INPUT_UNIT:
        raise ValueError(f"checkpoint expects unknown input unit '{input_unit}'")
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def tonemap_file(model: NeuralToneMapper, device: torch.device, jxr_path: Path, out_path: Path,
                 input_unit=NeuralToneMapper.INPUT_UNIT):
    hdr = load_jxr(jxr_path, input_unit).to(device)
    with torch.inference_mode():
        sdr = model(hdr).clamp(0, 1)
    arr = np.rint(sdr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=95, optimize=True)
    logging.info(f"{jxr_path.name} -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--jxr", type=str, default=None, help="Single JXR file")
    p.add_argument("--jxr-dir", type=str, default=None, help="Folder of JXRs")
    p.add_argument("--out", type=str, default=None, help="Single output path")
    p.add_argument("--out-dir", type=str, default=None, help="Output folder")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    input_unit = ckpt.get("input_unit")
    saved_args = ckpt.get("args", {})
    model = NeuralToneMapper(
        grid_d=saved_args.get("grid_d", 8),
        grid_h=saved_args.get("grid_hw", 16),
        grid_w=saved_args.get("grid_hw", 16),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])
    logging.info(f"Loaded checkpoint {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    if args.jxr and args.out:
        tonemap_file(model, device, Path(args.jxr), Path(args.out), input_unit)
    elif args.jxr_dir and args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for j in sorted(Path(args.jxr_dir).glob("*.jxr")):
            tonemap_file(model, device, j, out_dir / f"{j.stem}.jpg", input_unit)
    else:
        raise SystemExit("Pass either (--jxr and --out) or (--jxr-dir and --out-dir)")


if __name__ == "__main__":
    main()

"""
Compare checkpoints with each other and with a reference operator, on JXR captures.

    python compare.py --jxr-dir ../samples --teacher agx_punchy --out sheet.jpg ^
                      --ckpt checkpoints/latest.pth --ckpt checkpoints_teacher/best.pth

Prints, for every capture and every checkpoint: the distance to the reference render
(PSNR, higher is closer), the mean luma, and the black level (the luma below which the
darkest 1 % of pixels lie; a healthy render of a scene with shadows is close to 0).
Writes a contact sheet: reference in the first column, one column per checkpoint.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for entry in (str(HERE), str(HERE.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from dataset import SCRGB_WHITE_NITS, load_jxr_nits  # noqa: E402
from model import NeuralToneMapper  # noqa: E402
from teacher import TEACHERS, Teacher  # noqa: E402


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    args = ckpt.get("args", {}) or {}
    model = NeuralToneMapper(grid_d=args.get("grid_d", 8), grid_h=args.get("grid_hw", 16),
                             grid_w=args.get("grid_hw", 16)).to(device).eval()
    model.load_state_dict(ckpt["model"])
    # Checkpoints from before the unit was recorded were fed scRGB x 0.1 by the application.
    scale = 1.0 if ckpt.get("input_unit") == "nits" else 0.1 / SCRGB_WHITE_NITS
    return model, scale


def luma(image: torch.Tensor) -> torch.Tensor:
    return 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]


def describe(image: torch.Tensor, reference: torch.Tensor) -> dict:
    mse = float(((image - reference) ** 2).mean())
    y = luma(image)
    return {"psnr": 10.0 * math.log10(1.0 / max(mse, 1e-12)), "mean": float(y.mean()),
            "black": float(y.flatten()[::17].quantile(0.01))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", action="append", required=True)
    parser.add_argument("--jxr-dir", required=True)
    parser.add_argument("--teacher", default="punchy", choices=TEACHERS)
    parser.add_argument("--out", default=None, help="contact sheet to write (.jpg)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = Teacher(args.teacher, device)
    models = [load_model(p, device) for p in args.ckpt]
    captures = sorted(Path(args.jxr_dir).glob("*.jxr"))
    if not captures:
        print(f"no .jxr capture in {args.jxr_dir}")
        return 2

    rows, totals = [], [dict(psnr=0.0, mean=0.0, black=0.0) for _ in models]
    print(f"{'capture':24s} {'reference':>16s}   " + "   ".join(f"{Path(p).parent.name[:22]:>22s}" for p in args.ckpt))
    print(f"{'':24s} {'mean  black':>16s}   " + "   ".join(f"{'PSNR   mean  black':>22s}" for _ in models))
    for capture in captures:
        nits = torch.from_numpy(load_jxr_nits(capture)).permute(2, 0, 1)[None].to(device)
        reference = teacher(nits)
        row, line = [reference], f"{capture.stem[-24:]:24s} {float(luma(reference).mean()):9.3f} "
        line += f"{float(luma(reference).flatten()[::17].quantile(0.01)):6.3f}   "
        for (model, scale), total in zip(models, totals):
            with torch.inference_mode():
                out = model(nits * scale)
            stats = describe(out, reference)
            for key in total:
                total[key] += stats[key] / len(captures)
            line += f"{stats['psnr']:8.1f} {stats['mean']:6.3f} {stats['black']:6.3f}   "
            row.append(out)
        print(line)
        rows.append(row)
    print(f"{'average':24s} {'':16s}   " +
          "   ".join(f"{t['psnr']:8.1f} {t['mean']:6.3f} {t['black']:6.3f}" for t in totals))

    if args.out:
        from PIL import Image

        tile_w, tile_h = 640, 360
        sheet = Image.new("RGB", (tile_w * (len(models) + 1), tile_h * len(rows)))
        for r, row in enumerate(rows):
            for c, image in enumerate(row):
                array = (image[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
                sheet.paste(Image.fromarray(array).resize((tile_w, tile_h), Image.LANCZOS), (c * tile_w, r * tile_h))
        sheet.save(args.out, quality=88)
        print(f"contact sheet: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

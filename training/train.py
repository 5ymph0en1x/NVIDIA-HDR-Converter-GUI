"""
Training entry point for NeuralToneMapper.

Usage examples:
    # Synthetic-only sanity training on a folder of JPEGs
    python train.py --synthetic-dir D:/photos --epochs 20

    # Real HDRTV4K + synthetic mixed
    python train.py --hdrtv4k-dir D:/HDRTV4K --synthetic-dir D:/photos --amp

    # Fine-tune on user JXR pairs: start from trained weights, fresh schedule
    python train.py --user-pairs-dir D:/my_pairs --init-from checkpoints/best.pth \\
                    --out-dir checkpoints_user --epochs 10 --lr 5e-5

    # Real HDR frames from caches (prepare_cache.py), target rendered on the GPU by a teacher
    python train.py --cache-dir datasets/cache/games --cache-dir datasets/cache/polyhaven \\
                    --teacher agx_punchy --batch-size 64 --epochs 60 --amp

    # Continue an interrupted run exactly where it stopped
    python train.py --synthetic-dir D:/photos --epochs 20 --resume checkpoints/latest.pth

Validation:
    A fraction of the data (`--val-fraction`) is held out and never trained
    on. Its loss is logged every epoch and the best epoch is kept as
    `best.pth`. Every `val_every` epochs the predictions on held-out samples
    are also saved as PNGs next to the checkpoints for visual inspection.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from model import NeuralToneMapper
from losses import CompositeLoss
from cache import FrameCache, GpuLoader
from teacher import TEACHERS, Teacher, augment
from hdr_blaster.engine.colour import nonnegative_rgb
from dataset import (
    SyntheticHDRDataset,
    HDRTV4KDataset,
    UserPairsDataset,
)


def build_cache_datasets(args):
    """(train, validation) from frame caches. Whole origins are held out, decided when
    the cache was built, so the split is the same in every run."""
    train, val = [], []
    for folder in args.cache_dir:
        train.append(FrameCache(folder, "train", exclude=args.exclude_origin))
        val.append(FrameCache(folder, "val", exclude=args.exclude_origin))
        logging.info(f"+ Cache {folder}: {len(train[-1])} train + {len(val[-1])} held out "
                     f"({len(val[-1].origins)} origins)")
    if not any(len(v) for v in val) or not any(len(t) for t in train):
        raise SystemExit("The caches hold no frame for training or none for validation.")
    return train, val


def build_dataset(args) -> torch.utils.data.Dataset:
    parts: list[torch.utils.data.Dataset] = []
    if args.synthetic_dir:
        parts.append(SyntheticHDRDataset(args.synthetic_dir, args.patch_size))
        logging.info(f"+ Synthetic: {len(parts[-1])} images")
    if args.hdrtv4k_dir:
        parts.append(HDRTV4KDataset(args.hdrtv4k_dir, args.patch_size))
        logging.info(f"+ HDRTV4K:   {len(parts[-1])} pairs")
    if args.user_pairs_dir:
        parts.append(UserPairsDataset(args.user_pairs_dir, args.patch_size))
        logging.info(f"+ UserPairs: {len(parts[-1])} pairs")
    if not parts:
        raise SystemExit(
            "No dataset specified. Pass at least one of "
            "--synthetic-dir, --hdrtv4k-dir, --user-pairs-dir."
        )
    return ConcatDataset(parts) if len(parts) > 1 else parts[0]


def targets_for(batch, device, teacher, ev: float):
    """(hdr, sdr) on the device. With a teacher the frame is augmented, then the target
    is rendered from that very frame; otherwise the dataset's own pair is used."""
    hdr = batch["hdr"].to(device, non_blocking=True)
    if teacher is None:
        return nonnegative_rgb(hdr), batch["sdr"].to(device, non_blocking=True)
    if ev > 0.0:
        hdr = augment(hdr, ev=ev)
    return nonnegative_rgb(hdr), teacher(hdr)


def evaluate(model, val_loader, loss_fn, device, teacher=None) -> float:
    """Mean total loss on the held-out samples, with a fixed random crop and EV
    per sample so that epochs are comparable."""
    state = random.getstate()
    random.seed(0)
    model.eval()
    total, n = 0.0, 0
    try:
        with torch.no_grad():
            for batch in val_loader:
                hdr, sdr = targets_for(batch, device, teacher, ev=0.0)   # frames as stored
                loss, _ = loss_fn(model(hdr), sdr, hdr)
                total += loss.item() * hdr.size(0)
                n += hdr.size(0)
    finally:
        random.setstate(state)
        model.train()
    return total / max(n, 1)


def save_validation(model, val_loader, device, out_path: Path, max_items: int = 4, teacher=None):
    """Run model on a few validation samples and save side-by-side PNGs."""
    import numpy as np
    from PIL import Image

    out_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if saved >= max_items:
                break
            hdr, sdr_target = targets_for(batch, device, teacher, ev=0.0)
            sdr_target = sdr_target.cpu()
            pred = model(hdr).cpu().clamp(0, 1)
            # Build side-by-side: [log(HDR) | pred | target]
            hdr_vis = model.log_compress(hdr).cpu().clamp(0, 1)
            row = torch.cat([hdr_vis, pred, sdr_target], dim=-1)  # (B, 3, H, 3W)
            for b in range(row.size(0)):
                if saved >= max_items:
                    break
                arr = (row[b].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(out_path / f"val_{saved:02d}.png")
                saved += 1
    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-dir", type=str, default=None)
    p.add_argument("--hdrtv4k-dir", type=str, default=None)
    p.add_argument("--user-pairs-dir", type=str, default=None)
    p.add_argument("--cache-dir", action="append", default=[],
                   help="Frame cache built by prepare_cache.py (repeatable). Needs --teacher")
    p.add_argument("--exclude-origin", action="append", default=[],
                   help="Leave out cached frames whose source file name contains this text (repeatable)")
    p.add_argument("--teacher", type=str, default=None,
                   choices=TEACHERS + ("agx", "agx_punchy", "agx_golden"),
                   help="Render the SDR target on the GPU with this operator of the application")
    p.add_argument("--aug-ev", type=float, default=2.0,
                   help="With a teacher: random exposure of each frame, +/- this many stops")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", help="Mixed precision (FP16/BF16)")
    p.add_argument("--resume", type=str, default=None,
                   help="Continue an interrupted run: weights, optimizer, schedule and epoch count")
    p.add_argument("--init-from", type=str, default=None,
                   help="Fine-tune: start from these weights with a fresh optimizer and schedule")
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Share of the data held out for validation")
    p.add_argument("--grid-d", type=int, default=8)
    p.add_argument("--grid-hw", type=int, default=16)
    p.add_argument(
        "--loss-weights", type=str, default=None,
        help='JSON dict of override weights, e.g. \'{"struct": 3.0}\'',
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                     f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.resume and args.init_from:
        raise SystemExit("Pass either --resume (continue a run) or --init-from (fine-tune), not both.")

    legacy = args.synthetic_dir or args.hdrtv4k_dir or args.user_pairs_dir
    if args.cache_dir and legacy:
        raise SystemExit("Frame caches cannot be mixed with the other sources in one run "
                         "(different frame sizes). Build a cache from them with prepare_cache.py.")
    if args.cache_dir and not args.teacher:
        raise SystemExit("Frame caches hold HDR frames only: choose the target with --teacher.")
    teacher = Teacher(args.teacher, device) if args.teacher else None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    gpu_loaders = None
    if args.cache_dir:
        train_caches, val_caches = build_cache_datasets(args)
        train_set, val_set = ConcatDataset(train_caches), ConcatDataset(val_caches)
        n_val = len(val_set)
        needed = GpuLoader.bytes_needed(train_caches + val_caches)
        if device.type == "cuda" and needed < 0.5 * torch.cuda.mem_get_info()[0]:
            # Small enough to live on the GPU: no loader workers, no copies while training.
            gpu_loaders = (
                GpuLoader(train_caches, device, args.batch_size, shuffle=True, drop_last=True),
                GpuLoader(val_caches, device, min(args.batch_size, 64), shuffle=False))
            logging.info(f"Frames held in GPU memory: {needed / 1e9:.1f} GB")
    else:
        dataset = build_dataset(args)
        if len(dataset) < 2:
            raise SystemExit("Need at least 2 samples: one to train on, one held out for validation.")
        # Held-out validation split: the same samples every run, never trained on.
        n_val = min(len(dataset) - 1, max(1, round(len(dataset) * args.val_fraction)))
        train_set, val_set = torch.utils.data.random_split(
            dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=len(train_set) >= args.batch_size,
        persistent_workers=args.num_workers > 0,
    )
    logging.info(f"Samples: {len(train_set)} train + {n_val} held out  Iters/epoch: {len(loader)}")
    val_loader = DataLoader(val_set, batch_size=min(args.batch_size, 16), shuffle=False, num_workers=0)
    if gpu_loaders:
        loader, val_loader = gpu_loaders

    model = NeuralToneMapper(
        grid_d=args.grid_d, grid_h=args.grid_hw, grid_w=args.grid_hw
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: NeuralToneMapper  params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * max(len(loader), 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    weights = None
    if args.loss_weights:
        import json
        weights = json.loads(args.loss_weights)
    loss_fn = CompositeLoss(weights=weights).to(device)

    start_epoch = 0
    best_val = float("inf")
    source = args.resume or args.init_from
    if source:
        ckpt = torch.load(source, map_location=device, weights_only=True)
        if ckpt.get("input_unit") != NeuralToneMapper.INPUT_UNIT:
            logging.warning(
                f"{source} was trained on inputs in unit {ckpt.get('input_unit')!r}; this run "
                f"feeds {NeuralToneMapper.INPUT_UNIT!r}. Its weights are only a starting point.")
        model.load_state_dict(ckpt["model"])
    if args.resume:
        start_epoch = ckpt["epoch"] + 1
        if start_epoch >= args.epochs:
            raise SystemExit(
                f"{args.resume} already completed {start_epoch} epoch(s) and --epochs is "
                f"{args.epochs}: nothing left to train. To fine-tune from these weights with a "
                f"fresh schedule, use --init-from instead of --resume.")
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val = ckpt.get("best_val", best_val)
        logging.info(f"Resumed from {args.resume} @ epoch {start_epoch}")
    elif args.init_from:
        logging.info(f"Fine-tuning from the weights of {args.init_from}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = {"pixel": 0.0, "struct": 0.0, "chroma": 0.0, "range": 0.0, "total": 0.0}
        n_iters = 0

        for i, batch in enumerate(loader):
            hdr, sdr = targets_for(batch, device, teacher, ev=args.aug_ev)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred = model(hdr)
                total, parts = loss_fn(pred, sdr, hdr)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k, v in parts.items():
                running[k] += v
            n_iters += 1

            if i % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                logging.info(
                    f"E{epoch:03d}/{args.epochs} I{i:04d} lr={lr:.2e} "
                    + " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                )

        dt = time.time() - t0
        avg = {k: v / max(n_iters, 1) for k, v in running.items()}
        logging.info(
            f"== Epoch {epoch} done in {dt:.0f}s | "
            + " ".join(f"avg_{k}={v:.4f}" for k, v in avg.items())
        )

        val_loss = evaluate(model, val_loader, loss_fn, device, teacher)
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        logging.info(f"== Epoch {epoch} held-out loss={val_loss:.4f}" + ("  (best so far)" if is_best else ""))

        if (epoch + 1) % args.val_every == 0:
            save_validation(model, val_loader, device, out_dir / f"val_e{epoch:03d}", teacher=teacher)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "input_unit": NeuralToneMapper.INPUT_UNIT,
            "teacher": teacher.look if teacher else None,
            # The white the teacher was anchored on: the application divides by its own
            # screen's white and multiplies by this one before feeding the model.
            "anchor_white_nits": teacher.white_nits if teacher else None,
            "val_loss": val_loss,
            "best_val": best_val,
        }
        if is_best:
            torch.save(ckpt, out_dir / "best.pth")
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, out_dir / "latest.pth")
            torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pth")
            logging.info(f"Saved checkpoint -> latest.pth + epoch_{epoch:03d}.pth")

    logging.info("Training done.")


if __name__ == "__main__":
    main()

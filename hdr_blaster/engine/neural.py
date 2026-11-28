"""Loader for the learned tone mapper (moved verbatim from NHC.py)."""
import logging
import os
import sys

from typing import Optional

import torch

from ..paths import app_data_dir as _app_data_dir
from .constants import ENGINE_UNIT_NITS


# ============================================================
# Neural tone mapper integration
# ------------------------------------------------------------
# HDRNet-style learned HDR -> SDR mapper trained from scratch on
# project data (see ./training/). Loaded lazily and cached per
# device; falls back gracefully to classical operators if the
# training package or checkpoint is missing.
# ============================================================

_NEURAL_TONEMAP_AVAILABLE = False


_NeuralToneMapperImpl = None


try:
    _TRAINING_DIR = os.path.join(_app_data_dir(), "training")
    if os.path.isdir(_TRAINING_DIR) and _TRAINING_DIR not in sys.path:
        sys.path.insert(0, _TRAINING_DIR)
    from model import NeuralToneMapper as _NeuralToneMapperImpl  # noqa: E402
    _NEURAL_TONEMAP_AVAILABLE = True
except ImportError as _exc:
    logging.info(f"Neural tone mapper module unavailable: {_exc}")


NEURAL_TONEMAP_CHECKPOINT_PATH = os.path.join(
    _app_data_dir(), "training", "checkpoints", "latest.pth"
)


_NEURAL_MODEL_CACHE: dict = {}


def engine_input_scale(ckpt: dict) -> float:
    """Factor from the engine's working space to the unit a checkpoint was trained on.

    Checkpoints written before the unit was recorded are fed the engine's
    values unchanged, as the application always did: their rendering does not move.
    """
    unit = ckpt.get("input_unit")
    if unit is None:
        return 1.0
    if unit == "nits":
        return ENGINE_UNIT_NITS
    raise ValueError(f"unknown input unit '{unit}' in the neural tone mapper checkpoint")


def anchor_white_nits(ckpt: dict) -> Optional[float]:
    """The white its teacher was anchored on, for a checkpoint taught by a look that keeps
    the scene's exposure; ``None`` for the earlier ones, which expose a picture by its
    content and are fed absolute luminance."""
    value = ckpt.get("anchor_white_nits")
    if value is None:
        return None
    value = float(value)
    if not 10.0 <= value <= 10000.0:
        raise ValueError(f"implausible anchor white ({value} nits) in the neural tone mapper checkpoint")
    return value


def _get_neural_tonemap(device):
    """Cached per-device loader. Returns a ready-to-eval model or None."""
    if not _NEURAL_TONEMAP_AVAILABLE:
        return None
    key = str(device)
    if key in _NEURAL_MODEL_CACHE:
        return _NEURAL_MODEL_CACHE[key]
    if not os.path.exists(NEURAL_TONEMAP_CHECKPOINT_PATH):
        logging.info(
            f"Neural tone mapper checkpoint not found at "
            f"{NEURAL_TONEMAP_CHECKPOINT_PATH}; classical operators only."
        )
        _NEURAL_MODEL_CACHE[key] = None
        return None
    try:
        ckpt = torch.load(
            NEURAL_TONEMAP_CHECKPOINT_PATH,
            map_location=device, weights_only=True,
        )
        saved_args = ckpt.get("args", {}) or {}
        model = _NeuralToneMapperImpl(
            grid_d=saved_args.get("grid_d", 8),
            grid_h=saved_args.get("grid_hw", 16),
            grid_w=saved_args.get("grid_hw", 16),
        ).to(device).eval()
        model.load_state_dict(ckpt["model"])
        model.engine_input_scale = engine_input_scale(ckpt)
        model.anchor_white_nits = anchor_white_nits(ckpt)
        model.teacher_look = ckpt.get("teacher")
        logging.info(
            f"Loaded neural tone mapper from "
            f"{NEURAL_TONEMAP_CHECKPOINT_PATH} "
            f"(epoch {ckpt.get('epoch', '?')}) on {device}"
        )
        _NEURAL_MODEL_CACHE[key] = model
        return model
    except Exception as exc:
        logging.error(f"Failed to load neural tone mapper: {exc}")
        _NEURAL_MODEL_CACHE[key] = None
        return None

"""Image-adaptive 3D LUT colour enhancement (moved verbatim from NHC.py)."""
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..paths import app_data_dir as _app_data_dir
from .params import measure as _measure


# ============================================================
# Image-Adaptive 3D LUT — neural color enhancement
# ------------------------------------------------------------
# Zeng et al., "Learning Image-adaptive 3D Lookup Tables for
# High Performance Photo Enhancement in Real-time", TPAMI 2020.
# https://github.com/HuiZeng/Image-Adaptive-3DLUT (Apache 2.0)
#
# Pretrained on MIT-Adobe FiveK (sRGB, paired).
# Weights total ~2.4 MB, auto-downloaded on first launch.
# ============================================================

LUT3D_WEIGHTS_BASE_URL = (
    "https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/"
    "master/pretrained_models/sRGB"
)


LUT3D_FILES = {"LUTs.pth": 1294452, "classifier.pth": 1083261}


LUT3D_LOCAL_DIR = os.path.join(_app_data_dir(), "models", "3dlut", "sRGB")


def ensure_lut3d_weights():
    """Download the 3D LUT pretrained weights on first launch."""
    import urllib.request
    os.makedirs(LUT3D_LOCAL_DIR, exist_ok=True)
    for fname, expected_size in LUT3D_FILES.items():
        dst = os.path.join(LUT3D_LOCAL_DIR, fname)
        if os.path.exists(dst) and os.path.getsize(dst) == expected_size:
            continue
        url = f"{LUT3D_WEIGHTS_BASE_URL}/{fname}"
        logging.info(f"Downloading {fname} from {url}")
        urllib.request.urlretrieve(url, dst)


def _disc_block(in_filters, out_filters, normalization=False):
    """Downsampling block matching HuiZeng's discriminator_block convention."""
    layers = [
        nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class Classifier3DLUT(nn.Module):
    """
    Lightweight CNN that predicts 3 blending weights for the basis LUTs.
    Layer ordering mirrors HuiZeng's paired Classifier so classifier.pth
    loads with no key remapping.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *_disc_block(16, 32, normalization=True),
            *_disc_block(32, 64, normalization=True),
            *_disc_block(64, 128, normalization=True),
            *_disc_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, x):
        return self.model(x)


class ImageAdaptive3DLUT(nn.Module):
    """
    Image-adaptive 3D LUT color enhancement.

    Input  : (B, 3, H, W) RGB tensor in [0, 1]
    Output : (B, 3, H, W) enhanced RGB tensor in [0, 1]

    Sets `self.enabled = False` if weights cannot be loaded; callers
    should check this flag and skip the AI step gracefully.
    """

    LUT_DIM = 33

    def __init__(self, weights_dir=None, auto_download=True):
        super().__init__()
        if weights_dir is None:
            weights_dir = LUT3D_LOCAL_DIR
        d = self.LUT_DIM
        # Basis LUTs — inner axes are (R_index, G_index, B_index).
        self.LUT0 = nn.Parameter(torch.zeros(3, d, d, d))
        self.LUT1 = nn.Parameter(torch.zeros(3, d, d, d))
        self.LUT2 = nn.Parameter(torch.zeros(3, d, d, d))
        self.classifier = Classifier3DLUT()
        self.enabled = False
        try:
            if auto_download:
                ensure_lut3d_weights()
            self._load_pretrained(weights_dir)
            self.enabled = True
        except Exception as e:
            logging.error(
                f"Failed to load 3D LUT weights from {weights_dir}: {e}. "
                "AI color enhancement disabled."
            )

    def _load_pretrained(self, weights_dir):
        luts_path = os.path.join(weights_dir, "LUTs.pth")
        clf_path = os.path.join(weights_dir, "classifier.pth")
        luts_state = torch.load(luts_path, map_location='cpu', weights_only=True)
        clf_state = torch.load(clf_path, map_location='cpu', weights_only=True)
        with torch.no_grad():
            self.LUT0.copy_(luts_state["0"]["LUT"])
            self.LUT1.copy_(luts_state["1"]["LUT"])
            self.LUT2.copy_(luts_state["2"]["LUT"])
        self.classifier.load_state_dict(clf_state)
        logging.info(f"Loaded 3D LUT weights from {weights_dir}")

    def forward(self, x, params=None):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        weights = _measure(params, "lut3d.weights",
                           lambda: self.classifier(x_clamped).flatten(1))  # (B, 3)
        luts = torch.stack([self.LUT0, self.LUT1, self.LUT2], dim=0)
        combined = torch.einsum('bn,ncdhw->bcdhw', weights, luts)
        return self._trilinear_lookup(combined, x_clamped)

    @staticmethod
    def _trilinear_lookup(lut, image):
        """
        Trilinear 3D LUT lookup via F.grid_sample (no CUDA extension).
        lut   : (B, 3, dim, dim, dim) with inner axes (R_idx, G_idx, B_idx)
        image : (B, 3, H, W) in [0, 1]
        """
        # PyTorch grid_sample 3D expects last-dim coords as (x=W, y=H, z=D).
        # Our LUT D-axis = R, H-axis = G, W-axis = B, so:
        #   grid[..., 0] = B value, [..., 1] = G value, [..., 2] = R value
        r, g, b = image[:, 0], image[:, 1], image[:, 2]
        grid = torch.stack([b, g, r], dim=-1).mul(2.0).sub(1.0)
        grid = grid.unsqueeze(1)  # (B, 1, H, W, 3)
        out = F.grid_sample(
            lut, grid, mode='bilinear',
            padding_mode='border', align_corners=True,
        )
        return out.squeeze(2)

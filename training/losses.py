"""
Multi-component loss for HDR-faithful tone mapping.

Components:
    L_pixel   — L1 between predicted SDR and target SDR (baseline)
    L_struct  — per-pixel gradient alignment on log-luminance,
                weighted by HDR gradient magnitude (the fidelity term)
    L_chroma  — chromatic angle preservation between scale-normalized
                HDR and SDR (preserves hue)
    L_range   — penalizes clipping outside [0,1] and rewards full
                dynamic-range usage

Default weights bias the network toward structural fidelity:
    pixel=1.0  struct=2.0  chroma=0.5  range=0.3
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def luminance(x: torch.Tensor) -> torch.Tensor:
    """BT.709 relative luminance. Input (B, 3, H, W) -> (B, 1, H, W)."""
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def log_lum(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(luminance(x) + eps)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB EOTF. Input is clamped to [0, 1]."""
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class StructuralLoss(nn.Module):
    """
    Per-pixel gradient alignment between SDR_pred and HDR_input in
    log-luminance space, weighted by HDR gradient magnitude.

    Interprets each pixel's gradient as a 2D vector (∂L/∂x, ∂L/∂y).
    Cosine = 1 → perfect alignment, 0 → orthogonal, -1 → reversed.

    Loss = mean( (1 - cos) * |∇L_HDR| / mean(|∇L_HDR|) )

    The weighting focuses attention on regions where HDR actually has
    strong local contrast — flat regions get downweighted, so the
    network is judged on its handling of edges and texture.
    """

    def forward(self, sdr_pred: torch.Tensor, hdr_input: torch.Tensor) -> torch.Tensor:
        # The prediction is not clamped while training: keep the log defined.
        L_sdr = log_lum(sdr_pred.clamp(min=0.0))
        L_hdr = log_lum(hdr_input)

        dx_sdr = L_sdr[..., :, 1:] - L_sdr[..., :, :-1]
        dy_sdr = L_sdr[..., 1:, :] - L_sdr[..., :-1, :]
        dx_hdr = L_hdr[..., :, 1:] - L_hdr[..., :, :-1]
        dy_hdr = L_hdr[..., 1:, :] - L_hdr[..., :-1, :]

        # Crop to common spatial extent (one fewer row + one fewer col).
        dx_sdr = dx_sdr[..., :-1, :]
        dy_sdr = dy_sdr[..., :, :-1]
        dx_hdr = dx_hdr[..., :-1, :]
        dy_hdr = dy_hdr[..., :, :-1]

        grad_sdr = torch.stack([dx_sdr, dy_sdr], dim=-1)  # (B, 1, H-1, W-1, 2)
        grad_hdr = torch.stack([dx_hdr, dy_hdr], dim=-1)

        dot = (grad_sdr * grad_hdr).sum(dim=-1)
        nrm_sdr = grad_sdr.norm(dim=-1)
        nrm_hdr = grad_hdr.norm(dim=-1)
        cos = dot / (nrm_sdr * nrm_hdr + 1e-6)

        weight = nrm_hdr / (nrm_hdr.mean() + 1e-6)
        return ((1.0 - cos) * weight).mean()


class ChromaLoss(nn.Module):
    """
    Preserves chromatic angle: cosine similarity of per-pixel RGB
    vectors after scale-normalization. Penalizes hue shifts and
    saturation collapse, but is invariant to absolute brightness.

    Both sides are compared in LINEAR light. The prediction is sRGB-encoded,
    and gamma encoding pulls every colour toward grey: comparing it directly
    with the linear scene asked the network for over-saturated output. The
    scene is scaled by ONE factor per image; a factor per channel would be a
    white-balance change, i.e. a hue shift in the reference itself.
    """

    def forward(self, sdr_pred: torch.Tensor, hdr_input: torch.Tensor) -> torch.Tensor:
        hdr_norm = hdr_input / (hdr_input.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
        sdr_n = F.normalize(srgb_to_linear(sdr_pred) + 1e-6, dim=1)
        hdr_n = F.normalize(hdr_norm + 1e-6, dim=1)
        cos = (sdr_n * hdr_n).sum(dim=1)
        return (1.0 - cos).mean()


class RangeLoss(nn.Module):
    """
    Encourages full use of [0, 1] without clipping.
        - Penalizes any output value > 1 (over-bright)
        - Penalizes any output value < 0 (under-bright)
        - Rewards spread (max - min per image)
    """

    def forward(self, sdr_pred: torch.Tensor) -> torch.Tensor:
        over = (sdr_pred - 1.0).clamp(min=0.0).pow(2).mean()
        under = sdr_pred.clamp(max=0.0).pow(2).mean()
        # Spread of what will actually be displayed: the reward stops at [0, 1].
        shown = sdr_pred.clamp(0.0, 1.0)
        spread = (shown.amax(dim=(2, 3)) - shown.amin(dim=(2, 3))).mean()
        # Negative spread term so optimization pushes toward larger spread.
        return over + under - 0.1 * spread


class CompositeLoss(nn.Module):
    """
    Combined loss for HDR-faithful tone mapping.

    Returns (total_scalar, components_dict_of_floats).
    """

    DEFAULT_WEIGHTS = {"pixel": 1.0, "struct": 2.0, "chroma": 0.5, "range": 0.3}

    def __init__(self, weights: dict | None = None):
        super().__init__()
        self.w = dict(self.DEFAULT_WEIGHTS, **(weights or {}))
        self._struct = StructuralLoss()
        self._chroma = ChromaLoss()
        self._range = RangeLoss()

    def forward(
        self,
        sdr_pred: torch.Tensor,
        sdr_target: torch.Tensor,
        hdr_input: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Always in float32, even under mixed precision: the logs, the tiny
        # epsilons and the divisions below overflow or vanish in half precision.
        with torch.autocast(device_type=sdr_pred.device.type, enabled=False):
            sdr_pred = sdr_pred.float()
            sdr_target = sdr_target.float()
            hdr_input = hdr_input.float()

            l_pixel = F.l1_loss(sdr_pred, sdr_target)
            l_struct = self._struct(sdr_pred, hdr_input)
            l_chroma = self._chroma(sdr_pred, hdr_input)
            l_range = self._range(sdr_pred)

            total = (self.w["pixel"] * l_pixel
                     + self.w["struct"] * l_struct
                     + self.w["chroma"] * l_chroma
                     + self.w["range"] * l_range)

        return total, {
            "pixel": l_pixel.item(),
            "struct": l_struct.item(),
            "chroma": l_chroma.item(),
            "range": l_range.item(),
            "total": total.item(),
        }

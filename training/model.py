"""
NeuralToneMapper — HDRNet-style bilateral guided affine tone mapper.

Reference: Gharbi et al., "Deep Bilateral Learning for Real-Time Image
Enhancement", SIGGRAPH 2017 — adapted for HDR linear input → SDR output
with the fidelity-first design we want for game HDR screenshots.

Pipeline:
    HDR linear (B, 3, H, W), unbounded
        |
        v  log_compress  -> (B, 3, H, W) in [0, 1]
        |
        +--------------+---------------+
        |              |               |
    downsample        full-res     full-res
    to 256x256       guide path    apply path
        |              |               |
    Coeff predictor  Guide head       (input to affine)
    -> bilateral     -> per-pixel
       grid of        guide(x,y)
       affine
       coeffs
       (B, 12, gd, gh, gw)
                       |
                       v
                  Bilateral slice (trilinear in grid)
                       |
                       v
                  Per-pixel affine 3x4 matrices (B, 12, H, W)
                       |
                       v
                  Apply: out = W @ in + b   (per pixel)
                       |
                       v  clamp
    SDR (B, 3, H, W) in [0, 1]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int, ksize: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, ksize, stride=stride, padding=ksize // 2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CoefficientPredictor(nn.Module):
    """
    Predicts a bilateral grid of affine coefficients from a low-res input.

    Output: (B, 12, grid_d, grid_h, grid_w)
        12 = 3 output channels x 4 (3 input weights + 1 bias)
        grid_d = guide axis bins
        grid_h, grid_w = spatial bins
    """

    def __init__(self, grid_d: int = 8, grid_h: int = 16, grid_w: int = 16, base_ch: int = 8):
        super().__init__()
        self.grid_d = grid_d
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Splat: 256x256 -> 16x16 via four stride-2 blocks
        self.splat = nn.Sequential(
            _conv_bn_relu(3, base_ch, 3, 2),            # 128
            _conv_bn_relu(base_ch, base_ch * 2, 3, 2),  # 64
            _conv_bn_relu(base_ch * 2, base_ch * 4, 3, 2),  # 32
            _conv_bn_relu(base_ch * 4, base_ch * 8, 3, 2),  # 16
        )

        # Global features: 16x16 -> 1x1 -> 64-dim
        self.global_head = nn.Sequential(
            _conv_bn_relu(base_ch * 8, base_ch * 8, 3, 2),  # 8
            _conv_bn_relu(base_ch * 8, base_ch * 8, 3, 2),  # 4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, base_ch * 8),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch * 8, base_ch * 8),
            nn.ReLU(inplace=True),
        )

        # Local features kept at 16x16
        self.local_head = _conv_bn_relu(base_ch * 8, base_ch * 8, 3, 1)

        # Fusion -> 12 * grid_d channels (will be reshaped to grid_d depth axis)
        self.fusion = nn.Conv2d(base_ch * 8, 12 * grid_d, 1)

        # Initialize fusion to predict near-identity affine + zero bias
        self._init_identity()

    def _init_identity(self):
        # Bias init: identity matrix [I | 0] expanded across the grid.
        # For each grid depth bin, output channels are arranged as
        # [w00 w01 w02 b0 w10 w11 w12 b1 w20 w21 w22 b2].
        with torch.no_grad():
            self.fusion.weight.zero_()
            bias = torch.zeros(12 * self.grid_d)
            for d in range(self.grid_d):
                base = d * 12
                bias[base + 0] = 1.0  # w00
                bias[base + 5] = 1.0  # w11
                bias[base + 10] = 1.0  # w22
            self.fusion.bias.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.splat(x)              # (B, C, 16, 16)
        glob = self.global_head(feat)     # (B, C)
        local = self.local_head(feat)     # (B, C, 16, 16)
        fused = F.relu(local + glob[:, :, None, None])
        coeffs = self.fusion(fused)       # (B, 12 * grid_d, 16, 16)
        b = coeffs.size(0)
        # Reshape to (B, 12, grid_d, grid_h, grid_w) for trilinear slicing.
        coeffs = coeffs.view(b, 12, self.grid_d, self.grid_h, self.grid_w)
        return coeffs


class GuidePredictor(nn.Module):
    """
    Per-pixel guide map in [0, 1] from full-res log-input.
    Tiny pointwise CNN so it can run at any resolution.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def bilateral_slice(grid: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    """
    Sample the bilateral grid at every full-resolution pixel.

    grid  : (B, C, gd, gh, gw)  — C coefficients per bin
    guide : (B, 1, H, W) in [0, 1]
    return: (B, C, H, W) sliced coefficients

    Uses F.grid_sample 3D — no custom CUDA op needed.
    """
    b = grid.size(0)
    h, w = guide.shape[-2:]
    device = grid.device
    dtype = grid.dtype

    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    gz = guide.squeeze(1) * 2.0 - 1.0  # (B, H, W)
    gx_b = gx.unsqueeze(0).expand(b, -1, -1)
    gy_b = gy.unsqueeze(0).expand(b, -1, -1)

    # grid_sample 5D expects (x, y, z) order matching (W, H, D) input axes.
    sample_grid = torch.stack([gx_b, gy_b, gz], dim=-1).unsqueeze(1)  # (B, 1, H, W, 3)
    sliced = F.grid_sample(
        grid, sample_grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    return sliced.squeeze(2)  # (B, C, H, W)


class NeuralToneMapper(nn.Module):
    """
    HDR linear -> SDR sRGB.

    The network's affine transforms operate on the LOG-compressed input,
    not the raw HDR. The output is clamped to [0, 1] (gamma-encoded sRGB
    target convention).

    Forward signature:
        in : (B, 3, H, W) — HDR linear, absolute luminance in NITS,
             BT.709 primaries, non-negative
        out: (B, 3, H, W) — SDR sRGB; clamped to [0, 1] in eval mode,
             left un-clamped in training mode (see _apply_to_log)
    """

    # Every data source and every caller converts to this unit. Checkpoints
    # record it, so a caller knows how to scale its own working space.
    INPUT_UNIT = "nits"
    LOG_NORM = math.log(1.0 + 1e4)  # log1p of the 10000-nit PQ peak: 100 nits -> 0.50

    def __init__(self, grid_d: int = 8, grid_h: int = 16, grid_w: int = 16, lowres: int = 256):
        super().__init__()
        self.lowres = lowres
        self.coeffs = CoefficientPredictor(grid_d, grid_h, grid_w)
        self.guide = GuidePredictor()

    @classmethod
    def log_compress(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x.clamp(min=0.0)) / cls.LOG_NORM

    def forward(self, x_hdr: torch.Tensor) -> torch.Tensor:
        x_log = self.log_compress(x_hdr)
        return self._apply_to_log(x_log, self._grid_from_log(x_log))

    def predict_grid(self, x_hdr: torch.Tensor) -> torch.Tensor:
        """The one whole-image decision of the model: its bilateral grid of
        affine coefficients, (B, 12, gd, gh, gw), predicted from a low-res view.

        ``apply_grid(x, predict_grid(x))`` is exactly ``forward(x)``. Keeping the
        two apart lets a caller predict the grid once (or smooth it over time,
        for video) and apply it to other frames.
        """
        return self._grid_from_log(self.log_compress(x_hdr))

    def apply_grid(self, x_hdr: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply a grid from :meth:`predict_grid`. Per-pixel work only."""
        return self._apply_to_log(self.log_compress(x_hdr), coeffs)

    def _grid_from_log(self, x_log: torch.Tensor) -> torch.Tensor:
        # Low-res for coefficient prediction
        x_low = F.interpolate(
            x_log, size=(self.lowres, self.lowres),
            mode="bilinear", align_corners=False,
        )
        return self.coeffs(x_low)                 # (B, 12, gd, gh, gw)

    def _apply_to_log(self, x_log: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        guide = self.guide(x_log)                 # (B, 1, H, W)
        sliced = bilateral_slice(coeffs, guide)   # (B, 12, H, W)

        # Apply per-pixel 3x4 affine to log-compressed RGB.
        b, _, h, w = sliced.shape
        sliced = sliced.view(b, 3, 4, h, w)
        weights = sliced[:, :, :3]                # (B, 3 out, 3 in, H, W)
        bias = sliced[:, :, 3]                    # (B, 3, H, W)
        # out_c = sum_i W[c, i] * x_log[i] + b[c]
        out = (weights * x_log.unsqueeze(1)).sum(dim=2) + bias
        # Inference: the SDR result is clamped to [0, 1]. Training: left as it is.
        # A clamp has zero gradient outside its range, so a pixel that saturates
        # during training could never be pulled back, and the range loss, which
        # exists to penalise exactly that, would always read zero.
        return out if self.training else out.clamp(0.0, 1.0)


def smoke_test():
    """Verify a forward+backward pass works at typical sizes."""
    model = NeuralToneMapper().cuda() if torch.cuda.is_available() else NeuralToneMapper()
    device = next(model.parameters()).device
    x = torch.rand(2, 3, 512, 512, device=device) * 1000.0  # fake HDR, in nits
    with torch.no_grad():
        y = model.eval()(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert y.min() >= 0.0 and y.max() <= 1.0, f"Out of range: [{y.min()}, {y.max()}]"
    loss = (model.train()(x) - 0.5).abs().mean()
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OK  shape={tuple(y.shape)}  range=[{y.min().item():.3f}, {y.max().item():.3f}]")
    print(f"OK  loss={loss.item():.4f}  params={n_params:,}")


if __name__ == "__main__":
    smoke_test()

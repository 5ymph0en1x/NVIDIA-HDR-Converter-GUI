"""Colour-space helpers (moved verbatim from NHC.py)."""
import torch


class PerceptualColorPreserver:
    """Preserves perceptual color relationships during tone mapping."""

    def __init__(self, device):
        self.device = device
        # CIE LAB conversion matrices
        self.xyz_to_rgb = torch.tensor([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ], device=device)
        self.rgb_to_xyz = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], device=device)

    def rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to CIE LAB color space."""
        # Linearize RGB
        rgb = torch.where(rgb > 0.04045,
                          torch.pow((rgb + 0.055) / 1.055, 2.4),
                          rgb / 12.92)

        # Convert to XYZ
        B, C, H, W = rgb.shape
        rgb_flat = rgb.view(B, C, -1)
        xyz = torch.einsum('ij,bjk->bik', self.rgb_to_xyz, rgb_flat)
        xyz = xyz.view(B, 3, H, W)

        # Normalize by D65 white point
        xyz[:, 0] /= 0.95047
        xyz[:, 2] /= 1.08883

        # Convert to LAB
        f = torch.where(xyz > 0.008856,
                        torch.pow(xyz, 1 / 3),
                        7.787 * xyz + 16 / 116)

        L = 116 * f[:, 1:2] - 16
        a = 500 * (f[:, 0:1] - f[:, 1:2])
        b = 200 * (f[:, 1:2] - f[:, 2:3])

        return torch.cat([L, a, b], dim=1)

    def lab_to_rgb(self, lab: torch.Tensor) -> torch.Tensor:
        """Convert CIE LAB to RGB color space."""
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

        # Convert to XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        x_val = torch.where(fx > 0.206897, torch.pow(fx, 3), (fx - 16 / 116) / 7.787)
        y_val = torch.where(fy > 0.206897, torch.pow(fy, 3), (fy - 16 / 116) / 7.787)
        z_val = torch.where(fz > 0.206897, torch.pow(fz, 3), (fz - 16 / 116) / 7.787)

        xyz = torch.cat([x_val, y_val, z_val], dim=1)

        # Denormalize
        xyz[:, 0] *= 0.95047
        xyz[:, 2] *= 1.08883

        # Convert to RGB
        B, C, H, W = xyz.shape
        xyz_flat = xyz.view(B, C, -1)
        rgb = torch.einsum('ij,bjk->bik', self.xyz_to_rgb, xyz_flat)
        rgb = rgb.view(B, 3, H, W)

        # Apply sRGB gamma
        rgb = torch.where(rgb > 0.0031308,
                          1.055 * torch.pow(rgb, 1 / 2.4) - 0.055,
                          12.92 * rgb)

        return torch.clamp(rgb, 0, 1)

    def preserve_color_ratios(self, original: torch.Tensor, tonemapped: torch.Tensor) -> torch.Tensor:
        """Preserve color ratios from original in tonemapped image."""
        # Convert both to LAB
        orig_lab = self.rgb_to_lab(original)
        tone_lab = self.rgb_to_lab(tonemapped)

        # Preserve original color channels, use tonemapped luminance
        preserved_lab = torch.cat([tone_lab[:, 0:1], orig_lab[:, 1:3]], dim=1)

        # Scale color channels based on luminance change
        lum_ratio = torch.clamp(tone_lab[:, 0:1] / (orig_lab[:, 0:1] + 1e-6), 0.2, 5.0)
        preserved_lab[:, 1:3] *= lum_ratio

        # Convert back to RGB
        return self.lab_to_rgb(preserved_lab)


def _srgb_decode(x: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB EOTF. Used to undo the neural model's sRGB output
    so the downstream linear_to_srgb step re-encodes correctly."""
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def nonnegative_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Map signed scRGB into the positive BT.709 cone at constant linear luminance.

    Pull only colours with negative channels towards neutral until the lowest
    channel reaches zero. Highlights can still exceed white at this stage.
    """
    low = rgb.amin(dim=1, keepdim=True)
    y = (0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]).clamp_min(0.0)
    scale = y / (y - low).clamp_min(1e-20)
    return torch.where(low < 0, ((rgb - low) * scale).clamp_min(0.0), rgb)


def signed_power(x: torch.Tensor, exponent: float) -> torch.Tensor:
    """Apply an exposure-control power curve on both sides of scRGB zero."""
    return x.sign() * x.abs().pow(exponent)


def adjust_scene(linear: torch.Tensor, pre_gamma: float = 1.0, exposure: float = 1.0) -> torch.Tensor:
    """The user's Gamma and Exposure on a picture already in the engine's working space:
    the same arithmetic the loaders apply while reading (in scRGB units, signed values
    raised in magnitude to 1 / gamma, then multiplied by the exposure). The preview uses it to shape
    the result while the source view keeps showing the picture as recorded."""
    if pre_gamma == 1.0 and exposure == 1.0:
        return linear
    from .constants import ENGINE_UNIT_NITS

    scale = 80.0 / ENGINE_UNIT_NITS                      # engine units per scRGB unit
    scrgb = linear / scale
    if pre_gamma != 1.0:
        scrgb = signed_power(scrgb, 1.0 / pre_gamma)
    if exposure != 1.0:
        scrgb = scrgb * exposure
    return scrgb * scale

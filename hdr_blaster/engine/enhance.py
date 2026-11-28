"""Edge enhancement (moved verbatim from NHC.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import measure as _measure
from .renderings import srgb_shadow_weight


class EdgeEnhancementBlock(nn.Module):
    """
    Enhances edges in the image using Sobel filters.
    """

    def __init__(self):
        super(EdgeEnhancementBlock, self).__init__()
        gaussian_kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32) / 256.0
        self.register_buffer('gaussian_kernel',
                             gaussian_kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1))

        kernel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32) / 4.0
        kernel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0]
        ], dtype=torch.float32) / 4.0

        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def forward(self, x, edge_strength=0.0, params=None, keep_levels=False):
        """
        Performs edge enhancement using Sobel filters.

        :param x: Input tensor of shape (B, 3, H, W)
        :param edge_strength: Strength of the edge enhancement [0-100]
        """
        if edge_strength <= 0.0:
            return x

        if keep_levels:
            # Reinforce the edges and nothing else. The original path below stretches the
            # result back to the input's range, which dims the WHOLE picture as the
            # strength rises: fine for a look, not for a rendering that promises levels.
            luminance = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
            padded = F.pad(luminance, (1, 1, 1, 1), mode="replicate")
            # Difference each opposing pair first: a flat field must give exactly
            # zero before normalisation, including on the image border.
            dx = padded[..., 2:] - padded[..., :-2]
            dy = padded[..., 2:, :] - padded[..., :-2, :]
            edge_x = 0.25 * (dx[..., :-2, :] + 2 * dx[..., 1:-1, :] + dx[..., 2:, :])
            edge_y = 0.25 * (dy[..., :-2] + 2 * dy[..., 1:-1] + dy[..., 2:])
            magnitude = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
            mag_max = _measure(params, "edge.mag_max", lambda: magnitude.max())
            strength = magnitude / mag_max.clamp_min(torch.finfo(magnitude.dtype).tiny)
            gain = 1.0 + srgb_shadow_weight(x) * strength * ((edge_strength / 100.0) * 0.2)
            return (x * gain).clamp(0.0, 1.0)

        # Normalize input for edge detection
        orig_min = _measure(params, "edge.in_min", lambda: x.min())
        orig_max = _measure(params, "edge.in_max", lambda: x.max())
        x_norm = (x - orig_min) / (orig_max - orig_min + 1e-8)

        # Calculate luminance for edge detection
        luminance = 0.2989 * x_norm[:, 0:1] + 0.5870 * x_norm[:, 1:2] + 0.1140 * x_norm[:, 2:3]

        # Sobel filters
        edge_x = F.conv2d(luminance, self.kernel_x, padding=1)
        edge_y = F.conv2d(luminance, self.kernel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
        mag_max = _measure(params, "edge.mag_max", lambda: edge_magnitude.max())
        edge_magnitude = edge_magnitude / (mag_max + 1e-8)

        # Scale edge strength
        edge_strength = (edge_strength / 100.0) * 0.2
        enhancement = edge_magnitude * edge_strength

        # Apply enhancement
        result = torch.zeros_like(x_norm)
        for c in range(3):
            result[:, c:c + 1] = x_norm[:, c:c + 1] * (1.0 + enhancement)

        # Re-normalize to original range
        out_min = _measure(params, "edge.out_min", lambda: result.min())
        out_max = _measure(params, "edge.out_max", lambda: result.max())
        result = (result - out_min) / (out_max - out_min + 1e-8)
        result = result * (orig_max - orig_min) + orig_min
        result = torch.clamp(result, min=orig_min, max=orig_max)
        return result

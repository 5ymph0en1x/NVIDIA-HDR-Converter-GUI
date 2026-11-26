import os
import sys
import gc
import traceback
import io
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import _tkinter
from PIL import Image, ImageTk, ImageDraw

try:
    import TKinterModernThemes as TKMT
except ImportError:
    # Fallback if TKMT isn't available
    TKMT = None
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
try:
    import matplotlib
    matplotlib.use('Agg')  # headless backend; histograms are drawn with PIL
except Exception:
    matplotlib = None
import numpy as np
import imagecodecs
import struct
from typing import Dict, Tuple, Optional, List
import json
import tifffile

# Logging Configuration
logging.basicConfig(
    filename='hdr_converter.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Constants
DEFAULT_TONE_MAP = "perceptual"
DEFAULT_PREGAMMA = "1.0"
DEFAULT_AUTOEXPOSURE = "1.0"
SUPPORTED_TONE_MAPS = {"agx", "agx_punchy", "agx_golden", "perceptual", "adaptive", "hable", "reinhard",
                       "filmic", "aces", "uncharted2", "mantiuk06", "drago03", "neural"}

# Default render target: AgX is the modern, robust film-like operator (Blender 4+)
# used as the out-of-box default — natural midtones, graceful highlight
# desaturation, no hue skews on bright gaming HDR (sunsets, speculars).
DEFAULT_RENDER_METHOD = "agx_punchy"
# Auto-exposure target: the MEDIAN luminance of the (loader-scaled) scene is
# mapped to this value before AgX. The median is robust — unlike the geometric
# mean it is not dragged down by deep shadows, so high-dynamic-range scenes
# (e.g. bright cherry-blossom / snow captures with dark foliage) are no longer
# over-brightened into a washed-out result. Tuned on NVIDIA scRGB game captures.
AGX_AUTO_EXPOSURE_TARGET = 0.016

# Shadow-recovery for the classical operators. Global tone-map curves crush
# shadows on high-contrast / backlit scenes (a character silhouetted against a
# sunset); a luminance-gated lift in display space recovers that detail without
# touching midtones/highlights or washing well-exposed scenes.
SHADOW_LIFT_AMOUNT = 0.40   # base; an adaptive global gate fades it out on bright scenes

# Gentle "punch" on the classical-operator output (display space) so the result
# keeps the in-game contrast/saturation instead of looking flat on bright scenes.
PUNCH_CONTRAST = 1.12
PUNCH_BLACK = 0.008
PUNCH_SAT = 1.06

MODES = {
    'big': {
        'window_width': 2200,
        'window_height': 976,
        'preview_width': 720,
        'preview_height': 406,
    },
    'small': {
        'window_width': 1760,
        'window_height': 976,
        'preview_width': 512,
        'preview_height': 288,
    },
}

# Set PyTorch backend configurations for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class HDRMetadata:
    """Stores and manages HDR metadata extracted from JXR files."""

    def __init__(self):
        self.max_luminance = 10000.0
        self.min_luminance = 0.0
        self.white_point = 1000.0
        self.color_primaries = "bt2020"
        self.transfer_function = "pq"
        self.color_space = "rec2020"
        self.bit_depth = 10
        self.has_metadata = False

    def extract_from_jxr(self, jxr_data: bytes) -> bool:
        """Extract HDR metadata from JXR file data."""
        try:
            # Look for HDR metadata markers in JXR container
            # This is a simplified version - real implementation would parse JPEGXR container properly
            if b'hdrf' in jxr_data:
                # Extract luminance values
                idx = jxr_data.find(b'hdrf')
                if idx != -1 and idx + 16 <= len(jxr_data):
                    # Parse metadata block (simplified)
                    try:
                        self.max_luminance = struct.unpack('<f', jxr_data[idx + 4:idx + 8])[0]
                        self.min_luminance = struct.unpack('<f', jxr_data[idx + 8:idx + 12])[0]
                        self.white_point = struct.unpack('<f', jxr_data[idx + 12:idx + 16])[0]
                    except struct.error:
                        # If metadata format is different, use defaults
                        pass
                    self.has_metadata = True
                    return True
        except Exception as e:
            logging.debug(f"No HDR metadata found: {e}")
        return False

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            'max_luminance': self.max_luminance,
            'min_luminance': self.min_luminance,
            'white_point': self.white_point,
            'color_primaries': self.color_primaries,
            'transfer_function': self.transfer_function,
            'color_space': self.color_space,
            'bit_depth': self.bit_depth,
            'has_metadata': self.has_metadata
        }


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


class AdvancedToneMapper:
    """Advanced tone mapping with multiple algorithms and automatic selection."""

    def __init__(self, device, metadata: Optional[HDRMetadata] = None):
        self.device = device
        self.metadata = metadata or HDRMetadata()
        self.color_preserver = PerceptualColorPreserver(device)

        # Pre-compute tone mapping LUTs for efficiency
        self._build_luts()

        # Learned tone mapper (None if module or checkpoint missing).
        self.neural_model = _get_neural_tonemap(device)

    def _build_luts(self):
        """Build lookup tables for various tone mapping curves."""
        lut_size = 4096
        x = torch.linspace(0, 20, lut_size, device=self.device)

        # Build LUTs for different operators
        self.hable_lut = self._hable_curve(x)
        self.aces_lut = self._aces_curve(x)
        self.reinhard_lut = self._reinhard_curve(x, L_white=4.0)

        # --- AgX working-space matrices (Rec.709), column-vector convention ---
        # Forward = sRGB/Rec.709 -> AgX inset; Inverse = AgX -> display.
        self._agx_fwd = torch.tensor([
            [0.842479062253094, 0.0784335999999992, 0.0792237451477643],
            [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
            [0.0423756549057051, 0.0784336, 0.879142973793104],
        ], dtype=torch.float32, device=self.device)
        self._agx_inv = torch.tensor([
            [1.19687900512017, -0.0980208811401368, -0.0990297440797205],
            [-0.0528968517574562, 1.15190312990417, -0.0989611768448433],
            [-0.0529716355144438, -0.0980434501171241, 1.15107367264116],
        ], dtype=torch.float32, device=self.device)
        self._agx_min_ev = -12.47393
        self._agx_max_ev = 4.026069

    def _hable_curve(self, x):
        """Hable tone mapping curve."""
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

    def _aces_curve(self, x):
        """ACES RRT+ODT tone mapping curve."""
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

    def _reinhard_curve(self, x, L_white=4.0):
        """Extended Reinhard tone mapping curve."""
        return x * (1.0 + x / (L_white * L_white)) / (1.0 + x)

    def _apply_lut(self, image: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
        """Apply tone mapping LUT to image."""
        # Normalize image values to LUT range
        img_max = image.max()
        if img_max > 0:
            normalized = image / img_max * (len(lut) - 1)
            indices = torch.clamp(normalized.long(), 0, len(lut) - 1)

            # Apply LUT
            B, C, H, W = image.shape
            flat = indices.view(-1)
            mapped_flat = lut[flat]
            mapped = mapped_flat.view(B, C, H, W) * img_max

            return mapped
        return image

    def analyze_image_statistics(self, image: torch.Tensor) -> Dict:
        """Comprehensive HDR image analysis for optimal tone mapping selection."""
        luminance = 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]

        # Basic statistics
        stats = {
            'min_luminance': luminance.min().item(),
            'max_luminance': luminance.max().item(),
            'mean_luminance': luminance.mean().item(),
            'std_luminance': luminance.std().item(),
            'median_luminance': luminance.median().item(),
            'dynamic_range': (luminance.max() / (luminance.min() + 1e-8)).item(),
            'key': torch.exp(torch.log(luminance + 1e-8).mean()).item(),
        }

        # Enhanced histogram analysis
        hist, bins = torch.histogram(torch.log10(luminance.cpu() + 1e-8), bins=256)
        hist_norm = hist.float() / hist.sum()

        # Zone analysis (Ansel Adams Zone System inspired)
        zones = torch.split(hist_norm, 32)  # 8 zones
        zone_weights = torch.tensor([zone.sum().item() for zone in zones])

        stats['shadow_detail'] = zone_weights[:2].sum().item()  # Zones 0-I
        stats['midtone_detail'] = zone_weights[3:5].sum().item()  # Zones III-IV
        stats['highlight_detail'] = zone_weights[6:].sum().item()  # Zones VI-VII

        # Local contrast analysis
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                              dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        contrast = F.conv2d(luminance.unsqueeze(1), kernel, padding=1)
        stats['local_contrast'] = contrast.abs().mean().item()
        stats['contrast_variance'] = contrast.std().item()

        # Color saturation analysis
        rgb_max = image.max(dim=1)[0]
        rgb_min = image.min(dim=1)[0]
        saturation = (rgb_max - rgb_min) / (rgb_max + 1e-8)
        stats['mean_saturation'] = saturation.mean().item()
        stats['saturation_variance'] = saturation.std().item()

        # Specular highlight detection
        highlight_threshold = stats['mean_luminance'] + 2 * stats['std_luminance']
        stats['specular_ratio'] = (luminance > highlight_threshold).float().mean().item()

        # Scene classification
        stats['is_high_key'] = stats['mean_luminance'] > 0.6 and stats['shadow_detail'] < 0.1
        stats['is_low_key'] = stats['mean_luminance'] < 0.3 and stats['highlight_detail'] < 0.1
        stats['has_extreme_highlights'] = stats['specular_ratio'] > 0.05
        stats['needs_shadow_recovery'] = stats['shadow_detail'] > 0.3 and stats['min_luminance'] < 0.01

        # Perceptual metrics
        stats['contrast_score'] = stats['local_contrast'] / (stats['contrast_variance'] + 1e-8)
        stats['detail_score'] = (stats['shadow_detail'] + stats['midtone_detail'] + stats['highlight_detail']) / 3

        return stats

    def select_optimal_tonemap(self, stats: Dict) -> str:
        """Intelligent tone mapping selection based on comprehensive image analysis."""
        dr = stats['dynamic_range']

        # Decision tree with weighted scoring
        scores = {
            'perceptual': 0,
            'mantiuk06': 0,
            'drago03': 0,
            'hable': 0,
            'aces': 0,
            'reinhard': 0,
            'adaptive': 0
        }

        # Dynamic range scoring
        if dr > 10000:
            scores['perceptual'] += 30
            scores['mantiuk06'] += 25
        elif dr > 1000:
            scores['mantiuk06'] += 30
            scores['drago03'] += 20
        elif dr > 100:
            scores['drago03'] += 25
            scores['adaptive'] += 20
        else:
            scores['hable'] += 20
            scores['reinhard'] += 15

        # Scene type scoring
        if stats['is_high_key']:
            scores['reinhard'] += 15
            scores['aces'] += 10
        elif stats['is_low_key']:
            scores['drago03'] += 20
            scores['perceptual'] += 15

        # Detail preservation scoring
        if stats['needs_shadow_recovery']:
            scores['drago03'] += 25
            scores['adaptive'] += 20

        if stats['has_extreme_highlights']:
            scores['perceptual'] += 20
            scores['aces'] += 15

        # Local contrast scoring
        if stats['contrast_score'] > 1.0:
            scores['perceptual'] += 15
            scores['mantiuk06'] += 10
        else:
            scores['hable'] += 10

        # Color saturation scoring
        if stats['mean_saturation'] > 0.5:
            scores['aces'] += 15  # ACES preserves colors well
            scores['hable'] += 10
        elif stats['mean_saturation'] < 0.2:
            scores['mantiuk06'] += 10  # Better for low saturation

        # Metadata influence
        if self.metadata.has_metadata:
            if self.metadata.max_luminance > 4000:
                scores['perceptual'] += 20
            elif self.metadata.max_luminance > 1000:
                scores['mantiuk06'] += 15

        is_high_contrast_scene = stats['needs_shadow_recovery'] and stats['has_extreme_highlights']

        if is_high_contrast_scene:
            logging.info("High-contrast scene detected. Prioritizing local tone mapping operators.")
            # In this case, we give a massive bonus to local operators that can handle this conflict.
            # 'adaptive' is designed for this, and 'mantiuk06' is excellent for local contrast.
            scores['adaptive'] += 50  # Decisive bonus
            scores['mantiuk06'] += 40  # Strong preference
            scores['perceptual'] -= 20  # Penalty for global operators in this scenario

        # Select highest scoring method
        best_method = max(scores, key=scores.get)

        # Log decision
        logging.info(f"Tone mapping scores: {scores}")
        logging.info(f"Selected: {best_method} (score: {scores[best_method]})")
        logging.info(f"Key metrics - DR: {dr:.1f}, Shadow: {stats['shadow_detail']:.2f}, "
                     f"Highlight: {stats['highlight_detail']:.2f}, Contrast: {stats['contrast_score']:.2f}")

        return best_method

    def tone_map_perceptual(self, image: torch.Tensor) -> torch.Tensor:
        """Perceptual tone mapping preserving local contrast and color."""
        # Preserve original for better color mapping
        original = image.clone()

        # Work on luminance only to preserve colors
        luminance = 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]

        # Global tone mapping on luminance
        key_value = torch.exp(torch.log(luminance + 1e-8).mean())
        scaled_lum = luminance / key_value * 0.18

        # Reinhard tone mapping on luminance
        L_white = 2.0  # Reduced from implicit higher value
        tonemapped_lum = scaled_lum * (1.0 + scaled_lum / (L_white * L_white)) / (1.0 + scaled_lum)

        # Apply tone mapping ratio to preserve colors
        ratio = tonemapped_lum / (luminance + 1e-8)
        ratio = torch.clamp(ratio, 0, 2)  # Limit ratio to prevent oversaturation

        # Apply ratio to each channel
        result = original * ratio.unsqueeze(1)

        # Soft clip to prevent hard clipping
        result = torch.where(result > 0.9,
                             0.9 + 0.1 * torch.tanh((result - 0.9) * 2),
                             result)

        return torch.clamp(result, 0, 1)

    def tone_map_mantiuk06(self, image: torch.Tensor) -> torch.Tensor:
        """Proper Mantiuk06 tone mapping based on contrast perception."""
        epsilon = 1e-6

        # Convert to XYZ luminance for proper tone mapping
        luminance = 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]
        luminance = torch.clamp(luminance, epsilon, None)

        # Calculate log-average luminance
        log_avg_lum = torch.exp(torch.log(luminance + epsilon).mean())

        # Scale luminance
        key = 0.18  # Middle grey key value
        scaled_lum = (key / log_avg_lum) * luminance

        # Mantiuk's contrast processing
        # Convert to log domain for contrast processing
        log_lum = torch.log(scaled_lum + epsilon)

        # Calculate local adaptation using Gaussian blur approximation
        # Use separable Gaussian for efficiency
        kernel_size = 7
        sigma = 2.0
        x = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_1d = gaussian_1d.view(1, 1, 1, kernel_size)

        # Apply Gaussian blur
        log_lum_expanded = log_lum.unsqueeze(1)
        blurred = F.conv2d(log_lum_expanded, gaussian_1d, padding=(0, kernel_size // 2))
        blurred = F.conv2d(blurred, gaussian_1d.transpose(-1, -2), padding=(kernel_size // 2, 0))
        local_adaptation = blurred.squeeze(1)

        # Contrast-based tone mapping
        contrast_factor = 0.3  # Adjust contrast sensitivity
        max_contrast = 100.0  # Maximum displayable contrast

        # Calculate local contrast
        local_contrast = log_lum - local_adaptation

        # Compress contrast with smooth function
        compressed_contrast = torch.tanh(local_contrast * contrast_factor) / contrast_factor

        # Reconstruct tone-mapped luminance
        tone_mapped_log_lum = local_adaptation + compressed_contrast
        tone_mapped_lum = torch.exp(tone_mapped_log_lum)

        # Apply gamma correction and normalization
        max_lum = tone_mapped_lum.max()
        if max_lum > 1.0:
            tone_mapped_lum = tone_mapped_lum / max_lum

        # Preserve color ratios
        lum_ratio = tone_mapped_lum / (luminance + epsilon)
        lum_ratio = torch.clamp(lum_ratio, 0.1, 10.0)  # Prevent extreme ratios

        # Apply to all channels while preserving saturation
        result = image * lum_ratio.unsqueeze(1)

        # Boost saturation slightly to compensate for tone mapping
        saturation_boost = 1.1
        mean_rgb = result.mean(dim=1, keepdim=True)
        result = mean_rgb + (result - mean_rgb) * saturation_boost

        return torch.clamp(result, 0, 1)

    def tone_map_drago03(self, image: torch.Tensor) -> torch.Tensor:
        """Drago03 logarithmic tone mapping."""
        # Parameters
        bias = 0.85

        # Compute world adaptation luminance
        luminance = 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]
        Lwa = torch.exp(torch.log(luminance + 1e-8).mean())

        # Maximum luminance
        Lwmax = luminance.max()

        # Bias function
        b = torch.log(torch.tensor(bias, device=image.device)) / torch.log(torch.tensor(0.5, device=image.device))

        # Tone mapping
        Ld = (torch.log(luminance / Lwa + 1) / torch.log(Lwmax / Lwa + 1)) ** (
                    torch.log(b) / torch.log(torch.tensor(0.5, device=image.device)))

        # Apply to color channels
        scale = Ld / (luminance + 1e-8)
        result = image * scale.unsqueeze(1)

        return torch.clamp(result, 0, 1)

    # ----------------------------------------------------------------- AgX ---
    def _agx_mat(self, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply a 3x3 matrix to an (B, 3, H, W) image (column-vector convention)."""
        b, c, h, w = x.shape
        return (m.to(x.dtype) @ x.reshape(b, 3, -1)).reshape(b, 3, h, w)

    @staticmethod
    def _agx_contrast(x: torch.Tensor) -> torch.Tensor:
        """7th-order polynomial approximation of the AgX display sigmoid."""
        x2 = x * x
        x4 = x2 * x2
        return (15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
                - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232)

    def auto_expose(self, image: torch.Tensor, target: float = AGX_AUTO_EXPOSURE_TARGET) -> torch.Tensor:
        """Scale scene-linear RGB so its MEDIAN luminance hits `target`.

        The median is a robust midtone anchor: it ignores the deep-shadow tail
        that biased the old geometric-mean exposure and over-brightened
        high-dynamic-range scenes (washed-out highlights). It also adapts to
        genuinely bright scenes (a fixed exposure would blow those out).
        """
        lum = 0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]
        key = lum.median()
        scale = (target / key.clamp(min=1e-5)).clamp(0.01, 50.0)
        return image * scale

    @staticmethod
    def _srgb_oetf(x: torch.Tensor) -> torch.Tensor:
        """Linear -> gamma sRGB (IEC 61966-2-1)."""
        x = x.clamp(0.0, 1.0)
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055).clamp(0.0, 1.0)

    def tone_map_agx(self, image: torch.Tensor, look: str = "agx_punchy") -> torch.Tensor:
        """AgX tone mapping (Troy Sobotka / Blender 4) + auto-exposure + grade.

        AgX alone is deliberately *filmic-flat* (it lifts blacks and desaturates
        highlights toward white) which reads as washed-out / "no tone mapping" on
        bright game HDR. After the AgX transform we apply a display-space grade —
        black point, contrast, saturation, and a highlight-colour retain — to
        restore the deep-black, saturated, punchy look of the in-game render.

        Returns LINEAR [0, 1] (downstream ``linear_to_srgb`` re-encodes;
        ``_srgb_decode`` is the matching inverse), so the contract is uniform.
        """
        looks = {
            # neutral filmic AgX (no grade) — for users who want the soft look
            "agx":        dict(target=0.018, agx_sat=1.00, black=0.0,  contrast=1.00, sat=1.00, hl=0.0,  warm=(1.00, 1.00, 1.00)),
            # default: punchy, matched to the in-game SDR render
            "agx_punchy": dict(target=0.016, agx_sat=1.15, black=0.03, contrast=1.28, sat=1.20, hl=0.15, warm=(1.00, 1.00, 1.00)),
            # punchy + warm golden-hour tint
            "agx_golden": dict(target=0.016, agx_sat=1.12, black=0.03, contrast=1.26, sat=1.18, hl=0.15, warm=(1.05, 1.00, 0.94)),
        }
        p = looks.get(look, looks["agx_punchy"])

        # --- AgX core ---
        x = self.auto_expose(image, target=p["target"]).clamp(min=0.0)
        g = self._agx_mat(self._agx_fwd, x)
        g = torch.log2(g + 1e-10).clamp(self._agx_min_ev, self._agx_max_ev)
        g = (g - self._agx_min_ev) / (self._agx_max_ev - self._agx_min_ev)
        g = self._agx_contrast(g).clamp(0.0, 1.0)
        if p["agx_sat"] != 1.0:
            gl = 0.2126 * g[:, 0:1] + 0.7152 * g[:, 1:2] + 0.0722 * g[:, 2:3]
            g = (gl + p["agx_sat"] * (g - gl)).clamp(0.0, 1.0)
        srgb = self._srgb_oetf(self._agx_mat(self._agx_inv, g).clamp(0.0, 1.0))

        # --- display-space grade ---
        # Retain highlight colour (AgX whitens bright sky/snow): mix back the
        # exposed-original sRGB, weighted by output brightness.
        if p["hl"] > 0.0:
            ref = self._srgb_oetf(x.clamp(0.0, 1.0))
            m = (0.2126 * srgb[:, 0:1] + 0.7152 * srgb[:, 1:2] + 0.0722 * srgb[:, 2:3]).clamp(0.0, 1.0)
            srgb = srgb * (1.0 - p["hl"] * m) + ref * (p["hl"] * m)
        if p["black"] > 0.0:                                   # deepen blacks
            srgb = ((srgb - p["black"]) / (1.0 - p["black"])).clamp(0.0, 1.0)
        if p["contrast"] != 1.0:                               # contrast about mid grey
            srgb = (0.5 + (srgb - 0.5) * p["contrast"]).clamp(0.0, 1.0)
        if p["sat"] != 1.0:                                    # saturation
            sl = 0.2126 * srgb[:, 0:1] + 0.7152 * srgb[:, 1:2] + 0.0722 * srgb[:, 2:3]
            srgb = (sl + p["sat"] * (srgb - sl)).clamp(0.0, 1.0)
        if p["warm"] != (1.0, 1.0, 1.0):                       # warm tint (golden look)
            warm = torch.tensor(p["warm"], dtype=srgb.dtype, device=srgb.device).view(1, 3, 1, 1)
            srgb = (srgb * warm).clamp(0.0, 1.0)

        return _srgb_decode(srgb)  # -> linear; downstream linear_to_srgb re-encodes

    def apply_tone_mapping(self, image: torch.Tensor, method: Optional[str] = None) -> torch.Tensor:
        """Apply tone mapping with automatic method selection if not specified."""
        # Auto-select: the sophisticated statistics oracle scores the classical
        # operators and picks the best one (the user's preferred default).
        if method is None:
            stats = self.analyze_image_statistics(image)
            method = self.select_optimal_tonemap(stats)
            logging.info(f"Auto-selected tone mapping method: {method}")

        # Store original for color preservation
        original = image.clone()

        # Apply selected tone mapping
        if method in ('agx', 'agx_punchy', 'agx_golden'):
            tonemapped = self.tone_map_agx(image, look=method)
        elif method == 'neural':
            if self.neural_model is None:
                logging.warning(
                    "Neural tone mapper requested but unavailable; "
                    "falling back to perceptual."
                )
                tonemapped = self.tone_map_perceptual(image)
            else:
                tonemapped = self._tone_map_neural(image)
        elif method == 'perceptual':
            tonemapped = self.tone_map_perceptual(image)
        elif method == 'mantiuk06':
            tonemapped = self.tone_map_mantiuk06(image)
        elif method == 'drago03':
            tonemapped = self.tone_map_drago03(image)
        elif method == 'adaptive':
            # Combine multiple operators based on image regions
            stats = self.analyze_image_statistics(image)
            if stats['has_extreme_highlights']:
                highlights = self.tone_map_perceptual(image)
            else:
                highlights = self._apply_lut(image, self.aces_lut)

            if stats['needs_shadow_recovery']:
                shadows = self.tone_map_drago03(image)
            else:
                shadows = self._apply_lut(image, self.hable_lut)

            # Blend based on luminance
            luminance = 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]
            blend_mask = torch.sigmoid((luminance - 0.5) * 10)
            tonemapped = shadows * (1 - blend_mask.unsqueeze(1)) + highlights * blend_mask.unsqueeze(1)
        elif method == 'hable':
            tonemapped = self._hable_operator(image)
        elif method == 'aces':
            tonemapped = self._aces_operator(image)
        elif method == 'reinhard':
            tonemapped = self._reinhard_operator(image)
        else:
            # Fallback to simple Reinhard
            tonemapped = image / (1 + image)

        # Skip color preservation for now to avoid color issues
        # tonemapped = self.color_preserver.preserve_color_ratios(original, tonemapped)

        # AgX and the neural mapper already produce display-ready [0,1] with a
        # graceful highlight rolloff — an extra soft-clip would only crush them.
        # The classical curve operators still benefit from it.
        if method not in ('agx', 'agx_punchy', 'agx_golden', 'neural'):
            tonemapped = self.soft_clip(tonemapped)
            # Display-space finishing for the classical operators: recover backlit
            # shadow detail (adaptive), then a gentle punch for in-game contrast.
            # Mapped back to linear so the return contract (linear [0,1]) stays
            # uniform for the downstream linear_to_srgb.
            srgb = self._shadow_lift(self._srgb_oetf(tonemapped))
            srgb = self._punch(srgb)
            tonemapped = _srgb_decode(srgb)

        return tonemapped

    def _tone_map_neural(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply the learned neural tone mapper.

        Input  : HDR linear tensor (B, 3, H, W), unbounded non-negative.
        Output : tone-mapped tensor in linear-light [0, 1]. The neural
                 model emits gamma-encoded sRGB directly, so we undo
                 the sRGB EOTF here — the downstream `linear_to_srgb`
                 step re-encodes it back without double-encoding.
        """
        with torch.inference_mode():
            srgb_out = self.neural_model(image)
        return _srgb_decode(srgb_out)

    def _hable_operator(self, x):
        """Direct Hable operator without LUT."""
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        W = 11.2

        def hable(v):
            return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F

        # Reduce exposure to prevent oversaturation
        curr = hable(x * 1.0)  # Changed from 2.0
        white_scale = hable(torch.tensor(W, device=x.device, dtype=x.dtype))
        return curr / white_scale

    def _aces_operator(self, x):
        """Direct ACES operator without LUT."""
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

    def _reinhard_operator(self, x):
        """Direct Reinhard operator without LUT."""
        L_white = 4.0
        return x * (1.0 + x / (L_white * L_white)) / (1.0 + x)

    def apply_dithering(self, image: torch.Tensor, strength: float = 1.0 / 255.0) -> torch.Tensor:
        """
        Triangular-PDF (TPDF) dithering — the last operation before quantization.

        A fixed ordered (Bayer) pattern at +/-0.5 LSB is too weak and too
        structured to break banding on gradual gradients (e.g. a sunset sky).
        TPDF noise of +/-1 LSB, independent per pixel AND per channel, fully
        decorrelates the quantization error and removes the banding. ``strength``
        is one LSB of the target depth (1/255 for 8-bit JPEG, 1/65535 for TIFF).
        """
        # (U - U) is triangular on [-1, 1]; * strength -> +/-1 LSB TPDF dither.
        noise = (torch.rand_like(image) - torch.rand_like(image)) * strength
        return torch.clamp(image + noise, 0.0, 1.0)

    def soft_clip(self, image: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
        """Soft-clip HIGHLIGHTS to avoid a hard cutoff at 1.0.

        The old shadow branch pulled darks toward 0 and *crushed* shadow detail
        on backlit / high-contrast scenes — removed. Shadow detail is recovered
        by ``_shadow_lift`` instead.
        """
        image = image.clone()
        over_mask = image > threshold
        if over_mask.any():
            over_values = image[over_mask]
            compressed = threshold + (1 - threshold) * torch.tanh((over_values - threshold) / (1 - threshold))
            image[over_mask] = compressed
        return image

    def _shadow_lift(self, srgb: torch.Tensor, amount: float = SHADOW_LIFT_AMOUNT,
                     knee: float = 0.5) -> torch.Tensor:
        """Adaptive, luminance-gated shadow lift in display (sRGB) space.

        Two gates:
          * GLOBAL — fades the whole lift out on bright / high-key scenes (where
            lifting darks would only flatten the contrast) and applies it at full
            strength on dark, backlit scenes that need shadow recovery.
          * LOCAL  — within a lifted scene, only deep shadows are raised; midtones
            and highlights are untouched.
        """
        lum = 0.2126 * srgb[:, 0:1] + 0.7152 * srgb[:, 1:2] + 0.0722 * srgb[:, 2:3]
        # global gate from the scene's mean display brightness (full <0.30, off >0.46)
        sm = lum.mean()
        tg = ((sm - 0.30) / 0.16).clamp(0.0, 1.0)
        amt = amount * (1.0 - tg * tg * (3.0 - 2.0 * tg))
        if float(amt) < 1e-3:
            return srgb
        # local gate -> deep shadows only
        g = (1.0 - (lum / knee).clamp(0.0, 1.0))
        g = g * g * (3.0 - 2.0 * g)
        return (srgb.clamp(min=1e-6) ** (1.0 - amt * g)).clamp(0.0, 1.0)

    def _punch(self, srgb: torch.Tensor, contrast: float = PUNCH_CONTRAST,
               black: float = PUNCH_BLACK, sat: float = PUNCH_SAT) -> torch.Tensor:
        """Gentle display-space grade (black point + S-contrast about mid grey +
        saturation) so the classical operators keep the punchy in-game look instead
        of rendering flat. Mild enough not to re-crush the lifted shadows."""
        if black > 0.0:
            srgb = ((srgb - black) / (1.0 - black)).clamp(0.0, 1.0)
        srgb = (0.5 + (srgb - 0.5) * contrast).clamp(0.0, 1.0)
        if sat != 1.0:
            lum = 0.2126 * srgb[:, 0:1] + 0.7152 * srgb[:, 1:2] + 0.0722 * srgb[:, 2:3]
            srgb = (lum + sat * (srgb - lum)).clamp(0.0, 1.0)
        return srgb


class DeviceManager:
    """
    Manages the computational device (GPU or CPU) for the application.
    """

    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        if self.use_gpu:
            torch.cuda.empty_cache()
        logging.info(f"DeviceManager initialized on {self.device}")

    def switch_device(self, use_gpu: bool):
        """
        Switches the device between GPU and CPU based on availability and user preference.
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        new_device = torch.device("cuda" if self.use_gpu else "cpu")
        if new_device != self.device:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.device = new_device
            logging.info(f"DeviceManager switched to {self.device}")

    def get_device(self):
        """Returns the current computational device."""
        return self.device


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


def _app_data_dir():
    """Base directory for bundled app data (models, weights, model.py, icon).

    A frozen one-file build extracts bundled data to ``sys._MEIPASS``; read the
    assets from there so the .exe is fully self-contained. In dev, use the
    script directory.
    """
    if getattr(sys, 'frozen', False):  # PyInstaller bundle
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    # FROZEN ORACLE: the only edit to this copy of NHC.py. Assets live at the project root,
    # two levels above tests/legacy/.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    def forward(self, x):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        weights = self.classifier(x_clamped).flatten(1)  # (B, 3)
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
            map_location=device, weights_only=False,
        )
        saved_args = ckpt.get("args", {}) or {}
        model = _NeuralToneMapperImpl(
            grid_d=saved_args.get("grid_d", 8),
            grid_h=saved_args.get("grid_hw", 16),
            grid_w=saved_args.get("grid_hw", 16),
        ).to(device).eval()
        model.load_state_dict(ckpt["model"])
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


def _srgb_decode(x: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB EOTF. Used to undo the neural model's sRGB output
    so the downstream linear_to_srgb step re-encodes correctly."""
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


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

    def forward(self, x, edge_strength=0.0):
        """
        Performs edge enhancement using Sobel filters.

        :param x: Input tensor of shape (B, 3, H, W)
        :param edge_strength: Strength of the edge enhancement [0-100]
        """
        if edge_strength <= 0.0:
            return x

        # Normalize input for edge detection
        orig_min = x.min()
        orig_max = x.max()
        x_norm = (x - orig_min) / (orig_max - orig_min + 1e-8)

        # Calculate luminance for edge detection
        luminance = 0.2989 * x_norm[:, 0:1] + 0.5870 * x_norm[:, 1:2] + 0.1140 * x_norm[:, 2:3]

        # Sobel filters
        edge_x = F.conv2d(luminance, self.kernel_x, padding=1)
        edge_y = F.conv2d(luminance, self.kernel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)

        # Scale edge strength
        edge_strength = (edge_strength / 100.0) * 0.2
        enhancement = edge_magnitude * edge_strength

        # Apply enhancement
        result = torch.zeros_like(x_norm)
        for c in range(3):
            result[:, c:c + 1] = x_norm[:, c:c + 1] * (1.0 + enhancement)

        # Re-normalize to original range
        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        result = result * (orig_max - orig_min) + orig_min
        result = torch.clamp(result, min=orig_min, max=orig_max)
        return result


class OptimizedJXRLoader:
    """
    Handles loading and processing of JXR images.
    This class includes tone mapping operations for preview and final processing.
    """

    def __init__(self, device):
        self.device = device
        self.hdr_peak_luminance = 10000.0
        self.selected_pre_gamma = 1.0
        self.selected_auto_exposure = 1.0
        self.metadata = HDRMetadata()
        self.tone_mapper = None

    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Safely applies gamma correction to the input image array.
        """
        pos_mask = image > 0
        result = np.zeros_like(image)
        if np.any(pos_mask):
            result[pos_mask] = np.power(image[pos_mask], gamma)
        return result

    def load_jxr(self, file_path: str):
        """
        Loads and preprocesses a JXR image, returning a linear HDR tensor.
        Tone mapping is NOT applied here.
        """
        try:
            with open(file_path, 'rb') as f:
                jxr_data = f.read()

            # Extract metadata
            self.metadata.extract_from_jxr(jxr_data)

            # Decode JXR
            try:
                logging.debug(f"Attempting to decode JXR file: {file_path}")
                image = imagecodecs.jpegxr_decode(jxr_data)
                if not isinstance(image, np.ndarray):
                    raise ValueError(f"Unexpected decode result type: {type(image)}")
                logging.debug(f"Successfully decoded image with shape: {image.shape}, dtype: {image.dtype}")
            except Exception as decode_error:
                logging.error(f"Image decode failed: {decode_error}", exc_info=True)
                raise ValueError(f"Failed to decode JXR: {decode_error}")

            if image is None:
                raise ValueError("Failed to decode JXR image")
            image = image.astype(np.float32)

            # Update peak luminance from metadata if available
            if self.metadata.has_metadata and self.metadata.max_luminance > 0 and not np.isnan(
                    self.metadata.max_luminance):
                self.hdr_peak_luminance = self.metadata.max_luminance

            pre_gamma = self.selected_pre_gamma
            auto_exposure = self.selected_auto_exposure

            # Apply gamma correction
            if pre_gamma != 1.0:
                gamma = 1.0 / pre_gamma
                image = self._apply_gamma_correction(image, gamma)

            # Apply exposure correction
            if auto_exposure != 1.0:
                image *= auto_exposure

            # Scale image to display range
            display_peak = 1000.0
            if self.metadata.has_metadata and self.metadata.white_point > 0 and not np.isnan(self.metadata.white_point):
                display_peak = self.metadata.white_point
            image = image * (display_peak / self.hdr_peak_luminance)

            # Handle grayscale or RGBA
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] > 4:
                image = image[:, :, :3]

            # Replace NaNs and inf
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

            # Convert to torch Tensor
            tensor = torch.from_numpy(image).float().permute(2, 0, 1).contiguous().unsqueeze(0)
            tensor = torch.clamp(tensor, min=0.0)
            tensor = tensor.to(self.device, non_blocking=True)

            # Create tone mapper with metadata for later use
            if self.tone_mapper is None:
                self.tone_mapper = AdvancedToneMapper(self.device, self.metadata)

            return tensor, None, self.metadata.to_dict()
        except Exception as e:
            logging.error(f"Failed to load HDR image: {e}", exc_info=True)
            return None, str(e), {}

    def linear_to_srgb(self, linear_rgb):
        """Converts linear RGB to sRGB."""
        linear_rgb = torch.clamp(linear_rgb, 0.0, 1.0)
        a = 0.055
        gamma = 2.4
        srgb = torch.where(
            linear_rgb <= 0.0031308,
            12.92 * linear_rgb,
            (1 + a) * torch.pow(linear_rgb, 1.0 / gamma) - a
        )
        return torch.clamp(srgb, 0.0, 1.0)

    def process_preview(self, tensor_data, target_width: int, target_height: int):
        """
        Generates a lower-resolution preview from the loaded tensor.
        """
        try:
            tensor = tensor_data[0] if isinstance(tensor_data, tuple) else tensor_data
            if tensor is None:
                return None
            _, _, h, w = tensor.shape
            width_ratio = target_width / w
            height_ratio = target_height / h
            scale_factor = min(width_ratio, height_ratio)
            new_width = int(w * scale_factor)
            new_height = int(h * scale_factor)
            with torch.inference_mode():
                preview_tensor = F.interpolate(tensor, size=(new_height, new_width),
                                               mode='bicubic', align_corners=False, antialias=True)
            return torch.clamp(preview_tensor, 0.0, 1.0)
        except Exception as e:
            logging.error(f"Preview generation failed: {str(e)}")
            return None

    def tensor_to_pil(self, tensor: torch.Tensor):
        """
        Converts a tensor (0-1 range in float) to a PIL Image.
        """
        try:
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor = tensor.float()  # Convert to float32 if not already
            tensor = torch.clamp(tensor, 0.0, 1.0)
            tensor = (tensor * 255).byte()
            img_array = tensor.squeeze(0).permute(1, 2, 0).numpy()
            return Image.fromarray(img_array)
        except Exception as e:
            logging.error(f"Tensor to PIL conversion failed: {e}")
            return None


class HDRColorProcessor:
    """
    Processes HDR images with optional color correction and edge enhancements.
    Manages GPU/CPU usage and half-precision toggling if desired.
    """

    def __init__(self, device, jxr_loader, use_fp16=False):
        self.device = device
        self.jxr_loader = jxr_loader
        self.use_fp16 = use_fp16
        self.color_net = ImageAdaptive3DLUT().eval().to(device)

        if device.type == 'cuda' and self.use_fp16:
            self.color_net.half()
        elif device.type == 'cpu' and self.use_fp16:
            try:
                self.color_net.half()
            except Exception as e:
                logging.error(f"Failed to convert model to FP16 on CPU: {e}")
                self.use_fp16 = False

        self.edge_enhancement = EdgeEnhancementBlock().to(device)
        if device.type == 'cuda' and self.use_fp16:
            self.edge_enhancement.half()
        elif device.type == 'cpu' and self.use_fp16:
            try:
                self.edge_enhancement.half()
            except Exception as e:
                logging.error(f"Failed to convert EdgeEnhancementBlock to FP16 on CPU: {e}")
                self.use_fp16 = False

    def clear_gpu_memory(self):
        """Clears GPU memory cache if on CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def process_image(self, original_tensor, color_strength=0.0, edge_strength=0.0, use_enhancement=True):
        """
        Process an HDR image tensor with color and edge enhancements, returning the processed tensor.
        """
        try:
            tensor = original_tensor.to(self.device, dtype=torch.float16 if self.use_fp16 else torch.float32)

            pad_size = 32
            padded_tensor = F.pad(tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

            with torch.inference_mode():
                if use_enhancement and color_strength > 0 and self.color_net.enabled:
                    lut_output = self.color_net(padded_tensor)
                    alpha = color_strength / 100.0
                    # Luminance-aware blend: the FiveK LUT lifts shadows and may
                    # shift highlights — both undesirable on dramatic HDR captures.
                    # We protect deep shadows (lum < ~0.05) and specular highlights
                    # (lum > ~0.98) by smoothly attenuating the LUT effect there,
                    # leaving full effect on midtones where retouching makes sense.
                    lum = (0.2126 * padded_tensor[:, 0:1]
                           + 0.7152 * padded_tensor[:, 1:2]
                           + 0.0722 * padded_tensor[:, 2:3])
                    shadow_ramp = ((lum - 0.05) / 0.25).clamp(0.0, 1.0)
                    shadow_ramp = shadow_ramp * shadow_ramp * (3.0 - 2.0 * shadow_ramp)
                    highlight_ramp = (1.0 - (lum - 0.85) / 0.13).clamp(0.0, 1.0)
                    highlight_ramp = highlight_ramp * highlight_ramp * (3.0 - 2.0 * highlight_ramp)
                    tone_mask = shadow_ramp * highlight_ramp
                    effective_alpha = alpha * tone_mask
                    enhanced = padded_tensor * (1.0 - effective_alpha) + lut_output * effective_alpha
                else:
                    enhanced = padded_tensor

                if edge_strength > 0:
                    enhanced = self.edge_enhancement(enhanced, edge_strength=edge_strength)

                enhanced = enhanced[:, :, pad_size:-pad_size, pad_size:-pad_size]

                if self.device.type == 'cpu' or (self.device.type == 'cuda' and self.use_fp16):
                    enhanced = enhanced.float()

                self.clear_gpu_memory()
                return enhanced

        except Exception as e:
            logging.error(f"Processing failed: {str(e)}", exc_info=True)
            return original_tensor

    def save_tensor_as_image(self, tensor_to_save, output_path, output_format, is_hdr_data=False, quality=95):
        """
        Saves a tensor as an image file (JPEG or TIFF).
        Applies adaptive ordered dithering before quantization for LDR formats
        to prevent banding.
        """
        try:
            if tensor_to_save.is_cuda:
                tensor_to_save = tensor_to_save.cpu()

            tensor_to_save = tensor_to_save.detach()

            # --- Start of dithering logic ---
            # Apply dithering only to LDR (non-HDR) images that will be quantized.
            if not is_hdr_data:
                dither_strength = 0.0
                if output_format == 'JPEG':
                    # Dithering intensity is adjusted for 8-bit target.
                    dither_strength = 1.0 / 255.0
                elif output_format == 'TIFF':
                    # Dithering intensity is adjusted for 16-bit target.
                    dither_strength = 1.0 / 65535.0

                # If an intensity has been defined, apply dithering.
                if dither_strength > 0 and self.jxr_loader.tone_mapper is not None:
                    logging.info(
                        f"Applying dithering for {output_format} format with intensity {dither_strength:.6f}")
                    tensor_to_save = self.jxr_loader.tone_mapper.apply_dithering(tensor_to_save,
                                                                                 strength=dither_strength)
            # --- End of dithering logic ---

            if output_format == 'JPEG':
                # Convert tensor (potentially dithered) to NumPy array.
                array = tensor_to_save.squeeze(0).permute(1, 2, 0).contiguous().numpy()
                array = np.clip(array, 0.0, 1.0)
                # Round (not truncate) to 8-bit; the TPDF dither above already
                # broke the banding. Save 4:4:4 (subsampling=0): JPEG's default
                # 4:2:0 chroma subsampling re-bands smooth colour gradients.
                array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
                result_image = Image.fromarray(array)
                result_image.save(output_path, 'JPEG', quality=max(quality, 97),
                                  subsampling=0, optimize=True)

            elif output_format == 'TIFF':
                # Convert tensor (potentially dithered) to NumPy array.
                array = tensor_to_save.squeeze(0).permute(1, 2, 0).contiguous().numpy()

                if is_hdr_data:
                    # Save raw HDR data as float32, without dithering or quantization.
                    array = np.ascontiguousarray(array, dtype=np.float32)
                    tifffile.imwrite(output_path, array, photometric='rgb', compression='lzw')
                else:
                    # For LDR data, 16-bit quantization (rounded).
                    array = np.clip(array, 0.0, 1.0)
                    array = np.clip(np.rint(array * 65535.0), 0, 65535).astype(np.uint16)
                    tifffile.imwrite(output_path, array, photometric='rgb', compression='lzw')

            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            logging.info(f"Image saved successfully: {output_path}")

        except Exception as e:
            logging.error(f"Failed to save image to {output_path}: {e}", exc_info=True)
            raise

    def switch_device(self, use_gpu, use_fp16=False):
        """
        Switches the processing device and updates model data types accordingly.
        """
        new_device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

        if new_device.type != 'cuda':
            use_fp16 = False

        if new_device != self.device or use_fp16 != self.use_fp16:
            self.clear_gpu_memory()
            self.device = new_device
            self.use_fp16 = use_fp16

            desired_dtype = torch.float16 if (self.device.type == 'cuda' and self.use_fp16) else torch.float32

            try:
                self.color_net = self.color_net.to(device=self.device, dtype=desired_dtype)
                self.edge_enhancement = self.edge_enhancement.to(device=self.device, dtype=desired_dtype)

                self.jxr_loader.tone_mapper = AdvancedToneMapper(self.device, self.jxr_loader.metadata)

            except Exception as e:
                logging.error(f"Failed to switch device or precision: {e}")
                return False

            logging.info(f"Switched to {self.device} with {'FP16' if self.use_fp16 else 'FP32'} precision")
            return True
        else:
            logging.info(f"Already on {self.device} with {'FP16' if self.use_fp16 else 'FP32'} precision")
            return False


def validate_files(input_file: str, output_file: str) -> None:
    """Validates the existence of input and output file paths."""
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError("Input file does not exist")
    if not output_file:
        raise ValueError("Output file path not specified")


def validate_parameters(pre_gamma: str, auto_exposure: str) -> None:
    """Validates the pre-gamma and auto-exposure parameters."""
    try:
        float(pre_gamma)
        float(auto_exposure)
    except ValueError:
        raise ValueError("Pre-gamma and auto-exposure must be numeric")


class App(TKMT.ThemedTKinterFrame if TKMT else tk.Tk):
    """
    Main application class for the NVIDIA HDR Converter GUI.
    Sets up the UI, configures user options, and handles file/folder batch conversions.
    """

    def __init__(self, theme="park", mode="dark"):
        self.use_theme = False

        if TKMT:
            try:
                super().__init__("HDR Image Converter", theme, mode)
                self.use_theme = True
                self.root = self.master
            except (_tkinter.TclError, Exception) as e:
                logging.warning(f"Theme initialization failed: {e}. Using basic tkinter.")
                tk.Tk.__init__(self)
                self.title("HDR Image Converter")
                self.configure(bg='#2b2b2b')
                self.root = self
                self.master = self
        else:
            super().__init__()
            self.title("HDR Image Converter")
            self.configure(bg='#2b2b2b')
            self.root = self
            self.master = self

        self.device_manager = DeviceManager()
        self.jxr_loader = OptimizedJXRLoader(self.device_manager.get_device())
        self.use_fp16_var = tk.BooleanVar(value=False)
        self.color_processor = HDRColorProcessor(
            self.device_manager.get_device(),
            self.jxr_loader,
            use_fp16=self.use_fp16_var.get()
        )

        self.current_before_image_path = None
        self.current_after_image_path = None
        self.original_input_tensor = None
        self.current_enhanced_tensor = None

        self.before_image_ref = None
        self.after_image_ref = None
        self.before_hist_ref = None
        self.after_hist_ref = None

        # Modern visual pass (accent + typography) and window icon.
        self.accent = "#76B900"  # NVIDIA green
        self._apply_modern_style()
        self._set_window_icon()

        main_frame = ttk.Frame(self.master, padding=(10, 10))
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame, padding=(10, 10))
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        left_frame.grid_columnconfigure(0, weight=1)

        right_frame = ttk.Frame(main_frame, padding=(10, 10))
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        # Header band, then a container that holds the control sections.
        self._setup_header(left_frame)
        controls_frame = ttk.Frame(left_frame)
        controls_frame.grid(row=1, column=0, sticky="new")
        controls_frame.grid_columnconfigure(0, weight=1)

        # Setup various UI components
        self._setup_mode_selection(controls_frame)
        self._setup_file_selection(controls_frame)
        self._setup_device_selection(controls_frame)
        self._setup_parameters(controls_frame)
        self._setup_color_controls(controls_frame)
        self._setup_conversion_button(controls_frame)
        self._setup_progress_bar(controls_frame)
        self._setup_status_label(controls_frame)

        self.mode_size_var = tk.StringVar(value='small')
        self.current_mode = 'small'
        self.preview_width = MODES[self.current_mode]['preview_width']
        self.preview_height = MODES[self.current_mode]['preview_height']

        self._setup_previews(right_frame)
        self._setup_histograms(right_frame)
        self.update_mode_size()
        self.ui_lock = threading.Lock()

        mode_switch_frame = ttk.Frame(main_frame, padding=(10, 10))
        mode_switch_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(mode_switch_frame, text="UI Mode:").grid(row=0, column=0, sticky="e", padx=(0, 5))
        big_radio = ttk.Radiobutton(mode_switch_frame, text="Big", variable=self.mode_size_var, value='big',
                                    command=self.update_mode_size)
        big_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        small_radio = ttk.Radiobutton(mode_switch_frame, text="Small", variable=self.mode_size_var, value='small',
                                      command=self.update_mode_size)
        small_radio.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        self.center_window(MODES[self.current_mode]['window_width'], MODES[self.current_mode]['window_height'])

        if self.device_manager.use_gpu:
            self.enhance_checkbox.config(state='normal')
            self.edge_scale.config(state='normal')
            self.fp16_toggle.config(state='normal')
        else:
            self.enhance_checkbox.config(state='disabled')
            self.edge_scale.config(state='disabled')
            self.use_enhancement.set(False)
            self._update_enhancement_controls()
            self.fp16_toggle.config(state='disabled')
            self.use_fp16_var.set(False)

        logging.info(f"Application initialized on {self.device_manager.device}")

    def _apply_modern_style(self):
        """Layer modern typography + accent-coloured section titles on the theme."""
        try:
            style = ttk.Style()
            style.configure(".", font=("Segoe UI", 10))
            style.configure("TLabelframe.Label", font=("Segoe UI Semibold", 10),
                            foreground=self.accent)
            style.configure("AppTitle.TLabel", font=("Segoe UI Semibold", 19),
                            foreground="#F2F2F4")
            style.configure("AppSubtitle.TLabel", font=("Segoe UI", 9),
                            foreground="#8A8A95")
            style.configure("Accent.TButton", font=("Segoe UI Semibold", 11),
                            padding=(10, 9))
            style.configure("TButton", padding=(8, 6))
        except Exception as e:
            logging.warning(f"Style pass failed: {e}")

    def _set_window_icon(self):
        """Set the window icon from app.ico (frozen-aware), best-effort."""
        try:
            ico = os.path.join(_app_data_dir(), "app.ico")
            if os.path.exists(ico):
                self.root.iconbitmap(default=ico)
        except Exception as e:
            logging.debug(f"Window icon not set: {e}")

    def _setup_header(self, parent):
        """App header band: title, subtitle, accent rule."""
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(header, text="NVIDIA HDR Converter",
                  style="AppTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Intelligent HDR tone mapping  ·  CUDA-accelerated",
                  style="AppSubtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        tk.Frame(header, height=2, bg=self.accent).grid(row=2, column=0, sticky="ew", pady=(9, 0))

    def _setup_mode_selection(self, parent_frame):
        """Sets up the mode selection radio buttons (Single File or Folder)."""
        mode_frame = ttk.LabelFrame(parent_frame, text="Mode Selection", padding=(10, 10))
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.mode_var = tk.StringVar(value="single")
        single_radio = ttk.Radiobutton(mode_frame, text="Single File", variable=self.mode_var, value="single",
                                       command=self.update_mode)
        single_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        folder_radio = ttk.Radiobutton(mode_frame, text="Folder", variable=self.mode_var, value="folder",
                                       command=self.update_mode)
        folder_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    def update_mode(self):
        """Updates the UI based on the selected mode (Single File or Folder)."""
        mode = self.mode_var.get()
        if mode == "single":
            self.file_frame.config(text="File Selection")
            self.input_label.config(text="Input JXR:")
            self.output_label.config(text="Output Base Path:")
            self.output_label.grid()
            self.output_entry.grid()
            self.output_browse_button.grid()
            self.status_label.config(text="")
            self.convert_btn.config(state='normal')
        else:
            self.file_frame.config(text="Folder Selection")
            self.input_label.config(text="Input Folder:")
            self.output_label.config(text="Output Folder:")
            folder_path = self.input_entry.get().strip()
            if folder_path and os.path.isdir(folder_path):
                jxr_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jxr')]
                file_count = len(jxr_files)
                if file_count == 0:
                    self.status_label.config(text="No JXR files found in selected folder.", foreground="red")
                    self.convert_btn.config(state='disabled')
                else:
                    self.status_label.config(
                        text=f"Found {file_count} JXR files ready for conversion.",
                        foreground="#00FF00"
                    )
                    self.convert_btn.config(state='normal')
            else:
                self.status_label.config(text="Please select a folder containing JXR files.", foreground="#CCCCCC")
                self.convert_btn.config(state='disabled')

            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
            self.after_label.config(image="", text="No Preview")
            self.after_image_ref = None
            self.before_hist_label.config(image="", text="No Histogram")
            self.before_hist_ref = None
            self.after_hist_label.config(image="", text="No Histogram")
            self.after_hist_ref = None

    def _setup_file_selection(self, parent_frame):
        """Sets up file or folder selection widgets."""
        self.file_frame = ttk.LabelFrame(parent_frame, text="Input / Output", padding=(10, 10))
        self.file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Input Path
        self.input_label = ttk.Label(self.file_frame, text="Input JXR:")
        self.input_label.grid(row=0, column=0, sticky="w", padx=(0, 5), pady=5)
        self.input_entry = ttk.Entry(self.file_frame, width=50)
        self.input_entry.grid(row=1, column=0, columnspan=2, padx=(0, 5), pady=2, sticky="ew")
        self.browse_button = ttk.Button(self.file_frame, text="Browse...", command=self.browse_input)
        self.browse_button.grid(row=1, column=2, padx=(0, 5), pady=2)

        # Output Path
        self.output_label = ttk.Label(self.file_frame, text="Output Base Path:")
        self.output_label.grid(row=2, column=0, sticky="w", padx=(0, 5), pady=5)
        self.output_entry = ttk.Entry(self.file_frame, width=50)
        self.output_entry.grid(row=3, column=0, columnspan=2, padx=(0, 5), pady=2, sticky="ew")
        self.output_browse_button = ttk.Button(self.file_frame, text="Browse...", command=self.browse_output)
        self.output_browse_button.grid(row=3, column=2, padx=(0, 5), pady=2)

        # Output Format
        ttk.Label(self.file_frame, text="Output Format:").grid(row=4, column=0, sticky="w", padx=(0, 5), pady=(10, 5))
        self.output_format_var = tk.StringVar(value="JPEG")
        self.output_format_combo = ttk.Combobox(
            self.file_frame,
            textvariable=self.output_format_var,
            values=["JPEG", "TIFF", "Both"],
            state="readonly",
            width=15
        )
        self.output_format_combo.grid(row=4, column=1, padx=(0, 5), pady=(10, 5), sticky="w")

        # Tone Mapping Selection — the intelligent Auto-detect oracle is the default.
        ttk.Label(self.file_frame, text="Tone Mapping:").grid(row=5, column=0, sticky="w", padx=(0, 5), pady=(10, 5))
        _tm_values = ["Auto-detect"]
        _tm_default = "Auto-detect"
        if _NEURAL_TONEMAP_AVAILABLE and os.path.exists(NEURAL_TONEMAP_CHECKPOINT_PATH):
            _tm_values.append("Neural (AI)")
        _tm_values.extend(
            ["Perceptual", "Mantiuk06", "Drago03", "Hable", "ACES", "Reinhard", "Adaptive",
             "AgX (Punchy)", "AgX (Neutral)", "AgX (Golden)"]
        )
        self.tone_mapping_var = tk.StringVar(value=_tm_default)
        self.tone_mapping_combo = ttk.Combobox(
            self.file_frame,
            textvariable=self.tone_mapping_var,
            values=_tm_values,
            state="readonly",
            width=15
        )
        self.tone_mapping_combo.grid(row=5, column=1, padx=(0, 5), pady=(10, 5), sticky="w")

    def get_selected_tone_mapping_method(self):
        """Convert display name to internal tone mapping method name."""
        display_to_internal = {
            "AgX (Punchy)": "agx_punchy",
            "AgX (Neutral)": "agx",
            "AgX (Golden)": "agx_golden",
            "Auto-detect": None,  # None triggers the intelligent oracle (select_optimal_tonemap)
            "Neural (AI)": "neural",
            "Hable": "hable",
            "ACES": "aces",
            "Reinhard": "reinhard",
            "Mantiuk06": "mantiuk06",
            "Drago03": "drago03",
            "Perceptual": "perceptual",
            "Adaptive": "adaptive"
        }
        return display_to_internal.get(self.tone_mapping_var.get(), None)

    def _method_for(self, linear_tensor, tone_mapper):
        """Resolve the concrete tone-map method for previews: the user's pick, or
        the Auto-detect oracle's choice when 'Auto-detect' is selected."""
        m = self.get_selected_tone_mapping_method()
        if m is None:
            m = tone_mapper.select_optimal_tonemap(tone_mapper.analyze_image_statistics(linear_tensor))
        return m

    def _setup_device_selection(self, parent_frame):
        """Sets up device selection radio buttons (GPU or CPU) and precision toggle."""
        device_frame = ttk.LabelFrame(parent_frame, text="Processing Device", padding=(10, 10))
        device_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.use_gpu_var = tk.BooleanVar(value=self.device_manager.use_gpu)
        gpu_state = 'normal' if torch.cuda.is_available() else 'disabled'
        gpu_label = "GPU (CUDA)" if torch.cuda.is_available() else "GPU (Not Available)"
        gpu_radio = ttk.Radiobutton(device_frame, text=gpu_label,
                                    variable=self.use_gpu_var, value=True,
                                    command=self.update_device,
                                    state=gpu_state)
        gpu_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        cpu_radio = ttk.Radiobutton(device_frame, text="CPU", variable=self.use_gpu_var, value=False,
                                    command=self.update_device)
        cpu_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.fp16_toggle = ttk.Checkbutton(device_frame, text="Use Half-Precision (FP16)",
                                           variable=self.use_fp16_var,
                                           command=self.update_precision)
        self.fp16_toggle.grid(row=0, column=2, padx=20, pady=5, sticky="w")

    def update_precision(self):
        """Handles the precision switching logic based on user selection."""
        use_fp16 = self.use_fp16_var.get()
        try:
            if self.device_manager.use_gpu:
                success = self.color_processor.switch_device(use_gpu=True, use_fp16=use_fp16)
                if success:
                    precision = "FP16" if use_fp16 else "FP32"
                    self.status_label.config(text=f"Precision switched to {precision}", foreground="#CCCCCC")
            else:
                raise RuntimeError("Precision selection is only available when GPU is active.")
        except Exception as e:
            error_msg = f"Precision switching failed: {str(e)}"
            self.status_label.config(text=error_msg, foreground="red")
            logging.error(error_msg)
            self.use_fp16_var.set(False)

    def update_device(self):
        """Handles the device switching logic based on user selection."""
        use_gpu = self.use_gpu_var.get()
        try:
            if use_gpu and torch.cuda.is_available():
                use_fp16 = self.use_fp16_var.get()
            else:
                use_fp16 = False

            success = self.color_processor.switch_device(use_gpu, use_fp16=use_fp16)
            if success:
                device_name = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"
                precision = "FP16" if use_fp16 else "FP32"
                self.status_label.config(text=f"Switched to {device_name} with {precision} precision",
                                         foreground="#CCCCCC")
            else:
                device_name = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"
                precision = "FP16" if use_fp16 else "FP32"
                self.status_label.config(text=f"Already on {device_name} with {precision} precision",
                                         foreground="#CCCCCC")

            if use_gpu and torch.cuda.is_available():
                self.enhance_checkbox.config(state='normal')
                self.edge_scale.config(state='normal')
                self.fp16_toggle.config(state='normal')
            else:
                self.enhance_checkbox.config(state='normal')
                self.edge_scale.config(state='normal')
                self._update_enhancement_controls()
                self.fp16_toggle.config(state='disabled')
                self.use_fp16_var.set(False)

        except Exception as e:
            error_msg = f"Device switching failed: {str(e)}"
            self.status_label.config(text=error_msg, foreground="red")
            logging.error(error_msg)
            self.use_gpu_var.set(not use_gpu)
            if not use_gpu:
                self.use_fp16_var.set(False)

    def _setup_parameters(self, parent_frame):
        """Sets up parameter input fields for gamma and exposure."""
        self.params_frame = ttk.LabelFrame(parent_frame, text="Conversion Parameters", padding=(10, 10))
        self.params_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(self.params_frame, text="Tone Map:").grid(row=0, column=0, sticky="e", padx=(0, 5), pady=5)
        self.tonemap_label = ttk.Label(self.params_frame, text="Auto-detect", foreground="#00AA00")
        self.tonemap_label.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="w")

        ttk.Label(self.params_frame, text="Gamma:").grid(row=1, column=0, sticky="e", padx=(0, 5), pady=5)
        self.pregamma_var = tk.StringVar(value=DEFAULT_PREGAMMA)
        pregamma_entry = ttk.Entry(self.params_frame, textvariable=self.pregamma_var, width=15)
        pregamma_entry.grid(row=1, column=1, padx=(0, 5), pady=5)

        ttk.Label(self.params_frame, text="Exposure:").grid(row=2, column=0, sticky="e", padx=(0, 5), pady=5)
        self.autoexposure_var = tk.StringVar(value=DEFAULT_AUTOEXPOSURE)
        autoexposure_entry = ttk.Entry(self.params_frame, textvariable=self.autoexposure_var, width=15)
        autoexposure_entry.grid(row=2, column=1, padx=(0, 5), pady=5)

    def _setup_color_controls(self, parent_frame):
        """Image enhancement: AI Color grade (3D LUT) and Sharpness — independent."""
        enhance_frame = ttk.LabelFrame(parent_frame, text="Image Enhancement", padding=(12, 10))
        enhance_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.use_enhancement = tk.BooleanVar(value=True)

        self.enhance_checkbox = ttk.Checkbutton(enhance_frame, text="Enable enhancements",
                                                variable=self.use_enhancement,
                                                command=self._update_enhancement_controls)
        self.enhance_checkbox.grid(row=0, column=0, columnspan=3, pady=(0, 8), sticky="w")

        enhance_frame.grid_columnconfigure(0, minsize=130)
        enhance_frame.grid_columnconfigure(1, weight=1)
        enhance_frame.grid_columnconfigure(2, minsize=44)

        # AI Color grade (FiveK 3D LUT). OFF by default: it can tint dark gaming
        # HDR; AgX already nails the look. Opt-in for a photographic color grade.
        ttk.Label(enhance_frame, text="AI Color (LUT):").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(6, 0))
        self.ai_color_strength = tk.DoubleVar(value=0.0)
        self.color_scale = ttk.Scale(enhance_frame, from_=0.0, to=100.0,
                                     variable=self.ai_color_strength, orient="horizontal")
        self.color_scale.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(6, 0))
        self.color_label = ttk.Label(enhance_frame, text="0%", width=4, anchor="e")
        self.color_label.grid(row=1, column=2, sticky="e", pady=(6, 0))

        # Sharpness (Sobel edge enhancement).
        ttk.Label(enhance_frame, text="Sharpness:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(8, 0))
        self.edge_strength = tk.DoubleVar(value=25.0)
        self.edge_scale = ttk.Scale(enhance_frame, from_=0.0, to=100.0,
                                    variable=self.edge_strength, orient="horizontal")
        self.edge_scale.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(8, 0))
        self.edge_label = ttk.Label(enhance_frame, text="25%", width=4, anchor="e")
        self.edge_label.grid(row=2, column=2, sticky="e", pady=(8, 0))

        self.ai_color_strength.trace_add(
            "write", lambda *a: self.color_label.config(text=f"{int(self.ai_color_strength.get())}%"))
        self.edge_strength.trace_add(
            "write", lambda *a: self.edge_label.config(text=f"{int(self.edge_strength.get())}%"))
        self._update_enhancement_controls()

    def _setup_conversion_button(self, parent_frame):
        """Sets up the conversion button."""
        convert_frame = ttk.Frame(parent_frame, padding=(0, 0))
        convert_frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        convert_frame.grid_columnconfigure(0, weight=1)
        self.convert_btn = ttk.Button(convert_frame, text="Convert", command=self.convert_image, style="Accent.TButton")
        self.convert_btn.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        style = ttk.Style()
        try:
            if 'Accent.TButton' not in style.element_names():
                style.configure(
                    'Accent.TButton',
                    font=('Segoe UI', 10),
                    padding=5,
                    foreground='white',
                    background='#007ACC'
                )
                style.map('Accent.TButton',
                          background=[('active', '#005A9E')],
                          foreground=[('active', 'white')])
        except:
            pass

    def _setup_progress_bar(self, parent_frame):
        """Sets up the progress bar."""
        progress_frame = ttk.Frame(parent_frame, padding=(0, 0))
        progress_frame.grid(row=6, column=0, sticky="ew", pady=(5, 10))
        progress_frame.grid_columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")

    def _setup_status_label(self, parent_frame):
        """Sets up the status label to display messages."""
        self.status_label = ttk.Label(parent_frame, text="", foreground="#CCCCCC")
        self.status_label.grid(row=7, column=0, sticky="w", pady=(0, 10))

    def _setup_previews(self, parent_frame):
        """Sets up the preview image canvases."""
        previews_frame = ttk.Frame(parent_frame, padding=(10, 10))
        previews_frame.grid(row=0, column=0, sticky="nsew")
        previews_frame.grid_columnconfigure(0, weight=1)
        previews_frame.grid_columnconfigure(1, weight=1)
        previews_frame.grid_rowconfigure(0, weight=1)

        before_frame = ttk.LabelFrame(previews_frame, text="Before Conversion", padding=(10, 10))
        before_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.before_canvas = tk.Canvas(before_frame, width=self.preview_width, height=self.preview_height)
        self.before_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.before_label = ttk.Label(self.before_canvas)
        self.before_canvas.create_window(self.preview_width // 2, self.preview_height // 2, window=self.before_label)

        after_frame = ttk.LabelFrame(previews_frame, text="After Conversion", padding=(10, 10))
        after_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.after_canvas = tk.Canvas(after_frame, width=self.preview_width, height=self.preview_height)
        self.after_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.after_label = ttk.Label(self.after_canvas)
        self.after_canvas.create_window(self.preview_width // 2, self.preview_height // 2, window=self.after_label)

    def _setup_histograms(self, parent_frame):
        """Sets up the histogram canvases for before and after images."""
        histograms_frame = ttk.Frame(parent_frame, padding=(10, 10))
        histograms_frame.grid(row=1, column=0, sticky="nsew")

        before_hist_frame = ttk.LabelFrame(histograms_frame, text="Before Conversion Histogram", padding=(10, 10))
        before_hist_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.before_hist_canvas = tk.Canvas(before_hist_frame, width=self.preview_width, height=150)
        self.before_hist_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.before_hist_label = ttk.Label(self.before_hist_canvas)
        self.before_hist_canvas.create_window(self.preview_width // 2, 75, window=self.before_hist_label)

        after_hist_frame = ttk.LabelFrame(histograms_frame, text="After Conversion Histogram", padding=(10, 10))
        after_hist_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.after_hist_canvas = tk.Canvas(after_hist_frame, width=self.preview_width, height=150)
        self.after_hist_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.after_hist_label = ttk.Label(self.after_hist_canvas)
        self.after_hist_canvas.create_window(self.preview_width // 2, 75, window=self.after_hist_label)

    def update_mode_size(self):
        """Updates the window and preview sizes based on selected mode."""
        mode = self.mode_size_var.get()
        logging.info(f"Switching to mode: {mode}")
        self.current_mode = mode
        self.preview_width = MODES[mode]['preview_width']
        self.preview_height = MODES[mode]['preview_height']
        self.root.minsize(MODES[mode]['window_width'], MODES[mode]['window_height'])
        self.root.geometry(f"{MODES[mode]['window_width']}x{MODES[mode]['window_height']}")

        self.adjust_preview_canvases()
        self.adjust_histogram_canvases()

        if self.original_input_tensor is not None:
            self.before_hist_label.config(image="", text="")
            self.before_hist_ref = None

            # Recreate preview from original linear tensor
            tone_mapper = AdvancedToneMapper(self.device_manager.get_device(), self.jxr_loader.metadata)
            tonemapped_tensor = tone_mapper.apply_tone_mapping(
                self.original_input_tensor, method=self._method_for(self.original_input_tensor, tone_mapper))
            srgb_tensor = self.jxr_loader.linear_to_srgb(tonemapped_tensor)

            preview_tensor = self.jxr_loader.process_preview(
                srgb_tensor,
                self.preview_width,
                self.preview_height
            )
            self._update_preview_ui(preview_tensor, srgb_tensor)
            self.show_color_spectrum_from_tensor(srgb_tensor, is_before=True)

        # Regenerate after image from tensor if available, otherwise use file
        if getattr(self, 'current_enhanced_tensor', None) is not None:
            self.show_preview_from_tensor(self.current_enhanced_tensor, is_before=False)
            self.show_color_spectrum_from_tensor(self.current_enhanced_tensor, is_before=False)
        elif getattr(self, 'current_after_image_path', None):
            self.show_preview_from_file(self.current_after_image_path, is_before=False)
            self.show_color_spectrum(self.current_after_image_path, is_before=False)

        self.center_window(MODES[mode]['window_width'], MODES[mode]['window_height'])
        self.status_label.config(text=f"Switched to {mode} mode", foreground="#CCCCCC")

    def adjust_preview_canvases(self):
        """Adjusts the size of preview canvases based on the current mode."""
        self.before_canvas.config(width=self.preview_width, height=self.preview_height)
        self.before_canvas.delete("all")
        self.before_canvas.create_window(self.preview_width // 2, self.preview_height // 2, window=self.before_label)

        self.after_canvas.config(width=self.preview_width, height=self.preview_height)
        self.after_canvas.delete("all")
        self.after_canvas.create_window(self.preview_width // 2, self.preview_height // 2, window=self.after_label)

    def adjust_histogram_canvases(self):
        """Adjusts the size of histogram canvases based on the current mode."""
        self.before_hist_canvas.config(width=self.preview_width, height=150)
        self.before_hist_canvas.delete("all")
        self.before_hist_canvas.create_window(self.preview_width // 2, 75, window=self.before_hist_label)

        self.after_hist_canvas.config(width=self.preview_width, height=150)
        self.after_hist_canvas.delete("all")
        self.after_hist_canvas.create_window(self.preview_width // 2, 75, window=self.after_hist_label)

    def center_window(self, width, height):
        """Centers the application window on the screen."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def browse_input(self):
        """Handles the input file or folder browsing."""
        mode = self.mode_var.get()
        if mode == "single":
            filename = filedialog.askopenfilename(
                title="Select Input JXR File",
                filetypes=[("JXR files", "*.jxr"), ("All files", "*.*")]
            )
            if filename:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, filename)

                base_name, _ = os.path.splitext(filename)
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, base_name)

                self.status_label.config(text="Loading preview...", foreground="#CCCCCC")
                self.master.update_idletasks()
                self.create_preview_from_jxr(filename)
        else:
            foldername = filedialog.askdirectory(title="Select Folder Containing JXR Files")
            if foldername:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, foldername)
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, os.path.join(foldername, "Converted_Outputs"))
                jxr_files = [f for f in os.listdir(foldername) if f.lower().endswith('.jxr')]
                file_count = len(jxr_files)
                if file_count == 0:
                    self.status_label.config(text="No JXR files found in selected folder.", foreground="red")
                    self.convert_btn.config(state='disabled')
                else:
                    self.status_label.config(
                        text=f"Found {file_count} JXR files ready for conversion.",
                        foreground="#00FF00"
                    )
                    self.convert_btn.config(state='normal')

                self.before_label.config(image="", text="No Preview")
                self.before_image_ref = None
                self.after_label.config(image="", text="No Preview")
                self.after_image_ref = None
                self.before_hist_label.config(image="", text="No Histogram")
                self.before_hist_ref = None
                self.after_hist_label.config(image="", text="No Histogram")
                self.after_hist_ref = None

    def browse_output(self):
        """Handles the output file browsing."""
        mode = self.mode_var.get()
        if mode == "single":
            filename = filedialog.asksaveasfilename(
                title="Select Output File Base Name",
                filetypes=[("All files", "*.*")]
            )
            if filename:
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, filename)
        else:
            foldername = filedialog.askdirectory(title="Select Output Folder")
            if foldername:
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, foldername)

    def create_preview_from_jxr(self, jxr_file: str):
        """Generates a preview from a selected JXR file."""
        if not jxr_file or not jxr_file.lower().endswith('.jxr'):
            self._clear_preview()
            self.status_label.config(text="Invalid JXR file", foreground="red")
            return
        if not os.path.exists(jxr_file):
            self._clear_preview()
            self.status_label.config(text="File not found", foreground="red")
            return

        self.status_label.config(text="Loading preview...", foreground="#CCCCCC")
        self.master.update_idletasks()

        def process_jxr():
            try:
                linear_tensor, error, metadata = self.jxr_loader.load_jxr(jxr_file)
                if linear_tensor is None:
                    raise RuntimeError("Failed to load JXR file.")
                self.original_input_tensor = linear_tensor.clone()

                if metadata.get('has_metadata', False):
                    logging.info(f"HDR Metadata: Max Luminance={metadata.get('max_luminance')}nits, "
                                 f"White Point={metadata.get('white_point')}nits")

                # Apply tone mapping and sRGB for preview
                tone_mapper = AdvancedToneMapper(self.device_manager.get_device(), self.jxr_loader.metadata)
                tonemapped_tensor = tone_mapper.apply_tone_mapping(
                    linear_tensor, method=self._method_for(linear_tensor, tone_mapper))
                srgb_tensor = self.jxr_loader.linear_to_srgb(tonemapped_tensor)

                preview_tensor = self.jxr_loader.process_preview(
                    srgb_tensor,
                    self.preview_width,
                    self.preview_height
                )
                if preview_tensor is None:
                    raise RuntimeError("Failed to generate preview")

                self.master.after(0, lambda: self._update_preview_ui(preview_tensor, srgb_tensor))
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Preview generation error: {error_msg}", exc_info=True)
                self.master.after(0, lambda: self._handle_preview_error(error_msg))

        thread = threading.Thread(target=process_jxr, daemon=True)
        thread.start()

    def _handle_preview_error(self, error_msg: str):
        """Handles errors during preview generation."""
        self.status_label.config(text=f"Preview failed: {error_msg}", foreground="red")
        self.before_label.config(image="", text="Preview Failed")
        self.before_image_ref = None
        self.before_hist_label.config(image="", text="No Histogram")
        self.before_hist_ref = None

    def _clear_preview(self):
        """Clears the preview and histogram displays."""
        self.before_label.config(image="", text="No Preview")
        self.before_image_ref = None
        self.before_hist_label.config(image="", text="No Histogram")
        self.before_hist_ref = None

    def show_preview_from_file(self, filepath: str, is_before: bool):
        """Displays a preview image from a file."""
        label = self.before_label if is_before else self.after_label
        if is_before:
            if self.before_image_ref:
                self.before_label.config(image="")
                self.before_image_ref = None
            self.current_before_image_path = filepath
        else:
            if self.after_image_ref:
                self.after_label.config(image="")
                self.after_image_ref = None
            self.current_after_image_path = filepath

        try:
            img = Image.open(filepath).convert('RGB')
            logging.info(f"Loading preview from {filepath}, is_before={is_before}, original size: {img.size}")
            img = self._resize_image(img, self.preview_width, self.preview_height)
            logging.info(f"After resize: {img.size}, target: {self.preview_width}x{self.preview_height}")
            
            # Apply the same centering logic as _update_preview_ui for consistency
            if img.size[0] < self.preview_width or img.size[1] < self.preview_height:
                bg = Image.new('RGB', (self.preview_width, self.preview_height), (46, 46, 46))
                offset_x = (self.preview_width - img.size[0]) // 2
                offset_y = (self.preview_height - img.size[1]) // 2
                bg.paste(img, (offset_x, offset_y))
                img = bg
                logging.info(f"Centered on background: {img.size}")
            
            img_tk = ImageTk.PhotoImage(img)
            label.config(image=img_tk, text="")
            if is_before:
                self.before_image_ref = img_tk
            else:
                self.after_image_ref = img_tk
            img.close()
        except Exception as e:
            label.config(image="", text=f"Preview Error: {str(e)}")
            if is_before:
                self.before_image_ref = None
            else:
                self.after_image_ref = None
            logging.error(f"Error loading preview: {str(e)}")

    def show_preview_from_tensor(self, tensor: torch.Tensor, is_before: bool):
        """Displays a preview image from a tensor."""
        label = self.before_label if is_before else self.after_label
        image_ref_attr = 'before_image_ref' if is_before else 'after_image_ref'

        if getattr(self, image_ref_attr):
            label.config(image="")
            setattr(self, image_ref_attr, None)

        try:
            # Generate a preview-sized tensor
            preview_tensor = self.jxr_loader.process_preview(
                tensor,
                self.preview_width,
                self.preview_height
            )
            if preview_tensor is None:
                raise RuntimeError("Failed to generate preview tensor")

            img = self.jxr_loader.tensor_to_pil(preview_tensor)
            if img is None:
                raise RuntimeError("Failed to convert tensor to PIL image")

            # Center the image in the canvas if it's smaller
            orig_width, orig_height = img.size
            if orig_width < self.preview_width or orig_height < self.preview_height:
                bg = Image.new('RGB', (self.preview_width, self.preview_height), (46, 46, 46))
                offset_x = (self.preview_width - orig_width) // 2
                offset_y = (self.preview_height - orig_height) // 2
                bg.paste(img, (offset_x, offset_y))
                img = bg

            img_tk = ImageTk.PhotoImage(img)
            label.config(image=img_tk, text="")
            setattr(self, image_ref_attr, img_tk)

        except Exception as e:
            label.config(image="", text=f"Preview Error: {str(e)}")
            setattr(self, image_ref_attr, None)
            logging.error(f"Error loading preview from tensor: {str(e)}")

    def show_color_spectrum(self, filepath: str, is_before: bool):
        """Generates and displays a color spectrum histogram from an image file."""
        label = self.before_hist_label if is_before else self.after_hist_label
        if is_before and self.before_hist_ref:
            self.before_hist_label.config(image="", text="")
            self.before_hist_ref = None
        elif not is_before and self.after_hist_ref:
            self.after_hist_label.config(image="", text="")
            self.after_hist_ref = None

        try:
            img = Image.open(filepath).convert('RGB')
            vis_img = self._create_histogram_image(np.array(img))
            vis_tk = ImageTk.PhotoImage(vis_img)

            def update_visualization():
                label.config(image=vis_tk, text="")
                if is_before:
                    self.before_hist_ref = vis_tk
                else:
                    self.after_hist_ref = vis_tk

            self.master.after(0, update_visualization)
        except Exception as e:
            label.config(image="", text=f"Visualization Error: {str(e)}")
            if is_before:
                self.before_hist_ref = None
            else:
                self.after_hist_ref = None
            logging.error(f"Error generating color spectrum: {str(e)}")

    def show_color_spectrum_from_tensor(self, tensor: torch.Tensor, is_before: bool):
        """Generates and displays a color spectrum histogram from a tensor."""
        label = self.before_hist_label if is_before else self.after_hist_label
        if is_before and self.before_hist_ref:
            self.before_hist_label.config(image="", text="")
            self.before_hist_ref = None
        elif not is_before and self.after_hist_ref:
            self.after_hist_label.config(image="", text="")
            self.after_hist_ref = None

        try:
            img_tensor = tensor.squeeze(0)
            if img_tensor.shape[0] != 3:
                img_tensor = img_tensor[:3, :, :]
            img_data = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            vis_img = self._create_histogram_image(img_data)
            vis_tk = ImageTk.PhotoImage(vis_img)

            def update_visualization():
                label.config(image=vis_tk, text="")
                if is_before:
                    self.before_hist_ref = vis_tk
                else:
                    self.after_hist_ref = vis_tk

            self.master.after(0, update_visualization)
        except Exception as e:
            label.config(image="", text=f"Visualization Error: {str(e)}")
            if is_before:
                self.before_hist_ref = None
            else:
                self.after_hist_ref = None
            logging.error(f"Error generating color spectrum from tensor: {str(e)}")

    def _create_histogram_image(self, img_array: np.ndarray) -> Image.Image:
        """Creates a histogram visualization from an image array."""
        vis_img = Image.new('RGB', (self.preview_width, 150), '#1e1e1e')
        draw = ImageDraw.Draw(vis_img, 'RGBA')
        draw.rectangle([0, 0, self.preview_width, 150], fill='#2b2b2b')
        grid_spacing = 150 // 4
        for i in range(5):
            y = i * grid_spacing
            draw.line([(0, y), (self.preview_width, y)], fill='#40404040', width=1)
        channels = [
            (img_array[:, :, 0], '#ff000066'),
            (img_array[:, :, 1], '#00ff0066'),
            (img_array[:, :, 2], '#0000ff66')
        ]
        for channel_data, color in channels:
            hist, _ = np.histogram(channel_data, bins=self.preview_width, range=(0, 255))
            if hist.max() > 0:
                hist = hist / hist.max() * (150 - 10)
            points = [(0, 150)]
            for x in range(self.preview_width):
                y = 150 - hist[x]
                points.append((x, y))
            points.append((self.preview_width, 150))
            draw.polygon(points, fill=color)
        return vis_img

    def _resize_image(self, image: Image.Image, max_width: int, max_height: int) -> Image.Image:
        """Resizes an image while maintaining aspect ratio."""
        original_width, original_height = image.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _update_preview_ui(self, preview_tensor: torch.Tensor, full_res_srgb_tensor: torch.Tensor):
        """Updates the UI with the generated preview image."""
        try:
            preview_image = self.jxr_loader.tensor_to_pil(preview_tensor)
            if preview_image is None:
                raise RuntimeError("Failed to convert preview to image")

            orig_width, orig_height = preview_image.size
            width_ratio = self.preview_width / orig_width
            height_ratio = self.preview_height / orig_height
            scale_factor = min(width_ratio, height_ratio)
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)

            preview_image = preview_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if new_width < self.preview_width or new_height < self.preview_height:
                bg = Image.new('RGB', (self.preview_width, self.preview_height), (46, 46, 46))
                offset_x = (self.preview_width - new_width) // 2
                offset_y = (self.preview_height - new_height) // 2
                bg.paste(preview_image, (offset_x, offset_y))
                preview_image = bg

            buffer = io.BytesIO()
            preview_image.save(buffer, format='PNG')
            buffer.seek(0)
            preview_tk = ImageTk.PhotoImage(Image.open(buffer))
            self.before_label.config(image=preview_tk, text="")
            self.before_image_ref = preview_tk
            self.show_color_spectrum_from_tensor(full_res_srgb_tensor, is_before=True)
            self.status_label.config(text="Preview loaded successfully", foreground="#00FF00")

        except Exception as e:
            self._handle_preview_error(str(e))

    def _update_enhancement_controls(self):
        """Enables or disables enhancement controls based on user selection."""
        state = 'normal' if self.use_enhancement.get() else 'disabled'
        self.edge_scale.configure(state=state)
        if hasattr(self, 'color_scale'):
            self.color_scale.configure(state=state)

    def convert_image(self):
        """Initiates the image conversion process based on selected mode."""
        mode = self.mode_var.get()
        if mode == "single":
            self._convert_single_file()
        else:
            self._convert_folder()

    def _convert_single_file(self):
        """Converts a single JXR file."""
        input_file = self.input_entry.get().strip()
        output_base_file = self.output_entry.get().strip()
        pre_gamma_str = self.pregamma_var.get().strip()
        auto_exposure_str = self.autoexposure_var.get().strip()
        use_enhancement = self.use_enhancement.get()
        color_strength = self.ai_color_strength.get() if use_enhancement else 0.0
        edge_strength = self.edge_strength.get() if use_enhancement else 0.0

        try:
            validate_files(input_file, output_base_file)
            validate_parameters(pre_gamma_str, auto_exposure_str)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text=str(e), foreground="red")
            return

        self.jxr_loader.selected_pre_gamma = float(pre_gamma_str)
        self.jxr_loader.selected_auto_exposure = float(auto_exposure_str)

        self.status_label.config(text="Analyzing image...", foreground="#CCCCCC")
        self.progress['value'] = 0
        self.master.update_idletasks()
        self.convert_btn.config(state='disabled')

        def process_task():
            try:
                linear_tensor, _, metadata = self.jxr_loader.load_jxr(input_file)
                if linear_tensor is None:
                    raise RuntimeError("Failed to load JXR file.")

                tone_mapper = AdvancedToneMapper(self.device_manager.get_device(), self.jxr_loader.metadata)
                
                # Get user-selected tone mapping method or auto-detect
                selected_method = self.get_selected_tone_mapping_method()
                if selected_method is None:  # Auto-detect (intelligent oracle)
                    auto_method = tone_mapper.select_optimal_tonemap(tone_mapper.analyze_image_statistics(linear_tensor))
                    final_method = auto_method
                    method_display = f"{auto_method} (auto)"
                else:  # Manual selection
                    final_method = selected_method
                    method_display = f"{selected_method} (manual)"
                
                self.master.after(0, lambda: self.tonemap_label.config(text=method_display, foreground="#00FF00"))
                self.master.after(0, lambda: self.status_label.config(
                    text=f"Converting with {final_method} tone mapping...", foreground="#CCCCCC"))

                tonemapped_tensor = tone_mapper.apply_tone_mapping(linear_tensor.clone(), method=final_method)
                srgb_tonemapped_tensor = self.jxr_loader.linear_to_srgb(tonemapped_tensor)

                enhanced_tensor = self.color_processor.process_image(
                    srgb_tonemapped_tensor.clone(),
                    color_strength=color_strength,
                    edge_strength=edge_strength,
                    use_enhancement=use_enhancement
                )

                output_format = self.output_format_var.get()

                if output_format in ("JPEG", "Both"):
                    self.color_processor.save_tensor_as_image(enhanced_tensor, f"{output_base_file}.jpg", "JPEG")

                if output_format in ("TIFF", "Both"):
                    self.color_processor.save_tensor_as_image(linear_tensor, f"{output_base_file}_raw.tiff", "TIFF",
                                                              is_hdr_data=True)
                    self.color_processor.save_tensor_as_image(enhanced_tensor, f"{output_base_file}_tonemapped.tiff",
                                                              "TIFF", is_hdr_data=False)

                # Store the processed tensor for mode switching
                self.current_enhanced_tensor = enhanced_tensor.clone()
                
                # Generate "After" preview directly from the processed tensor
                self.master.after(0, lambda: self.show_preview_from_tensor(enhanced_tensor.clone(), is_before=False))
                self.master.after(0, lambda: self.show_color_spectrum_from_tensor(enhanced_tensor.clone(),
                                                                                  is_before=False))

                self.safe_update_ui("Conversion successful!", "#00FF00")

            except Exception as e:
                error_msg = str(e)
                logging.error(f"Conversion failed with error: {error_msg}", exc_info=True)
                self.safe_update_ui(f"Conversion failed: {error_msg}", "red")
                messagebox.showerror("Error", error_msg)
            finally:
                self.safe_enable_convert_button()

        threading.Thread(target=process_task, daemon=True).start()

    def _convert_folder(self):
        """Converts all JXR files in a selected folder."""
        folder_path = self.input_entry.get().strip()
        output_folder = self.output_entry.get().strip()

        if not folder_path or not os.path.isdir(folder_path):
            error_msg = "Please select a valid input folder."
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg, foreground="red")
            return

        if not output_folder:
            error_msg = "Please select a valid output folder."
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg, foreground="red")
            return

        jxr_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jxr')]
        if not jxr_files:
            error_msg = "No JXR files found in the selected folder."
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg, foreground="red")
            return

        os.makedirs(output_folder, exist_ok=True)

        pre_gamma_str = self.pregamma_var.get().strip()
        auto_exposure_str = self.autoexposure_var.get().strip()

        try:
            validate_parameters(pre_gamma_str, auto_exposure_str)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text=str(e), foreground="red")
            return

        self.jxr_loader.selected_pre_gamma = float(pre_gamma_str)
        self.jxr_loader.selected_auto_exposure = float(auto_exposure_str)

        use_enhancement = self.use_enhancement.get()
        color_strength = self.ai_color_strength.get() if use_enhancement else 0.0
        edge_strength = self.edge_strength.get() if use_enhancement else 0.0
        output_format_choice = self.output_format_var.get()

        self.status_label.config(text=f"Converting {len(jxr_files)} files...", foreground="#CCCCCC")
        self.progress['maximum'] = len(jxr_files)
        self.progress['value'] = 0
        self.convert_btn.config(state='disabled')

        def process_files():
            successful_conversions = 0
            failed_conversions = 0
            failed_files = []
            try:
                for jxr_file in jxr_files:
                    try:
                        input_path = os.path.join(folder_path, jxr_file)
                        output_base_name = os.path.join(output_folder, os.path.splitext(jxr_file)[0])
                        self.safe_update_ui(f"Processing: {jxr_file}", "#CCCCCC")
                        logging.info(f"Processing file: {jxr_file}")

                        linear_tensor, _, metadata = self.jxr_loader.load_jxr(input_path)
                        if linear_tensor is None:
                            raise RuntimeError(f"Failed to load {jxr_file}")

                        tone_mapper = AdvancedToneMapper(self.device_manager.get_device(), self.jxr_loader.metadata)
                        
                        # Get user-selected tone mapping method or auto-detect
                        selected_method = self.get_selected_tone_mapping_method()
                        if selected_method is None:  # Auto-detect (intelligent oracle)
                            auto_method = tone_mapper.select_optimal_tonemap(
                                tone_mapper.analyze_image_statistics(linear_tensor))
                            final_method = auto_method
                        else:  # Manual selection
                            final_method = selected_method

                        tonemapped_tensor = tone_mapper.apply_tone_mapping(linear_tensor.clone(), method=final_method)
                        srgb_tonemapped_tensor = self.jxr_loader.linear_to_srgb(tonemapped_tensor)

                        enhanced_tensor = self.color_processor.process_image(
                            srgb_tonemapped_tensor.clone(),
                            color_strength=color_strength,
                            edge_strength=edge_strength,
                            use_enhancement=use_enhancement
                        )

                        if output_format_choice in ("JPEG", "Both"):
                            self.color_processor.save_tensor_as_image(enhanced_tensor, f"{output_base_name}.jpg",
                                                                      "JPEG")

                        if output_format_choice in ("TIFF", "Both"):
                            self.color_processor.save_tensor_as_image(linear_tensor, f"{output_base_name}_raw.tiff",
                                                                      "TIFF", is_hdr_data=True)
                            self.color_processor.save_tensor_as_image(enhanced_tensor,
                                                                      f"{output_base_name}_tonemapped.tiff", "TIFF", is_hdr_data=False)

                        successful_conversions += 1
                    except Exception as e:
                        logging.error(f"Failed to process {jxr_file}: {e}", exc_info=True)
                        failed_conversions += 1
                        failed_files.append(jxr_file)
                    finally:
                        self.safe_increment_progress()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if failed_conversions == 0:
                    self.safe_update_ui(
                        f"Batch conversion completed! All {successful_conversions} files processed.",
                        "#00FF00"
                    )
                else:
                    error_msg = (
                        f"Batch completed with errors. Success: {successful_conversions}, Failed: {failed_conversions}\n"
                        f"Failed files: {', '.join(failed_files)}")
                    self.safe_update_ui(error_msg, "orange" if successful_conversions > 0 else "red")
                    logging.error(error_msg)
            except Exception as e:
                logging.exception("Batch processing error.")
                self.safe_update_ui(f"Batch processing error: {str(e)}", "red")
            finally:
                self.safe_enable_convert_button()

        threading.Thread(target=process_files, daemon=True).start()

    def safe_update_ui(self, message, color):
        """Safely updates the UI status label from any thread."""
        with self.ui_lock:
            self.status_label.config(text=message, foreground=color)

    def safe_increment_progress(self):
        """Safely increments the progress bar from any thread."""
        with self.ui_lock:
            self.progress['value'] += 1
            self.master.update_idletasks()

    def safe_enable_convert_button(self):
        """Safely enables the convert button from any thread."""
        with self.ui_lock:
            self.convert_btn.config(state='normal')


def _cli_convert(argv) -> int:
    """Headless one-shot conversion (bypasses the GUI).

        NHC_HDR_Converter.exe --selftest INPUT.jxr [--out BASE] [--method agx_punchy]

    Used as the bundled-exe self-test and as a minimal CLI. Returns an exit code.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="NHC", add_help=True)
    parser.add_argument("--selftest", "--convert", dest="input", metavar="INPUT.jxr", required=True)
    parser.add_argument("--out", dest="out", default=None, help="output base path (no extension)")
    parser.add_argument("--method", dest="method", default=None,
                        help="tone map (default: Auto-detect oracle)")
    parser.add_argument("--color", type=float, default=0.0, help="AI Color (3D LUT) strength 0-100")
    parser.add_argument("--edge", type=float, default=25.0, help="Sharpness 0-100")
    parser.add_argument("--format", dest="fmt", default="JPEG", choices=["JPEG", "TIFF", "Both"])
    args = parser.parse_args(argv)

    dev = DeviceManager().get_device()
    loader = OptimizedJXRLoader(dev)
    lin, err, _ = loader.load_jxr(args.input)
    if lin is None:
        print(f"[selftest] FAILED to load {args.input}: {err}")
        return 2
    tm = AdvancedToneMapper(dev, loader.metadata)
    proc = HDRColorProcessor(dev, loader, use_fp16=False)
    tonemapped = tm.apply_tone_mapping(lin.clone(), method=args.method)  # None -> Auto-detect oracle
    method = args.method or "auto-detect"
    srgb = loader.linear_to_srgb(tonemapped)
    enhanced = proc.process_image(srgb.clone(), color_strength=args.color,
                                  edge_strength=args.edge, use_enhancement=True)
    out_base = args.out or os.path.splitext(args.input)[0]
    if args.fmt in ("JPEG", "Both"):
        proc.save_tensor_as_image(enhanced, out_base + ".jpg", "JPEG")
    if args.fmt in ("TIFF", "Both"):
        proc.save_tensor_as_image(enhanced, out_base + "_tonemapped.tiff", "TIFF", is_hdr_data=False)
    print(f"[selftest] OK  device={dev.type}  method={method}  -> {out_base}.jpg")
    return 0


if __name__ == "__main__":
    # Headless CLI / self-test path (no GUI). Lets the bundled .exe verify itself:
    #   NHC_HDR_Converter.exe --selftest sample.jxr
    if len(sys.argv) > 1 and ("--selftest" in sys.argv or "--convert" in sys.argv):
        sys.exit(_cli_convert(sys.argv[1:]))
    try:
        app = App("park", "dark")
        root = app.master
        root.title("HDR Image Converter - Enhanced Edition")
        root.minsize(1760, 976)
        window_width = 1760
        window_height = 976
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        logging.info("Starting main event loop...")
        root.mainloop()
    except Exception as e:
        logging.exception("Error starting application.")
        messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{str(e)}")
        sys.exit(1)

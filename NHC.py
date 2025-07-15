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
import torchvision.models as models
import logging
import matplotlib
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

# Use 'Agg' backend for matplotlib to prevent GUI issues
matplotlib.use('Agg')

# Constants
DEFAULT_TONE_MAP = "perceptual"
DEFAULT_PREGAMMA = "1.0"
DEFAULT_AUTOEXPOSURE = "1.0"
SUPPORTED_TONE_MAPS = {"perceptual", "adaptive", "hable", "reinhard", "filmic", "aces", "uncharted2", "mantiuk06",
                       "drago03"}

PRETRAINED_MODELS = {
    'vgg': models.vgg16,
    'resnet': models.resnet34,
    'densenet': models.densenet121
}

MODES = {
    'big': {
        'window_width': 2200,
        'window_height': 900,
        'preview_width': 720,
        'preview_height': 406,
    },
    'small': {
        'window_width': 1760,
        'window_height': 900,
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

    def _build_luts(self):
        """Build lookup tables for various tone mapping curves."""
        lut_size = 4096
        x = torch.linspace(0, 20, lut_size, device=self.device)

        # Build LUTs for different operators
        self.hable_lut = self._hable_curve(x)
        self.aces_lut = self._aces_curve(x)
        self.reinhard_lut = self._reinhard_curve(x, L_white=4.0)

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

    def apply_tone_mapping(self, image: torch.Tensor, method: Optional[str] = None) -> torch.Tensor:
        """Apply tone mapping with automatic method selection if not specified."""
        # Analyze image if method not specified
        if method is None:
            stats = self.analyze_image_statistics(image)
            method = self.select_optimal_tonemap(stats)
            logging.info(f"Auto-selected tone mapping method: {method}")

        # Store original for color preservation
        original = image.clone()

        # Apply selected tone mapping
        if method == 'perceptual':
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

        # Avoid clipping with soft compression
        tonemapped = self.soft_clip(tonemapped)

        return tonemapped

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

    def soft_clip(self, image: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
        """Soft clipping to avoid hard cutoffs at 0 and 1."""
        # Highlights
        over_mask = image > threshold
        if over_mask.any():
            over_values = image[over_mask]
            compressed = threshold + (1 - threshold) * torch.tanh((over_values - threshold) / (1 - threshold))
            image[over_mask] = compressed

        # Shadows
        under_mask = image < (1 - threshold)
        if under_mask.any():
            under_values = image[under_mask]
            compressed = (1 - threshold) * torch.tanh(under_values / (1 - threshold))
            image[under_mask] = compressed

        return image


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


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Provides channel and spatial attention to enhance relevant features.
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        return x


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


class ColorBalanceBlock(nn.Module):
    """
    Balances colors across different luminance ranges (shadows, midtones, highlights).
    Also applies optional color temperature adjustments and channel correlations.
    """

    def __init__(self, channels, color_preservation=0.5):
        super(ColorBalanceBlock, self).__init__()
        self.color_preservation = color_preservation
        self.shadows_param = nn.Parameter(torch.zeros(channels))
        self.midtones_param = nn.Parameter(torch.zeros(channels))
        self.highlights_param = nn.Parameter(torch.zeros(channels))
        self.color_temp = nn.Parameter(torch.tensor([0.0]))
        self.channel_corr = nn.Linear(channels, channels, bias=False)
        self.shadow_threshold = nn.Parameter(torch.tensor([0.2]))
        self.highlight_threshold = nn.Parameter(torch.tensor([0.8]))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_transform = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        luminance = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        shadows_mask = (luminance < self.shadow_threshold).float()
        highlights_mask = (luminance > self.highlight_threshold).float()
        midtones_mask = 1.0 - shadows_mask - highlights_mask

        shadows_adjust = self.shadows_param.view(1, C, 1, 1)
        midtones_adjust = self.midtones_param.view(1, C, 1, 1)
        highlights_adjust = self.highlights_param.view(1, C, 1, 1)
        adjust_map = shadows_mask * shadows_adjust + midtones_mask * midtones_adjust + highlights_mask * highlights_adjust
        x_balanced = x + adjust_map

        # Linear transformation across channels for color correlation
        x_reshaped = x_balanced.view(B, C, -1).transpose(1, 2)
        x_reshaped = x_reshaped.to(self.channel_corr.weight.dtype)

        x_corr = self.channel_corr(x_reshaped).transpose(1, 2).view(B, C, H, W)

        # Apply color temperature offset (simple red/blue shift)
        temp_val = self.color_temp
        x_corr[:, 0, :, :] += temp_val * 0.05
        x_corr[:, 2, :, :] -= temp_val * 0.05

        # Context-based scaling
        context = self.global_pool(x_corr)
        context = self.context_transform(context)
        x_corr = x_corr * context

        # Blend between original and corrected
        x_final = self.color_preservation * x + (1.0 - self.color_preservation) * x_corr
        return x_final


class ColorCorrectionNet(nn.Module):
    """
    Neural network for color correction using pretrained models (VGG16, ResNet34, DenseNet121).
    Combines feature extraction from all three models and fuses them into a final color transform.
    """

    def __init__(self):
        super(ColorCorrectionNet, self).__init__()
        try:
            self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        except Exception as e:
            print(f"Error loading pre-trained models: {e}")
            sys.exit(1)

        # Freeze pretrained models
        for model in [self.vgg, self.resnet, self.densenet]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        # Simplified adaptation layers (reduces dimension to a common size)
        self.vgg_adapt = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.resnet_adapt = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.densenet_adapt = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Fusion and post-processing
        self.fusion = nn.Sequential(
            nn.Conv2d(768, 384, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # CBAM attention and final color transform
        self.cbam = CBAM(192, reduction=8)
        self.color_transform = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 3, 3, padding=1),
            nn.Sigmoid()  # Sigmoid for smoother color transitions
        )

        # Minimal edge enhancement block, if desired
        self.edge_enhance = EdgeEnhancementBlock()

    def extract_features(self, x):
        """
        Extracts lower-level features from VGG, ResNet, and DenseNet.
        We only take the earliest layers for a simpler memory footprint.
        """
        # VGG features
        vgg_feat = self.vgg.features[:5](x)
        vgg_feat = self.vgg_adapt(vgg_feat)

        # ResNet features
        x_res = self.resnet.conv1(x)
        x_res = self.resnet.bn1(x_res)
        x_res = self.resnet.relu(x_res)
        resnet_feat = self.resnet_adapt(x_res)

        # DenseNet features
        densenet_feat = self.densenet.features[:4](x)
        densenet_feat = self.densenet_adapt(densenet_feat)

        return vgg_feat, resnet_feat, densenet_feat

    def forward(self, x):
        """
        Forward pass for color correction and minor enhancement.
        """
        with torch.inference_mode():
            if x.device.type == 'cpu':
                x = x.float()

            # Preserve original range
            x_min = x.amin(dim=[2, 3], keepdim=True)
            x_max = x.amax(dim=[2, 3], keepdim=True)
            x_range = x_max - x_min
            eps = 1e-8

            # Normalize
            x_normalized = (x - x_min) / (x_range + eps)

            # Extract features
            vgg_feat, resnet_feat, densenet_feat = self.extract_features(x_normalized)

            # Ensure consistent spatial dimensions before fusion
            target_size = vgg_feat.shape[-2:]
            resnet_feat = F.interpolate(resnet_feat, size=target_size, mode='bilinear', align_corners=False)
            densenet_feat = F.interpolate(densenet_feat, size=target_size, mode='bilinear', align_corners=False)

            # Fuse features
            fused = self.fusion(torch.cat([vgg_feat, resnet_feat, densenet_feat], dim=1))
            del vgg_feat, resnet_feat, densenet_feat

            # Apply CBAM attention
            fused = self.cbam(fused)

            # Generate color adjustment
            delta = self.color_transform(fused)
            del fused

            # Upsample to match input resolution
            delta = F.interpolate(delta, size=x.shape[-2:], mode='bilinear', align_corners=False)

            # Scale adjustments to ±0.1 range
            delta = (delta - 0.5) * 0.2
            enhanced = x_normalized * (1.0 + delta)

            # Restore original range
            enhanced = enhanced * x_range + x_min
            enhanced = torch.clamp(enhanced, min=x_min, max=x_max)

            # Cleanup
            if x.device.type == 'cuda':
                torch.cuda.empty_cache()

            return enhanced


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
        self.color_net = ColorCorrectionNet().eval().to(device)

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
                if use_enhancement and color_strength > 0:
                    enhanced = self.color_net(padded_tensor)
                    normalized_strength = (color_strength / 100.0) * 1.5
                    enhanced_strength = pow(normalized_strength, 0.7)
                    enhanced = padded_tensor * (1 - enhanced_strength) + enhanced * enhanced_strength
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
        """Saves a tensor as an image file (JPEG or TIFF)."""
        try:
            if tensor_to_save.is_cuda:
                tensor_to_save = tensor_to_save.cpu()

            tensor_to_save = tensor_to_save.detach()

            if output_format == 'JPEG':
                array = tensor_to_save.squeeze(0).permute(1, 2, 0).contiguous().numpy()
                array = np.clip(array, 0.0, 1.0)
                array = (array * 255).astype(np.uint8)
                result_image = Image.fromarray(array)
                result_image.save(output_path, 'JPEG', quality=quality, optimize=True)

            elif output_format == 'TIFF':
                array = tensor_to_save.squeeze(0).permute(1, 2, 0).contiguous().numpy()
                array = np.ascontiguousarray(array, dtype=np.float32)

                if is_hdr_data:
                    min_val, max_val = array.min(), array.max()
                    if max_val > min_val:
                        array = (array - min_val) / (max_val - min_val)
                else:
                    array = np.clip(array, 0.0, 1.0)

                array = (array * 65535).astype(np.uint16)

                # Use tifffile.imwrite for robust TIFF saving.
                tifffile.imwrite(output_path, array, photometric='rgb', compression='lzw')

            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            logging.info(f"Successfully saved image to {output_path}")

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
                for model in [self.color_net.vgg, self.color_net.resnet, self.color_net.densenet]:
                    model.to(device=self.device, dtype=desired_dtype)

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

        self.before_image_ref = None
        self.after_image_ref = None
        self.before_hist_ref = None
        self.after_hist_ref = None

        main_frame = ttk.Frame(self.master, padding=(10, 10))
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame, padding=(10, 10))
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        right_frame = ttk.Frame(main_frame, padding=(10, 10))
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        # Setup various UI components
        self._setup_mode_selection(left_frame)
        self._setup_file_selection(left_frame)
        self._setup_device_selection(left_frame)
        self._setup_parameters(left_frame)
        self._setup_color_controls(left_frame)
        self._setup_conversion_button(left_frame)
        self._setup_progress_bar(left_frame)
        self._setup_status_label(left_frame)

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
        """Sets up image enhancement controls (AI Enhancement and Edge Strength)."""
        enhance_frame = ttk.LabelFrame(parent_frame, text="Image Enhancement", padding=(10, 10))
        enhance_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.use_enhancement = tk.BooleanVar(value=True)

        self.enhance_checkbox = ttk.Checkbutton(enhance_frame, text="Enable AI Enhancement",
                                                variable=self.use_enhancement,
                                                command=self._update_enhancement_controls)
        self.enhance_checkbox.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky="w")

        enhance_frame.grid_columnconfigure(0, minsize=120)
        enhance_frame.grid_columnconfigure(1, weight=1)
        enhance_frame.grid_columnconfigure(2, minsize=50)

        ttk.Label(enhance_frame, text="Strength:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        self.edge_strength = tk.DoubleVar(value=50.0)
        self.edge_scale = ttk.Scale(enhance_frame, from_=0.0, to=100.0,
                                    variable=self.edge_strength, orient="horizontal")
        self.edge_scale.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        self.edge_label = ttk.Label(enhance_frame, text="50%", width=4, anchor="e")
        self.edge_label.grid(row=1, column=2, sticky="e", pady=(10, 0))

        def update_edge_label(*args):
            value = self.edge_strength.get()
            self.edge_label.config(text=f"{int(value)}%")

        self.edge_strength.trace_add("write", update_edge_label)
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
            tonemapped_tensor = tone_mapper.apply_tone_mapping(self.original_input_tensor, method='hable')
            srgb_tensor = self.jxr_loader.linear_to_srgb(tonemapped_tensor)

            preview_tensor = self.jxr_loader.process_preview(
                srgb_tensor,
                self.preview_width,
                self.preview_height
            )
            self._update_preview_ui(preview_tensor, srgb_tensor)
            self.show_color_spectrum_from_tensor(srgb_tensor, is_before=True)

        if getattr(self, 'current_after_image_path', None):
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
                tonemapped_tensor = tone_mapper.apply_tone_mapping(linear_tensor, method='hable')
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
            img = self._resize_image(img, self.preview_width, self.preview_height)
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
        color_strength = 100.0 if use_enhancement else 0.0
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
                auto_method = tone_mapper.select_optimal_tonemap(tone_mapper.analyze_image_statistics(linear_tensor))
                self.master.after(0, lambda: self.tonemap_label.config(text=auto_method, foreground="#00FF00"))
                self.master.after(0, lambda: self.status_label.config(
                    text=f"Converting with {auto_method} tone mapping...", foreground="#CCCCCC"))

                tonemapped_tensor = tone_mapper.apply_tone_mapping(linear_tensor.clone(), method=auto_method)
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
                                                              "TIFF")

                if output_format == "JPEG" or output_format == "Both":
                    self.master.after(0,
                                      lambda: self.show_preview_from_file(f"{output_base_file}.jpg", is_before=False))
                    self.master.after(0, lambda: self.show_color_spectrum(f"{output_base_file}.jpg", is_before=False))

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
        color_strength = 100.0 if use_enhancement else 0.0
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
                        auto_method = tone_mapper.select_optimal_tonemap(
                            tone_mapper.analyze_image_statistics(linear_tensor))

                        tonemapped_tensor = tone_mapper.apply_tone_mapping(linear_tensor.clone(), method=auto_method)
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
                                                                      f"{output_base_name}_tonemapped.tiff", "TIFF")

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


if __name__ == "__main__":
    try:
        app = App("park", "dark")
        root = app.master
        root.title("HDR Image Converter - Enhanced Edition")
        root.minsize(1760, 900)
        window_width = 1760
        window_height = 900
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

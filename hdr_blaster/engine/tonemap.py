"""Tone mapping: the dispatcher, the Auto-select oracle and the learned tone mapper.

The renderings themselves (what each look and operator does to a picture) are in
``renderings.py``, with the rules they all obey.
"""
import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from . import renderings as R
from .colour import _srgb_decode
from .constants import ENGINE_UNIT_NITS, REFERENCE_WHITE_NITS
from .metadata import HDRMetadata
from .neural import _get_neural_tonemap
from .params import measure as _measure


# Kept under their first names for the callers that import them from here.
_pq_encode, _pq_decode, faithful_curve = R.pq_encode, R.pq_decode, R.faithful_curve


class AdvancedToneMapper:
    """Every rendering of the application, and the choice among the classical operators."""

    def __init__(self, device, metadata: Optional[HDRMetadata] = None):
        self.device = device
        self.metadata = metadata or HDRMetadata()
        # Learned tone mapper (None if module or checkpoint missing).
        self.neural_model = _get_neural_tonemap(device)

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

    # ------------------------------------------------------------ renderings
    @staticmethod
    def _relative(image: torch.Tensor, white_nits: float) -> torch.Tensor:
        """Engine units -> linear RGB relative to SDR white, negatives dropped."""
        return image.float().clamp(min=0.0) * (ENGINE_UNIT_NITS / float(white_nits))

    @staticmethod
    def _peak(x: torch.Tensor, params) -> torch.Tensor:
        """Where the picture's highlights end, relative to white: the one thing a
        rendering measures (a video smooths it over each shot)."""
        return _measure(params, "scene.peak", lambda: R.scene_peak(x))

    def tone_map_faithful(self, image: torch.Tensor, white_nits: float = REFERENCE_WHITE_NITS,
                          params=None) -> torch.Tensor:
        """The scene at its own luminance, SDR white standing for ``white_nits``.

        A tone of 20 nits comes out as 20 / white of the SDR range, whatever surrounds
        it. Only what is brighter than white is compressed (:func:`faithful_curve`). The
        curve is driven by the brightest channel and applied as one gain to the three,
        so a highlight keeps its hue and saturation and no channel clips.

        Returns LINEAR [0, 1] like every rendering.
        """
        return self._roll_off(self._relative(image, white_nits), white_nits, params)

    def _roll_off(self, x: torch.Tensor, white_nits: float, params) -> torch.Tensor:
        top = R.brightest(x)
        peak = self._peak(x, params)
        white = float(white_nits)
        mapped = faithful_curve(top * white, white, peak * white) / white
        gain = torch.where(top > 1e-9, mapped / top.clamp(min=1e-9), torch.ones_like(top))
        return (x * gain).clamp(0.0, 1.0)

    def tone_map_look(self, image: torch.Tensor, look: str, white_nits: float = REFERENCE_WHITE_NITS,
                      params=None) -> torch.Tensor:
        """Natural, Punchy, Golden: the faithful picture, graded above its shadows."""
        p = R.LOOKS[look]
        x = R.grade(self._relative(image, white_nits), p["contrast"], p["saturation"], p["warm"])
        return R.fade_highlights(self._roll_off(x, white_nits, params), p["fade"])

    def tone_map_operator(self, image: torch.Tensor, method: str, white_nits: float = REFERENCE_WHITE_NITS,
                          params=None) -> torch.Tensor:
        """The classical operators, each with the trait it is known for (see renderings.py)."""
        x = self._relative(image, white_nits)
        peak = self._peak(x, params)
        if method == "reinhard":
            return R.soft(x, peak)
        if method == "hable":
            return R.film(x, peak)
        if method == "aces":
            return R.per_channel(R._aces, x, peak)
        if method == "drago03":
            return R.hue_kept(R._drago, x, peak)
        if method == "perceptual":
            return R.perceptual(x, peak, white_nits)
        if method == "mantiuk06":
            return R.local_contrast(x, peak)
        if method == "adaptive":
            # Two operators chosen from the picture, one for its lower half and one for its
            # lights, blended by brightness.
            flags = {}

            def flag(name):
                if not flags:
                    flags.update(self.analyze_image_statistics(image))
                return bool(flags[name])

            if _measure(params, "adaptive.extreme_highlights", lambda: flag("has_extreme_highlights")):
                lights = R.soft_hue_kept(x, peak)
            else:
                lights = R.per_channel(R._aces, x, peak)
            if _measure(params, "adaptive.shadow_recovery", lambda: flag("needs_shadow_recovery")):
                lower = R.hue_kept(R._drago, x, peak)
            else:
                lower = R.film(x, peak)
            mask = torch.sigmoid((R.brightest(x) - 0.5) * 10.0)
            return lower * (1.0 - mask) + lights * mask
        raise ValueError(f"unknown tone mapping method '{method}'")

    def keeps_scene_levels(self, method: str) -> bool:
        """Kept for callers: every rendering now keeps the scene's levels, so sharpening
        never rescales the picture. Only a learned tone mapper taught before the looks
        were anchored still goes through the original path."""
        if R.canonical(method) == "neural":
            return getattr(self.neural_model, "anchor_white_nits", None) is not None
        return True

    def apply_tone_mapping(self, image: torch.Tensor, method: Optional[str] = None,
                           params=None, white_nits: float = REFERENCE_WHITE_NITS) -> torch.Tensor:
        """Render ``image`` (engine units) with ``method``; ``None`` lets the Auto-select
        oracle pick a classical operator from the picture's statistics. Linear [0, 1]."""
        if method is None:
            method = self.select_optimal_tonemap(self.analyze_image_statistics(image))
            logging.info(f"Auto-selected tone mapping method: {method}")
        method = R.canonical(method)
        if method == "neural":
            if self.neural_model is not None:
                return self._tone_map_neural(image, params, white_nits)
            logging.warning("Neural tone mapper requested but unavailable; falling back to perceptual.")
            method = "perceptual"
        if method == "faithful":
            return self.tone_map_faithful(image, white_nits, params)
        if method in R.LOOKS:
            return self.tone_map_look(image, method, white_nits, params)
        return self.tone_map_operator(image, method, white_nits, params)

    def _tone_map_neural(self, image: torch.Tensor, params=None,
                         white_nits: float = REFERENCE_WHITE_NITS) -> torch.Tensor:
        """
        Apply the learned neural tone mapper.

        Input  : HDR linear tensor (B, 3, H, W), unbounded non-negative.
        Output : tone-mapped tensor in linear-light [0, 1]. The neural
                 model emits gamma-encoded sRGB directly, so we undo
                 the sRGB EOTF here — the downstream `linear_to_srgb`
                 step re-encodes it back without double-encoding.
        """
        # The model's only whole-image decision is its bilateral grid of affine
        # coefficients; everything after it is per-pixel. Recording the grid makes
        # the operator replayable like the others.
        scale = getattr(self.neural_model, "engine_input_scale", 1.0)
        anchor = getattr(self.neural_model, "anchor_white_nits", None)
        if anchor is not None:
            # Taught by a look anchored on a white of `anchor` nits: show it the scene as a
            # screen with that white would, so that it renders this screen's white the same.
            scale = scale * (anchor / float(white_nits))
        x = image if scale == 1.0 else image * scale     # engine units -> the checkpoint's unit
        with torch.inference_mode():
            grid = _measure(params, "neural.grid", lambda: self.neural_model.predict_grid(x))
            srgb_out = self.neural_model.apply_grid(x, grid)
        return _srgb_decode(srgb_out)

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

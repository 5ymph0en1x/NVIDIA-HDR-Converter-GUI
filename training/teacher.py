"""
Teacher rendering and GPU-side augmentation for training on real HDR frames.

There is no ready-made "correct SDR" for a game's HDR picture. The target the network
learns is therefore produced by a look of the application itself (Punchy by default: the
one its author tuned to match the in-game SDR render), computed on the GPU for every
batch, from the very frame the network is given.

The looks keep the scene's luminance: SDR white stands for a fixed luminance, dark tones
and mid-grey stay where the scene put them. The teacher therefore renders every
frame against ONE white (``TEACHER_WHITE_NITS``), which the checkpoint records. The
application then shows the model each scene as a screen with that white would
(luminance x teacher white / this screen's white), so the model renders any screen's
white the way it was taught.

Because the target is computed after the augmentation, each frame is seen at many
exposures: that is what covers screens brighter or dimmer than the teacher's white.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEACHERS = ("natural", "punchy", "golden")
_EARLIER_NAMES = {"agx": "natural", "agx_punchy": "punchy", "agx_golden": "golden"}
TEACHER_WHITE_NITS = 203.0          # ITU-R BT.2408 reference white


class Teacher:
    """The application's own look, returning sRGB in [0, 1]."""

    def __init__(self, look: str, device, white_nits: float = TEACHER_WHITE_NITS):
        look = _EARLIER_NAMES.get(look, look)
        if look not in TEACHERS:
            raise ValueError(f"unknown teacher '{look}' (expected one of {TEACHERS})")
        from hdr_blaster.engine.constants import ENGINE_UNIT_NITS
        from hdr_blaster.engine.tonemap import AdvancedToneMapper
        from hdr_blaster.io.jxr import OptimizedJXRLoader

        self.look = look
        self.white_nits = float(white_nits)
        self.unit = ENGINE_UNIT_NITS
        self._loader = OptimizedJXRLoader(device)
        self._mapper = AdvancedToneMapper(device)

    @torch.no_grad()
    def __call__(self, nits: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) nits -> (B, 3, H, W) sRGB. A whole batch is rendered in one go;
        the only thing a look measures, the peak, is taken per picture, so no picture
        influences another."""
        linear = self._mapper.tone_map_look(nits.float() / self.unit, self.look, white_nits=self.white_nits)
        return self._loader.linear_to_srgb(linear)


def augment(nits: torch.Tensor, ev: float = 2.0, peak_range=(600.0, 4000.0),
            generator: torch.Generator | None = None) -> torch.Tensor:
    """Random exposure (+/- ``ev`` stops), display peak (log-uniform, values above it are
    clipped as an HDR10 display pipeline does) and horizontal flip, per image, on the GPU."""
    batch = nits.shape[0]

    def rand(*shape):
        return torch.rand(*shape, device=nits.device, generator=generator)

    stops = (rand(batch, 1, 1, 1) * 2.0 - 1.0) * ev
    out = nits * torch.pow(2.0, stops)
    lo, hi = math.log(peak_range[0]), math.log(peak_range[1])
    peak = torch.exp(lo + (hi - lo) * rand(batch, 1, 1, 1))
    out = torch.minimum(out, peak)
    flip = rand(batch) < 0.5
    return torch.where(flip.view(batch, 1, 1, 1), out.flip(-1), out)

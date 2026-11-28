"""Colour conversions between a video frame and the engine's working space.

In:  HDR10 planes (Y'CbCr, PQ transfer, BT.2020 primaries, 4:2:0, chroma sited
     'left' as in every HEVC/H.264 stream that does not say otherwise).
Out: the tensor the engine expects, identical in meaning to what the JXR loader
     produces: linear BT.709 primaries, scRGB x 0.1 (1.0 = 800 nits), including signed wide-gamut colours.

And back: a finished sRGB tensor to 8-bit BT.709 limited-range 4:2:0 planes.
"""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from ..engine.colour import signed_power
from ..engine.constants import ENGINE_UNIT_NITS

SCRGB_WHITE_NITS = 80.0
ENGINE_SCALE = SCRGB_WHITE_NITS / ENGINE_UNIT_NITS       # 0.1, the JXR loader's display scale

# SMPTE ST 2084 (PQ)
_M1, _M2 = 0.1593017578125, 78.84375
_C1, _C2, _C3 = 0.8359375, 18.8515625, 18.6875
PQ_PEAK_NITS = 10000.0

# Y'CbCr luma coefficients
_BT2020 = (0.2627, 0.0593)       # Kr, Kb
_BT709 = (0.2126, 0.0722)

_BT2020_TO_BT709 = (
    (1.660491, -0.587641, -0.072850),
    (-0.124550, 1.132900, -0.008349),
    (-0.018151, -0.100579, 1.118730),
)


@dataclass(frozen=True)
class FrameFormat:
    """What is needed to interpret the planes of a decoded frame."""
    width: int
    height: int
    bit_depth: int = 10
    full_range: bool = False
    matrix: str = "bt2020nc"


def pq_eotf(code: torch.Tensor) -> torch.Tensor:
    """PQ code value in [0, 1] -> absolute luminance in nits."""
    p = code.clamp(0.0, 1.0) ** (1.0 / _M2)
    return PQ_PEAK_NITS * ((p - _C1).clamp(min=0.0) / (_C2 - _C3 * p)) ** (1.0 / _M1)


_BT709_TO_BT2020 = (
    (0.6274040, 0.3292820, 0.0433136),
    (0.0690970, 0.9195400, 0.0113612),
    (0.0163916, 0.0880132, 0.8955950),
)


def pq_oetf(nits: torch.Tensor) -> torch.Tensor:
    """Absolute luminance in nits -> PQ code value in [0, 1] (inverse of :func:`pq_eotf`)."""
    y = (nits.clamp(0.0, PQ_PEAK_NITS) / PQ_PEAK_NITS) ** _M1
    return ((_C1 + _C2 * y) / (1.0 + _C3 * y)) ** _M2


def unconverted_look(linear: torch.Tensor) -> torch.Tensor:
    """What an SDR application shows of an HDR10 video it does not understand: the
    PQ-encoded BT.2020 values taken for ordinary sRGB. Dull and washed out; this is the
    'before' picture. ``linear`` is in the engine's working space."""
    m = torch.tensor(_BT709_TO_BT2020, dtype=torch.float32, device=linear.device)
    b, _, h, w = linear.shape
    nits2020 = (m @ (linear.float() * ENGINE_UNIT_NITS).reshape(b, 3, -1)).reshape(b, 3, h, w)
    return pq_oetf(nits2020).clamp(0.0, 1.0)


def _upsample_chroma_left(c: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(1, 1, H/2, W/2) -> (1, 1, H, W) for 'left' siting: chroma samples sit on the even
    luma columns and halfway between luma rows."""
    c = F.interpolate(c, size=(height, c.shape[-1]), mode="bilinear", align_corners=False)
    out = c.new_empty(c.shape[0], c.shape[1], height, width)
    out[..., 0::2] = c
    nxt = torch.cat([c[..., 1:], c[..., -1:]], dim=-1)
    out[..., 1::2] = 0.5 * (c + nxt)
    return out


def yuv_to_engine(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, fmt: FrameFormat,
                  pre_gamma: float = 1.0, exposure: float = 1.0) -> torch.Tensor:
    """HDR10 planes -> (1, 3, H, W) float32 in the engine's working space.

    ``y`` is (H, W), ``u`` and ``v`` are (H/2, W/2), integer code values, already on
    the device to compute on. ``pre_gamma`` and ``exposure`` are the user's
    controls, applied where the JXR loader applies them: on scRGB values, gamma
    first, before the x 0.1 display scale.
    """
    return scrgb_to_engine(yuv_to_scrgb(y, u, v, fmt), pre_gamma, exposure)


def scrgb_to_engine(scrgb: torch.Tensor, pre_gamma: float = 1.0, exposure: float = 1.0) -> torch.Tensor:
    """The scene's light (:func:`yuv_to_scrgb`) with the user's controls, in the engine's
    working space."""
    if pre_gamma != 1.0:
        scrgb = signed_power(scrgb, 1.0 / pre_gamma)
    if exposure != 1.0:
        scrgb = scrgb * exposure
    return scrgb * ENGINE_SCALE


def yuv_to_scrgb(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, fmt: FrameFormat) -> torch.Tensor:
    """HDR10 planes -> (1, 3, H, W) float32, the scene's light as recorded: linear BT.709
    primaries, signed scRGB (1.0 = 80 nits), before any control of the user.
    Light adds up on this scale: the average of several frames is what a longer exposure
    would have recorded."""
    if fmt.matrix != "bt2020nc":
        raise ValueError(f"unsupported colour matrix '{fmt.matrix}'")
    height, width = y.shape[-2:]
    top = float((1 << fmt.bit_depth) - 1)
    unit = float(1 << (fmt.bit_depth - 8))

    y = y.to(torch.float32).view(1, 1, height, width)
    u = u.to(torch.float32).view(1, 1, *u.shape[-2:])
    v = v.to(torch.float32).view(1, 1, *v.shape[-2:])
    if fmt.full_range:
        yp, cb, cr = y / top, (u - 128.0 * unit) / top, (v - 128.0 * unit) / top
    else:
        yp = (y - 16.0 * unit) / (219.0 * unit)
        cb = (u - 128.0 * unit) / (224.0 * unit)
        cr = (v - 128.0 * unit) / (224.0 * unit)
    cb = _upsample_chroma_left(cb, height, width)
    cr = _upsample_chroma_left(cr, height, width)

    kr, kb = _BT2020
    kg = 1.0 - kr - kb
    r = yp + 2.0 * (1.0 - kr) * cr
    b = yp + 2.0 * (1.0 - kb) * cb
    g = (yp - kr * r - kb * b) / kg
    nits2020 = pq_eotf(torch.cat([r, g, b], dim=1))

    m = torch.tensor(_BT2020_TO_BT709, dtype=torch.float32, device=y.device)
    scrgb = (m @ nits2020.reshape(1, 3, -1)).reshape(1, 3, height, width) / SCRGB_WHITE_NITS
    return scrgb


def srgb_to_yuv420(srgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finished sRGB image (1, 3, H, W) in [0, 1] -> 8-bit BT.709 limited-range planes.

    Returns ``(y (H, W), u (H/2, W/2), v (H/2, W/2))`` as uint8, chroma sited left.
    """
    _, _, height, width = srgb.shape
    if height % 2 or width % 2:
        raise ValueError(f"4:2:0 needs even dimensions, got {width}x{height}")
    srgb = srgb.to(torch.float32).clamp(0.0, 1.0)
    kr, kb = _BT709
    kg = 1.0 - kr - kb
    r, g, b = srgb[:, 0:1], srgb[:, 1:2], srgb[:, 2:3]
    yp = kr * r + kg * g + kb * b
    cb = (b - yp) / (2.0 * (1.0 - kb))
    cr = (r - yp) / (2.0 * (1.0 - kr))

    def subsample(c):
        # [1, 2, 1] / 4 centred on the even columns (left siting), then average row pairs.
        c = F.pad(c, (1, 1, 0, 0), mode="replicate")
        c = 0.25 * c[..., 0:-2:2] + 0.5 * c[..., 1:-1:2] + 0.25 * c[..., 2::2]
        return 0.5 * (c[..., 0::2, :] + c[..., 1::2, :])

    def quantise(plane, offset, span):
        return (offset + span * plane).round().clamp(0.0, 255.0).to(torch.uint8)[0, 0]

    return quantise(yp, 16.0, 219.0), quantise(subsample(cb), 128.0, 224.0), quantise(subsample(cr), 128.0, 224.0)

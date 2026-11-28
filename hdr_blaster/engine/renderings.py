"""The renderings: every look and every operator as a variation of the SAME picture.

A game's HDR picture is finished work: its shadows, its mid-tones and its colours were
decided by the game. Converting it to SDR only has to bring what is brighter than SDR
white under it. The first version of this application treated the picture as raw scene
light instead, and let each operator expose it and bend its whole range: dark zones came
out two to three times too bright (or crushed), tinted, and the lift made the noise and
the compression blocks hidden in the dark visible.

So every rendering here obeys the same rules, enforced by ``tests/test_renderings.py``:

* below a quarter of mid-grey the picture is the scene itself, untouched;
* mid-grey (18 % of white) stays mid-grey: nothing re-exposes the picture;
* the scene's peak lands on white: highlights are never dull;
* tones keep their order.

What distinguishes the renderings lives above the shadows: how the highlights roll off,
contrast, saturation, tint, what becomes of a colour brighter than white, local detail.

That is less room than it seems. A smooth curve that must pass through the shadows,
mid-grey and the peak is almost entirely decided: held to those three points, the curves
of Reinhard, Hable and Drago are the same curve to within a level or two (measured on
the author's captures: Perceptual, Reinhard and Mantiuk06 shared one curve and Drago's
lay on it). So each operator is given, above the shadows, the trait it is known for:

* Reinhard    soft: nothing touched up to mid-grey, its shoulder from there, per channel
              (bright colours go pastel);
* Hable       film: more contrast in the mid-tones, muted colours, its shoulder with the
              colours kept, lights that fade towards white at the very top;
* ACES        vivid: its own strong contrast, rich mid-tones, lights that burn to white;
* Drago03     flat: its logarithm compresses the whole range evenly, colours kept;
* Perceptual  luminous: a positive log slope on ICtCp intensity. A coloured light keeps its hue
              and its brightness; its colour yields (Faithful keeps the weakest
              channel of a light as the scene has it: a neon stays a neon, and dark);
* Guided local contrast: a soft roll-off of the large shapes, their detail reinforced.

Everything here works on ``x``: linear RGB relative to SDR white (1.0 = white, 0.18 =
mid-grey), and on ``peak``: the brightest channel value of the scene on that scale.
"""
import math
from functools import lru_cache
from typing import Callable, Optional

import torch
import torch.nn.functional as F

GREY = 0.18                 # mid-grey, as a share of white
SHADOW_STOPS = 2.0          # below GREY / 2**SHADOW_STOPS a rendering is the scene itself

# SMPTE ST 2084 (PQ): the scale on which equal steps look equally large, from black to 10 000 nits.
_PQ_M1, _PQ_M2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
_PQ_C1, _PQ_C2, _PQ_C3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0

_ALIASES = {"agx": "natural", "agx_punchy": "punchy", "agx_golden": "golden",
            "uncharted2": "hable", "filmic": "hable"}


def canonical(method: Optional[str]) -> Optional[str]:
    """The current name of a rendering (the looks were called after AgX, which they no
    longer use; ``uncharted2`` and ``filmic`` always were Hable's curve)."""
    return _ALIASES.get(method, method)


def pq_encode(nits: torch.Tensor) -> torch.Tensor:
    y = (nits.clamp(0.0, 10000.0) / 10000.0) ** _PQ_M1
    return ((_PQ_C1 + _PQ_C2 * y) / (1.0 + _PQ_C3 * y)) ** _PQ_M2


def pq_decode(code: torch.Tensor) -> torch.Tensor:
    p = code.clamp(0.0, 1.0) ** (1.0 / _PQ_M2)
    return 10000.0 * ((p - _PQ_C1).clamp(min=0.0) / (_PQ_C2 - _PQ_C3 * p)) ** (1.0 / _PQ_M1)


def faithful_curve(nits: torch.Tensor, white_nits, peak_nits) -> torch.Tensor:
    """Bring a scene that peaks at ``peak_nits`` under ``white_nits`` without touching
    what already fits: luminance in, luminance out, both in nits.

    This is the roll-off of ITU-R BT.2390 (the reference way to show HDR on a dimmer
    display). On the PQ scale, tones below a knee are returned EXACTLY as they came;
    above it a smooth shoulder (a Hermite curve leaving the knee with slope 1 and
    arriving flat) lands the peak on white. A scene that fits under white is unchanged.
    """
    nits = nits.float()
    white = torch.as_tensor(white_nits, dtype=nits.dtype, device=nits.device)
    peak = torch.maximum(torch.as_tensor(peak_nits, dtype=nits.dtype, device=nits.device), white)
    top = pq_encode(peak)
    target = pq_encode(white) / top                     # where white sits when the peak is 1
    knee = (1.5 * target - 0.5).clamp(min=0.0)
    position = pq_encode(nits) / top
    t = ((position.clamp(max=1.0) - knee) / (1.0 - knee).clamp(min=1e-6)).clamp(0.0, 1.0)
    t2, t3 = t * t, t * t * t
    shoulder = ((2.0 * t3 - 3.0 * t2 + 1.0) * knee + (t3 - 2.0 * t2 + t) * (1.0 - knee)
                + (-2.0 * t3 + 3.0 * t2) * target)
    return torch.where(position <= knee, nits, torch.minimum(pq_decode(shoulder * top), white))


def faithful_stops(peak, device, width: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Log2 headroom above white, and the fraction of highlight contrast the display can hold."""
    peak = torch.as_tensor(peak, dtype=torch.float32, device=device)
    headroom = torch.log2(peak.clamp_min(1.0))
    # The transition uses half the reserved interval. Its average slope is
    # halfway between unity and the final slope.
    room = 0.75 * width
    return headroom, room / (room + headroom)


def log_shoulder(level: torch.Tensor, peak, width: float) -> torch.Tensor:
    """Compress the upper width stops and HDR headroom, joining at unit log slope.

    The slope falls linearly through the first half of the reserved interval,
    then stays positive up to the measured peak. Integrating gives the roll-off.
    """
    headroom, slope = faithful_stops(peak, level.device, width)
    stops = (torch.log2(level.clamp_min(1e-9)) + width).clamp_min(0.0)
    stops = torch.minimum(stops, width + headroom)
    bend = stops.clamp_max(0.5 * width)
    log_out = slope * stops + (1.0 - slope) * (bend * (1.0 - bend / width))
    compressed = torch.exp2(log_out - width).clamp_max(1.0)
    return torch.where((stops == 0.0) | ((headroom == 0) & (level <= 1)), level, compressed)


def faithful_highlights(x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """Faithful in linear RGB relative to SDR white. Half a stop below white, a
    quarter-stop transition joins the unchanged tones to a power law. Its positive
    log slope keeps two bright clouds distinct up to the peak, which lands on white.

    At each pixel map the weakest channel between half-white and the mapped maximum.
    Its slope starts at one, falls linearly, then stays positive. Integrating those
    slopes gives the expression below. The middle channel keeps its position between
    the other two. This preserves the RGB hue of a glow and leaves a neon's dark
    channel alone. Peak is the only scene statistic, including on video.
    """
    high, low = x.amax(dim=1, keepdim=True), x.amin(dim=1, keepdim=True)
    mapped_high = log_shoulder(high, peak, 0.5)

    room = (mapped_high - 0.5).clamp_min(0.0)
    excess = (low - 0.5).clamp_min(0.0)
    # The transition occupies a quarter of the available room. Choose the final slope so
    # the integral reaches mapped_high exactly when low reaches high.
    end_slope = (0.875 * room / (high - 0.5 - 0.125 * room).clamp_min(1e-9)).clamp(0.0, 1.0)
    toe = torch.minimum(excess, 0.25 * room)
    transition = toe * (1.0 - 2.0 * toe / room.clamp_min(1e-9))
    mapped_low = 0.5 + end_slope * excess + (1.0 - end_slope) * transition
    mapped_low = torch.where(low <= 0.5, low, torch.minimum(mapped_low, mapped_high))
    position = (x - low) / (high - low).clamp_min(1e-9)
    result = mapped_low + position * (mapped_high - mapped_low)
    unchanged = (high <= 2.0 ** -0.5) | ((peak <= 1.0) & (high <= 1.0))
    return torch.where(unchanged, x, result).clamp(0.0, 1.0)


def roll_off(x: torch.Tensor, white_nits, peak) -> torch.Tensor:
    """The 3.1.2 shoulder formerly used for the three graded looks. ``x`` is
    relative to white and has no negative; ``peak`` is the scene's, on the same scale.

    The brightest and the weakest channel of each pixel go through :func:`faithful_curve`,
    each for itself; the middle one is placed between them where it stood between them in
    the scene, so the hue does not move. What this gives:

    * a pixel whose brightest channel is under the knee is handed back as it came, to the
      last bit: nothing is computed for it;
    * the brightest channel of the scene's peak lands on white, and no channel passes it;
    * a light keeps the colour its weakest channel gives it. Under the knee that channel
      is the scene's own: a neon stays a neon, as the source view shows it. Above white it
      is rolled off with the brightest one, which it nearly joins: a glow brighter than
      white on its three channels burns towards white, as it does for a camera and for the eye.

    Until 3.1.1 the curve was read on the brightest channel alone and applied as one gain
    to the three. A light kept its whole colour and paid for it in luminance: the glow of a
    rising sun (999 / 852 / 684 nits under a white of 480) was shown as a flat peach
    surface, 255 / 238 / 216, over a fifth of the picture."""
    white = float(white_nits)
    peak_nits = torch.as_tensor(peak, dtype=torch.float32, device=x.device) * white
    top, low = x.amax(dim=1, keepdim=True), x.amin(dim=1, keepdim=True)
    top_nits, low_nits = top * white, low * white
    top_mapped = faithful_curve(top_nits, white, peak_nits)
    low_mapped = faithful_curve(low_nits, white, peak_nits)
    top_out, low_out = top_mapped / white, low_mapped / white
    span = top - low
    place = torch.where(span > 1e-9, (x - low) / span.clamp(min=1e-9), torch.zeros_like(x))
    rolled = low_out + place * (top_out - low_out)
    return torch.where(top_mapped == top_nits, x, rolled).clamp(0.0, 1.0)


def luminance(x: torch.Tensor) -> torch.Tensor:
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def brightest(x: torch.Tensor) -> torch.Tensor:
    """The brightest channel of each pixel, (B, 1, H, W). Curves are driven by it rather
    than by luminance: a saturated light then stays within white and keeps its hue.
    Faithful and the graded looks also use the weakest channel to render a coloured light."""
    return x.amax(dim=1, keepdim=True)


def scene_peak(x: torch.Tensor) -> torch.Tensor:
    """Where the roll-off ends: the level only one pixel in ten thousand exceeds. The true
    maximum belongs to a stray pixel as often as to the picture (the chroma of a 4:2:0
    video overshoots on sharp coloured edges), and would dim every highlight for it; what
    exceeds the peak is simply rendered white. One value per picture: a plain number for a
    single picture, (B, 1, 1, 1) for a batch."""
    top = brightest(x).flatten(1).float()
    step = max(1, top.shape[1] // 2_000_000)
    peak = torch.quantile(top[:, ::step], 0.9999, dim=1)
    return peak.reshape(()) if x.shape[0] == 1 else peak.view(-1, 1, 1, 1)


def shadow_weight(level: torch.Tensor) -> torch.Tensor:
    """How much of a rendering applies at this level: 0 below a quarter of mid-grey (the
    scene is left alone), 1 from mid-grey up, smooth in between, counted in stops."""
    stops = torch.log2(level.clamp(min=1e-9) / GREY)
    t = ((stops + SHADOW_STOPS) / SHADOW_STOPS).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def srgb_shadow_weight(srgb: torch.Tensor) -> torch.Tensor:
    """The shadow protection used by tone mapping, on the finished sRGB picture."""
    from .colour import _srgb_decode

    return shadow_weight(_srgb_decode(brightest(srgb).float())).to(srgb.dtype)


# ---------------------------------------------------------------- the curves
# Each takes values relative to white and the scene's peak, and returns 1.0 at the peak.

def _reinhard(x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """Extended Reinhard: a long, gentle roll-off that starts early."""
    return x * (1.0 + x / (peak * peak)) / (1.0 + x)


def _hable_shape(x):
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f


def _hable(x: torch.Tensor, peak: torch.Tensor, exposure: float = 2.0) -> torch.Tensor:
    """Hable's filmic curve (Uncharted 2): firm mid-tones, long shoulder."""
    return _hable_shape(exposure * x) / _hable_shape(exposure * peak)


def _aces_shape(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return (x * (a * x + b)) / (x * (c * x + d) + e)


def _aces(x: torch.Tensor, peak: torch.Tensor, exposure: float = 0.6) -> torch.Tensor:
    """ACES filmic fit (Narkowicz): contrasty, highlights fade towards white."""
    return _aces_shape(exposure * x) / _aces_shape(exposure * peak)


def _drago(x: torch.Tensor, peak: torch.Tensor, bias: float = 0.85) -> torch.Tensor:
    """Drago's adaptive logarithm: the softest highlights, most detail kept near the peak."""
    power = math.log(bias) / math.log(0.5)
    ratio = (x / peak).clamp(0.0, 1.0)
    return (torch.log1p(x) / torch.log1p(peak)) * (math.log(10.0) / torch.log(2.0 + 8.0 * ratio ** power))


def anchored(curve: Callable, x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """A curve with two fixed points whatever the scene: the peak on white (the curves are
    built for it) and mid-grey on mid-grey, by raising the curve to the one power that
    puts it there. A scene that already fits under white needs no roll-off at all: for
    Reinhard this gives back the picture unchanged."""
    peak = peak.clamp(min=1.0)
    grey = curve(torch.full_like(peak, GREY), peak).clamp(1e-6, 1.0 - 1e-6)
    power = math.log(GREY) / torch.log(grey)
    return curve(x.clamp(min=0.0), peak).clamp(0.0, 1.0) ** power


def reinhard_shoulder(u: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
    """Extended Reinhard read as a shoulder: it leaves 0 at slope 1 and reaches 1 at ``top``."""
    return u * (1.0 + u / (top * top)) / (1.0 + u)


def from_knee(shoulder: Callable, x: torch.Tensor, peak: torch.Tensor, knee: float = GREY) -> torch.Tensor:
    """The scene itself up to the knee; from there a shoulder that leaves the knee without a
    corner and lands the peak on white. With the knee on mid-grey nothing below it is
    touched, and no power is needed to hold it in place. A scene that fits under white
    comes back unchanged."""
    span = 1.0 - knee
    top = ((torch.as_tensor(peak, dtype=x.dtype, device=x.device) - knee) / span).clamp(min=1.0)
    u = ((x - knee) / span).clamp(min=0.0)
    return torch.where(x <= knee, x, knee + span * shoulder(torch.minimum(u, top), top))


def soft(x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """Reinhard: its shoulder from mid-grey up, on each channel. Bright colours go pastel."""
    return from_knee(reinhard_shoulder, x.clamp(min=0.0), peak).clamp(0.0, 1.0)


def soft_hue_kept(x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """The same shoulder on the brightest channel, as one gain: colours are kept."""
    top = brightest(x)
    mapped = from_knee(reinhard_shoulder, top, peak)
    gain = torch.where(top > 1e-9, mapped / top.clamp(min=1e-9), torch.ones_like(top))
    return (x * gain).clamp(0.0, 1.0)


def protect_shadows(x: torch.Tensor, rendered: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    """The rendering above the shadows, the scene itself in them."""
    weight = shadow_weight(level)
    return weight * rendered + (1.0 - weight) * x


def per_channel(curve: Callable, x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """The curve on each channel, the classic way: bright colours drift towards white."""
    return protect_shadows(x, anchored(curve, x, peak), brightest(x)).clamp(0.0, 1.0)


def hue_kept(curve: Callable, x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """The curve on the brightest channel, as one gain for the three: colours are kept."""
    top = brightest(x)
    mapped = protect_shadows(top, anchored(curve, top, peak), top)
    gain = torch.where(top > 1e-9, mapped / top.clamp(min=1e-9), torch.ones_like(top))
    return (x * gain).clamp(0.0, 1.0)


# --------------------------------------------------------- local (Mantiuk06)

def _box(image: torch.Tensor, radius: int) -> torch.Tensor:
    size = 2 * radius + 1
    padded = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(padded, size, stride=1)


def _edge_aware_base(log_level: torch.Tensor, radius: int, epsilon: float = 0.05) -> torch.Tensor:
    """A smoothed picture that keeps its strong edges (guided filter, guided by itself):
    a plain blur would leave a dark halo around every light."""
    mean = _box(log_level, radius)
    variance = (_box(log_level * log_level, radius) - mean * mean).clamp_min(0.0)
    a = variance / (variance + epsilon)
    b = mean - a * mean
    return _box(a, radius) * log_level + _box(b, radius)


LOCAL_BOOST = 1.5            # Mantiuk06: local contrast above the shadows is rendered this much stronger


def local_contrast(x: torch.Tensor, peak: torch.Tensor, boost: float = LOCAL_BOOST) -> torch.Tensor:
    """Mantiuk's idea, kept to what a finished picture needs. The picture is split in
    three: its large shapes (smoothed over a wide radius, edges kept), its local contrast
    (what a narrower smoothing still sees and the wide one does not: the relief of a
    cloud, a facade, a face), and its finest detail (the rest: texture, but grain too).

    The large shapes are rolled off softly. The local contrast is not flattened with
    them: it is reinforced, above the shadows only. The finest detail is passed as it is,
    so that grain and compression are never amplified. A soft limit then holds what would
    pass white."""
    top = brightest(x)
    height, width = top.shape[-2:]
    wide = max(4, round(max(height, width) / 96))
    narrow = max(1, wide // 6)
    log_top = torch.log(top.clamp(min=1e-6))
    log_shapes = _edge_aware_base(log_top, wide)
    log_relief = _edge_aware_base(log_top, narrow)
    shapes = torch.exp(log_shapes)
    rolled = from_knee(reinhard_shoulder, shapes, peak)
    relief = (log_relief - log_shapes) * (1.0 + boost * shadow_weight(shapes))
    out = rolled * torch.exp(relief + (log_top - log_relief))
    # Limit only positive relief to the headroom above the mapped base. A flat
    # peak now reaches white, while detail approaching the ceiling rolls off.
    room = (1.0 - rolled).clamp_min(0.0)
    lift = (out - rolled).clamp_min(0.0)
    out = torch.minimum(out, rolled) + room * torch.tanh(lift / room.clamp_min(1e-9))
    gain = torch.where(top > 1e-9, out / top.clamp(min=1e-9), torch.ones_like(top))
    # in the shadows the scene itself, as everywhere else
    weight = shadow_weight(top)
    return (x * (weight * gain + (1.0 - weight))).clamp(0.0, 1.0)


# ------------------------------------------------------------ ICtCp (Perceptual)
# ITU-R BT.2100: the colour space built on PQ in which hue lines are straight and equal
# steps look equal. I is intensity, Ct and Cp the two chroma axes.
_BT709_TO_BT2020 = ((0.6274040, 0.3292820, 0.0433136), (0.0690970, 0.9195400, 0.0113612),
                    (0.0163916, 0.0880132, 0.8955950))
_BT2020_TO_LMS = ((1688.0, 2146.0, 262.0), (683.0, 2951.0, 462.0), (99.0, 309.0, 3688.0))
_LMS_TO_ICTCP = ((2048.0, 2048.0, 0.0), (6610.0, -13613.0, 7003.0), (17933.0, -17390.0, -543.0))


@lru_cache(maxsize=16)
def _ictcp_matrices(device: torch.device, dtype: torch.dtype):
    to_lms = (torch.tensor(_BT2020_TO_LMS, dtype=torch.float64) / 4096.0) @ torch.tensor(_BT709_TO_BT2020, dtype=torch.float64)
    to_ictcp = torch.tensor(_LMS_TO_ICTCP, dtype=torch.float64) / 4096.0
    return tuple(m.to(device=device, dtype=dtype)
                 for m in (to_lms, to_ictcp, torch.linalg.inv(to_ictcp), torch.linalg.inv(to_lms)))


def _mix(matrix: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    b, _, h, w = image.shape
    return (matrix @ image.reshape(b, 3, -1)).reshape(b, 3, h, w)


def to_ictcp(nits: torch.Tensor) -> torch.Tensor:
    """Linear BT.709 RGB in nits, (B, 3, H, W) -> ICtCp."""
    to_lms, to_ictcp_, _, _ = _ictcp_matrices(nits.device, nits.dtype)
    return _mix(to_ictcp_, pq_encode(_mix(to_lms, nits).clamp(min=0.0)))


def from_ictcp(ictcp: torch.Tensor) -> torch.Tensor:
    """ICtCp -> linear BT.709 RGB in nits. A colour outside BT.709 comes back with a negative channel."""
    _, _, to_pq_lms, to_rgb = _ictcp_matrices(ictcp.device, ictcp.dtype)
    return _mix(to_rgb, pq_decode(_mix(to_pq_lms, ictcp).clamp(0.0, 1.0)))


def _fit_ictcp(ictcp: torch.Tensor, white: float) -> torch.Tensor:
    """Reduce chroma along a ray of constant I and hue until its RGB fits the display."""
    _, _, to_lms_code, to_rgb = _ictcp_matrices(ictcp.device, ictcp.dtype)
    lms_code = _mix(to_lms_code, ictcp)
    rgb = _mix(to_rgb, pq_decode(lms_code)) / white
    outside = ((rgb < 0) | (rgb > 1)).any(dim=1)
    # Work only on the affected pixels, in bounded chunks even for a full 4K sky.
    indices = outside.flatten().nonzero().flatten()
    if indices.numel() == 0:
        return rgb
    result = rgb.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    intensity = ictcp[:, 0].reshape(-1, 1)
    codes = lms_code.permute(0, 2, 3, 1).reshape(-1, 3)
    for ix in indices.split(262144):
        neutral = intensity[ix]
        offset = codes[ix] - neutral
        lo, hi = torch.zeros_like(neutral), torch.ones_like(neutral)
        for _ in range(16):
            mid = (lo + hi) * 0.5
            # Inverse PQ and inverse LMS are nonlinear together; test actual RGB.
            trial = pq_decode(neutral + mid * offset) @ to_rgb.T
            fits = ((trial >= 0.0) & (trial <= white)).all(dim=1, keepdim=True)
            lo, hi = torch.where(fits, mid, lo), torch.where(fits, hi, mid)
        result[ix] = (pq_decode(neutral + lo * offset) @ to_rgb.T) / white
    b, _, h, w = ictcp.shape
    return result.view(b, h, w, 3).permute(0, 3, 1, 2).clamp(0.0, 1.0)


def perceptual(x: torch.Tensor, peak: torch.Tensor, white_nits: float) -> torch.Tensor:
    """Roll off ICtCp intensity in log light and fit chroma at that intensity.

    Reserve one stop below white for bright structure. In-gamut colours below
    the shoulder pass through exactly, including all representable shadows.
    """
    white = float(white_nits)
    ictcp = to_ictcp(x * white)
    intensity = ictcp[:, 0:1]
    level = pq_decode(intensity) / white
    target = log_shoulder(level, peak, 1.0)
    mapped = torch.where(target == level, intensity, pq_encode(target * white))
    ratio = (mapped / intensity.clamp_min(1e-9)).clamp_max(1.0)
    rendered = _fit_ictcp(torch.cat((mapped, ictcp[:, 1:] * ratio), dim=1), white)
    high, low = brightest(x), x.amin(dim=1, keepdim=True)
    untouched = (target == level) & (low >= 0) & (high <= 1)
    rendered = torch.where(untouched, x, rendered)
    # A neutral's intensity in nits is its own channel value. Evaluating that
    # axis directly avoids PQ round-trip noise on fine grey ramps.
    return torch.where(high == low, log_shoulder(high.clamp_min(0), peak, 1.0), rendered)


# ------------------------------------------------------------------ film (Hable)

# Hable, the film curve of Uncharted 2. Its toe (what deepens a film's low tones) lies in
# the shadows, which no rendering may touch; held to mid-grey and the peak, what is left of
# it is Reinhard's shoulder to a level and a half. Its character is given back above the
# shadows: the contrast of the toe as a grade around mid-grey, film's muted colours, and
# lights that lose their colour only at the very top.
FILM = dict(contrast=1.15, saturation=0.90, fade=0.35)


def _rise_at_grey(curve: Callable, peak: torch.Tensor) -> torch.Tensor:
    """How fast an anchored curve rises through mid-grey, compared with the scene (1.0 = as
    the scene). It depends on the peak: the higher the peak, the flatter the curve."""
    peak = peak.double()
    low, high = torch.full_like(peak, GREY / 1.02), torch.full_like(peak, GREY * 1.02)
    return torch.log(anchored(curve, high, peak) / anchored(curve, low, peak)) / (2.0 * math.log(1.02))


def film(x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """Hable: film's contrast and muted colours above the shadows, then its own shoulder on
    the brightest channel (colours kept), and lights that fade at the very top.
    The grade is set so that the finished picture rises through mid-grey ``FILM["contrast"]``
    times faster than the scene, whatever the peak (the shoulder alone flattens the
    mid-tones more and more as the peak rises)."""
    peak = torch.as_tensor(peak, dtype=torch.float32, device=x.device).clamp(min=1.0)
    grade_by, lifted = torch.ones_like(peak), peak
    for _ in range(3):                      # the grade lifts the peak, which flattens the shoulder: settle
        grade_by = (FILM["contrast"] / _rise_at_grey(_hable, lifted)).float().clamp(1.0, 2.0)
        lifted = peak * (peak / GREY) ** (grade_by - 1.0)
    top = brightest(x)
    weight = shadow_weight(top)
    graded = x * (1.0 + weight * ((top.clamp(min=1e-9) / GREY) ** (grade_by - 1.0) - 1.0))
    graded = grade(graded, 1.0, FILM["saturation"], None)
    return fade_highlights(hue_kept(_hable, graded, lifted), FILM["fade"])


# ------------------------------------------------------------------ the looks

LOOKS = {
    # softer than the scene: gentler contrast, colours that fade a little in the lights
    "natural": dict(contrast=0.90, saturation=0.96, warm=None, fade=0.45),
    # more contrast and colour above the shadows, which stay the scene's own
    "punchy": dict(contrast=1.16, saturation=1.16, warm=None, fade=0.0),
    # punchy, in a warm light
    "golden": dict(contrast=1.13, saturation=1.12, warm=(1.07, 1.00, 0.88), fade=0.0),
}


def grade(x: torch.Tensor, contrast: float, saturation: float, warm) -> torch.Tensor:
    """The look, on the scene relative to white and BEFORE the highlights are rolled off
    (so that what the contrast pushes up is rolled off with the rest). Each step is
    weighted by :func:`shadow_weight`: the dark tones are not graded."""
    top = brightest(x)
    weight = shadow_weight(top)
    if contrast != 1.0:                      # around mid-grey, as one gain: hue kept
        gain = (top.clamp(min=1e-9) / GREY) ** (contrast - 1.0)
        x = x * (1.0 + weight * (gain - 1.0))
    if saturation != 1.0:
        grey = luminance(x)
        x = (grey + (1.0 + weight * (saturation - 1.0)) * (x - grey)).clamp(min=0.0)
    if warm is not None:                     # a tint that keeps the luminance
        tint = torch.tensor(warm, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        tint = tint / (0.2126 * warm[0] + 0.7152 * warm[1] + 0.0722 * warm[2])
        x = x * (1.0 + weight * (tint - 1.0))
    return x


def fade_highlights(y: torch.Tensor, amount: float) -> torch.Tensor:
    """Let colours fade towards white in the brightest part of the picture, as film does."""
    if amount <= 0.0:
        return y
    top = brightest(y)
    t = ((top - 0.6) / 0.4).clamp(0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t) * amount
    return y + t * (top - y)

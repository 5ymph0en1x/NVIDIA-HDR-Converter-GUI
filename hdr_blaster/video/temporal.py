"""Temporal stabilisation: from per-frame measurements to flicker-free parameters.

The first pass of a conversion records, for every frame, what the engine
measured on it (a :class:`RenderParams`). Used as they are, those values change
a little from one frame to the next and the picture flickers. This module turns
the sequence into one that moves smoothly inside a shot and jumps only at cuts.

Each recorded key follows one rule, by nature:

``log_level``   multiplicative quantities (exposure, keys): smoothed in the log domain
``level``       additive quantities (amounts, weights, the neural grid): smoothed
``upper``       maxima used to normalise: sliding maximum, then smoothed, so the
                value never drops below what any nearby frame needs
``lower``       minima used to normalise: the mirror image
``vote``        yes/no branch choices: majority over the shot

Smoothing has no lag (both passes exist, so the future is known) and works in
seconds, because recordings have an irregular frame rate.
"""
import re
from bisect import bisect_left, bisect_right
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from ..engine import params as _params_module
from ..engine.params import RenderParams

KEY_POLICY: Dict[str, str] = {
    "scene.peak": "upper",
    "lut3d.weights": "level",
    "neural.grid": "level",
    "edge.in_max": "upper",
    "edge.mag_max": "upper",
    "edge.out_max": "upper",
    "edge.in_min": "lower",
    "edge.out_min": "lower",
    "adaptive.extreme_highlights": "vote",
    "adaptive.shadow_recovery": "vote",
}

SIGNATURE_SIZE = (9, 16)          # rows, columns of the luma thumbnail
CUT_THRESHOLD = 0.06              # mean absolute change of the thumbnail, in PQ code value (0..1)
CUT_RATIO = 4.0                   # ...and this many times the recent frame-to-frame change
_LOG_FLOOR = 1e-12
_LARGE_VALUE = 64                 # values bigger than this many numbers are smoothed with a running sum
_LOOK_FRAMES = 4                  # frames on each side used to decide whether a change lasts


def documented_keys() -> set:
    """The keys listed in the docstring of ``engine/params.py``."""
    return set(re.findall(r"``([a-z0-9_]+\.[a-z0-9_.]+)``", _params_module.__doc__))


def frame_signature(y: torch.Tensor, bit_depth: int = 10) -> torch.Tensor:
    """A tiny luma thumbnail of a frame, on the CPU, in PQ code value 0..1.

    PQ is close to perceptually uniform, so plain differences of code values are a
    fair measure of how much the picture changed.
    """
    plane = y.to(torch.float32)[None, None] / float((1 << bit_depth) - 1)
    return F.adaptive_avg_pool2d(plane, SIGNATURE_SIZE).flatten().cpu()


def detect_cuts(signatures: torch.Tensor, threshold: float = CUT_THRESHOLD,
                ratio: float = CUT_RATIO) -> List[int]:
    """Indices of the first frame of each shot (always starts with 0).

    A cut is a change that is

    * large in absolute terms,
    * large compared with the motion just before it (a fast pan or a fade changes
      every frame a little and is not a cut),
    * and lasting: the picture after it stays different from the picture before it.
      A muzzle flash, a lightning strike or a strobe changes everything for a frame
      and then comes back; treating those as cuts would restart the stabilisation at
      every flash, which is exactly when it is needed.
    """
    count = signatures.shape[0]
    if count == 0:
        return []
    change = [0.0] + (signatures[1:] - signatures[:-1]).abs().mean(dim=1).tolist()
    cuts = [0]
    for i in range(1, count):
        value = change[i]
        if value <= threshold:
            continue
        recent = [change[j] for j in range(max(1, i - 10), i) if j not in cuts]
        usual = sorted(recent)[len(recent) // 2] if recent else 0.0
        if value <= ratio * usual:
            continue
        # The typical look on each side: an element-wise median over an even number of
        # frames ignores a single flash and sees a strobe as the same on both sides.
        before = signatures[max(cuts[-1], i - _LOOK_FRAMES):i].median(dim=0).values
        after = signatures[i:i + _LOOK_FRAMES].median(dim=0).values
        if float((after - before).abs().mean()) > threshold:
            cuts.append(i)
    return cuts


def _window_bounds(times: Sequence[float], half_width: float):
    """For each time, the index range [lo, hi) of the samples within +/- half_width."""
    los = [bisect_left(times, t - half_width) for t in times]
    his = [bisect_right(times, t + half_width) for t in times]
    return los, his


def _box_mean(values: torch.Tensor, present: torch.Tensor, los, his) -> torch.Tensor:
    """Mean over each window of the samples that exist. ``values`` is (N, ...) float64,
    ``present`` (N,) bool. A window that holds no sample takes the result of the nearest
    window that does, so the value arrives without a jump (the shot-wide mean used
    before made the picture step by a factor of 2.6 in one frame)."""
    shape = (-1,) + (1,) * (values.dim() - 1)
    mask = present.to(values.dtype).view(shape)
    zero = torch.zeros_like(values[:1])
    sums = torch.cat([zero, (values * mask).cumsum(dim=0)])
    counts = torch.cat([torch.zeros(1, dtype=values.dtype), present.to(values.dtype).cumsum(dim=0)])
    lo = torch.tensor(los)
    hi = torch.tensor(his)
    total = sums[hi] - sums[lo]
    n = (counts[hi] - counts[lo]).view(shape)
    out = total / n.clamp(min=1.0)
    return out[_nearest_filled((n.flatten() > 0).tolist())]


def _nearest_filled(filled) -> torch.Tensor:
    """For each position, the index of the nearest position that is filled (itself if it is)."""
    have = [i for i, ok in enumerate(filled) if ok]
    if not have:
        return torch.arange(len(filled))
    out = []
    for i in range(len(filled)):
        k = bisect_left(have, i)
        left = have[max(k - 1, 0)]
        right = have[min(k, len(have) - 1)]
        out.append(left if i - left <= right - i else right)
    return torch.tensor(out)


def _running_mean(samples: list, los, his) -> list:
    """Windowed mean for LARGE values (the neural grid is 24 576 numbers per frame): a
    running sum that adds and removes one frame at a time, so the memory used is the
    frames themselves, not several float64 copies of the whole sequence. Frames that did
    not record the key are skipped; an empty window takes the nearest result."""
    template = next(s for s in samples if s is not None)
    total = torch.zeros_like(template, dtype=torch.float64, device="cpu")
    count, lo, hi, out = 0, 0, 0, []
    for want_lo, want_hi in zip(los, his):
        while hi < want_hi:
            if samples[hi] is not None:
                total += samples[hi].detach().to("cpu", torch.float64)
                count += 1
            hi += 1
        while lo < want_lo:
            if samples[lo] is not None:
                total -= samples[lo].detach().to("cpu", torch.float64)
                count -= 1
            lo += 1
        out.append((total / count).to(template.device, template.dtype) if count else None)
    nearest = _nearest_filled([o is not None for o in out]).tolist()
    return [out[j] for j in nearest]


def _sliding_extreme(values: torch.Tensor, present: torch.Tensor, los, his, upper: bool) -> torch.Tensor:
    """Sliding maximum (or minimum) of the samples that exist, per window."""
    fill = float("-inf") if upper else float("inf")
    shape = (-1,) + (1,) * (values.dim() - 1)
    masked = torch.where(present.view(shape), values, torch.full_like(values, fill))
    reduce = torch.amax if upper else torch.amin
    overall = reduce(masked, dim=0)
    out = torch.empty_like(values)
    for i, (lo, hi) in enumerate(zip(los, his)):
        out[i] = reduce(masked[lo:hi], dim=0)
    empty = torch.isinf(out)
    return torch.where(empty, overall.expand_as(out), out)


def _stabilise_key(policy: str, samples: list, times: Sequence[float], window_s: float) -> list:
    """One key over one shot. ``samples`` holds a tensor/bool per frame, or None when the
    frame did not record the key. Returns one value per frame."""
    present = torch.tensor([s is not None for s in samples])
    template = next(s for s in samples if s is not None)

    if policy == "vote":
        votes = [bool(s) for s in samples if s is not None]
        yes = sum(votes)
        winner = yes * 2 > len(votes) or (yes * 2 == len(votes) and votes[0])
        return [winner] * len(samples)

    if policy == "level" and template.numel() > _LARGE_VALUE:
        los, his = _window_bounds(times, window_s)
        return _running_mean(samples, los, his)

    zero = torch.zeros_like(template, dtype=torch.float64, device="cpu")
    values = torch.stack([zero if s is None else s.detach().to("cpu", torch.float64) for s in samples])
    los, his = _window_bounds(times, window_s)
    if policy == "log_level":
        smooth = torch.exp(_box_mean(torch.log(values.clamp(min=_LOG_FLOOR)), present, los, his))
    elif policy == "level":
        smooth = _box_mean(values, present, los, his)
    elif policy in ("upper", "lower"):
        envelope = _sliding_extreme(values, present, los, his, upper=policy == "upper")
        # Averaging the envelope over half the window keeps it on the safe side of every
        # frame: each envelope value inside that half-window already covers the frame.
        inner_lo, inner_hi = _window_bounds(times, window_s / 2.0)
        smooth = _box_mean(envelope, torch.ones_like(present), inner_lo, inner_hi)
    else:
        raise ValueError(f"unknown policy '{policy}'")
    return [smooth[i].to(template.device, template.dtype) for i in range(len(samples))]


def stabilise(params: Sequence[RenderParams], times_s: Sequence[float], cuts: Sequence[int],
              window_s: float = 1.0) -> List[RenderParams]:
    """Per-frame recorded parameters -> frozen parameters that are steady inside each shot.

    ``times_s`` are the presentation times of the frames, ``cuts`` the index of the first
    frame of each shot (as :func:`detect_cuts` returns them). The inputs are not modified.
    """
    if len(params) != len(times_s):
        raise ValueError("one time per frame is required")
    if not params:
        return []
    methods = {p.method for p in params}
    if len(methods) != 1:
        raise ValueError(f"a video is rendered with one operator, got {sorted(methods)}")
    method = methods.pop()
    starts = sorted(set(cuts) | {0})
    if starts[-1] >= len(params) or starts[0] < 0:
        raise ValueError("cut index outside the video")

    out: List[Dict] = [dict() for _ in params]
    for start, end in zip(starts, starts[1:] + [len(params)]):
        shot = params[start:end]
        times = list(times_s[start:end])
        keys = []
        for p in shot:
            keys.extend(k for k in p.stats if k not in keys)
        for key in keys:
            policy = KEY_POLICY.get(key)
            if policy is None:
                raise KeyError(f"no temporal rule for render parameter '{key}': add it to KEY_POLICY")
            smoothed = _stabilise_key(policy, [p.stats.get(key) for p in shot], times, window_s)
            for offset, value in enumerate(smoothed):
                out[start + offset][key] = value
    return [RenderParams(method=method, stats=stats, frozen=True) for stats in out]

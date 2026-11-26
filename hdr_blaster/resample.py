"""Bringing a picture to the size it is shown at: once, with a Spline36 kernel.

A picture resampled carelessly shows things that are not in it. Averaging over uneven
groups of pixels (what an "area" reduction does when the ratio is not a whole number)
turns thin lines into staircases; a bilinear enlargement blurs them on top of that. The
kernel used here is the one video renderers use for the same job: Spline36, three pixels
wide on each side, widened when the picture shrinks so that detail finer than the new
grid is filtered out before the picture is sampled.

The kernel has small negative lobes (that is what keeps edges sharp). Left alone they
draw a thin halo along every hard edge: 7 % of the step, a dark line next to each light.
So each new pixel is held between the darkest and the brightest of the pixels it sits
on (the anti-ringing of video renderers): the edge stays sharp, the halo cannot exist,
and a result never leaves the range of the picture.
"""
import math
from typing import Tuple

import torch


def spline36(x: torch.Tensor) -> torch.Tensor:
    """The Spline36 kernel: 1 at 0, 0 at every other whole number, nothing beyond 3."""
    x = x.abs()
    near = ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0
    t = x - 1.0
    middle = ((-6.0 / 11.0 * t + 270.0 / 209.0) * t - 156.0 / 209.0) * t
    t = x - 2.0
    far = ((1.0 / 11.0 * t - 45.0 / 209.0) * t + 26.0 / 209.0) * t
    return torch.where(x < 1.0, near, torch.where(x < 2.0, middle, torch.where(x < 3.0, far, torch.zeros_like(x))))


def _taps(n_in: int, n_out: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each pixel of the new line: which pixels of the old one it is made of, how much
    of each, and which of them it sits on. (n_out, taps) indices, weights adding up to 1,
    and flags."""
    scale = n_out / n_in
    stretch = max(1.0, 1.0 / scale)                 # shrinking: a wider kernel, which filters
    support = 3.0 * stretch
    centre = (torch.arange(n_out, dtype=torch.float64, device=device) + 0.5) / scale - 0.5
    first = torch.floor(centre - support).long() + 1
    count = int(math.ceil(2.0 * support)) + 1
    index = first[:, None] + torch.arange(count, device=device)[None, :]
    distance = (index.double() - centre[:, None]) / stretch
    weight = spline36(distance)
    weight = weight / weight.sum(dim=1, keepdim=True)
    under = distance.abs() < 1.0                    # the kernel's main lobe: the pixels it sits on
    return index.clamp(0, n_in - 1), weight.float(), under   # past the border, the border pixel repeats


def _along(image: torch.Tensor, n_out: int, dim: int, antiring: bool = True) -> torch.Tensor:
    n_in = image.shape[dim]
    if n_in == n_out:
        return image
    index, weight, under = _taps(n_in, n_out, image.device)
    shape = [1] * image.dim()
    shape[dim] = n_out
    out = darkest = brightest = None
    for k in range(index.shape[1]):
        w = weight[:, k]
        if not bool(w.any()):
            continue
        taken = image.index_select(dim, index[:, k])
        part = taken * w.view(shape).to(image.dtype)
        out = part if out is None else out + part
        if antiring and bool(under[:, k].any()):
            here = under[:, k].view(shape)
            low = torch.where(here, taken, torch.full_like(taken, float("inf")))
            high = torch.where(here, taken, torch.full_like(taken, float("-inf")))
            darkest = low if darkest is None else torch.minimum(darkest, low)
            brightest = high if brightest is None else torch.maximum(brightest, high)
    return torch.minimum(torch.maximum(out, darkest), brightest) if antiring else out


def resize(image: torch.Tensor, height: int, width: int, antiring: bool = True) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, height, width). The same picture object when nothing changes."""
    if image.shape[-2:] == (height, width):
        return image
    # the direction that shrinks most first: less to compute for the other
    if width * image.shape[-2] <= height * image.shape[-1]:
        return _along(_along(image, width, 3, antiring), height, 2, antiring)
    return _along(_along(image, height, 2, antiring), width, 3, antiring)


def fit_size(box_width: int, box_height: int, width: int, height: int) -> Tuple[int, int]:
    """(width, height) of the largest picture of this shape that fits the box. A picture
    made for a box is recognised by it: ``fit_size(box, *fit_size(box, w, h))`` is itself."""
    scale = min(box_width / width, box_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))

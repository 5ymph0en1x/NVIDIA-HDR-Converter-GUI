"""The size a picture is shown at.

Apart from :mod:`resample`, which makes the picture at that size: the window needs this
arithmetic before PyTorch is loaded.
"""
from typing import Tuple


def fit_size(box_width: int, box_height: int, width: int, height: int) -> Tuple[int, int]:
    """(width, height) of the largest picture of this shape that fits the box. A picture
    made for a box is recognised by it: ``fit_size(box, *fit_size(box, w, h))`` is itself."""
    scale = min(box_width / width, box_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))

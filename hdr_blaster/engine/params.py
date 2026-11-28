"""Render parameters: every decision the engine takes by measuring an image.

The engine measures things on each image (median luminance, log-average,
maxima, LUT weights, ...) and uses them straight away. For a photo that is
fine. For a video, measuring every frame independently makes the result
flicker. So each measurement goes through :func:`measure`:

* while **recording**, the value is computed and stored under a key;
* while **replaying** (``frozen``), the stored value is returned and nothing is
  measured.

A photo is rendered in recording mode, in one pass, exactly as before. A video
records on analysis frames, stabilises the values over time, then replays them.

Values are kept as tensors (or plain bools for branch decisions), never
converted to Python floats: that would change the arithmetic downstream.

Keys
----
``scene.peak``                    where the picture's highlights end, relative to white (after grading for looks and their anchored neural model)
``adaptive.extreme_highlights``, ``adaptive.shadow_recovery``   branch choices (bool)
``lut3d.weights``                 blend weights of the three basis colour LUTs
``edge.in_min``, ``edge.in_max``, ``edge.mag_max``, ``edge.out_min``, ``edge.out_max``
``neural.grid``                   bilateral grid of the learned tone mapper, (B, 12, gd, gh, gw)
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


class MissingRenderParam(KeyError):
    """A replay asked for a value that the analysis never recorded."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


@dataclass
class RenderParams:
    method: str
    stats: Dict[str, Any] = field(default_factory=dict)
    frozen: bool = False

    def measure(self, key: str, compute: Callable[[], Any]) -> Any:
        if self.frozen:
            try:
                return self.stats[key]
            except KeyError:
                raise MissingRenderParam(
                    f"render parameter '{key}' was not recorded by the analysis "
                    f"(method '{self.method}'): analyse with the same settings you render with"
                ) from None
        value = compute()
        self.stats[key] = value
        return value

    def freeze(self) -> "RenderParams":
        """A read-only copy: rendering with it measures nothing."""
        return RenderParams(method=self.method, stats=dict(self.stats), frozen=True)


def measure(params: Optional[RenderParams], key: str, compute: Callable[[], Any]) -> Any:
    """Measurement point. Without params, simply compute (original behaviour)."""
    if params is None:
        return compute()
    return params.measure(key, compute)

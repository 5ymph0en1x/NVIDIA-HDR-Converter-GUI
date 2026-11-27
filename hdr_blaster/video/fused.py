"""The stages of a video frame in a handful of operations instead of dozens.

A frame of a recording goes through four stages on the graphics card: its planes become
the scene's light, the light is rendered (tone mapping, sRGB encoding), and the picture
becomes 8-bit planes again. Written with PyTorch's own operations, that is about 180 of
them, each one a pass over the whole picture ordered separately. Measured on a 2560x1440
recording, ordering them took more time than computing them: 3.8 ms of the 6.4 ms a frame
cost, while the card waited for the next order.

Here each stage is one operation compiled for the card (two for the stages that look at
neighbouring samples), with the SAME arithmetic: every product, sum, power and rounding
of the plain stage, in the same order, in 32 bits. The result is the same picture, bit
for bit, not a close one. Three things make that true, each found by measuring against
PyTorch:

* the compiler must not merge ``a * b + c`` into one fused multiply-add (it rounds once
  instead of twice): every operation is written with the card's ``__f..._rn`` forms, which
  are never merged. Where PyTorch's own compiled code does use a fused multiply-add
  (enlarging the chroma, the 3x3 colour matrix), the same one is written here;
* PyTorch divides by a constant by multiplying by its reciprocal, taken in 64 bits and
  rounded to 32;
* constants are written as the 32-bit numbers PyTorch turns them into.

None of this is trusted blindly. A stage is used only after it has given, on the first
frame it meets and on pictures made to be difficult, exactly the bits the plain stage
gives (:meth:`Chain._checked`); otherwise the plain stage goes on being used, and the log
says so. Another card, driver or PyTorch that rounds differently therefore costs speed,
never a different picture. The plain stages remain the reference: the tests compare the
two, and ``reference=True`` conversions never come here.

Only the graphics card is concerned (the operations are compiled for it by PyTorch's
``jiterator``); on the processor the plain stages are used.
"""
import logging
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from ..engine import renderings as R
from ..engine.constants import ENGINE_UNIT_NITS
from . import colour


def _f32(value: float) -> str:
    """``value`` as PyTorch uses it next to a 32-bit picture: the nearest 32-bit number,
    written so that the compiler reads exactly that number."""
    text = f"{float(np.float32(value)):.9g}"
    return (text if any(mark in text for mark in ".e") else text + ".0") + "f"


def _inverse(value: float) -> str:
    """What PyTorch multiplies by when asked to divide by the constant ``value``."""
    return _f32(1.0 / value)


# ------------------------------------------------------ planes -> the scene's light
# Chroma is enlarged in two steps, as in colour._upsample_chroma_left: between rows first
# (bilinear, whose compiled code is fma(w0, a, w1 * b)), then between columns.
_CHROMA_ROWS = """
template <typename T> T chroma_rows(T a, T b, T w0, T w1, T offset, T scale) {
    float A = __fmul_rn(__fsub_rn((float)a, (float)offset), (float)scale);
    float B = __fmul_rn(__fsub_rn((float)b, (float)offset), (float)scale);
    return (T)__fmaf_rn((float)w0, A, __fmul_rn((float)w1, B));
}"""


def _light_code() -> str:
    kr, kb = colour._BT2020
    kg = 1.0 - kr - kb
    m = colour._BT2020_TO_BT709

    def eotf(name: str) -> str:              # colour.pq_eotf
        return (f"    float p{name} = powf(fminf(fmaxf({name}, 0.0f), 1.0f), {_f32(1.0 / colour._M2)});\n"
                f"    float n{name} = __fmul_rn({_f32(colour.PQ_PEAK_NITS)}, powf(__fdiv_rn(fmaxf(__fsub_rn(p{name}, "
                f"{_f32(colour._C1)}), 0.0f), __fsub_rn({_f32(colour._C2)}, __fmul_rn({_f32(colour._C3)}, p{name}))), "
                f"{_f32(1.0 / colour._M1)}));\n")

    def mixed(row: int) -> str:
        # The 3x3 matrix is a matrix product in the plain stage, and the library that
        # computes it adds the first two terms with a fused multiply-add, then the third.
        return (f"fmaxf(__fmul_rn(__fadd_rn(__fmaf_rn({_f32(m[row][1])}, ng, __fmul_rn({_f32(m[row][0])}, nr)), "
                f"__fmul_rn({_f32(m[row][2])}, nb)), {_inverse(colour.SCRGB_WHITE_NITS)}), 0.0f)")

    return f"""
template <typename T> void scene_light(T y, T cb_left, T cb_right, T cr_left, T cr_right, T y_offset, T y_scale,
                                       T& red, T& green, T& blue) {{
    float yp = __fmul_rn(__fsub_rn((float)y, (float)y_offset), (float)y_scale);
    float cb = __fmul_rn(0.5f, __fadd_rn((float)cb_left, (float)cb_right));
    float cr = __fmul_rn(0.5f, __fadd_rn((float)cr_left, (float)cr_right));
    float r = __fadd_rn(yp, __fmul_rn({_f32(2.0 * (1.0 - kr))}, cr));
    float b = __fadd_rn(yp, __fmul_rn({_f32(2.0 * (1.0 - kb))}, cb));
    float g = __fmul_rn(__fsub_rn(__fsub_rn(yp, __fmul_rn({_f32(kr)}, r)), __fmul_rn({_f32(kb)}, b)), {_inverse(kg)});
{eotf("r")}{eotf("g")}{eotf("b")}    red = (T){mixed(0)};
    green = (T){mixed(1)};
    blue = (T){mixed(2)};
}}"""


# ------------------------------------------- the scene, rendered Faithful, as sRGB
def _faithful_code() -> str:
    def encode(name: str, nits: str) -> str:          # renderings.pq_encode
        return (f"    float y_{name} = powf(__fmul_rn(fminf(fmaxf({nits}, 0.0f), 10000.0f), {_inverse(10000.0)}), "
                f"{_f32(R._PQ_M1)});\n"
                f"    float {name} = powf(__fdiv_rn(__fadd_rn({_f32(R._PQ_C1)}, __fmul_rn({_f32(R._PQ_C2)}, y_{name})), "
                f"__fadd_rn(1.0f, __fmul_rn({_f32(R._PQ_C3)}, y_{name}))), {_f32(R._PQ_M2)});\n")

    def srgb(name: str) -> str:                       # the loader's linear_to_srgb, on (x * gain).clamp(0, 1)
        return (f"    float l{name} = fminf(fmaxf(fminf(fmaxf(__fmul_rn(x{name}, gain), 0.0f), 1.0f), 0.0f), 1.0f);\n"
                f"    float s{name} = l{name} <= {_f32(0.0031308)} ? __fmul_rn({_f32(12.92)}, l{name}) : "
                f"__fsub_rn(__fmul_rn({_f32(1 + 0.055)}, powf(l{name}, {_f32(1.0 / 2.4)})), {_f32(0.055)});\n")

    return f"""
template <typename T> void faithful_srgb(T r, T g, T b, T top_code, T target, T knee, T rest, T span,
                                         T relative, T white, T per_white, T& red, T& green, T& blue) {{
    float xr = __fmul_rn(fmaxf((float)r, 0.0f), (float)relative);
    float xg = __fmul_rn(fmaxf((float)g, 0.0f), (float)relative);
    float xb = __fmul_rn(fmaxf((float)b, 0.0f), (float)relative);
    float top = fmaxf(fmaxf(xr, xg), xb);
    float nits = __fmul_rn(top, (float)white);
{encode("code", "nits")}    float position = __fdiv_rn(code, (float)top_code);
    float t = fminf(fmaxf(__fdiv_rn(__fsub_rn(fminf(position, 1.0f), (float)knee), (float)span), 0.0f), 1.0f);
    float t2 = __fmul_rn(t, t);
    float t3 = __fmul_rn(t2, t);
    float leave = __fmul_rn(__fadd_rn(__fsub_rn(__fmul_rn(2.0f, t3), __fmul_rn(3.0f, t2)), 1.0f), (float)knee);
    float bend = __fmul_rn(__fadd_rn(__fsub_rn(t3, __fmul_rn(2.0f, t2)), t), (float)rest);
    float land = __fmul_rn(__fadd_rn(__fmul_rn(-2.0f, t3), __fmul_rn(3.0f, t2)), (float)target);
    float shoulder = __fmul_rn(__fadd_rn(__fadd_rn(leave, bend), land), (float)top_code);
    float p = powf(fminf(fmaxf(shoulder, 0.0f), 1.0f), {_f32(1.0 / R._PQ_M2)});
    float decoded = __fmul_rn(10000.0f, powf(__fdiv_rn(fmaxf(__fsub_rn(p, {_f32(R._PQ_C1)}), 0.0f),
                              __fsub_rn({_f32(R._PQ_C2)}, __fmul_rn({_f32(R._PQ_C3)}, p))), {_f32(1.0 / R._PQ_M1)}));
    float mapped = __fmul_rn(position <= (float)knee ? nits : fminf(decoded, (float)white), (float)per_white);
    float gain = top > {_f32(1e-9)} ? __fdiv_rn(mapped, fmaxf(top, {_f32(1e-9)})) : 1.0f;
{srgb("r")}{srgb("g")}{srgb("b")}    red = (T)fminf(fmaxf(sr, 0.0f), 1.0f);
    green = (T)fminf(fmaxf(sg, 0.0f), 1.0f);
    blue = (T)fminf(fmaxf(sb, 0.0f), 1.0f);
}}"""


# --------------------------------------------- the finished picture -> 8-bit planes
def _luma_code() -> str:
    kr, kb = colour._BT709
    kg = 1.0 - kr - kb
    return f"""
template <typename T> void luma_and_differences(T r, T g, T b, T& y, T& cb, T& cr) {{
    float R = fminf(fmaxf((float)r, 0.0f), 1.0f);
    float G = fminf(fmaxf((float)g, 0.0f), 1.0f);
    float B = fminf(fmaxf((float)b, 0.0f), 1.0f);
    float yp = __fadd_rn(__fadd_rn(__fmul_rn({_f32(kr)}, R), __fmul_rn({_f32(kg)}, G)), __fmul_rn({_f32(kb)}, B));
    y = (T)fminf(fmaxf(nearbyintf(__fadd_rn(16.0f, __fmul_rn(219.0f, yp))), 0.0f), 255.0f);
    cb = (T)__fmul_rn(__fsub_rn(B, yp), {_inverse(2.0 * (1.0 - kb))});
    cr = (T)__fmul_rn(__fsub_rn(R, yp), {_inverse(2.0 * (1.0 - kr))});
}}"""


# [1, 2, 1] / 4 across, centred on the even columns, then the two rows averaged; quantised.
_CHROMA_SAMPLE = """
template <typename T> T chroma_sample(T a0, T a1, T a2, T b0, T b1, T b2) {
    float top = __fadd_rn(__fadd_rn(__fmul_rn(0.25f, (float)a0), __fmul_rn(0.5f, (float)a1)), __fmul_rn(0.25f, (float)a2));
    float low = __fadd_rn(__fadd_rn(__fmul_rn(0.25f, (float)b0), __fmul_rn(0.5f, (float)b1)), __fmul_rn(0.25f, (float)b2));
    float c = __fmul_rn(0.5f, __fadd_rn(top, low));
    return (T)fminf(fmaxf(nearbyintf(__fadd_rn(128.0f, __fmul_rn(224.0f, c))), 0.0f), 255.0f);
}"""


# The plain stage mixes the three colours with a matrix product, and the library behind it
# (cuBLAS) does not add the three terms the same way for every size: plain sums up to some
# 8 000 pixels, a chain of fused multiply-adds up to 58 184, and from 58 185 pixels up to
# 7680x4320 at least, the form written in _light_code. Measured on an RTX 4090 with
# PyTorch 2.11 (CUDA 13.0). Smaller pictures keep the plain stage: they are quick anyway.
_MATRIX_PIXELS = 65536


def _identical(a, b) -> bool:
    """The same tensors (or tuples of tensors): shape, type and every bit, the sign of a zero included."""
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        a, b = a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    return bool(torch.equal(a, b))


class Chain:
    """The fused stages for one graphics card. Every method gives what its plain twin gives."""

    def __init__(self, device):
        from torch.cuda.jiterator import _create_jit_fn, _create_multi_output_jit_fn

        self.device = torch.device(device)
        self._chroma_rows = _create_jit_fn(_CHROMA_ROWS, offset=0.0, scale=1.0)
        self._light = _create_multi_output_jit_fn(_light_code(), num_outputs=3, y_offset=0.0, y_scale=1.0)
        self._faithful = _create_multi_output_jit_fn(_faithful_code(), num_outputs=3,
                                                     relative=1.0, white=1.0, per_white=1.0)
        self._luma = _create_multi_output_jit_fn(_luma_code(), num_outputs=3)
        self._chroma_sample = _create_jit_fn(_CHROMA_SAMPLE)
        self._verdicts = {}                 # (stage, kind of input) -> the fused stage may be used
        self._rows = {}                     # chroma height -> what enlarging its rows needs
        self._columns = {}                  # chroma width -> which samples stand left and right of each pixel

    # ------------------------------------------------------------ the check
    def _checked(self, stage: str, kind, plain: Callable, fused: Callable, hard: Callable, args: tuple):
        """``fused(*args)`` once the fused stage has been seen to give, on a first input of
        this kind and on ``hard(*args)`` (inputs made to be difficult), the very bits of
        the plain stage; ``plain(*args)`` until then, and for good if it does not."""
        verdict = self._verdicts.get((stage, kind))
        if verdict:
            return fused(*args)
        expected = plain(*args)
        if verdict is None:
            try:
                agree = _identical(expected, fused(*args)) and all(
                    _identical(plain(*case), fused(*case)) for case in hard(*args))
                reason = "it does not give the same picture on this machine"
            except Exception as exc:        # no compiler for the card, an unknown card...
                agree, reason = False, f"{type(exc).__name__}: {str(exc)[:200]}"
            self._verdicts[(stage, kind)] = agree
            if not agree:
                logging.warning(f"Fused stage '{stage}' is not used ({reason}): the plain stage goes on, more slowly.")
            else:
                logging.info(f"Fused stage '{stage}' checked against the plain one: the same bits.")
        return expected

    # ------------------------------------------------ planes -> scene light
    def _rows_of(self, half: int):
        if half not in self._rows:
            rows = torch.arange(2 * half, device=self.device)
            source = ((rows.float() + 0.5) * 0.5 - 0.5).clamp(min=0.0)       # where each row falls among the chroma rows
            first = source.floor().long()
            second = (first + 1).clamp(max=half - 1)
            w1 = (source - first.float()).view(1, -1, 1)
            self._rows[half] = (first, second, 1.0 - w1, w1)
        return self._rows[half]

    def _columns_of(self, half: int):
        if half not in self._columns:
            # pixel 2i: sample i twice (their mean is the sample itself); pixel 2i+1: samples i and i+1,
            # the last one repeated at the edge
            index = torch.arange(2 * half + 1, device=self.device) // 2
            self._columns[half] = index.clamp(max=half - 1)
        return self._columns[half]

    def _yuv_to_scrgb(self, y, u, v, fmt):
        height, width = y.shape[-2:]
        top = float((1 << fmt.bit_depth) - 1)
        unit = float(1 << (fmt.bit_depth - 8))
        if fmt.full_range:
            y_offset, y_scale, c_scale = 0.0, 1.0 / top, 1.0 / top
        else:
            y_offset, y_scale, c_scale = 16.0 * unit, 1.0 / (219.0 * unit), 1.0 / (224.0 * unit)
        first, second, w0, w1 = self._rows_of(height // 2)
        chroma = torch.stack((u, v))
        tall = self._chroma_rows(chroma.index_select(1, first), chroma.index_select(1, second), w0, w1,
                                 offset=128.0 * unit, scale=c_scale)
        wide = tall.index_select(2, self._columns_of(width // 2))
        red, green, blue = self._light(y, wide[0, :, :width], wide[0, :, 1:], wide[1, :, :width], wide[1, :, 1:],
                                       y_offset=y_offset, y_scale=y_scale)
        return torch.stack((red, green, blue)).unsqueeze(0)

    @staticmethod
    def _hard_planes(y, u, v, fmt):
        """Every code value, in every neighbourhood: noise over the whole range, then the extremes."""
        g = torch.Generator(device=y.device).manual_seed(20261127)
        top = (1 << fmt.bit_depth) - 1

        def noise(like, low, high):
            return torch.randint(low, high + 1, like.shape, device=like.device, generator=g).to(like.dtype)

        flat = [torch.full_like(y, value) for value in (0, top)]
        return [(noise(y, 0, top), noise(u, 0, top), noise(v, 0, top), fmt),
                (noise(y, 0, top), torch.full_like(u, top), torch.full_like(v, 0), fmt),
                (flat[0], noise(u, 0, top), noise(v, 0, top), fmt),
                (flat[1], torch.full_like(u, 0), torch.full_like(v, top), fmt)]

    def yuv_to_scrgb(self, y, u, v, fmt):
        """:func:`colour.yuv_to_scrgb`."""
        height, width = y.shape[-2:]
        usual = (fmt.matrix == "bt2020nc" and y.dim() == 2 and u.shape == v.shape == (height // 2, width // 2)
                 and height % 2 == 0 and width % 2 == 0 and height * width >= _MATRIX_PIXELS
                 and y.dtype == u.dtype == v.dtype == torch.int16 and y.device.type == "cuda")
        if not usual:
            return colour.yuv_to_scrgb(y, u, v, fmt)
        return self._checked("planes to scene light", (height, width, fmt), colour.yuv_to_scrgb,
                             self._yuv_to_scrgb, self._hard_planes, (y, u, v, fmt))

    # ------------------------------------------------- the Faithful rendering
    def _faithful_srgb(self, linear, peak, white_nits):
        white = float(white_nits)
        # What the curve needs of the picture as a whole, by the very operations of
        # renderings.faithful_curve: a few numbers, not worth a kernel of their own.
        white_t = torch.as_tensor(white, dtype=torch.float32, device=linear.device)
        peak_t = torch.maximum(torch.as_tensor(peak * white, dtype=torch.float32, device=linear.device), white_t)
        top_code = R.pq_encode(peak_t)
        target = R.pq_encode(white_t) / top_code
        knee = (1.5 * target - 0.5).clamp(min=0.0)
        rest = 1.0 - knee
        one = ()                            # each of them one number, whatever shape the peak came in
        red, green, blue = self._faithful(linear[0, 0], linear[0, 1], linear[0, 2], top_code.reshape(one),
                                          target.reshape(one), knee.reshape(one), rest.reshape(one),
                                          rest.clamp(min=1e-6).reshape(one), relative=ENGINE_UNIT_NITS / white,
                                          white=white, per_white=1.0 / white)
        return torch.stack((red, green, blue)).unsqueeze(0)

    @staticmethod
    def _hard_scenes(linear, peak, white_nits):
        """Scenes that reach every part of the curve, with peaks under, at and far above white."""
        g = torch.Generator(device=linear.device).manual_seed(20261127)
        relative = float(white_nits) / ENGINE_UNIT_NITS            # white, in the engine's unit
        noise = torch.rand(linear.shape, device=linear.device, generator=g)
        wide = noise * torch.rand(linear.shape, device=linear.device, generator=g) * (10000.0 / ENGINE_UNIT_NITS)
        near_black = noise * 0.02 * relative - 0.002 * relative    # the foot of the sRGB curve, and negatives
        about_white = (0.5 + noise) * relative
        cases = []
        for scene in (wide, near_black, about_white):
            for level in (0.5, 1.0, 1.0001, 3.0, 10000.0 / float(white_nits)):
                cases.append((scene, torch.as_tensor(level, dtype=peak.dtype, device=peak.device).reshape(peak.shape),
                              white_nits))
        return cases

    def render(self, plain: Callable, linear, peak, white_nits):
        """The Faithful rendering of a scene, sRGB encoded: what ``plain(linear, peak,
        white_nits)`` gives, which is how the caller says what the plain way is."""
        # (the plain way refuses a picture of 32 pixels or less a side: it is left to refuse it)
        usual = (linear.dim() == 4 and linear.shape[0] == 1 and linear.shape[1] == 3 and min(linear.shape[-2:]) > 32
                 and linear.dtype == torch.float32 and linear.device.type == "cuda"
                 and torch.is_tensor(peak) and peak.numel() == 1 and peak.device == linear.device)
        if not usual:
            return plain(linear, peak, white_nits)
        kind = (tuple(linear.shape[-2:]), float(white_nits), peak.dtype, tuple(peak.shape))
        return self._checked("Faithful rendering", kind, plain, self._faithful_srgb, self._hard_scenes,
                             (linear, peak, white_nits))

    # ---------------------------------------------- the picture -> 8-bit planes
    def _srgb_to_yuv420(self, srgb):
        y, cb, cr = self._luma(srgb[:, 0:1], srgb[:, 1:2], srgb[:, 2:3])
        both = torch.nn.functional.pad(torch.cat((cb, cr), dim=1), (1, 1, 0, 0), mode="replicate")
        top, low = both[..., 0::2, :], both[..., 1::2, :]
        chroma = self._chroma_sample(top[..., 0:-2:2], top[..., 1:-1:2], top[..., 2::2],
                                     low[..., 0:-2:2], low[..., 1:-1:2], low[..., 2::2]).to(torch.uint8)
        return y.to(torch.uint8)[0, 0], chroma[0, 0], chroma[0, 1]

    @staticmethod
    def _hard_pictures(srgb):
        g = torch.Generator(device=srgb.device).manual_seed(20261127)
        noise = torch.rand(srgb.shape, device=srgb.device, generator=g)
        steps = (torch.arange(srgb.numel(), device=srgb.device, dtype=torch.float32).reshape(srgb.shape) % 4097) / 4096
        return [(noise * 1.4 - 0.2,), (noise,), (steps,), (torch.round(noise * 255) / 255,)]

    def srgb_to_yuv420(self, srgb) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:func:`colour.srgb_to_yuv420`."""
        usual = (srgb.dim() == 4 and srgb.shape[0] == 1 and srgb.shape[1] == 3 and srgb.dtype == torch.float32
                 and srgb.shape[-2] % 2 == 0 and srgb.shape[-1] % 2 == 0 and srgb.device.type == "cuda")
        if not usual:
            return colour.srgb_to_yuv420(srgb)
        return self._checked("picture to 8-bit planes", tuple(srgb.shape[-2:]), colour.srgb_to_yuv420,
                             self._srgb_to_yuv420, self._hard_pictures, (srgb,))


_chains = {}


def chain_for(device) -> Optional[Chain]:
    """The fused stages for ``device``, or None where there are none (the processor, a
    PyTorch without the compiler for the card): the plain stages are then used."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _chains:
        try:
            _chains[key] = Chain(device)
        except Exception as exc:
            logging.info(f"No fused stages on this machine ({type(exc).__name__}: {exc}); the plain ones are used.")
            _chains[key] = None
    return _chains[key]

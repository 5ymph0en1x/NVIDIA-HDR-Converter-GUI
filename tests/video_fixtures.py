"""Reference implementations and small clips for the video tests.

The colour maths here is written independently of ``hdr_blaster.video.colour``
(numpy, float64, straight from the standards) so that the tests compare two
implementations rather than one with itself.
"""
import numpy as np

# SMPTE ST 2084 (PQ)
M1, M2 = 0.1593017578125, 78.84375
C1, C2, C3 = 0.8359375, 18.8515625, 18.6875
# BT.2020 non-constant luminance
KR, KB = 0.2627, 0.0593
KG = 1.0 - KR - KB

BT709_TO_BT2020 = np.array([
    [0.6274040, 0.3292820, 0.0433136],
    [0.0690970, 0.9195400, 0.0113612],
    [0.0163916, 0.0880132, 0.8955950],
])


def pq_encode(nits):
    y = (np.clip(np.asarray(nits, dtype=np.float64), 0.0, 10000.0) / 10000.0) ** M1
    return ((C1 + C2 * y) / (1.0 + C3 * y)) ** M2


def pq_decode(code):
    p = np.asarray(code, dtype=np.float64) ** (1.0 / M2)
    return 10000.0 * (np.maximum(p - C1, 0.0) / (C2 - C3 * p)) ** (1.0 / M1)


def quantise(yp, cb, cr, bit_depth=10, full_range=False):
    """Y'CbCr in [0,1] / [-0.5,0.5] -> integer code values."""
    top = (1 << bit_depth) - 1
    unit = 1 << (bit_depth - 8)
    if full_range:
        y, u, v = yp * top, cb * top + 128 * unit, cr * top + 128 * unit
    else:
        y, u, v = 16 * unit + 219 * unit * yp, 128 * unit + 224 * unit * cb, 128 * unit + 224 * unit * cr
    return [np.clip(np.rint(p), 0, top).astype(np.uint16) for p in (y, u, v)]


def subsample_left(c):
    """4:2:0 with 'left' chroma siting: [1,2,1]/4 horizontally at even columns, row pairs averaged."""
    padded = np.pad(c, ((0, 0), (1, 1)), mode="edge")
    horizontal = 0.25 * padded[:, 0:-2:2] + 0.5 * padded[:, 1:-1:2] + 0.25 * padded[:, 2::2]
    return 0.5 * (horizontal[0::2] + horizontal[1::2])


def nits709_to_hdr10(rgb_nits, bit_depth=10, full_range=False):
    """(H, W, 3) linear BT.709 in nits -> (y, u, v) HDR10 planes (PQ, BT.2020, 4:2:0 left-sited)."""
    rgb2020 = np.clip(np.asarray(rgb_nits, dtype=np.float64) @ BT709_TO_BT2020.T, 0.0, None)
    r, g, b = np.moveaxis(pq_encode(rgb2020), -1, 0)
    yp = KR * r + KG * g + KB * b
    cb = (b - yp) / (2.0 * (1.0 - KB))
    cr = (r - yp) / (2.0 * (1.0 - KR))
    return quantise(yp, subsample_left(cb), subsample_left(cr), bit_depth, full_range)


def flat_hdr10(rgb_nits_2020, height=8, width=8, bit_depth=10, full_range=False):
    """A flat patch given directly in BT.2020 nits -> (y, u, v) planes."""
    r, g, b = pq_encode(np.asarray(rgb_nits_2020, dtype=np.float64))
    yp = KR * r + KG * g + KB * b
    cb = (b - yp) / (2.0 * (1.0 - KB))
    cr = (r - yp) / (2.0 * (1.0 - KR))
    y, u, v = quantise(yp, cb, cr, bit_depth, full_range)
    return (np.full((height, width), y, np.uint16),
            np.full((height // 2, width // 2), u, np.uint16),
            np.full((height // 2, width // 2), v, np.uint16))

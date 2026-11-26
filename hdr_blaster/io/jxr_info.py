"""What a JXR file says about itself, read from its container before any pixel is decoded.

NVIDIA's HDR screenshots (and Windows' own) are JPEG XR files whose pixel format is a
floating-point RGBA one. That format IS the colour metadata: it means scRGB, i.e. linear
light, BT.709 primaries, 1.0 = 80 nits, values below 0 and above 1 allowed. Such files
carry no other luminance information (no mastering level, no ICC profile); none is
needed, since scRGB is absolute.

A JXR file can also hold an ordinary 8-bit picture. Treating its integers as scRGB
would give nonsense, so the pixel format is checked before a capture is accepted.
"""
import os
import struct
from dataclasses import dataclass
from typing import Optional

_TAG_PIXEL_FORMAT, _TAG_WIDTH, _TAG_HEIGHT = 0xBC01, 0xBC80, 0xBC81
_TAG_ICC, _TAG_XMP, _TAG_ALPHA_OFFSET = 0x8773, 0x02BC, 0xBCC2
# Every WIC pixel format GUID is 6FDDC324-4E03-4BFE-B185-3D77768DC9xx: the last byte names it.
_WIC_PREFIX = bytes.fromhex("24c3dd6f034efe4bb1853d77768dc9")
_FORMATS = {
    0x19: ("128-bit RGBA float", True), 0x1A: ("128-bit premultiplied RGBA float", True),
    0x1B: ("128-bit RGB float", True), 0x3A: ("64-bit RGBA half float", True),
    0x3B: ("48-bit RGB half float", True), 0x42: ("64-bit RGB half float", True),
    0x0C: ("24-bit BGR", False), 0x0D: ("24-bit RGB", False), 0x0E: ("32-bit BGR", False),
    0x0F: ("32-bit BGRA", False), 0x15: ("48-bit RGB", False), 0x16: ("64-bit RGBA", False),
    0x08: ("8-bit grey", False), 0x0B: ("16-bit grey", False),
}


class UnsupportedCapture(ValueError):
    """The file is not an HDR capture we can convert. The message is written for the user."""


@dataclass(frozen=True)
class JxrInfo:
    path: str
    width: int
    height: int
    pixel_format: str             # a readable name, or the GUID when it is not one we know
    is_float: bool                # floating point, hence scRGB
    has_alpha: bool
    has_icc_profile: bool
    has_xmp: bool


def read_jxr_info(path) -> JxrInfo:
    path = os.fspath(path)
    name = os.path.basename(path)
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) < 8 or head[:3] != b"II\xbc":
                raise UnsupportedCapture(f"{name} is not a JPEG XR file.")
            handle.seek(struct.unpack_from("<I", head, 4)[0])
            count = struct.unpack("<H", handle.read(2))[0]
            entries = handle.read(12 * count)
            tags = {}
            for i in range(count):
                tag, kind, items, value = struct.unpack_from("<HHII", entries, 12 * i)
                tags[tag] = (kind, items, value)
            guid: Optional[bytes] = None
            if _TAG_PIXEL_FORMAT in tags:
                handle.seek(tags[_TAG_PIXEL_FORMAT][2])
                guid = handle.read(16)
    except (OSError, struct.error) as exc:
        raise UnsupportedCapture(f"Could not read {name}: {exc}") from exc

    known = guid is not None and len(guid) == 16 and guid[:15] == _WIC_PREFIX
    label, is_float = _FORMATS.get(guid[15], (None, False)) if known else (None, False)
    if label is None:
        label = guid.hex() if guid else "unknown"
    return JxrInfo(path=path, width=tags.get(_TAG_WIDTH, (0, 0, 0))[2], height=tags.get(_TAG_HEIGHT, (0, 0, 0))[2],
                   pixel_format=label, is_float=is_float, has_alpha=_TAG_ALPHA_OFFSET in tags,
                   has_icc_profile=_TAG_ICC in tags, has_xmp=_TAG_XMP in tags)


def check_hdr_capture(info: JxrInfo) -> None:
    """Raise :class:`UnsupportedCapture` unless the file is a floating-point (scRGB) capture."""
    if not info.is_float:
        raise UnsupportedCapture(
            f"{os.path.basename(info.path)} is not an HDR capture (its pixels are {info.pixel_format}, "
            "not floating point). HDR screenshots from the NVIDIA app or Windows can be converted.")

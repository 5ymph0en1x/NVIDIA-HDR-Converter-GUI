"""What the screen does with an SDR window, so that an HDR picture can be shown truthfully in one.

A window like ours is SDR: its white is whatever luminance the system gives to SDR
content. On a Windows display with HDR switched on, that is the "SDR content
brightness" setting, anywhere from 80 to 480 nits. To show the tones of an HDR picture
at the luminance they have in the game, on the same screen, sRGB white must stand for
exactly that many nits. This module asks Windows for it.

Without HDR on the display (or off Windows) the screen's white is not known; the HDR
reference white of ITU-R BT.2408, 203 nits, is used instead.
"""
import ctypes
import sys
from typing import Tuple

REFERENCE_WHITE_NITS = 203.0          # ITU-R BT.2408: reference white of HDR programmes
_QDC_ONLY_ACTIVE_PATHS = 0x2
_GET_ADVANCED_COLOR_INFO = 9
_GET_SDR_WHITE_LEVEL = 11


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _PathSource(ctypes.Structure):
    _fields_ = [("adapterId", _LUID), ("id", ctypes.c_uint32), ("modeInfoIdx", ctypes.c_uint32),
                ("statusFlags", ctypes.c_uint32)]


class _PathTarget(ctypes.Structure):
    _fields_ = [("adapterId", _LUID), ("id", ctypes.c_uint32), ("modeInfoIdx", ctypes.c_uint32),
                ("outputTechnology", ctypes.c_uint32), ("rotation", ctypes.c_uint32),
                ("scaling", ctypes.c_uint32), ("refreshNumerator", ctypes.c_uint32),
                ("refreshDenominator", ctypes.c_uint32), ("scanLineOrdering", ctypes.c_uint32),
                ("targetAvailable", ctypes.c_int32), ("statusFlags", ctypes.c_uint32)]


class _PathInfo(ctypes.Structure):
    _fields_ = [("sourceInfo", _PathSource), ("targetInfo", _PathTarget), ("flags", ctypes.c_uint32)]


class _ModeInfo(ctypes.Structure):
    _fields_ = [("infoType", ctypes.c_uint32), ("id", ctypes.c_uint32), ("adapterId", _LUID),
                ("mode", ctypes.c_byte * 48)]


class _Header(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("size", ctypes.c_uint32), ("adapterId", _LUID),
                ("id", ctypes.c_uint32)]


class _AdvancedColorInfo(ctypes.Structure):
    _fields_ = [("header", _Header), ("value", ctypes.c_uint32), ("colorEncoding", ctypes.c_uint32),
                ("bitsPerColorChannel", ctypes.c_uint32)]


class _SdrWhiteLevel(ctypes.Structure):
    _fields_ = [("header", _Header), ("SDRWhiteLevel", ctypes.c_uint32)]


class _SourceName(ctypes.Structure):
    _fields_ = [("header", _Header), ("viewGdiDeviceName", ctypes.c_wchar * 32)]


_GET_SOURCE_NAME = 1


def _windows_hdr_displays() -> list:
    """(name, SDR white in nits) of every active display that has HDR switched on. The name
    is the one Windows gives the screen (DISPLAY1, with its device prefix); see
    :func:`display_of_window`."""
    assert ctypes.sizeof(_PathInfo) == 72 and ctypes.sizeof(_ModeInfo) == 64
    user32 = ctypes.windll.user32
    paths_n, modes_n = ctypes.c_uint32(0), ctypes.c_uint32(0)
    if user32.GetDisplayConfigBufferSizes(_QDC_ONLY_ACTIVE_PATHS, ctypes.byref(paths_n), ctypes.byref(modes_n)):
        return []
    paths = (_PathInfo * paths_n.value)()
    modes = (_ModeInfo * modes_n.value)()
    if user32.QueryDisplayConfig(_QDC_ONLY_ACTIVE_PATHS, ctypes.byref(paths_n), paths,
                                 ctypes.byref(modes_n), modes, None):
        return []
    found = []
    for path in list(paths)[:paths_n.value]:
        colour = _AdvancedColorInfo()
        colour.header.type, colour.header.size = _GET_ADVANCED_COLOR_INFO, ctypes.sizeof(colour)
        colour.header.adapterId, colour.header.id = path.targetInfo.adapterId, path.targetInfo.id
        if user32.DisplayConfigGetDeviceInfo(ctypes.byref(colour)) or not colour.value & 0x2:
            continue                                          # bit 1: advanced colour (HDR) enabled
        white = _SdrWhiteLevel()
        white.header.type, white.header.size = _GET_SDR_WHITE_LEVEL, ctypes.sizeof(white)
        white.header.adapterId, white.header.id = path.targetInfo.adapterId, path.targetInfo.id
        if user32.DisplayConfigGetDeviceInfo(ctypes.byref(white)) == 0 and white.SDRWhiteLevel:
            name = _SourceName()
            name.header.type, name.header.size = _GET_SOURCE_NAME, ctypes.sizeof(name)
            name.header.adapterId, name.header.id = path.sourceInfo.adapterId, path.sourceInfo.id
            label = name.viewGdiDeviceName if user32.DisplayConfigGetDeviceInfo(ctypes.byref(name)) == 0 else ""
            found.append((label, white.SDRWhiteLevel / 1000.0 * 80.0))   # documented unit: thousandths of 80 nits
    return found


def sdr_white_nits(screen: str = "") -> Tuple[float, str]:
    """(nits that sRGB white stands for on a screen, where the figure comes from).

    ``screen``: the name of the screen the window is on (:func:`display_of_window`). The
    source is "display" when Windows reports the figure for that screen with HDR switched
    on, "reference" when the BT.2408 reference white is used instead: the screen is not in
    HDR, or this is not Windows. Without a name, or with one Windows does not know, the
    first display in HDR answers.
    """
    if sys.platform == "win32":
        try:
            displays = [(name, v) for name, v in _windows_hdr_displays() if 40.0 <= v <= 2000.0]
            named = [v for name, v in displays if screen and name == screen]
            if named:
                return named[0], "display"
            if displays and (not screen or not _is_a_display_name(screen)):
                return displays[0][1], "display"
        except Exception:                                      # an API oddity must never stop a preview
            pass
    return REFERENCE_WHITE_NITS, "reference"


class _MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint32), ("rcMonitor", ctypes.c_int32 * 4), ("rcWork", ctypes.c_int32 * 4),
                ("dwFlags", ctypes.c_uint32), ("szDevice", ctypes.c_wchar * 32)]


def display_of_window(window_id: int) -> str:
    """The name Windows gives the screen a window is (mostly) on, or "" when it cannot be
    told. ``window_id``: the window's native handle (Qt's ``winId()``)."""
    if sys.platform != "win32" or not window_id:
        return ""
    try:
        user32 = ctypes.windll.user32
        user32.MonitorFromWindow.restype = ctypes.c_void_p
        user32.MonitorFromWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        monitor = user32.MonitorFromWindow(ctypes.c_void_p(int(window_id)), 2)      # 2: the nearest screen
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(info)
        if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return info.szDevice
    except Exception:
        pass
    return ""


def _is_a_display_name(screen: str) -> bool:
    """Does this look like one of Windows' own display names? If it does and it is not among
    the displays in HDR, that screen is simply not in HDR."""
    return screen.upper().startswith("\\\\.\\DISPLAY")

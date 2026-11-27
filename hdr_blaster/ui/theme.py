"""Colours, type and the style sheet. The picture is the jewel; the window is its case."""
from PySide6.QtGui import QColor, QFont

INK = "#17161B"        # window background
VELVET = "#211F27"     # raised surfaces: bars, panel, chips
VELVET_HI = "#2C2A34"  # hovered or pressed surfaces
SEAM = "#34313D"       # hairlines
LIFT = "#3B3846"       # what is chosen or pressed: the pill of a segmented choice
RIM = "#4C4859"        # the fine edge of a raised or hovered control
DIM = "#6B6775"        # what cannot be used right now
IVORY = "#ECE9E4"      # text, and the one prominent button
MIST = "#9C98A6"       # secondary text
GREEN = "#76B900"      # the single accent: sliders, switches, progress
EMBER = "#E0795B"      # errors

FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans"]


def app_font() -> QFont:
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPixelSize(13)
    return font


def motion() -> bool:
    """May the interface animate? Not when the system is set to show no animation
    (Windows: Settings, Accessibility, Visual effects): everything then lands at once."""
    import sys

    if sys.platform != "win32":
        return True
    try:
        import ctypes

        shown = ctypes.c_int(1)
        ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(shown), 0)   # SPI_GETCLIENTAREAANIMATION
        return bool(shown.value)
    except Exception:
        return True


def colour(hex_value: str, alpha: int = 255) -> QColor:
    c = QColor(hex_value)
    c.setAlpha(alpha)
    return c


STYLE = f"""
QWidget {{ background: {INK}; color: {IVORY}; }}
QWidget#bar, QWidget#header {{ background: {VELVET}; }}
QWidget#panel {{ background: {VELVET}; border-left: 1px solid {SEAM}; }}
QWidget#queue_panel {{ background: {VELVET}; border-right: 1px solid {SEAM}; }}
QWidget#row {{ background: transparent; }}
QLabel {{ background: transparent; }}
QLabel[role="muted"] {{ color: {MIST}; }}
QLabel[role="done"] {{ color: {IVORY}; }}
QLabel[role="section"] {{ color: {MIST}; font-size: 12px; padding-top: 12px; }}
QLabel#banner {{ background: {VELVET_HI}; color: {IVORY}; padding: 9px 14px; border-left: 3px solid {EMBER}; }}
QLabel#banner[kind="info"] {{ border-left: 3px solid {GREEN}; }}

/* Buttons are pills. Every one carries a one-pixel edge, shown or not, so that nothing
   moves when the edge appears (hover, keyboard focus). */
QPushButton {{
    background: {VELVET_HI}; color: {IVORY}; border: 1px solid transparent; border-radius: 15px;
    padding: 5px 16px; min-height: 18px;
}}
QPushButton:hover {{ background: {LIFT}; border-color: {RIM}; }}
QPushButton:pressed {{ background: #454150; }}
QPushButton:disabled {{ color: {DIM}; }}
QPushButton:focus {{ outline: none; border-color: {MIST}; }}

/* the quiet button: an outline until it is needed */
QPushButton[role="ghost"] {{ background: transparent; color: {MIST}; border-color: {SEAM}; }}
QPushButton[role="ghost"]:hover {{ background: {VELVET_HI}; color: {IVORY}; border-color: {RIM}; }}
QPushButton[role="ghost"]:pressed {{ background: {LIFT}; }}
QPushButton[role="ghost"]:checked {{ background: {LIFT}; color: {IVORY}; border-color: {RIM}; }}
QPushButton[role="ghost"]:focus {{ border-color: {MIST}; }}

/* one choice of a segmented control: only its text; the pill is painted by the control */
QPushButton[role="segment"] {{
    background: transparent; color: {MIST}; border: 1px solid transparent; border-radius: 15px;
    padding: 5px 17px; font-weight: 500;
}}
QPushButton[role="segment"]:hover {{ background: transparent; color: {IVORY}; border-color: transparent; }}
QPushButton[role="segment"]:pressed {{ background: transparent; }}
QPushButton[role="segment"]:checked {{ color: {IVORY}; }}
QPushButton[role="segment"]:disabled {{ color: {DIM}; }}
QPushButton[role="segment"]:focus {{ border-color: {MIST}; }}

QSlider {{ background: transparent; }}
QSlider:horizontal {{ min-height: 22px; }}
QSlider::groove:horizontal {{ height: 4px; background: {SEAM}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {GREEN}; border-radius: 2px; }}
QSlider::sub-page:horizontal:disabled {{ background: {DIM}; }}
QSlider::handle:horizontal {{
    width: 12px; margin: -6px 0; border-radius: 8px; background: {IVORY}; border: 2px solid {VELVET};
}}
QSlider::handle:horizontal:hover {{ background: #FFFFFF; }}
QSlider::handle:horizontal:disabled {{ background: {DIM}; }}

QComboBox {{
    background: {INK}; border: 1px solid {SEAM}; border-radius: 9px; padding: 6px 30px 6px 11px; min-height: 18px;
}}
QComboBox:hover {{ border-color: {RIM}; }}
QComboBox:focus {{ border-color: {RIM}; }}
QComboBox:on {{ border-color: {MIST}; }}
QComboBox:disabled {{ color: {DIM}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; }}
QComboBox QAbstractItemView {{
    background: {VELVET}; border: 1px solid {RIM}; padding: 4px; outline: none;
    selection-background-color: {LIFT}; selection-color: {IVORY};
}}
QComboBox QAbstractItemView::item {{ min-height: 26px; padding: 2px 8px; border-radius: 6px; }}
QComboBox QAbstractItemView::item:selected {{ background: {LIFT}; color: {IVORY}; }}
QCheckBox {{ background: transparent; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {MIST}; background: {INK}; }}
QCheckBox::indicator:checked {{ background: {GREEN}; border: 1px solid {GREEN}; }}
QCheckBox:disabled {{ color: {DIM}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:horizontal {{ height: 6px; background: transparent; }}
QScrollBar:vertical {{ width: 6px; background: transparent; }}
QScrollBar::handle {{ background: {SEAM}; border-radius: 3px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QToolTip {{ background: {VELVET_HI}; color: {IVORY}; border: 1px solid {SEAM}; padding: 4px 8px; }}
"""

"""Colours, type and the style sheet. The picture is the jewel; the window is its case."""
from PySide6.QtGui import QColor, QFont

INK = "#17161B"        # window background
VELVET = "#211F27"     # raised surfaces: bars, panel, chips
VELVET_HI = "#2C2A34"  # hovered or pressed surfaces
SEAM = "#34313D"       # hairlines
IVORY = "#ECE9E4"      # text, and the one prominent button
MIST = "#9C98A6"       # secondary text
GREEN = "#76B900"      # the single accent: chosen look, progress
EMBER = "#E0795B"      # errors

FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans"]


def app_font() -> QFont:
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPixelSize(13)
    return font


def colour(hex_value: str, alpha: int = 255) -> QColor:
    c = QColor(hex_value)
    c.setAlpha(alpha)
    return c


STYLE = f"""
QWidget {{ background: {INK}; color: {IVORY}; }}
QWidget#bar, QWidget#header, QWidget#queue {{ background: {VELVET}; }}
QWidget#panel {{ background: {VELVET}; border-left: 1px solid {SEAM}; }}
QLabel {{ background: transparent; }}
QLabel[role="muted"] {{ color: {MIST}; }}
QLabel[role="done"] {{ color: {IVORY}; }}
QLabel[role="section"] {{ color: {MIST}; font-size: 12px; padding-top: 10px; }}
QLabel#banner {{ background: {VELVET_HI}; color: {IVORY}; padding: 9px 14px; border-left: 3px solid {EMBER}; }}
QLabel#banner[kind="info"] {{ border-left: 3px solid {GREEN}; }}

QPushButton {{
    background: {VELVET_HI}; color: {IVORY}; border: none; border-radius: 6px;
    padding: 7px 14px; min-height: 18px;
}}
QPushButton:hover {{ background: #383542; }}
QPushButton:pressed {{ background: #413E4C; }}
QPushButton:disabled {{ color: #6B6775; }}
QPushButton:focus {{ outline: none; border: 1px solid {MIST}; padding: 6px 13px; }}

QPushButton[role="look"] {{ border-radius: 15px; padding: 6px 16px; background: transparent; color: {MIST}; }}
QPushButton[role="look"]:hover {{ color: {IVORY}; background: {VELVET_HI}; }}
QPushButton[role="look"]:checked {{ background: {GREEN}; color: #121B02; }}
QPushButton[role="look"]:focus {{ border: 1px solid {MIST}; padding: 5px 15px; }}

QPushButton[role="ghost"] {{ background: transparent; color: {MIST}; }}
QPushButton[role="ghost"]:hover {{ color: {IVORY}; background: {VELVET_HI}; }}

QPushButton[role="chip"] {{ background: transparent; color: {MIST}; text-align: left; padding: 6px 12px; }}
QPushButton[role="chip"]:hover {{ background: {VELVET_HI}; color: {IVORY}; }}
QPushButton[role="chip"]:checked {{ background: {VELVET_HI}; color: {IVORY}; }}
QPushButton[role="chip"][state="done"] {{ color: {GREEN}; }}
QPushButton[role="chip"][state="failed"] {{ color: {EMBER}; }}

QSlider::groove:horizontal {{ height: 4px; background: {SEAM}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {GREEN}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: {IVORY}; }}
QSlider::handle:horizontal:disabled {{ background: #6B6775; }}

QComboBox {{
    background: {INK}; border: 1px solid {SEAM}; border-radius: 6px; padding: 5px 8px; min-height: 18px;
}}
QComboBox:focus {{ border: 1px solid {MIST}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {VELVET}; border: 1px solid {SEAM}; selection-background-color: {VELVET_HI}; }}
QCheckBox {{ background: transparent; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {MIST}; background: {INK}; }}
QCheckBox::indicator:checked {{ background: {GREEN}; border: 1px solid {GREEN}; }}
QCheckBox:disabled {{ color: #6B6775; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:horizontal {{ height: 6px; background: transparent; }}
QScrollBar:vertical {{ width: 6px; background: transparent; }}
QScrollBar::handle {{ background: {SEAM}; border-radius: 3px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QToolTip {{ background: {VELVET_HI}; color: {IVORY}; border: 1px solid {SEAM}; padding: 4px 8px; }}
"""

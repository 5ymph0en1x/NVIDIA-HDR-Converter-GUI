"""The picture: before and after, side by side under a divider the user drags.

The two pictures arrive at the exact size of their place, in screen pixels, and are drawn
as they are: this widget never resamples them. (It did: a picture reduced elsewhere was
stretched here to its place, and thin lines came out as staircases.) Only while the
window is being resized is the last picture stretched, until the one made for the new
size arrives.

On a screen with HDR switched on, the pictures are not painted by this widget at all:
they are presented by an HDR surface laid over it (see ``hdr_surface``), the source at
its own luminance, far above the window's white where the capture goes there. This
widget then only says where things are and paints the labels for that surface.
"""
import math
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QPushButton, QWidget

from ..resample import fit_size
from . import theme

MARGIN = 16                                    # around the picture, in the widget's units


def image_from_rgb8(array: np.ndarray) -> QImage:
    """(H, W, 3) uint8 -> QImage owning its pixels."""
    height, width, _ = array.shape
    return QImage(array.data, width, height, 3 * width, QImage.Format_RGB888).copy()


class CompareView(QWidget):
    """Empty: an invitation to drop a file. Loaded: the HDR source on the left of the
    divider, the SDR result on the right."""

    browse_requested = Signal()
    resized = Signal()
    hdr_lost = Signal()                        # the HDR surface gave up: the SDR view is back
    box_changed = Signal()                     # the room for the picture changed: another size is needed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self._before: Optional[QImage] = None
        self._after: Optional[QImage] = None
        self._previous_after: Optional[QImage] = None
        self._scaled = {}                      # id(image) -> (size, pixmap)
        self._split = 0.5
        self._dragging = False
        self._drop_hover = False
        self._note = ""                        # small text over the picture ("Preview", "Loading")
        self._source_tag = "HDR source"
        self._histogram = None                 # (3, 64) float in 0..1, or None
        self._blend = 1.0
        self._inset_right = 0                  # room kept free on the right (the advanced panel)
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(150)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.valueChanged.connect(self._on_fade)
        self._browse = QPushButton("Browse…", self)
        self._browse.clicked.connect(self.browse_requested)
        self._browse.setCursor(Qt.PointingHandCursor)
        self._hdr = None                       # the HDR surface, on a screen that takes one
        self._hdr_box = None                   # the widget that holds it
        self._hdr_pictures = None              # (source, result) for it: half floats, 1.0 = 80 nits
        self._hdr_white = 203.0                # what SDR white is on this screen, in nits
        self._overlay_key = None

    # ---------------------------------------------------------------------- HDR
    def use_hdr_surface(self) -> None:
        """Present the pictures in HDR from now on (the screen has HDR switched on)."""
        from .hdr_surface import HdrSurface

        self._hdr = HdrSurface()
        self._hdr.pointer.connect(self._on_surface_pointer)
        self._hdr.key.connect(self._on_surface_key)
        self._hdr.failed.connect(self._on_surface_failed)
        self._hdr_box = QWidget.createWindowContainer(self._hdr, self)
        self._hdr_box.setFocusPolicy(Qt.StrongFocus)
        self._hdr_box.hide()

    def hdr_ready(self) -> bool:
        """Is there an HDR surface to give pictures to?"""
        return self._hdr is not None

    def shows_hdr(self) -> bool:
        """Is the source on screen right now shown in real HDR?"""
        return self._hdr is not None and self._hdr_pictures is not None and self._before is not None

    def _sync_hdr(self) -> None:
        """Put the HDR surface over this widget (but never over the advanced panel) and tell
        it where the picture sits; or hide it when there is nothing for it to show."""
        if self._hdr is None:
            return
        if not self.shows_hdr():
            self._hdr_box.hide()
            return
        self._hdr_box.setGeometry(0, 0, max(1, self.width() - self._inset_right), self.height())
        self._hdr.drop_target = self.window()            # files dropped on the picture go to the window
        ratio = self.devicePixelRatioF()
        rect = self.picture_rect()
        place = tuple(round(v * ratio) for v in (rect.left(), rect.top(), rect.width(), rect.height()))
        ivory, ink = theme.colour(theme.IVORY), theme.colour(theme.INK)
        self._hdr.set_layout(place, self._hdr_white, 0.75 * ratio,
                             (ivory.redF(), ivory.greenF(), ivory.blueF(), 230 / 255),
                             (ink.redF(), ink.greenF(), ink.blueF()))
        self._hdr.set_split(self._split)
        self._refresh_overlay()
        self._hdr_box.show()

    def _refresh_overlay(self) -> None:
        """Paint again what the surface shows over the pictures, if any of it changed."""
        if not self.shows_hdr():
            return
        rect = self.picture_rect()
        x = rect.left() + rect.width() * self._split
        ratio = self.devicePixelRatioF()
        key = (round(rect.width() * ratio), round(rect.height() * ratio), ratio, self._source_tag, self._note,
               id(self._histogram), self._tags_visible(rect, x))
        if key == self._overlay_key:
            return
        self._overlay_key = key
        image = QImage(key[0], key[1], QImage.Format_RGBA8888_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        painter.setFont(self.font())
        painter.translate(-rect.topLeft())
        # the picture's rounded corners: outside them, the window's own ground
        whole, rounded = QPainterPath(), QPainterPath()
        whole.addRect(rect)
        rounded.addRoundedRect(rect, 6, 6)
        painter.fillPath(whole.subtracted(rounded), theme.colour(theme.INK))
        self._paint_labels(painter, rect, x)
        painter.end()
        self._hdr.set_overlay(image)
        handle = QImage(round(32 * ratio), round(32 * ratio), QImage.Format_RGBA8888_Premultiplied)
        handle.setDevicePixelRatio(ratio)
        handle.fill(Qt.transparent)
        painter = QPainter(handle)
        painter.setRenderHints(QPainter.Antialiasing)
        self._paint_handle(painter, QPointF(16, 16))
        painter.end()
        self._hdr.set_handle(handle, 32 * ratio)

    def _on_surface_pointer(self, kind: str, x: float, y: float) -> None:
        if self._before is None:
            return
        self._hdr.setCursor(Qt.SplitHCursor if self.picture_rect().contains(QPointF(x, y)) else Qt.ArrowCursor)
        if kind == "press":
            self._dragging = True
            self._split_from(x)
        elif kind == "move" and self._dragging:
            self._split_from(x)
        elif kind == "release":
            self._dragging = False
        elif kind == "double":
            self.set_split(0.5)

    def _on_surface_key(self, key: int) -> None:
        if self._before is not None and key in (Qt.Key_Left, Qt.Key_Right):
            self.set_split(self._split + (0.02 if key == Qt.Key_Right else -0.02))

    def _on_surface_failed(self, _why: str) -> None:
        """No HDR after all (the display changed, the driver refused): the SDR view again."""
        self._hdr_box.hide()
        self._hdr, self._hdr_pictures = None, None
        self.update()
        self.hdr_lost.emit()

    # ------------------------------------------------------------------ content
    def has_picture(self) -> bool:
        return self._before is not None

    def clear(self) -> None:
        self._before = self._after = self._previous_after = None
        self._hdr_pictures = None
        self._sync_hdr()
        self._scaled.clear()
        self._note, self._histogram = "", None
        self._browse.show()
        self.update()

    def set_pictures(self, before: np.ndarray, after: np.ndarray, fade: bool = True,
                     hdr=None, white_nits: float = 203.0) -> None:
        """The two pictures as 8-bit sRGB, for the SDR view. ``hdr``: the same two as half
        floats in scRGB, when there is an HDR surface to present them (``hdr_ready``);
        ``white_nits`` is then what SDR white stands for on this screen."""
        self._before = image_from_rgb8(before)
        self._set_after(image_from_rgb8(after), fade)
        self._browse.hide()
        self._hdr_pictures = hdr if self._hdr is not None else None
        self._hdr_white = float(white_nits)
        if self._hdr_pictures is not None:
            self._hdr.set_pictures(*self._hdr_pictures)
        self._sync_hdr()

    def set_after(self, after: np.ndarray, fade: bool = True) -> None:
        if self._before is not None:
            self._set_after(image_from_rgb8(after), fade)

    def _set_after(self, image: QImage, fade: bool) -> None:
        same_size = self._after is not None and self._after.size() == image.size()
        self._previous_after = self._after if (fade and same_size) else None
        self._after = image
        self._scaled.clear()
        if self._previous_after is not None:
            self._fade.stop()
            self._blend = 0.0
            self._fade.start()
        else:
            self._blend = 1.0
        self.update()

    def set_note(self, text: str) -> None:
        if text != self._note:
            self._note = text
            self.update()
            self._refresh_overlay()

    def set_source_tag(self, text: str) -> None:
        """The label of the left picture, e.g. 'HDR source, white at 480 nits'."""
        if text != self._source_tag:
            self._source_tag = text
            self.update()
            self._refresh_overlay()

    def set_histogram(self, histogram) -> None:
        self._histogram = histogram
        self.update()
        self._refresh_overlay()

    def set_drop_hover(self, hovering: bool) -> None:
        if hovering != self._drop_hover:
            self._drop_hover = hovering
            self.update()

    @property
    def split(self) -> float:
        return self._split

    def set_split(self, value: float) -> None:
        value = min(1.0, max(0.0, float(value)))
        if value != self._split:
            self._split = value
            self.update()
            if self._hdr is not None:
                self._hdr.set_split(value)
                self._refresh_overlay()

    def _on_fade(self, value) -> None:
        self._blend = float(value)
        if self._blend >= 1.0:
            self._previous_after = None
        self.update()

    # ----------------------------------------------------------------- geometry
    def picture_box(self) -> Tuple[int, int]:
        """The room there is for the picture, (width, height) in SCREEN pixels: the size a
        picture is asked for."""
        ratio = self.devicePixelRatioF()
        return (max(40, math.floor((self.width() - self._inset_right - 2 * MARGIN) * ratio)),
                max(40, math.floor((self.height() - 2 * MARGIN) * ratio)))

    def picture_rect(self) -> QRectF:
        """Where the picture sits: as large as its room allows, keeping its shape, and on
        the screen's pixel grid (a picture drawn between two pixels is a blurred picture)."""
        if self._before is None:
            return QRectF()
        ratio = self.devicePixelRatioF()
        box_w, box_h = self.picture_box()
        w, h = fit_size(box_w, box_h, self._before.width(), self._before.height())
        left = round(MARGIN * ratio + (box_w - w) / 2)
        top = round((self.height() * ratio - h) / 2)
        return QRectF(left / ratio, top / ratio, w / ratio, h / ratio)

    def set_inset_right(self, pixels: int) -> None:
        """Keep this much of the right side free: the picture moves over, it is never covered."""
        pixels = max(0, int(pixels))
        if pixels != self._inset_right:
            self._inset_right = pixels
            self._scaled.clear()
            self._place_browse()
            self.update()
            self._sync_hdr()
            self.box_changed.emit()

    def _made_for(self, image: QImage, rect: QRectF) -> bool:
        """Is this picture the size of its place, pixel for pixel?"""
        ratio = self.devicePixelRatioF()
        return (image.width(), image.height()) == (round(rect.width() * ratio), round(rect.height() * ratio))

    def _pixmap(self, image: QImage, rect: QRectF) -> QPixmap:
        ratio = self.devicePixelRatioF()
        size = (round(rect.width() * ratio), round(rect.height() * ratio))
        cached = self._scaled.get(id(image))
        if cached is None or cached[0] != size:
            # Made for this place: drawn as it is. Otherwise the window is being resized and
            # the picture for the new size is on its way: stretch this one meanwhile.
            scaled = image if self._made_for(image, rect) else \
                image.scaled(size[0], size[1], Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            pixmap = QPixmap.fromImage(scaled)
            pixmap.setDevicePixelRatio(ratio)
            cached = (size, pixmap)
            self._scaled[id(image)] = cached
        return cached[1]

    def resizeEvent(self, event) -> None:
        self._scaled.clear()
        self._place_browse()
        self._sync_hdr()
        self.resized.emit()
        self.box_changed.emit()

    def moveEvent(self, event) -> None:
        self.resized.emit()

    def _place_browse(self) -> None:
        hint = self._browse.sizeHint()
        free = self.width() - self._inset_right
        self._browse.setGeometry((free - hint.width()) // 2, self.height() // 2 + 34, hint.width(), hint.height())

    # -------------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        painter.fillRect(self.rect(), theme.colour(theme.INK))
        if self._before is None:
            self._paint_empty(painter)
        else:
            self._paint_pictures(painter)
        painter.end()

    def _paint_empty(self, painter: QPainter) -> None:
        free = self.width() - self._inset_right
        box = QRectF(0, 0, free, self.height()).adjusted(28, 28, -28, -28)
        pen = QPen(theme.colour(theme.GREEN if self._drop_hover else theme.SEAM), 1.5, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(theme.colour(theme.VELVET, 120 if self._drop_hover else 0))
        painter.drawRoundedRect(box, 14, 14)
        centre = self.height() / 2
        title = painter.font()
        title.setPixelSize(20)
        painter.setFont(title)
        painter.setPen(theme.colour(theme.IVORY))
        painter.drawText(QRectF(0, centre - 52, free, 32), Qt.AlignCenter, "Drop an HDR capture or recording")
        body = painter.font()
        body.setPixelSize(13)
        painter.setFont(body)
        painter.setPen(theme.colour(theme.MIST))
        painter.drawText(QRectF(0, centre - 16, free, 24), Qt.AlignCenter,
                         "JXR screenshots and HDR10 videos from the NVIDIA app")

    def _paint_pictures(self, painter: QPainter) -> None:
        rect = self.picture_rect()
        clip = QPainterPath()
        clip.addRoundedRect(rect, 6, 6)
        painter.save()
        painter.setClipPath(clip)
        x = rect.left() + rect.width() * self._split
        # A picture made for its place goes to the screen pixel for pixel: no filtering at all.
        painter.setRenderHint(QPainter.SmoothPixmapTransform, not self._made_for(self._after, rect))
        # after: the whole picture, then the unconverted one over its left part
        if self._previous_after is not None and self._blend < 1.0:
            painter.drawPixmap(rect.topLeft(), self._pixmap(self._previous_after, rect))
            painter.setOpacity(self._blend)
        painter.drawPixmap(rect.topLeft(), self._pixmap(self._after, rect))
        painter.setOpacity(1.0)
        painter.setClipRect(QRectF(rect.left(), rect.top(), x - rect.left(), rect.height()), Qt.IntersectClip)
        painter.drawPixmap(rect.topLeft(), self._pixmap(self._before, rect))
        painter.restore()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # the divider and its handle
        painter.setPen(QPen(theme.colour(theme.IVORY, 230), 1.5))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        self._paint_handle(painter, QPointF(x, rect.center().y()))
        self._paint_labels(painter, rect, x)

    def _paint_handle(self, painter, centre: QPointF) -> None:
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.colour(theme.IVORY))
        painter.drawEllipse(centre, 13, 13)
        painter.setPen(QPen(theme.colour(theme.INK), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for direction in (-1, 1):
            tip = centre.x() + direction * 7
            painter.drawLine(QPointF(tip, centre.y()), QPointF(tip - direction * 4, centre.y() - 4))
            painter.drawLine(QPointF(tip, centre.y()), QPointF(tip - direction * 4, centre.y() + 4))

    def _tags_visible(self, rect: QRectF, x: float):
        """(source label, result label): each shows only while its side is wide enough for it."""
        return (x - rect.left() > self.fontMetrics().horizontalAdvance(self._source_tag) + 60,
                rect.right() - x > 130)

    def _paint_labels(self, painter, rect: QRectF, x: float) -> None:
        """What sits on the pictures besides the divider: the two labels and the histogram."""
        source, result = self._tags_visible(rect, x)
        self._paint_tag(painter, self._source_tag, rect.left() + 10, rect.top() + 10, left=True, visible=source)
        after_text = "SDR result" if not self._note else f"SDR result, {self._note.lower()}"
        self._paint_tag(painter, after_text, rect.right() - 10, rect.top() + 10, left=False, visible=result)
        if self._histogram is not None:
            self._paint_histogram(painter, rect)

    def _paint_tag(self, painter, text, x, y, left, visible) -> None:
        if not visible:
            return
        font = painter.font()
        font.setPixelSize(12)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(text) + 18
        box = QRectF(x if left else x - width, y, width, 24)
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.colour(theme.INK, 190))
        painter.drawRoundedRect(box, 12, 12)
        painter.setPen(theme.colour(theme.IVORY))
        painter.drawText(box, Qt.AlignCenter, text)

    def _paint_histogram(self, painter, rect) -> None:
        box = QRectF(rect.left() + 10, rect.bottom() - 84, 168, 74)
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.colour(theme.INK, 190))
        painter.drawRoundedRect(box, 8, 8)
        inner = box.adjusted(8, 8, -8, -8)
        for channel, hex_value in enumerate(("#E0795B", "#76B900", "#6FA8DC")):
            values = self._histogram[channel]
            path = QPainterPath(QPointF(inner.left(), inner.bottom()))
            for i, v in enumerate(values):
                path.lineTo(inner.left() + inner.width() * i / (len(values) - 1), inner.bottom() - inner.height() * float(v))
            painter.setPen(QPen(theme.colour(hex_value, 220), 1.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

    # -------------------------------------------------------------- interaction
    def _split_from(self, x: float) -> None:
        rect = self.picture_rect()
        if rect.width() > 0:
            self.set_split((x - rect.left()) / rect.width())

    def mousePressEvent(self, event) -> None:
        if self._before is not None and event.button() == Qt.LeftButton:
            self._dragging = True
            self._split_from(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if self._before is None:
            return
        self.setCursor(Qt.SplitHCursor if self.picture_rect().contains(event.position()) else Qt.ArrowCursor)
        if self._dragging:
            self._split_from(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False

    def mouseDoubleClickEvent(self, event) -> None:
        if self._before is not None:
            self.set_split(0.5)

    def keyPressEvent(self, event) -> None:
        if self._before is not None and event.key() in (Qt.Key_Left, Qt.Key_Right):
            self.set_split(self._split + (0.02 if event.key() == Qt.Key_Right else -0.02))
        else:
            super().keyPressEvent(event)


def histogram_of(rgb8: np.ndarray, bins: int = 64) -> np.ndarray:
    """(3, bins) curves in 0..1 for the overlay, from an (H, W, 3) uint8 picture."""
    levels = rgb8.astype(np.int32)                       # uint8 * 64 would overflow
    curves = np.stack([np.bincount(levels[..., c].ravel() * bins // 256, minlength=bins)[:bins] for c in range(3)])
    curves = np.sqrt(curves.astype(np.float64))          # square root: keeps small peaks visible
    return curves / max(curves.max(), 1.0)

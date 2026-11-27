"""The queue: the files to convert, in a panel that slides in from the left.

One row per file: a picture of it, what it is (a capture, a video), its name, and where its
conversion stands. The row of the file on screen carries a mark that glides from row to
row. A file at work fills a ring and its row; one that is done draws its tick.

Rows are kept from one refresh of the window to the next (the window refreshes often):
a row is made when its file comes, moved when the list changes, and let go when its file
leaves. That is also what lets it animate: it has a past.
"""
import math
import os

from PySide6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSize, Qt, QTimer,
                            QVariantAnimation, Signal)
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from . import theme
from .widgets import CloseButton

QUEUE_WIDTH = 320
ROW_HEIGHT = 66
ROW_GAP = 4
EDGE = 10                              # between the panel's edge and a row
THUMB = (96, 54)                       # the picture of a file, 16:9
KIND_NAMES = {"image": "JXR capture", "video": "HDR10 video"}


def _animation(owner, duration: int, on_value, curve=QEasingCurve.OutCubic) -> QVariantAnimation:
    """An animation from 0 to 1. Where the system asks for no animation it is over at once."""
    animation = QVariantAnimation(owner)
    animation.setDuration(duration if theme.motion() else 0)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setEasingCurve(curve)
    animation.valueChanged.connect(on_value)
    return animation


class QueueRow(QWidget):
    """One file of the queue. Painted by hand: the ring, the tick and the wash are not
    things a style sheet draws."""

    def __init__(self, panel: "QueuePanel", path: str, kind: str, parent=None):
        super().__init__(parent)
        self._panel = panel
        self.path, self.kind = path, kind
        self.name = os.path.basename(path)
        self.state, self.message, self.outputs = "waiting", "", []
        self.selected = False
        self.removable = True
        self.thumbnail = None                      # QImage, once the file has been read
        self._activity = ""                        # at work: "Converting… 42 %"
        self._target = 0.0                         # progress: where it stands
        self._shown = 0.0                          # and what is drawn, catching up smoothly
        self._known = True                         # False: at work, how far is not known
        self._hover = self._hover_remove = False
        self._arrived = 1.0                        # 0..1: the row slides in
        self._ticked = 1.0                         # 0..1: the tick draws itself
        self._pictured = 1.0                       # 0..1: the picture fades in
        self.setFixedHeight(ROW_HEIGHT)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover, True)
        self.setToolTip(path)
        self.setAccessibleName(self.name)
        self.setAccessibleDescription(self.caption())
        self._arrive = _animation(self, 240, lambda v: self._set("_arrived", v))
        self._tick = _animation(self, 520, lambda v: self._set("_ticked", v))
        self._picture = _animation(self, 220, lambda v: self._set("_pictured", v))
        self._catch_up = QVariantAnimation(self)
        self._catch_up.setDuration(200 if theme.motion() else 0)
        self._catch_up.valueChanged.connect(lambda v: self._set("_shown", v))
        self._move = QPropertyAnimation(self, b"pos", self)
        self._move.setDuration(180 if theme.motion() else 0)
        self._move.setEasingCurve(QEasingCurve.OutCubic)

    def _set(self, name: str, value) -> None:
        setattr(self, name, float(value))
        self.update()

    # ---------------------------------------------------------------- state
    def arrive(self, delay_ms: int) -> None:
        """Slide in, a little after the row above: files dropped together come in one after the other."""
        if not theme.motion() or not self.isVisible():
            return
        self._arrived = 0.0
        QTimer.singleShot(delay_ms, self._arrive.start)

    def set_state(self, state: str, message: str, outputs, removable: bool) -> None:
        became_done = state == "done" and self.state != "done"
        changed = (state, message, list(outputs), removable) != (self.state, self.message, self.outputs, self.removable)
        self.state, self.message, self.outputs, self.removable = state, message, list(outputs), removable
        if state == "done":
            self._progress_to(1.0)
        elif state in ("waiting", "cancelled", "failed"):
            self._catch_up.stop()
            self._target = self._shown = 0.0
            self._activity = ""
        if became_done and self.isVisible() and theme.motion():
            self._tick.stop()
            self._ticked = 0.0
            self._tick.start()
        if changed:
            self.setToolTip(self.message or (os.linesep.join(self.outputs) if self.outputs else self.path))
            self.setAccessibleDescription(self.caption())
            self.update()

    def set_activity(self, text: str, fraction) -> None:
        """At work: what is being done, and how far (``None`` when that cannot be known)."""
        self._activity = text
        self._known = fraction is not None
        if fraction is not None:
            self._progress_to(float(fraction))
        self.update()

    def _progress_to(self, fraction: float) -> None:
        fraction = min(1.0, max(0.0, fraction))
        if fraction == self._target:
            return
        self._target = fraction
        self._catch_up.stop()
        if not self.isVisible() or fraction < self._shown:
            self._shown = fraction
            self.update()
            return
        self._catch_up.setStartValue(self._shown)
        self._catch_up.setEndValue(fraction)
        self._catch_up.start()

    def progress(self) -> float:
        return self._target

    def set_thumbnail(self, image: QImage) -> None:
        first = self.thumbnail is None
        self.thumbnail = image
        if first and self.isVisible() and theme.motion():
            self._pictured = 0.0
            self._picture.start()
        self.update()

    def caption(self) -> str:
        """The second line: what the file is, or where its conversion stands."""
        if self.state == "working":
            return self._activity or "Starting…"
        if self.state == "done":
            if not self.outputs:
                return "Done"
            more = f" and {len(self.outputs) - 1} more" if len(self.outputs) > 1 else ""
            return f"Saved {os.path.basename(self.outputs[0])}{more}"
        if self.state == "failed":
            text = self.message or "Could not be converted"
            return text[len(self.name) + 2:] if text.startswith(self.name + ": ") else text
        if self.state == "cancelled":
            return "Stopped"
        return KIND_NAMES.get(self.kind, self.kind)

    def slide_to(self, y: int, at_once: bool) -> None:
        target = self.pos()
        target.setY(y)
        target.setX(EDGE)
        self._move.stop()
        if at_once or not self.isVisible() or self.pos() == target:
            self.move(target)
            return
        self._move.setStartValue(self.pos())
        self._move.setEndValue(target)
        self._move.start()

    # -------------------------------------------------------------- pointer
    def _status_centre(self) -> QPointF:
        return QPointF(self.width() - 24.0, self.height() / 2.0)

    def _over_remove(self, position) -> bool:
        centre = self._status_centre()
        return self.removable and abs(position.x() - centre.x()) <= 14 and abs(position.y() - centre.y()) <= 14

    def mouseMoveEvent(self, event) -> None:
        over = self._over_remove(event.position())
        if over != self._hover_remove:
            self._hover_remove = over
            self.update()

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover = self._hover_remove = False
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self.rect().contains(event.position().toPoint()):
            return
        if event.button() == Qt.RightButton or (event.button() == Qt.LeftButton and self._over_remove(event.position())):
            self.remove()
        elif event.button() == Qt.LeftButton:
            self.click()

    def click(self) -> None:
        self._panel.choose(self)

    def remove(self) -> None:
        if self.removable:
            self._panel.take_out(self)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        if self._arrived < 1.0:
            painter.setOpacity(self._arrived)
            painter.translate(-EDGE * (1.0 - self._arrived), 0.0)     # from the panel's edge, no further
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        shape = QPainterPath()
        shape.addRoundedRect(box, 10, 10)
        if self._hover and not self.selected:
            painter.fillPath(shape, theme.colour(theme.VELVET_HI, 150))
        self._paint_wash(painter, box, shape)
        self._paint_picture(painter)
        self._paint_text(painter)
        self._paint_status(painter)
        painter.end()

    def _paint_wash(self, painter: QPainter, box: QRectF, shape: QPainterPath) -> None:
        """At work, the row fills with green as the file advances, and a soft light runs
        over what is filled. Done, the whole row glows once and settles.

        The filling starts at the right edge of the picture, not at the row's: the picture
        would hide the first third of it. So the row at work is told from the others at
        once, by the green that frames its picture, and every step forward is seen."""
        if self.state == "working":
            painter.save()
            painter.setClipPath(shape)
            start = self._thumb_box().right() + 8.0
            reach = start + (box.right() - start) * self._shown if self._known else box.right()
            filled = QRectF(box.left(), box.top(), reach - box.left(), box.height())
            painter.fillRect(filled, theme.colour(theme.GREEN, 30 if self._known else 16))
            phase = self._panel.phase()
            if theme.motion() and filled.width() > 4:
                x = filled.left() - 70 + (filled.width() + 140) * phase
                light = QLinearGradient(x - 70, 0, x + 70, 0)
                light.setColorAt(0.0, theme.colour(theme.GREEN, 0))
                light.setColorAt(0.5, theme.colour(theme.GREEN, 46))
                light.setColorAt(1.0, theme.colour(theme.GREEN, 0))
                painter.fillRect(filled, light)
            painter.restore()
        elif self.state == "done" and self._ticked < 1.0:
            painter.fillPath(shape, theme.colour(theme.GREEN, round(46 * (1.0 - self._ticked))))

    def _thumb_box(self) -> QRectF:
        return QRectF(8.0, (self.height() - THUMB[1]) / 2.0, THUMB[0], THUMB[1])

    def _paint_picture(self, painter: QPainter) -> None:
        frame = self._thumb_box()
        shape = QPainterPath()
        shape.addRoundedRect(frame, 6, 6)
        painter.fillPath(shape, theme.colour(theme.INK))
        if self.thumbnail is not None:
            painter.save()
            painter.setClipPath(shape)
            painter.setOpacity(painter.opacity() * self._pictured)
            image = self.thumbnail
            scale = max(frame.width() / image.width(), frame.height() / image.height())     # fills the frame
            w, h = image.width() * scale, image.height() * scale
            painter.drawImage(QRectF(frame.center().x() - w / 2, frame.center().y() - h / 2, w, h), image)
            painter.restore()
        painter.setPen(QPen(theme.colour(theme.RIM if self.selected else theme.SEAM), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(shape)
        # what it is, in the corner of its picture: a play mark for a video, a small
        # landscape for a capture
        badge = QRectF(frame.left() + 5, frame.bottom() - 21, 22, 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.colour(theme.INK, 205))
        painter.drawRoundedRect(badge, 5, 5)
        centre = badge.center()
        painter.setBrush(theme.colour(theme.IVORY))
        if self.kind == "video":
            mark = QPainterPath()
            mark.moveTo(centre.x() - 3.2, centre.y() - 4.2)
            mark.lineTo(centre.x() + 4.2, centre.y())
            mark.lineTo(centre.x() - 3.2, centre.y() + 4.2)
            mark.closeSubpath()
            painter.drawPath(mark)
        else:
            painter.drawEllipse(QPointF(centre.x() + 3.6, centre.y() - 2.4), 1.5, 1.5)
            hills = QPainterPath()
            hills.moveTo(centre.x() - 6.5, centre.y() + 4.5)
            hills.lineTo(centre.x() - 2.2, centre.y() - 1.6)
            hills.lineTo(centre.x() + 0.6, centre.y() + 1.8)
            hills.lineTo(centre.x() + 2.4, centre.y() + 0.2)
            hills.lineTo(centre.x() + 6.5, centre.y() + 4.5)
            hills.closeSubpath()
            painter.drawPath(hills)

    def _paint_text(self, painter: QPainter) -> None:
        left = self._thumb_box().right() + 12
        width = self.width() - left - 46
        name = painter.font()
        name.setPixelSize(13)
        name.setWeight(name.Weight.Medium)
        painter.setFont(name)
        painter.setPen(theme.colour(theme.IVORY if self.state != "cancelled" else theme.MIST))
        text = painter.fontMetrics().elidedText(self.name, Qt.ElideMiddle, int(width))
        painter.drawText(QRectF(left, 13, width, 20), Qt.AlignLeft | Qt.AlignVCenter, text)
        small = painter.font()
        small.setPixelSize(12)
        small.setWeight(small.Weight.Normal)
        painter.setFont(small)
        painter.setPen(theme.colour(theme.EMBER if self.state == "failed" else theme.MIST))
        # the name of the file written is cut in its middle: what it ends with says what it is
        cut = Qt.ElideMiddle if self.state == "done" else Qt.ElideRight
        caption = painter.fontMetrics().elidedText(self.caption(), cut, int(width))
        painter.drawText(QRectF(left, 34, width, 18), Qt.AlignLeft | Qt.AlignVCenter, caption)

    def _paint_status(self, painter: QPainter) -> None:
        centre, radius = self._status_centre(), 9.0
        ring = QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius)
        painter.setBrush(Qt.NoBrush)
        if self._hover and self.removable:
            # the way out of the list, in place of the state while the pointer is on the row
            if self._hover_remove:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.colour(theme.LIFT))
                painter.drawEllipse(centre, 13, 13)
            painter.setPen(QPen(theme.colour(theme.IVORY if self._hover_remove else theme.MIST), 1.5, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(centre + QPointF(-4, -4), centre + QPointF(4, 4))
            painter.drawLine(centre + QPointF(-4, 4), centre + QPointF(4, -4))
            return
        if self.state == "working":
            painter.setPen(QPen(theme.colour(theme.RIM if self.selected else theme.SEAM), 2))   # seen on the mark too
            painter.drawEllipse(ring)
            painter.setPen(QPen(theme.colour(theme.GREEN), 2, Qt.SolidLine, Qt.RoundCap))
            if self._known:
                painter.drawArc(ring, 90 * 16, -round(360 * 16 * self._shown))
            else:
                painter.drawArc(ring, round((90 - 360 * self._panel.phase()) * 16), -90 * 16)
        elif self.state == "done":
            drawn = min(1.0, self._ticked * 1.7)
            if self._ticked < 1.0:                 # a ring of light leaves the tick as it lands
                grow = radius + 11 * self._ticked
                painter.setPen(QPen(theme.colour(theme.GREEN, round(150 * (1.0 - self._ticked))), 1.5))
                painter.drawEllipse(centre, grow, grow)
            painter.setPen(QPen(theme.colour(theme.GREEN, 90), 1.5))
            painter.drawEllipse(ring)
            tick = QPainterPath()
            tick.moveTo(centre.x() - 4.2, centre.y() + 0.2)
            tick.lineTo(centre.x() - 1.2, centre.y() + 3.2)
            tick.lineTo(centre.x() + 4.4, centre.y() - 3.4)
            painter.setPen(QPen(theme.colour(theme.GREEN), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if drawn >= 1.0:
                painter.drawPath(tick)
            else:
                part = QPainterPath()
                part.moveTo(tick.pointAtPercent(0.0))
                steps = max(1, round(24 * drawn))
                for step in range(1, steps + 1):
                    part.lineTo(tick.pointAtPercent(drawn * step / steps))
                painter.drawPath(part)
        elif self.state == "failed":
            painter.setPen(QPen(theme.colour(theme.EMBER), 1.5))
            painter.drawEllipse(ring)
            painter.setPen(QPen(theme.colour(theme.EMBER), 2, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(centre + QPointF(0, -4.2), centre + QPointF(0, 1.0))
            painter.drawPoint(centre + QPointF(0, 4.4))
        elif self.state == "cancelled":
            painter.setPen(QPen(theme.colour(theme.DIM), 1.5))
            painter.drawEllipse(ring)
            painter.setPen(QPen(theme.colour(theme.DIM), 2, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(centre + QPointF(-3.5, 0), centre + QPointF(3.5, 0))
        else:
            painter.setPen(QPen(theme.colour(theme.RIM if self.selected else theme.SEAM), 1.5))
            painter.drawEllipse(ring)


class _Rows(QWidget):
    """The rows, one under the other, and under them the mark of the chosen one."""

    def __init__(self, panel: "QueuePanel"):
        super().__init__()
        self._panel = panel
        self.setObjectName("row")
        self._place = None                          # where the mark is painted
        self._glide = QVariantAnimation(self)
        self._glide.setDuration(190 if theme.motion() else 0)
        self._glide.setEasingCurve(QEasingCurve.OutCubic)
        self._glide.valueChanged.connect(self._on_glide)

    def _on_glide(self, place) -> None:
        self._place = QRectF(place)
        self.update()

    def mark(self, place, at_once: bool) -> None:
        """Put the mark on ``place`` (None: no row is chosen). It glides there from where it was."""
        self._glide.stop()
        if place is None or self._place is None or at_once or not self.isVisible():
            self._place = QRectF(place) if place is not None else None
            self.update()
            return
        if QRectF(place) == self._place:
            return
        self._glide.setStartValue(QRectF(self._place))
        self._glide.setEndValue(QRectF(place))
        self._glide.start()

    def place(self):
        return QRectF(self._glide.endValue()) if self._glide.state() == QVariantAnimation.Running else self._place

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        if self._place is not None:
            pill = self._place.adjusted(0.5, 0.5, -0.5, -0.5)
            # a brighter edge while the keys go to the list
            painter.setPen(QPen(theme.colour(theme.MIST if self._panel.hasFocus() else theme.RIM), 1))
            painter.setBrush(theme.colour(theme.LIFT))
            painter.drawRoundedRect(pill, 10, 10)
        elif not self._panel.rows:
            painter.setPen(theme.colour(theme.MIST))
            painter.drawText(QRectF(EDGE + 8, 18, self.width() - 2 * EDGE - 16, 60), Qt.AlignLeft | Qt.TextWordWrap,
                             "No file yet. Drop captures or recordings anywhere in the window.")
        painter.end()


class _Line(QWidget):
    """How far the whole run is: a hairline under the title, green as far as it has got."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(2)
        self.fraction = None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.colour(theme.SEAM))
        if self.fraction is not None:
            painter.fillRect(QRectF(0, 0, self.width() * self.fraction, self.height()), theme.colour(theme.GREEN))
        painter.end()


class QueuePanel(QWidget):
    selected = Signal(int)
    removed = Signal(int)
    add_requested = Signal()
    clear_requested = Signal()                      # take the finished files out
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("queue_panel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        # A child of the picture area: a click that none of its rows takes stops here.
        self.setAttribute(Qt.WA_NoMousePropagation, True)
        self.setFixedWidth(QUEUE_WIDTH)
        self.setFocusPolicy(Qt.StrongFocus)         # Up and Down walk the list, Delete takes a file out
        self.rows = []
        self._by_path = {}
        self._followed = None                       # the row at work the list was last scrolled to
        self._chosen = None                         # the row of the file on screen
        self._keep_chosen = False                   # it is kept in view until the user scrolls away
        self._revealing = False
        self._clamped = False                       # the bar is being brought back into a shorter range

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(18, 14, 12, 8)
        title = QLabel("Queue")
        font = title.font()
        font.setPixelSize(15)
        font.setWeight(font.Weight.DemiBold)
        title.setFont(font)
        head.addWidget(title)
        head.addSpacing(10)
        self.summary = QLabel("")
        self.summary.setProperty("role", "muted")
        head.addWidget(self.summary, 1)
        self.close_button = CloseButton()
        self.close_button.clicked.connect(self.close_requested)
        head.addWidget(self.close_button)
        outer.addLayout(head)
        self._line = _Line()
        outer.addWidget(self._line)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list = _Rows(self)
        self._scroll.setWidget(self._list)
        self._scroll.verticalScrollBar().rangeChanged.connect(self._on_range)
        self._scroll.verticalScrollBar().valueChanged.connect(self._on_scrolled)
        outer.addWidget(self._scroll, 1)

        foot = QHBoxLayout()
        foot.setContentsMargins(18, 10, 18, 14)
        foot.setSpacing(8)
        self.add_button = QPushButton("Add files…")
        self.clear_button = QPushButton("Clear finished")
        for button in (self.add_button, self.clear_button):
            button.setProperty("role", "ghost")
            button.setCursor(Qt.PointingHandCursor)
            button.setFocusPolicy(Qt.TabFocus)
            foot.addWidget(button)
        foot.addStretch(1)
        self.add_button.clicked.connect(self.add_requested)
        self.clear_button.clicked.connect(self.clear_requested)
        self.clear_button.hide()
        outer.addLayout(foot)

        # one clock for every row at work: the light that runs over them, the turning ring
        self._phase = 0.0
        self._clock = QVariantAnimation(self)
        self._clock.setDuration(1500)
        self._clock.setStartValue(0.0)
        self._clock.setEndValue(1.0)
        self._clock.setLoopCount(-1)
        self._clock.valueChanged.connect(self._tick)

    # ---------------------------------------------------------------- rows
    def set_items(self, items, selected: int, converting: bool = False) -> None:
        """Show these files, in this order. Rows of files already shown are kept."""
        fresh, kept = [], []
        for item in items:
            row = self._by_path.get(item.path)
            if row is None:
                row = QueueRow(self, item.path, item.kind, self._list)
                row.resize(self._row_width(), ROW_HEIGHT)
                self._by_path[item.path] = row
                fresh.append(row)
            kept.append(row)
        for path in [p for p, row in self._by_path.items() if row not in kept]:
            row = self._by_path.pop(path)
            row.hide()
            row.setParent(None)
            row.deleteLater()
        self.rows = kept
        for index, (row, item) in enumerate(zip(kept, items)):
            row.selected = index == selected
            row.set_state(item.state, item.message, item.outputs, removable=not converting)
            new = row in fresh
            row.slide_to(index * (ROW_HEIGHT + ROW_GAP) + 4, at_once=new)
            if new:
                row.show()
                row.arrive(45 * min(fresh.index(row), 8))
            row.update()
        self._list.setMinimumHeight(len(kept) * (ROW_HEIGHT + ROW_GAP) + 8)
        chosen = kept[selected] if 0 <= selected < len(kept) else None
        if chosen is not self._chosen:
            # Another file is on screen: its row is brought into view, and stays there when
            # the panel changes height (a video brings its timeline, the window is resized).
            self._chosen, self._keep_chosen = chosen, chosen is not None
            QTimer.singleShot(0, self._reveal_chosen)
        place = None
        if chosen is not None:
            place = QRectF(EDGE, selected * (ROW_HEIGHT + ROW_GAP) + 4, self._row_width(), ROW_HEIGHT)
        self._list.mark(place, at_once=chosen is not None and chosen in fresh and len(fresh) == len(kept))
        self._list.update()
        done = sum(item.state == "done" for item in items)
        failed = sum(item.state == "failed" for item in items)
        words = [f"{len(items)} file{'s' if len(items) != 1 else ''}"] if items else []
        words += [f"{done} done"] if done else []
        words += [f"{failed} failed"] if failed else []
        self.summary.setText(", ".join(words))
        self.clear_button.setVisible(done > 0)
        self.clear_button.setEnabled(not converting)
        if not converting:
            self._followed = None
        self._run_clock()

    def _row_width(self) -> int:
        return QUEUE_WIDTH - 2 * EDGE

    def _reveal(self, row: QueueRow) -> None:
        """Scroll the list so that this row is whole in view. (Where the row is going, not
        where it is: it may still be sliding there.)"""
        if row not in self.rows:
            return
        top = self.rows.index(row) * (ROW_HEIGHT + ROW_GAP) + 4
        bar, height = self._scroll.verticalScrollBar(), self._scroll.viewport().height()
        self._revealing = True
        if top - 4 < bar.value():
            bar.setValue(top - 4)
        elif top + ROW_HEIGHT + 4 > bar.value() + height:
            bar.setValue(top + ROW_HEIGHT + 4 - height)
        self._revealing = False

    def _reveal_chosen(self) -> None:
        if self._keep_chosen and self._chosen is not None:
            self._reveal(self._chosen)

    def _on_range(self, _low: int, high: int) -> None:
        """The panel got taller or the list shorter. A bar that stood beyond its new range is
        brought back into it at once: that is not the user scrolling. (Choosing a video does
        it: its timeline leaves until the picture comes, and the panel is taller meanwhile.)"""
        self._clamped = self._scroll.verticalScrollBar().value() > high

    def _on_scrolled(self, _value) -> None:
        if self._clamped:
            self._clamped = False
        elif not self._revealing:
            self._keep_chosen = False               # the user looks elsewhere: the list stays there

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._reveal_chosen)   # once the list has its new height

    def pill_place(self):
        """Where the mark of the chosen row is (or is gliding to)."""
        return self._list.place()

    def choose(self, row: QueueRow) -> None:
        if row in self.rows:
            self.setFocus(Qt.MouseFocusReason)      # the arrow keys now walk the list
            self.selected.emit(self.rows.index(row))

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        self._list.update()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self._list.update()

    def take_out(self, row: QueueRow) -> None:
        if row in self.rows:
            self.removed.emit(self.rows.index(row))

    def set_progress(self, index: int, text: str, fraction) -> None:
        if 0 <= index < len(self.rows):
            row = self.rows[index]
            row.set_activity(text, fraction)
            if row is not self._followed:
                # shown when its turn comes, not at every step of it: the user may be
                # looking at another part of a long list
                self._followed = row
                self._reveal(row)

    def set_thumbnail(self, path: str, image: QImage) -> None:
        row = self._by_path.get(path)
        if row is not None and not image.isNull():
            row.set_thumbnail(image)

    def wants_thumbnail(self, path: str) -> bool:
        row = self._by_path.get(path)
        return row is not None and row.thumbnail is None

    def set_overall(self, fraction) -> None:
        """How far the whole run is, 0..1; ``None`` when nothing is being converted."""
        self._line.fraction = None if fraction is None else min(1.0, max(0.0, float(fraction)))
        self._line.update()

    def overall(self):
        return self._line.fraction

    # --------------------------------------------------------------- clock
    def phase(self) -> float:
        return self._phase

    def _tick(self, value) -> None:
        self._phase = float(value)
        for row in self.rows:
            if row.state == "working":
                row.update()

    def _run_clock(self) -> None:
        wanted = theme.motion() and self.isVisible() and any(row.state == "working" for row in self.rows)
        if wanted and self._clock.state() != QVariantAnimation.Running:
            self._clock.start()
        elif not wanted and self._clock.state() == QVariantAnimation.Running:
            self._clock.stop()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._run_clock()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._run_clock()

    def keyPressEvent(self, event) -> None:
        chosen = next((i for i, row in enumerate(self.rows) if row.selected), -1)
        if event.key() in (Qt.Key_Up, Qt.Key_Down) and self.rows:
            self.selected.emit(min(len(self.rows) - 1, max(0, chosen + (1 if event.key() == Qt.Key_Down else -1))))
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and chosen >= 0:
            self.rows[chosen].remove()
        else:
            super().keyPressEvent(event)


class QueueButton(QPushButton):
    """The button that opens the queue: its name, and how many files wait in it."""

    def __init__(self, parent=None):
        super().__init__("Queue", parent)
        self._count = 0
        self._popped = 1.0                         # 0..1: the badge swells as a file is counted
        self._pop = _animation(self, 320, self._on_pop, QEasingCurve.Linear)
        self.setProperty("role", "ghost")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)

    def _on_pop(self, value) -> None:
        self._popped = float(value)
        self.update()

    def set_count(self, count: int) -> None:
        if count != self._count:
            if count > self._count and self.isVisible() and theme.motion():
                self._pop.stop()                    # one more file: the badge says it took it
                self._pop.start()
            self._count = int(count)
            # the word, then blank room that the badge is painted over
            self.setText("Queue" + " " * (5 + 2 * len(str(self._count))) if self._count else "Queue")
            self.updateGeometry()
            self.update()

    def count(self) -> int:
        return self._count

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        """The style sheet draws the pill and the word; the count is drawn here, as a small
        round badge in the room the text leaves blank after the word."""
        super().paintEvent(event)
        if not self._count:
            return
        figure = str(self._count)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        metrics = self.fontMetrics()
        whole = metrics.horizontalAdvance(self.text())
        width = max(18.0, metrics.horizontalAdvance(figure) + 10.0)
        left = (self.width() + whole) / 2 - width
        badge = QRectF(left, (self.height() - 18) / 2, width, 18)
        if self._popped < 1.0:
            swell = 1.0 + 0.28 * math.sin(math.pi * self._popped)
            painter.translate(badge.center())
            painter.scale(swell, swell)
            painter.translate(-badge.center())
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(theme.GREEN) if self.isChecked() or self.underMouse() else theme.colour(theme.LIFT))
        painter.drawRoundedRect(badge, 9, 9)
        small = self.font()
        small.setPixelSize(11)
        small.setWeight(small.Weight.DemiBold)
        painter.setFont(small)
        painter.setPen(theme.colour(theme.INK if self.isChecked() or self.underMouse() else theme.IVORY))
        painter.drawText(badge, Qt.AlignCenter, figure)
        painter.end()

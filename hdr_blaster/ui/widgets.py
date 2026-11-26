"""The small controls drawn by hand, where a style sheet cannot do the job: the segmented
choice, the switch, the list with its chevron, the round close button."""
from PySide6.QtCore import QEasingCurve, QEvent, QPointF, QRectF, QSize, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QCheckBox, QComboBox, QHBoxLayout, QPushButton, QSizePolicy,
                               QStyledItemDelegate, QWidget)

from . import theme


def mixed(a: str, b: str, share: float) -> QColor:
    """The colour ``share`` of the way from ``a`` to ``b``."""
    first, second = QColor(a), QColor(b)
    return QColor(*(round(x + (y - x) * share) for x, y in zip(first.getRgb()[:3], second.getRgb()[:3])))


class Segmented(QWidget):
    """A few choices in one well; the chosen one is a raised pill that glides from choice to
    choice. The buttons are ordinary checkable buttons, kept exclusive by their owner."""

    INSET = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(self.INSET, self.INSET, self.INSET, self.INSET)
        self._layout.setSpacing(0)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._buttons = []
        self._gliding = None                        # the pill's place while it travels
        self._glide = QVariantAnimation(self)
        self._glide.setDuration(180)
        self._glide.setEasingCurve(QEasingCurve.OutCubic)
        self._glide.valueChanged.connect(self._on_glide)
        self._glide.finished.connect(self._landed)

    def add(self, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setCheckable(True)
        button.setProperty("role", "segment")
        button.setCursor(Qt.PointingHandCursor)
        button.setFocusPolicy(Qt.TabFocus)
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        button.toggled.connect(lambda on, b=button: self._chosen(b) if on else self.update())
        self._layout.addWidget(button)
        self._buttons.append(button)
        return button

    def _place_of(self, button) -> QRectF:
        return QRectF(button.geometry())

    def _chosen(self, button) -> None:
        start = self._shown                         # where the pill was last painted
        self._glide.stop()
        if start is None or not self.isVisible():
            self._gliding = None                    # nothing to come from: the pill is simply there
            self.update()
            return
        self._gliding = QRectF(start)
        self._glide.setStartValue(QRectF(start))
        self._glide.setEndValue(self._place_of(button))
        self._glide.start()

    _shown = None

    def _on_glide(self, place) -> None:
        self._gliding = QRectF(place)
        self.update()

    def _landed(self) -> None:
        self._gliding = None
        self.update()

    def resizeEvent(self, event) -> None:
        self._glide.stop()
        self._gliding = None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        well = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(theme.colour(theme.SEAM), 1))
        painter.setBrush(theme.colour(theme.INK))
        painter.drawRoundedRect(well, well.height() / 2, well.height() / 2)
        chosen = next((b for b in self._buttons if b.isChecked()), None)
        place = None
        if chosen is not None:
            place = self._gliding if self._gliding is not None else self._place_of(chosen)
        self._shown = place
        if place is not None:
            pill = place.adjusted(0.5, 0.5, -0.5, -0.5)
            painter.setPen(QPen(theme.colour(theme.RIM if self.isEnabled() else theme.SEAM), 1))
            painter.setBrush(theme.colour(theme.LIFT if self.isEnabled() else theme.VELVET_HI))
            painter.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)
        painter.end()


class Switch(QCheckBox):
    """An on/off setting: its name on the left, a switch on the right. The whole row answers."""

    TRACK = (34, 20)
    GAP = 14

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._place = 0.0                           # 0: off, 1: on, in between while the knob travels
        self._travel = QVariantAnimation(self)
        self._travel.setDuration(140)
        self._travel.setEasingCurve(QEasingCurve.OutCubic)
        self._travel.valueChanged.connect(self._on_travel)
        self.toggled.connect(self._switched)

    def _switched(self, on: bool) -> None:
        end = 1.0 if on else 0.0
        self._travel.stop()
        if not self.isVisible():
            self._place = end
            self.update()
            return
        self._travel.setStartValue(self._place)
        self._travel.setEndValue(end)
        self._travel.start()

    def _on_travel(self, value) -> None:
        self._place = float(value)
        self.update()

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self.text()) + self.GAP + self.TRACK[0] + 2,
                     max(self.TRACK[1] + 8, metrics.height() + 8))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, position) -> bool:
        return self.rect().contains(position)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.EnabledChange:
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        enabled = self.isEnabled()
        width, height = self.TRACK
        painter.setFont(self.font())
        painter.setPen(theme.colour(theme.IVORY if enabled else theme.DIM))
        painter.drawText(QRectF(0, 0, self.width() - width - self.GAP, self.height()),
                         Qt.AlignLeft | Qt.AlignVCenter, self.text())
        track = QRectF(self.width() - width - 1, (self.height() - height) / 2, width, height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(mixed(theme.LIFT, theme.GREEN, self._place) if enabled else theme.colour(theme.VELVET_HI))
        painter.drawRoundedRect(track, height / 2, height / 2)
        knob = QPointF(track.left() + height / 2 + self._place * (width - height), track.center().y())
        painter.setBrush(mixed(theme.MIST, "#FFFFFF", self._place) if enabled else theme.colour(theme.DIM))
        painter.drawEllipse(knob, height / 2 - 3, height / 2 - 3)
        if self.hasFocus():
            painter.setPen(QPen(theme.colour(theme.MIST), 1))
            painter.setBrush(Qt.NoBrush)
            ring = track.adjusted(-2.5, -2.5, 2.5, 2.5)
            painter.drawRoundedRect(ring, ring.height() / 2, ring.height() / 2)
        painter.end()


class Combo(QComboBox):
    """A list of choices. The style sheet gives it its field; the chevron is drawn here."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(QStyledItemDelegate(self))      # so that the style sheet reaches the rows
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bright = self.isEnabled() and (self.underMouse() or self.hasFocus())
        colour = theme.IVORY if bright else (theme.MIST if self.isEnabled() else theme.DIM)
        painter.setPen(QPen(theme.colour(colour), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        x, y = self.width() - 17.0, self.height() / 2.0
        painter.drawPolyline([QPointF(x - 4, y - 2), QPointF(x, y + 2), QPointF(x + 4, y - 2)])
        painter.end()

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.update()


class CloseButton(QAbstractButton):
    """A round button with a cross."""

    SIDE = 28

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIDE, self.SIDE)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.setToolTip("Close")
        self.setAccessibleName("Close")

    def sizeHint(self) -> QSize:
        return QSize(self.SIDE, self.SIDE)

    def enterEvent(self, event) -> None:
        self.update()

    def leaveEvent(self, event) -> None:
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hovered = self.underMouse()
        if self.isDown() or hovered:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme.colour(theme.LIFT if self.isDown() else theme.VELVET_HI))
            painter.drawEllipse(box)
        painter.setPen(QPen(theme.colour(theme.IVORY if hovered else theme.MIST), 1.5, Qt.SolidLine, Qt.RoundCap))
        centre, arm = box.center(), 4.5
        painter.drawLine(centre + QPointF(-arm, -arm), centre + QPointF(arm, arm))
        painter.drawLine(centre + QPointF(-arm, arm), centre + QPointF(arm, -arm))
        if self.hasFocus():
            painter.setPen(QPen(theme.colour(theme.MIST), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(box)
        painter.end()

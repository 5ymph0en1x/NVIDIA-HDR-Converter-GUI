"""The bar under the picture: four looks, what will be written, and the one action."""
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPainterPath
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from ..session import LOOKS
from . import theme


class ElidedLabel(QLabel):
    """A label that shows as much of its text as fits, with an ellipsis, and the whole text
    as a tooltip. It never forces the window wider."""

    def __init__(self, mode=Qt.ElideRight, parent=None):
        super().__init__("", parent)
        self._mode, self._full = mode, ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)

    def set_full_text(self, text: str, tooltip: str = "", mode=None) -> None:
        self._full = text
        if mode is not None:
            self._mode = mode
        self.setToolTip(tooltip or text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(self._full, self._mode, max(self.width(), 0)))


class LookBar(QWidget):
    """Auto, Natural, Punchy, Golden. Exactly one is chosen."""

    look_chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel("Look")
        label.setProperty("role", "muted")
        layout.addWidget(label)
        layout.addSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons = {}
        for name in LOOKS:
            button = QPushButton(name)
            button.setCheckable(True)
            button.setProperty("role", "look")
            button.setCursor(Qt.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            self._group.addButton(button)
            layout.addWidget(button)
            self.buttons[name] = button
            button.clicked.connect(lambda _checked, n=name: self.look_chosen.emit(n))

    def set_look(self, name) -> None:
        """Light one look, or none (``None``) when an advanced operator replaces the looks."""
        if name is None:
            self._group.setExclusive(False)
            for button in self.buttons.values():
                button.setChecked(False)
            self._group.setExclusive(True)
            return
        self.buttons.get(name, next(iter(self.buttons.values()))).setChecked(True)

    def look(self):
        return next((name for name, button in self.buttons.items() if button.isChecked()), None)


class ConvertButton(QPushButton):
    """The one prominent button. While working it becomes its own progress bar."""

    def __init__(self, parent=None):
        super().__init__("Convert", parent)
        self._progress = None                       # None: idle; 0..1 while working
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(150)
        font = self.font()
        font.setPixelSize(14)
        font.setWeight(font.Weight.DemiBold)
        self.setFont(font)

    def set_idle(self, text: str = "Convert") -> None:
        self._progress = None
        self.setText(text)
        self.update()

    def set_working(self, text: str, fraction: float) -> None:
        self._progress = min(1.0, max(0.0, float(fraction)))
        self.setText(text)
        self.update()

    def working(self) -> bool:
        return self._progress is not None

    def sizeHint(self):
        hint = super().sizeHint()
        width = self.fontMetrics().horizontalAdvance("99 of 99, choosing the operator… 100 %") + 36
        hint.setWidth(max(hint.width(), width))
        hint.setHeight(max(hint.height(), 38))
        return hint

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        shape = QPainterPath()
        shape.addRoundedRect(box, 8, 8)
        painter.setClipPath(shape)
        if self._progress is None:
            base = theme.IVORY if self.isEnabled() else theme.VELVET_HI
            painter.fillRect(box, theme.colour("#FFFFFF" if (self.underMouse() and self.isEnabled()) else base))
            painter.setPen(theme.colour(theme.INK if self.isEnabled() else "#6B6775"))
        else:
            painter.fillRect(box, theme.colour(theme.VELVET_HI))
            painter.fillRect(QRectF(box.left(), box.top(), box.width() * self._progress, box.height()),
                             theme.colour(theme.GREEN, 150))
            painter.setPen(theme.colour(theme.IVORY))
        painter.setFont(self.font())
        painter.drawText(box, Qt.AlignCenter, self.text())
        if self.hasFocus():
            painter.setClipping(False)
            painter.setPen(theme.colour(theme.MIST))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(box, 8, 8)
        painter.end()


class ActionBar(QWidget):
    convert_clicked = Signal()
    cancel_clicked = Signal()
    reveal_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("bar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(12)
        self.looks = LookBar()
        layout.addWidget(self.looks)
        layout.addStretch(1)
        self.summary = ElidedLabel(Qt.ElideLeft)
        self.summary.setProperty("role", "muted")
        self.summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.summary, 2)
        self.reveal = QPushButton("Show in folder")
        self.reveal.setProperty("role", "ghost")
        self.reveal.setCursor(Qt.PointingHandCursor)
        self.reveal.hide()
        self.reveal.clicked.connect(self.reveal_clicked)
        layout.addWidget(self.reveal)
        self.cancel = QPushButton("Cancel")
        self.cancel.setProperty("role", "ghost")
        self.cancel.setCursor(Qt.PointingHandCursor)
        self.cancel.hide()
        self.cancel.clicked.connect(self.cancel_clicked)
        layout.addWidget(self.cancel)
        self.convert = ConvertButton()
        self.convert.clicked.connect(self._on_convert)
        layout.addWidget(self.convert)

    def _on_convert(self) -> None:
        if not self.convert.working():
            self.convert_clicked.emit()

    def set_summary(self, text: str, done: bool = False, tooltip: str = "") -> None:
        """What a conversion will write, or (``done``) what the last one wrote: the news of
        a success is given here, in the place that already exists, not in a line of its own."""
        self._full_summary = text
        role = "done" if done else "muted"
        if self.summary.property("role") != role:
            self.summary.setProperty("role", role)
            self.summary.style().unpolish(self.summary)
            self.summary.style().polish(self.summary)
        # A file's name is cut in its middle: "Saved" and the end of the name both stay.
        self.summary.set_full_text(text, tooltip, Qt.ElideMiddle if done else Qt.ElideLeft)

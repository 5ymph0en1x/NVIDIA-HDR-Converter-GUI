"""The files waiting: one chip each, shown only when there is more than one file."""
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QScrollArea, QWidget

MARKS = {"waiting": "", "working": "…  ", "done": "✓  ", "failed": "!  ", "cancelled": ""}


class QueueStrip(QScrollArea):
    selected = Signal(int)
    removed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("queue")
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFixedHeight(52)
        inner = QWidget()
        inner.setObjectName("queue")
        inner.setAttribute(Qt.WA_StyledBackground, True)
        self._layout = QHBoxLayout(inner)
        self._layout.setContentsMargins(14, 8, 14, 8)
        self._layout.setSpacing(4)
        self._layout.addStretch(1)
        self.setWidget(inner)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.chips = []

    def set_items(self, items, selected: int) -> None:
        for chip in self.chips:                    # gone now, not at some later turn of the event loop
            self._group.removeButton(chip)
            self._layout.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self.chips = []
        for index, item in enumerate(items):
            chip = QPushButton(MARKS.get(item.state, "") + os.path.basename(item.path))
            chip.setCheckable(True)
            chip.setChecked(index == selected)
            chip.setProperty("role", "chip")
            chip.setProperty("state", item.state)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setFocusPolicy(Qt.TabFocus)
            chip.setToolTip(item.message or item.path)
            chip.setContextMenuPolicy(Qt.CustomContextMenu)
            chip.clicked.connect(lambda _checked, i=index: self.selected.emit(i))
            chip.customContextMenuRequested.connect(lambda _pos, i=index: self.removed.emit(i))
            self._group.addButton(chip)
            self._layout.insertWidget(index, chip)
            self.chips.append(chip)
        self.setVisible(len(items) > 1)

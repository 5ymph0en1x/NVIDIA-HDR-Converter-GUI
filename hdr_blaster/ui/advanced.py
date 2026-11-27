"""Everything that is not one of the four looks: a panel that slides in from the right."""
import os
from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget)

from .widgets import CloseButton, Combo, Switch
from ..session import (IMAGE_FORMATS, OPERATORS, RANGES, VIDEO_FRAME_RATES, VIDEO_HEIGHTS, VIDEO_QUALITIES,
                       Options)

PANEL_WIDTH = 340
QUALITY_NAMES = {"high": "High", "balanced": "Balanced", "compact": "Compact"}
HEIGHT_NAMES = {0: "Original size", 1080: "1080p", 720: "720p"}
RATE_NAMES = {0: "Original", 120: "120 fps", 60: "60 fps", 30: "30 fps"}
FORMAT_NAMES = {"JPEG": "JPEG", "TIFF": "TIFF", "Both": "JPEG and TIFF"}


def breakable(path: str, run: int = 18) -> str:
    """The same path with invisible places where a line may break: after each separator,
    and inside a name too long for the panel. A label breaks a text between its words only,
    and a path is one word: left as it is, it asks for its whole length in one line."""
    out, since = [], 0
    for char in path:
        out.append(char)
        since += 1
        if char in "\\/_-. " or since >= run:
            out.append("\u200b")
            since = 0
    return "".join(out)


class AdvancedPanel(QWidget):
    changed = Signal(object)            # the new Options
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        # The panel is a child of the picture area. A click that none of its controls takes
        # (its ground, a label) stops here: it must not reach the picture and move the divider.
        self.setAttribute(Qt.WA_NoMousePropagation, True)
        self.setFixedWidth(PANEL_WIDTH)
        self._options = Options()
        self._loading = False
        self._cuda = None                    # whether a graphics card can be used: not known until the engine is loaded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.setContentsMargins(18, 14, 12, 6)
        title = QLabel("Advanced")
        font = title.font()
        font.setPixelSize(15)
        font.setWeight(font.Weight.DemiBold)
        title.setFont(font)
        head.addWidget(title)
        head.addStretch(1)
        close = CloseButton()
        close.clicked.connect(self.close_requested)
        head.addWidget(close)
        outer.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(18, 4, 18, 18)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self._grid, self._row = grid, 0

        self._section("Picture")
        self.operator = self._combo("Operator", list(OPERATORS))
        # Exposure and gamma in hundredths: 100 on the slider is 1.00.
        self.exposure, self.exposure_value = self._slider("Exposure", *[round(v * 100) for v in RANGES["exposure"]], 100)
        self.gamma, self.gamma_value = self._slider("Gamma", *[round(v * 100) for v in RANGES["pre_gamma"]], 100)
        self.sharpness, self.sharpness_value = self._slider("Sharpness", 0, 100, 1)
        self.ai_color, self.ai_color_value = self._slider("AI colour", 0, 100, 1)
        self.video_enhance = self._check("Sharpness and AI colour on videos")
        self.video_enhance.setToolTip("Off, a video is converted without sharpening or AI colour, and analysed\n"
                                      "about twice as fast. On, both apply to videos as they do to captures.")

        self._section("Output")
        self.image_format = self._combo("Captures", [FORMAT_NAMES[f] for f in IMAGE_FORMATS])
        self.video_quality = self._combo("Video quality", [QUALITY_NAMES[q] for q in VIDEO_QUALITIES])
        self.video_height = self._combo("Video size", [HEIGHT_NAMES[h] for h in VIDEO_HEIGHTS])
        self.video_fps = self._combo("Frame rate", [RATE_NAMES[r] for r in VIDEO_FRAME_RATES])
        self.video_fps.setToolTip("Frames are dropped, never invented: the ones that stay are the ones the\n"
                                  "full conversion writes. A rate above the recording's keeps every frame.")
        self.video_blur = self._check("Motion blur at a lower frame rate")
        self.video_blur.setToolTip("Each frame written holds the light of the frames it replaces, added up\n"
                                   "as a longer exposure adds it. Motion is smoother, and softer.\n"
                                   "Nothing is invented; nothing changes at the original frame rate.")
        self.destination = QLabel()
        self.destination.setProperty("role", "muted")
        self.destination.setTextFormat(Qt.PlainText)
        self.destination.setWordWrap(True)
        # The folder takes the width the panel has and as many lines as it needs. It never
        # asks for more width: the panel does not scroll sideways, so whatever is wider than
        # it pushes every control of the panel out of sight.
        policy = self.destination.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Ignored)
        self.destination.setSizePolicy(policy)
        self._add("Save to", self.destination)
        # one under the other: side by side, the two are wider than their column
        buttons = QVBoxLayout()
        buttons.setSpacing(6)
        self.choose_folder = QPushButton("Choose a folder…")
        self.choose_folder.setCursor(Qt.PointingHandCursor)
        self.choose_folder.setFocusPolicy(Qt.TabFocus)
        self.choose_folder.clicked.connect(self._choose_folder)
        self.reset_folder = QPushButton("Next to the source")
        self.reset_folder.setProperty("role", "ghost")
        self.reset_folder.setCursor(Qt.PointingHandCursor)
        self.reset_folder.setFocusPolicy(Qt.TabFocus)
        self.reset_folder.clicked.connect(lambda: self._set("destination", ""))
        buttons.addWidget(self.choose_folder, 0, Qt.AlignLeft)
        buttons.addWidget(self.reset_folder, 0, Qt.AlignLeft)
        grid.addLayout(buttons, self._row, 1)
        self._row += 1

        self._section("Processing")
        self.use_gpu = self._check("Use the graphics card")
        self.use_fp16 = self._check("Half precision (less memory)")
        self.show_histogram = self._check("Show the histogram")
        grid.setRowStretch(self._row, 1)

        self.operator.currentTextChanged.connect(lambda text: self._set("operator", OPERATORS[text]))
        self.exposure.valueChanged.connect(lambda v: self._set("exposure", v / 100.0))
        self.gamma.valueChanged.connect(lambda v: self._set("pre_gamma", v / 100.0))
        self.sharpness.valueChanged.connect(lambda v: self._set("sharpness", float(v)))
        self.ai_color.valueChanged.connect(lambda v: self._set("ai_color", float(v)))
        self.video_enhance.toggled.connect(lambda v: self._set("video_enhance", bool(v)))
        self.image_format.currentIndexChanged.connect(lambda i: self._set("image_format", IMAGE_FORMATS[i]))
        self.video_quality.currentIndexChanged.connect(lambda i: self._set("video_quality", VIDEO_QUALITIES[i]))
        self.video_height.currentIndexChanged.connect(lambda i: self._set("video_height", VIDEO_HEIGHTS[i]))
        self.video_fps.currentIndexChanged.connect(lambda i: self._set("video_fps", VIDEO_FRAME_RATES[i]))
        self.video_blur.toggled.connect(lambda v: self._set("video_blur", bool(v)))
        self.use_gpu.toggled.connect(lambda v: self._set("use_gpu", bool(v)))
        self.use_fp16.toggled.connect(lambda v: self._set("use_fp16", bool(v)))
        self.show_histogram.toggled.connect(lambda v: self._set("show_histogram", bool(v)))
        self.set_options(self._options)

    # ------------------------------------------------------------- building
    def _section(self, text: str) -> None:
        label = QLabel(text)
        label.setProperty("role", "section")
        self._grid.addWidget(label, self._row, 0, 1, 2)
        self._row += 1

    def _add(self, text: str, widget: QWidget) -> None:
        self._grid.addWidget(QLabel(text), self._row, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self._grid.addWidget(widget, self._row, 1)
        self._row += 1

    def _combo(self, text: str, entries) -> QComboBox:
        box = Combo()
        box.addItems(entries)
        self._add(text, box)
        return box

    def _slider(self, text: str, low: int, high: int, divisor: int):
        """A slider with its value shown beside it. ``divisor`` 100 shows hundredths as 1.00."""
        row = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setPageStep(max(1, (high - low) // 10))
        value = QLabel()
        value.setProperty("role", "muted")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def show(v):
            value.setText(f"{v / divisor:.2f}" if divisor != 1 else str(v))

        show(high)
        value.setMinimumWidth(value.sizeHint().width())      # room for the widest value
        slider.valueChanged.connect(show)
        show(slider.value())
        row.addWidget(slider, 1)
        row.addWidget(value)
        holder = QWidget()
        holder.setObjectName("row")               # no ground of its own: the panel shows through
        holder.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        self._add(text, holder)
        return slider, value

    def _check(self, text: str) -> QCheckBox:
        box = Switch(text)
        self._grid.addWidget(box, self._row, 0, 1, 2)
        self._row += 1
        return box

    # ---------------------------------------------------------------- state
    def options(self) -> Options:
        return self._options

    def set_cuda(self, available: bool) -> None:
        """What the engine found once loaded. Without a graphics card the switch is off and
        cannot be turned on; the option itself is left as the user set it."""
        self._cuda = bool(available)
        self.use_gpu.setEnabled(self._cuda)
        self.use_gpu.setToolTip("" if self._cuda
                                else "No CUDA graphics card was found: conversions run on the processor.")
        self.set_options(self._options)

    def set_options(self, options: Options) -> None:
        """Show these options without announcing a change."""
        self._options = options
        self._loading = True
        try:
            self.operator.setCurrentText(next(k for k, v in OPERATORS.items() if v == options.operator))
            self.exposure.setValue(int(round(options.exposure * 100)))
            self.gamma.setValue(int(round(options.pre_gamma * 100)))
            self.sharpness.setValue(int(round(options.sharpness)))
            self.ai_color.setValue(int(round(options.ai_color)))
            self.video_enhance.setChecked(options.video_enhance)
            self.image_format.setCurrentIndex(IMAGE_FORMATS.index(options.image_format))
            self.video_quality.setCurrentIndex(VIDEO_QUALITIES.index(options.video_quality))
            self.video_height.setCurrentIndex(VIDEO_HEIGHTS.index(options.video_height))
            self.video_fps.setCurrentIndex(VIDEO_FRAME_RATES.index(options.video_fps))
            self.video_blur.setChecked(options.video_blur)
            self.use_gpu.setChecked(options.use_gpu and self._cuda is not False)
            self.use_fp16.setChecked(options.use_fp16)
            self.show_histogram.setChecked(options.show_histogram)
            self._refresh()
        finally:
            self._loading = False

    def _refresh(self) -> None:
        folder = self._options.destination
        self.destination.setText(breakable(folder) if folder else "Next to the source file")
        self.destination.setToolTip(folder)
        self.reset_folder.setVisible(bool(folder))
        self.use_fp16.setEnabled(self.use_gpu.isChecked() and self.use_gpu.isEnabled())
        self.video_blur.setEnabled(bool(self._options.video_fps))

    def _set(self, name: str, value) -> None:
        if self._loading or getattr(self._options, name) == value:
            return
        self._options = replace(self._options, **{name: value})
        self._refresh()
        self.changed.emit(self._options)

    def _choose_folder(self) -> None:
        start = self._options.destination or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Save converted files to", start)
        if folder:
            self._set("destination", folder)

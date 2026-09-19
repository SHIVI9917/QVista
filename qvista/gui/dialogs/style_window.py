"""
The design window that sits beside a plot.

One of these opens next to any figure in QVista. Everything it changes
is presentation only -- :mod:`qvista.visualization.plot_style` applies it
to the finished figure, so nothing here can alter a number.

Changes take effect as they are made rather than behind an Apply button,
because choosing a line width is a matter of looking at the result.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from qvista.gui import theme
from qvista.gui.dialogs.side_window import SideWindow
from qvista.visualization import plot_style as ps

#: Wide enough for two columns of controls, tall enough to avoid
#: scrolling on a laptop screen.
PREFERRED = 420

#: Shown in a spin box that is currently leaving the figure alone.
AUTO_TEXT = "auto"


class OptionalDouble(QDoubleSpinBox):
    """
    A spin box with an "leave it alone" state below its minimum.

    An axis limit, a line width and a font size all need a value that
    means "whatever the plot chose", and that is not a number. Stepping
    below the minimum shows ``auto`` and reports ``None``.
    """

    def __init__(
        self,
        low: float,
        high: float,
        step: float = 0.1,
        decimals: int = 2,
        parent=None,
    ) -> None:

        super().__init__(parent)

        # Before any Qt setter: setDecimals and setRange both call back
        # into textFromValue, which needs this to exist.
        self._empty = low - step

        self.setDecimals(decimals)
        self.setRange(self._empty, high)
        self.setSingleStep(step)
        self.setValue(self._empty)

    def textFromValue(self, value: float) -> str:

        if value <= self._empty:
            return AUTO_TEXT

        return super().textFromValue(value)

    def valueFromText(self, text: str) -> float:

        if text.strip().lower() in (AUTO_TEXT, ""):
            return self._empty

        return super().valueFromText(text)

    def value_or_none(self) -> float | None:
        """The number, or ``None`` when it is on ``auto``."""

        value = super().value()

        return None if value <= self._empty else float(value)

    def set_value_or_none(self, value: float | None) -> None:

        self.setValue(self._empty if value is None else float(value))

    def clear_to_auto(self) -> None:

        self.setValue(self._empty)


def _weight(box: QCheckBox) -> str:
    """A tick box, as the weight the style layer wants."""

    return "bold" if box.isChecked() else ps.AS_DRAWN


class StyleWindow(SideWindow):
    """Controls for how one figure looks."""

    #: Emitted whenever a control moves, with the current style.
    changed = Signal(object)

    #: Emitted when Reset is pressed.
    cleared = Signal()

    def __init__(self, title: str = "Plot Design", parent=None) -> None:

        super().__init__(f"QVista - {title}", parent, PREFERRED)

        self.style = ps.PlotStyle()

        # Set while loading a style into the controls, so that filling
        # them in does not fire a hundred changed signals.
        self._loading = False

        self.resize(PREFERRED, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        tabs = QTabWidget()

        tabs.addTab(self._wrap(self._build_lines()), "Lines")
        tabs.addTab(self._wrap(self._build_axes()), "Axes")
        tabs.addTab(self._wrap(self._build_text()), "Text")
        tabs.addTab(self._wrap(self._build_figure()), "Figure")

        layout.addWidget(tabs, 1)

        self.status = QLabel("Everything is as the plot drew it.")
        self.status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.status.setWordWrap(True)

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        reset = QPushButton("Reset")
        reset.setToolTip("Put the figure back exactly as it was drawn.")
        reset.clicked.connect(self._reset)

        footer.addWidget(reset)
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.hide)

        footer.addWidget(close)

        layout.addLayout(footer)

    # ------------------------------------------------------------

    def _wrap(self, widget: QWidget) -> QScrollArea:
        """Let a tab scroll rather than squash its controls."""

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        area.setFrameShape(QScrollArea.NoFrame)

        return area

    # ------------------------------------------------------------
    # Lines and colour
    # ------------------------------------------------------------

    def _build_lines(self) -> QWidget:

        page = QWidget()
        column = QVBoxLayout(page)

        box = QGroupBox("Lines")
        form = QFormLayout(box)

        self.line_width = OptionalDouble(ps.MIN_WIDTH, ps.MAX_WIDTH, 0.1)
        self.line_width.setToolTip(
            "Applies to the data only. Fermi levels, zero lines and "
            "high-symmetry separators keep the width they were drawn "
            "with."
        )
        form.addRow("Width", self.line_width)

        self.line_style = QComboBox()
        self.line_style.addItems(list(ps.LINE_STYLES))
        form.addRow("Style", self.line_style)

        self.alpha = OptionalDouble(0.05, 1.0, 0.05)
        self.alpha.setToolTip(
            "Transparency. Useful where many bands overlap."
        )
        form.addRow("Opacity", self.alpha)

        self.marker_size = OptionalDouble(0.5, 20.0, 0.5)
        self.marker_size.setToolTip(
            "Markers and scatter points, including the fat-band circles."
        )
        form.addRow("Marker size", self.marker_size)

        column.addWidget(box)

        colour = QGroupBox("Colour")
        colour_form = QFormLayout(colour)

        self.palette = QComboBox()
        self.palette.addItems(list(ps.PALETTES) + [ps.CUSTOM])
        self.palette.setToolTip(
            "Handed out along the labelled series, in the order the "
            "legend lists them."
        )
        colour_form.addRow("Palette", self.palette)

        self.swatches = QLabel("")
        self.swatches.setMinimumHeight(18)

        colour_form.addRow("", self.swatches)

        # Typed hex codes, for matching a journal's house style or a
        # figure somebody else already made.
        self.custom = QLineEdit()
        self.custom.setPlaceholderText("#0072b2, #e69f00, #009e73")
        self.custom.setToolTip(
            "Hex codes, separated by commas or spaces. Three, six or "
            "eight digits; the leading # is optional. Used when the "
            "palette above is set to Custom."
        )

        colour_form.addRow("Hex codes", self.custom)

        self.custom_note = QLabel("")
        self.custom_note.setWordWrap(True)

        colour_form.addRow("", self.custom_note)

        note = QLabel(
            "Colourblind safe is the Okabe-Ito set: it stays readable "
            "under the common forms of colour blindness and in "
            "greyscale."
        )
        note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        note.setWordWrap(True)

        colour_form.addRow(note)

        column.addWidget(colour)
        column.addStretch(1)

        for widget in (self.line_width, self.alpha, self.marker_size):
            widget.valueChanged.connect(self._emit)

        for widget in (self.line_style, self.palette):
            widget.currentIndexChanged.connect(self._emit)

        self.palette.currentIndexChanged.connect(self._draw_swatches)

        # Typing a colour switches to Custom, so the codes take effect
        # without a second click on a combo box the user has already
        # told what they want by typing into the box below it.
        self.custom.textChanged.connect(self._custom_typed)

        self._draw_swatches()

        return page

    def _custom_typed(self) -> None:
        """React to hex codes being typed."""

        good, bad = ps.parse_colours(self.custom.text())

        if bad:
            self.custom_note.setStyleSheet(f"color: {theme.AMBER};")
            self.custom_note.setText(
                f"Not a hex colour, ignored: {', '.join(bad)}"
            )
        elif good:
            self.custom_note.setStyleSheet(f"color: {theme.TEXT_DIM};")
            self.custom_note.setText(
                f"{len(good)} colour(s), cycled across the series."
            )
        else:
            self.custom_note.setText("")

        if good and self.palette.currentText() != ps.CUSTOM:
            # Switching fires _emit through currentIndexChanged.
            self.palette.setCurrentText(ps.CUSTOM)
            return

        self._draw_swatches()
        self._emit()

    def _draw_swatches(self) -> None:
        """Show the palette as a row of blocks."""

        chosen = self.palette.currentText()

        if chosen == ps.CUSTOM:
            colours = tuple(ps.parse_colours(self.custom.text())[0])
        else:
            colours = ps.PALETTES.get(chosen, ())

        if not colours:

            if chosen == ps.CUSTOM:
                self.swatches.setText(
                    "<span style='color:#7f8c8d'>type some hex codes "
                    "below</span>"
                )
                return
            self.swatches.setText(
                "<span style='color:#7f8c8d'>colours left as drawn</span>"
            )
            return

        self.swatches.setText(
            "".join(
                f"<span style='background:{colour}; color:{colour}'>"
                f"&nbsp;&nbsp;&nbsp;</span>&nbsp;"
                for colour in colours
            )
        )

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------

    def _build_axes(self) -> QWidget:

        page = QWidget()
        column = QVBoxLayout(page)

        box = QGroupBox("Range")
        form = QFormLayout(box)

        self.x_min = OptionalDouble(-1e6, 1e6, 0.5, 3)
        self.x_max = OptionalDouble(-1e6, 1e6, 0.5, 3)
        self.y_min = OptionalDouble(-1e6, 1e6, 0.5, 3)
        self.y_max = OptionalDouble(-1e6, 1e6, 0.5, 3)

        for label, widget in (
            ("X from", self.x_min),
            ("X to", self.x_max),
            ("Y from", self.y_min),
            ("Y to", self.y_max),
        ):
            widget.setToolTip(
                "Leave on auto to keep whatever the plot chose. Either "
                "end can be set on its own."
            )
            form.addRow(label, widget)

        clear = QPushButton("Back to auto")
        clear.clicked.connect(self._clear_ranges)

        form.addRow("", clear)

        column.addWidget(box)

        grid_box = QGroupBox("Grid and ticks")
        grid_form = QFormLayout(grid_box)

        self.grid = QComboBox()
        self.grid.addItems(list(ps.GRID_CHOICES))
        grid_form.addRow("Grid", self.grid)

        self.grid_alpha = QDoubleSpinBox()
        self.grid_alpha.setRange(0.05, 1.0)
        self.grid_alpha.setSingleStep(0.05)
        self.grid_alpha.setValue(0.3)
        grid_form.addRow("Grid opacity", self.grid_alpha)

        self.tick_direction = QComboBox()
        self.tick_direction.addItems(list(ps.TICK_DIRECTIONS))
        grid_form.addRow("Ticks point", self.tick_direction)

        column.addWidget(grid_box)
        column.addStretch(1)

        for widget in (self.x_min, self.x_max, self.y_min, self.y_max,
                       self.grid_alpha):
            widget.valueChanged.connect(self._emit)

        for widget in (self.grid, self.tick_direction):
            widget.currentIndexChanged.connect(self._emit)

        return page

    def _clear_ranges(self) -> None:

        self._loading = True

        for widget in (self.x_min, self.x_max, self.y_min, self.y_max):
            widget.clear_to_auto()

        self._loading = False

        self._emit()

    # ------------------------------------------------------------
    # Text
    # ------------------------------------------------------------

    def _build_text(self) -> QWidget:

        page = QWidget()
        column = QVBoxLayout(page)

        box = QGroupBox("Labels")
        form = QFormLayout(box)

        self.title = QLineEdit()
        self.title.setPlaceholderText("leave empty to keep the plot's own")

        self.x_label = QLineEdit()
        self.x_label.setPlaceholderText("as drawn")

        self.y_label = QLineEdit()
        self.y_label.setPlaceholderText("as drawn")

        for field in (self.title, self.x_label, self.y_label):
            field.setToolTip(
                "Underscore for a subscript and caret for a "
                "superscript: LiCoO_2, E_F, eV^{-1}. Braces for more "
                "than one character."
            )

        form.addRow("Title", self.title)
        form.addRow("X label", self.x_label)
        form.addRow("Y label", self.y_label)

        self.mathtext = QCheckBox("Read _ and ^ as subscript and superscript")
        self.mathtext.setChecked(True)
        self.mathtext.setToolTip(
            "LiCoO_2 becomes LiCoO with a subscript 2. Turn this off "
            "for a label that genuinely contains an underscore, or "
            r"escape it as \_."
        )

        form.addRow("", self.mathtext)

        self.preview = QLabel("")
        self.preview.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.preview.setWordWrap(True)

        form.addRow("Reads as", self.preview)

        column.addWidget(box)

        sizes = QGroupBox("Font sizes")
        size_form = QFormLayout(sizes)

        self.title_size = OptionalDouble(ps.MIN_FONT, ps.MAX_FONT, 1.0, 1)
        self.label_size = OptionalDouble(ps.MIN_FONT, ps.MAX_FONT, 1.0, 1)
        self.tick_size = OptionalDouble(ps.MIN_FONT, ps.MAX_FONT, 1.0, 1)
        self.legend_size = OptionalDouble(ps.MIN_FONT, ps.MAX_FONT, 1.0, 1)

        size_form.addRow("Title", self.title_size)
        size_form.addRow("Axis labels", self.label_size)
        size_form.addRow("Tick labels", self.tick_size)
        size_form.addRow("Legend", self.legend_size)

        column.addWidget(sizes)

        weights = QGroupBox("Weight")
        weight_form = QFormLayout(weights)

        self.title_bold = QCheckBox("Bold title")
        self.label_bold = QCheckBox("Bold axis labels")
        self.tick_bold = QCheckBox("Bold tick labels")

        for widget in (self.title_bold, self.label_bold, self.tick_bold):
            weight_form.addRow("", widget)

        column.addWidget(weights)

        legend = QGroupBox("Legend")
        legend_form = QFormLayout(legend)

        self.legend = QComboBox()
        self.legend.addItems(list(ps.LEGEND_CHOICES))
        legend_form.addRow("Show", self.legend)

        self.legend_location = QComboBox()
        self.legend_location.addItems(list(ps.LEGEND_LOCATIONS))
        legend_form.addRow("Position", self.legend_location)

        self.legend_columns = QSpinBox()
        self.legend_columns.setRange(1, 8)
        self.legend_columns.setToolTip(
            "More than one column keeps a long legend from covering the "
            "figure."
        )
        legend_form.addRow("Columns", self.legend_columns)

        self.legend_frame = QCheckBox("Draw a box around it")
        self.legend_frame.setChecked(True)
        legend_form.addRow("", self.legend_frame)

        column.addWidget(legend)
        column.addStretch(1)

        for widget in (self.title, self.x_label, self.y_label):
            widget.textChanged.connect(self._emit)
            widget.textChanged.connect(self._show_preview)

        self.mathtext.toggled.connect(self._emit)
        self.mathtext.toggled.connect(self._show_preview)

        for widget in (self.title_bold, self.label_bold, self.tick_bold):
            widget.toggled.connect(self._emit)

        for widget in (self.title_size, self.label_size, self.tick_size,
                       self.legend_size):
            widget.valueChanged.connect(self._emit)

        for widget in (self.legend, self.legend_location):
            widget.currentIndexChanged.connect(self._emit)

        self.legend_columns.valueChanged.connect(self._emit)
        self.legend_frame.toggled.connect(self._emit)

        return page

    def _show_preview(self) -> None:
        """Say what the label text will actually come out as."""

        typed = [
            text
            for text in (
                self.title.text(),
                self.x_label.text(),
                self.y_label.text(),
            )
            if text
        ]

        if not typed:
            self.preview.setText("")
            return

        if not self.mathtext.isChecked():
            self.preview.setText("taken literally")
            return

        self.preview.setText(
            " | ".join(ps.to_mathtext(text) for text in typed)
        )

    # ------------------------------------------------------------
    # The figure itself
    # ------------------------------------------------------------

    def _build_figure(self) -> QWidget:

        page = QWidget()
        column = QVBoxLayout(page)

        box = QGroupBox("Size")
        form = QFormLayout(box)

        self.width = OptionalDouble(ps.MIN_INCHES, ps.MAX_INCHES, 0.5, 1)
        self.height = OptionalDouble(ps.MIN_INCHES, ps.MAX_INCHES, 0.5, 1)

        self.width.setToolTip(
            "Inches. Sets the aspect ratio of the exported file; on "
            "screen the window still decides how big it looks."
        )

        form.addRow("Width", self.width)
        form.addRow("Height", self.height)

        self.tight = QCheckBox("Fit the labels inside the canvas")
        self.tight.setChecked(True)

        form.addRow("", self.tight)

        column.addWidget(box)

        export = QGroupBox("Export")
        export_form = QFormLayout(export)

        self.export_dpi = QSpinBox()
        self.export_dpi.setRange(ps.MIN_DPI, ps.MAX_DPI)
        self.export_dpi.setSingleStep(50)
        self.export_dpi.setValue(600)
        self.export_dpi.setToolTip(
            "Resolution used by Export figure. 300 is enough for a "
            "report, 600 for most journals."
        )

        export_form.addRow("Resolution", self.export_dpi)

        note = QLabel(
            "Everything here is presentation only. None of it can "
            "change a number on the figure."
        )
        note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        note.setWordWrap(True)

        export_form.addRow(note)

        column.addWidget(export)
        column.addStretch(1)

        for widget in (self.width, self.height):
            widget.valueChanged.connect(self._emit)

        self.export_dpi.valueChanged.connect(self._emit)
        self.tight.toggled.connect(self._emit)

        return page

    # ------------------------------------------------------------
    # Reading the controls
    # ------------------------------------------------------------

    def current(self) -> ps.PlotStyle:
        """Build a style from where the controls are now."""

        return ps.PlotStyle(
            line_width=self.line_width.value_or_none(),
            line_style=self.line_style.currentText(),
            alpha=self.alpha.value_or_none(),
            marker_size=self.marker_size.value_or_none(),
            palette=self.palette.currentText(),
            custom_colours=tuple(
                ps.parse_colours(self.custom.text())[0]
            ),
            x_min=self.x_min.value_or_none(),
            x_max=self.x_max.value_or_none(),
            y_min=self.y_min.value_or_none(),
            y_max=self.y_max.value_or_none(),
            grid=self.grid.currentText(),
            grid_alpha=self.grid_alpha.value(),
            tick_direction=self.tick_direction.currentText(),
            title=self.title.text(),
            x_label=self.x_label.text(),
            y_label=self.y_label.text(),
            title_size=self.title_size.value_or_none(),
            label_size=self.label_size.value_or_none(),
            tick_size=self.tick_size.value_or_none(),
            legend_size=self.legend_size.value_or_none(),
            title_weight=_weight(self.title_bold),
            label_weight=_weight(self.label_bold),
            tick_weight=_weight(self.tick_bold),
            mathtext=self.mathtext.isChecked(),
            legend=self.legend.currentText(),
            legend_location=self.legend_location.currentText(),
            legend_frame=self.legend_frame.isChecked(),
            legend_columns=self.legend_columns.value(),
            width=self.width.value_or_none(),
            height=self.height.value_or_none(),
            export_dpi=self.export_dpi.value(),
            tight=self.tight.isChecked(),
        )

    def _emit(self) -> None:

        if self._loading:
            return

        self.style = self.current()

        rows = ps.summary(self.style)

        self.status.setText(
            "; ".join(f"{name} {value}" for name, value in rows)
            or "Everything is as the plot drew it."
        )

        self.changed.emit(self.style)

    def _reset(self) -> None:
        """Put every control back to "as drawn"."""

        self._loading = True

        for widget in (
            self.line_width, self.alpha, self.marker_size,
            self.x_min, self.x_max, self.y_min, self.y_max,
            self.title_size, self.label_size, self.tick_size,
            self.legend_size, self.width, self.height,
        ):
            widget.clear_to_auto()

        self.line_style.setCurrentIndex(0)
        self.palette.setCurrentIndex(0)
        self.custom.clear()
        self.custom_note.setText("")
        self.grid.setCurrentIndex(0)
        self.grid_alpha.setValue(0.3)
        self.tick_direction.setCurrentIndex(0)

        for widget in (self.title, self.x_label, self.y_label):
            widget.clear()

        for widget in (self.title_bold, self.label_bold, self.tick_bold):
            widget.setChecked(False)

        self.mathtext.setChecked(True)

        self.legend.setCurrentIndex(0)
        self.legend_location.setCurrentIndex(0)
        self.legend_columns.setValue(1)
        self.legend_frame.setChecked(True)

        self.export_dpi.setValue(600)
        self.tight.setChecked(True)

        self._loading = False

        self.style = ps.PlotStyle()

        self._draw_swatches()
        self._show_preview()

        self.status.setText("Everything is as the plot drew it.")

        self.cleared.emit()

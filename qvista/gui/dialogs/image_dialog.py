"""
Export the viewport as an image.

Format, resolution, background. Everything else about the picture --
orientation, boundary, representation, which bonds are drawn -- is
already set in the viewport, and this window deliberately offers no way
to change any of it: the thing being exported is what is on screen.

Resolution is a multiple of the viewport rather than a pixel size, with
the result shown live. See :mod:`qvista.services.image_export` for why.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.services.image_export import (
    FORMATS,
    MAX_SCALE,
    MIN_SCALE,
    DEFAULT_SCALE,
    dialog_filter,
    output_size,
    save_image,
    supports_transparency,
)

#: The live pixel readout.
SIZE_STYLE = f"color: {theme.AQUA}; font-size: 10pt; font-weight: 600;"

#: What the chosen format is for.
NOTE_STYLE = theme.CAPTION_STYLE

#: Where a choice is unavailable and the reason matters.
WARN_STYLE = f"color: {theme.AMBER}; font-size: 9pt;"


class ImageDialog(QDialog):
    """Choose a format and a resolution, then write the viewport out."""

    def __init__(self, viewer, suggested: str = "structure", parent=None):

        super().__init__(parent)

        #: The plotter being photographed. Held rather than copied --
        #: the export renders it again at the chosen scale, so it has to
        #: be the live one.
        self.viewer = viewer

        #: What the save box opens on, from the structure's own filename
        #: so a folder of exports is not full of "structure.png".
        self.suggested = suggested or "structure"

        self.setWindowTitle("Export Image")
        fit_to_screen(self, 420, 300)
        self.resize(480, 380)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(8)

        # ----------------------------------------------------------
        # Format
        # ----------------------------------------------------------

        self.format_box = QComboBox()

        for fmt in FORMATS:
            self.format_box.addItem(f"{fmt.name}  ({fmt.suffix})", fmt)

        self.format_box.currentIndexChanged.connect(self._format_changed)

        form.addRow("Format", self.format_box)

        # ----------------------------------------------------------
        # Resolution
        # ----------------------------------------------------------

        self.scale_box = QSpinBox()
        self.scale_box.setRange(MIN_SCALE, MAX_SCALE)
        self.scale_box.setValue(DEFAULT_SCALE)
        self.scale_box.setPrefix("x ")
        self.scale_box.setToolTip(
            "Render at this multiple of the viewport. The view itself "
            "does not change -- only how finely it is sampled."
        )
        self.scale_box.valueChanged.connect(self._refresh_size)

        form.addRow("Resolution", self.scale_box)

        self.size_label = QLabel()
        self.size_label.setStyleSheet(SIZE_STYLE)

        form.addRow("", self.size_label)

        # ----------------------------------------------------------
        # Background
        # ----------------------------------------------------------

        self.transparent = QCheckBox("Transparent background")
        self.transparent.setToolTip(
            "Drop the viewport's background, so the structure can sit "
            "on a page or a slide of any colour."
        )

        form.addRow("Background", self.transparent)

        self.transparent_note = QLabel()
        self.transparent_note.setStyleSheet(WARN_STYLE)
        self.transparent_note.setWordWrap(True)

        form.addRow("", self.transparent_note)

        # ----------------------------------------------------------
        # Orientation marker
        # ----------------------------------------------------------

        self.include_axes = QCheckBox("Include orientation axes")
        self.include_axes.setChecked(False)
        self.include_axes.setToolTip(
            "The marker's letters are drawn at a fixed size while its "
            "arrows are not, so enlarging the render pulls the two "
            "apart and the labels come out oversized. Off by default "
            "for that reason."
        )

        form.addRow("Axes", self.include_axes)

        layout.addLayout(form)

        layout.addWidget(_hline())

        self.note = QLabel()
        self.note.setStyleSheet(NOTE_STYLE)
        self.note.setWordWrap(True)
        self.note.setAlignment(Qt.AlignTop)
        self.note.setMinimumHeight(48)

        layout.addWidget(self.note)

        layout.addStretch(1)

        layout.addLayout(self._build_buttons())

        self._format_changed()

    # ------------------------------------------------------------

    def _build_buttons(self) -> QHBoxLayout:

        buttons = QHBoxLayout()

        buttons.addStretch(1)

        save = QPushButton("Save Image...")
        save.setStyleSheet(SUCCESS_BUTTON_STYLE)
        save.clicked.connect(self._save)

        buttons.addWidget(save)

        close = QPushButton("Cancel")
        close.clicked.connect(self.reject)

        buttons.addWidget(close)

        return buttons

    # ------------------------------------------------------------
    # State
    # ------------------------------------------------------------

    def current_format(self):

        return self.format_box.currentData()

    def _format_changed(self, *_args) -> None:
        """
        Follow the chosen format.

        A vector export has no resolution to choose and no background to
        drop, so both controls are switched off rather than left looking
        as though they still apply.
        """

        fmt = self.current_format()

        self.note.setText(fmt.note)

        self.scale_box.setEnabled(not fmt.vector)

        allowed = supports_transparency(fmt)

        self.transparent.setEnabled(allowed)

        if not allowed:

            self.transparent.setChecked(False)

            self.transparent_note.setText(
                f"{fmt.name} carries no transparency."
                if not fmt.vector
                else "Vector output has no background to drop."
            )

        else:
            self.transparent_note.setText("")

        self._refresh_size()

    def _refresh_size(self, *_args) -> None:
        """Say what the export will actually come to."""

        fmt = self.current_format()

        if fmt.vector:
            self.size_label.setText("resolution independent")
            return

        width, height = output_size(
            self.viewport_size(), self.scale_box.value()
        )

        megapixels = width * height / 1.0e6

        self.size_label.setText(
            f"{width} x {height} px      {megapixels:.1f} MP"
        )

    def viewport_size(self) -> tuple[int, int]:
        """
        The viewport in pixels, as VTK sees it.

        Falls back to the Qt widget size if the render window has not
        been asked yet, which is the case before the first paint.
        """

        try:
            size = self.viewer.window_size

            if size is not None and size[0] and size[1]:
                return (int(size[0]), int(size[1]))

        except Exception:
            pass

        return (max(1, self.viewer.width()), max(1, self.viewer.height()))

    # ------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------

    def _save(self) -> None:

        fmt = self.current_format()

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            f"{self.suggested}{fmt.suffix}",
            dialog_filter(),
            fmt.filter,
        )

        if not filename:
            return

        try:
            written = save_image(
                self.viewer,
                filename,
                fmt,
                scale=self.scale_box.value(),
                transparent=self.transparent.isChecked(),
                axes=self.include_axes.isChecked(),
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export failed",
                f"Could not write {filename}:\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return

        QMessageBox.information(
            self,
            "Exported",
            f"Image written to\n{written}",
        )

        self.accept()

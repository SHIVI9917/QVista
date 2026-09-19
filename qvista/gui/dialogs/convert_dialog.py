"""
Format conversion.

One entry per conversion people actually ask for, rather than a source
format and a target format to be paired up. Picking one sets the file
filters at both ends, so the dialog cannot be driven into a combination
it does not support.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.sizing import fit_to_screen
from qvista.services import convert as cv

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

SOURCE_KEY = "convert/source"
TARGET_KEY = "convert/target"

NOTES_LINES = 4


class ConvertDialog(QDialog):
    """Pick a conversion, a file in and a file out."""

    def __init__(self, start: str = "", parent=None) -> None:
        """
        ``start`` may be a folder or a file.

        A folder is only where the browser opens. A file is the
        structure that is open in QVista, and it is filled in as the
        source -- converting the thing on screen is what the dialog is
        opened for nine times out of ten, and making the user find it
        again on disk is asking them to repeat work they have done.
        """

        super().__init__(parent)

        self.result: cv.Result | None = None

        #: The last output path this dialog suggested. Anything else in
        #: the target box was typed or browsed for, and is left alone.
        self._suggested = ""

        self.setWindowTitle("Convert")
        fit_to_screen(self, 760, 480)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ----------------------------------------------------------
        # Which conversion
        # ----------------------------------------------------------

        box = QGroupBox("Conversion")

        form = QFormLayout(box)

        self.choice = QComboBox()

        for item in cv.CONVERSIONS:
            self.choice.addItem(item.label, item.key)

        self.choice.currentIndexChanged.connect(self._picked)

        form.addRow("Convert", self.choice)

        self.source = QLineEdit()
        self.source.setPlaceholderText("the file to read")

        source_row = QHBoxLayout()
        source_row.addWidget(self.source, 1)

        browse_source = QPushButton("Browse...")
        browse_source.clicked.connect(self._browse_source)

        source_row.addWidget(browse_source)

        form.addRow("From", source_row)

        self.target = QLineEdit()
        self.target.setPlaceholderText("where to write it")

        target_row = QHBoxLayout()
        target_row.addWidget(self.target, 1)

        browse_target = QPushButton("Browse...")
        browse_target.clicked.connect(self._browse_target)

        target_row.addWidget(browse_target)

        form.addRow("To", target_row)

        # ----------------------------------------------------------
        # Only for the two that have to invent a cell
        # ----------------------------------------------------------

        self.vacuum_row = QWidget()

        vacuum_layout = QHBoxLayout(self.vacuum_row)
        vacuum_layout.setContentsMargins(0, 0, 0, 0)

        self.vacuum = QDoubleSpinBox()
        self.vacuum.setRange(0.0, 40.0)
        self.vacuum.setSingleStep(1.0)
        self.vacuum.setValue(cv.DEFAULT_VACUUM)
        self.vacuum.setSuffix(" A")
        self.vacuum.setToolTip(
            "On each side, so the gap between periodic images is twice "
            "this. Only used when the file carries no lattice of its "
            "own."
        )
        self.vacuum.valueChanged.connect(self._describe)

        vacuum_layout.addWidget(self.vacuum)
        vacuum_layout.addStretch(1)

        form.addRow("Vacuum", self.vacuum_row)

        layout.addWidget(box)

        self.hint = QLabel("")
        self.hint.setStyleSheet(STATUS_STYLE)
        self.hint.setWordWrap(True)

        layout.addWidget(self.hint)

        layout.addWidget(_hline())

        # ----------------------------------------------------------

        self.notes = QLabel("")
        self.notes.setStyleSheet(theme.WARNING_STYLE)
        self.notes.setWordWrap(True)
        self.notes.setAlignment(Qt.AlignTop)

        self.notes_area = QScrollArea()
        self.notes_area.setWidgetResizable(True)
        self.notes_area.setWidget(self.notes)
        self.notes_area.setFrameShape(QFrame.NoFrame)
        self.notes_area.setVisible(False)
        self.notes_area.setMaximumHeight(
            NOTES_LINES * self.notes.fontMetrics().lineSpacing()
            + 2 * self.notes_area.frameWidth()
        )

        layout.addWidget(self.notes_area, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self._say("Choose a conversion and the file to convert.")

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.run_button = QPushButton("Convert")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.clicked.connect(self._run)

        footer.addWidget(self.run_button)
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        # A file gets filled in; a folder is only a starting point for
        # the browser.
        loaded = Path(start) if start else None

        if loaded is not None and loaded.is_file():

            self._start = str(loaded.parent)

            # Point the conversion at the format actually loaded before
            # filling anything in, so the file filters and the suggested
            # output name agree with the source from the first moment.
            match = cv.conversion_for(loaded)

            if match is not None:

                index = self.choice.findData(match.key)

                if index >= 0:
                    self.choice.blockSignals(True)
                    self.choice.setCurrentIndex(index)
                    self.choice.blockSignals(False)

            self.source.setText(str(loaded))

            self._suggest_target(loaded.parent)

        else:
            self._start = start or ""

        self._picked()

        if self.source.text().strip():
            self._say(
                f"Loaded structure {Path(self.source.text()).name} is "
                f"filled in. Browse for a different one if that is not "
                f"the file to convert."
            )

    # ------------------------------------------------------------

    def _say(self, text: str, style: str = STATUS_STYLE) -> None:

        self.status.setStyleSheet(style)
        self.status.setText(text)

    @property
    def conversion(self) -> cv.Conversion:

        return cv.find(self.choice.currentData())

    # ------------------------------------------------------------

    def _picked(self) -> None:
        """Follow the choice: the vacuum row only applies to some."""

        conversion = self.conversion

        self.vacuum_row.setVisible(conversion.needs_cell)

        # The suggested output name follows the conversion, but never
        # overwrites something the user typed. A name this dialog put
        # there itself is not something the user typed -- leaving a
        # suggested .cif in place after switching to XYZ would write a
        # file whose contents and whose extension disagree.
        current = self.target.text().strip()

        if not current:
            self.target.setPlaceholderText(conversion.target_name)

        elif current == self._suggested:
            self._suggest_target(Path(current).parent)

        self._describe()

    def _suggest_target(self, folder: Path) -> None:
        """Offer an output beside ``folder``, and remember that we did."""

        self._suggested = str(folder / self.conversion.target_name)

        self.target.setText(self._suggested)

    def _describe(self) -> None:

        conversion = self.conversion

        parts = []

        if conversion.note:
            parts.append(conversion.note)

        if conversion.needs_cell:
            parts.append(
                f"A Gaussian file has no lattice unless it carries Tv "
                f"vectors. Without them a box is built with "
                f"{self.vacuum.value():g} A of vacuum on each side, so "
                f"{2 * self.vacuum.value():g} A between periodic images."
            )

        self.hint.setText("  ".join(parts))

    # ------------------------------------------------------------

    def _browse_source(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = (
            self.source.text()
            or self._start
            or settings.value(SOURCE_KEY, "", type=str)
        )

        chosen, _ = QFileDialog.getOpenFileName(
            self, "File to convert", start, self.conversion.source_filter
        )

        if not chosen:
            return

        self.source.setText(chosen)

        settings.setValue(SOURCE_KEY, chosen)

        # Suggest an output beside the input, which is where it is
        # almost always wanted.
        current = self.target.text().strip()

        if not current or current == self._suggested:
            self._suggest_target(Path(chosen).parent)

    def _browse_target(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.target.text() or settings.value(
            TARGET_KEY, "", type=str
        )

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Write to", start, self.conversion.target_filter
        )

        if not chosen:
            return

        self.target.setText(chosen)

        settings.setValue(TARGET_KEY, chosen)

    # ------------------------------------------------------------

    def _run(self) -> None:

        source = self.source.text().strip()
        target = self.target.text().strip()

        if not source:
            self._say("Choose a file to convert.", theme.WARNING_STYLE)
            return

        if not target:
            target = str(
                Path(source).parent / self.conversion.target_name
            )

            self.target.setText(target)

        try:
            self.result = cv.run(
                self.choice.currentData(),
                source,
                target,
                vacuum=self.vacuum.value(),
            )

        except cv.ConversionError as exc:
            self.result = None
            self.notes_area.setVisible(False)
            self._say(str(exc), theme.WARNING_STYLE)
            return

        except Exception as exc:  # noqa: BLE001 - shown, never swallowed
            self.result = None
            self.notes_area.setVisible(False)
            self._say(
                f"{type(exc).__name__}: {exc}", theme.WARNING_STYLE
            )
            return

        result = self.result

        self._say(
            f"Wrote {result.target.name}: {result.formula}, "
            f"{result.natoms} atoms."
        )

        self.notes.setText("\n\n".join(result.notes))
        self.notes_area.setVisible(bool(result.notes))

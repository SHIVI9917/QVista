"""
Read the elastic tensor out of a finished VASP run.

The other half of Mechanical Properties. The CHGNet route strains the
cell here and answers in a minute; this one reads what VASP already
worked out, and is the first-principles answer.

Both end in the same :class:`~qvista.analyses.elastic.ElasticResult` and
the same panel, so the two are directly comparable -- which is the point
of having both.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.sizing import fit_to_screen
from qvista.services import elastic_vasp as ev

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

RUN_KEY = "elastic_vasp/run_root"

NOTES_LINES = 5


class Reader(QThread):
    """Reading a long OUTCAR is slow enough to be worth a thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: str, parent=None) -> None:

        super().__init__(parent)

        self.folder = folder

    def run(self) -> None:

        try:
            self.completed.emit(ev.read_outcar(self.folder))

        except ev.ElasticVaspError as exc:
            self.failed.emit(str(exc))

        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ElasticVaspDialog(QDialog):
    """Point at a VASP elastic run and read its tensor."""

    def __init__(self, start: str = "", parent=None) -> None:

        super().__init__(parent)

        self.elastic: ev.VaspElastic | None = None
        self.result = None
        self.reader: Reader | None = None

        self.setWindowTitle("Mechanical Properties from VASP")
        fit_to_screen(self, 820, 520)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_source(layout)

        layout.addWidget(_hline())

        self._build_choice(layout)

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
        self._say("Choose the folder holding the finished VASP run.")

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.show_button = QPushButton("Show tensor")
        self.show_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.show_button.setEnabled(False)
        self.show_button.clicked.connect(self.accept)

        footer.addWidget(self.show_button)
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        if start:
            self.folder.setText(start)

    # ------------------------------------------------------------

    def _say(self, text: str, style: str = STATUS_STYLE) -> None:

        self.status.setStyleSheet(style)
        self.status.setText(text)

    def _build_source(self, layout: QVBoxLayout) -> None:

        row = QHBoxLayout()

        row.addWidget(QLabel("Run folder"))

        self.folder = QLineEdit()
        self.folder.setPlaceholderText("the directory holding OUTCAR")
        self.folder.editingFinished.connect(self._read)

        row.addWidget(self.folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        read = QPushButton("Read")
        read.clicked.connect(self._read)

        row.addWidget(read)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(140)

        row.addWidget(self.progress)

        layout.addLayout(row)

    def _build_choice(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Which tensor")

        column = QVBoxLayout(box)

        row = QHBoxLayout()

        self.block = QComboBox()

        for key, (_heading, description) in ev.BLOCKS.items():
            self.block.addItem(description, key)

        self.block.setEnabled(False)
        self.block.currentIndexChanged.connect(self._derive)

        row.addWidget(self.block, 1)

        column.addLayout(row)

        self.hint = QLabel(
            "VASP prints the clamped-ion tensor and what the ions "
            "relaxing adds. The total is the one to quote."
        )
        self.hint.setStyleSheet(STATUS_STYLE)
        self.hint.setWordWrap(True)

        column.addWidget(self.hint)

        layout.addWidget(box)

    # ------------------------------------------------------------

    def _browse(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.folder.text() or settings.value(RUN_KEY, "", type=str)

        chosen = QFileDialog.getExistingDirectory(
            self, "VASP elastic run folder", start
        )

        if not chosen:
            return

        self.folder.setText(chosen)

        settings.setValue(RUN_KEY, chosen)

        self._read()

    def _read(self) -> None:

        folder = self.folder.text().strip()

        if not folder:
            return

        if self.reader is not None and self.reader.isRunning():
            return

        self.progress.setVisible(True)
        self._say(f"Reading {Path(folder).name}...")

        self.reader = Reader(folder, self)
        self.reader.completed.connect(self._loaded)
        self.reader.failed.connect(self._failed)
        self.reader.start()

    def _loaded(self, elastic: ev.VaspElastic) -> None:

        self.progress.setVisible(False)

        self.elastic = elastic

        # Only offer the blocks this run actually printed, so a choice
        # cannot lead to an error message.
        available = elastic.available()

        self.block.blockSignals(True)
        self.block.clear()

        for key in available:
            self.block.addItem(ev.BLOCKS[key][1], key)

        self.block.blockSignals(False)
        self.block.setEnabled(len(available) > 1)

        self._derive()

    def _failed(self, message: str) -> None:

        self.progress.setVisible(False)

        self.elastic = None
        self.result = None

        self.block.setEnabled(False)
        self.show_button.setEnabled(False)

        self._say(message, theme.WARNING_STYLE)

    def _derive(self) -> None:
        """Turn the chosen block into moduli."""

        if self.elastic is None:
            return

        which = self.block.currentData() or "total"

        try:
            self.result = ev.to_result(self.elastic, which=which)

        except ev.ElasticVaspError as exc:
            self._failed(str(exc))
            return

        self.show_button.setEnabled(True)

        moduli = self.result.moduli

        self._say(
            f"Read {len(self.elastic.available())} block(s). "
            f"K = {moduli.bulk_hill:.1f} GPa, G = {moduli.shear_hill:.1f} "
            f"GPa, E = {moduli.youngs:.1f} GPa."
        )

        warnings = [
            note
            for note in self.result.notes
            if note.startswith("WARNING") or "Ionic relaxation" in note
        ]

        self.notes.setText("\n\n".join(warnings))
        self.notes_area.setVisible(bool(warnings))

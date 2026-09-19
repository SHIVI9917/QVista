"""
Read a finished VASP molecular-dynamics run.

The other half of MD. The CHGNet route integrates the trajectory here;
this one reads one VASP already produced, from OSZICAR and XDATCAR.

It ends in the same :class:`~qvista.analyses.md.MDResult` as the CHGNet
route, so the traces figure, the CSV and the trajectory export are shared
and the diffusion window opens straight onto the same trajectory.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.dialogs.plot_window import FigurePlaceholder, FigureWindow
from qvista.gui.sizing import fit_to_screen
from qvista.services import md_vasp
from qvista.visualization import dynamics_plots

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

RUN_KEY = "md_vasp/run_root"

NOTES_LINES = 5


class Reader(QThread):
    """XDATCAR for a long run takes a while to parse."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: str, parent=None) -> None:

        super().__init__(parent)

        self.folder = folder

    def run(self) -> None:

        try:
            self.completed.emit(md_vasp.read_run(self.folder))

        except md_vasp.MdVaspError as exc:
            self.failed.emit(str(exc))

        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MdVaspDialog(QDialog):
    """Point at a VASP AIMD run and read its traces."""

    def __init__(self, start: str = "", parent=None) -> None:

        super().__init__(parent)

        self.run: md_vasp.VaspMd | None = None
        self.result = None
        self.reader: Reader | None = None

        self.setWindowTitle("Molecular Dynamics from VASP")
        fit_to_screen(self, 900, 660)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_source(layout)

        layout.addWidget(_hline())

        self.figures = FigureWindow(
            "QVista - Molecular Dynamics", ("Energy and temperature",), self
        )

        self.panel = self.figures.panel_for()

        layout.addWidget(FigurePlaceholder(self.figures, self))

        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        layout.addWidget(self.table, 1)

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

        layout.addWidget(self.notes_area)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self._say("Choose the folder holding the finished VASP MD run.")

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.diffusion_button = QPushButton("Diffusion...")
        self.diffusion_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.diffusion_button.setEnabled(False)
        self.diffusion_button.clicked.connect(self._diffusion)

        footer.addWidget(self.diffusion_button)

        self.save_button = QPushButton("Save CSV...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)

        footer.addWidget(self.save_button)
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
        self.folder.setPlaceholderText(
            "the directory holding OSZICAR and XDATCAR"
        )
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

    # ------------------------------------------------------------

    def _browse(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.folder.text() or settings.value(RUN_KEY, "", type=str)

        chosen = QFileDialog.getExistingDirectory(
            self, "VASP molecular-dynamics run folder", start
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

    def _loaded(self, run: md_vasp.VaspMd) -> None:

        self.progress.setVisible(False)

        self.run = run
        self.result = md_vasp.to_result(run)

        self._fill()
        self._draw()

        self.diffusion_button.setEnabled(run.has_trajectory)
        self.save_button.setEnabled(True)

        self._say(
            f"{run.steps} steps, {run.simulated_time:.0f} fs, "
            f"mean T = {run.mean_temperature:.0f} K."
        )

        warnings = [
            note for note in run.notes if note.startswith("WARNING")
        ]

        others = [
            note for note in run.notes if not note.startswith("WARNING")
        ]

        self.notes.setText("\n\n".join(warnings + others))
        self.notes_area.setVisible(bool(run.notes))

    def _failed(self, message: str) -> None:

        self.progress.setVisible(False)

        self.run = None
        self.result = None

        self.diffusion_button.setEnabled(False)
        self.save_button.setEnabled(False)

        self._say(message, theme.WARNING_STYLE)

    # ------------------------------------------------------------

    def _draw(self) -> None:

        if self.result is None:
            return

        dynamics_plots.draw_md_traces(
            self.panel.figure,
            self.result,
            title=(
                f"{self.run.ensemble_label}, "
                f"{self.run.simulated_time:.0f} fs"
            ),
        )

        self.panel.refresh()

        self.figures.show_figure(reference=self)

    def _fill(self) -> None:

        if self.run is None:
            return

        rows = md_vasp._rows(self.run)

        self.table.clear()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.table.setRowCount(len(rows))

        for index, (label, value) in enumerate(rows):

            for column, text in enumerate((label, value)):

                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                self.table.setItem(index, column, item)

    # ------------------------------------------------------------

    def _diffusion(self) -> None:
        """Hand the trajectory to the diffusion window."""

        if self.run is None or self.run.trajectory is None:
            return

        from qvista.gui.dialogs.msd_dialog import MsdDialog

        MsdDialog(
            trajectory=self.run.trajectory,
            start=str(self.run.directory or ""),
            parent=self,
        ).exec()

    def _save(self) -> None:

        if self.result is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save traces", "vasp_md.csv", "CSV (*.csv)"
        )

        if not filename:
            return

        try:
            Path(filename).write_text(self.result.to_csv(), encoding="utf-8")

        except OSError as exc:
            self._say(f"Could not write the file: {exc}", theme.WARNING_STYLE)
            return

        self._say(f"Saved to {filename}.")

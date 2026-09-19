"""
Phonon density of states from cached force constants.

The forces are the expensive part and they are already done by the time
this opens, so everything here is re-runnable in seconds: change the
mesh, the projection or the broadening and press Run again.

:mod:`qvista.services.phonon_dos` does the work.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.dialogs.plot_window import FigurePlaceholder, FigureWindow
from qvista.services import phonon_dos as pd
from qvista.visualization import phonon_plots

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

#: Rows of the DOS worth putting in a table.  The curve itself has of
#: order 200 points, which is a plot, not a table; the peaks are what
#: anyone would read off it.
PEAK_COUNT = 8

#: Lines of notes shown before the area starts scrolling. Whole lines,
#: never a fraction of one.
NOTES_LINES = 4


def _status(label: QLabel, text: str, style: str = STATUS_STYLE) -> None:

    label.setStyleSheet(style)
    label.setText(text)


class Chip(QPushButton):
    """A small checkable box, matching the other analysis dialogs."""

    def __init__(self, text: str, parent=None) -> None:

        super().__init__(text, parent)

        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)


class DosRun(QThread):
    """The DOS and the thermal properties, off the GUI thread."""

    progressed = Signal(str)
    completed = Signal(object, object)
    failed = Signal(str)

    def __init__(self, phonon, settings: dict, thermal: dict, parent=None):

        super().__init__(parent)

        self.phonon = phonon
        self.settings = settings
        self.thermal = thermal

    def run(self) -> None:

        try:
            result = pd.run_dos(
                self.phonon,
                progress=self.progressed.emit,
                **self.settings,
            )

            # Deliberately after the DOS: it reuses the mesh that call
            # left behind, so the two always describe the same sampling.
            thermal = pd.run_thermal(
                self.phonon,
                progress=self.progressed.emit,
                **self.thermal,
            )

            self.completed.emit(result, thermal)

        except pd.PhononDosError as exc:
            self.failed.emit(str(exc))

        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class PhononDosDialog(QDialog):
    """Mesh, projection and broadening, over a plot and a table."""

    def __init__(self, phonon, force_field: str = "", parent=None) -> None:

        super().__init__(parent)

        self.phonon = phonon
        self.force_field = force_field

        self.result: pd.DosResult | None = None
        self.thermal: pd.ThermalResult | None = None
        self.worker: DosRun | None = None

        self.setWindowTitle("Phonon Density of States")
        fit_to_screen(self, 1120, 820)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_settings(layout)

        layout.addWidget(_hline())

        # The figures live in their own window now.
        layout.addWidget(self._build_figures())
        layout.addWidget(self._build_tables(), 1)

        self.notes = QLabel("")
        self.notes.setStyleSheet(theme.WARNING_STYLE)
        self.notes.setWordWrap(True)
        self.notes.setAlignment(Qt.AlignTop)

        self.notes_area = QScrollArea()
        self.notes_area.setWidgetResizable(True)
        self.notes_area.setWidget(self.notes)
        self.notes_area.setFrameShape(QFrame.NoFrame)
        self.notes_area.setVisible(False)

        # A whole number of lines: an arbitrary pixel height ends the
        # area mid-glyph, and half a line of text reads as two lines
        # printed over each other.
        self.notes_area.setMaximumHeight(
            NOTES_LINES * self.notes.fontMetrics().lineSpacing()
            + 2 * self.notes_area.frameWidth()
        )

        layout.addWidget(self.notes_area)

        self.status = QLabel("")
        _status(self.status, "Ready. Press Run.")
        self.status.setWordWrap(True)

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.save_button = QPushButton("Save table...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)

        footer.addWidget(self.save_button)
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        self._describe()

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_settings(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Sampling")

        column = QVBoxLayout(box)

        # -- mesh and projection --------------------------------
        top = QHBoxLayout()
        top.setSpacing(10)

        top.addWidget(QLabel("Mesh"))

        self.mesh = []

        for index in range(3):

            spin = QSpinBox()
            spin.setRange(1, 200)
            spin.setValue(pd.DEFAULT_MESH[index])
            spin.valueChanged.connect(lambda _: self._describe())

            self.mesh.append(spin)
            top.addWidget(spin)

        top.addSpacing(16)

        top.addWidget(QLabel("Resolve"))

        self.projection = "total"
        self.projection_chips: dict[str, Chip] = {}

        for key, label in (
            ("total", "Total"),
            ("element", "Element wise"),
            ("atom", "Atom wise"),
        ):
            chip = Chip(label)
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_projection(name)
            )

            self.projection_chips[key] = chip
            top.addWidget(chip)

        self.projection_chips["total"].setChecked(True)

        top.addStretch(1)

        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.clicked.connect(self._run)

        top.addWidget(self.run_button)

        column.addLayout(top)

        # -- broadening and units -------------------------------
        row = QHBoxLayout()
        row.setSpacing(10)

        row.addWidget(QLabel("Broadening"))

        self.method = QComboBox()

        for key in pd.METHODS:
            self.method.addItem(key, key)

        self.method.currentIndexChanged.connect(self._describe)

        row.addWidget(self.method)

        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.001, 10.0)
        self.sigma.setDecimals(3)
        self.sigma.setSingleStep(0.05)
        self.sigma.setValue(pd.DEFAULT_SIGMA)
        self.sigma.setPrefix("sigma ")
        self.sigma.setSuffix(" THz")
        self.sigma.setEnabled(False)

        row.addWidget(self.sigma)

        row.addSpacing(16)

        row.addWidget(QLabel("Units"))

        self.unit = QComboBox()

        for key in pd.UNITS:
            self.unit.addItem(key, key)

        self.unit.currentIndexChanged.connect(self._describe)

        row.addWidget(self.unit)

        row.addSpacing(16)

        row.addWidget(QLabel("Temperature to"))

        self.t_max = QDoubleSpinBox()
        self.t_max.setRange(10.0, 5000.0)
        self.t_max.setDecimals(0)
        self.t_max.setSingleStep(100.0)
        self.t_max.setValue(pd.DEFAULT_TEMPERATURES[1])
        self.t_max.setSuffix(" K")

        row.addWidget(self.t_max)

        row.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(180)

        row.addWidget(self.progress)

        column.addLayout(row)

        self.choice_note = QLabel("")
        self.choice_note.setStyleSheet(STATUS_STYLE)
        self.choice_note.setWordWrap(True)

        column.addWidget(self.choice_note)

        layout.addWidget(box)

    def _build_figures(self) -> QWidget:
        """Put the figures in a window of their own."""

        self.figures = FigureWindow(
            "QVista - Phonon Density of States",
            ("Density of states", "Thermal properties"),
            self,
        )

        self.dos_panel = self.figures.panel_for("Density of states")
        self.thermal_panel = self.figures.panel_for("Thermal properties")

        return FigurePlaceholder(self.figures, self)

    def _build_tables(self) -> QWidget:

        self.tables = QTabWidget()

        self.peaks = QTableWidget()
        self.peaks.verticalHeader().setVisible(False)
        self.peaks.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.thermal_table = QTableWidget()
        self.thermal_table.verticalHeader().setVisible(False)
        self.thermal_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.tables.addTab(self.peaks, "Peaks and weights")
        self.tables.addTab(self.thermal_table, "Thermal properties")

        return self.tables

    # ------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------

    def _pick_projection(self, name: str) -> None:

        self.projection = name

        for key, chip in self.projection_chips.items():
            chip.setChecked(key == name)

        self._describe()

    def _describe(self) -> None:
        """Say what the current settings will cost and mean."""

        self.sigma.setEnabled(self.method.currentData() == "smearing")

        grid = [spin.value() for spin in self.mesh]

        points = grid[0] * grid[1] * grid[2]

        text = pd.METHODS[self.method.currentData()]

        if self.projection == "total":
            text += (
                f"  --  {points} mesh points, reduced by symmetry."
            )
        else:
            text += (
                f"  --  a projection needs the eigenvectors, so all "
                f"{points} mesh points are sampled rather than the "
                f"symmetry-reduced set."
            )

        self.choice_note.setText(text)

    # ------------------------------------------------------------
    # Running
    # ------------------------------------------------------------

    def _run(self) -> None:

        if self.worker is not None and self.worker.isRunning():
            return

        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress.setVisible(True)

        settings = {
            "mesh": [spin.value() for spin in self.mesh],
            "projection": self.projection,
            "method": self.method.currentData(),
            "sigma": self.sigma.value(),
            "unit": self.unit.currentData(),
        }

        thermal = {"t_max": self.t_max.value()}

        self.worker = DosRun(self.phonon, settings, thermal, parent=self)

        self.worker.progressed.connect(
            lambda text: _status(self.status, text)
        )
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.worker.start()

    def _on_completed(
        self,
        result: pd.DosResult,
        thermal: pd.ThermalResult,
    ) -> None:

        self.result = result
        self.thermal = thermal

        self._draw()
        self._fill_peaks()
        self._fill_thermal()

        notes = list(result.notes) + list(thermal.notes)

        self.notes.setText("\n".join(f"- {note}" for note in notes))
        self.notes_area.setVisible(bool(notes))

        self.save_button.setEnabled(True)

        _status(
            self.status,
            f"Done. {result.integral:.3f} states over "
            f"{result.natoms} atoms, against {result.expected_integral:.0f} "
            f"expected, on {result.qpoints} q-points.",
            theme.SUCCESS_STYLE,
        )

    def _on_failed(self, message: str) -> None:

        _status(self.status, message, theme.ERROR_STYLE)

        QMessageBox.warning(self, "Phonon DOS", message)

    def _on_finished(self) -> None:

        self.progress.setVisible(False)
        self.run_button.setEnabled(True)

        self.worker = None

    # ------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------

    def _draw(self) -> None:

        if self.result is None:
            return

        phonon_plots.draw_dos_result(
            self.dos_panel.figure,
            self.result,
            force_field=self.force_field,
        )

        self.dos_panel.refresh()

        if self.thermal is not None:

            phonon_plots.draw_thermal(
                self.thermal_panel.figure,
                self.thermal,
                force_field=self.force_field,
            )

            self.thermal_panel.refresh()

        self.figures.show_figure("Density of states", reference=self)

    # ------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------

    def _fill_peaks(self) -> None:
        """Where the DOS peaks are, and what each curve is worth."""

        result = self.result

        if result is None:
            return

        peaks = pd.peak_frequencies(result, count=PEAK_COUNT)
        weights = pd.band_weights(result)

        columns = ["", "Frequency", "Height / modes", "Share"]

        self.peaks.clear()
        self.peaks.setColumnCount(len(columns))
        self.peaks.setHorizontalHeaderLabels(columns)
        self.peaks.setRowCount(len(peaks) + len(weights) + bool(weights))

        unit = pd.UNITS[result.unit]

        row = 0

        for frequency, height in peaks:

            self._cell(row, 0, "Peak" if row == 0 else "")
            self._cell(
                row, 1, f"{frequency:.{unit.decimals}f} {result.unit}"
            )
            self._cell(row, 2, f"{height:.4f}")
            self._cell(row, 3, "")

            row += 1

        if not weights:
            return

        self._cell(row, 0, "Modes", bold=True)
        self._cell(row, 1, "", bold=True)
        self._cell(row, 2, f"{result.integral:.3f}", bold=True)
        self._cell(
            row, 3, f"of {result.expected_integral:.0f}", bold=True
        )

        row += 1

        for label, modes in weights:

            share = (
                modes / result.expected_integral
                if result.expected_integral
                else 0.0
            )

            self._cell(row, 0, label)
            self._cell(row, 1, "")
            self._cell(row, 2, f"{modes:.3f}")
            self._cell(row, 3, f"{share * 100:.1f} %")

            row += 1

    def _fill_thermal(self) -> None:
        """A row per 100 K, which is what anyone quotes."""

        thermal = self.thermal

        if thermal is None:
            return

        columns = [
            "T (K)",
            "F (kJ/mol)",
            "S (J/K/mol)",
            "Cv (J/K/mol)",
            "Cv / 3NR",
        ]

        wanted = [
            temperature
            for temperature in thermal.temperatures
            if float(temperature) % 100.0 == 0.0
        ]

        self.thermal_table.clear()
        self.thermal_table.setColumnCount(len(columns))
        self.thermal_table.setHorizontalHeaderLabels(columns)
        self.thermal_table.setRowCount(len(wanted))

        limit = thermal.dulong_petit

        for row, temperature in enumerate(wanted):

            free, entropy, capacity = thermal.at(float(temperature))

            self._cell(row, 0, f"{float(temperature):.0f}",
                       table=self.thermal_table)
            self._cell(row, 1, f"{free:.4f}", table=self.thermal_table)
            self._cell(row, 2, f"{entropy:.4f}", table=self.thermal_table)
            self._cell(row, 3, f"{capacity:.4f}", table=self.thermal_table)
            self._cell(
                row,
                4,
                f"{capacity / limit:.3f}" if limit else "",
                table=self.thermal_table,
            )

    def _cell(
        self,
        row: int,
        column: int,
        text: str,
        bold: bool = False,
        table: QTableWidget | None = None,
    ) -> None:

        target = self.peaks if table is None else table

        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)

        target.setItem(row, column, item)

    # ------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------

    def _save(self) -> None:
        """Write the whole curve, not just the peaks, plus the thermals."""

        if self.result is None:
            return

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save density of states", "phonon_dos.csv", "CSV (*.csv)"
        )

        if not chosen:
            return

        result = self.result

        try:
            with open(chosen, "w", newline="", encoding="utf-8") as handle:

                writer = csv.writer(handle)

                writer.writerow(["# QVista phonon density of states"])
                writer.writerow(
                    [
                        f"# mesh {result.mesh[0]}x{result.mesh[1]}"
                        f"x{result.mesh[2]}  ({result.qpoints} q-points)"
                    ]
                )
                writer.writerow([f"# broadening {result.method}"])

                if result.sigma is not None:
                    writer.writerow([f"# sigma {result.sigma} THz"])

                writer.writerow(
                    [
                        f"# integral {result.integral:.6f} of "
                        f"{result.expected_integral:.0f}"
                    ]
                )

                if self.force_field:
                    writer.writerow([f"# forces from {self.force_field}"])

                for note in result.notes:
                    writer.writerow([f"# {note}"])

                labels = list(result.curves)

                writer.writerow(
                    [f"frequency ({result.unit})", "total", *labels]
                )

                for index, frequency in enumerate(result.frequencies):
                    writer.writerow(
                        [
                            f"{float(frequency):.6f}",
                            f"{float(result.total[index]):.6f}",
                            *(
                                f"{float(result.curves[label][index]):.6f}"
                                for label in labels
                            ),
                        ]
                    )

                if self.thermal is not None:

                    writer.writerow([])
                    writer.writerow(["# Thermal properties"])

                    for note in self.thermal.notes:
                        writer.writerow([f"# {note}"])

                    writer.writerow(
                        [
                            "T (K)",
                            "F (kJ/mol)",
                            "S (J/K/mol)",
                            "Cv (J/K/mol)",
                        ]
                    )

                    for index, temperature in enumerate(
                        self.thermal.temperatures
                    ):
                        writer.writerow(
                            [
                                f"{float(temperature):.1f}",
                                f"{float(self.thermal.free_energy[index]):.6f}",
                                f"{float(self.thermal.entropy[index]):.6f}",
                                f"{float(self.thermal.heat_capacity[index]):.6f}",
                            ]
                        )

        except OSError as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not write that file: {exc}"
            )
            return

        _status(self.status, f"Written to {Path(chosen).name}.")

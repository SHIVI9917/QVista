"""
Diffusion kinetics and ionic conductivity across temperatures.

One trajectory gives one diffusivity, which on its own says little. This
takes several -- one XDATCAR per temperature -- fits the Arrhenius
relation through them, and reports the activation energy and prefactor,
plus the conductivity those imply.

Adding runs is the whole workflow, so the table of them is the middle of
the window rather than a setting hidden behind a button.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
from qvista.services import arrhenius as ar
from qvista.services import conductivity as cond
from qvista.services import msd as msd_service
from qvista.visualization import transport_plots as tp

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

RUN_KEY = "kinetics/trajectory"

NOTES_LINES = 6


class Runner(QThread):
    """Reading and fitting several long trajectories takes a while."""

    progressed = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, entries, settings, parent=None) -> None:

        super().__init__(parent)

        self.entries = entries
        self.settings = settings

    def run(self) -> None:

        try:
            points = []

            for path, temperature, timestep in self.entries:

                self.progressed.emit(
                    f"{Path(path).name} at {temperature:.0f} K..."
                )

                trajectory = msd_service.read_xdatcar(
                    path, timestep=timestep, temperature=temperature
                )

                result = msd_service.analyse(
                    trajectory,
                    elements=[self.settings["element"]],
                )

                diffusion = result.fit_for(self.settings["element"])

                if diffusion is None:
                    raise msd_service.TrajectoryError(
                        f"{Path(path).name} has no "
                        f"{self.settings['element']} to follow. It has "
                        f"{', '.join(trajectory.elements())}."
                    )

                points.append(
                    ar.Point(
                        temperature=temperature,
                        diffusivity=diffusion.diffusivity,
                        label=Path(path).name,
                    )
                )

                self.settings.setdefault(
                    "volume", trajectory.volume
                )
                self.settings.setdefault(
                    "count", len(trajectory.indices_of(
                        self.settings["element"]
                    ))
                )

            self.completed.emit(points)

        except msd_service.TrajectoryError as exc:
            self.failed.emit(str(exc))

        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class KineticsDialog(QDialog):
    """Arrhenius fit and ionic conductivity from several temperatures."""

    def __init__(self, start: str = "", parent=None) -> None:

        super().__init__(parent)

        #: ``(path, temperature, timestep)``.
        self.entries: list[tuple[str, float, float]] = []

        self.points: list[ar.Point] = []
        self.result: ar.ArrheniusResult | None = None
        self.ionic: cond.IonicResult | None = None

        self.runner: Runner | None = None

        self._start = start

        self.setWindowTitle("Diffusion Kinetics")
        fit_to_screen(self, 980, 760)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_runs(layout)

        layout.addWidget(_hline())

        self._build_settings(layout)

        self.figures = FigureWindow(
            "QVista - Diffusion Kinetics",
            ("Arrhenius", "Conductivity"),
            self,
        )

        layout.addWidget(FigurePlaceholder(self.figures, self))

        self.results = QTableWidget()
        self.results.verticalHeader().setVisible(False)
        self.results.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        layout.addWidget(self.results, 1)

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
        self._say(
            f"Add one XDATCAR per temperature. At least "
            f"{ar.MIN_POINTS} are needed: two points define a line "
            f"through any two numbers."
        )

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.run_button = QPushButton("Fit")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run)

        footer.addWidget(self.run_button)

        self.save_button = QPushButton("Save CSV...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)

        footer.addWidget(self.save_button)
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

    # ------------------------------------------------------------

    def _say(self, text: str, style: str = STATUS_STYLE) -> None:

        self.status.setStyleSheet(style)
        self.status.setText(text)

    def _build_runs(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Runs")

        column = QVBoxLayout(box)

        row = QHBoxLayout()

        add = QPushButton("Add XDATCAR...")
        add.clicked.connect(self._add)

        row.addWidget(add)

        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove)

        row.addWidget(remove)

        row.addWidget(QLabel("at"))

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(1.0, 5000.0)
        self.temperature.setSingleStep(50.0)
        self.temperature.setValue(600.0)
        self.temperature.setSuffix(" K")

        row.addWidget(self.temperature)

        row.addWidget(QLabel("timestep"))

        self.timestep = QDoubleSpinBox()
        self.timestep.setDecimals(4)
        self.timestep.setRange(0.0001, 10.0)
        self.timestep.setValue(0.002)
        self.timestep.setSuffix(" ps")

        row.addWidget(self.timestep)

        row.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(160)

        row.addWidget(self.progress)

        column.addLayout(row)

        self.runs = QTableWidget()
        self.runs.setColumnCount(3)
        self.runs.setHorizontalHeaderLabels(
            ["Trajectory", "Temperature", "Timestep"]
        )
        self.runs.verticalHeader().setVisible(False)
        self.runs.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.runs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.runs.setMaximumHeight(160)

        column.addWidget(self.runs)

        layout.addWidget(box)

    def _build_settings(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Mobile ion")

        row = QHBoxLayout(box)

        row.addWidget(QLabel("Element"))

        self.element = QComboBox()
        self.element.setEditable(True)
        self.element.setMinimumWidth(90)

        row.addWidget(self.element)

        row.addWidget(QLabel("Charge"))

        self.charge = QDoubleSpinBox()
        self.charge.setRange(-8.0, 8.0)
        self.charge.setSingleStep(1.0)
        self.charge.setValue(1.0)
        self.charge.setToolTip(
            "Formal charge, for the Nernst-Einstein conversion."
        )

        row.addWidget(self.charge)

        hint = QLabel(
            "Ea comes from the slope and D0 from the intercept; the "
            "conductivity follows through Nernst-Einstein and carries "
            "the same Ea."
        )
        hint.setStyleSheet(STATUS_STYLE)
        hint.setWordWrap(True)

        row.addWidget(hint, 1)

        layout.addWidget(box)

    # ------------------------------------------------------------

    def _add(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self._start or settings.value(RUN_KEY, "", type=str)

        chosen, _ = QFileDialog.getOpenFileNames(
            self,
            "XDATCAR files",
            start,
            "Trajectories (XDATCAR*);;All Files (*)",
        )

        if not chosen:
            return

        settings.setValue(RUN_KEY, chosen[0])

        for path in chosen:

            self.entries.append(
                (path, self.temperature.value(), self.timestep.value())
            )

        self._fill_runs()

        # Offer the species actually in the first trajectory.
        if self.element.count() == 0:

            try:
                trajectory = msd_service.read_xdatcar(
                    chosen[0], timestep=self.timestep.value()
                )

                self.element.addItems(trajectory.elements())

            except msd_service.TrajectoryError:
                pass

        self.run_button.setEnabled(len(self.entries) >= ar.MIN_POINTS)

        self._say(
            f"{len(self.entries)} run(s). "
            + (
                "Ready to fit."
                if len(self.entries) >= ar.MIN_POINTS
                else f"At least {ar.MIN_POINTS} temperatures are needed."
            )
        )

    def _remove(self) -> None:

        rows = sorted(
            {index.row() for index in self.runs.selectedIndexes()},
            reverse=True,
        )

        for row in rows:
            if 0 <= row < len(self.entries):
                self.entries.pop(row)

        self._fill_runs()

        self.run_button.setEnabled(len(self.entries) >= ar.MIN_POINTS)

    def _fill_runs(self) -> None:

        self.runs.setRowCount(len(self.entries))

        for row, (path, temperature, timestep) in enumerate(self.entries):

            for column, text in enumerate(
                (Path(path).name, f"{temperature:.0f} K", f"{timestep:g} ps")
            ):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                if column == 0:
                    item.setToolTip(path)

                self.runs.setItem(row, column, item)

    # ------------------------------------------------------------

    def _run(self) -> None:

        element = self.element.currentText().strip()

        if not element:
            self._say("Name the mobile ion.", theme.WARNING_STYLE)
            return

        self.settings = {"element": element}

        self.progress.setVisible(True)
        self.run_button.setEnabled(False)

        self.runner = Runner(list(self.entries), self.settings, self)
        self.runner.progressed.connect(lambda text: self._say(text))
        self.runner.completed.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.finished.connect(
            lambda: (
                self.progress.setVisible(False),
                self.run_button.setEnabled(True),
            )
        )
        self.runner.start()

    def _done(self, points) -> None:

        self.points = points

        try:
            self.result = ar.fit(points)

            self.ionic = cond.ionic(
                points,
                element=self.settings["element"],
                charge=self.charge.value(),
                count=int(self.settings.get("count", 1)),
                volume=float(self.settings.get("volume", 0.0)),
            )

        except Exception as exc:  # noqa: BLE001
            self._failed(f"{type(exc).__name__}: {exc}")
            return

        tp.draw_arrhenius(
            self.figures.figure_for("Arrhenius"),
            self.result,
            title=f"{self.settings['element']} diffusion",
        )

        self.figures.refresh("Arrhenius")

        tp.draw_conductivity(
            self.figures.figure_for("Conductivity"),
            self.ionic,
            title=f"{self.settings['element']} conductivity",
        )

        self.figures.refresh("Conductivity")

        self.figures.show_figure("Arrhenius", reference=self)

        self._fill()

        self.save_button.setEnabled(True)

        notes = list(self.ionic.notes)

        warnings = [n for n in notes if n.startswith("WARNING")]
        others = [n for n in notes if not n.startswith("WARNING")]

        self.notes.setText("\n\n".join(warnings + others))
        self.notes_area.setVisible(bool(notes))

        self._say(
            f"Ea = {self.result.activation_energy:.4f} eV, "
            f"D0 = {self.result.prefactor:.3e} cm^2/s, "
            f"R^2 = {self.result.r_squared:.4f}.",
            theme.SUCCESS_STYLE,
        )

    def _failed(self, message: str) -> None:

        self.result = None
        self.save_button.setEnabled(False)
        self._say(message, theme.WARNING_STYLE)

    def _fill(self) -> None:

        rows: list[tuple[str, str]] = []

        if self.ionic is not None:
            rows.extend(self.ionic.as_rows())

        rows.append(("", ""))

        for point in self.points:
            rows.append(
                (
                    f"  {point.temperature:.0f} K",
                    f"D = {point.diffusivity:.4e} cm^2/s",
                )
            )

        self.results.clear()
        self.results.setColumnCount(2)
        self.results.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.results.setRowCount(len(rows))

        for index, (label, value) in enumerate(rows):

            for column, text in enumerate((label, value)):

                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                self.results.setItem(index, column, item)

    def _save(self) -> None:

        if self.result is None:
            return

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save", "diffusion_kinetics.csv", "CSV (*.csv)"
        )

        if not chosen:
            return

        try:
            Path(chosen).write_text(
                self.result.to_csv(), encoding="utf-8"
            )

        except OSError as exc:
            self._say(f"Could not write: {exc}", theme.WARNING_STYLE)
            return

        self._say(f"Saved to {chosen}.")

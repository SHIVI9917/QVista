"""
Trajectory, mean squared displacement and diffusion.

Takes either the run this session just did or an XDATCAR from a VASP
AIMD calculation, and turns it into a diffusion coefficient and, for a
mobile ion, a conductivity.

:mod:`qvista.services.msd` does the work and carries the caveats.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.dialogs.plot_window import FigurePlaceholder, FigureWindow
from qvista.services import msd
from qvista.visualization import dynamics_plots

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

#: Where the trajectory last opened is remembered between sessions.
RUN_KEY = "msd/run_root"

#: Lines of notes shown before the area scrolls. Whole lines only.
NOTES_LINES = 4

#: Common formal charges, offered for the conductivity.
CHARGES = ("1", "2", "3", "-1", "-2")


def _status(label: QLabel, text: str, style: str = STATUS_STYLE) -> None:

    label.setStyleSheet(style)
    label.setText(text)


class MsdRun(QThread):
    """One pass over the trajectory, off the GUI thread."""

    progressed = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, trajectory, settings: dict, parent=None) -> None:

        super().__init__(parent)

        self.trajectory = trajectory
        self.settings = settings

    def run(self) -> None:

        try:
            self.completed.emit(
                msd.analyse(
                    self.trajectory,
                    progress=self.progressed.emit,
                    **self.settings,
                )
            )

        except msd.TrajectoryError as exc:
            self.failed.emit(str(exc))

        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MsdDialog(QDialog):
    """The window behind Analysis > MD > MSD and Diffusion."""

    def __init__(self, trajectory=None, start: str = "", parent=None) -> None:

        super().__init__(parent)

        self.trajectory = trajectory
        self.result: msd.MsdResult | None = None
        self.worker: MsdRun | None = None

        self.setWindowTitle("Trajectory and Diffusion")
        fit_to_screen(self, 1140, 820)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_source(layout)

        layout.addWidget(_hline())

        # The figures live in their own window now, so the dialog is
        # tables plus the one line that says so.
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
        self.notes_area.setMaximumHeight(
            NOTES_LINES * self.notes.fontMetrics().lineSpacing()
            + 2 * self.notes_area.frameWidth()
        )

        layout.addWidget(self.notes_area)

        self.status = QLabel("")
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

        if trajectory is not None:
            self._describe_trajectory()
        elif start:
            self.folder.setText(start)

        self._describe()

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_source(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Trajectory")

        column = QVBoxLayout(box)

        # -- where it comes from --------------------------------
        row = QHBoxLayout()

        row.addWidget(QLabel("XDATCAR"))

        self.folder = QLineEdit()
        self.folder.setPlaceholderText(
            "the folder holding an XDATCAR from a VASP run with IBRION = 0"
        )

        row.addWidget(self.folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        load = QPushButton("Load")
        load.clicked.connect(self._load)

        row.addWidget(load)

        column.addLayout(row)

        # -- what the frames mean -------------------------------
        timing = QHBoxLayout()
        timing.setSpacing(10)

        timing.addWidget(QLabel("Time between frames"))

        self.timestep = QDoubleSpinBox()
        self.timestep.setRange(0.0001, 1000.0)
        self.timestep.setDecimals(4)
        self.timestep.setValue(0.002)
        self.timestep.setSingleStep(0.001)
        self.timestep.setSuffix(" ps")
        self.timestep.setToolTip(
            "POTIM x NBLOCK for a VASP run, since VASP writes every "
            "NBLOCK-th step rather than every step."
        )
        self.timestep.valueChanged.connect(self._retime)

        timing.addWidget(self.timestep)

        timing.addSpacing(14)

        timing.addWidget(QLabel("Temperature"))

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(1.0, 5000.0)
        self.temperature.setDecimals(0)
        self.temperature.setValue(300.0)
        self.temperature.setSingleStep(50.0)
        self.temperature.setSuffix(" K")

        timing.addWidget(self.temperature)

        timing.addSpacing(14)

        self.drift = QCheckBox("Remove cell drift")
        self.drift.setChecked(True)
        self.drift.setToolTip(
            "Subtract the motion of the cell as a whole, which would "
            "otherwise appear as diffusion of every atom equally."
        )

        timing.addWidget(self.drift)

        timing.addStretch(1)

        self.run_button = QPushButton("Analyse")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run)

        timing.addWidget(self.run_button)

        column.addLayout(timing)

        # -- fitting and conductivity ---------------------------
        fitting = QHBoxLayout()
        fitting.setSpacing(10)

        fitting.addWidget(QLabel("Fit from"))

        self.fit_start = QDoubleSpinBox()
        self.fit_start.setRange(0.0, 0.9)
        self.fit_start.setDecimals(2)
        self.fit_start.setSingleStep(0.05)
        self.fit_start.setValue(msd.DEFAULT_FIT_START)
        self.fit_start.valueChanged.connect(self._describe)

        fitting.addWidget(self.fit_start)

        fitting.addWidget(QLabel("to"))

        self.fit_end = QDoubleSpinBox()
        self.fit_end.setRange(0.1, 1.0)
        self.fit_end.setDecimals(2)
        self.fit_end.setSingleStep(0.05)
        self.fit_end.setValue(msd.DEFAULT_FIT_END)
        self.fit_end.valueChanged.connect(self._describe)

        fitting.addWidget(self.fit_end)

        fitting.addWidget(QLabel("of the curve"))

        fitting.addSpacing(14)

        fitting.addWidget(QLabel("Mobile ion"))

        self.mobile = QComboBox()
        self.mobile.setMinimumWidth(90)
        self.mobile.currentIndexChanged.connect(self._fill_diffusion)

        fitting.addWidget(self.mobile)

        fitting.addWidget(QLabel("charge"))

        self.charge = QComboBox()
        self.charge.setEditable(True)

        for value in CHARGES:
            self.charge.addItem(value)

        self.charge.currentTextChanged.connect(
            lambda _: self._fill_diffusion()
        )

        fitting.addWidget(self.charge)

        fitting.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(170)

        fitting.addWidget(self.progress)

        column.addLayout(fitting)

        self.choice_note = QLabel("")
        self.choice_note.setStyleSheet(STATUS_STYLE)
        self.choice_note.setWordWrap(True)

        column.addWidget(self.choice_note)

        layout.addWidget(box)

    def _build_figures(self) -> QWidget:
        """
        Put the figures in a window of their own.

        An MSD curve is read for the straightness of its middle section,
        which needs height. Sharing a splitter with the tables left it a
        strip too short to judge that in.
        """

        self.figures = FigureWindow(
            "QVista - Mean Squared Displacement",
            ("MSD", "MSD (log-log)", "Sampling"),
            self,
        )

        self.msd_panel = self.figures.panel_for("MSD")
        self.log_panel = self.figures.panel_for("MSD (log-log)")
        self.origins_panel = self.figures.panel_for("Sampling")

        return FigurePlaceholder(self.figures, self)

    def _build_tables(self) -> QWidget:

        self.tables = QTabWidget()

        self.diffusion = QTableWidget()
        self.diffusion.verticalHeader().setVisible(False)
        self.diffusion.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.curve = QTableWidget()
        self.curve.verticalHeader().setVisible(False)
        self.curve.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.tables.addTab(self.diffusion, "Diffusion")
        self.tables.addTab(self.curve, "Curve")

        return self.tables

    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------

    def _browse(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.folder.text() or settings.value(RUN_KEY, "", type=str)

        chosen = QFileDialog.getExistingDirectory(
            self, "Folder holding the XDATCAR", start
        )

        if chosen:
            self.folder.setText(chosen)
            settings.setValue(RUN_KEY, chosen)
            self._load()

    def _load(self) -> None:
        """Read an XDATCAR from the chosen folder."""

        text = self.folder.text().strip()

        if not text:
            _status(
                self.status,
                "Choose a folder holding an XDATCAR, or run some dynamics "
                "first and come back.",
                theme.WARNING_STYLE,
            )
            return

        path = Path(text)

        if path.is_dir():
            path = path / "XDATCAR"

        try:
            self.trajectory = msd.read_xdatcar(
                path,
                timestep=self.timestep.value(),
                temperature=self.temperature.value(),
            )

        except msd.TrajectoryError as exc:
            _status(self.status, str(exc), theme.ERROR_STYLE)
            return

        self._describe_trajectory()

    def _describe_trajectory(self) -> None:
        """Report what was loaded and offer its species."""

        trajectory = self.trajectory

        if trajectory is None:
            return

        if trajectory.temperature:
            self.temperature.setValue(trajectory.temperature)

        self.timestep.blockSignals(True)
        self.timestep.setValue(trajectory.timestep)
        self.timestep.blockSignals(False)

        self.mobile.blockSignals(True)
        self.mobile.clear()

        for element in trajectory.elements():
            self.mobile.addItem(element, element)

        self.mobile.blockSignals(False)

        self.run_button.setEnabled(True)

        _status(
            self.status,
            f"{trajectory.frames} frames, {trajectory.natoms} atoms, "
            f"{trajectory.duration:.3f} ps. Press Analyse.",
        )

        self._describe()

    def _retime(self, value: float) -> None:
        """A new frame spacing rescales the whole time axis."""

        if self.trajectory is not None:
            self.trajectory.timestep = float(value)

        self._describe()

    def _describe(self) -> None:
        """Say what the current settings will actually fit."""

        trajectory = self.trajectory

        if trajectory is None:
            self.choice_note.setText(
                "No trajectory yet. Load an XDATCAR, or run "
                "Analysis > MD > Run Dynamics and open this again."
            )
            return

        window = trajectory.duration * msd.DEFAULT_MAX_LAG_FRACTION

        self.choice_note.setText(
            f"{trajectory.duration:.3f} ps of trajectory. The curve is "
            f"taken to {window:.3f} ps -- a quarter of the run, beyond "
            f"which too few time origins are left to average -- and "
            f"fitted between {self.fit_start.value():.2f} and "
            f"{self.fit_end.value():.2f} of that, "
            f"{window * self.fit_start.value():.3f} to "
            f"{window * self.fit_end.value():.3f} ps."
        )

    # ------------------------------------------------------------
    # Running
    # ------------------------------------------------------------

    def _run(self) -> None:

        if self.trajectory is None:
            return

        if self.worker is not None and self.worker.isRunning():
            return

        if self.fit_start.value() >= self.fit_end.value():
            _status(
                self.status,
                "The fit window starts after it ends.",
                theme.WARNING_STYLE,
            )
            return

        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress.setVisible(True)

        self.trajectory.temperature = self.temperature.value()

        settings = {
            "drift": self.drift.isChecked(),
            "start": self.fit_start.value(),
            "end": self.fit_end.value(),
        }

        self.worker = MsdRun(self.trajectory, settings, parent=self)

        self.worker.progressed.connect(
            lambda text: _status(self.status, text)
        )
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.worker.start()

    def _on_completed(self, result: msd.MsdResult) -> None:

        self.result = result

        self._draw()
        self._fill_diffusion()
        self._fill_curve()

        self.notes.setText("\n".join(f"- {note}" for note in result.notes))
        self.notes_area.setVisible(bool(result.notes))

        self.save_button.setEnabled(True)

        fit = result.fit_for(self.mobile.currentData() or "all")

        _status(
            self.status,
            (
                f"Done. {self.mobile.currentText()} "
                f"D = {fit.diffusivity:.4e} cm2/s "
                f"(R^2 = {fit.r_squared:.4f})."
                if fit
                else "Done."
            ),
            theme.SUCCESS_STYLE,
        )

    def _on_failed(self, message: str) -> None:

        _status(self.status, message, theme.ERROR_STYLE)

        QMessageBox.warning(self, "Diffusion", message)

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

        trajectory = self.result.trajectory

        title = (
            f"{trajectory.natoms} atoms, {trajectory.duration:.2f} ps"
            f" at {trajectory.temperature:.0f} K"
            if trajectory
            else ""
        )

        dynamics_plots.draw_msd(
            self.msd_panel.figure, self.result, title=title
        )
        self.msd_panel.refresh()

        dynamics_plots.draw_msd(
            self.log_panel.figure, self.result, title=title, log_scale=True
        )
        self.log_panel.refresh()

        dynamics_plots.draw_msd_origins(
            self.origins_panel.figure,
            self.result,
            title="How many time origins each point averages",
        )
        self.origins_panel.refresh()

        # The figures are the point of the run, so raise them when one
        # finishes rather than making the user go and find the window.
        self.figures.show_figure("MSD", reference=self)

    # ------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------

    def _fill_diffusion(self) -> None:
        """One row per species, plus the conductivity for the mobile one."""

        result = self.result

        if result is None:
            return

        columns = [
            "Species",
            "Atoms",
            "D (cm2/s)",
            "D (A2/ps)",
            "R^2",
            "Fit window (ps)",
            "Conductivity",
        ]

        self.diffusion.clear()
        self.diffusion.setColumnCount(len(columns))
        self.diffusion.setHorizontalHeaderLabels(columns)
        self.diffusion.setRowCount(len(result.fits))

        mobile = self.mobile.currentData()

        try:
            charge = float(self.charge.currentText())

        except ValueError:
            charge = 0.0

        trajectory = result.trajectory

        for row, fit in enumerate(result.fits):

            self._cell(row, 0, "all atoms" if fit.label == "all" else fit.label)
            self._cell(row, 1, str(fit.count))
            self._cell(row, 2, f"{fit.diffusivity:.4e}")
            self._cell(row, 3, f"{fit.diffusivity_a2_ps:.5f}")
            self._cell(
                row,
                4,
                f"{fit.r_squared:.4f}",
                warn=not fit.trustworthy,
            )
            self._cell(row, 5, f"{fit.fit_from:.3f} - {fit.fit_to:.3f}")

            # Only for the ion actually named as mobile: a conductivity
            # quoted for the framework is meaningless, and quoting one
            # for every row invites reading the wrong number.
            text = ""

            if (
                fit.label == mobile
                and charge
                and trajectory is not None
                and trajectory.volume > 0
                and fit.diffusivity > 0
            ):
                try:
                    value = msd.conductivity(
                        fit.diffusivity,
                        fit.count,
                        charge,
                        trajectory.volume,
                        trajectory.temperature,
                    )
                    text = f"{value:.4g} mS/cm"

                except msd.TrajectoryError:
                    text = ""

            self._cell(row, 6, text)

    def _fill_curve(self) -> None:
        """The curve itself, thinned to something a table can hold."""

        result = self.result

        if result is None:
            return

        columns = ["Lag (ps)", "MSD (A2)", "Origins"]

        lags = np.asarray(result.lags, dtype=float)

        step = max(1, lags.size // 60)
        rows = range(0, lags.size, step)

        self.curve.clear()
        self.curve.setColumnCount(len(columns))
        self.curve.setHorizontalHeaderLabels(columns)
        self.curve.setRowCount(len(list(rows)))

        origins = result.origins

        for row, index in enumerate(range(0, lags.size, step)):

            self._cell(row, 0, f"{lags[index]:.4f}", table=self.curve)
            self._cell(
                row, 1, f"{float(result.total[index]):.5f}", table=self.curve
            )
            self._cell(
                row,
                2,
                "" if origins is None else f"{float(origins[index]):.0f}",
                table=self.curve,
            )

    def _cell(
        self,
        row: int,
        column: int,
        text: str,
        warn: bool = False,
        table: QTableWidget | None = None,
    ) -> None:

        target = self.diffusion if table is None else table

        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        if warn:
            item.setForeground(Qt.red)

        target.setItem(row, column, item)

    # ------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------

    def _save(self) -> None:

        if self.result is None:
            return

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save diffusion analysis", "diffusion.csv", "CSV (*.csv)"
        )

        if not chosen:
            return

        result = self.result
        trajectory = result.trajectory

        try:
            with open(chosen, "w", newline="", encoding="utf-8") as handle:

                writer = csv.writer(handle)

                writer.writerow(["# QVista diffusion analysis"])

                if trajectory is not None:
                    writer.writerow([f"# source {trajectory.source}"])
                    writer.writerow(
                        [
                            f"# {trajectory.frames} frames, "
                            f"{trajectory.natoms} atoms, "
                            f"{trajectory.duration:.4f} ps at "
                            f"{trajectory.temperature:.0f} K"
                        ]
                    )

                writer.writerow(
                    ["# D from MSD = 2 d D t, d = 3, fitted by least squares"]
                )
                writer.writerow(
                    [
                        "# the standard error below is a lower bound: MSD "
                        "points share trajectory and are correlated"
                    ]
                )

                for note in result.notes:
                    writer.writerow([f"# {note}"])

                writer.writerow([])
                writer.writerow(
                    [
                        "species",
                        "atoms",
                        "D (cm^2/s)",
                        "D (A^2/ps)",
                        "slope (A^2/ps)",
                        "stderr of slope",
                        "R^2",
                        "fit from (ps)",
                        "fit to (ps)",
                    ]
                )

                for fit in result.fits:
                    writer.writerow(
                        [
                            fit.label,
                            fit.count,
                            f"{fit.diffusivity:.6e}",
                            f"{fit.diffusivity_a2_ps:.6f}",
                            f"{fit.slope:.6f}",
                            f"{fit.stderr:.6f}",
                            f"{fit.r_squared:.6f}",
                            f"{fit.fit_from:.4f}",
                            f"{fit.fit_to:.4f}",
                        ]
                    )

                writer.writerow([])
                writer.writerow(["# the curve"])

                labels = list(result.curves)

                writer.writerow(
                    ["lag (ps)", "MSD all (A^2)", *labels, "origins"]
                )

                for index, lag in enumerate(result.lags):
                    writer.writerow(
                        [
                            f"{float(lag):.6f}",
                            f"{float(result.total[index]):.6f}",
                            *(
                                f"{float(result.curves[label][index]):.6f}"
                                for label in labels
                            ),
                            ""
                            if result.origins is None
                            else f"{float(result.origins[index]):.0f}",
                        ]
                    )

        except OSError as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not write that file: {exc}"
            )
            return

        _status(self.status, f"Written to {Path(chosen).name}.")

"""
RDF, Van Hove and residence time from one trajectory.

Three analyses in one window because they all read the same XDATCAR and
differ only in what they ask of it. Loading a long trajectory is the
slow part, so it happens once and all three run off it.
"""

from __future__ import annotations

from pathlib import Path

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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.dialogs.plot_window import FigurePlaceholder, FigureWindow
from qvista.gui.sizing import fit_to_screen
from qvista.services import correlations as co
from qvista.services import msd as msd_service
from qvista.visualization import transport_plots as tp

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

RUN_KEY = "correlations/trajectory"

NOTES_LINES = 5

RDF = "rdf"
VAN_HOVE = "vanhove"
RESIDENCE = "residence"

MODES = {
    RDF: "RDF",
    VAN_HOVE: "Van Hove",
    RESIDENCE: "Residence time",
}


class Chip(QPushButton):
    """A small checkable box, matching the other analysis dialogs."""

    def __init__(self, text: str, parent=None) -> None:

        super().__init__(text, parent)

        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)


class Worker(QThread):
    """Correlations on a long trajectory take seconds to minutes."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, job, parent=None) -> None:

        super().__init__(parent)

        self.job = job

    def run(self) -> None:

        try:
            self.completed.emit(self.job())

        except (co.TrajectoryError, msd_service.TrajectoryError) as exc:
            self.failed.emit(str(exc))

        except Exception as exc:  # noqa: BLE001 - shown, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CorrelationsDialog(QDialog):
    """Pair and time correlations from an XDATCAR."""

    def __init__(
        self,
        start: str = "",
        mode: str = RDF,
        trajectory: "msd_service.Trajectory | None" = None,
        parent=None,
    ):

        super().__init__(parent)

        self.trajectory: msd_service.Trajectory | None = None
        self.result = None
        self.worker: Worker | None = None

        self.mode = mode if mode in MODES else RDF

        self.setWindowTitle("Trajectory Correlations")
        fit_to_screen(self, 1000, 720)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_source(layout)

        layout.addWidget(_hline())

        self._build_controls(layout)

        self.figures = FigureWindow(
            "QVista - Trajectory Correlations",
            ("RDF", "Van Hove", "Van Hove slices", "Residence time"),
            self,
        )

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
        self._say("Choose an XDATCAR from a molecular-dynamics run.")

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.save_button = QPushButton("Save CSV...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)

        footer.addWidget(self.save_button)
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        self._pick_mode(self.mode)

        if start:
            self.source.setText(start)

        if trajectory is not None:
            self._adopt(trajectory)

    # ------------------------------------------------------------

    def _say(self, text: str, style: str = STATUS_STYLE) -> None:

        self.status.setStyleSheet(style)
        self.status.setText(text)

    def _build_source(self, layout: QVBoxLayout) -> None:

        row = QHBoxLayout()

        row.addWidget(QLabel("XDATCAR"))

        self.source = QLineEdit()
        self.source.setPlaceholderText("the trajectory to analyse")
        self.source.editingFinished.connect(self._load)

        row.addWidget(self.source, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        row.addWidget(QLabel("Timestep"))

        self.timestep = QDoubleSpinBox()
        self.timestep.setDecimals(4)
        self.timestep.setRange(0.0001, 10.0)
        self.timestep.setValue(0.002)
        self.timestep.setSuffix(" ps")
        self.timestep.setToolTip(
            "Between the frames in the file, which is POTIM * NBLOCK "
            "and not POTIM."
        )

        row.addWidget(self.timestep)

        row.addWidget(QLabel("T"))

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 5000.0)
        self.temperature.setSingleStep(50.0)
        self.temperature.setValue(300.0)
        self.temperature.setSuffix(" K")

        row.addWidget(self.temperature)

        load = QPushButton("Load")
        load.clicked.connect(self._load)

        row.addWidget(load)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(140)

        row.addWidget(self.progress)

        layout.addLayout(row)

    def _build_controls(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Analysis")

        column = QVBoxLayout(box)

        row = QHBoxLayout()
        row.setSpacing(6)

        self.chips: dict[str, Chip] = {}

        for key, label in MODES.items():

            chip = Chip(label)
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_mode(name)
            )

            self.chips[key] = chip
            row.addWidget(chip)

        row.addStretch(1)

        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run)

        row.addWidget(self.run_button)

        column.addLayout(row)

        # -- shared -------------------------------------------------
        shared = QHBoxLayout()
        shared.setSpacing(10)

        shared.addWidget(QLabel("Skip"))

        self.skip = QDoubleSpinBox()
        self.skip.setRange(0.0, 0.9)
        self.skip.setSingleStep(0.05)
        self.skip.setDecimals(2)
        self.skip.setValue(co.DEFAULT_SKIP)
        self.skip.setToolTip(
            "Fraction of the run discarded at the start as "
            "equilibration. Those frames are not the ensemble."
        )

        shared.addWidget(self.skip)

        shared.addWidget(QLabel("Bins"))

        self.bins = QSpinBox()
        self.bins.setRange(20, 1000)
        self.bins.setValue(co.DEFAULT_BINS)

        shared.addWidget(self.bins)

        shared.addWidget(QLabel("r max"))

        self.r_max = QDoubleSpinBox()
        self.r_max.setRange(0.0, 50.0)
        self.r_max.setSingleStep(0.5)
        self.r_max.setValue(0.0)
        self.r_max.setSpecialValueText("half the cell")
        self.r_max.setSuffix(" A")

        shared.addWidget(self.r_max)

        # -- per mode -----------------------------------------------
        self.element_label = QLabel("Element")
        self.element = QComboBox()

        shared.addWidget(self.element_label)
        shared.addWidget(self.element)

        self.lags_label = QLabel("Lags")
        self.lags = QSpinBox()
        self.lags.setRange(2, 100)
        self.lags.setValue(20)

        shared.addWidget(self.lags_label)
        shared.addWidget(self.lags)

        self.distinct = QCheckBox("Distinct part")
        self.distinct.setChecked(True)
        self.distinct.setToolTip(
            "The expensive half: every pair at every lag. Turn it off "
            "for a first look at a long run."
        )

        shared.addWidget(self.distinct)

        self.cutoff_label = QLabel("Cutoff")
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.1, 10.0)
        self.cutoff.setSingleStep(0.25)
        self.cutoff.setValue(co.DEFAULT_RESIDENCE_CUTOFF)
        self.cutoff.setSuffix(" A")
        self.cutoff.setToolTip(
            "An ion counts as having stayed while it is within this "
            "of where it was. Between the vibrational amplitude and "
            "the hop distance."
        )

        shared.addWidget(self.cutoff_label)
        shared.addWidget(self.cutoff)

        shared.addStretch(1)

        column.addLayout(shared)

        self.hint = QLabel("")
        self.hint.setStyleSheet(STATUS_STYLE)
        self.hint.setWordWrap(True)

        column.addWidget(self.hint)

        layout.addWidget(box)

    # ------------------------------------------------------------

    def _pick_mode(self, name: str) -> None:

        self.mode = name

        for key, chip in self.chips.items():
            chip.setChecked(key == name)

        needs_element = name in (VAN_HOVE, RESIDENCE)

        for widget in (self.element_label, self.element):
            widget.setVisible(needs_element)

        for widget in (self.lags_label, self.lags, self.distinct):
            widget.setVisible(name == VAN_HOVE)

        for widget in (self.cutoff_label, self.cutoff):
            widget.setVisible(name == RESIDENCE)

        for widget in (self.bins, self.r_max):
            widget.setEnabled(name != RESIDENCE)

        self.hint.setText(
            {
                RDF: "g(r): how likely one species is to be found a "
                     "distance r from another, against a uniform gas "
                     "of the same density. The first peak is the bond "
                     "length; the area under it is the coordination "
                     "number.",
                VAN_HOVE: "G(r,t): the RDF with time in it. The self "
                          "part follows one ion away from where it "
                          "started, the distinct part follows every "
                          "other one -- so Gd at t = 0 is the RDF.",
                RESIDENCE: "How long an ion stays put before it hops, "
                           "from the fraction still within the cutoff "
                           "of where it was.",
            }[name]
        )

    # ------------------------------------------------------------

    def _browse(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.source.text() or settings.value(
            RUN_KEY, "", type=str
        )

        chosen, _ = QFileDialog.getOpenFileName(
            self, "XDATCAR", start, "Trajectories (XDATCAR*);;All Files (*)"
        )

        if not chosen:
            return

        self.source.setText(chosen)

        settings.setValue(RUN_KEY, chosen)

        self._load()

    def _load(self) -> None:

        path = self.source.text().strip()

        if not path:
            return

        try:
            self.trajectory = msd_service.read_xdatcar(
                path,
                timestep=self.timestep.value(),
                temperature=self.temperature.value(),
            )

        except msd_service.TrajectoryError as exc:
            self.trajectory = None
            self.run_button.setEnabled(False)
            self._say(str(exc), theme.WARNING_STYLE)
            return

        self._describe()

    def _adopt(self, trajectory) -> None:
        """
        Take a trajectory recorded in this session rather than read.

        A CHGNet run hands its frames over directly, so there is no file
        to browse for. The timestep and temperature come with it and are
        shown rather than asked for -- they are properties of the run
        that produced it, and typing a different number here would only
        mislabel the axes.
        """

        self.trajectory = trajectory

        self.source.clear()
        self.source.setPlaceholderText(
            "from this session's CHGNet run"
        )

        for box, value in (
            (self.timestep, trajectory.timestep),
            (self.temperature, trajectory.temperature),
        ):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)

        self._describe()

    def _describe(self) -> None:
        """Fill the element list and say what was loaded."""

        if self.trajectory is None:
            return

        self.element.clear()
        self.element.addItems(self.trajectory.elements())

        self.run_button.setEnabled(True)

        self._say(
            f"{self.trajectory.frames} frames, "
            f"{self.trajectory.natoms} atoms "
            f"({', '.join(self.trajectory.elements())}), "
            f"{self.trajectory.duration:.3f} ps."
        )

    # ------------------------------------------------------------

    def _run(self) -> None:

        if self.trajectory is None:
            return

        trajectory = self.trajectory

        if self.mode == RDF:

            job = lambda: co.rdf(
                trajectory,
                bins=self.bins.value(),
                r_max=self.r_max.value(),
                skip=self.skip.value(),
            )

        elif self.mode == VAN_HOVE:

            job = lambda: co.van_hove(
                trajectory,
                element=self.element.currentText(),
                bins=self.bins.value(),
                r_max=self.r_max.value(),
                lags=self.lags.value(),
                skip=self.skip.value(),
                distinct=self.distinct.isChecked(),
            )

        else:

            job = lambda: co.residence_time(
                trajectory,
                element=self.element.currentText(),
                cutoff=self.cutoff.value(),
                skip=self.skip.value(),
            )

        self.progress.setVisible(True)
        self.run_button.setEnabled(False)
        self._say("Running...")

        self.worker = Worker(job, self)
        self.worker.completed.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(
            lambda: (
                self.progress.setVisible(False),
                self.run_button.setEnabled(True),
            )
        )
        self.worker.start()

    def _done(self, result) -> None:

        self.result = result

        try:
            self._draw()
            self._fill()

        except Exception as exc:  # noqa: BLE001
            self._say(f"{type(exc).__name__}: {exc}", theme.ERROR_STYLE)
            return

        self.save_button.setEnabled(True)

        notes = list(getattr(result, "notes", []) or [])

        warnings = [n for n in notes if n.startswith("WARNING")]
        others = [n for n in notes if not n.startswith("WARNING")]

        self.notes.setText("\n\n".join(warnings + others))
        self.notes_area.setVisible(bool(notes))

        self._say(self._summary(), theme.SUCCESS_STYLE)

    def _failed(self, message: str) -> None:

        self.result = None
        self.save_button.setEnabled(False)
        self._say(message, theme.WARNING_STYLE)

    # ------------------------------------------------------------

    def _draw(self) -> None:

        if self.mode == RDF:

            tp.draw_rdf(
                self.figures.figure_for("RDF"),
                self.result,
                title=f"g(r) at {self.temperature.value():.0f} K",
            )

            self.figures.refresh("RDF")
            self.figures.show_figure("RDF", reference=self)

            return

        if self.mode == VAN_HOVE:

            tp.draw_van_hove(
                self.figures.figure_for("Van Hove"),
                self.result,
                title=(
                    f"Van Hove self part, {self.result.element} at "
                    f"{self.temperature.value():.0f} K"
                ),
            )

            self.figures.refresh("Van Hove")

            tp.draw_van_hove_slices(
                self.figures.figure_for("Van Hove slices"),
                self.result,
                title=f"Self part, {self.result.element}",
            )

            self.figures.refresh("Van Hove slices")

            self.figures.show_figure("Van Hove", reference=self)

            return

        tp.draw_residence(
            self.figures.figure_for("Residence time"),
            self.result,
            title=(
                f"{self.result.element} survival at "
                f"{self.temperature.value():.0f} K"
            ),
        )

        self.figures.refresh("Residence time")
        self.figures.show_figure("Residence time", reference=self)

    def _rows(self) -> list[tuple[str, str]]:

        result = self.result

        if self.mode == RDF:

            rows: list[tuple[str, str]] = [
                ("Frames averaged", str(result.frames)),
            ]

            for name in result.curves:

                rows.append(("", ""))
                rows.append((f"{name} first peak",
                             f"{result.first_peak.get(name, float('nan')):.3f} A"))

                minimum = result.first_minimum.get(name)

                if minimum is not None:
                    rows.append(
                        (f"{name} first minimum", f"{minimum:.3f} A")
                    )
                    rows.append(
                        (
                            f"{name} coordination",
                            f"{result.shell.get(name, float('nan')):.2f}",
                        )
                    )

            return rows

        if self.mode == VAN_HOVE:

            return [
                ("Element", result.element),
                ("Frames used", str(result.frames)),
                ("Lags", str(len(result.t))),
                ("Longest lag", f"{result.t[-1]:.4f} ps"),
                ("r points", str(len(result.r))),
                ("Distinct part", "yes" if result.has_distinct else "no"),
            ]

        return [
            ("Element", result.element),
            ("Cutoff", f"{result.cutoff:g} A"),
            (
                "Residence time",
                f"{result.tau:.4f} ps" if result.tau == result.tau
                else "not measurable",
            ),
            ("Hop rate", f"{result.rate:.3f} /ps"),
            (
                "Decayed past 1/e",
                "yes" if result.decayed else "NO - see the warning",
            ),
        ]

    def _fill(self) -> None:

        rows = self._rows()

        self.table.clear()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.table.setRowCount(len(rows))

        for index, (label, value) in enumerate(rows):

            for column, text in enumerate((label, value)):

                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                self.table.setItem(index, column, item)

    def _summary(self) -> str:

        result = self.result

        if self.mode == RDF:

            return (
                f"{len(result.curves)} pair(s) over "
                f"{result.frames} frames."
            )

        if self.mode == VAN_HOVE:

            return (
                f"{result.element}: {len(result.t)} lags to "
                f"{result.t[-1]:.3f} ps."
            )

        return (
            f"{result.element}: tau = {result.tau:.4f} ps "
            f"({result.rate:.2f} hops/ps)."
        )

    # ------------------------------------------------------------

    def _save(self) -> None:

        if self.result is None:
            return

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save", f"{self.mode}.csv", "CSV (*.csv)"
        )

        if not chosen:
            return

        text = getattr(self.result, "to_csv", None)

        if text is None:

            # Van Hove and residence have no CSV of their own; write
            # the table, which is what is actually read off them.
            lines = ["quantity,value"]

            lines.extend(
                f"{label},{value}" for label, value in self._rows()
            )

            body = "\n".join(lines) + "\n"

        else:
            body = text()

        try:
            Path(chosen).write_text(body, encoding="utf-8")

        except OSError as exc:
            self._say(f"Could not write: {exc}", theme.WARNING_STYLE)
            return

        self._say(f"Saved to {chosen}.")

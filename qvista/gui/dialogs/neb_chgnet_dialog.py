"""
A machine-learned NEB, start to finish, in one window.

The VASP route is split in two because the calculation happens somewhere
else: write the folders, take them to a machine with a licence, bring
them back. A force field has no such gap -- the path relaxes here, in
seconds -- so splitting it would ask the user to walk through a door
that leads back into the same room.

So: the two endpoints, the settings, Run, and then the same figure and
the same tables the VASP route produces, in the panel below. Everything
after the run is :class:`~qvista.gui.dialogs.neb_vasp_dialog.NebResultsView`,
shared with the other route, because a relaxed path is a relaxed path
and the questions asked of one are the questions asked of the other.
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
    QVBoxLayout,
    QWidget,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.dialogs.neb_vasp_dialog import (
    ANALYSIS_ROW_HEIGHT,
    MEP_STAGE,
    NOTES_LINES,
    NebResultsView,
    _status,
)
from qvista.gui.dialogs.plot_window import FigurePlaceholder, FigureWindow
from qvista.gui.sizing import fit_to_screen
from qvista.services import neb_vasp

START_KEY = "neb_chgnet/start"
END_KEY = "neb_chgnet/end"

#: Defaults, matching the analysis they are handed to.
DEFAULT_IMAGES = 5
DEFAULT_FMAX = 0.05
DEFAULT_STEPS = 200


class Runner(QThread):
    """Relax the band off the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)
    progressed = Signal(str)

    def __init__(self, job, parent=None) -> None:

        super().__init__(parent)

        self._job = job

    def run(self) -> None:

        try:
            self.completed.emit(self._job(self.report))

        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def report(self, message: str, percent: int | None = None) -> None:
        """
        Publish one progress message to the console and the window.

        The console print happens here, on the worker thread, rather
        than in the slot on the other end of the signal: a relaxation
        that stalls should still have left a trail, and that trail
        should not depend on the GUI thread being free to deliver a
        queued signal. Relaxing a band with a force field takes minutes,
        and a window that says nothing for minutes is indistinguishable
        from one that has hung.
        """

        # The window gets the message whatever happens to the console.
        # Windows consoles run a code page that need not be able to
        # spell every character in a progress line -- the Angstrom sign,
        # most often -- and an encoding error raised in here would be
        # caught upstream and reported as a failed relaxation, which is
        # a lie about a run that is going fine.
        try:
            print(f"[NEB] {message}", flush=True)

        except Exception:  # noqa: BLE001 - a log is not worth a run
            try:
                print(
                    f"[NEB] {message}".encode("ascii", "replace").decode(
                        "ascii"
                    ),
                    flush=True,
                )

            except Exception:  # noqa: BLE001 - then there is no console
                pass

        self.progressed.emit(message)


class NebChgnetDialog(NebResultsView, QDialog):
    """Endpoints, settings, run, and the result -- in one window."""

    def __init__(self, start: str = "", parent=None) -> None:

        super().__init__(parent)

        self.result = None
        self.runner: Runner | None = None

        self.setWindowTitle("NEB - CHGNet")
        fit_to_screen(self, 1000, 820)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_endpoints(layout)
        self._build_settings(layout)

        layout.addWidget(_hline())

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        layout.addWidget(self.progress)

        self._build_results(layout)

        if start:
            self.start.setText(start)

    # ------------------------------------------------------------
    # Asking
    # ------------------------------------------------------------

    def _build_endpoints(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Endpoints")

        column = QVBoxLayout(box)

        self.start = QLineEdit()
        self.start.setPlaceholderText("POSCAR of the starting site")

        self.end = QLineEdit()
        self.end.setPlaceholderText("POSCAR of the ending site")

        for label, field, key in (
            ("Initial", self.start, START_KEY),
            ("Final", self.end, END_KEY),
        ):
            row = QHBoxLayout()

            row.addWidget(QLabel(label))
            row.addWidget(field, 1)

            browse = QPushButton("Browse...")
            browse.clicked.connect(
                lambda _=False, f=field, k=key, t=label: self._browse(
                    f, k, t
                )
            )

            row.addWidget(browse)

            column.addLayout(row)

        hint = QLabel(
            "Both must already be relaxed, with the same atoms in the "
            "same order. NEB holds them fixed."
        )
        hint.setStyleSheet(theme.CAPTION_STYLE)
        hint.setWordWrap(True)

        column.addWidget(hint)

        layout.addWidget(box)

    def _build_settings(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Path")

        column = QVBoxLayout(box)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.climb = QCheckBox("CI-NEB")
        self.climb.setChecked(True)
        self.climb.setToolTip(
            "Climbing image: puts the highest image on the saddle "
            "point, so the barrier is the barrier rather than a lower "
            "bound."
        )

        row.addWidget(self.climb)

        row.addWidget(QLabel("Fmax"))

        self.fmax = QDoubleSpinBox()
        self.fmax.setDecimals(3)
        self.fmax.setRange(0.001, 1.0)
        self.fmax.setSingleStep(0.01)
        self.fmax.setValue(DEFAULT_FMAX)
        self.fmax.setSuffix(" eV/A")
        self.fmax.setToolTip(
            "Force convergence. The band stops when no image carries a "
            "force larger than this."
        )

        row.addWidget(self.fmax)

        row.addWidget(QLabel("Images"))

        self.images = QSpinBox()
        self.images.setRange(1, 30)
        self.images.setValue(DEFAULT_IMAGES)
        self.images.setToolTip(
            "Intermediate images, endpoints excluded."
        )

        row.addWidget(self.images)

        row.addWidget(QLabel("Max steps"))

        self.steps = QSpinBox()
        self.steps.setRange(1, 2000)
        self.steps.setValue(DEFAULT_STEPS)

        row.addWidget(self.steps)

        row.addStretch(1)

        self.run_button = QPushButton("Run NEB")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.clicked.connect(self._run)

        row.addWidget(self.run_button)

        column.addLayout(row)

        self._build_outputs(column)

        layout.addWidget(box)

    def _build_outputs(self, column: QVBoxLayout) -> None:
        """
        Where the run goes when it finishes.

        A relaxation costs minutes and, until now, left nothing behind:
        close the window and it had to be run again. Written out, the
        folder is the same shape the reader expects, so a saved run
        reopens through Post-Processing instead.

        Checked by default. The cost of writing a few files is nothing
        against the cost of discovering afterwards that the run was
        never kept.
        """

        column.addWidget(_hline())

        where = QHBoxLayout()
        where.setSpacing(6)

        self.save_outputs = QCheckBox("Save outputs to")
        self.save_outputs.setChecked(True)
        self.save_outputs.setToolTip(
            "Write the relaxed images and the profile when the run "
            "finishes, so it can be reopened without relaxing again."
        )

        where.addWidget(self.save_outputs)

        self.output_folder = QLineEdit()
        self.output_folder.setPlaceholderText(
            "chosen from the initial structure if left empty"
        )

        where.addWidget(self.output_folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_output)

        where.addWidget(browse)

        column.addLayout(where)

        what = QHBoxLayout()
        what.setSpacing(12)

        #: Keyed by the argument name ``save_run`` takes for each.
        self.save_what: dict[str, QCheckBox] = {}

        for key, label in neb_vasp.SAVEABLE:

            box = QCheckBox(label)
            box.setChecked(True)

            self.save_what[key] = box

            what.addWidget(box)

        what.addStretch(1)

        column.addLayout(what)

        # The image folders and the profile are always written -- they
        # are what makes the folder readable, and choosing not to write
        # them would only produce a folder nothing can reopen.
        self.save_outputs.toggled.connect(self._outputs_toggled)

    def _outputs_toggled(self, on: bool) -> None:

        self.output_folder.setEnabled(on)

        for box in self.save_what.values():
            box.setEnabled(on)

    def _browse_output(self) -> None:

        chosen = QFileDialog.getExistingDirectory(
            self,
            "Save the NEB run in",
            self.output_folder.text() or self._default_output(),
        )

        if chosen:
            self.output_folder.setText(chosen)

    def _default_output(self) -> str:
        """
        Beside the structure the path starts from.

        Somewhere predictable and next to the input, rather than
        wherever the file dialog happened to be last.
        """

        start = self.start.text().strip()

        if not start:
            return ""

        return str(Path(start).parent / "neb_chgnet")

    # ------------------------------------------------------------
    # Showing
    # ------------------------------------------------------------

    def _build_results(self, layout: QVBoxLayout) -> None:
        """
        The same output the VASP route shows.

        Built here rather than inherited because a mixin that lays out
        widgets is a mixin that dictates the window around them; what is
        shared is the behaviour, and the two windows put it in different
        places.
        """

        self.figures = FigureWindow(
            "QVista - NEB Barrier", ("Energy profile",), self
        )

        self.panel = self.figures.panel_for()

        layout.addWidget(FigurePlaceholder(self.figures, self))

        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        layout.addWidget(self.table)

        self.analysis_label = QLabel("Migrating atom")
        self.analysis_label.setStyleSheet(
            f"color: {theme.IRIS_GLOW}; font-weight: 700;"
        )
        self.analysis_label.setVisible(False)

        layout.addWidget(self.analysis_label)

        self.analysis_row = QWidget()

        controls = QHBoxLayout(self.analysis_row)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        controls.addWidget(QLabel("Bond cutoff"))

        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(neb_vasp.MIN_CUTOFF, neb_vasp.MAX_CUTOFF)
        self.cutoff.setSingleStep(0.1)
        self.cutoff.setDecimals(2)
        self.cutoff.setValue(neb_vasp.NEIGHBOUR_CUTOFF)
        self.cutoff.setSuffix(" A")
        self.cutoff.valueChanged.connect(self._recount)

        controls.addWidget(self.cutoff)

        controls.addSpacing(12)

        controls.addWidget(QLabel("Show"))

        self.stage = QComboBox()

        controls.addWidget(self.stage)

        self.show_button = QPushButton("Show structure")
        self.show_button.clicked.connect(self._show_structure)

        controls.addWidget(self.show_button)

        controls.addStretch(1)

        self.analysis_row.setVisible(False)

        layout.addWidget(self.analysis_row)

        self.analysis = QTableWidget()
        self.analysis.verticalHeader().setVisible(False)
        self.analysis.setAlternatingRowColors(True)
        self.analysis.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.analysis.setVisible(False)
        self.analysis.verticalHeader().setDefaultSectionSize(
            ANALYSIS_ROW_HEIGHT
        )
        self.analysis.setShowGrid(False)
        self.analysis.setStyleSheet(
            "QTableWidget::item { padding: 6px 10px; }"
        )

        layout.addWidget(self.analysis, 1)

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
        _status(self.status, "Choose the two relaxed endpoints.")

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

    # ------------------------------------------------------------
    # Running
    # ------------------------------------------------------------

    def _browse(self, field: QLineEdit, key: str, title: str) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        chosen, _ = QFileDialog.getOpenFileName(
            self,
            f"{title} structure",
            field.text() or settings.value(key, "", type=str),
            "VASP structure (POSCAR* CONTCAR* *.vasp);;All files (*)",
        )

        if not chosen:
            return

        field.setText(chosen)

        settings.setValue(key, chosen)

    def _run(self) -> None:

        if self.runner is not None and self.runner.isRunning():
            return

        start = self.start.text().strip()
        end = self.end.text().strip()

        if not start or not end:
            _status(
                self.status,
                "Choose both endpoints first.",
                theme.WARNING_STYLE,
            )
            return

        images = self.images.value()
        fmax = self.fmax.value()
        steps = self.steps.value()
        climb = self.climb.isChecked()

        def job(report):

            from qvista.analyses.neb import NEBAnalysis
            from qvista.models.project import Project
            from qvista.models.project_info import ProjectInfo
            from qvista.services.structure_loader import StructureLoader

            # A project built around the starting file alone. NEB needs
            # a structure and a name and nothing else from one, and
            # requiring an open project would stop this working on two
            # POSCARs that were never opened as a project.
            source = Path(start)

            project = Project(
                info=ProjectInfo(
                    name=source.stem,
                    directory=source.parent,
                    source_file=source,
                ),
                structure=StructureLoader.load(source),
            )

            analysis = NEBAnalysis(
                project,
                Path(end),
                n_images=images,
                fmax=fmax,
                steps=steps,
                climb=climb,
            )

            return analysis.run(progress=report)

        self.progress.setVisible(True)
        self.run_button.setEnabled(False)

        _status(
            self.status,
            f"Relaxing {images} images to {fmax:g} eV/A...",
        )

        self.runner = Runner(job, self)
        self.runner.progressed.connect(self._progressed)
        self.runner.completed.connect(self._loaded)
        self.runner.failed.connect(self._failed)
        self.runner.finished.connect(self._finished)
        self.runner.start()

    def _progressed(self, message: str) -> None:

        _status(self.status, message)

    def _finished(self) -> None:

        self.progress.setVisible(False)
        self.run_button.setEnabled(True)

    def _loaded(self, result) -> None:

        self.result = result

        self._fill()
        self._fill_analysis()
        self._draw()

        self.save_button.setEnabled(True)

        _status(
            self.status,
            f"Barrier {result.barrier_forward:.4f} eV, "
            f"saddle at image "
            f"{result.saddle_index:02d} of {len(result.energies)}.",
        )

        warnings = [
            note for note in result.notes if note.startswith("WARNING")
        ]
        others = [
            note
            for note in result.notes
            if not note.startswith("WARNING")
        ]

        self.notes.setText("\n\n".join(warnings + others))
        self.notes_area.setVisible(bool(result.notes))

        self._save_outputs()

    def _save_outputs(self) -> None:
        """
        Write the finished run out, if that was asked for.

        After the figure is drawn rather than before, because the plot
        is one of the things being saved and it does not exist until
        then. A failure here is reported and does not touch the result
        already on screen -- the run happened either way, and losing
        the window over a full disk would be the worse outcome.
        """

        if self.result is None or not self.save_outputs.isChecked():
            return

        folder = self.output_folder.text().strip() or self._default_output()

        if not folder:
            _status(
                self.status,
                "Nowhere to save the run: choose an output folder.",
                theme.WARNING_STYLE,
            )
            return

        try:
            saved = neb_vasp.save_run(
                self.result,
                folder,
                figure=self.panel.figure,
                **{
                    key: box.isChecked()
                    for key, box in self.save_what.items()
                },
            )

        except Exception as exc:  # noqa: BLE001 - the result stands
            _status(
                self.status,
                f"The run finished, but saving it failed: "
                f"{type(exc).__name__}: {exc}",
                theme.WARNING_STYLE,
            )
            return

        self.output_folder.setText(str(saved.root))

        message = (
            f"Barrier {self.result.barrier_forward:.4f} eV, saddle at "
            f"image {self.result.saddle_index:02d} of "
            f"{len(self.result.energies)}. Saved {saved.count} files "
            f"to {saved.root}."
        )

        if saved.skipped:

            message += " Not written: " + "; ".join(saved.skipped)

            _status(self.status, message, theme.WARNING_STYLE)
            return

        _status(self.status, message)

    def _failed(self, message: str) -> None:

        self.result = None

        self.save_button.setEnabled(False)

        _status(self.status, message, theme.WARNING_STYLE)

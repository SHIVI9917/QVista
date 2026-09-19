"""
The two ends of a VASP nudged elastic band.

Preprocessing takes two relaxed endpoints and writes the numbered image
folders VASP expects. Postprocessing reads the barrier back out of them.
Nothing here runs VASP; in between, the folder goes to whatever machine
holds the licence.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QComboBox,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.dialogs.plot_window import FigurePlaceholder, FigureWindow
from qvista.gui.sizing import fit_to_screen
from qvista.services import neb_vasp
from qvista.visualization import dynamics_plots

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

START_KEY = "neb_vasp/start"
END_KEY = "neb_vasp/end"
ROOT_KEY = "neb_vasp/root"
READ_KEY = "neb_vasp/read_root"

ANALYSIS_ROW_HEIGHT = 34

#: What the Show dropdown carries for the whole-path entry. Not an image
#: index, and deliberately out of range of one.
MEP_STAGE = -1

NOTES_LINES = 5


def _status(label: QLabel, text: str, style: str = STATUS_STYLE) -> None:

    label.setStyleSheet(style)
    label.setText(text)


def _pick_file(parent, title: str, start: str) -> str:

    chosen, _ = QFileDialog.getOpenFileName(
        parent,
        title,
        start,
        "VASP structure (POSCAR* CONTCAR* *.vasp);;All files (*)",
    )

    return chosen


# ================================================================
# Preprocessing
# ================================================================


class NebPreprocessDialog(QDialog):
    """Interpolate between two endpoints and write the image folders."""

    def __init__(self, start: str = "", parent=None) -> None:

        super().__init__(parent)

        self.result: neb_vasp.WriteResult | None = None

        self.setWindowTitle("NEB Pre-Processing")
        fit_to_screen(self, 880, 620)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_endpoints(layout)

        layout.addWidget(_hline())

        self._build_settings(layout)

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
        _status(self.status, "Choose the two relaxed endpoints.")

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.preview_button = QPushButton("Preview path")
        self.preview_button.clicked.connect(self._preview)

        footer.addWidget(self.preview_button)

        self.write_button = QPushButton("Write folders")
        self.write_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.write_button.clicked.connect(self._write)

        footer.addWidget(self.write_button)
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        if start:
            self.root.setText(str(Path(start) / neb_vasp.EXPERIMENT_DIR))

    # ------------------------------------------------------------

    def _build_endpoints(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Endpoints")

        column = QVBoxLayout(box)

        self.start = QLineEdit()
        self.start.setPlaceholderText("POSCAR of the starting site")

        self.end = QLineEdit()
        self.end.setPlaceholderText("POSCAR of the ending site")

        for label, field, key in (
            ("Start", self.start, START_KEY),
            ("End", self.end, END_KEY),
        ):
            row = QHBoxLayout()

            row.addWidget(QLabel(label))
            row.addWidget(field, 1)

            browse = QPushButton("Browse...")
            browse.clicked.connect(
                lambda _=False, f=field, k=key, t=label: self._browse(f, k, t)
            )

            row.addWidget(browse)

            column.addLayout(row)

        hint = QLabel(
            "Both must already be relaxed, with the same atoms in the "
            "same order. NEB holds them fixed."
        )
        hint.setStyleSheet(STATUS_STYLE)
        hint.setWordWrap(True)

        column.addWidget(hint)

        layout.addWidget(box)

    def _build_settings(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Path")

        column = QVBoxLayout(box)

        row = QHBoxLayout()

        row.addWidget(QLabel("Images"))

        self.images = QSpinBox()
        self.images.setRange(1, 50)
        self.images.setValue(neb_vasp.DEFAULT_IMAGES)
        self.images.setToolTip(
            "Intermediate images, endpoints excluded. This is the "
            "number IMAGES takes in the INCAR."
        )

        row.addWidget(self.images)

        self.climbing = QCheckBox("Climbing image")
        self.climbing.setChecked(True)
        self.climbing.setToolTip(
            "Puts the highest image on the saddle point, so the barrier "
            "is the barrier rather than a lower bound."
        )

        row.addWidget(self.climbing)

        row.addWidget(QLabel("EDIFFG"))

        self.ediffg = QDoubleSpinBox()
        self.ediffg.setRange(-1.0, -0.001)
        self.ediffg.setDecimals(3)
        self.ediffg.setSingleStep(0.01)
        self.ediffg.setValue(-0.03)
        self.ediffg.setSuffix(" eV/A")

        row.addWidget(self.ediffg)

        self.write_incar = QCheckBox("Write INCAR")
        self.write_incar.setChecked(True)

        row.addWidget(self.write_incar)

        row.addStretch(1)

        column.addLayout(row)

        target = QHBoxLayout()

        target.addWidget(QLabel("Write into"))

        self.root = QLineEdit()
        self.root.setPlaceholderText("a folder for this NEB")

        target.addWidget(self.root, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_root)

        target.addWidget(browse)

        column.addLayout(target)

        layout.addWidget(box)

    # ------------------------------------------------------------

    def _browse(self, field: QLineEdit, key: str, title: str) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        chosen = _pick_file(
            self,
            f"{title} structure",
            field.text() or settings.value(key, "", type=str),
        )

        if not chosen:
            return

        field.setText(chosen)

        settings.setValue(key, chosen)

    def _browse_root(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        chosen = QFileDialog.getExistingDirectory(
            self,
            "Where to write the NEB",
            self.root.text() or settings.value(ROOT_KEY, "", type=str),
        )

        if not chosen:
            return

        self.root.setText(str(Path(chosen) / neb_vasp.EXPERIMENT_DIR))

        settings.setValue(ROOT_KEY, chosen)

    # ------------------------------------------------------------

    def _endpoints(self):
        """Read the two structures, or report why not."""

        from pymatgen.core import Structure

        paths = [self.start.text().strip(), self.end.text().strip()]

        if not all(paths):
            _status(
                self.status, "Name both endpoints.", theme.WARNING_STYLE
            )
            return None

        loaded = []

        for path in paths:

            try:
                loaded.append(Structure.from_file(path))

            except Exception as exc:
                _status(
                    self.status,
                    f"Could not read {Path(path).name}: "
                    f"{type(exc).__name__}: {exc}",
                    theme.WARNING_STYLE,
                )
                return None

        return loaded

    def _preview(self) -> None:
        """Interpolate without writing anything."""

        endpoints = self._endpoints()

        if endpoints is None:
            return

        try:
            chain = neb_vasp.interpolate(
                endpoints[0], endpoints[1], self.images.value()
            )

            coordinates = neb_vasp.path_length(chain)

            moving = neb_vasp.moving_atoms(endpoints[0], endpoints[1])

        except neb_vasp.NebVaspError as exc:
            _status(self.status, str(exc), theme.WARNING_STYLE)
            return

        self._fill(chain, coordinates, moving, written=False)

        _status(
            self.status,
            f"{len(chain)} folders would be written. Path length "
            f"{coordinates[-1]:.3f} A, {len(moving)} atom(s) moving.",
        )

    def _write(self) -> None:
        """Write the folders."""

        endpoints = self._endpoints()

        if endpoints is None:
            return

        root = self.root.text().strip()

        if not root:
            _status(
                self.status,
                "Choose where to write the NEB.",
                theme.WARNING_STYLE,
            )
            return

        try:
            self.result = neb_vasp.write_path(
                root,
                endpoints[0],
                endpoints[1],
                images=self.images.value(),
                incar=(
                    neb_vasp.incar_text(
                        images=self.images.value(),
                        climbing=self.climbing.isChecked(),
                        ediffg=self.ediffg.value(),
                    )
                    if self.write_incar.isChecked()
                    else ""
                ),
            )

        except (neb_vasp.NebVaspError, OSError) as exc:
            _status(self.status, str(exc), theme.WARNING_STYLE)
            return

        chain = neb_vasp.interpolate(
            endpoints[0], endpoints[1], self.images.value()
        )

        self._fill(
            chain,
            self.result.coordinates,
            self.result.moving,
            written=True,
        )

        self.notes.setText("\n\n".join(self.result.notes))
        self.notes_area.setVisible(bool(self.result.notes))

        _status(
            self.status,
            f"Wrote {self.result.total_folders} folders under "
            f"{self.result.root}.",
        )

    def _fill(self, chain, coordinates, moving, written: bool) -> None:

        headers = ["Folder", "Role", "Reaction coordinate (A)"]

        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(chain))

        last = len(chain) - 1

        for index in range(len(chain)):

            role = (
                "start (fixed)"
                if index == 0
                else "end (fixed)"
                if index == last
                else f"image {index}"
            )

            for column, text in enumerate(
                (
                    neb_vasp.folder_name(index),
                    role,
                    f"{coordinates[index]:.3f}",
                )
            ):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                self.table.setItem(index, column, item)

class Reader(QThread):
    """Reading a dozen OUTCARs is slow enough to be worth a thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: str, parent=None) -> None:

        super().__init__(parent)

        self.folder = folder

    def run(self) -> None:

        try:
            self.completed.emit(neb_vasp.read_path(self.folder))

        except neb_vasp.NebVaspError as exc:
            self.failed.emit(str(exc))

        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class NebResultsView:
    """
    Everything that shows a finished NEB, wherever it came from.

    Both routes end with the same object and the same question -- where
    is the barrier, and what is the migrating atom doing -- so both get
    the same figure, the same two tables and the same structure viewer.
    Reading a folder and running a force field differ only in how the
    result is produced, and that difference belongs above this, not
    inside it.

    Expects the host to provide ``result``, ``table``, ``analysis``,
    ``analysis_label``, ``analysis_row``, ``cutoff``, ``stage``,
    ``status``, ``panel`` and ``figures``.
    """

    def _fill(self) -> None:

        if self.result is None:
            return

        rows = self.result.rows

        self.table.clear()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.table.setRowCount(len(rows))

        for index, (label, value) in enumerate(rows):

            for column, text in enumerate((label, value)):

                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                self.table.setItem(index, column, item)

    def _draw(self) -> None:

        if self.result is None:
            return

        dynamics_plots.draw_neb_profile(
            self.panel.figure,
            self.result,
            title=(
                f"{self.result.n_images} images, barrier "
                f"{self.result.barrier_forward:.3f} eV"
            ),
        )

        self.panel.refresh()

        self.figures.show_figure(reference=self)

    def _fill_analysis(self) -> None:
        """The migrating atom's environment, or nothing if there is none."""

        from qvista.services import neb_vasp

        rows = neb_vasp.analysis_rows(self.result)

        self.analysis.clear()

        if not rows:

            self.analysis.setVisible(False)
            self.analysis_label.setVisible(False)
            self.analysis_row.setVisible(False)

            return

        columns = neb_vasp.ANALYSIS_COLUMNS

        self.analysis.setColumnCount(len(columns))
        self.analysis.setHorizontalHeaderLabels(list(columns))
        self.analysis.setRowCount(len(rows))

        for index, values in enumerate(rows):

            for column, text in enumerate(values):

                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                self.analysis.setItem(index, column, item)

        self.analysis_label.setText(
            f"Migrating atom - index {self.result.moving_atom + 1} "
            f"(1-based), bonds within "
            f"{self.result.neighbour_cutoff:g} A"
        )

        chosen = self.stage.currentIndex()

        self.stage.clear()

        for environment in self.result.environments:
            self.stage.addItem(
                f"{environment.label}  ({environment.image:02d})",
                environment.image,
            )

        # The whole path in one picture, as against one point on it.
        self.stage.addItem("MEP  (all images)", MEP_STAGE)

        if 0 <= chosen < self.stage.count():
            self.stage.setCurrentIndex(chosen)

        self.analysis.setVisible(True)
        self.analysis_label.setVisible(True)
        self.analysis_row.setVisible(True)

    def _recount(self) -> None:
        """Redo the coordination at the cutoff now set."""

        if self.result is None:
            return

        neb_vasp.recount(self.result, self.cutoff.value())

        self._fill_analysis()

    def _show_structure(self) -> None:
        """
        Put the chosen image in the 3D viewer, migrating atom selected.

        The main window's viewer rather than one of our own: it already
        has the representation, the colours and the boundary the user
        set up, and a second viewer with its own settings would be a
        different picture of the same thing.
        """

        if self.result is None or self.result.moving_atom is None:
            return

        image = self.stage.currentData()

        structures = list(getattr(self.result, "structures", []) or [])

        if image == MEP_STAGE:

            structure = neb_vasp.mep_structure(self.result)

            if structure is None:
                _status(
                    self.status,
                    "There is no migration path to draw.",
                    theme.WARNING_STYLE,
                )
                return

            # No highlight. The trail *is* the subject here, and a
            # selection cage around one copy of it points at the least
            # interesting thing in the picture.
            self._load_into_viewer(
                structure,
                f"the path, {len(structures)} images of atom "
                f"{self.result.moving_atom + 1}",
                highlight=False,
            )
            return

        if image is None or not (0 <= image < len(structures)):
            return

        structure = structures[image]

        if structure is None:
            _status(
                self.status,
                f"Image {image:02d} has no POSCAR or CONTCAR to show.",
                theme.WARNING_STYLE,
            )
            return

        self._load_into_viewer(structure, f"Image {image:02d}")

    def _load_into_viewer(
        self, structure, described: str, highlight: bool = True
    ) -> None:
        """
        Put one pymatgen structure in the main window's viewer.

        The main viewer rather than one of our own: it already has the
        representation, the colours and the boundary the user set up,
        and a second viewer with its own settings would be a different
        picture of the same thing.
        """

        window = self.parent()

        workspace = getattr(window, "workspace", None)

        if workspace is None:
            _status(
                self.status,
                "No 3D viewer to show it in.",
                theme.WARNING_STYLE,
            )
            return

        from ase import Atoms

        from qvista.models.structure import Structure

        atoms = Atoms(
            symbols=[str(site.specie) for site in structure],
            positions=[site.coords for site in structure],
            cell=structure.lattice.matrix,
            pbc=True,
        )

        workspace.load_structure(Structure(atoms))

        if not highlight:
            _status(self.status, f"{described} in the viewer.")
            return

        found = workspace.viewer.select_index(self.result.moving_atom)

        _status(
            self.status,
            f"{described} in the viewer"
            + (
                f", atom {self.result.moving_atom + 1} highlighted."
                if found
                else " -- the migrating atom is outside the display "
                "boundary."
            ),
        )

    def _save(self) -> None:

        if self.result is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save profile", "neb.csv", "CSV (*.csv)"
        )

        if not filename:
            return

        try:
            Path(filename).write_text(
                self.result.to_csv(), encoding="utf-8"
            )

        except OSError as exc:
            _status(
                self.status,
                f"Could not write the file: {exc}",
                theme.WARNING_STYLE,
            )
            return

        _status(self.status, f"Saved to {filename}.")


# ================================================================
# Postprocessing
# ================================================================


class NebPostprocessDialog(NebResultsView, QDialog):
    """Read the barrier out of a finished VASP NEB."""

    def __init__(self, start: str = "", parent=None) -> None:

        super().__init__(parent)

        self.result = None
        self.reader: Reader | None = None

        self.setWindowTitle("NEB Post-Processing")
        fit_to_screen(self, 900, 640)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        row = QHBoxLayout()

        row.addWidget(QLabel("NEB folder"))

        self.folder = QLineEdit()
        self.folder.setPlaceholderText(
            "the directory holding 00, 01, 02, ..."
        )
        self.folder.editingFinished.connect(self._read)

        row.addWidget(self.folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        read = QPushButton("Read")
        read.clicked.connect(self._read)

        row.addWidget(read)

        layout.addLayout(row)

        layout.addWidget(_hline())

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

        # The analysis gets a table of its own rather than more rows in
        # the summary. It answers a different question -- not "what did
        # the calculation find" but "why is the barrier that height" --
        # and it has columns the summary has no use for.
        self.analysis_label = QLabel("Migrating atom")
        self.analysis_label.setStyleSheet(
            f"color: {theme.IRIS_GLOW}; font-weight: 700;"
        )
        self.analysis_label.setVisible(False)

        layout.addWidget(self.analysis_label)

        # The cutoff is a convention, not a measurement, so it is a
        # control rather than a constant -- and the structure it applies
        # to is one click away beside it.
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
        self.cutoff.setToolTip(
            "Longest separation counted as a bond. Wide enough and the "
            "next shell of metal atoms joins the coordination number, "
            "which is how a six-coordinate lithium came back as "
            "sixteen-coordinate."
        )
        self.cutoff.valueChanged.connect(self._recount)

        controls.addWidget(self.cutoff)

        controls.addSpacing(12)

        controls.addWidget(QLabel("Show"))

        self.stage = QComboBox()
        self.stage.setToolTip(
            "Load this image into the viewer with the migrating atom "
            "highlighted."
        )

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

        # Taller rows and room around the text. The default packs six
        # columns of numbers into a strip that reads as one block.
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
        _status(self.status, "Choose the folder holding the image folders.")

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

        if start:
            self.folder.setText(start)

    # ------------------------------------------------------------

    def _browse(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        chosen = QFileDialog.getExistingDirectory(
            self,
            "NEB folder",
            self.folder.text() or settings.value(READ_KEY, "", type=str),
        )

        if not chosen:
            return

        self.folder.setText(chosen)

        settings.setValue(READ_KEY, chosen)

        self._read()

    def _read(self) -> None:

        folder = self.folder.text().strip()

        if not folder:
            return

        if self.reader is not None and self.reader.isRunning():
            return

        _status(self.status, f"Reading {Path(folder).name}...")

        self.reader = Reader(folder, self)
        self.reader.completed.connect(self._loaded)
        self.reader.failed.connect(self._failed)
        self.reader.start()

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

        warnings = [n for n in result.notes if n.startswith("WARNING")]
        others = [n for n in result.notes if not n.startswith("WARNING")]

        self.notes.setText("\n\n".join(warnings + others))
        self.notes_area.setVisible(bool(result.notes))

    def _failed(self, message: str) -> None:

        self.result = None

        self.save_button.setEnabled(False)

        _status(self.status, message, theme.WARNING_STYLE)



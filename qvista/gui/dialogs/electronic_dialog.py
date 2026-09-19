"""
The Electrochemical Tools dialogs: DOS, bands, band centres.

Three windows over one service.  Each menu entry opens the window it
belongs to with the right mode already picked, so a button that says
"Projected Orbital wise DOS" lands on exactly that, and the other modes
are one click away rather than another trip through the menu.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
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
from qvista.gui.dialogs.side_window import SideWindow
from qvista.services import electronic as el
from qvista.gui.widgets.flow_layout import flow_widget
from qvista.services.orbital import parse_atom_list
from qvista.visualization import electronic_plots

#: How tall the band-centre selection table starts. Five rows and a
#: header. This is the control the window is built around -- a row is
#: one thing on the plot -- so it gets the room to be read at a glance
#: rather than squinted at.
TABLE_HEIGHT = 230

#: How tall the per-element band blocks scroll before they scroll.
BANDS_HEIGHT = 210

#: Checkboxes per row in the element and orbital grids. Five keeps the
#: nine orbitals to two tidy rows at any dialog width.
BOXES_PER_ROW = 5

#: How tall one row of the results table is assumed to be before there
#: is a row to measure.
ROW_HEIGHT = 30

#: Border and scrollbar allowance around the results table.
TABLE_PADDING = 8

#: The results window never shrinks below one useful row, and never
#: grows past a screenful -- past that it scrolls instead.
MIN_TABLE_HEIGHT = 90
MAX_TABLE_HEIGHT = 520

#: Room for the summary line and the buttons under the table.
WINDOW_FURNITURE = 130

#: The configuration table sits inside the scrolling controls, so it
#: gets less room before it scrolls than a window of its own would.
CONFIG_MOST = 220

#: ...and the configuration summary under them.
CONFIG_HEIGHT = 120

#: How tall one row of it is. Qt's default leaves a combo box wedged
#: into its cell with no breathing room, which is what made the table
#: hard to read at a glance.
TABLE_ROW_HEIGHT = 34

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

RUN_KEY = "electronic/run_root"

NOTES_LINES = 4

#: "Let the run decide" in the combo boxes that offer it.
AUTO = "auto"


def _status(label: QLabel, text: str, style: str = STATUS_STYLE) -> None:

    label.setStyleSheet(style)
    label.setText(text)


class Chip(QPushButton):
    """A small checkable box, matching the other analysis dialogs."""

    def __init__(self, text: str, parent=None) -> None:

        super().__init__(text, parent)

        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)


class Loader(QThread):
    """Parsing a vasprun with projections takes a while."""

    progressed = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: str, projected: bool, dos: bool, parent=None):

        super().__init__(parent)

        self.folder = folder
        self.projected = projected
        self.dos = dos

    def run(self) -> None:

        try:
            self.progressed.emit("Parsing vasprun.xml...")

            self.completed.emit(
                el.load(
                    self.folder, projected=self.projected, dos=self.dos
                )
            )

        except el.ElectronicError as exc:
            self.failed.emit(str(exc))

        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class Worker(QThread):
    """Whatever the dialog asked the service for."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, job, parent=None) -> None:

        super().__init__(parent)

        self.job = job

    def run(self) -> None:

        try:
            self.completed.emit(self.job())

        except el.ElectronicError as exc:
            self.failed.emit(str(exc))

        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def _heading(text: str) -> QLabel:
    """A section title, in the reference's ### style."""

    label = QLabel(text)
    label.setStyleSheet(
        f"color: {theme.IRIS_GLOW}; font-weight: 700; font-size: 10pt;"
    )

    return label


class ElectronicDialog(QDialog):
    """Shared scaffolding: pick a folder, load it, plot, tabulate, save."""

    #: Filled in by each subclass.
    TITLE = "Electronic structure"
    NEEDS_PROJECTIONS = True
    NEEDS_DOS = True

    #: Tab name inside the figure window.
    FIGURE = "Figure"

    #: Whether the Contribution tab is offered at all.
    #:
    #: It answers "how much of the occupied manifold is this series",
    #: which needs curves to integrate. A band-centre window has none --
    #: every row of it is already a single number -- so it shows the
    #: summary alone rather than a tab that is permanently empty.
    SHOW_CONTRIBUTION = True

    #: Whether the footer offers its own Save button. False for a window
    #: whose results live in a window of their own, which carries one.
    SHOWS_FOOTER_SAVE = True

    def __init__(self, start: str = "", mode: str = "", parent=None) -> None:

        super().__init__(parent)

        self.run: el.Run | None = None
        self.result = None
        self.loader: Loader | None = None
        self.worker: Worker | None = None

        self.setWindowTitle(self.TITLE)
        fit_to_screen(self, 1140, 800)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_source(layout)
        self._build_controls(layout)

        layout.addWidget(_hline())

        # The figure gets a window of its own. A band structure and a
        # projected DOS both want height, and sharing a splitter with the
        # table left them a strip.
        self.figures = FigureWindow(self.TITLE, (self.FIGURE,), self)

        self.panel = self.figures.panel_for()

        self.tabs = QTabWidget()

        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.tabs.addTab(self.table, "Summary")

        # What each projected series actually accounts for. The curves
        # show the shape; this shows the size, which is the question a
        # projected figure is usually being asked.
        self.contribution: QTableWidget | None = None

        if self.SHOW_CONTRIBUTION:

            self.contribution = QTableWidget()
            self.contribution.verticalHeader().setVisible(False)
            self.contribution.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch
            )

            self.tabs.addTab(self.contribution, "Contribution")

        layout.addWidget(FigurePlaceholder(self.figures, self))
        layout.addWidget(self.tabs, 1)

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
        _status(self.status, "Choose the folder holding the VASP run.")

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.save_button = QPushButton("Save table...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)

        # A window that puts its results in a window of their own has a
        # Save button there, beside the table it saves. A second one
        # down here would save the same thing from further away.
        self.save_button.setVisible(self.SHOWS_FOOTER_SAVE)

        footer.addWidget(self.save_button)
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        if mode:
            self._pick_mode(mode)

        if start:
            self.folder.setText(start)

    # ------------------------------------------------------------

    def _build_source(self, layout: QVBoxLayout) -> None:

        row = QHBoxLayout()

        row.addWidget(QLabel("Run folder"))

        self.folder = QLineEdit()
        self.folder.setPlaceholderText("the directory holding vasprun.xml")
        self.folder.editingFinished.connect(self._load)

        row.addWidget(self.folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        load = QPushButton("Load")
        load.clicked.connect(self._load)

        row.addWidget(load)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(150)

        row.addWidget(self.progress)

        layout.addLayout(row)

    def _build_controls(self, layout: QVBoxLayout) -> None:
        """Subclasses put their own row here."""

    # ------------------------------------------------------------
    # The orbital projection, at two levels of detail
    # ------------------------------------------------------------

    def _build_orbital_views(self, column: QVBoxLayout) -> None:
        """
        The row that appears when the projection is "orbital".

        Same two choices as the Orbital window, from the same constants,
        so "resolved" means the same nine channels in the same order
        wherever it is offered.
        """

        self.orbital_row = QWidget()

        row = QHBoxLayout(self.orbital_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        row.addWidget(QLabel("Orbitals"))

        self.orbital_view = el.DEFAULT_ORBITAL_VIEW
        self.view_chips: dict[str, Chip] = {}

        for key, label in el.ORBITAL_VIEWS.items():

            chip = Chip(label)
            chip.setChecked(key == self.orbital_view)
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_orbital_view(name)
            )

            self.view_chips[key] = chip
            row.addWidget(chip)

        row.addStretch(1)

        column.addWidget(self.orbital_row)

        self.orbital_row.setVisible(False)

    def _pick_orbital_view(self, name: str) -> None:

        self.orbital_view = name

        for key, chip in self.view_chips.items():
            chip.setChecked(key == name)

    def _show_orbital_views(self) -> None:
        """Only meaningful when the projection is onto orbitals."""

        row = getattr(self, "orbital_row", None)

        if row is not None:
            row.setVisible(getattr(self, "mode", "") == "orbital")

    # ------------------------------------------------------------
    # Which species, as a row of tick boxes
    #
    # Shared by every window with an element-wise mode, so ticking a
    # species means the same thing and looks the same wherever it is
    # offered.
    # ------------------------------------------------------------

    def _build_element_row(self, column: QVBoxLayout) -> None:
        """
        Which species to draw, as a row of tick boxes.

        Filled in when a run is loaded, because the species are not
        known until then. Everything is ticked to begin with, so the
        mode behaves as it always did until something is unticked.
        """

        self.element_row = QWidget()

        row = QHBoxLayout(self.element_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        row.addWidget(QLabel("Elements"))

        self.element_boxes: dict[str, QCheckBox] = {}

        self.element_layout = row

        self.all_elements = QCheckBox("All")
        self.all_elements.setChecked(True)
        self.all_elements.toggled.connect(self._toggle_all_elements)

        row.addWidget(self.all_elements)

        row.addStretch(1)

        column.addWidget(self.element_row)

        self.element_row.setVisible(False)

    def _offer_elements(self, run: el.Run) -> None:
        """Rebuild the tick boxes for the run just loaded."""

        for box in self.element_boxes.values():
            self.element_layout.removeWidget(box)
            box.deleteLater()

        self.element_boxes = {}

        # Before the trailing stretch, after the All box.
        position = 2

        for element in run.elements():

            box = QCheckBox(element)
            box.setChecked(True)
            box.toggled.connect(self._element_toggled)

            self.element_boxes[element] = box

            self.element_layout.insertWidget(position, box)

            position += 1

        self._show_elements()

    def _toggle_all_elements(self, on: bool) -> None:

        for box in self.element_boxes.values():
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)

    def _element_toggled(self) -> None:
        """Keep the All box honest without letting it fight the others."""

        every = all(
            box.isChecked() for box in self.element_boxes.values()
        )

        self.all_elements.blockSignals(True)
        self.all_elements.setChecked(every)
        self.all_elements.blockSignals(False)

    def _chosen_elements(self) -> list[str]:
        """The species ticked, or empty for all of them."""

        if self.all_elements.isChecked():
            return []

        return [
            name for name, box in self.element_boxes.items()
            if box.isChecked()
        ]

    def _show_elements(self) -> None:

        row = getattr(self, "element_row", None)

        if row is None:
            return

        basis = getattr(self, "basis", None)

        # Shown when a projection is being made *and* it is made of
        # species. The mode used to be called "element", which is what
        # this tested for; after the split that name lives on the basis
        # combo instead, and testing the old one hid the row for good.
        row.setVisible(
            getattr(self, "mode", "") == "projected"
            and (basis is None or basis.currentData() == "element")
            and bool(self.element_boxes)
        )

    # ------------------------------------------------------------

    def _browse(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.folder.text() or settings.value(RUN_KEY, "", type=str)

        chosen = QFileDialog.getExistingDirectory(
            self, "VASP run folder", start
        )

        if chosen:
            self.folder.setText(chosen)
            settings.setValue(RUN_KEY, chosen)
            self._load()

    def _load(self) -> None:

        folder = self.folder.text().strip()

        if not folder or not Path(folder).is_dir():
            _status(
                self.status, "That folder does not exist.",
                theme.WARNING_STYLE,
            )
            return

        if self.loader is not None and self.loader.isRunning():
            return

        self.progress.setVisible(True)

        self.loader = Loader(
            folder, self.NEEDS_PROJECTIONS, self.NEEDS_DOS, parent=self
        )

        self.loader.progressed.connect(
            lambda text: _status(self.status, text)
        )
        self.loader.completed.connect(self._on_loaded)
        self.loader.failed.connect(self._on_failed)
        self.loader.finished.connect(
            lambda: self.progress.setVisible(False)
        )

        self.loader.start()

    def _on_loaded(self, run: el.Run) -> None:

        self.run = run

        self._offer(run)

        self.notes.setText("\n".join(f"- {note}" for note in run.notes))
        self.notes_area.setVisible(bool(run.notes))

        _status(
            self.status,
            f"{run.natoms} atoms, {run.nbands} bands, {run.nkpoints} "
            f"k-points"
            + (", spin polarised" if run.spin_polarised else "")
            + ". Press Run.",
            theme.SUCCESS_STYLE,
        )

    def _offer(self, run: el.Run) -> None:
        """Subclasses fill in whatever depends on the structure."""

    def _on_failed(self, message: str) -> None:

        _status(self.status, message, theme.ERROR_STYLE)

    def _pick_mode(self, mode: str) -> None:
        """Subclasses select their starting mode."""

    def _basis_changed(self, _index: int = 0) -> None:
        """
        Show the selector the chosen basis actually uses.

        Element wise reads the ticked species and ignores the atom
        field; atom wise does the reverse. Leaving both live invites
        filling in one and wondering why the other was used.
        """

        basis = self.basis.currentData()

        by_atom = basis == "atom"

        if hasattr(self, "atoms"):
            self.atoms.setEnabled(by_atom)

        if hasattr(self, "all_elements"):
            self.all_elements.setEnabled(not by_atom)

        for box in getattr(self, "element_boxes", {}).values():
            box.setEnabled(not by_atom)

        # The row itself follows the basis too, not just its contents.
        self._show_elements()

    # ------------------------------------------------------------

    def _start(self, job) -> None:
        """Run ``job`` off the GUI thread and hand the result back."""

        if self.worker is not None and self.worker.isRunning():
            return

        self.progress.setVisible(True)

        self.worker = Worker(job, parent=self)

        self.worker.completed.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(
            lambda: self.progress.setVisible(False)
        )

        self.worker.start()

    def _on_result(self, result) -> None:

        self.result = result

        try:
            self._draw()
            self._fill()
            self._fill_contribution()

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            _status(self.status, f"{type(exc).__name__}: {exc}",
                    theme.ERROR_STYLE)
            return

        notes = list(getattr(result, "notes", []) or [])

        if notes:
            self.notes.setText("\n".join(f"- {note}" for note in notes))
            self.notes_area.setVisible(True)

        self.save_button.setEnabled(True)

        _status(self.status, self._summary(), theme.SUCCESS_STYLE)

    def _draw(self) -> None:
        raise NotImplementedError

    def _fill(self) -> None:
        raise NotImplementedError

    def _shares(self) -> list:
        """
        What each series contributes. Empty when the idea does not apply.

        Overridden by the two windows that project onto something; the
        band-centre window already *is* a contribution table.
        """

        return []

    def _fill_contribution(self) -> None:
        """The Contribution tab: how much each series accounts for."""

        table = self.contribution

        if table is None:
            return

        try:
            shares = self._shares()

        except Exception:  # noqa: BLE001 - an empty tab, never a crash
            shares = []

        table.clear()

        if not shares:
            table.setRowCount(0)
            table.setColumnCount(0)
            self.tabs.setTabEnabled(1, False)
            return

        self.tabs.setTabEnabled(1, True)

        spinful = any(share.spin for share in shares)

        headers = ["Series"]

        if spinful:
            headers.append("Spin")

        headers.extend(self.SHARE_COLUMNS)

        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(shares))

        for row, share in enumerate(shares):

            column = 0

            self._cell(row, column, share.label, table=table)
            column += 1

            if spinful:
                self._cell(row, column, share.spin, table=table)
                column += 1

            self._cell(
                row, column, f"{share.occupied:.4f}", table=table
            )
            self._cell(
                row, column + 1, f"{share.whole:.4f}", table=table
            )
            self._cell(
                row,
                column + 2,
                f"{share.percent:.2f}%",
                dim=share.empty,
                table=table,
            )

    #: The three numeric columns of the Contribution tab. Named by the
    #: subclass, because "occupied" means an integrated density in one
    #: window and a mean weight per state in the other.
    SHARE_COLUMNS = ("Below E_F", "All energies", "Share")

    def _summary(self) -> str:
        return "Done."

    # ------------------------------------------------------------

    def _cell(
        self,
        row: int,
        column: int,
        text: str,
        dim: bool = False,
        table: QTableWidget | None = None,
    ):

        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        if dim:
            item.setForeground(Qt.gray)

        (self.table if table is None else table).setItem(row, column, item)

    def _save(self) -> None:
        """Write whatever is on screen as CSV."""

        if self.result is None:
            return

        showing = self.tabs.currentWidget()

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save table", f"{self.TITLE.lower()}.csv", "CSV (*.csv)"
        )

        if not chosen:
            return

        headers = [
            showing.horizontalHeaderItem(column).text()
            for column in range(showing.columnCount())
        ]

        try:
            with open(chosen, "w", newline="", encoding="utf-8") as handle:

                writer = csv.writer(handle)

                writer.writerow([f"# QVista {self.TITLE}"])

                if self.run is not None:
                    writer.writerow([f"# {self.run.directory}"])

                    for note in self.run.notes:
                        writer.writerow([f"# {note}"])

                writer.writerow(headers)

                for row in range(showing.rowCount()):
                    writer.writerow(
                        [
                            showing.item(row, column).text()
                            if showing.item(row, column)
                            else ""
                            for column in range(showing.columnCount())
                        ]
                    )

        except OSError as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not write that file: {exc}"
            )
            return

        _status(self.status, f"Written to {Path(chosen).name}.")


# ================================================================
# Density of states
# ================================================================


class DosDialog(ElectronicDialog):
    """The four DOS entries, on one window."""

    TITLE = "Projected Density of States"
    FIGURE = "Density of states"

    MODES = {
        # Two questions, not six. "Do I want the whole thing, or a part
        # of it" comes first; *which* part is a separate choice, and
        # putting them on one row made the pairing everybody actually
        # wants -- the d states of one species -- unreachable, because
        # picking "Element wise" ruled out "Orbital wise".
        #
        # A chart of its own rather than a dashed line laid over every
        # other one: the total is the sum over every atom, so on a
        # forty-atom cell it dwarfs any projection and flattens it.
        "plain": "Plain (TDOS)",
        "projected": "Projected",
        "kpoint": "By k-point",
        "band": "By band index",
    }

    def _build_controls(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Projection")

        column = QVBoxLayout(box)

        # -- what a curve is ------------------------------------
        top = QHBoxLayout()
        top.setSpacing(6)

        self.mode = "projected"
        self.chips: dict[str, Chip] = {}

        for key, label in self.MODES.items():

            chip = Chip(label)
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_mode(name)
            )

            self.chips[key] = chip
            top.addWidget(chip)

        self.chips[self.mode].setChecked(True)

        top.addStretch(1)

        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.clicked.connect(self._run)

        top.addWidget(self.run_button)

        column.addLayout(top)

        # -- selection ------------------------------------------
        second = QHBoxLayout()
        second.setSpacing(10)

        second.addWidget(QLabel("Atoms"))

        self.atoms = QLineEdit()
        self.atoms.setPlaceholderText("1-6, 9   (numbers or ranges)")
        self.atoms.setMaximumWidth(200)

        second.addWidget(self.atoms)

        second.addWidget(QLabel("k-points"))

        self.kpoints = QLineEdit()
        self.kpoints.setPlaceholderText("all")
        self.kpoints.setMaximumWidth(140)

        second.addWidget(self.kpoints)

        second.addWidget(QLabel("Bands"))

        self.bands = QLineEdit()
        self.bands.setPlaceholderText("all")
        self.bands.setMaximumWidth(140)

        second.addWidget(self.bands)

        second.addWidget(QLabel("Broadening"))

        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.005, 2.0)
        self.sigma.setDecimals(3)
        self.sigma.setSingleStep(0.01)
        self.sigma.setValue(el.DEFAULT_SIGMA)
        self.sigma.setSuffix(" eV")

        second.addWidget(self.sigma)

        self.shells: dict[str, QCheckBox] = {}

        for shell in el.SHELLS:

            box_ = QCheckBox(shell)
            box_.setChecked(True)

            self.shells[shell] = box_
            second.addWidget(box_)

        # What the selection is made of. Independent of the shells
        # above, which is the point: the two together are what lets a
        # single species' d states be asked for.
        second.addWidget(QLabel("of"))

        self.basis = QComboBox()

        for key, label in el.BASES.items():
            self.basis.addItem(label, key)

        self.basis.setToolTip(
            "Element wise sums over every atom of a species. Atom wise "
            "takes the ones you name, which is what to use when two "
            "sites of the same element are not equivalent."
        )
        self.basis.currentIndexChanged.connect(self._basis_changed)

        second.addWidget(self.basis)

        # Splits each shell into its individual orbitals. For d that is
        # the t2g/eg distinction, which summing the shell erases -- and
        # which is the whole content of a crystal-field argument.
        self.resolved = QCheckBox("Resolved")
        self.resolved.setToolTip(
            "One curve per orbital instead of one per shell: d becomes "
            "dxy, dyz, dz2, dxz, dx2, so the t2g and eg sets can be "
            "told apart."
        )

        second.addWidget(self.resolved)

        # Apply it once now. The combo starts on the first entry and
        # setting it there again emits nothing, so without this the
        # window opens element wise with the atom field still live.
        self._basis_changed()

        second.addWidget(QLabel("Plot"))

        # How the curve is drawn between the calculated points. A
        # redraw, not a re-run: the states are already integrated, so
        # switching costs a repaint.
        self.curve_style = QComboBox()

        for key, label in electronic_plots.CURVE_STYLES.items():
            self.curve_style.addItem(label, key)

        self.curve_style.setToolTip(
            "Beta-spline draws a smooth curve through every calculated "
            "point, so peaks keep their height and position. Line joins "
            "them straight, which is what to use when judging whether "
            "the energy grid is fine enough -- smoothing hides the "
            "corners that say it is not."
        )
        self.curve_style.currentIndexChanged.connect(self._restyle)

        second.addWidget(self.curve_style)

        second.addStretch(1)

        column.addLayout(second)

        self._build_orbital_views(column)
        self._build_element_row(column)

        self.hint = QLabel("")
        self.hint.setStyleSheet(STATUS_STYLE)
        self.hint.setWordWrap(True)

        column.addWidget(self.hint)

        layout.addWidget(box)

        self._describe()

    def _pick_mode(self, name: str) -> None:

        if name not in self.MODES:
            return

        self.mode = name

        for key, chip in self.chips.items():
            chip.setChecked(key == name)

        self.atoms.setEnabled(name == "atom")

        for widget in (self.kpoints, self.bands, self.sigma):
            widget.setEnabled(name in ("kpoint", "band"))

        self._show_orbital_views()
        self._show_elements()

        self._describe()

    def _offer(self, run: el.Run) -> None:

        self._offer_elements(run)

        self._describe()

        self.hint.setText(
            self.hint.text()
            + f"  Elements: {', '.join(run.elements())}."
        )

    def _describe(self) -> None:

        over = {
            "total": "the whole cell, unprojected",
            "atom": "each atom named",
            "element": "each species",
            "orbital": "each orbital of the whole cell",
            "kpoint": "the k-points named",
            "band": "the bands named",
        }.get(self.mode, self.mode)

        self.hint.setText(
            f"One curve per {over}, referenced to the Fermi level."
        )

    def _restyle(self) -> None:
        """
        Redraw with the chosen curve style, when there is something to
        redraw.

        The states are already integrated, so this never re-runs
        anything -- and it does nothing at all before the first run,
        rather than complaining about a result that was never asked for.
        """

        if self.result is not None:
            self._draw()

    def _run(self) -> None:

        run = self.run

        if run is None:
            _status(self.status, "Load a run first.", theme.WARNING_STYLE)
            return

        shells = [
            shell for shell, box in self.shells.items() if box.isChecked()
        ]

        if self.mode in ("kpoint", "band"):

            kpoints = _numbers(self.kpoints.text())
            bands = _numbers(self.bands.text())

            projection = (
                "orbital" if len(shells) < len(el.SHELLS) else "total"
            )

            self._start(
                lambda: el.broadened_dos(
                    run,
                    kpoints=kpoints,
                    bands=bands,
                    projection=projection,
                    shells=shells,
                    orbital_view=self.orbital_view,
                    sigma=self.sigma.value(),
                )
            )
            return

        if self.mode == "plain":

            self._start(
                lambda: el.projected_dos(run, projection="total")
            )
            return

        # Projected: the selection and the shells together, which is
        # what projected_dos could not express.
        self._start(
            lambda: el.projected_curves(
                run,
                basis=self.basis.currentData(),
                elements=self._chosen_elements(),
                atoms=self.atoms.text().strip(),
                shells=shells,
                resolved=self.resolved.isChecked(),
            )
        )

    def _draw(self) -> None:

        electronic_plots.draw_dos(
            self.panel.figure,
            self.result,
            title=f"{self.TITLE} -- {self.MODES.get(self.mode, self.mode)}",
            curve_style=self.curve_style.currentData(),
            # Never as an overlay. On a projected chart the total is a
            # different quantity on a different scale and flattens what
            # it is laid over; on the TDOS chart it is already the
            # curves, and drawing it again would double every line.
            with_total=False,
        )

        self.panel.refresh()

        self.figures.show_figure(reference=self)

    def _shares(self) -> list:
        """Integrated weight per curve, below the Fermi level."""

        return el.contributions(self.result)

    COLUMNS = (
        "Curve",
        "Spin",
        "States below E_F",
        "States total",
    )

    def _fill(self) -> None:
        """Where each curve has its weight, which is what a table adds."""

        shares = self._shares()

        self.table.clear()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setRowCount(len(shares))

        for row, share in enumerate(shares):

            self._cell(row, 0, share.label)
            self._cell(row, 1, share.spin)
            self._cell(row, 2, f"{share.occupied:.4f}")
            self._cell(row, 3, f"{share.whole:.4f}")

    def _summary(self) -> str:

        curves = self.result.curves if self.result is not None else []

        if not curves:
            return "No curves."

        return f"{len(curves)} curve(s)."


class BandDialog(ElectronicDialog):
    """
    The dispersion, with the projection drawn as a fat band.

    Two sources of eigenvalues. A proper band structure is a
    non-self-consistent run along a k-path; most people have only the
    SCF mesh they already ran, and plotting that against k-point index
    still answers where the band edges are and which atoms they sit on.
    Refusing until they go away and run VASP again throws that away, so
    both are offered and the figure says which it drew.
    """

    TITLE = "Band Structure"
    FIGURE = "Band structure"

    MODES = {
        # The same two-question split as the DOS window, for the same
        # reason: "Element wise" and "Orbital wise" as alternatives made
        # a fat band showing one species' d character impossible to ask
        # for, and that is the figure a crystal-field argument needs.
        "plain": "Plain bands",
        "projected": "Projected",
    }

    def _build_controls(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Projection")

        column = QVBoxLayout(box)

        # -- what a series is -----------------------------------
        top = QHBoxLayout()
        top.setSpacing(6)

        self.mode = "projected"
        self.chips: dict[str, Chip] = {}

        for key, label in self.MODES.items():

            chip = Chip(label)
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_mode(name)
            )

            self.chips[key] = chip
            top.addWidget(chip)

        self.chips[self.mode].setChecked(True)

        top.addStretch(1)

        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.clicked.connect(self._run)

        top.addWidget(self.run_button)

        column.addLayout(top)

        # -- where the eigenvalues come from --------------------
        second = QHBoxLayout()
        second.setSpacing(10)

        second.addWidget(QLabel("Source"))

        self.source = QComboBox()

        for key, label in el.BAND_MODES.items():
            self.source.addItem(label, key)

        self.source.setToolTip(
            "A k-path needs a non-self-consistent run. The SCF mesh is "
            "not a path, but plotting it against k-point index still "
            "shows the band edges and the gap."
        )

        second.addWidget(self.source)

        second.addWidget(QLabel("Atoms"))

        self.atoms = QLineEdit()
        self.atoms.setPlaceholderText("1-6, 9   (numbers or ranges)")
        self.atoms.setMaximumWidth(200)

        second.addWidget(self.atoms)

        self.shells: dict[str, QCheckBox] = {}

        for shell in el.SHELLS:

            box_ = QCheckBox(shell)
            box_.setChecked(True)

            self.shells[shell] = box_
            second.addWidget(box_)

        # What the selection is made of. Independent of the shells
        # above, which is the point: the two together are what lets a
        # single species' d states be asked for.
        second.addWidget(QLabel("of"))

        self.basis = QComboBox()

        for key, label in el.BASES.items():
            self.basis.addItem(label, key)

        self.basis.setToolTip(
            "Element wise sums over every atom of a species. Atom wise "
            "takes the ones you name, which is what to use when two "
            "sites of the same element are not equivalent."
        )
        self.basis.currentIndexChanged.connect(self._basis_changed)

        second.addWidget(self.basis)

        # Splits each shell into its individual orbitals. For d that is
        # the t2g/eg distinction, which summing the shell erases -- and
        # which is the whole content of a crystal-field argument.
        self.resolved = QCheckBox("Resolved")
        self.resolved.setToolTip(
            "One curve per orbital instead of one per shell: d becomes "
            "dxy, dyz, dz2, dxz, dx2, so the t2g and eg sets can be "
            "told apart."
        )

        second.addWidget(self.resolved)

        # Apply it once now. The combo starts on the first entry and
        # setting it there again emits nothing, so without this the
        # window opens element wise with the atom field still live.
        self._basis_changed()

        second.addStretch(1)

        column.addLayout(second)

        # -- how it is drawn ------------------------------------
        third = QHBoxLayout()
        third.setSpacing(10)

        third.addWidget(QLabel("Style"))

        self.style = QComboBox()

        for key, label in electronic_plots.BAND_STYLES.items():
            self.style.addItem(label, key)

        self.style.currentIndexChanged.connect(self._restyle)

        third.addWidget(self.style)

        third.addWidget(QLabel("Weights"))

        self.weight_scale = QComboBox()

        for key, label in electronic_plots.WEIGHT_SCALES.items():
            self.weight_scale.addItem(label, key)

        self.weight_scale.currentIndexChanged.connect(self._restyle)

        third.addWidget(self.weight_scale)

        third.addWidget(QLabel("Window"))

        self.low = QDoubleSpinBox()
        self.low.setRange(-60.0, 0.0)
        self.low.setDecimals(1)
        self.low.setValue(-6.0)
        self.low.setSuffix(" eV")
        self.low.valueChanged.connect(self._restyle)

        third.addWidget(self.low)

        third.addWidget(QLabel("to"))

        self.high = QDoubleSpinBox()
        self.high.setRange(0.0, 60.0)
        self.high.setDecimals(1)
        self.high.setValue(6.0)
        self.high.setSuffix(" eV")
        self.high.valueChanged.connect(self._restyle)

        third.addWidget(self.high)

        third.addStretch(1)

        column.addLayout(third)

        self._build_orbital_views(column)
        self._build_element_row(column)

        self.hint = QLabel("")
        self.hint.setStyleSheet(STATUS_STYLE)
        self.hint.setWordWrap(True)

        column.addWidget(self.hint)

        layout.addWidget(box)

        self._describe()

    def _pick_mode(self, name: str) -> None:

        if name not in self.MODES:
            return

        self.mode = name

        for key, chip in self.chips.items():
            chip.setChecked(key == name)

        self.atoms.setEnabled(name == "atom")

        self._show_orbital_views()
        self._show_elements()

        self._describe()

    def _offer(self, run: el.Run) -> None:

        self._offer_elements(run)

        # Open on whichever source this run can actually answer, rather
        # than on a k-path the run does not have.
        wanted = el.PATH_MODE if run.line_mode else el.SCF_MODE

        found = self.source.findData(wanted)

        if found >= 0:
            self.source.setCurrentIndex(found)

        self._describe()

    def _describe(self) -> None:

        over = {
            "total": "no projection",
            "atom": "one series per atom named",
            "element": "one series per species",
            "orbital": "one series per orbital",
        }.get(self.mode, self.mode)

        self.hint.setText(
            f"Eigenvalues relative to the Fermi level, with {over}. "
            f"A fat band's width is the fraction of that state sitting "
            f"on the series."
        )

    def _restyle(self) -> None:
        """Redraw without recomputing: style is a property of the figure."""

        if self.result is not None:
            self._draw()

    def _run(self) -> None:

        run = self.run

        if run is None:
            _status(self.status, "Load a run first.", theme.WARNING_STYLE)
            return

        shells = [
            shell for shell, box in self.shells.items() if box.isChecked()
        ]

        source = self.source.currentData()

        atoms = self.atoms.text().strip()
        elements = self._chosen_elements()

        # "Plain" is the whole dispersion with nothing projected onto
        # it. "Projected" asks the selection and the shells together --
        # the weighting already intersects the two, so a fat band of one
        # species' d character comes out of the same call.
        projection = (
            "total" if self.mode == "plain" else self.basis.currentData()
        )

        # ORBITAL_VIEWS is a mapping of key to label, not a sequence --
        # indexing it with a position raised KeyError the moment
        # Resolved was ticked, so the band never drew.
        view = (
            el.VIEW_RESOLVED
            if self.resolved.isChecked()
            else self.orbital_view
        )

        def job():

            reader = (
                el.band_structure if source == el.PATH_MODE else el.scf_bands
            )

            return reader(
                run,
                projection=projection,
                atoms=atoms,
                elements=elements,
                shells=shells,
                orbital_view=view,
            )

        self._start(job)

    def _draw(self) -> None:

        electronic_plots.draw_band(
            self.panel.figure,
            self.result,
            title=f"{self.TITLE} -- {self.MODES.get(self.mode, self.mode)}",
            window=(self.low.value(), self.high.value()),
            style=self.style.currentData(),
            weight_scale=self.weight_scale.currentData(),
        )

        self.panel.refresh()

        self.figures.show_figure(reference=self)

    def _shares(self) -> list:
        """How much of the occupied states each series carries."""

        return el.band_contributions(self.result)

    COLUMNS = ("Series", "Mean weight", "Largest weight")

    def _fill(self) -> None:

        result = self.result

        weights = getattr(result, "weights", {}) or {}

        self.table.clear()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setRowCount(len(weights))

        for row, (label, values) in enumerate(weights.items()):

            array = np.asarray(values, dtype=float)

            self._cell(row, 0, label)
            self._cell(
                row, 1, f"{float(array.mean()):.4f}" if array.size else ""
            )
            self._cell(
                row, 2, f"{float(array.max()):.4f}" if array.size else ""
            )

    def _summary(self) -> str:

        result = self.result

        if result is None:
            return ""

        kind = "metal" if result.is_metal else f"gap {result.gap:.3f} eV"

        return f"{kind}. {len(result.weights)} projected series."


def fit_table(table, rows: int, most: int = MAX_TABLE_HEIGHT) -> None:
    """
    Size a table to the rows it actually holds.

    A fixed height is wrong at both ends: mostly empty under two rows,
    and scrolling under twenty. Measuring the table's own header and row
    height is the only thing that is right at either.
    """

    header = table.horizontalHeader().height()

    row_height = table.rowHeight(0) if rows else ROW_HEIGHT

    needed = header + rows * row_height + TABLE_PADDING

    height = min(max(needed, MIN_TABLE_HEIGHT), most)

    table.setMinimumHeight(height)
    table.setMaximumHeight(height)


class ResultsWindow(SideWindow):
    """
    The numbers, in a window of their own.

    Calculating and plotting are two different questions, and a run that
    answers the first by opening the second is answering the wrong one:
    the band centres are the result, and the diagram is a way of looking
    at them. So the table comes up on Calculate, and the figure waits
    until it is asked for.

    Modeless, so it can be left open beside the plot and compared
    against it, or dragged onto a second screen.
    """

    def __init__(self, title: str, parent=None) -> None:

        super().__init__(title, parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        layout.addWidget(self.table, 1)

        self.summary = QLabel("")
        self.summary.setStyleSheet(STATUS_STYLE)
        self.summary.setWordWrap(True)

        layout.addWidget(self.summary)

        buttons = QHBoxLayout()

        buttons.addStretch(1)

        self.save_button = QPushButton("Save as CSV...")
        self.save_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.save_button.clicked.connect(self._save)

        buttons.addWidget(self.save_button)

        close = QPushButton("Close")
        close.clicked.connect(self.hide)

        buttons.addWidget(close)

        layout.addLayout(buttons)

    # ------------------------------------------------------------

    def show_rows(self, columns, rows, summary: str = "") -> None:
        """Put a table of strings on screen."""

        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(list(columns))
        self.table.setRowCount(len(rows))

        for row, values in enumerate(rows):

            for column, text in enumerate(values):

                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                self.table.setItem(row, column, item)

        self.summary.setText(summary)

        self.save_button.setEnabled(bool(rows))

        self._fit_to_rows(len(rows))

    def _fit_to_rows(self, rows: int) -> None:
        """Size the window to what it holds."""

        fit_table(self.table, rows)

        self.resize(
            self.width(),
            min(
                self.sizeHint().height(),
                MAX_TABLE_HEIGHT + WINDOW_FURNITURE,
            ),
        )

    def _save(self) -> None:

        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save table",
            "band_centres.csv",
            "CSV (*.csv)",
        )

        if not chosen:
            return

        headers = [
            self.table.horizontalHeaderItem(column).text()
            for column in range(self.table.columnCount())
        ]

        try:
            with open(chosen, "w", newline="", encoding="utf-8") as handle:

                writer = csv.writer(handle)
                writer.writerow(headers)

                for row in range(self.table.rowCount()):

                    writer.writerow(
                        [
                            (
                                self.table.item(row, column).text()
                                if self.table.item(row, column)
                                else ""
                            )
                            for column in range(self.table.columnCount())
                        ]
                    )

        except OSError as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not write {chosen}: {exc}"
            )
            return

        QMessageBox.information(
            self, "Saved", f"Table written to\n{chosen}"
        )


class BandCentreDialog(ElectronicDialog):
    """
    Norskov band centres, following the reference workflow.

    Pick the energy window, pick the atoms -- either every atom of some
    elements, or a list of indices -- and then, for each element that
    selection contains, tick which bands to take the moment of and
    whether to resolve them into individual orbitals. The configuration
    is shown back before anything is calculated, because a band centre
    is cheap to compute and easy to compute over the wrong thing.

    Two diagrams of the same numbers: the real projected states with
    each centre drawn across them, and a schematic that keeps only the
    energies.
    """

    TITLE = "Band Centre"
    FIGURE = "Band centre"

    #: Every row is one number already; there is nothing to integrate.
    SHOW_CONTRIBUTION = False

    #: The numbers get a window of their own, and it carries the Save
    #: button beside the table it writes.
    SHOWS_FOOTER_SAVE = False

    #: The three shells, as the reference names them.
    BANDS = ("s-band", "p-band", "d-band")

    #: The nine orbitals, in the reference's order.
    ORBITALS = (
        "s",
        "px",
        "py",
        "pz",
        "dxy",
        "dyz",
        "dz2",
        "dxz",
        "dx2-y2",
    )

    ELEMENT_WISE = "Element-wise"
    INDEX_WISE = "Atom Index-wise"

    def _build_controls(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Band Center Analysis")

        outer = QVBoxLayout(box)
        outer.setContentsMargins(6, 0, 6, 6)

        # The controls scroll as one. Fixed heights inside a fixed
        # dialog is what made the layout collapse onto itself when the
        # window was short: a scroll area is the only thing that holds
        # at any size.
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        page = QWidget()

        column = QVBoxLayout(page)
        column.setContentsMargins(4, 4, 4, 4)

        scroller.setWidget(page)

        outer.addWidget(scroller)

        # -- energy window --------------------------------------
        window = QHBoxLayout()
        window.setSpacing(6)

        window.addWidget(QLabel("Energy minimum"))

        self.low = QDoubleSpinBox()
        self.low.setRange(-200.0, 200.0)
        self.low.setDecimals(2)
        self.low.setSingleStep(0.5)
        self.low.setValue(-10.0)
        self.low.setSuffix(" eV")

        window.addWidget(self.low)

        window.addWidget(QLabel("Energy maximum"))

        self.high = QDoubleSpinBox()
        self.high.setRange(-200.0, 200.0)
        self.high.setDecimals(2)
        self.high.setSingleStep(0.5)
        self.high.setValue(0.0)
        self.high.setSuffix(" eV")

        window.addWidget(self.high)

        window.addStretch(1)

        column.addLayout(window)

        # -- how atoms are chosen -------------------------------
        column.addWidget(_heading("Select Atoms"))

        choose = QHBoxLayout()
        choose.setSpacing(6)

        self.selection_mode = QButtonGroup(self)

        self.element_wise = QRadioButton(self.ELEMENT_WISE)
        self.element_wise.setChecked(True)

        self.index_wise = QRadioButton(self.INDEX_WISE)

        for button in (self.element_wise, self.index_wise):
            self.selection_mode.addButton(button)
            choose.addWidget(button)
            button.toggled.connect(self._selection_changed)

        choose.addSpacing(12)

        self.atom_indices = QLineEdit()
        self.atom_indices.setPlaceholderText("Example: 1 2 5 8-12 20")
        self.atom_indices.setToolTip(
            "1-based VASP atom indices. Ranges such as 8-12 are "
            "supported."
        )
        self.atom_indices.setEnabled(False)
        self.atom_indices.editingFinished.connect(self._selection_changed)

        choose.addWidget(self.atom_indices, 1)

        column.addLayout(choose)

        self.elements_row = QWidget()
        self.elements_flow = QGridLayout(self.elements_row)
        self.elements_flow.setContentsMargins(0, 0, 0, 0)
        self.elements_flow.setHorizontalSpacing(14)
        self.elements_flow.setVerticalSpacing(4)

        column.addWidget(self.elements_row)

        self.selection_note = QLabel("")
        self.selection_note.setStyleSheet(STATUS_STYLE)
        self.selection_note.setWordWrap(True)

        column.addWidget(self.selection_note)

        # -- per-element band selection -------------------------
        column.addWidget(_heading("Element-Specific Band Selection"))

        caption = QLabel(
            "Select one or more band contributions for each element "
            "independently."
        )
        caption.setStyleSheet(STATUS_STYLE)

        column.addWidget(caption)

        self.bands_host = QWidget()
        self.bands_layout = QVBoxLayout(self.bands_host)
        self.bands_layout.setContentsMargins(0, 0, 0, 0)
        self.bands_layout.setSpacing(8)
        self.bands_layout.addStretch(1)

        column.addWidget(self.bands_host)

        #: element -> {"bands": {name: box}, "resolved": box,
        #: "orbitals": {name: box}}
        self.per_element: dict[str, dict] = {}

        # -- what will be calculated ----------------------------
        column.addWidget(_heading("Band Center Configuration"))

        self.configuration = QTableWidget(0, 3)
        self.configuration.setHorizontalHeaderLabels(
            ["Element", "Atoms", "Contributions"]
        )
        self.configuration.verticalHeader().setVisible(False)
        self.configuration.setMinimumHeight(CONFIG_HEIGHT)
        self.configuration.setAlternatingRowColors(True)
        self.configuration.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.configuration.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.configuration.verticalHeader().setDefaultSectionSize(28)

        header = self.configuration.horizontalHeader()

        for index in range(3):
            header.setSectionResizeMode(index, QHeaderView.Stretch)

        column.addWidget(self.configuration)

        # -- calculate ------------------------------------------
        buttons = QHBoxLayout()

        buttons.addStretch(1)

        self.run_button = QPushButton("Calculate Band Center")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.clicked.connect(self._run)

        buttons.addWidget(self.run_button)

        column.addLayout(buttons)

        # -- how the answer is drawn ----------------------------
        column.addWidget(_heading("Band Center Plot"))

        drawing = QHBoxLayout()
        drawing.setSpacing(6)

        drawing.addWidget(QLabel("Plotting method"))

        self.plot_type = QComboBox()

        for key, label in electronic_plots.CENTRE_PLOTS.items():
            self.plot_type.addItem(label, key)

        self.plot_type.currentIndexChanged.connect(self._restyle)

        drawing.addWidget(self.plot_type)

        drawing.addWidget(QLabel("Smoothing"))

        self.smooth_window = QSpinBox()
        self.smooth_window.setRange(5, 101)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.setValue(electronic_plots.SMOOTH_WINDOW)
        self.smooth_window.valueChanged.connect(self._restyle)

        drawing.addWidget(self.smooth_window)

        drawing.addWidget(QLabel("Order"))

        self.smooth_order = QSpinBox()
        self.smooth_order.setRange(2, 5)
        self.smooth_order.setValue(electronic_plots.SMOOTH_ORDER)
        self.smooth_order.valueChanged.connect(self._restyle)

        drawing.addWidget(self.smooth_order)

        drawing.addWidget(QLabel("PDOS scale"))

        self.pdos_scale = QDoubleSpinBox()
        self.pdos_scale.setRange(0.1, 2.0)
        self.pdos_scale.setSingleStep(0.1)
        self.pdos_scale.setDecimals(1)
        self.pdos_scale.setValue(electronic_plots.PDOS_SCALE)
        self.pdos_scale.valueChanged.connect(self._restyle)

        drawing.addWidget(self.pdos_scale)

        drawing.addWidget(QLabel("Schematic width"))

        self.band_width = QDoubleSpinBox()
        self.band_width.setRange(0.05, 1.0)
        self.band_width.setSingleStep(0.05)
        self.band_width.setDecimals(2)
        self.band_width.setValue(electronic_plots.SCHEMATIC_WIDTH)
        self.band_width.valueChanged.connect(self._restyle)

        drawing.addWidget(self.band_width)

        drawing.addStretch(1)

        column.addLayout(drawing)

        self.hint = QLabel("")
        self.hint.setStyleSheet(STATUS_STYLE)
        self.hint.setWordWrap(True)

        column.addWidget(self.hint)

        layout.addWidget(box)

        self._describe()

    # ------------------------------------------------------------
    # Choosing atoms
    # ------------------------------------------------------------

    def _offer(self, run: el.Run) -> None:
        """A run arrived: offer its elements to tick."""

        while self.elements_flow.count():

            item = self.elements_flow.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self.element_boxes: dict[str, QCheckBox] = {}

        for symbol in run.elements():

            box = QCheckBox(symbol)
            box.toggled.connect(self._selection_changed)

            self.element_boxes[symbol] = box

            position = len(self.element_boxes) - 1

            self.elements_flow.addWidget(
                box, position // BOXES_PER_ROW, position % BOXES_PER_ROW
            )

        # The window the DOS actually covers, so the default is not a
        # range the run has nothing in.
        self.low.setValue(-10.0)
        self.high.setValue(0.0)

        self._selection_changed()

    def _selection_changed(self, *_args) -> None:
        """Follow the atom selection through to the band blocks."""

        self.atom_indices.setEnabled(self.index_wise.isChecked())

        for symbol, box in getattr(self, "element_boxes", {}).items():
            box.setEnabled(self.element_wise.isChecked())

        self._rebuild_band_blocks()

        self._describe_selection()

        self._refresh_configuration()

    def selected_atoms(self) -> dict[str, list[int]]:
        """
        Which atoms are selected, grouped by element.

        Zero-based indices, as the projection arrays are. The text field
        counts from one, as VASP and every other table here does.
        """

        run = self.run

        if run is None:
            return {}

        if self.element_wise.isChecked():

            chosen = [
                symbol
                for symbol, box in getattr(
                    self, "element_boxes", {}
                ).items()
                if box.isChecked()
            ]

            return {
                symbol: run.indices_of(symbol)
                for symbol in chosen
                if run.indices_of(symbol)
            }

        text = self.atom_indices.text().strip()

        if not text:
            return {}

        try:
            indices = el.parse_indices(text)

        except Exception:
            return {}

        grouped: dict[str, list[int]] = {}

        for index in indices:

            if index < 0 or index >= run.natoms:
                continue

            symbol = run.symbols[index]

            grouped.setdefault(symbol, []).append(index)

        return grouped

    def _describe_selection(self) -> None:

        grouped = self.selected_atoms()

        if not grouped:
            self.selection_note.setText("No atoms selected.")
            return

        total = sum(len(indices) for indices in grouped.values())

        parts = [
            f"{symbol} x{len(indices)}"
            for symbol, indices in grouped.items()
        ]

        self.selection_note.setText(
            f"{total} atom(s) selected:  " + "   ".join(parts)
        )

    # ------------------------------------------------------------
    # Per-element band selection
    # ------------------------------------------------------------

    def _rebuild_band_blocks(self) -> None:
        """
        One block per element in the selection.

        Rebuilt rather than reconciled, but the ticks are carried over:
        changing which atoms are selected should not silently undo the
        bands already chosen for an element that is still there.
        """

        remembered = {
            element: {
                "bands": [
                    name
                    for name, box in widgets["bands"].items()
                    if box.isChecked()
                ],
                "resolved": widgets["resolved"].isChecked(),
                "orbitals": [
                    name
                    for name, box in widgets["orbitals"].items()
                    if box.isChecked()
                ],
            }
            for element, widgets in self.per_element.items()
        }

        while self.bands_layout.count() > 1:

            item = self.bands_layout.takeAt(0)
            widget = item.widget()

            if widget is None:
                continue

            # Unparented before being scheduled for deletion.
            # deleteLater alone only queues the delete, and the widget
            # stays a child of the host until an event loop gets round
            # to it -- so a second rebuild found the old blocks still
            # there and stacked new ones underneath them.
            widget.setParent(None)
            widget.deleteLater()

        self.per_element = {}

        for element in self.selected_atoms():

            was = remembered.get(element, {})

            self.bands_layout.insertWidget(
                self.bands_layout.count() - 1,
                self._band_block(element, was),
            )

    def _band_block(self, element: str, was: dict) -> QWidget:
        """The bands and orbitals offered for one element."""

        host = QGroupBox(element)

        column = QVBoxLayout(host)

        # Top margin left alone. The theme draws a group box's title on
        # the frame itself and reserves room for it; a layout that
        # claims that space back prints the first row through the
        # lettering, which is why the first block appeared to have no
        # element name at all.
        column.setContentsMargins(10, 0, 10, 8)
        column.setSpacing(6)

        bands = QHBoxLayout()
        bands.setSpacing(10)

        bands.addWidget(QLabel("Band contributions"))

        band_boxes: dict[str, QCheckBox] = {}

        for name in self.BANDS:

            box = QCheckBox(name)
            box.setChecked(name in was.get("bands", []))
            box.toggled.connect(self._refresh_configuration)

            band_boxes[name] = box
            bands.addWidget(box)

        bands.addStretch(1)

        column.addLayout(bands)

        resolved = QCheckBox("Enable orbital-resolved analysis")
        resolved.setChecked(bool(was.get("resolved", False)))

        column.addWidget(resolved)

        orbital_row = QWidget()

        orbital_flow = QGridLayout(orbital_row)
        orbital_flow.setContentsMargins(18, 0, 0, 0)
        orbital_flow.setHorizontalSpacing(14)
        orbital_flow.setVerticalSpacing(4)

        orbital_boxes: dict[str, QCheckBox] = {}

        for position, name in enumerate(self.ORBITALS):

            box = QCheckBox(name)
            box.setChecked(name in was.get("orbitals", []))
            box.toggled.connect(self._refresh_configuration)

            orbital_boxes[name] = box

            orbital_flow.addWidget(
                box, position // BOXES_PER_ROW, position % BOXES_PER_ROW
            )

        orbital_row.setVisible(resolved.isChecked())

        resolved.toggled.connect(orbital_row.setVisible)
        resolved.toggled.connect(self._refresh_configuration)

        column.addWidget(orbital_row)

        self.per_element[element] = {
            "bands": band_boxes,
            "resolved": resolved,
            "orbitals": orbital_boxes,
        }

        return host

    def _requests(self) -> list:
        """Every contribution that was ticked, as something to compute."""

        wanted = []

        grouped = self.selected_atoms()

        for element, widgets in self.per_element.items():

            indices = grouped.get(element, [])

            if not indices:
                continue

            names = [
                name
                for name, box in widgets["bands"].items()
                if box.isChecked()
            ]

            if widgets["resolved"].isChecked():

                names += [
                    name
                    for name, box in widgets["orbitals"].items()
                    if box.isChecked()
                ]

            for name in names:

                wanted.append(
                    el.CentreRequest(
                        atoms=el.ATOMS_INDICES,
                        indices=" ".join(
                            str(index + 1) for index in indices
                        ),
                        element=element,
                        orbital=el.orbital_of(name),
                    )
                )

        return wanted

    def _refresh_configuration(self, *_args) -> None:
        """Show back exactly what will be computed, before it is."""

        grouped = self.selected_atoms()

        rows = []

        for element, widgets in self.per_element.items():

            names = [
                name
                for name, box in widgets["bands"].items()
                if box.isChecked()
            ]

            if widgets["resolved"].isChecked():

                names += [
                    name
                    for name, box in widgets["orbitals"].items()
                    if box.isChecked()
                ]

            if names:
                rows.append(
                    (element, len(grouped.get(element, [])), names)
                )

        self.configuration.setRowCount(len(rows))

        for row, (element, count, names) in enumerate(rows):

            self._config_cell(row, 0, element)
            self._config_cell(row, 1, str(count))
            self._config_cell(row, 2, ", ".join(names))

        fit_table(self.configuration, len(rows), most=CONFIG_MOST)

        self._describe()

    def _config_cell(self, row: int, column: int, text: str) -> None:

        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        self.configuration.setItem(row, column, item)

    # ------------------------------------------------------------

    def _describe(self) -> None:

        count = len(self._requests()) if self.per_element else 0

        self.hint.setText(
            f"{count} contribution(s) will be calculated. The Norskov "
            f"band centre is the first moment of the projected states "
            f"over the window, referenced to the Fermi level: "
            f"the integral of E D(E) dE divided by the integral of "
            f"D(E) dE."
        )

    def _restyle(self, *_args) -> None:
        """Redraw with the current plotting settings, never re-running."""

        pdos = (
            self.plot_type.currentData() == electronic_plots.PLOT_PDOS
        )

        for widget in (
            self.smooth_window,
            self.smooth_order,
            self.pdos_scale,
        ):
            widget.setEnabled(pdos)

        self.band_width.setEnabled(not pdos)

        if self.result:
            self._draw()

    def _run(self) -> None:

        run = self.run

        if run is None:
            _status(self.status, "Load a run first.", theme.WARNING_STYLE)
            return

        requests = self._requests()

        if not requests:
            _status(
                self.status,
                "Select atoms and tick at least one band contribution.",
                theme.WARNING_STYLE,
            )
            return

        if self.low.value() >= self.high.value():
            _status(
                self.status,
                "The energy minimum has to be below the maximum.",
                theme.WARNING_STYLE,
            )
            return

        erange = (self.low.value(), self.high.value())

        def job():
            """One ticked contribution, one curve and one centre."""

            return el.centre_curves(run, requests, erange=erange)

        self._start(job)

    def _draw(self) -> None:
        """
        Redraw the figure where it stands, without raising its window.

        Deliberately silent. A plot setting changed is a request to
        change the plot, not to be shown it -- and a window jumping to
        the front on every nudge of a spin box is unusable. Bringing it
        up is what the Show plots button is for.
        """

        if not self.result:
            return

        electronic_plots.draw_band_centre(
            self.panel.figure,
            self.result,
            plot=self.plot_type.currentData(),
            window=(self.low.value(), self.high.value()),
            smooth_window=self.smooth_window.value(),
            smooth_order=self.smooth_order.value(),
            scale=self.pdos_scale.value(),
            band_width=self.band_width.value(),
        )

        self.panel.refresh()

    def results_window(self) -> "ResultsWindow":
        """The window the numbers go in, built the first time it is asked."""

        existing = getattr(self, "_results_window", None)

        if existing is None:
            existing = ResultsWindow("Band Centres", self)
            self._results_window = existing

        return existing

    def _on_result(self, result) -> None:
        """
        A calculation finished: show the numbers.

        The figure is drawn at the same time but left where it is. What
        was asked for was the band centres; the diagram is one way of
        reading them, and it opens when it is asked for.
        """

        self.result = result

        try:
            self._fill()
            self._draw()

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            _status(
                self.status,
                f"{type(exc).__name__}: {exc}",
                theme.ERROR_STYLE,
            )
            return

        curves = result or []

        window = self.results_window()

        window.show_rows(
            self.COLUMNS,
            [
                (
                    curve.element,
                    str(curve.atoms),
                    curve.contribution,
                    (
                        f"{curve.centre:.6f}"
                        if curve.defined
                        else "no states"
                    ),
                )
                for curve in curves
            ],
            summary=self._summary(),
        )

        window.present(self)

        _status(
            self.status,
            f"{len(curves)} contribution(s) calculated.",
            theme.SUCCESS_STYLE,
        )

    #: Exactly what the reference reports.
    COLUMNS = ("Element", "Atoms", "Contribution", "Band Center (eV)")

    def _fill(self) -> None:

        curves = self.result or []

        self.table.clear()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setRowCount(len(curves))

        for row, curve in enumerate(curves):

            self._cell(row, 0, curve.element)
            self._cell(row, 1, str(curve.atoms))
            self._cell(row, 2, curve.contribution)

            if not curve.defined:
                self._cell(row, 3, "no states", dim=True)
                continue

            # Six decimals, as the reference rounds to.
            self._cell(row, 3, f"{curve.centre:.6f}")

        fit_table(self.table, len(curves))

    def _summary(self) -> str:

        curves = self.result or []

        usable = [curve for curve in curves if curve.defined]

        if not usable:
            return "No states in any of these selections."

        parts = [
            f"{curve.element} {curve.contribution} {curve.centre:.3f} eV"
            for curve in usable
        ]

        return "   |   ".join(parts)


def _numbers(text: str) -> list[int]:
    """``1-4, 7`` as a list, or empty for everything."""

    text = text.strip()

    if not text or text.lower() == "all":
        return []

    from qvista.services.orbital import parse_atom_list

    return parse_atom_list(text)

"""
Orbital contributions and wavefunctions for one electronic state.

Point the dialog at a finished VASP run and it resolves a single state --
the valence edge, the conduction edge, or a band you name -- then shows
two things about it side by side:

    left    how much of the state sits on each element and orbital,
            narrowed by clicking an element and then a shell.

    right   the same state as a real-space wavefunction, written out in
            VASP's volumetric format so VESTA can open it.

Both halves describe the one state named in the heading, which is the
reason they share a window rather than being two menu entries.

The run must have been made with ``LORBIT = 11`` for the table and
``LWAVE = .TRUE.`` for the wavefunction.  Neither is VASP's default, so
:mod:`qvista.services.orbital` names the missing tag when it finds the
data absent rather than reporting a parse failure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pymatgen.electronic_structure.core import Spin

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.services import orbital as ob

#: Where the run folder last opened is remembered between sessions.
RUN_KEY = "orbital/run_root"

#: The three sections the menu offers, and the heading each carries.
MODES = {
    "vbm": "VBM",
    "cbm": "CBM",
    "custom": "Custom Band",
}

#: The shells offered above the table, in the order they are shown.
SHELLS = ("all", "s", "p", "d")

#: How finely the columns are broken down. The two menu entries open on
#: one each; the chips switch between them without reloading the run.
VIEWS = ob.VIEWS

DEFAULT_VIEW = ob.VIEW_SPD

#: Shown on the element row before any element has been picked.
ALL_ELEMENTS = "All"

#: Decimal places in the table.  Five matches the reference analyser's
#: per-atom listing, where values run to the fourth decimal.
DECIMALS = 5

#: At or below this a contribution is nothing.  VASP rounds projections
#: to four decimals, so anything under half of that last digit is zero
#: as far as the file is concerned.
EMPTY = 5e-5

#: Drawn in place of a zero, so a sparse row reads as a shape rather
#: than a wall of 0.00000.
EMPTY_MARK = "-"

STATUS_STYLE = f"color: {theme.TEXT_DIM};"


def _status(label: QLabel, text: str, style: str = STATUS_STYLE) -> None:
    """Set the dialog's one line of feedback."""

    label.setStyleSheet(style)
    label.setText(text)


class Chip(QPushButton):
    """
    A small checkable box, used for the element and shell rows.

    Checkable push buttons rather than a combo box because the whole
    point is to see every element at once and click one -- with three
    species that is one click instead of two.
    """

    def __init__(self, text: str, parent=None) -> None:

        super().__init__(text, parent)

        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(52)


class OrbitalDialog(QDialog):
    """The one popup behind VBM, CBM and Custom Band."""

    def __init__(
        self,
        mode: str = "vbm",
        start: str = "",
        view: str = DEFAULT_VIEW,
        parent=None,
    ):

        super().__init__(parent)

        self.view = view if view in VIEWS else DEFAULT_VIEW

        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}")

        self.mode = mode

        # Everything below is filled in by _load().
        self.run = None
        self.poscar = None
        self.state: ob.State | None = None
        self.breakdown: ob.Breakdown | None = None
        self.rows: list[ob.Contribution] = []
        self.element: str | None = None
        self.shell = "all"

        self.setWindowTitle(
            f"Orbital {VIEWS[self.view].split(' - ')[0]} - {MODES[mode]}"
        )
        fit_to_screen(self, 1040, 640)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_source(layout)

        layout.addWidget(_hline())

        panes = QHBoxLayout()
        panes.setSpacing(12)

        panes.addWidget(self._build_table_pane(), 3)
        panes.addWidget(self._build_wave_pane(), 2)

        layout.addLayout(panes, 1)

        self.status = QLabel("Choose the folder holding the VASP run.")
        _status(self.status, self.status.text())

        layout.addWidget(self.status)

        footer = QHBoxLayout()
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        if start:
            self.folder.setText(start)

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_source(self, layout: QVBoxLayout) -> None:
        """The folder row, the state row and the heading."""

        row = QHBoxLayout()

        row.addWidget(QLabel("Run folder"))

        self.folder = QLineEdit()
        self.folder.setPlaceholderText(
            "the directory holding vasprun.xml, POSCAR and WAVECAR"
        )

        row.addWidget(self.folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        layout.addLayout(row)

        # Custom Band names its own state, so the three spinners only
        # appear for that mode.
        self.custom = QWidget()

        custom_row = QHBoxLayout(self.custom)
        custom_row.setContentsMargins(0, 0, 0, 0)

        custom_row.addWidget(QLabel("k-point"))

        self.kpoint = QSpinBox()
        self.kpoint.setRange(1, 100000)

        custom_row.addWidget(self.kpoint)

        custom_row.addWidget(QLabel("Band"))

        self.band = QSpinBox()
        self.band.setRange(1, 100000)

        custom_row.addWidget(self.band)

        custom_row.addWidget(QLabel("Spin"))

        self.spin = QComboBox()
        self.spin.addItems(["up", "down"])

        custom_row.addWidget(self.spin)

        custom_row.addStretch(1)

        layout.addWidget(self.custom)

        self.custom.setVisible(self.mode == "custom")

        # Which state, chosen here rather than by which menu entry was
        # clicked. The three used to be three menu entries; keeping them
        # as chips means comparing the valence and conduction edges is
        # one click instead of closing the window and reopening it.
        load_row = QHBoxLayout()
        load_row.setSpacing(6)

        load_row.addWidget(QLabel("State"))

        self.mode_chips: dict[str, Chip] = {}

        for key, label in MODES.items():

            chip = Chip(label)
            chip.setChecked(key == self.mode)
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_mode(name)
            )

            self.mode_chips[key] = chip

            load_row.addWidget(chip)

        self.load_button = QPushButton(f"Load {MODES[self.mode]}")
        self.load_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.load_button.clicked.connect(self._load)

        load_row.addWidget(self.load_button)
        load_row.addStretch(1)

        layout.addLayout(load_row)

        self.heading = QLabel("")
        self.heading.setStyleSheet(
            f"color: {theme.TEXT}; font-weight: 600;"
        )

        layout.addWidget(self.heading)

    def _build_table_pane(self) -> QGroupBox:
        """Element row, shell row, table and its save button."""

        box = QGroupBox("Orbital contribution")

        layout = QVBoxLayout(box)

        self.element_row = QHBoxLayout()
        self.element_row.setSpacing(6)

        layout.addLayout(self.element_row)

        # ----------------------------------------------------------
        # How finely to break the columns down
        # ----------------------------------------------------------

        view_row = QHBoxLayout()
        view_row.setSpacing(6)

        view_row.addWidget(QLabel("Columns"))

        self.view_chips: dict[str, Chip] = {}

        for key, label in VIEWS.items():

            chip = Chip(label)
            chip.setChecked(key == self.view)
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_view(name)
            )

            self.view_chips[key] = chip

            view_row.addWidget(chip)

        view_row.addStretch(1)

        layout.addLayout(view_row)

        shell_row = QHBoxLayout()
        shell_row.setSpacing(6)

        shell_row.addWidget(QLabel("Orbital"))

        self.shell_chips: dict[str, Chip] = {}

        for name in SHELLS:

            chip = Chip("All" if name == "all" else name)
            chip.setChecked(name == "all")
            chip.clicked.connect(
                lambda _=False, n=name: self._pick_shell(n)
            )

            self.shell_chips[name] = chip

            shell_row.addWidget(chip)

        # A d state has no p character at all, so nine columns of which
        # four are 0.00000 is mostly noise. Dropping the empty ones
        # leaves the orbitals that actually carry the state.
        self.hide_empty = QCheckBox("Hide empty")
        self.hide_empty.setChecked(True)
        self.hide_empty.toggled.connect(lambda _: self._refresh_table())

        shell_row.addWidget(self.hide_empty)

        shell_row.addStretch(1)

        layout.addLayout(shell_row)

        # ----------------------------------------------------------
        # Two tables: the one you narrow, and the whole state at once
        # ----------------------------------------------------------

        self.tabs = QTabWidget()

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.tabs.addTab(self.table, "By element")

        self.contribution = QTableWidget()
        self.contribution.setAlternatingRowColors(True)
        self.contribution.verticalHeader().setVisible(False)
        self.contribution.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.tabs.addTab(self.contribution, "Contribution")

        layout.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self.save_table = QPushButton("Save table...")
        self.save_table.setEnabled(False)
        self.save_table.clicked.connect(self._save_table)

        buttons.addWidget(self.save_table)

        layout.addLayout(buttons)

        return box

    def _build_wave_pane(self) -> QGroupBox:
        """The three wavefunction components and where to write them."""

        box = QGroupBox("Wavefunction")

        layout = QVBoxLayout(box)

        note = QLabel(
            "Written as VASP volumetric files, one per component."
        )
        note.setStyleSheet(STATUS_STYLE)
        note.setWordWrap(True)

        layout.addWidget(note)

        self.want_real_up = QCheckBox("Real psi, spin up")
        self.want_real_down = QCheckBox("Real psi, spin down")
        self.want_square = QCheckBox("psi squared")

        self.want_real_up.setChecked(True)
        self.want_square.setChecked(True)

        for widget in (
            self.want_real_up,
            self.want_real_down,
            self.want_square,
        ):
            layout.addWidget(widget)

        layout.addWidget(_hline())

        layout.addWidget(QLabel("Write into"))

        self.output = QLineEdit("wavefunctions")

        layout.addWidget(self.output)

        self.visualize = QPushButton("Visualize...")
        self.visualize.setEnabled(False)
        self.visualize.clicked.connect(self._visualize)

        layout.addWidget(self.visualize)

        self.write = QPushButton("Write .vasp files")
        self.write.setEnabled(False)
        self.write.clicked.connect(self._write_wavefunctions)

        layout.addWidget(self.write)

        self.written = QLabel("")
        self.written.setStyleSheet(STATUS_STYLE)
        self.written.setWordWrap(True)
        self.written.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        layout.addWidget(self.written)

        layout.addStretch(1)

        return box

    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------

    def _browse(self) -> None:
        """Pick the run folder, starting where one was last opened."""

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.folder.text() or settings.value(
            RUN_KEY, "", type=str
        )

        chosen = QFileDialog.getExistingDirectory(
            self,
            "VASP run folder",
            start,
        )

        if chosen:
            self.folder.setText(chosen)
            settings.setValue(RUN_KEY, chosen)

    def _load(self) -> None:
        """Parse the run and resolve the state this dialog is about."""

        folder = self.folder.text().strip()

        if not folder:
            QMessageBox.information(
                self,
                "No folder",
                "Choose the folder holding the VASP run first.",
            )
            return

        QGuiApplication.setOverrideCursor(Qt.WaitCursor)

        _status(self.status, "Reading vasprun.xml...")

        QApplication.processEvents()

        try:
            # Projections are the slow part of parsing, so they are
            # asked for once, here, rather than on every element click.
            run = ob.load_run(folder, projected=True)

            state = self._resolve(run)

        except ob.OrbitalError as exc:
            QGuiApplication.restoreOverrideCursor()
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.warning(self, MODES[self.mode], str(exc))
            return

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            QGuiApplication.restoreOverrideCursor()
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.critical(self, MODES[self.mode], str(exc))
            return

        finally:
            QGuiApplication.restoreOverrideCursor()

        self.run = run
        self.state = state
        self.poscar = None

        structure = run.final_structure

        self.heading.setText(
            f"{MODES[self.mode]}   {state.label()}   "
            f"E = {state.energy:.6f} eV   "
            f"occupation = {state.occupation:.6f}"
        )

        self._build_element_chips(structure)

        try:
            self.rows = ob.element_table(run, state, structure)

        except ob.OrbitalError as exc:
            self.rows = []
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.save_table.setEnabled(False)
            _status(self.status, str(exc), theme.WARNING_STYLE)

        else:
            self._refresh_table()
            self.save_table.setEnabled(True)
            self._report_occupancy(run)

        # Tick the channel this state is actually in. Opening on the
        # other one draws the same band in a channel the table is not
        # describing, which reads as the picture being wrong.
        in_up = state.spin == Spin.up

        self.want_real_up.setChecked(in_up)
        self.want_real_down.setChecked(not in_up)

        self.write.setEnabled(True)
        self.visualize.setEnabled(True)

    def _resolve(self, run) -> ob.State:
        """The state each mode is about."""

        if self.mode == "custom":

            spin = (
                Spin.up
                if self.spin.currentText() == "up"
                else Spin.down
            )

            return ob.state_at(
                run,
                self.kpoint.value(),
                self.band.value(),
                spin,
            )

        return ob.find_state(run, self.mode)

    def _report_occupancy(self, run) -> None:
        """
        Say when the edges are smeared rather than a real band gap.

        In a metal this is the normal case, and the message is what
        stops the two numbers in the heading being read as a gap.
        """

        smeared = ob.partially_occupied(run)

        if not smeared:
            _status(self.status, "Loaded.")
            return

        _status(
            self.status,
            f"{len(smeared)} partially occupied states: these edges are "
            "the highest occupied and lowest unoccupied states, not a "
            "band gap.",
            theme.WARNING_STYLE,
        )

    # ------------------------------------------------------------
    # The table
    # ------------------------------------------------------------

    def _build_element_chips(self, structure) -> None:
        """One chip per species, plus All, rebuilt for each run."""

        while self.element_row.count():

            item = self.element_row.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

        self.element_row.addWidget(QLabel("Element"))

        self.element = None
        self.element_chips: dict[str, Chip] = {}

        for name in [ALL_ELEMENTS] + ob.elements_of(structure):

            chip = Chip(name)
            chip.setChecked(name == ALL_ELEMENTS)
            chip.clicked.connect(
                lambda _=False, n=name: self._pick_element(n)
            )

            self.element_chips[name] = chip

            self.element_row.addWidget(chip)

        self.element_row.addStretch(1)

    def _pick_element(self, name: str) -> None:
        """Select one element, or All."""

        self.element = None if name == ALL_ELEMENTS else name

        for key, chip in self.element_chips.items():
            chip.setChecked(key == name)

        self._refresh_table()

    def _pick_shell(self, name: str) -> None:
        """Restrict the columns to one shell."""

        self.shell = name

        for key, chip in self.shell_chips.items():
            chip.setChecked(key == name)

        self._refresh_table()

    def _pick_mode(self, name: str) -> None:
        """Switch which state is being described."""

        if name not in MODES:
            return

        self.mode = name

        for key, chip in self.mode_chips.items():
            chip.setChecked(key == name)

        self.custom.setVisible(name == "custom")

        self.load_button.setText(f"Load {MODES[name]}")

        self._retitle()

        # A run already open is re-read for the new state straight away;
        # otherwise wait for the folder.
        if self.run is not None:
            self._load()

    def _retitle(self) -> None:
        """Keep the window title matching the two choices."""

        self.setWindowTitle(
            f"Orbital {VIEWS[self.view].split(' - ')[0]} - "
            f"{MODES[self.mode]}"
        )

    def _pick_view(self, name: str) -> None:
        """Switch between the s/p/d columns and all nine."""

        self.view = name

        for key, chip in self.view_chips.items():
            chip.setChecked(key == name)

        self._retitle()

        self._refresh_table()

    def _labels(self) -> tuple[str, ...]:
        """The column names under the current view."""

        return ob.view_labels(self.view)

    def _as_view(self, values: np.ndarray) -> np.ndarray:
        """
        Nine raw channels, collapsed to whatever the view shows.

        Everything upstream stays in the nine-channel space VASP wrote,
        so switching view is a redraw and never a reload.
        """

        return ob.view_values(values, self.view)

    def _shell_indices(self) -> list[int]:
        """The columns the shell buttons select, before any hiding."""

        labels = self._labels()

        if self.shell == ob.ALL_ORBITALS:
            return list(range(len(labels)))

        # In the s/p/d view a shell is a single column; in the resolved
        # view it is the range of channels that make it up.
        if self.view == ob.VIEW_SPD:
            return [labels.index(self.shell)]

        start, stop = ob.GROUPS[self.shell]

        return list(range(start, stop))

    def _visible(self, matrix: np.ndarray) -> list[int]:
        """
        Which of those columns to draw.

        An empty column is dropped rather than filled with zeros -- but
        never all of them, because a shell that is genuinely empty is
        worth seeing as empty rather than as a missing table.
        """

        indices = self._shell_indices()

        if not self.hide_empty.isChecked() or matrix.size == 0:
            return indices

        carried = [
            index
            for index in indices
            if float(np.abs(matrix[:, index]).max()) > EMPTY
        ]

        return carried or indices

    def _cell_text(self, value: float) -> str:
        """A number, or a mark where there is nothing."""

        if abs(float(value)) <= EMPTY:
            return EMPTY_MARK

        return f"{value:.{DECIMALS}f}"

    def _refresh_table(self) -> None:
        """
        Redraw for the current element and shell.

        With no element chosen the rows are the elements, which is the
        overview the section opens on.  Choosing one swaps in its atoms,
        with the element's own total and percentage beneath them, so the
        answer to "which orbital" is on the same screen as "which atom".
        """

        if not self.rows:
            return

        if self.element is None:
            self._fill_element_rows()
        else:
            self._fill_atom_rows()

        self._fill_contribution()

    def _fill_contribution(self) -> None:
        """
        Every orbital of the state, against every element, as percent.

        Independent of the element and shell chips on purpose: this is
        the tab you look at to find out what the state *is*, before
        narrowing to a part of it. Always all nine channels, plus the
        three shell subtotals, whichever view the other tab is in.
        """

        table = self.contribution

        try:
            self.breakdown = ob.breakdown(self.rows)

        except ob.OrbitalError:
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)
            return

        data = self.breakdown

        headers = [
            "Orbital", "Shell", *data.elements, "Total", "% of state",
        ]

        table.clear()
        table.setRowCount(len(data.orbitals))
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)

        for row, name in enumerate(data.orbitals):

            bold = data.is_subtotal[row]

            self._set(row, 0, name, bold=bold, table=table)
            self._set(row, 1, data.shells[row], bold=bold, table=table)

            for column, value in enumerate(data.values[row], start=2):
                self._set(
                    row,
                    column,
                    self._cell_text(value),
                    bold=bold,
                    table=table,
                )

            self._set(
                row,
                len(headers) - 2,
                f"{float(data.totals[row]):.{DECIMALS}f}",
                bold=bold,
                table=table,
            )

            self._set(
                row,
                len(headers) - 1,
                f"{float(data.percent[row]):.2f}%",
                bold=bold,
                table=table,
            )

    def _fill_element_rows(self) -> None:
        """One row per element."""

        matrix = self._as_view(
            np.vstack([entry.totals for entry in self.rows])
        )

        shell = self._shell_indices()
        indices = self._visible(matrix)

        labels = self._labels()

        columns = [labels[index] for index in indices]

        self.table.clear()
        self.table.setRowCount(len(self.rows))
        self.table.setColumnCount(len(columns) + 2)
        self.table.setHorizontalHeaderLabels(
            ["Element", *columns, "Total"]
        )

        for row, entry in enumerate(self.rows):

            values = matrix[row]

            self._set(row, 0, entry.label)

            for column, index in enumerate(indices, start=1):
                self._set(row, column, self._cell_text(values[index]))

            # Summed over the whole shell, not the shown columns, so
            # hiding an empty one never moves the number beside it.
            self._set(
                row,
                len(columns) + 1,
                f"{float(values[shell].sum()):.{DECIMALS}f}",
            )

    def _fill_atom_rows(self) -> None:
        """One row per atom of the chosen element, then two summaries."""

        entry = next(
            (row for row in self.rows if row.label == self.element),
            None,
        )

        if entry is None:
            return

        per_atom = self._as_view(entry.per_atom)
        totals = self._as_view(entry.totals)

        shell = self._shell_indices()
        indices = self._visible(per_atom)

        labels = self._labels()

        columns = [labels[index] for index in indices]

        self.table.clear()
        self.table.setRowCount(len(entry.atoms) + 2)
        self.table.setColumnCount(len(columns) + 2)
        self.table.setHorizontalHeaderLabels(["Atom", *columns, "Total"])

        for row, atom in enumerate(entry.atoms):

            values = per_atom[row]

            self._set(row, 0, f"{atom + 1}  {entry.label}")

            for column, index in enumerate(indices, start=1):
                self._set(row, column, self._cell_text(values[index]))

            self._set(
                row,
                len(columns) + 1,
                f"{float(values[shell].sum()):.{DECIMALS}f}",
            )

        summary = len(entry.atoms)

        self._set(summary, 0, "Total", bold=True)

        for column, index in enumerate(indices, start=1):
            self._set(
                summary,
                column,
                self._cell_text(totals[index]),
                bold=True,
            )

        self._set(
            summary,
            len(columns) + 1,
            f"{float(totals[shell].sum()):.{DECIMALS}f}",
            bold=True,
        )

        # Percentages are within the shell, which is what makes "98.82%
        # dz2" meaningful -- it is 98.82% of the d character, not of
        # everything the atom does. Taken over the whole shell so that
        # hiding an empty column cannot change them.
        percent = ob.percentages(totals[shell])

        place = {index: n for n, index in enumerate(shell)}

        self._set(summary + 1, 0, "Percent")

        for column, index in enumerate(indices, start=1):
            self._set(
                summary + 1,
                column,
                f"{percent[place[index]]:.2f}%",
            )

        self._set(summary + 1, len(columns) + 1, "100.00%")

    def _set(
        self,
        row: int,
        column: int,
        text: str,
        bold: bool = False,
        table: QTableWidget | None = None,
    ) -> None:
        """Place one read-only cell, in the main table unless told."""

        target = self.table if table is None else table

        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        # The Contribution tab has a text column in slot 1 as well.
        numeric = column > 0 and not (
            target is getattr(self, "contribution", None) and column == 1
        )

        if numeric:
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)

        target.setItem(row, column, item)

    def _save_table(self) -> None:
        """Write what is on screen as CSV."""

        if self.state is None:
            return

        suggested = (
            f"orbital_{self.mode}_{self.state.filename_stem()}.csv"
        )

        start = str(Path(self.folder.text().strip()) / suggested)

        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save table",
            start,
            "CSV (*.csv)",
        )

        if not chosen:
            return

        showing = self.tabs.currentWidget()

        headers = [
            showing.horizontalHeaderItem(column).text()
            for column in range(showing.columnCount())
        ]

        try:
            with open(chosen, "w", newline="", encoding="utf-8") as handle:

                writer = csv.writer(handle)

                writer.writerow([f"# {MODES[self.mode]}"])
                writer.writerow([f"# {self.state.label()}"])
                writer.writerow(
                    [f"# energy {self.state.energy:.8f} eV"]
                )
                writer.writerow(
                    [f"# occupation {self.state.occupation:.8f}"]
                )
                writer.writerow(headers)

                for row in range(showing.rowCount()):
                    writer.writerow(
                        [
                            (
                                showing.item(row, column).text()
                                if showing.item(row, column)
                                else ""
                            )
                            for column in range(showing.columnCount())
                        ]
                    )

        except OSError as exc:
            QMessageBox.warning(self, "Save table", str(exc))
            return

        _status(self.status, f"Saved {Path(chosen).name}")

    # ------------------------------------------------------------
    # Wavefunctions
    # ------------------------------------------------------------

    def _wanted(self) -> list[tuple[str, Spin | None]]:
        """The ticked components, as (part, spin) with None meaning
        whichever spin the loaded state is in."""

        wanted: list[tuple[str, Spin | None]] = []

        if self.want_real_up.isChecked():
            wanted.append(("real", Spin.up))

        if self.want_real_down.isChecked():
            wanted.append(("real", Spin.down))

        if self.want_square.isChecked():
            wanted.append(("square", None))

        # Whatever is in the state's own channel comes first, so the
        # viewer opens on exactly what the table describes. Holding the
        # band fixed is not enough on its own: the other channel is a
        # different wavefunction of the same band, and opening on it
        # reads as the picture disagreeing with the table.
        if self.state is not None:
            wanted.sort(
                key=lambda item: (
                    item[1] is not None and item[1] != self.state.spin
                )
            )

        return wanted

    def _visualize(self) -> None:
        """
        Open the ticked components as isosurfaces over the structure.

        Nothing is written to disk: the grids are contoured straight
        from the WAVECAR, so a level can be tried out before committing
        a few hundred megabytes of ``.vasp`` file.
        """

        from qvista.gui.dialogs.wavefunction_viewer import (
            WavefunctionViewer,
        )

        if self.run is None or self.state is None:
            return

        wanted = self._wanted()

        if not wanted:
            QMessageBox.information(
                self,
                "Wavefunction",
                "Tick at least one component to visualise.",
            )
            return

        folder = Path(self.folder.text().strip())

        QGuiApplication.setOverrideCursor(Qt.WaitCursor)

        _status(self.status, "Building the grids...")

        QApplication.processEvents()

        try:
            if self.poscar is None:
                self.poscar = ob.load_poscar(folder)

            wavecar = ob.load_wavecar(folder)

            components = self._grids(wanted, wavecar)

            note = ob.wavefunction_note(wavecar, self.state)

        except ob.OrbitalError as exc:
            QGuiApplication.restoreOverrideCursor()
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.warning(self, "Wavefunction", str(exc))
            return

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            QGuiApplication.restoreOverrideCursor()
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.critical(self, "Wavefunction", str(exc))
            return

        finally:
            QGuiApplication.restoreOverrideCursor()

        _status(self.status, "Showing the wavefunction.")

        # The element the table is narrowed to is the one the question
        # is about, so the picture opens narrowed to it as well. Drawing
        # all 42 atoms at once is how a state that is 78% dz2 ends up
        # looking like nothing in particular.
        viewer = WavefunctionViewer(
            components,
            self.poscar.structure,
            heading=f"{MODES[self.mode]}   {self.state.label()}",
            isolate=self.element,
            note=note,
            parent=self,
        )

        viewer.show()

    def _grids(
        self,
        wanted: list[tuple[str, Spin | None]],
        wavecar,
    ) -> dict[str, tuple[np.ndarray, bool]]:
        """
        One grid per ticked component, labelled for the viewer's list.

        ``signed`` is False only for the density, which is what tells
        the viewer to draw one surface instead of a pair.
        """

        components: dict[str, tuple[np.ndarray, bool]] = {}

        for part, spin in wanted:

            state = self._state_for(part, spin)

            data = ob.wavefunction_data(
                wavecar,
                self.poscar,
                state,
                part,
            )

            # The state goes in the label, not just the spin. With two
            # spin channels on screen at once, "which one am I looking
            # at" has to be answerable without going back to the table.
            label = (
                f"{ob.WAVE_PARTS[part]}  "
                f"({ob.spin_name(state.spin).lower()})  "
                f"k={state.kpoint} band={state.band}"
            )

            components[label] = (data, part != "square")

        return components

    def _state_for(self, part: str, spin: Spin | None) -> ob.State:
        """
        The state a component belongs to.

        Always the band and k-point in the heading, with only the spin
        channel varying.

        Resolving the edge again inside each spin channel is defensible
        in the abstract -- the spin-up VBM really is a different band
        from the spin-down one -- and it is what the reference analyser
        does when asked for both spins at once. But it breaks the one
        promise this window makes: that the table on the left and the
        surface on the right are the same state. A table reading 78%
        dz2 beside a picture of some other band is worse than no
        picture, so the heading wins and the band is held fixed.
        """

        if spin is None or spin == self.state.spin:
            return self.state

        return ob.state_at(
            self.run,
            self.state.kpoint,
            self.state.band,
            spin,
        )

    def _write_wavefunctions(self) -> None:
        """
        Write each ticked component as a ``.vasp`` volumetric file.

        The spin-up and spin-down boxes each resolve the section's state
        within that channel -- the spin-up VBM is not generally the same
        band as the spin-down VBM -- which is what the reference
        analyser does when asked for both spins at once.
        """

        if self.run is None or self.state is None:
            return

        wanted = self._wanted()

        if not wanted:
            QMessageBox.information(
                self,
                "Wavefunction",
                "Tick at least one component to write.",
            )
            return

        folder = Path(self.folder.text().strip())

        QGuiApplication.setOverrideCursor(Qt.WaitCursor)

        _status(self.status, "Reading WAVECAR...")

        QApplication.processEvents()

        try:
            if self.poscar is None:
                self.poscar = ob.load_poscar(folder)

            wavecar = ob.load_wavecar(folder)

            written = self._write_each(wanted, wavecar, folder)

        except ob.OrbitalError as exc:
            QGuiApplication.restoreOverrideCursor()
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.warning(self, "Wavefunction", str(exc))
            return

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            QGuiApplication.restoreOverrideCursor()
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.critical(self, "Wavefunction", str(exc))
            return

        finally:
            QGuiApplication.restoreOverrideCursor()

        self.written.setText("\n".join(written))

        _status(self.status, f"Wrote {len(written)} file(s).")

    def _write_each(
        self,
        wanted: list[tuple[str, Spin | None]],
        wavecar,
        folder: Path,
    ) -> list[str]:
        """Resolve a state per component and write it."""

        target = folder / (self.output.text().strip() or "wavefunctions")

        written: list[str] = []

        for part, spin in wanted:

            state = self._state_for(part, spin)

            name = ob.wavefunction_filename(part, state)

            ob.write_wavefunction(
                wavecar,
                self.poscar,
                state,
                part,
                target / name,
            )

            written.append(name)

        return written

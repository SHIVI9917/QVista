"""
DDEC6 charges and bond orders from a Chargemol run.

Four views of the same output: per atom, per element, per bond, and per
kind of bond.  The bond views carry the bond length, which Chargemol
does not print -- it names the periodic image instead, and
:mod:`qvista.services.ddec` turns that back into a distance.

Chargemol is a separate program and is not shipped with QVista.  If it
is installed, point the two boxes at it and this dialog will write the
``job_control.txt`` and run it; otherwise it reads output that is
already in the folder.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.bader_dialog import Chip
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.services import ddec as dd

RUN_KEY = "ddec/run_root"

#: The four groupings, and the chip each one gets.
SCOPES: dict[str, str] = {
    "atoms": "Atoms",
    "elements": "Elements",
    "bonds": "Bonds",
    "types": "Bond types",
}

DECIMALS = 4

STATUS_STYLE = f"color: {theme.TEXT_DIM};"


def _status(label: QLabel, text: str, style: str = STATUS_STYLE) -> None:

    label.setStyleSheet(style)
    label.setText(text)


class Cell(QTableWidgetItem):
    """
    A cell that sorts by its number rather than by its text.

    Qt compares the displayed string, under which 10 comes before 9 and
    -0.5 before -1.2, so a column of bond lengths sorted by clicking its
    header would come out in an order that means nothing.
    """

    def __lt__(self, other) -> bool:

        mine = self.data(Qt.UserRole)
        theirs = (
            other.data(Qt.UserRole)
            if isinstance(other, QTableWidgetItem)
            else None
        )

        if mine is None or theirs is None:
            return super().__lt__(other)

        return float(mine) < float(theirs)


class ChargemolRun(QThread):
    """Chargemol, off the GUI thread. It takes minutes."""

    progress = Signal(str, int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        folder: str,
        executable: str,
        densities: str,
        net_charge: float,
        parent=None,
    ) -> None:

        super().__init__(parent)

        self.folder = folder
        self.executable = executable
        self.densities = densities
        self.net_charge = net_charge

    def run(self) -> None:

        try:
            result = dd.run(
                self.folder,
                self.executable,
                self.densities,
                net_charge=self.net_charge,
                progress=self.progress.emit,
            )

        except dd.DdecError as exc:
            self.failed.emit(str(exc))
            return

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        self.completed.emit(result)


class DdecDialog(QDialog):
    """The window behind Analysis > Bonding Tools > DDEC6."""

    def __init__(self, start: str = "", parent=None):

        super().__init__(parent)

        self.result: dd.DdecResult | None = None
        self.scope = "atoms"
        self.worker: ChargemolRun | None = None

        self.setWindowTitle("DDEC6")
        fit_to_screen(self, 1100, 720)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._build_source(layout)

        layout.addWidget(_hline())

        layout.addWidget(self._build_results(), 1)

        self.status = QLabel("Choose the folder holding the Chargemol run.")
        _status(self.status, self.status.text())
        self.status.setWordWrap(True)

        layout.addWidget(self.status)

        footer = QHBoxLayout()
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        if start:
            self.folder.setText(start)
            self._look()

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_source(self, layout: QVBoxLayout) -> None:

        row = QHBoxLayout()

        row.addWidget(QLabel("Folder"))

        self.folder = QLineEdit()
        self.folder.setPlaceholderText(
            "the directory holding the DDEC6 .xyz output"
        )
        self.folder.editingFinished.connect(self._look)

        row.addWidget(self.folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        self.load_button = QPushButton("Load")
        self.load_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.load_button.clicked.connect(self._load)

        row.addWidget(self.load_button)

        layout.addLayout(row)

        layout.addWidget(self._build_runner())

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)

        layout.addWidget(self.progress)

    def _build_runner(self) -> QWidget:
        """
        Where Chargemol is, for folders that have no output yet.

        Hidden once output is found, since by then it is answered.
        """

        self.runner = QGroupBox("Run Chargemol")

        settings = QSettings(ORGANIZATION, APP_NAME)

        outer = QVBoxLayout(self.runner)

        row = QHBoxLayout()

        row.addWidget(QLabel("Program"))

        self.executable = QLineEdit(
            settings.value(dd.EXE_KEY, "", type=str)
            or (dd.find_executable() or "")
        )
        self.executable.setPlaceholderText("path to the Chargemol binary")

        row.addWidget(self.executable, 1)

        pick_exe = QPushButton("...")
        pick_exe.setMaximumWidth(34)
        pick_exe.clicked.connect(self._pick_executable)

        row.addWidget(pick_exe)

        row.addWidget(QLabel("Atomic densities"))

        self.densities = QLineEdit(
            settings.value(dd.DENSITIES_KEY, "", type=str)
        )
        self.densities.setPlaceholderText(
            "the atomic_densities folder that ships with Chargemol"
        )

        row.addWidget(self.densities, 1)

        pick_dir = QPushButton("...")
        pick_dir.setMaximumWidth(34)
        pick_dir.clicked.connect(self._pick_densities)

        row.addWidget(pick_dir)

        row.addWidget(QLabel("Net charge"))

        self.net_charge = QDoubleSpinBox()
        self.net_charge.setRange(-20.0, 20.0)
        self.net_charge.setSingleStep(1.0)
        self.net_charge.setDecimals(2)

        row.addWidget(self.net_charge)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._run)

        row.addWidget(self.run_button)

        outer.addLayout(row)

        return self.runner

    def _build_results(self) -> QGroupBox:

        box = QGroupBox("DDEC6")

        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel("Show"))

        self.scope_chips: dict[str, Chip] = {}

        for key, text in SCOPES.items():

            chip = Chip(text)
            chip.setChecked(key == "atoms")
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_scope(name)
            )

            self.scope_chips[key] = chip

            row.addWidget(chip)

        self.order_label = QLabel("order >=")

        row.addWidget(self.order_label)

        self.minimum = QDoubleSpinBox()
        self.minimum.setRange(0.0, 3.0)
        self.minimum.setSingleStep(0.05)
        self.minimum.setDecimals(3)
        self.minimum.setValue(dd.DEFAULT_MIN_ORDER)
        self.minimum.setToolTip(
            "Chargemol prints every bond order down to 0.001; most of "
            "those are the tail of the exchange hole, not a bond."
        )
        self.minimum.valueChanged.connect(lambda _: self._refresh_table())

        row.addWidget(self.minimum)

        # Chargemol lists each bond twice, once from each end. Showing it
        # that way is occasionally what you want -- it is how you read
        # one atom's whole coordination off the table.
        self.both_ends = QCheckBox("Both ends")
        self.both_ends.setToolTip(
            "List each bond once from each atom, as Chargemol prints it."
        )
        self.both_ends.toggled.connect(lambda _: self._refresh_table())

        row.addWidget(self.both_ends)

        row.addStretch(1)

        self.summary = QLabel("")
        self.summary.setStyleSheet(STATUS_STYLE)

        row.addWidget(self.summary)

        layout.addLayout(row)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.table.setSortingEnabled(True)

        layout.addWidget(self.table, 1)

        self.notes = QLabel("")
        self.notes.setStyleSheet(theme.WARNING_STYLE)
        self.notes.setWordWrap(True)
        self.notes.setAlignment(Qt.AlignTop)

        self.notes_area = QScrollArea()
        self.notes_area.setWidgetResizable(True)
        self.notes_area.setWidget(self.notes)
        self.notes_area.setMaximumHeight(70)
        self.notes_area.setFrameShape(QFrame.NoFrame)
        self.notes_area.setVisible(False)

        layout.addWidget(self.notes_area)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self.save_button = QPushButton("Save table...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_table)

        buttons.addWidget(self.save_button)

        self.visualize = QPushButton("Visualization...")
        self.visualize.setEnabled(False)
        self.visualize.clicked.connect(self._visualize)

        buttons.addWidget(self.visualize)

        layout.addLayout(buttons)

        self._show_bond_controls(False)

        return box

    # ------------------------------------------------------------
    # Source
    # ------------------------------------------------------------

    def _browse(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.folder.text() or settings.value(RUN_KEY, "", type=str)

        chosen = QFileDialog.getExistingDirectory(
            self, "Chargemol run folder", start
        )

        if chosen:
            self.folder.setText(chosen)
            settings.setValue(RUN_KEY, chosen)
            self._look()

    def _pick_executable(self) -> None:

        chosen, _ = QFileDialog.getOpenFileName(
            self, "Chargemol program", self.executable.text()
        )

        if chosen:
            self.executable.setText(chosen)
            QSettings(ORGANIZATION, APP_NAME).setValue(dd.EXE_KEY, chosen)

    def _pick_densities(self) -> None:

        chosen = QFileDialog.getExistingDirectory(
            self, "Atomic densities folder", self.densities.text()
        )

        if chosen:
            self.densities.setText(chosen)
            QSettings(ORGANIZATION, APP_NAME).setValue(
                dd.DENSITIES_KEY, chosen
            )

    def _look(self) -> None:
        """Say what is in the folder, and offer the runner only if needed."""

        folder = self.folder.text().strip()

        if not folder or not Path(folder).is_dir():
            self.runner.setVisible(True)
            _status(self.status, "That folder does not exist.", theme.WARNING_STYLE)
            return

        found = dd.available(folder)

        self.runner.setVisible("charges" not in found)

        if "charges" not in found:
            _status(
                self.status,
                "No DDEC6 output here. Run Chargemol, or pick a folder "
                "that already has its .xyz files.",
                theme.WARNING_STYLE,
            )
            return

        _status(
            self.status,
            "Found " + ", ".join(sorted(path.name for path in found.values())),
        )

    # ------------------------------------------------------------
    # Loading and running
    # ------------------------------------------------------------

    def _load(self) -> None:
        """Read output that is already there."""

        folder = self.folder.text().strip()

        if not folder:
            QMessageBox.information(
                self, "No folder", "Choose the folder first."
            )
            return

        try:
            result = dd.read(folder)

        except dd.DdecError as exc:
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.warning(self, "DDEC6", str(exc))
            return

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.critical(self, "DDEC6", str(exc))
            return

        self._accept(result)

    def _run(self) -> None:
        """Write job_control.txt and run Chargemol."""

        folder = self.folder.text().strip()
        executable = self.executable.text().strip()
        densities = self.densities.text().strip()

        if not folder:
            QMessageBox.information(
                self, "No folder", "Choose the folder first."
            )
            return

        if not executable or not densities:
            QMessageBox.information(
                self,
                "Chargemol",
                "Chargemol is a separate program. Point the two boxes at "
                "the binary and at its atomic_densities folder.",
            )
            return

        if self.worker is not None and self.worker.isRunning():
            return

        settings = QSettings(ORGANIZATION, APP_NAME)
        settings.setValue(dd.EXE_KEY, executable)
        settings.setValue(dd.DENSITIES_KEY, densities)

        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)

        self.progress.setValue(0)
        self.progress.setVisible(True)

        self.worker = ChargemolRun(
            folder,
            executable,
            densities,
            self.net_charge.value(),
            parent=self,
        )

        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._accept)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.worker.start()

    def _on_progress(self, message: str, percent: int) -> None:

        _status(self.status, message)

        if percent >= 0:
            self.progress.setValue(percent)

    def _on_failed(self, message: str) -> None:

        _status(self.status, message, theme.ERROR_STYLE)

        QMessageBox.warning(self, "Chargemol", message)

    def _on_finished(self) -> None:

        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.load_button.setEnabled(True)

        self.worker = None

    def _accept(self, result: dd.DdecResult) -> None:

        self.result = result

        self.runner.setVisible(False)

        spin = any(atom.spin is not None for atom in result.atoms)

        self.summary.setText(
            f"{result.method}   {len(result.atoms)} atoms   "
            f"{len(result.bonds)} printed bonds"
            + ("   spin polarised" if spin else "")
            + ("" if result.periodic else "   non-periodic")
        )

        self.notes.setText("\n".join(f"- {note}" for note in result.notes))
        self.notes_area.setVisible(bool(result.notes))

        self._refresh_table()

        self.save_button.setEnabled(True)
        self.visualize.setEnabled(True)

        _status(
            self.status,
            f"Loaded. Total charge {result.total_charge:+.4f}.",
            theme.SUCCESS_STYLE,
        )

    # ------------------------------------------------------------
    # The table
    # ------------------------------------------------------------

    def _pick_scope(self, name: str) -> None:

        self.scope = name

        for key, chip in self.scope_chips.items():
            chip.setChecked(key == name)

        self._show_bond_controls(name in ("bonds", "types"))

        self._refresh_table()

    def _show_bond_controls(self, shown: bool) -> None:

        self.order_label.setVisible(shown)
        self.minimum.setVisible(shown)
        self.both_ends.setVisible(shown and self.scope == "bonds")

    def _bonds(self) -> list[dd.Bond]:
        """The bonds the current settings ask for."""

        bonds = dd.bonds_above(self.result, self.minimum.value())

        if self.scope == "bonds" and self.both_ends.isChecked():
            return bonds

        return dd.unique_bonds(bonds)

    def _refresh_table(self) -> None:

        if self.result is None:
            return

        # Sorting has to be off while the rows go in, or Qt reorders them
        # underneath the loop and the cells land in the wrong rows.
        self.table.setSortingEnabled(False)

        if self.scope == "atoms":
            self._fill_atoms()

        elif self.scope == "elements":
            self._fill_elements()

        elif self.scope == "bonds":
            self._fill_bonds()

        else:
            self._fill_types()

        self.table.setSortingEnabled(True)

    def _fill_atoms(self) -> None:

        atoms = self.result.atoms

        # Only the columns this run actually has: a spin column full of
        # dashes for a non-magnetic run is noise.
        optional = [
            ("Spin", "spin"),
            ("Dipole", "dipole"),
            ("SBO", "sbo"),
            ("CM5", "cm5"),
        ]

        extra = [
            (title, name)
            for title, name in optional
            if any(getattr(atom, name) is not None for atom in atoms)
        ]

        columns = ["Atom", "X", "Y", "Z", "Net charge"] + [
            title for title, _ in extra
        ]

        self._start(columns, len(atoms))

        for row, atom in enumerate(atoms):

            self._set(row, 0, atom.label(), sort=atom.index)

            for offset, value in enumerate(atom.coords):
                self._set(row, 1 + offset, f"{float(value):.4f}", sort=value)

            self._set(row, 4, f"{atom.charge:+.{DECIMALS}f}", sort=atom.charge)

            for offset, (_, name) in enumerate(extra):

                value = getattr(atom, name)

                self._set(
                    row,
                    5 + offset,
                    "-" if value is None else f"{value:+.{DECIMALS}f}",
                    sort=value,
                )

    def _fill_elements(self) -> None:

        rows = dd.by_element(self.result)

        columns = ["Element", "Atoms", "Net charge", "Charge/atom", "Spin", "SBO"]

        self._start(columns, len(rows))

        for row, entry in enumerate(rows):

            self._set(row, 0, entry.label)
            self._set(row, 1, str(entry.count), sort=entry.count)
            self._set(
                row, 2, f"{entry.charge:+.{DECIMALS}f}", sort=entry.charge
            )
            self._set(
                row,
                3,
                f"{entry.charge_per_atom:+.{DECIMALS}f}",
                sort=entry.charge_per_atom,
            )
            self._set(
                row,
                4,
                "-" if entry.spin is None else f"{entry.spin:+.{DECIMALS}f}",
                sort=entry.spin,
            )
            self._set(
                row,
                5,
                "-" if entry.sbo is None else f"{entry.sbo:.{DECIMALS}f}",
                sort=entry.sbo,
            )

    def _fill_bonds(self) -> None:

        bonds = self._bonds()

        columns = ["Bond", "Pair", "Order", "Length", "Image", "Spin pol."]

        self._start(columns, len(bonds))

        for row, bond in enumerate(bonds):

            self._set(row, 0, f"{bond.a} {bond.a_symbol} - {bond.b} {bond.b_symbol}")
            self._set(row, 1, bond.pair)
            self._set(row, 2, f"{bond.order:.{DECIMALS}f}", sort=bond.order)
            self._set(row, 3, f"{bond.length:.{DECIMALS}f}", sort=bond.length)
            self._set(row, 4, "{},{},{}".format(*bond.image))
            self._set(
                row,
                5,
                f"{bond.spin_polarization:+.{DECIMALS}f}",
                sort=bond.spin_polarization,
            )

    def _fill_types(self) -> None:

        rows = dd.by_bond_type(self._bonds())

        columns = [
            "Pair",
            "Count",
            "Order",
            "Order min",
            "Order max",
            "Length",
            "Length min",
            "Length max",
        ]

        self._start(columns, len(rows))

        for row, entry in enumerate(rows):

            self._set(row, 0, entry.pair)
            self._set(row, 1, str(entry.count), sort=entry.count)
            self._set(row, 2, f"{entry.order_mean:.{DECIMALS}f}", sort=entry.order_mean)
            self._set(row, 3, f"{entry.order_min:.{DECIMALS}f}", sort=entry.order_min)
            self._set(row, 4, f"{entry.order_max:.{DECIMALS}f}", sort=entry.order_max)
            self._set(row, 5, f"{entry.length_mean:.{DECIMALS}f}", sort=entry.length_mean)
            self._set(row, 6, f"{entry.length_min:.{DECIMALS}f}", sort=entry.length_min)
            self._set(row, 7, f"{entry.length_max:.{DECIMALS}f}", sort=entry.length_max)

    def _start(self, columns: list[str], rows: int) -> None:

        self.table.clear()
        self.table.setRowCount(rows)
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

    def _set(
        self,
        row: int,
        column: int,
        text: str,
        sort: float | None = None,
    ) -> None:
        """
        Place one read-only cell.

        ``sort`` carries the number behind the text, so clicking a header
        orders by value rather than by the string -- without it "10" sorts
        before "9" and a bond length column is useless.
        """

        item = Cell()
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setData(Qt.DisplayRole, text)

        if sort is not None:
            item.setData(Qt.UserRole, float(sort))

        if column > 0:
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.table.setItem(row, column, item)

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    def _save_table(self) -> None:

        if self.result is None:
            return

        start = str(
            Path(self.folder.text().strip()) / f"ddec6_{self.scope}.csv"
        )

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save table", start, "CSV (*.csv)"
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

                writer.writerow([f"# {self.result.method} {self.scope}"])
                writer.writerow([f"# folder {self.result.folder}"])

                if self.scope in ("bonds", "types"):
                    writer.writerow(
                        [f"# bond order cutoff {self.minimum.value():.3f}"]
                    )

                for note in self.result.notes:
                    writer.writerow([f"# {note}"])

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
            QMessageBox.warning(self, "Save table", str(exc))
            return

        _status(self.status, f"Saved {Path(chosen).name}")

    def _visualize(self) -> None:

        from qvista.gui.dialogs.ddec_viewer import DdecViewer

        if self.result is None:
            return

        try:
            viewer = DdecViewer(
                self.result,
                heading=(
                    f"{self.result.method}   {len(self.result.atoms)} atoms"
                ),
                parent=self,
            )

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.critical(self, "Visualization", str(exc))
            return

        viewer.show()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name

        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(15000)

        super().closeEvent(event)

"""
Bonding analysis of a LOBSTER run.

Three views of the same interaction list -- one row per bond, per kind of
bond, or per atom -- with the bond length beside the bond strength, since
neither means much without the other.  Selecting rows and pressing Plot
draws their curves.

COHP, COOP and COBI are all read; which one is loaded only changes what
the strength column means, and :mod:`qvista.services.cohp` keeps the sign
conventions straight so a positive number is always a bonding one.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from PySide6.QtCore import QSettings, Qt
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
    QLineEdit,
    QMessageBox,
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
from qvista.gui.dialogs.ddec_dialog import Cell
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.services import cohp as ch

RUN_KEY = "cohp/run_root"

SCOPES: dict[str, str] = {
    "bonds": "Bonds",
    "pairs": "Bond types",
    "atoms": "Atoms",
}

DECIMALS = 4

STATUS_STYLE = f"color: {theme.TEXT_DIM};"


def _status(label: QLabel, text: str, style: str = STATUS_STYLE) -> None:

    label.setStyleSheet(style)
    label.setText(text)


class CohpDialog(QDialog):
    """The window behind Analysis > Bonding Tools > COHP."""

    def __init__(self, start: str = "", parent=None):

        super().__init__(parent)

        self.result: ch.CohpResult | None = None
        self.scope = "bonds"
        self.rows: list = []

        self.setWindowTitle("COHP")
        fit_to_screen(self, 1140, 760)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._build_source(layout)

        layout.addWidget(_hline())

        layout.addWidget(self._build_results(), 1)

        self.status = QLabel("Choose the folder holding the LOBSTER output.")
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
            "the directory holding ICOHPLIST.lobster and COHPCAR.lobster"
        )
        self.folder.editingFinished.connect(self._look)

        row.addWidget(self.folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        row.addWidget(QLabel("Quantity"))

        self.quantity = QComboBox()
        self.quantity.setMinimumWidth(90)
        self.quantity.currentIndexChanged.connect(self._on_quantity)

        row.addWidget(self.quantity)

        self.load_button = QPushButton("Load")
        self.load_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.load_button.clicked.connect(self._load)

        row.addWidget(self.load_button)

        layout.addLayout(row)

    def _build_results(self) -> QGroupBox:

        box = QGroupBox("Bonding")

        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel("Show"))

        self.scope_chips: dict[str, Chip] = {}

        for key, text in SCOPES.items():

            chip = Chip(text)
            chip.setChecked(key == "bonds")
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_scope(name)
            )

            self.scope_chips[key] = chip

            row.addWidget(chip)

        row.addWidget(QLabel("length <="))

        self.max_length = QDoubleSpinBox()
        self.max_length.setRange(0.5, 30.0)
        self.max_length.setSingleStep(0.1)
        self.max_length.setValue(30.0)
        self.max_length.setSuffix(" A")
        self.max_length.valueChanged.connect(lambda _: self._refresh_table())

        row.addWidget(self.max_length)

        self.strength_label = QLabel("strength >=")

        row.addWidget(self.strength_label)

        self.min_strength = QDoubleSpinBox()
        self.min_strength.setRange(-1000.0, 1000.0)
        self.min_strength.setSingleStep(0.05)
        self.min_strength.setDecimals(3)
        self.min_strength.setValue(-1000.0)
        self.min_strength.valueChanged.connect(lambda _: self._refresh_table())

        row.addWidget(self.min_strength)

        row.addWidget(QLabel("Pair"))

        self.pair = QComboBox()
        self.pair.addItem("All", None)
        self.pair.currentIndexChanged.connect(lambda _: self._refresh_table())

        row.addWidget(self.pair)

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
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        layout.addWidget(self.table, 1)

        self.notes = QLabel("")
        self.notes.setStyleSheet(theme.WARNING_STYLE)
        self.notes.setWordWrap(True)
        self.notes.setAlignment(Qt.AlignTop)

        self.notes_area = QScrollArea()
        self.notes_area.setWidgetResizable(True)
        self.notes_area.setWidget(self.notes)
        self.notes_area.setMaximumHeight(64)
        self.notes_area.setFrameShape(QFrame.NoFrame)
        self.notes_area.setVisible(False)

        layout.addWidget(self.notes_area)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self.plot_button = QPushButton("Plot selected")
        self.plot_button.setEnabled(False)
        self.plot_button.clicked.connect(lambda: self._plot(False))

        buttons.addWidget(self.plot_button)

        self.plot_all_button = QPushButton("Plot all shown")
        self.plot_all_button.setEnabled(False)
        self.plot_all_button.clicked.connect(lambda: self._plot(True))

        buttons.addWidget(self.plot_all_button)

        self.save_button = QPushButton("Save table...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_table)

        buttons.addWidget(self.save_button)

        self.visualize = QPushButton("Visualization...")
        self.visualize.setEnabled(False)
        self.visualize.clicked.connect(self._visualize)

        buttons.addWidget(self.visualize)

        layout.addLayout(buttons)

        return box

    # ------------------------------------------------------------
    # Source
    # ------------------------------------------------------------

    def _browse(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.folder.text() or settings.value(RUN_KEY, "", type=str)

        chosen = QFileDialog.getExistingDirectory(
            self, "LOBSTER run folder", start
        )

        if chosen:
            self.folder.setText(chosen)
            settings.setValue(RUN_KEY, chosen)
            self._look()

    def _look(self) -> None:
        """Offer only the quantities this folder has files for."""

        folder = self.folder.text().strip()

        self.quantity.blockSignals(True)
        self.quantity.clear()

        if not folder or not Path(folder).is_dir():
            self.quantity.blockSignals(False)
            _status(self.status, "That folder does not exist.", theme.WARNING_STYLE)
            return

        found = ch.available(folder)

        for name in found:
            self.quantity.addItem(name, name)

        self.quantity.blockSignals(False)

        if not found:
            _status(
                self.status,
                "No LOBSTER bonding output here. Expected "
                "ICOHPLIST.lobster or COHPCAR.lobster.",
                theme.WARNING_STYLE,
            )
            return

        _status(self.status, f"Found {', '.join(found)}. Press Load.")

    def _on_quantity(self, _index: int) -> None:

        # Switching quantity invalidates the loaded numbers, so reload if
        # something is already on screen rather than showing COHP values
        # under a COOP heading.
        if self.result is not None and self.quantity.currentData():
            self._load()

    def _load(self) -> None:

        folder = self.folder.text().strip()
        quantity = self.quantity.currentData()

        if not folder or not quantity:
            QMessageBox.information(
                self, "No folder", "Choose a folder with LOBSTER output."
            )
            return

        try:
            result = ch.read(folder, quantity)

        except ch.CohpError as exc:
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.warning(self, "COHP", str(exc))
            return

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.critical(self, "COHP", str(exc))
            return

        self.result = result

        self.summary.setText(
            f"{result.kind.name}   {len(result.bonds)} interactions"
            + ("   spin polarised" if result.spin_polarized else "")
            + ("" if result.has_curves else "   no curves")
        )

        self.notes.setText("\n".join(f"- {note}" for note in result.notes))
        self.notes_area.setVisible(bool(result.notes))

        self.strength_label.setText(f"{result.kind.strength_label} >=")

        self.pair.blockSignals(True)
        self.pair.clear()
        self.pair.addItem("All", None)

        for pair in result.pairs():
            self.pair.addItem(pair, pair)

        self.pair.blockSignals(False)

        longest = max(bond.length for bond in result.bonds)

        self.max_length.blockSignals(True)
        self.max_length.setValue(round(longest + 0.05, 2))
        self.max_length.blockSignals(False)

        self.min_strength.blockSignals(True)
        self.min_strength.setValue(
            round(min(bond.strength for bond in result.bonds) - 0.01, 3)
        )
        self.min_strength.blockSignals(False)

        self._refresh_table()

        for button in (
            self.save_button,
            self.visualize,
            self.plot_button,
            self.plot_all_button,
        ):
            button.setEnabled(True)

        if not result.has_curves:
            self.plot_button.setEnabled(False)
            self.plot_all_button.setEnabled(False)

        _status(
            self.status,
            f"Loaded {len(result.bonds)} interactions.",
            theme.SUCCESS_STYLE,
        )

    # ------------------------------------------------------------
    # The table
    # ------------------------------------------------------------

    def _pick_scope(self, name: str) -> None:

        self.scope = name

        for key, chip in self.scope_chips.items():
            chip.setChecked(key == name)

        self._refresh_table()

    def _selected_bonds(self) -> list[ch.Bond]:
        """The bonds the filters let through, strongest first."""

        if self.result is None:
            return []

        return ch.select(
            self.result.bonds,
            max_length=self.max_length.value(),
            min_strength=self.min_strength.value(),
            pairs=[self.pair.currentData()] if self.pair.currentData() else None,
        )

    def _refresh_table(self) -> None:

        if self.result is None:
            return

        self.table.setSortingEnabled(False)

        bonds = self._selected_bonds()

        if self.scope == "bonds":
            self._fill_bonds(bonds)

        elif self.scope == "pairs":
            self._fill_pairs(bonds)

        else:
            self._fill_atoms(bonds)

        self.table.setSortingEnabled(True)

    def _fill_bonds(self, bonds: list[ch.Bond]) -> None:

        self.rows = bonds

        kind = self.result.kind

        unit = f" ({kind.unit})" if kind.unit else ""

        columns = [
            "Bond",
            "Pair",
            "Length (A)",
            f"{kind.strength_label}{unit}",
            "Image",
        ]

        self._start(columns, len(bonds))

        for row, bond in enumerate(bonds):

            self._set(row, 0, bond.label())
            self._set(row, 1, bond.pair)
            self._set(row, 2, f"{bond.length:.{DECIMALS}f}", bond.length)
            self._set(row, 3, f"{bond.strength:.{DECIMALS}f}", bond.strength)
            self._set(row, 4, "{},{},{}".format(*bond.translation))

    def _fill_pairs(self, bonds: list[ch.Bond]) -> None:

        rows = ch.by_pair(bonds)

        self.rows = rows

        kind = self.result.kind

        columns = [
            "Pair",
            "Count",
            f"{kind.strength_label} mean",
            "min",
            "max",
            "total",
            "Length mean",
            "min",
            "max",
        ]

        self._start(columns, len(rows))

        for row, entry in enumerate(rows):

            self._set(row, 0, entry.pair)
            self._set(row, 1, str(entry.count), entry.count)
            self._set(row, 2, f"{entry.strength_mean:.{DECIMALS}f}", entry.strength_mean)
            self._set(row, 3, f"{entry.strength_min:.{DECIMALS}f}", entry.strength_min)
            self._set(row, 4, f"{entry.strength_max:.{DECIMALS}f}", entry.strength_max)
            self._set(row, 5, f"{entry.strength_total:.{DECIMALS}f}", entry.strength_total)
            self._set(row, 6, f"{entry.length_mean:.{DECIMALS}f}", entry.length_mean)
            self._set(row, 7, f"{entry.length_min:.{DECIMALS}f}", entry.length_min)
            self._set(row, 8, f"{entry.length_max:.{DECIMALS}f}", entry.length_max)

    def _fill_atoms(self, bonds: list[ch.Bond]) -> None:

        rows = ch.by_atom(bonds)

        self.rows = rows

        kind = self.result.kind

        columns = [
            "Atom",
            "Element",
            "Bonds",
            f"{kind.strength_label} total",
            "mean",
            "Length mean",
        ]

        self._start(columns, len(rows))

        for row, entry in enumerate(rows):

            self._set(row, 0, entry.label)
            self._set(row, 1, entry.element)
            self._set(row, 2, str(entry.count), entry.count)
            self._set(row, 3, f"{entry.strength_total:.{DECIMALS}f}", entry.strength_total)
            self._set(row, 4, f"{entry.strength_mean:.{DECIMALS}f}", entry.strength_mean)
            self._set(row, 5, f"{entry.length_mean:.{DECIMALS}f}", entry.length_mean)

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

        item = Cell()
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setData(Qt.DisplayRole, text)

        if sort is not None:
            item.setData(Qt.UserRole, float(sort))

        # Which bond a row is remains findable after the user sorts the
        # table, which reorders the rows underneath us.
        if column == 0:
            item.setData(Qt.UserRole + 1, row)

        if column > 0:
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.table.setItem(row, column, item)

    # ------------------------------------------------------------
    # Plot, save, draw
    # ------------------------------------------------------------

    def _rows_selected(self) -> list[int]:
        """The original row numbers of whatever is highlighted."""

        found = []

        for index in self.table.selectionModel().selectedRows():

            item = self.table.item(index.row(), 0)

            if item is None:
                continue

            original = item.data(Qt.UserRole + 1)

            if original is not None:
                found.append(int(original))

        return found

    def _plot(self, everything: bool) -> None:
        """Open the curves for the chosen bonds."""

        from qvista.gui.dialogs.cohp_plot import CohpPlotWindow

        if self.result is None:
            return

        bonds = self._selected_bonds()

        if not everything:

            picked = self._rows_selected()

            if not picked:
                QMessageBox.information(
                    self,
                    "Plot",
                    "Select one or more rows, or use Plot all shown.",
                )
                return

            if self.scope == "bonds":
                bonds = [bonds[index] for index in picked]

            elif self.scope == "pairs":
                wanted = {self.rows[index].pair for index in picked}
                bonds = [bond for bond in bonds if bond.pair in wanted]

            else:
                wanted = {self.rows[index].label for index in picked}
                bonds = [
                    bond
                    for bond in bonds
                    if bond.a_label in wanted or bond.b_label in wanted
                ]

        if not bonds:
            QMessageBox.information(self, "Plot", "Nothing to plot.")
            return

        try:
            window = CohpPlotWindow(
                self.result,
                bonds,
                heading=(
                    f"{self.result.kind.name}   {len(bonds)} bond(s)   "
                    f"{Path(self.folder.text().strip()).name}"
                ),
                parent=self,
            )

        except ValueError as exc:
            QMessageBox.information(self, "Plot", str(exc))
            return

        window.show()

    def _save_table(self) -> None:

        if self.result is None:
            return

        start = str(
            Path(self.folder.text().strip())
            / f"{self.result.kind.name.lower()}_{self.scope}.csv"
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

                writer.writerow([f"# {self.result.kind.name} {self.scope}"])
                writer.writerow([f"# folder {self.result.folder}"])
                writer.writerow(
                    [
                        f"# length <= {self.max_length.value():.3f} A, "
                        f"{self.result.kind.strength_label} >= "
                        f"{self.min_strength.value():.3f}"
                    ]
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

        from qvista.gui.dialogs.cohp_viewer import CohpViewer

        if self.result is None:
            return

        try:
            viewer = CohpViewer(
                self.result,
                heading=(
                    f"{self.result.kind.name}   "
                    f"{len(self.result.bonds)} interactions"
                ),
                parent=self,
            )

        except ValueError as exc:
            QMessageBox.information(self, "Visualization", str(exc))
            return

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.critical(self, "Visualization", str(exc))
            return

        viewer.show()

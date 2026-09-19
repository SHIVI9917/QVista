"""
Elastic tensor panel.

Two tables, because an elastic result is two different things.

The 6x6 tensor genuinely needs a grid: read as console text its symmetry,
and any obviously wrong entry, are easy to miss.  Both tensors are here --
the stiffness C, which says what stress a strain costs, and its inverse
the compliance S, which says what strain a stress produces -- on one
selector, since they are the same result seen from either side and nobody
wants two 6x6 grids on screen at once.

The derived moduli are the other table: a column of numbers meant to be
read down and compared, so the symbol and the unit get columns of their
own rather than being run into the name and the value.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qvista.gui import theme

VOIGT_HEADERS = ("xx", "yy", "zz", "yz", "xz", "xy")

#: The two tensors, as ``key -> (label, unit, decimals)``.
TENSORS: dict[str, tuple[str, str, int]] = {
    "stiffness": ("Stiffness C", "GPa", 2),
    "compliance": ("Compliance S = C^-1", "1/GPa", 6),
}

MODULI_HEADERS = ("Property", "Symbol", "Value", "Unit")

WARNING_STYLE = theme.WARNING_STYLE
ERROR_STYLE = theme.ERROR_STYLE
DIM_STYLE = f"color: {theme.TEXT_DIM};"


class ElasticPanel(QWidget):
    """Displays the elastic tensors and the moduli derived from them."""

    def __init__(self, parent=None) -> None:

        super().__init__(parent)

        self.result = None

        #: Both tensors, filled in when a result arrives.
        self._tensors: dict[str, object] = {}

        self.monospace = QFont("Consolas")
        self.monospace.setStyleHint(QFont.Monospace)

        layout = QVBoxLayout(self)

        # ----------------------------------------------------------
        # Warnings
        # ----------------------------------------------------------

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(ERROR_STYLE)
        self.warning.setVisible(False)

        layout.addWidget(self.warning)

        # ----------------------------------------------------------
        # Tensor
        # ----------------------------------------------------------

        header = QHBoxLayout()

        header.addWidget(QLabel("Tensor"))

        self.tensor_choice = QComboBox()

        for key, (label, unit, _) in TENSORS.items():
            self.tensor_choice.addItem(f"{label}  ({unit})", key)

        self.tensor_choice.currentIndexChanged.connect(
            lambda _: self._fill_tensor()
        )

        header.addWidget(self.tensor_choice)

        self.tensor_note = QLabel("")
        self.tensor_note.setStyleSheet(DIM_STYLE)

        header.addWidget(self.tensor_note, 1)

        layout.addLayout(header)

        self.tensor_table = QTableWidget(6, 6)

        self.tensor_table.setHorizontalHeaderLabels(VOIGT_HEADERS)
        self.tensor_table.setVerticalHeaderLabels(VOIGT_HEADERS)

        self.tensor_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )

        self.tensor_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        layout.addWidget(self.tensor_table)

        # ----------------------------------------------------------
        # Derived properties
        # ----------------------------------------------------------

        layout.addWidget(QLabel("Derived properties"))

        self.moduli_table = QTableWidget(0, len(MODULI_HEADERS))

        self.moduli_table.setHorizontalHeaderLabels(MODULI_HEADERS)

        self.moduli_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )

        head = self.moduli_table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.Stretch)

        for column in range(1, len(MODULI_HEADERS)):
            head.setSectionResizeMode(column, QHeaderView.ResizeToContents)

        self.moduli_table.verticalHeader().setVisible(False)

        layout.addWidget(self.moduli_table, 1)

        # ----------------------------------------------------------
        # Export
        # ----------------------------------------------------------

        buttons = QHBoxLayout()

        self.provenance = QLabel("")
        self.provenance.setStyleSheet(DIM_STYLE)

        buttons.addWidget(self.provenance)
        buttons.addStretch(1)

        self.save_button = QPushButton("Save tables...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)

        buttons.addWidget(self.save_button)

        layout.addLayout(buttons)

        self.clear()

    # ------------------------------------------------------------

    def clear(self) -> None:
        """Empty both tables."""

        self.result = None
        self._tensors = {}

        for row in range(6):
            for column in range(6):
                self.tensor_table.setItem(row, column, QTableWidgetItem(""))

        self.moduli_table.setRowCount(0)

        self.tensor_note.setText("")
        self.provenance.setText("")

        self.warning.setVisible(False)
        self.save_button.setEnabled(False)

    # ------------------------------------------------------------

    def show_result(self, result) -> None:
        """Populate the panel from an :class:`ElasticResult`."""

        self.clear()

        self.result = result

        # ----------------------------------------------------------
        # Tensors
        # ----------------------------------------------------------

        if result.tensor is not None:

            import numpy as np

            from qvista.analyses.elastic import compliance_tensor

            stiffness = np.asarray(result.tensor, dtype=float)

            self._tensors["stiffness"] = stiffness

            # A singular tensor has no compliance, but it still has a
            # stiffness worth looking at -- and looking at it is how the
            # user finds out why it is singular.
            try:
                self._tensors["compliance"] = compliance_tensor(stiffness)

            except Exception:  # noqa: BLE001 - reported by the selector
                pass

        # Landing on an empty grid because the selector happens to be on
        # the tensor this result does not have would hide the one it
        # does; the note still says why the other is missing.
        if (
            self.tensor_choice.currentData() not in self._tensors
            and "stiffness" in self._tensors
        ):
            self.tensor_choice.setCurrentIndex(0)

        self._fill_tensor()

        # ----------------------------------------------------------
        # Derived properties
        # ----------------------------------------------------------

        rows: list[tuple[str, str, str, str, str]] = []

        if result.moduli is not None:
            rows.extend(result.moduli.as_table())

        if result.stability is not None:
            rows.extend(result.stability.as_table())

        self._fill_moduli(rows)

        self.provenance.setText(
            f"{result.spacegroup_after} (#{result.spacegroup_number_after})"
            f"     forces from {result.force_field}"
        )

        self.save_button.setEnabled(bool(self._tensors) or bool(rows))

        # ----------------------------------------------------------
        # Warnings
        # ----------------------------------------------------------

        warnings: list[str] = []

        if result.orientation_is_arbitrary:
            warnings.append(
                "Refinement returned P1: the crystal axes are arbitrary, so "
                "C44/C55/C66 are not the shear constants they are labelled "
                "as. Do not trust these moduli."
            )

        if result.stability is not None and not result.stability.stable:
            warnings.extend(result.stability.failures)

        if warnings:
            self.warning.setText("\n\n".join(warnings))
            self.warning.setStyleSheet(ERROR_STYLE)
            self.warning.setVisible(True)

    # ------------------------------------------------------------

    def _fill_tensor(self) -> None:
        """Draw whichever of the two tensors is selected."""

        key = self.tensor_choice.currentData()

        _, unit, decimals = TENSORS[key]

        matrix = self._tensors.get(key)

        for row in range(6):
            for column in range(6):

                item = QTableWidgetItem(
                    "" if matrix is None
                    else f"{float(matrix[row][column]):.{decimals}f}"
                )

                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFont(self.monospace)

                self.tensor_table.setItem(row, column, item)

        # Exactly six rows tall, so the grid neither clips its last row
        # nor sits in a pool of empty space with the moduli pushed off
        # the bottom of the window.
        self.tensor_table.setFixedHeight(
            self.tensor_table.horizontalHeader().height()
            + sum(
                self.tensor_table.rowHeight(row) for row in range(6)
            )
            + 2 * self.tensor_table.frameWidth()
        )

        if matrix is None and self._tensors:
            self.tensor_note.setText(
                "C is singular, so it has no inverse."
            )
            self.tensor_note.setStyleSheet(WARNING_STYLE)

        elif matrix is None:
            self.tensor_note.setText("")

        else:
            self.tensor_note.setText(
                "what strain a stress produces"
                if key == "compliance"
                else "what stress a strain costs"
            )
            self.tensor_note.setStyleSheet(DIM_STYLE)

    def _fill_moduli(
        self,
        rows: list[tuple[str, str, str, str, str]],
    ) -> None:
        """
        One row per property, under a heading per section.

        The sections are rows of the same table rather than separate
        tables: it keeps the value column in one line down the page,
        which is the point of putting them in a table at all.
        """

        sections = []
        section = None

        for entry in rows:

            if entry[0] != section:
                section = entry[0]
                sections.append(section)

        self.moduli_table.setRowCount(len(rows) + len(sections))

        index = 0
        section = None

        for group, name, symbol, value, unit in rows:

            if group != section:

                section = group

                heading = QTableWidgetItem(group)

                font = heading.font()
                font.setBold(True)
                heading.setFont(font)

                heading.setFlags(Qt.ItemIsEnabled)

                self.moduli_table.setItem(index, 0, heading)

                for column in range(1, len(MODULI_HEADERS)):
                    blank = QTableWidgetItem("")
                    blank.setFlags(Qt.ItemIsEnabled)
                    self.moduli_table.setItem(index, column, blank)

                index += 1

            self.moduli_table.setItem(index, 0, QTableWidgetItem(f"   {name}"))
            self.moduli_table.setItem(index, 1, QTableWidgetItem(symbol))

            number = QTableWidgetItem(value)
            number.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            number.setFont(self.monospace)

            self.moduli_table.setItem(index, 2, number)
            self.moduli_table.setItem(index, 3, QTableWidgetItem(unit))

            index += 1

    # ------------------------------------------------------------

    def save(self) -> None:
        """Write both tensors and the moduli as one CSV."""

        if self.result is None:
            return

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save elastic tables", "elastic.csv", "CSV (*.csv)"
        )

        if not chosen:
            return

        try:
            with open(chosen, "w", newline="", encoding="utf-8") as handle:

                writer = csv.writer(handle)

                writer.writerow(["# QVista elastic properties"])
                writer.writerow(
                    [
                        f"# space group {self.result.spacegroup_after} "
                        f"(#{self.result.spacegroup_number_after})"
                    ]
                )
                writer.writerow([f"# forces from {self.result.force_field}"])

                for key, (label, unit, decimals) in TENSORS.items():

                    matrix = self._tensors.get(key)

                    if matrix is None:
                        continue

                    writer.writerow([])
                    writer.writerow([f"# {label} ({unit})"])
                    writer.writerow(["", *VOIGT_HEADERS])

                    for row in range(6):
                        writer.writerow(
                            [
                                VOIGT_HEADERS[row],
                                *(
                                    f"{float(matrix[row][column]):.{decimals}f}"
                                    for column in range(6)
                                ),
                            ]
                        )

                writer.writerow([])
                writer.writerow(["# Derived properties"])
                writer.writerow(["Section", *MODULI_HEADERS])

                if self.result.moduli is not None:
                    for entry in self.result.moduli.as_table():
                        writer.writerow(entry)

                if self.result.stability is not None:
                    for entry in self.result.stability.as_table():
                        writer.writerow(entry)

        except OSError as exc:
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not write {Path(chosen).name}: {exc}",
            )
            return

        QMessageBox.information(
            self, "Saved", f"Written to {Path(chosen).name}."
        )

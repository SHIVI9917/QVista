"""
Selective dynamics: choosing which atoms VASP may move.

Four ways to pick them, one at a time, with an "invert" that takes
everything the current choice left behind. Almost every real selection
is "these, and not those", and the complement is where the mistakes
happen when it is written out by hand.

The selection always names the atoms that are *free*. That is stated on
the window, not just in the docstring, because both conventions are in
use and getting it backwards produces a POSCAR that runs perfectly and
relaxes exactly the atoms meant to be held.

Picking atoms in the viewer
---------------------------
"Choose atoms" used to mean typing ``1-6, 9`` -- which means counting
lines in a POSCAR to find out which atom is which, the very thing this
dialog exists to avoid. When the structure being edited is also the one
on screen, clicking the atoms is offered instead, and the numbers are
written into the same box.

The two are only tied together when they are genuinely the same cell:
the symbols of the loaded file are compared against the symbols the
viewer is drawing, in order, and picking stays off unless they match.
A click that named atom 12 of a different structure would freeze the
wrong atom and give no sign of it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
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
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.sizing import fit_to_screen
from qvista.services import selective as sd

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

SOURCE_KEY = "selective/source"

NOTES_LINES = 4

#: How many atoms may be picked in the viewer at once. Freezing the
#: bottom of a slab by hand is a lot of clicks, so this is generous;
#: the axis cutoff is the tool for the really large selections.
PICK_LIMIT = 256


class Chip(QPushButton):
    """A small checkable box, matching the other analysis dialogs."""

    def __init__(self, text: str, parent=None) -> None:

        super().__init__(text, parent)

        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)


class SelectiveDynamicsDialog(QDialog):
    """Build a selective-dynamics POSCAR."""

    def __init__(self, start: str = "", viewer=None, parent=None) -> None:
        """
        ``start`` may be a folder or a file.

        A folder is only where the browser opens. A file is the
        structure QVista has loaded, and it is filled in and read
        straight away -- the atoms being frozen are almost always the
        ones on screen.

        ``viewer`` is the 3D view, when there is one. Given it, atoms
        can be chosen by clicking them; without it the dialog behaves
        exactly as it did.
        """

        super().__init__(parent)

        self.cell: sd.Cell | None = None
        self.selection: sd.Selection | None = None

        self.viewer = viewer

        #: True while this dialog owns the viewer's picking. Checked
        #: before every disarm so that closing this window cannot take
        #: picking away from another tool that armed it.
        self._picking = False

        self.mode = sd.BY_AXIS

        self.setWindowTitle("Selective Dynamics")
        fit_to_screen(self, 900, 720)

        # Modeless when there is a viewer: a modal dialog swallows the
        # clicks that would land on the atoms.
        #
        # Qt.Tool, not Qt.Window: an independent top-level window sinks
        # behind the main window the moment you click an atom in it,
        # which is exactly what this dialog asks you to do. See
        # RemoveAtomDialog for the longer version.
        if viewer is not None:
            self.setModal(False)
            self.setWindowFlags(
                Qt.Tool | Qt.WindowCloseButtonHint | Qt.WindowTitleHint
            )

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_source(layout)

        layout.addWidget(_hline())

        self._build_selection(layout)

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
        self._say("Choose a POSCAR, CONTCAR or CIF.")

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.write_button = QPushButton("Write POSCAR...")
        self.write_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.write_button.setEnabled(False)
        self.write_button.clicked.connect(self._write)

        footer.addWidget(self.write_button)
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        # A file is the loaded structure and is read now; a folder is
        # only where the browser will open, and goes nowhere visible.
        if start and Path(start).is_file():
            self.source.setText(start)
            self._load()

        self._arm_picking()

    # ------------------------------------------------------------

    def _say(self, text: str, style: str = STATUS_STYLE) -> None:

        self.status.setStyleSheet(style)
        self.status.setText(text)

    def _build_source(self, layout: QVBoxLayout) -> None:

        row = QHBoxLayout()

        row.addWidget(QLabel("Structure"))

        self.source = QLineEdit()
        self.source.setPlaceholderText("POSCAR, CONTCAR or CIF")
        self.source.editingFinished.connect(self._load)

        row.addWidget(self.source, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        load = QPushButton("Load")
        load.clicked.connect(self._load)

        row.addWidget(load)

        layout.addLayout(row)

    # ------------------------------------------------------------

    def _build_selection(self, layout: QVBoxLayout) -> None:

        box = QGroupBox("Atoms free to move")

        column = QVBoxLayout(box)

        # -- how to choose ------------------------------------------
        chips = QHBoxLayout()
        chips.setSpacing(6)

        self.chips: dict[str, Chip] = {}

        for key, label in sd.MODES.items():

            chip = Chip(
                {
                    sd.BY_AXIS: "Axis cutoff",
                    sd.BY_ELEMENT: "Element wise",
                    sd.BY_HAND: "Choose atoms",
                }[key]
            )
            chip.setToolTip(label)
            chip.setChecked(key == sd.BY_AXIS)
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_mode(name)
            )

            self.chips[key] = chip
            chips.addWidget(chip)

        chips.addStretch(1)

        # Everything the current choice left behind.
        self.invert_button = QPushButton("Select the rest")
        self.invert_button.setToolTip(
            "Take every atom the current selection did not, and make "
            "those the free ones instead."
        )
        self.invert_button.setEnabled(False)
        self.invert_button.clicked.connect(self._invert)

        chips.addWidget(self.invert_button)

        column.addLayout(chips)

        # -- axis ---------------------------------------------------
        self.axis_row = QWidget()

        axis_layout = QHBoxLayout(self.axis_row)
        axis_layout.setContentsMargins(0, 0, 0, 0)

        axis_layout.addWidget(QLabel("Axis"))

        self.axis = QComboBox()
        self.axis.addItems(list(sd.AXES))
        self.axis.setCurrentIndex(2)
        self.axis.currentIndexChanged.connect(self._axis_changed)

        axis_layout.addWidget(self.axis)

        axis_layout.addWidget(QLabel("Cut at"))

        self.cut = QDoubleSpinBox()
        self.cut.setDecimals(3)
        self.cut.setRange(-1000.0, 1000.0)
        self.cut.setSingleStep(0.5)
        self.cut.setSuffix(" A")
        self.cut.valueChanged.connect(self._refresh)

        axis_layout.addWidget(self.cut)

        self.side = QComboBox()

        for key, label in sd.SIDES.items():
            self.side.addItem(label, key)

        self.side.currentIndexChanged.connect(self._refresh)

        axis_layout.addWidget(self.side)

        self.span = QLabel("")
        self.span.setStyleSheet(STATUS_STYLE)

        axis_layout.addWidget(self.span)

        axis_layout.addStretch(1)

        column.addWidget(self.axis_row)

        # -- element ------------------------------------------------
        self.element_row = QWidget()

        self.element_layout = QHBoxLayout(self.element_row)
        self.element_layout.setContentsMargins(0, 0, 0, 0)
        self.element_layout.setSpacing(6)

        self.element_layout.addWidget(QLabel("Elements"))
        self.element_layout.addStretch(1)

        self.element_boxes: dict[str, QCheckBox] = {}

        column.addWidget(self.element_row)

        self.element_row.setVisible(False)

        # -- by hand ------------------------------------------------
        self.atoms_row = QWidget()

        atoms_column = QVBoxLayout(self.atoms_row)
        atoms_column.setContentsMargins(0, 0, 0, 0)
        atoms_column.setSpacing(4)

        atoms_layout = QHBoxLayout()

        atoms_layout.addWidget(QLabel("Atoms"))

        self.atoms = QLineEdit()
        self.atoms.setPlaceholderText("1-6, 9   or   1 2 3")
        self.atoms.textChanged.connect(self._refresh)

        atoms_layout.addWidget(self.atoms, 1)

        self.clear_picks = QPushButton("Clear")
        self.clear_picks.setToolTip(
            "Drop the picked atoms and empty the list."
        )
        self.clear_picks.clicked.connect(self._clear_picks)

        atoms_layout.addWidget(self.clear_picks)

        atoms_column.addLayout(atoms_layout)

        # Says whether clicking is available, and why not when it is
        # not -- a viewer showing a different cell is the usual reason,
        # and silence there would look like a broken feature.
        self.pick_hint = QLabel("")
        self.pick_hint.setStyleSheet(STATUS_STYLE)
        self.pick_hint.setWordWrap(True)

        atoms_column.addWidget(self.pick_hint)

        column.addWidget(self.atoms_row)

        self.atoms_row.setVisible(False)

        # -- which directions ---------------------------------------
        directions = QHBoxLayout()
        directions.setSpacing(6)

        directions.addWidget(QLabel("Free along"))

        self.direction_boxes: list[QCheckBox] = []

        for name in sd.AXES:

            check = QCheckBox(name)
            check.setChecked(True)
            check.toggled.connect(self._refresh)

            self.direction_boxes.append(check)
            directions.addWidget(check)

        directions.addStretch(1)

        note = QLabel(
            "The selection names the atoms that may move. "
            "Everything else is written F F F and held."
        )
        note.setStyleSheet(STATUS_STYLE)

        directions.addWidget(note)

        column.addLayout(directions)

        layout.addWidget(box)

    # ------------------------------------------------------------

    def _browse(self) -> None:

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.source.text() or settings.value(
            SOURCE_KEY, "", type=str
        )

        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Structure",
            start,
            "Structures (POSCAR CONTCAR *.vasp *.cif);;All Files (*)",
        )

        if not chosen:
            return

        self.source.setText(chosen)

        settings.setValue(SOURCE_KEY, chosen)

        self._load()

    def _load(self) -> None:

        path = self.source.text().strip()

        if not path:
            return

        try:
            self.cell = sd.load(path)

        except sd.SelectiveError as exc:
            self.cell = None
            self.write_button.setEnabled(False)
            self._arm_picking()
            self._say(str(exc), theme.WARNING_STYLE)
            return

        self._offer_elements()
        self._axis_changed()

        # Whether clicking is available follows the file that is loaded.
        self._arm_picking()

        self._say(
            f"{self.cell.natoms} atoms: "
            f"{', '.join(self.cell.elements())}."
        )

        self.invert_button.setEnabled(True)

        self._refresh()

    def _offer_elements(self) -> None:
        """Rebuild the element boxes for the structure just loaded."""

        for box in self.element_boxes.values():
            self.element_layout.removeWidget(box)
            box.deleteLater()

        self.element_boxes = {}

        if self.cell is None:
            return

        for position, element in enumerate(self.cell.elements(), start=1):

            box = QCheckBox(element)
            box.toggled.connect(self._refresh)

            self.element_boxes[element] = box

            self.element_layout.insertWidget(position, box)

    def _axis_changed(self) -> None:
        """Show the span, and put the cut in the middle of it."""

        if self.cell is None:
            return

        low, high = self.cell.span(self.axis.currentIndex())

        self.span.setText(f"spans {low:.2f} to {high:.2f} A")

        self.cut.blockSignals(True)
        self.cut.setRange(low - 1.0, high + 1.0)
        self.cut.setValue(0.5 * (low + high))
        self.cut.blockSignals(False)

        self._refresh()

    def _pick_mode(self, name: str) -> None:

        self.mode = name

        for key, chip in self.chips.items():
            chip.setChecked(key == name)

        self.axis_row.setVisible(name == sd.BY_AXIS)
        self.element_row.setVisible(name == sd.BY_ELEMENT)
        self.atoms_row.setVisible(name == sd.BY_HAND)

        # Clicking only means anything while atoms are being named one
        # by one; left armed, it would select an atom every time the
        # view was turned.
        self._arm_picking()

        self._refresh()

    # ------------------------------------------------------------
    # Choosing atoms by clicking them
    # ------------------------------------------------------------

    def _pickable(self) -> tuple[bool, str]:
        """Whether a click in the viewer names an atom of this file."""

        if self.viewer is None:
            return False, "Type the atom numbers, counting from one."

        structure = getattr(self.viewer, "structure", None)

        if structure is None:
            return (
                False,
                "Open this structure in QVista as well to pick atoms "
                "by clicking them.",
            )

        if self.cell is None:
            return False, "Load a structure above to pick atoms by clicking."

        shown = [
            str(symbol)
            for symbol in structure.atoms.get_chemical_symbols()
        ]

        # Same species in the same order is what makes atom 12 here and
        # atom 12 on screen the same atom. Anything else and a click
        # would freeze something the user never chose.
        if list(self.cell.symbols) != shown:
            return (
                False,
                "The viewer is showing a different structure, so "
                "clicking would name the wrong atoms. Open this file "
                "in QVista to pick them.",
            )

        # "Build" rather than "add to": the list follows the picks, so
        # the first click after typing replaces what was typed.
        return (
            True,
            "Click atoms in the viewer to build the list; click one "
            "again to drop it, or right-click to clear.",
        )

    def _arm_picking(self) -> None:
        """Turn viewer picking on or off to match the current mode."""

        allowed, message = self._pickable()

        self.pick_hint.setText(message)

        wanted = allowed and self.mode == sd.BY_HAND

        if wanted == self._picking:
            return

        if wanted:

            # Set first: enabling re-caps the selection, which can call
            # back into this dialog before the call returns.
            self._picking = True

            self.viewer.enable_atom_picking(
                self._viewer_picked, limit=PICK_LIMIT
            )

            return

        self._picking = False

        if self.viewer is not None:
            self.viewer.disable_atom_picking()

    def _viewer_picked(self, atoms) -> None:
        """The viewer's selection moved: write it into the atom list."""

        if not self._picking:
            return

        # A periodic image is a visible atom of its own, so clicking the
        # same ion on two faces of the cell is two picks and one atom.
        numbers = sorted(
            {int(atom.original_index) + 1 for atom in atoms}
        )

        # textChanged runs _refresh, so the table and the counts follow.
        self.atoms.setText(_ranges(numbers))

    def _clear_picks(self) -> None:
        """Empty the list, and the highlights that go with it."""

        if self._picking and self.viewer is not None:
            self.viewer.clear_selection(notify=False)

        self.atoms.clear()

    def done(self, result: int) -> None:
        """
        Disarm picking, or every later camera drag selects an atom.

        Hooked here rather than on ``closeEvent`` because that is the
        one point every way out passes through: the Close button, the
        window's cross and the Escape key all end in ``done``.
        """

        if self._picking and self.viewer is not None:
            self._picking = False
            self.viewer.disable_atom_picking()

        super().done(result)

    # ------------------------------------------------------------

    def _chosen(self) -> list[int]:
        """The atoms the current controls select."""

        if self.cell is None:
            return []

        if self.mode == sd.BY_AXIS:

            return sd.by_axis(
                self.cell,
                self.axis.currentIndex(),
                self.cut.value(),
                self.side.currentData(),
            )

        if self.mode == sd.BY_ELEMENT:

            wanted = [
                name
                for name, box in self.element_boxes.items()
                if box.isChecked()
            ]

            if not wanted:
                return []

            return sd.by_element(self.cell, wanted)

        return sd.by_hand(self.cell, self.atoms.text())

    def _description(self) -> str:
        """What the selection is, in words, for the table and the file."""

        if self.mode == sd.BY_AXIS:

            return (
                f"{sd.SIDES[self.side.currentData()].lower()} "
                f"{self.axis.currentText()} = {self.cut.value():g} A"
            )

        if self.mode == sd.BY_ELEMENT:

            wanted = [
                name
                for name, box in self.element_boxes.items()
                if box.isChecked()
            ]

            return ", ".join(wanted) or "no element chosen"

        return self.atoms.text().strip() or "no atoms named"

    def _invert(self) -> None:
        """
        Swap to everything the current selection left behind.

        Switches to the by-hand mode carrying the complement, because
        that is the only mode that can hold an arbitrary set -- and
        because the result should then be editable rather than snapping
        back the moment another control moves.
        """

        if self.cell is None:
            return

        try:
            current = self._chosen()

        except sd.SelectiveError as exc:
            self._say(str(exc), theme.WARNING_STYLE)
            return

        complement = sd.rest(self.cell, current)

        self.atoms.blockSignals(True)
        self.atoms.setText(_ranges(complement))
        self.atoms.blockSignals(False)

        self._pick_mode(sd.BY_HAND)

    # ------------------------------------------------------------

    def _refresh(self) -> None:
        """Rebuild the selection and everything shown about it."""

        if self.cell is None:
            return

        try:
            free = self._chosen()

            self.selection = sd.build(
                self.cell,
                free=free,
                axes=tuple(
                    box.isChecked() for box in self.direction_boxes
                ),
                mode=self.mode,
                description=self._description(),
            )

        except sd.SelectiveError as exc:
            self.selection = None
            self.write_button.setEnabled(False)
            self._fill_empty()
            self._say(str(exc), theme.WARNING_STYLE)
            return

        self.write_button.setEnabled(True)

        self._fill()

        selection = self.selection

        self._say(
            f"{len(selection.free)} of {self.cell.natoms} atoms free, "
            f"{len(selection.frozen)} held."
        )

        warnings = [
            note for note in selection.notes
            if note.startswith("WARNING")
        ]

        others = [
            note for note in selection.notes
            if not note.startswith("WARNING")
        ]

        self.notes.setText("\n\n".join(warnings + others))
        self.notes_area.setVisible(bool(selection.notes))

    def _fill_empty(self) -> None:

        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

    def _fill(self) -> None:
        """Summary rows, then the per-element split."""

        selection = self.selection

        if selection is None:
            return

        rows = list(selection.summary())

        rows.append(("", ""))

        for element, free, frozen in selection.per_element():
            rows.append(
                (f"  {element}", f"{free} free, {frozen} held")
            )

        self.table.clear()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.table.setRowCount(len(rows))

        for index, (label, value) in enumerate(rows):

            for column, text in enumerate((label, value)):

                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                self.table.setItem(index, column, item)

    # ------------------------------------------------------------

    def _write(self) -> None:

        if self.selection is None:
            return

        start = ""

        if self.cell is not None and self.cell.source is not None:
            start = str(self.cell.source.parent / "POSCAR_sd")

        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Write POSCAR",
            start,
            "VASP POSCAR (POSCAR *.vasp);;All Files (*)",
        )

        if not chosen:
            return

        try:
            written = sd.write(self.selection, chosen)

        except sd.SelectiveError as exc:
            self._say(str(exc), theme.WARNING_STYLE)
            return

        self._say(
            f"Wrote {written.name}: {len(self.selection.free)} free, "
            f"{len(self.selection.frozen)} held."
        )


def _ranges(numbers) -> str:
    """
    Collapse a sorted list into ``1-6, 9, 12-14``.

    A complement is usually most of the structure, and forty numbers in
    a text box is unreadable and unusable.
    """

    numbers = sorted(int(value) for value in numbers)

    if not numbers:
        return ""

    parts: list[str] = []

    start = previous = numbers[0]

    for value in numbers[1:]:

        if value == previous + 1:
            previous = value
            continue

        parts.append(
            str(start) if start == previous else f"{start}-{previous}"
        )

        start = previous = value

    parts.append(
        str(start) if start == previous else f"{start}-{previous}"
    )

    return ", ".join(parts)

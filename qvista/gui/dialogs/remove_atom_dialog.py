"""
Remove atoms by clicking them.

Click an atom in the viewer, it joins the list; click Remove and it is
gone. The usual reason is making a vacancy -- pull one Li out of a
cathode and relax -- which is otherwise a matter of counting lines in a
POSCAR and deleting the right one.

Nothing is written to disk until asked. The structure in the viewer is
edited in place so the result can be looked at, and Undo puts it back;
the file on disk is untouched either way until Save.

Why the atom numbers are watched so carefully
---------------------------------------------
Removing atom 7 renumbers every atom above it. Click 7 and 9, remove
them one at a time, and the second click removes what used to be atom
10. Everything here therefore collects the whole selection first and
deletes in one pass, from the highest index down -- which is the only
order in which the earlier indices stay valid.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from qvista.gui import theme
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.gui.sizing import fit_to_screen

STATUS_STYLE = f"color: {theme.TEXT_DIM};"

#: How many atoms may be picked at once. Removing a whole coordination
#: shell is a normal thing to want, so this is generous.
SELECTION_LIMIT = 32


class RemoveAtomDialog(QDialog):
    """Pick atoms in the viewer and delete them."""

    def __init__(self, viewer, parent=None) -> None:

        super().__init__(parent)

        self.viewer = viewer

        #: Structures pushed before each removal, for Undo.
        self.history: list = []

        self.setWindowTitle("Remove Atom")
        fit_to_screen(self, 520, 560)

        # Modeless: the viewer has to stay clickable behind it.
        #
        # Qt.Tool rather than Qt.Window, which is what this was. A
        # Qt.Window is an independent top-level window, and clicking the
        # main window to pick an atom raises the main window over it --
        # so the panel showing the selection disappears behind the thing
        # you are selecting in. Windows hides the mistake by keeping a
        # parented dialog above its parent regardless; on X11 and
        # Wayland the window manager decides, and it decides otherwise.
        # Qt.Tool is a utility window: above its parent, and it does not
        # take focus from the viewport.
        self.setModal(False)
        self.setWindowFlags(
            Qt.Tool | Qt.WindowCloseButtonHint | Qt.WindowTitleHint
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        hint = QLabel(
            "Click atoms in the viewer to select them. Right-click "
            "clears the selection."
        )
        hint.setStyleSheet(STATUS_STYLE)
        hint.setWordWrap(True)

        layout.addWidget(hint)

        layout.addWidget(_hline())

        # ----------------------------------------------------------

        box = QGroupBox("Selected")

        column = QVBoxLayout(box)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(
            ["Atom", "Element", "Position (Å)"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)

        column.addWidget(self.table, 1)

        buttons = QHBoxLayout()

        clear = QPushButton("Clear selection")
        clear.clicked.connect(self.viewer.clear_selection)

        buttons.addWidget(clear)

        buttons.addStretch(1)

        column.addLayout(buttons)

        layout.addWidget(box, 1)

        # ----------------------------------------------------------

        self.also_bonded = QCheckBox(
            "Also remove atoms bonded only to these"
        )
        self.also_bonded.setToolTip(
            "Takes any atom whose every bond went to something being "
            "removed. Pulling out a Co leaves its O with nothing to "
            "hold them, and this catches that in one step."
        )

        layout.addWidget(self.also_bonded)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self._say("Click an atom to begin.")

        layout.addWidget(self.status)

        footer = QHBoxLayout()

        self.remove_button = QPushButton("Remove")
        self.remove_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self._remove)

        footer.addWidget(self.remove_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._undo)

        footer.addWidget(self.undo_button)

        self.save_button = QPushButton("Save as...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)

        footer.addWidget(self.save_button)

        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.close)

        footer.addWidget(close)

        layout.addLayout(footer)

        # ----------------------------------------------------------

        self.viewer.enable_atom_picking(
            self._selection_changed, limit=SELECTION_LIMIT
        )

        self._selection_changed(self.viewer.selection.atoms)

    # ------------------------------------------------------------

    def _say(self, text: str, style: str = STATUS_STYLE) -> None:

        self.status.setStyleSheet(style)
        self.status.setText(text)

    def closeEvent(self, event) -> None:
        """Disarm picking, or every later camera drag selects an atom."""

        self.viewer.disable_atom_picking()

        super().closeEvent(event)

    # ------------------------------------------------------------

    def _chosen_sites(self) -> list[int]:
        """
        The atoms of the cell the selection names, counting from zero.

        The viewer selects *visible* atoms, and a periodic image is a
        visible atom of its own -- so clicking the same ion twice on
        opposite faces of the cell is two selections and one atom.
        ``original_index`` is what identifies the atom in the cell, and
        collapsing on it is what stops the second click removing a
        neighbour instead.
        """

        return sorted(
            {
                int(atom.original_index)
                for atom in self.viewer.selection.atoms
            }
        )

    def _selection_changed(self, atoms) -> None:
        """The viewer's selection moved."""

        structure = self.viewer.structure

        if structure is None:
            return

        chosen = self._chosen_sites()

        self.table.setRowCount(len(chosen))

        symbols = structure.atoms.get_chemical_symbols()
        positions = structure.atoms.get_positions()

        for row, index in enumerate(chosen):

            if not 0 <= index < len(symbols):
                continue

            x, y, z = positions[index]

            for column, text in enumerate(
                (
                    str(index + 1),
                    str(symbols[index]),
                    f"{x:.3f}  {y:.3f}  {z:.3f}",
                )
            ):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                self.table.setItem(row, column, item)

        self.remove_button.setEnabled(bool(chosen))

        if chosen:

            clicked = len(self.viewer.selection.atoms)

            extra = (
                f"  ({clicked} clicks; periodic images of the same "
                f"atom count once)"
                if clicked > len(chosen)
                else ""
            )

            self._say(
                f"{len(chosen)} atom(s) selected, of "
                f"{len(symbols)} in the cell.{extra}"
            )
        else:
            self._say("Click an atom to begin.")

    # ------------------------------------------------------------

    def _companions(self, chosen: list[int]) -> list[int]:
        """
        Atoms left with no neighbour once ``chosen`` has gone.

        Pulling out a cation leaves its anions holding nothing, and
        deleting those by hand afterwards means finding them first.
        """

        if not self.also_bonded.isChecked():
            return []

        structure = self.viewer.structure

        scene = getattr(self.viewer, "scene", None)
        data = getattr(scene, "scene_data", None)
        graph = getattr(data, "graph", None)

        if graph is None:
            return []

        from qvista.visualization.visible_bond_builder import bond_pairs

        neighbours: dict[int, set[int]] = {}

        # bond_pairs works in atom keys -- (site, tx, ty, tz) -- because
        # a bond to a periodic image is a real bond drawn between two
        # visible atoms. The site is the first element, and reducing to
        # it is what turns "this image of atom 9" into "atom 9".
        for first, second in bond_pairs(graph):

            left, right = int(first[0]), int(second[0])

            if left == right:
                continue

            neighbours.setdefault(left, set()).add(right)
            neighbours.setdefault(right, set()).add(left)

        going = set(chosen)

        return [
            atom
            for atom, partners in neighbours.items()
            if atom not in going and partners and partners <= going
        ]

    def _remove(self) -> None:
        """Delete the selection, highest index first."""

        structure = self.viewer.structure

        if structure is None:
            return

        chosen = self._chosen_sites()

        if not chosen:
            return

        chosen = sorted(set(chosen) | set(self._companions(chosen)))

        total = len(structure.atoms)

        if len(chosen) >= total:
            QMessageBox.warning(
                self,
                "Remove Atom",
                "That would remove every atom, leaving nothing to "
                "show or to save.",
            )
            return

        # Keep a copy for Undo before anything changes.
        self.history.append(structure.atoms.copy())

        atoms = structure.atoms

        # Highest index first: deleting atom 7 renumbers everything
        # above it, so any lower index collected earlier would then
        # point at the wrong atom.
        for index in reversed(chosen):
            del atoms[index]

        symbols = atoms.get_chemical_symbols()

        self.viewer.clear_selection(notify=False)

        self.viewer.load_structure(structure)

        # load_structure disarms nothing, but it does clear the
        # selection, so picking has to be re-armed against this dialog.
        self.viewer.enable_atom_picking(
            self._selection_changed, limit=SELECTION_LIMIT
        )

        self.undo_button.setEnabled(True)
        self.save_button.setEnabled(True)

        # Refresh first: _selection_changed writes its own status, and
        # would otherwise wipe the one below.
        self._selection_changed([])

        self._say(
            f"Removed {len(chosen)} atom(s). "
            f"{len(symbols)} left: {_formula(symbols)}.",
            theme.SUCCESS_STYLE,
        )

    def _undo(self) -> None:
        """Put the last removal back."""

        if not self.history:
            return

        structure = self.viewer.structure

        if structure is None:
            return

        before = self.history.pop()

        structure.atoms = before

        self.viewer.clear_selection(notify=False)
        self.viewer.load_structure(structure)

        self.viewer.enable_atom_picking(
            self._selection_changed, limit=SELECTION_LIMIT
        )

        self.undo_button.setEnabled(bool(self.history))

        self._selection_changed([])

        self._say(
            f"Restored. {len(before)} atoms: "
            f"{_formula(before.get_chemical_symbols())}."
        )

    # ------------------------------------------------------------

    def _save(self) -> None:
        """Write the edited structure out."""

        structure = self.viewer.structure

        if structure is None:
            return

        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save structure",
            "POSCAR_edited",
            "VASP POSCAR (POSCAR *.vasp);;CIF (*.cif);;XYZ (*.xyz)",
        )

        if not chosen:
            return

        from qvista.services import structure_writer

        suffix = Path(chosen).suffix.lower()

        fmt = {".cif": "cif", ".xyz": "xyz"}.get(suffix, "poscar")

        from pymatgen.io.ase import AseAtomsAdaptor

        try:
            written = structure_writer.write(
                AseAtomsAdaptor.get_structure(structure.atoms),
                chosen,
                fmt,
                periodic=True,
            )

        except structure_writer.StructureWriteError as exc:
            self._say(str(exc), theme.WARNING_STYLE)
            return

        except Exception as exc:  # noqa: BLE001 - shown, never swallowed
            self._say(
                f"{type(exc).__name__}: {exc}", theme.WARNING_STYLE
            )
            return

        self._say(f"Written to {written}.", theme.SUCCESS_STYLE)


def _formula(symbols) -> str:
    """A reduced-looking formula, for the status line."""

    counts: dict[str, int] = {}

    for symbol in symbols:
        counts[str(symbol)] = counts.get(str(symbol), 0) + 1

    return " ".join(
        f"{element}{count if count > 1 else ''}"
        for element, count in counts.items()
    )

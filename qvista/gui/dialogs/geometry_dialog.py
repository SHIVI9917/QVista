"""
Edit Structure: edit a bond length or a bond angle by hand.

Non-modal, like the electronegativity panel, because the whole point is
to keep clicking the structure behind it. Click two atoms and the panel
shows the distance between them; click three and it shows the angle at
the middle one. Type a new value, press Apply, and the structure moves.

What moves
----------
The selected atom, by default -- which is always well defined. Moving the
whole group attached to it is what a molecule editor does, and it is
offered whenever cutting the bond genuinely separates the structure into
two pieces. In a crystal it never does, so the option stays off and says
why: every atom is bonded to every other through the periodic network,
and there is no side to move.

Edits stack, and Undo walks back through them one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from qvista.gui import theme
from qvista.gui.dialogs.incar_dialog import CardButton
from qvista.gui.dialogs.kpath_dialog import (
    INFO_BUTTON_STYLE,
    SUCCESS_BUTTON_STYLE,
    _hline,
)
from qvista.services.geometry import (
    MOVE_BOTH,
    MOVE_FIRST,
    MOVE_SECOND,
    GeometryError,
    angle,
    attached_group,
    distance,
    set_angle,
    set_distance,
)
from qvista.visualization.selection import atom_key
from qvista.visualization.viewer import SELECTION_COLORS

#: The two things this panel changes.
LENGTH = "Bond Length"
ANGLE = "Bond Angle"

#: How many atoms each needs.
NEEDS = {LENGTH: 2, ANGLE: 3}

#: The line under each card.
TAGLINE = {
    LENGTH: "Two atoms",
    ANGLE: "Three atoms, corner in the middle",
}

#: Monospace, so the readout lines up as it changes.
READOUT_STYLE = (
    f"{theme.MONO_STYLE} color: {theme.TEXT}; font-size: 12pt;"
)


@dataclass
class Step:
    """One applied edit, kept so it can be taken back."""

    label: str
    positions: np.ndarray


class GeometryDialog(QDialog):
    """Live bond length and bond angle editor for the current selection."""

    def __init__(self, viewer, mode: str = LENGTH, parent=None) -> None:

        super().__init__(parent)

        self.viewer = viewer

        self.mode = mode if mode in NEEDS else LENGTH

        #: Positions before each applied edit, newest last.
        self.history: list[Step] = []

        self.cards: dict[str, CardButton] = {}

        self.setWindowTitle("Edit Structure")
        self.setMinimumWidth(520)

        # Non-modal: the point is to keep clicking the structure behind
        # it. Qt.Tool keeps it above the main window without stealing
        # focus from the viewport.
        self.setModal(False)
        self.setWindowFlag(Qt.Tool, True)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ----------------------------------------------------------

        cards = QHBoxLayout()
        cards.setSpacing(8)

        for name in (LENGTH, ANGLE):

            card = CardButton(name, TAGLINE[name])
            card.picked.connect(self._select)

            cards.addWidget(card, 1)

            self.cards[name] = card

        layout.addLayout(cards)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(theme.CAPTION_STYLE)

        layout.addWidget(self.hint)

        layout.addWidget(_hline())

        # ----------------------------------------------------------

        self.readout = QLabel()
        self.readout.setTextFormat(Qt.PlainText)
        self.readout.setStyleSheet(READOUT_STYLE)

        layout.addWidget(self.readout)

        # ----------------------------------------------------------

        row = QHBoxLayout()

        self.value = QDoubleSpinBox()
        self.value.setDecimals(4)
        self.value.setSingleStep(0.05)
        self.value.setMinimumWidth(130)

        row.addWidget(self.value)

        self.move_box = QComboBox()
        self.move_box.setMinimumWidth(170)
        self.move_box.currentIndexChanged.connect(self._on_move_changed)

        row.addWidget(self.move_box)

        self.group_box = QCheckBox("Move the attached group")
        self.group_box.setChecked(False)

        row.addWidget(self.group_box)

        row.addStretch(1)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.apply_button.clicked.connect(self._apply)

        row.addWidget(self.apply_button)

        layout.addLayout(row)

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet(theme.CAPTION_STYLE)

        layout.addWidget(self.note)

        layout.addWidget(_hline())

        # ----------------------------------------------------------

        buttons = QHBoxLayout()

        self.undo_button = QPushButton("Undo")
        self.undo_button.setStyleSheet(INFO_BUTTON_STYLE)
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._undo)

        buttons.addWidget(self.undo_button)

        self.revert_button = QPushButton("Revert all")
        self.revert_button.setStyleSheet(INFO_BUTTON_STYLE)
        self.revert_button.setEnabled(False)
        self.revert_button.clicked.connect(self._revert)

        buttons.addWidget(self.revert_button)

        buttons.addStretch(1)

        clear = QPushButton("Clear selection")
        clear.clicked.connect(self.viewer.clear_selection)

        buttons.addWidget(clear)

        close = QPushButton("Close")
        close.clicked.connect(self.close)

        buttons.addWidget(close)

        layout.addLayout(buttons)

        # ----------------------------------------------------------

        self._select(self.mode)

    # ------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Disarm picking, so ordinary rotation stops selecting atoms."""

        self.viewer.disable_atom_picking()

        super().closeEvent(event)

    # ------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------

    def _select(self, name: str) -> None:
        """A card was clicked, or the panel has just opened."""

        if name not in NEEDS:
            return

        self.mode = name

        for title, card in self.cards.items():
            card.set_checked(title == name)

        colours = ", ".join(SELECTION_COLORS[: NEEDS[name]])

        if name == LENGTH:
            self.hint.setText(
                f"Click two atoms ({colours}). Click one again to "
                f"deselect, right-click to clear. Dragging still "
                f"rotates the view."
            )
            self.value.setRange(0.1, 100.0)
            self.value.setSuffix(" A")

        else:
            self.hint.setText(
                f"Click three atoms ({colours}). The second one is the "
                f"corner. Right-click clears; dragging still rotates."
            )
            self.value.setRange(0.1, 179.9)
            self.value.setSuffix(" deg")

        self._fill_move_box()

        # Re-arms with the new limit, which trims a third atom when
        # going back to a bond length.
        self.viewer.enable_atom_picking(
            self.on_selection_changed, limit=NEEDS[name]
        )

        self.on_selection_changed(self.viewer.selection.atoms)

    def _fill_move_box(self) -> None:
        """What may move, named for the mode."""

        self.move_box.clear()

        if self.mode == LENGTH:
            entries = [
                ("Move atom 2", MOVE_SECOND),
                ("Move atom 1", MOVE_FIRST),
                ("Move both", MOVE_BOTH),
            ]

        else:
            entries = [
                ("Swing atom 3", MOVE_SECOND),
                ("Swing atom 1", MOVE_FIRST),
                ("Swing both", MOVE_BOTH),
            ]

        for label, key in entries:
            self.move_box.addItem(label, key)

    # ------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------

    def on_selection_changed(self, atoms) -> None:
        """Rewrite the readout for the current selection."""

        problem = self._problem(atoms)

        ready = problem is None

        self.apply_button.setEnabled(ready)
        self.value.setEnabled(ready)
        self.move_box.setEnabled(ready)

        if not ready:
            self.readout.setText(problem)
            self.note.setText("")
            self.group_box.setEnabled(False)
            self.group_box.setChecked(False)
            return

        self.readout.setText(self._describe(atoms))

        # The box always opens on the measured value, so Apply with
        # nothing typed is a no-op rather than a surprise.
        self.value.setValue(self._measure(atoms))

        self._update_group_box(atoms)

    def _problem(self, atoms) -> str | None:
        """Why the current selection cannot be used, or ``None``."""

        if self.viewer.structure is None:
            return "No structure loaded."

        needed = NEEDS[self.mode]

        if len(atoms) < needed:
            missing = needed - len(atoms)

            return (
                f"Select {needed} atoms -- {missing} more to go."
                if len(atoms)
                else f"Select {needed} atoms."
            )

        sites = [atom.original_index for atom in atoms]

        if len(set(sites)) != len(sites):
            return (
                "Two of those are periodic images of the same atom, so "
                "moving one moves the other. Pick different atoms."
            )

        return None

    def _describe(self, atoms) -> str:
        """The measurement, spelled out."""

        labels = [self._label(atom, i) for i, atom in enumerate(atoms)]

        if self.mode == LENGTH:
            return (
                f"{labels[0]}  -  {labels[1]}"
                f"      {self._measure(atoms):.4f} A"
            )

        return (
            f"{labels[0]}  -  {labels[1]}  -  {labels[2]}"
            f"      {self._measure(atoms):.3f} deg"
        )

    def _label(self, atom, index: int) -> str:
        """One atom, named the way the readout names it."""

        site, tx, ty, tz = atom_key(atom)

        image = "" if (tx, ty, tz) == (0, 0, 0) else "*"

        return f"{atom.symbol}{site}{image}"

    def _measure(self, atoms) -> float:
        """The current value of whatever is being changed."""

        if self.mode == LENGTH:
            return distance(atoms[0].position, atoms[1].position)

        return angle(
            atoms[0].position, atoms[1].position, atoms[2].position
        )

    # ------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------

    def _on_move_changed(self, _index: int) -> None:
        """Which end moves decides which group is the attached one."""

        atoms = self.viewer.selection.atoms

        if self._problem(atoms) is None:
            self._update_group_box(atoms)

    def _update_group_box(self, atoms) -> None:
        """
        Offer the group move only where there is a group to move.

        Cutting a bond in a crystal separates nothing, and neither does
        cutting one bond of a ring, so the box is disabled and says
        which it is rather than moving something arbitrary.
        """

        if self.move_box.currentData() == MOVE_BOTH:
            self.group_box.setEnabled(False)
            self.group_box.setChecked(False)

            self.note.setText(
                "Both ends move, so there is no fixed side for a group "
                "to hang off. Pick one end to move a group with it."
            )
            return

        group = self._group_for(atoms)

        if group is None:
            self.group_box.setEnabled(False)
            self.group_box.setChecked(False)

            self.note.setText(
                "Only the selected atom moves: cutting this bond leaves "
                "the structure in one piece -- a ring, or the periodic "
                "network of a crystal -- so there is no attached group "
                "to swing."
            )
            return

        self.group_box.setEnabled(True)

        count = len(group)

        self.note.setText(
            f"{count} atom{'' if count == 1 else 's'} attached beyond "
            f"this bond can move with it."
        )

    def _group_for(self, atoms) -> set[int] | None:
        """
        The atoms travelling with the moving end, or ``None``.

        Worked out on the anchor-moving pair the current mode uses: the
        bond itself for a length, and the bond from the corner outwards
        for an angle.
        """

        if self.viewer.structure is None:
            return None

        pair = self._anchor_and_moving(atoms)

        if pair is None:
            return None

        anchor, moving, offset = pair

        try:
            return attached_group(
                self.viewer.structure.atoms, anchor, moving, offset
            )

        except Exception:
            return None

    def _anchor_and_moving(self, atoms):
        """
        ``(anchor, moving, offset)`` for the chosen move mode.

        ``offset`` is which periodic image of the moving atom the bond
        runs to, taken from the two clicked atoms' own translations --
        so clicking an atom in a neighbouring image cuts the bond that
        is actually drawn on screen rather than one in the home cell.
        """

        mode = self.move_box.currentData()

        if mode not in (MOVE_FIRST, MOVE_SECOND):
            # Symmetric: both ends move, so neither is an anchor and a
            # rigid group on either side is not a well-posed question.
            return None

        if self.mode == LENGTH:
            near, far = (
                (atoms[0], atoms[1])
                if mode == MOVE_SECOND
                else (atoms[1], atoms[0])
            )

        else:
            # The bond that is cut runs from the corner to the arm that
            # swings -- not from one arm to the other, which is not a
            # bond at all and would make every angle look inseparable.
            near = atoms[1]
            far = atoms[2] if mode == MOVE_SECOND else atoms[0]

        offset = tuple(
            f - n
            for n, f in zip(near.translation, far.translation)
        )

        return near.original_index, far.original_index, offset

    # ------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------

    def _apply(self) -> None:
        """Move the atoms so the value becomes what is in the box."""

        atoms = self.viewer.selection.atoms

        if self._problem(atoms) is not None:
            return

        structure = self.viewer.structure

        before = np.array(
            structure.atoms.get_positions(), dtype=float, copy=True
        )

        try:
            shift = self._shift_for(atoms, before)

        except GeometryError as exc:
            QMessageBox.warning(self, "Edit Structure", str(exc))
            return

        after = shift.applied_to(before)

        self.history.append(
            Step(label=f"{self.mode}  {shift.label}", positions=before)
        )

        self.viewer.update_positions(after)

        self.undo_button.setEnabled(True)
        self.revert_button.setEnabled(True)

        self.on_selection_changed(self.viewer.selection.atoms)

    def _shift_for(self, atoms, positions):
        """Build the displacement for the current mode and selection."""

        target = self.value.value()

        mode = self.move_box.currentData()

        group = self._group_for(atoms) if self.group_box.isChecked() else None

        sites = [atom.original_index for atom in atoms]

        # The group belongs to whichever end is moving; the other end
        # never carries one, because it is not going anywhere.
        first_group = group if mode in (MOVE_FIRST,) else None
        last_group = group if mode in (MOVE_SECOND,) else None

        if self.mode == LENGTH:
            return set_distance(
                atoms[0].position,
                atoms[1].position,
                target,
                first_index=sites[0],
                second_index=sites[1],
                move=mode,
                first_group=first_group,
                second_group=last_group,
            )

        return set_angle(
            atoms[0].position,
            atoms[1].position,
            atoms[2].position,
            target,
            first_index=sites[0],
            last_index=sites[2],
            move=mode,
            first_group=first_group,
            last_group=last_group,
            positions=positions,
        )

    # ------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------

    def _undo(self) -> None:
        """Put the last edit back."""

        if not self.history:
            return

        step = self.history.pop()

        self.viewer.update_positions(step.positions)

        self.undo_button.setEnabled(bool(self.history))
        self.revert_button.setEnabled(bool(self.history))

        self.on_selection_changed(self.viewer.selection.atoms)

    def _revert(self) -> None:
        """Back to the geometry the panel opened on."""

        if not self.history:
            return

        first = self.history[0]

        self.history.clear()

        self.viewer.update_positions(first.positions)

        self.undo_button.setEnabled(False)
        self.revert_button.setEnabled(False)

        self.on_selection_changed(self.viewer.selection.atoms)

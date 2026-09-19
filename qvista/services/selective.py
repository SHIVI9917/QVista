"""
Selective dynamics: choosing which atoms VASP is allowed to move.

A POSCAR with ``Selective dynamics`` carries three flags per atom --
``T T T`` for free, ``F F F`` for frozen, and any mixture for an atom
that may move along some axes and not others. Freezing the bottom of a
slab so it holds the bulk lattice constant while the surface relaxes is
the reason this exists.

Four ways to choose the atoms, because four different questions get
asked:

**By axis** -- everything above or below a plane. A slab is defined by
its height, and typing out the seventy atoms below z = 5 A is not a
reasonable thing to ask of anyone.

**By element** -- freeze the framework, let the adsorbate move.

**By hand** -- a list of atom numbers, for the cases the other three do
not cover.

**The rest** -- everything the previous choice did not take. Almost
every real selection is "these, and not those", and writing out the
complement by hand is where the mistakes happen.

Which way round
---------------
A selection says which atoms are *free*; everything else is frozen. That
is stated rather than assumed because both conventions are in use, and
getting it backwards produces a POSCAR that runs perfectly and relaxes
exactly the atoms that were meant to be held.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class SelectiveError(Exception):
    """Raised when a selection cannot be made, with a reason."""


#: The Cartesian axes, as a user names them.
AXES: tuple[str, str, str] = ("X", "Y", "Z")

#: How a selection is built.
BY_AXIS = "axis"
BY_ELEMENT = "element"
BY_HAND = "atoms"

MODES: dict[str, str] = {
    BY_AXIS: "Everything above or below a plane",
    BY_ELEMENT: "Every atom of the chosen species",
    BY_HAND: "Atom numbers, typed in",
}

#: Which side of the cutting plane is taken.
BELOW = "below"
ABOVE = "above"

SIDES: dict[str, str] = {
    BELOW: "Below the cut",
    ABOVE: "Above the cut, and on it",
}

#: What the flags mean, spelled out so the direction is never assumed.
FREE = "T"
FROZEN = "F"


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


@dataclass
class Cell:
    """A structure, and the parts of it a selection needs."""

    structure: Any

    source: Path | None = None

    symbols: tuple[str, ...] = ()
    coords: Any = None
    """Cartesian, in angstrom."""

    @property
    def natoms(self) -> int:

        return len(self.symbols)

    def elements(self) -> list[str]:
        """The species, in the order they first appear."""

        return list(dict.fromkeys(self.symbols))

    def indices_of(self, element: str) -> list[int]:
        """Atom numbers of one species, counting from one."""

        return [
            index + 1
            for index, symbol in enumerate(self.symbols)
            if symbol == element
        ]

    def span(self, axis: int) -> tuple[float, float]:
        """
        How far the atoms reach along one axis, in angstrom.

        A cutting plane is only choosable against the span it cuts:
        "5 A" means nothing until you know z runs from 0 to 14.
        """

        if not 0 <= axis <= 2:
            raise SelectiveError(f"{axis} is not one of the three axes.")

        column = np.asarray(self.coords, dtype=float)[:, axis]

        return float(column.min()), float(column.max())


def load(source: Path | str) -> Cell:
    """Read a structure and pull out what a selection works on."""

    from qvista.services.structure_analyzer import (
        StructureAnalysisError,
        StructureAnalyzer,
    )

    path = Path(source)

    if not path.is_file():
        raise SelectiveError(f"There is no such file: {path}")

    try:
        parsed = StructureAnalyzer.parse(path)

    except StructureAnalysisError as exc:
        raise SelectiveError(str(exc)) from exc

    if not parsed.periodic:
        raise SelectiveError(
            f"{path.name} was read as a molecule, with no lattice.\n\n"
            f"Selective dynamics is a POSCAR feature, so it needs a "
            f"periodic structure."
        )

    structure = parsed.structure

    return Cell(
        structure=structure,
        source=path,
        symbols=tuple(str(site.specie.symbol) for site in structure),
        coords=np.asarray(structure.cart_coords, dtype=float),
    )


# ------------------------------------------------------------
# Choosing
# ------------------------------------------------------------


def by_axis(
    cell: Cell, axis: int, cut: float, side: str = BELOW
) -> list[int]:
    """
    Atom numbers on one side of a plane normal to a Cartesian axis.

    Atoms sitting exactly on the plane go with the upper half, so that
    the two sides always partition the structure and nothing can be
    claimed twice or lost.
    """

    if not 0 <= axis <= 2:
        raise SelectiveError(f"{axis} is not one of the three axes.")

    if side not in SIDES:
        raise SelectiveError(
            f"Unknown side {side!r}. Try {BELOW} or {ABOVE}."
        )

    column = np.asarray(cell.coords, dtype=float)[:, axis]

    lower = column < cut

    keep = lower if side == BELOW else ~lower

    return [int(index) + 1 for index in np.flatnonzero(keep)]


def by_element(cell: Cell, elements: Sequence[str]) -> list[int]:
    """Every atom of the named species."""

    wanted = [str(name).strip() for name in elements if str(name).strip()]

    if not wanted:
        raise SelectiveError("Name at least one element.")

    known = set(cell.elements())

    for name in wanted:
        if name not in known:
            raise SelectiveError(
                f"There is no {name} in this structure. It has "
                f"{', '.join(sorted(known))}."
            )

    chosen: list[int] = []

    for name in wanted:
        chosen.extend(cell.indices_of(name))

    return sorted(set(chosen))


def by_hand(cell: Cell, text: str) -> list[int]:
    """
    Atom numbers typed in, as ``1-6, 9`` or ``1 2 3``.

    Counting from one, matching every table in QVista and the way VASP
    itself numbers atoms in its output.

    Spaces and newlines separate as well as commas, because a list
    pasted out of a terminal or a spreadsheet arrives that way and
    refusing it is pedantry.
    """

    import re

    from qvista.services.orbital import parse_atom_list, OrbitalError

    normalised = re.sub(r"[\s;]+", ",", str(text).strip())

    try:
        chosen = parse_atom_list(normalised)

    except OrbitalError as exc:
        raise SelectiveError(str(exc)) from exc

    if not chosen:
        raise SelectiveError("Name at least one atom.")

    outside = [
        number for number in chosen if not 1 <= number <= cell.natoms
    ]

    if outside:
        raise SelectiveError(
            f"This structure has {cell.natoms} atoms, so "
            f"{', '.join(str(n) for n in outside)} "
            f"{'is' if len(outside) == 1 else 'are'} outside it."
        )

    return sorted(set(chosen))


def rest(cell: Cell, chosen: Sequence[int]) -> list[int]:
    """
    Every atom the selection did not take.

    Almost every real selection is "these, and not those", and writing
    out the complement by hand is where the mistakes happen.
    """

    taken = {int(number) for number in chosen}

    return [
        number
        for number in range(1, cell.natoms + 1)
        if number not in taken
    ]


# ------------------------------------------------------------
# Writing
# ------------------------------------------------------------


@dataclass
class Selection:
    """Which atoms are free, and what that produced."""

    cell: Cell

    free: list[int] = field(default_factory=list)
    """Atom numbers allowed to move, counting from one."""

    axes: tuple[bool, bool, bool] = (True, True, True)
    """Which directions a free atom may move in."""

    mode: str = BY_AXIS
    description: str = ""

    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------

    @property
    def frozen(self) -> list[int]:

        return rest(self.cell, self.free)

    @property
    def flags(self) -> list[tuple[bool, bool, bool]]:
        """One ``(x, y, z)`` triple per atom, in structure order."""

        allowed = set(self.free)

        return [
            self.axes if number in allowed else (False, False, False)
            for number in range(1, self.cell.natoms + 1)
        ]

    def summary(self) -> list[tuple[str, str]]:
        """Rows for the table."""

        free = len(self.free)
        total = self.cell.natoms

        rows = [
            ("Structure", str(self.cell.source or "")),
            ("Atoms", str(total)),
            ("Free to move", f"{free}  ({free / total:.0%})"),
            ("Held fixed", f"{total - free}"),
            (
                "Free along",
                ", ".join(
                    name
                    for name, on in zip(AXES, self.axes)
                    if on
                )
                or "nothing",
            ),
        ]

        if self.description:
            rows.insert(1, ("Selection", self.description))

        return rows

    def per_element(self) -> list[tuple[str, int, int]]:
        """``(element, free, frozen)`` for each species."""

        allowed = set(self.free)

        out: list[tuple[str, int, int]] = []

        for element in self.cell.elements():

            numbers = self.cell.indices_of(element)

            free = sum(1 for number in numbers if number in allowed)

            out.append((element, free, len(numbers) - free))

        return out

    def to_poscar(self) -> str:
        """The structure, with the flags written in."""

        from pymatgen.io.vasp.inputs import Poscar

        return str(
            Poscar(
                self.cell.structure,
                selective_dynamics=[
                    list(triple) for triple in self.flags
                ],
            )
        )


def build(
    cell: Cell,
    free: Sequence[int],
    axes: tuple[bool, bool, bool] = (True, True, True),
    mode: str = BY_AXIS,
    description: str = "",
) -> Selection:
    """Assemble a selection and check it says something useful."""

    chosen = sorted({int(number) for number in free})

    outside = [
        number for number in chosen if not 1 <= number <= cell.natoms
    ]

    if outside:
        raise SelectiveError(
            f"Atom {outside[0]} is outside this structure, which has "
            f"{cell.natoms}."
        )

    selection = Selection(
        cell=cell,
        free=chosen,
        axes=axes,
        mode=mode,
        description=description,
    )

    selection.notes = _notes(selection)

    return selection


def _notes(selection: Selection) -> list[str]:
    """Everything worth saying about a selection before it is written."""

    notes: list[str] = []

    total = selection.cell.natoms
    free = len(selection.free)

    if free == 0:
        notes.append(
            "WARNING: nothing is free to move. VASP will run and every "
            "ionic step will change nothing, which looks like instant "
            "convergence."
        )

    elif free == total:
        notes.append(
            "Every atom is free, so this POSCAR behaves exactly as one "
            "without selective dynamics. Written anyway, since the "
            "flags make the intent explicit."
        )

    if not any(selection.axes):
        notes.append(
            "WARNING: no direction is allowed, so the free atoms cannot "
            "move either. Tick at least one axis."
        )

    elif not all(selection.axes):
        held = ", ".join(
            name for name, on in zip(AXES, selection.axes) if not on
        )

        notes.append(
            f"Free atoms are held along {held}. That is a constraint on "
            f"the relaxation, not a symmetry: the forces along {held} "
            f"are still there and are simply not acted on."
        )

    notes.append(
        "T means free and F means fixed. The selection above names the "
        "atoms that may move; everything else is held."
    )

    return notes


def write(selection: Selection, target: Path | str) -> Path:
    """Write the POSCAR out."""

    path = Path(target)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(selection.to_poscar(), encoding="utf-8")

    except OSError as exc:
        raise SelectiveError(
            f"Could not write {path.name}: {exc}"
        ) from exc

    return path

"""
Atom selection in the crystal viewer.

Selecting by clicking in the 3D view brings two problems a list widget
would not have.

A click lands on the *surface* of a sphere glyph, not on an atom, and the
spheres are merged into one mesh per element -- so there is no atom index
to read back. The picked point is resolved to the nearest atom centre
instead, restricted to the element whose mesh was hit, which makes the
answer unambiguous even where spheres overlap.

And what is on screen are periodic images, rebuilt from scratch on every
redraw. A selection is therefore remembered as *which* atom it is -- site
index plus periodic translation -- and re-resolved against the new list
after a redraw, dropping anything that has moved out of the display
boundary.

This module holds no VTK and no Qt: it is the bookkeeping only.
"""

from __future__ import annotations

import numpy as np

#: Identifies one visible atom: its site in the cell, plus which periodic
#: image of it this is.
AtomKey = tuple[int, int, int, int]


def atom_key(atom) -> AtomKey:
    """Stable identity for a visible atom, across redraws."""

    return (atom.original_index, atom.tx, atom.ty, atom.tz)


def nearest_atom(
    atoms,
    point,
    symbol: str | None = None,
    max_distance: float | None = None,
):
    """
    The atom closest to ``point``, or ``None``.

    ``symbol`` restricts the search to one element, which is what the
    caller knows from the mesh that was hit. ``max_distance`` rejects a
    pick that landed too far from any atom -- a click on a bond, say.
    """

    point = np.asarray(point, dtype=float)

    best = None
    best_distance = float("inf")

    for atom in atoms:

        if symbol is not None and atom.symbol != symbol:
            continue

        offset = np.asarray(atom.position, dtype=float) - point

        distance = float(np.sqrt(offset @ offset))

        if distance < best_distance:
            best = atom
            best_distance = distance

    if best is None:
        return None

    if max_distance is not None and best_distance > max_distance:
        return None

    return best


class AtomSelection:
    """
    The atoms the user has clicked, newest last.

    At most :attr:`limit` are held. Clicking one more drops the oldest
    rather than clearing, so walking along a chain of bonds keeps showing
    the last few atoms clicked. Clicking a selected atom again removes it.

    The limit belongs to the tool doing the asking: a pair for a distance
    or an electronegativity comparison, three for a bond angle.
    """

    #: The default, and the most highlight colours the viewer has.
    MAX = 3

    def __init__(self, limit: int = 2) -> None:

        self._atoms: list = []
        self._limit = max(1, int(limit))

    @property
    def limit(self) -> int:
        """How many atoms may be held at once."""

        return self._limit

    @limit.setter
    def limit(self, value: int) -> None:
        """Change the cap, dropping the oldest if it has shrunk."""

        self._limit = max(1, int(value))

        while len(self._atoms) > self._limit:
            self._atoms.pop(0)

    def __len__(self) -> int:

        return len(self._atoms)

    @property
    def atoms(self) -> list:
        """The selected atoms, in the order they were clicked."""

        return list(self._atoms)

    def is_selected(self, atom) -> bool:
        """True when ``atom`` is already in the selection."""

        key = atom_key(atom)

        return any(atom_key(other) == key for other in self._atoms)

    def index_of(self, atom) -> int | None:
        """Position of ``atom`` in the selection, or ``None``."""

        key = atom_key(atom)

        for index, other in enumerate(self._atoms):

            if atom_key(other) == key:
                return index

        return None

    def pick(self, atom) -> None:
        """Add ``atom``, or remove it if it was already selected."""

        index = self.index_of(atom)

        if index is not None:
            self._atoms.pop(index)
            return

        self._atoms.append(atom)

        while len(self._atoms) > self._limit:
            self._atoms.pop(0)

    def clear(self) -> None:
        """Drop the whole selection."""

        self._atoms = []

    def resync(self, visible_atoms) -> bool:
        """
        Re-resolve the selection against a freshly built scene.

        Returns True when the selection changed -- because an atom is no
        longer displayed, so the caller has to redraw its highlights and
        tell whatever is showing the numbers.
        """

        by_key = {atom_key(atom): atom for atom in visible_atoms}

        kept = [
            by_key[key]
            for key in (atom_key(atom) for atom in self._atoms)
            if key in by_key
        ]

        changed = len(kept) != len(self._atoms)

        self._atoms = kept

        return changed

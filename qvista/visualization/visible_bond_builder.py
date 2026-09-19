"""
Bond builder for the visible atoms.

This builder is independent of ASE NeighborList and is used only
for visualization.

Bonds are normally decided by distance, recomputed on every redraw. That
is right for a structure being edited and wrong for one being animated:
a vibrating atom crosses the cutoff twice a period, so bonds blink in
and out and the molecule appears to come apart and reassemble. Passing
``pairs`` fixes the topology instead -- the same bonds are drawn between
the same atoms wherever they have moved to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from qvista.models.visible_atom import VisibleAtom
from qvista.visualization.bonding import (
    DEFAULT_SCALE,
    BondRules,
    pair_key,
)
from qvista.visualization.selection import AtomKey, atom_key


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass(slots=True)
class VisibleBond:
    """
    One visible bond.
    """

    atom_i: VisibleAtom
    atom_j: VisibleAtom

    distance: float


@dataclass(slots=True)
class VisibleBondGraph:
    """
    Bond graph for visualization.
    """

    atoms: list[VisibleAtom]

    bonds: list[VisibleBond]


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------

class VisibleBondBuilder:
    """
    Construct bonds directly from the visible atoms.
    """

    def __init__(
        self,
        scale: float = DEFAULT_SCALE,
    ) -> None:

        self.scale = scale

    # ------------------------------------------------------------------

    def build(
        self,
        atoms: list[VisibleAtom],
        pairs: Iterable[tuple[AtomKey, AtomKey]] | None = None,
        rules: BondRules | None = None,
    ) -> VisibleBondGraph:
        """
        Bonds between ``atoms``.

        By distance, unless ``pairs`` names the bonds to draw -- see the
        module docstring for why an animation needs the second.

        ``rules`` decides which element pairs count and how far each one
        reaches. Left out, every pair follows its covalent radii, which
        is what this builder did before the rules existed.
        """

        if rules is None:
            rules = BondRules(scale=self.scale)

        if pairs is not None:
            return self._fixed(atoms, pairs, rules)

        bonds: list[VisibleBond] = []

        # One lookup per element pair rather than two radii and a
        # multiply per candidate. Squared, so the loop below never takes
        # a square root: on a wide boundary this runs into the hundreds
        # of thousands of candidates and it is the whole cost of a
        # redraw.
        table = rules.cutoff_table(atom.symbol for atom in atoms)

        n = len(atoms)

        for i in range(n):

            ai = atoms[i]

            for j in range(i + 1, n):

                aj = atoms[j]

                limit = table.get(pair_key(ai.symbol, aj.symbol))

                # Absent means the pair is switched off entirely.
                if limit is None:
                    continue

                delta = aj.position - ai.position

                squared = float(delta @ delta)

                if squared <= limit:

                    bonds.append(
                        VisibleBond(
                            atom_i=ai,
                            atom_j=aj,
                            distance=squared**0.5,
                        )
                    )

        return VisibleBondGraph(
            atoms=atoms,
            bonds=bonds,
        )

    # ------------------------------------------------------------------

    def _fixed(
        self,
        atoms: list[VisibleAtom],
        pairs: Iterable[tuple[AtomKey, AtomKey]],
        rules: BondRules,
    ) -> VisibleBondGraph:
        """
        Draw exactly the bonds named, wherever their atoms now are.

        A pair naming an atom that is no longer on screen is skipped
        rather than being an error: the display boundary can change
        under a frozen topology.

        The rules are consulted for whether a pair is drawn at all, but
        not for how far it reaches. Holding the topology fixed is the
        entire point here -- re-testing the distance every frame is what
        makes a vibrating bond blink -- while a pair the user has
        switched off should be gone whether or not it is mid-animation.
        """

        index = {atom_key(atom): atom for atom in atoms}

        bonds: list[VisibleBond] = []

        for key_i, key_j in pairs:

            ai = index.get(key_i)
            aj = index.get(key_j)

            if ai is None or aj is None:
                continue

            if not rules.is_enabled(ai.symbol, aj.symbol):
                continue

            bonds.append(
                VisibleBond(
                    atom_i=ai,
                    atom_j=aj,
                    distance=float(
                        np.linalg.norm(aj.position - ai.position)
                    ),
                )
            )

        return VisibleBondGraph(atoms=atoms, bonds=bonds)


def bond_pairs(graph: VisibleBondGraph) -> list[tuple[AtomKey, AtomKey]]:
    """The atom-key pairs of a graph, for freezing its topology."""

    return [
        (atom_key(bond.atom_i), atom_key(bond.atom_j))
        for bond in graph.bonds
    ]

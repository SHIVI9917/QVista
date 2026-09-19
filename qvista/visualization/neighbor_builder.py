"""
Neighbor builder for QVista.

Constructs periodic neighbor lists using ASE.
"""

from __future__ import annotations

from dataclasses import dataclass

from ase.neighborlist import NeighborList

from qvista.visualization.radii import covalent_radius


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass(slots=True)
class Bond:
    """
    One bond between two atoms.
    """

    i: int
    j: int

    position_i: object
    position_j: object

    symbol_i: str
    symbol_j: str

    distance: float


@dataclass(slots=True)
class NeighborGraph:
    """
    Complete neighbor graph.
    """

    atoms: object
    bonds: list[Bond]
    neighbors: dict[int, list[int]]


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------

class NeighborBuilder:
    """
    Builds an ASE NeighborList once and converts it into a reusable graph.
    """

    def __init__(self, scale: float = 1.15):

        self.scale = scale

    # ------------------------------------------------------------------

    def build(self, structure) -> NeighborGraph:

        atoms = structure.atoms

        cutoffs = [
            self.scale * covalent_radius(atom.symbol)
            for atom in atoms
        ]

        nl = NeighborList(
            cutoffs=cutoffs,
            self_interaction=False,
            bothways=True,
        )

        nl.update(atoms)

        cell = atoms.cell.array

        bonds: list[Bond] = []

        neighbors: dict[int, list[int]] = {
            i: []
            for i in range(len(atoms))
        }

        drawn = set()

        for i, atom in enumerate(atoms):

            indices, offsets = nl.get_neighbors(i)

            for j, offset in zip(indices, offsets):

                key = tuple(sorted((i, j)))

                if key in drawn:
                    continue

                drawn.add(key)

                p1 = atom.position

                p2 = (
                    atoms[j].position
                    + offset @ cell
                )

                distance = ((p2 - p1) ** 2).sum() ** 0.5

                bonds.append(
                    Bond(
                        i=i,
                        j=j,
                        position_i=p1,
                        position_j=p2,
                        symbol_i=atom.symbol,
                        symbol_j=atoms[j].symbol,
                        distance=distance,
                    )
                )

                neighbors[i].append(j)
                neighbors[j].append(i)

        return NeighborGraph(
            atoms=atoms,
            bonds=bonds,
            neighbors=neighbors,
        )

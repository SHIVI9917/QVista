"""
Tests for the visualization bond builder.

Bonds are normally decided by distance on every redraw, which is right
for a structure being looked at and wrong for one being animated: a
vibrating atom crosses the cutoff twice a period, so the molecule
visibly comes apart and reassembles. A fixed topology is the fix, and
these are the tests that it stays fixed.
"""

from __future__ import annotations

import numpy as np

from qvista.models.visible_atom import VisibleAtom
from qvista.visualization.radii import covalent_radius
from qvista.visualization.selection import atom_key
from qvista.visualization.visible_bond_builder import (
    VisibleBondBuilder,
    bond_pairs,
)


def _atoms(*positions, symbol: str = "Si") -> list[VisibleAtom]:
    """Visible atoms at the given points, one site each."""

    return [
        VisibleAtom(
            symbol=symbol,
            position=np.asarray(point, dtype=float),
            original_index=index,
        )
        for index, point in enumerate(positions)
    ]


def _pair_distance(symbol: str = "Si") -> float:
    """A separation comfortably inside the bond cutoff."""

    return 1.15 * 2.0 * covalent_radius(symbol) * 0.9


# ------------------------------------------------------------
# By distance
# ------------------------------------------------------------


def test_close_atoms_bond():

    near = _pair_distance()

    graph = VisibleBondBuilder().build(_atoms([0, 0, 0], [near, 0, 0]))

    assert len(graph.bonds) == 1
    assert graph.bonds[0].distance == np.float64(near)


def test_distant_atoms_do_not():

    graph = VisibleBondBuilder().build(_atoms([0, 0, 0], [9.0, 0, 0]))

    assert graph.bonds == []


def test_stretching_a_bond_past_the_cutoff_breaks_it():
    """
    The behaviour a frozen topology exists to override.

    This is not a bug -- it is what you want while editing a structure.
    It is only wrong when the same atoms are oscillating.
    """

    builder = VisibleBondBuilder()

    near = _pair_distance()

    assert len(builder.build(_atoms([0, 0, 0], [near, 0, 0])).bonds) == 1
    assert builder.build(_atoms([0, 0, 0], [near * 2, 0, 0])).bonds == []


# ------------------------------------------------------------
# Frozen
# ------------------------------------------------------------


def test_a_frozen_bond_survives_being_stretched():
    """The point of the whole exercise."""

    builder = VisibleBondBuilder()

    near = _pair_distance()

    at_rest = builder.build(_atoms([0, 0, 0], [near, 0, 0]))

    pairs = bond_pairs(at_rest)

    assert len(pairs) == 1

    stretched = builder.build(
        _atoms([0, 0, 0], [near * 3, 0, 0]), pairs=pairs
    )

    assert len(stretched.bonds) == 1

    # ...and it reports the distance it actually has now.
    assert stretched.bonds[0].distance > at_rest.bonds[0].distance


def test_a_frozen_topology_gains_nothing_new():
    """
    Atoms squeezed together do not acquire a bond.

    Both directions matter: an animation that grows bonds as atoms
    approach flickers exactly as much as one that loses them.
    """

    builder = VisibleBondBuilder()

    apart = builder.build(_atoms([0, 0, 0], [9.0, 0, 0]))

    assert apart.bonds == []

    together = builder.build(
        _atoms([0, 0, 0], [_pair_distance(), 0, 0]),
        pairs=bond_pairs(apart),
    )

    assert together.bonds == []


def test_the_frozen_bonds_follow_the_atoms_that_moved():
    """A bond is drawn between the atoms' new positions, not the old."""

    builder = VisibleBondBuilder()

    near = _pair_distance()

    pairs = bond_pairs(
        builder.build(_atoms([0, 0, 0], [near, 0, 0]))
    )

    moved = _atoms([1.0, 2.0, 3.0], [1.0 + near, 2.0, 3.0])

    graph = builder.build(moved, pairs=pairs)

    assert len(graph.bonds) == 1

    bond = graph.bonds[0]

    assert np.allclose(bond.atom_i.position, [1.0, 2.0, 3.0])
    assert np.allclose(bond.atom_j.position, [1.0 + near, 2.0, 3.0])


def test_a_pair_whose_atom_left_the_view_is_skipped():
    """
    A frozen topology outlives the display boundary that made it.

    Naming an atom that is no longer on screen is not an error; the
    bond simply is not drawn.
    """

    builder = VisibleBondBuilder()

    near = _pair_distance()

    pairs = bond_pairs(
        builder.build(_atoms([0, 0, 0], [near, 0, 0]))
    )

    graph = builder.build(_atoms([0, 0, 0]), pairs=pairs)

    assert graph.bonds == []


def test_bond_pairs_names_atoms_by_identity_not_by_position():
    """
    Which is why the topology survives the atoms moving.

    The key is site plus periodic image -- the two things that do not
    change when an atom is displaced.
    """

    graph = VisibleBondBuilder().build(
        _atoms([0, 0, 0], [_pair_distance(), 0, 0])
    )

    (first, second), = bond_pairs(graph)

    assert first == atom_key(graph.bonds[0].atom_i)
    assert second == atom_key(graph.bonds[0].atom_j)

    assert first == (0, 0, 0, 0)
    assert second == (1, 0, 0, 0)

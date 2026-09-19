"""
Tests for the bond length and bond angle editor.

The measurements are easy to check against numbers anyone can work out
by hand. What the tests are really for is the invariants: setting a
length must not change any angle, setting an angle must not change any
length, and neither must move an atom nobody asked to move.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services.geometry import (
    MOVE_BOTH,
    MOVE_FIRST,
    MOVE_SECOND,
    GeometryError,
    angle,
    attached_group,
    bond_edges,
    bond_graph,
    distance,
    set_angle,
    set_distance,
)


# ------------------------------------------------------------
# Structures
# ------------------------------------------------------------


@pytest.fixture
def water():
    """A water molecule, no cell: the case a molecule editor is for."""

    from ase import Atoms

    return Atoms(
        symbols="OHH",
        positions=[
            [0.000, 0.000, 0.000],
            [0.958, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ],
    )


@pytest.fixture
def ethanol():
    """
    Something with a group worth moving.

    C-C-O-H with hydrogens: cutting the C-O bond separates the hydroxyl
    from the rest, which is exactly the case the group move exists for.
    """

    from ase.build import molecule

    return molecule("CH3CH2OH")


@pytest.fixture
def silicon():
    """A crystal: bonded to itself through the periodic network."""

    from ase.build import bulk

    return bulk("Si", "diamond", a=5.43)


# ------------------------------------------------------------
# Measuring
# ------------------------------------------------------------


def test_distance_is_the_ordinary_one():

    assert distance([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)


def test_angle_is_measured_at_the_middle_argument():
    """A right angle at the origin, whichever way round the arms go."""

    assert angle([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert angle([0, 1, 0], [0, 0, 0], [1, 0, 0]) == pytest.approx(90.0)


def test_a_straight_line_is_one_hundred_and_eighty_degrees():

    assert angle([-1, 0, 0], [0, 0, 0], [1, 0, 0]) == pytest.approx(180.0)


def test_a_rounding_error_does_not_become_a_domain_error():
    """
    Nearly-parallel arms must not fall out of arccos.

    A dot product that comes back as 1.0000000002 is a crash rather than
    a zero angle unless it is clamped.
    """

    value = angle([1.0, 0.0, 0.0], [0, 0, 0], [1.0, 1e-12, 0.0])

    assert not np.isnan(value)
    assert value == pytest.approx(0.0, abs=1e-3)


def test_water_measures_as_water(water):
    """Against the numbers the fixture was built from."""

    positions = water.get_positions()

    assert distance(positions[0], positions[1]) == pytest.approx(
        0.958, abs=1e-3
    )
    assert angle(
        positions[1], positions[0], positions[2]
    ) == pytest.approx(104.5, abs=0.5)


# ------------------------------------------------------------
# Bond length
# ------------------------------------------------------------


def test_setting_a_length_gives_that_length(water):

    positions = water.get_positions()

    shift = set_distance(
        positions[0], positions[1], 1.2, 0, 1, move=MOVE_SECOND
    )

    moved = shift.applied_to(positions)

    assert distance(moved[0], moved[1]) == pytest.approx(1.2)


def test_moving_the_second_atom_leaves_the_first_alone(water):

    positions = water.get_positions()

    moved = set_distance(
        positions[0], positions[1], 1.2, 0, 1, move=MOVE_SECOND
    ).applied_to(positions)

    assert np.allclose(moved[0], positions[0])
    assert not np.allclose(moved[1], positions[1])


def test_moving_the_first_atom_leaves_the_second_alone(water):

    positions = water.get_positions()

    moved = set_distance(
        positions[0], positions[1], 1.2, 0, 1, move=MOVE_FIRST
    ).applied_to(positions)

    assert np.allclose(moved[1], positions[1])
    assert distance(moved[0], moved[1]) == pytest.approx(1.2)


def test_moving_both_keeps_the_midpoint(water):
    """Symmetric means symmetric: the centre of the bond does not drift."""

    positions = water.get_positions()

    before = 0.5 * (positions[0] + positions[1])

    moved = set_distance(
        positions[0], positions[1], 1.4, 0, 1, move=MOVE_BOTH
    ).applied_to(positions)

    after = 0.5 * (moved[0] + moved[1])

    assert np.allclose(before, after)
    assert distance(moved[0], moved[1]) == pytest.approx(1.4)


def test_a_length_change_slides_along_the_bond(water):
    """
    The direction is preserved, so no angle elsewhere changes.

    Stretching O-H must not bend H-O-H; if it does, the move had a
    component off the bond axis.
    """

    positions = water.get_positions()

    before = angle(positions[1], positions[0], positions[2])

    moved = set_distance(
        positions[0], positions[1], 1.5, 0, 1, move=MOVE_SECOND
    ).applied_to(positions)

    after = angle(moved[1], moved[0], moved[2])

    assert after == pytest.approx(before, abs=1e-9)


def test_nothing_else_moves(ethanol):
    """Exactly the atoms asked for, and no others."""

    positions = ethanol.get_positions()

    shift = set_distance(
        positions[0], positions[1], 2.0, 0, 1, move=MOVE_SECOND
    )

    assert set(shift.moves) == {1}

    moved = shift.applied_to(positions)

    for index in range(len(positions)):

        if index == 1:
            continue

        assert np.allclose(moved[index], positions[index])


def test_two_atoms_on_top_of_each_other_are_refused():
    """There is no direction to move along, so there is no answer."""

    with pytest.raises(GeometryError):
        set_distance([0, 0, 0], [0, 0, 0], 1.0, 0, 1)


def test_a_negative_length_is_refused():

    with pytest.raises(GeometryError):
        set_distance([0, 0, 0], [1, 0, 0], -1.0, 0, 1)


# ------------------------------------------------------------
# Bond angle
# ------------------------------------------------------------


def test_setting_an_angle_gives_that_angle(water):

    positions = water.get_positions()

    moved = set_angle(
        positions[1],
        positions[0],
        positions[2],
        120.0,
        first_index=1,
        last_index=2,
        move=MOVE_SECOND,
    ).applied_to(positions)

    assert angle(moved[1], moved[0], moved[2]) == pytest.approx(120.0)


def test_an_angle_change_preserves_both_bond_lengths(water):
    """
    That is what makes it an angle change.

    A rotation about the vertex moves an atom along an arc, so its
    distance to the corner is unchanged -- and the other arm never moves
    at all.
    """

    positions = water.get_positions()

    before = (
        distance(positions[0], positions[1]),
        distance(positions[0], positions[2]),
    )

    moved = set_angle(
        positions[1],
        positions[0],
        positions[2],
        80.0,
        first_index=1,
        last_index=2,
        move=MOVE_SECOND,
    ).applied_to(positions)

    after = (
        distance(moved[0], moved[1]),
        distance(moved[0], moved[2]),
    )

    assert after[0] == pytest.approx(before[0], abs=1e-9)
    assert after[1] == pytest.approx(before[1], abs=1e-9)


def test_swinging_either_arm_reaches_the_same_angle(water):
    """The choice is which atom moves, not what the answer is."""

    positions = water.get_positions()

    for move in (MOVE_FIRST, MOVE_SECOND, MOVE_BOTH):

        moved = set_angle(
            positions[1],
            positions[0],
            positions[2],
            97.0,
            first_index=1,
            last_index=2,
            move=move,
        ).applied_to(positions)

        assert angle(moved[1], moved[0], moved[2]) == pytest.approx(97.0)


def test_swinging_the_last_atom_leaves_the_first_alone(water):

    positions = water.get_positions()

    shift = set_angle(
        positions[1],
        positions[0],
        positions[2],
        97.0,
        first_index=1,
        last_index=2,
        move=MOVE_SECOND,
    )

    assert set(shift.moves) == {2}


def test_the_vertex_never_moves(water):
    """It is the centre of the rotation, so it cannot."""

    positions = water.get_positions()

    for move in (MOVE_FIRST, MOVE_SECOND, MOVE_BOTH):

        shift = set_angle(
            positions[1],
            positions[0],
            positions[2],
            70.0,
            first_index=1,
            last_index=2,
            move=move,
        )

        assert 0 not in shift.moves


def test_a_collinear_fragment_can_still_be_bent():
    """
    Three atoms in a line define no plane, so one has to be chosen.

    Refusing would mean a linear molecule could never be bent, which is
    the one thing anyone wants to do to one.
    """

    positions = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    moved = set_angle(
        positions[0],
        positions[1],
        positions[2],
        150.0,
        first_index=0,
        last_index=2,
        move=MOVE_SECOND,
    ).applied_to(positions)

    assert angle(moved[0], moved[1], moved[2]) == pytest.approx(150.0)


def test_an_angle_outside_the_range_is_refused():

    positions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )

    for target in (0.0, 180.0, 200.0, -5.0):

        with pytest.raises(GeometryError):
            set_angle(
                positions[0],
                positions[1],
                positions[2],
                target,
                first_index=0,
                last_index=2,
            )


def test_an_atom_on_the_vertex_is_refused():

    with pytest.raises(GeometryError):
        set_angle(
            [0, 0, 0], [0, 0, 0], [1, 0, 0], 90.0,
            first_index=0, last_index=2,
        )


# ------------------------------------------------------------
# Connectivity
# ------------------------------------------------------------


def test_water_is_bonded_the_way_water_is(water):

    graph = bond_graph(water)

    assert graph[0] == {1, 2}
    assert graph[1] == {0}
    assert graph[2] == {0}


def test_cutting_a_bond_in_a_molecule_separates_it(water):
    """One hydrogen comes away; the rest stays with the oxygen."""

    group = attached_group(water, anchor=0, moving=1)

    assert group == {1}


def test_the_group_is_everything_beyond_the_bond(ethanol):
    """
    Not just the neighbour: the whole fragment travels.

    Cutting the C-O bond in ethanol has to take the hydroxyl hydrogen
    with the oxygen, or moving the O leaves its H behind.
    """

    graph = bond_graph(ethanol)

    symbols = ethanol.get_chemical_symbols()

    oxygen = symbols.index("O")

    carbon = next(
        index
        for index in graph[oxygen]
        if symbols[index] == "C"
    )

    group = attached_group(ethanol, anchor=carbon, moving=oxygen)

    assert group is not None
    assert oxygen in group

    # The hydroxyl hydrogen is bonded to the oxygen and to nothing else.
    for index in graph[oxygen]:
        if symbols[index] == "H":
            assert index in group

    assert carbon not in group


def test_a_crystal_has_no_side_to_move(silicon):
    """
    Cutting one bond of a periodic network separates nothing.

    Every atom is reachable from every other the long way round, so the
    honest answer is that there is no group -- not an arbitrary half.
    """

    graph = bond_graph(silicon)

    first = 0
    second = next(iter(graph[first]))

    assert attached_group(silicon, anchor=first, moving=second) is None


def test_the_same_pair_bonded_through_several_images_is_not_separable(
    silicon,
):
    """
    Silicon's two atoms are bonded to each other four times over.

    Each bond runs through a different translation. Collapsing them into
    one edge makes the lattice look like a dumbbell that comes apart
    when cut -- and cutting it would drag half a crystal across the
    cell.
    """

    edges = bond_edges(silicon)

    to_other = [offset for other, offset in edges[0] if other == 1]

    assert len(to_other) == 4, "the four tetrahedral bonds"
    assert len(set(to_other)) == 4, "each through its own image"

    for offset in to_other:
        assert attached_group(silicon, 0, 1, offset) is None


def test_a_molecule_in_a_periodic_box_still_separates():
    """
    A periodic cell is not by itself a reason to refuse.

    An isolated molecule in vacuum is what half of VASP input is; its
    bonds all run through the home image, so cutting one separates it
    exactly as it would with no cell at all.
    """

    from ase import Atoms

    boxed = Atoms(
        symbols="OHH",
        positions=[
            [5.000, 5.000, 5.000],
            [5.958, 5.000, 5.000],
            [4.760, 5.927, 5.000],
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )

    assert attached_group(boxed, anchor=0, moving=1) == {1}


def test_a_ring_has_no_side_to_move():
    """Same reason as a crystal: the far end is reachable anyway."""

    from ase.build import molecule

    benzene = molecule("C6H6")

    graph = bond_graph(benzene)

    symbols = benzene.get_chemical_symbols()

    # Two carbons bonded to each other, both in the ring.
    first = symbols.index("C")
    second = next(
        index for index in graph[first] if symbols[index] == "C"
    )

    assert attached_group(benzene, anchor=first, moving=second) is None


# ------------------------------------------------------------
# A molecule is not a lattice
# ------------------------------------------------------------


def test_a_molecule_keeps_its_geometry_on_screen(water):
    """
    No cell means no periodic images, and no division by one.

    Fractional coordinates are undefined without a lattice, and
    multiplying them back by a cell of zeros puts every atom on the
    origin -- a molecule drawn as a single dot, and nothing to select.
    """

    from qvista.models.structure import Structure
    from qvista.visualization.periodic_image_generator import (
        PeriodicImageGenerator,
    )
    from qvista.visualization.settings import ViewerSettings

    visible = PeriodicImageGenerator().generate(
        Structure(water), ViewerSettings().boundary
    )

    assert len(visible) == len(water)

    for atom, site in zip(visible, water):
        assert np.allclose(atom.position, site.position)
        assert atom.translation == (0, 0, 0)

    # ...and the distances that come off it are the real ones.
    assert distance(
        visible[0].position, visible[1].position
    ) == pytest.approx(0.958, abs=1e-3)


# ------------------------------------------------------------
# Group moves
# ------------------------------------------------------------


def test_a_group_travels_whole_when_a_bond_is_stretched(ethanol):
    """
    The fragment keeps its own shape.

    Every distance inside the group is unchanged; only its distance to
    the anchor moved.
    """

    positions = ethanol.get_positions()

    symbols = ethanol.get_chemical_symbols()

    graph = bond_graph(ethanol)

    oxygen = symbols.index("O")
    carbon = next(i for i in graph[oxygen] if symbols[i] == "C")

    group = attached_group(ethanol, anchor=carbon, moving=oxygen)

    moved = set_distance(
        positions[carbon],
        positions[oxygen],
        2.1,
        first_index=carbon,
        second_index=oxygen,
        move=MOVE_SECOND,
        second_group=group,
    ).applied_to(positions)

    assert distance(moved[carbon], moved[oxygen]) == pytest.approx(2.1)

    members = sorted(group)

    for a in members:
        for b in members:
            assert distance(moved[a], moved[b]) == pytest.approx(
                distance(positions[a], positions[b]), abs=1e-9
            )


def test_a_group_swings_rather_than_slides(ethanol):
    """
    A rotated group is rotated, not translated.

    Translating it by the pivot's displacement would keep every internal
    distance right and leave the group pointing the wrong way -- which is
    why the rotation needs every member's position, not just the pivot's.
    """

    positions = ethanol.get_positions()

    symbols = ethanol.get_chemical_symbols()

    graph = bond_graph(ethanol)

    oxygen = symbols.index("O")
    carbon = next(i for i in graph[oxygen] if symbols[i] == "C")
    other = next(i for i in graph[carbon] if i != oxygen)

    group = attached_group(ethanol, anchor=carbon, moving=oxygen)

    moved = set_angle(
        positions[other],
        positions[carbon],
        positions[oxygen],
        95.0,
        first_index=other,
        last_index=oxygen,
        move=MOVE_SECOND,
        last_group=group,
        positions=positions,
    ).applied_to(positions)

    assert angle(
        moved[other], moved[carbon], moved[oxygen]
    ) == pytest.approx(95.0)

    # Rigid: internal distances survive, and so does the distance from
    # every member to the vertex it turned about.
    members = sorted(group)

    for a in members:

        assert distance(moved[a], moved[carbon]) == pytest.approx(
            distance(positions[a], positions[carbon]), abs=1e-9
        )

        for b in members:
            assert distance(moved[a], moved[b]) == pytest.approx(
                distance(positions[a], positions[b]), abs=1e-9
            )


def test_rotating_a_group_without_positions_is_refused(ethanol):
    """A rigid rotation cannot be built from the pivot alone."""

    positions = ethanol.get_positions()

    with pytest.raises(GeometryError):
        set_angle(
            positions[0],
            positions[1],
            positions[2],
            95.0,
            first_index=0,
            last_index=2,
            move=MOVE_SECOND,
            last_group={2, 3},
        )

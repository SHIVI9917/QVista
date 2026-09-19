"""
Editing a structure's internal coordinates.

Two operations, the ones worth doing by hand: set the distance between
two atoms, and set the angle between two bonds. Both are the same shape
of problem -- measure what is there, decide what should move, move it --
and both are done on Cartesian positions, with the cell left alone.

What moves, and why it matters
------------------------------
In a molecule, stretching a bond means moving everything on one side of
it: the C-H distance in methanol changes by sliding the whole hydroxyl
group, not by leaving the O behind. That works because cutting the bond
separates the molecule into two pieces.

In a crystal it does not. Every atom is bonded to every other through the
periodic network, so there is no "side" of a bond to move, and asking for
one is asking for something that does not exist. :func:`attached_group`
returns ``None`` in that case rather than guessing, and the caller falls
back to moving the selected atom alone -- which is always well defined,
in a molecule and in a crystal alike.

Nothing here knows about periodic images. The caller works out the
displacement in the frame the user is looking at and applies it to the
home-cell atom; because an image is a pure translation of its home atom,
the two are the same displacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Which end of the pair moves.
MOVE_FIRST = "first"
MOVE_SECOND = "second"
MOVE_BOTH = "both"

#: Below this, two atoms are on top of each other and the direction
#: between them is noise rather than a bond.
MIN_SEPARATION = 1e-6

#: Below this the cross product of two bonds is numerical dust, so the
#: three atoms are collinear and the rotation plane is undetermined.
MIN_NORMAL = 1e-8

#: Multiplies the sum of covalent radii to decide what counts as bonded.
#: The same value the viewer draws bonds at, so the graph edited here is
#: the graph on screen.
BOND_SCALE = 1.15


# ----------------------------------------------------------------
# Measuring
# ----------------------------------------------------------------


def distance(first, second) -> float:
    """The distance between two Cartesian points, in Angstrom."""

    offset = np.asarray(second, dtype=float) - np.asarray(
        first, dtype=float
    )

    return float(np.linalg.norm(offset))


def angle(first, vertex, last) -> float:
    """
    The angle first-vertex-last, in degrees.

    The middle argument is the corner. Clamped before the arccos: a
    dot product that comes out at 1.0000000002 through rounding is a
    domain error rather than a zero angle.
    """

    a = np.asarray(first, dtype=float) - np.asarray(vertex, dtype=float)
    c = np.asarray(last, dtype=float) - np.asarray(vertex, dtype=float)

    na = float(np.linalg.norm(a))
    nc = float(np.linalg.norm(c))

    if na < MIN_SEPARATION or nc < MIN_SEPARATION:
        return 0.0

    cosine = float(np.dot(a, c) / (na * nc))

    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


# ----------------------------------------------------------------
# Connectivity
# ----------------------------------------------------------------


#: One neighbour: which atom, and which periodic image of it.
Neighbour = tuple[int, tuple[int, int, int]]

#: No translation -- the atom in the home cell.
HOME = (0, 0, 0)


def bond_edges(
    atoms,
    scale: float = BOND_SCALE,
) -> dict[int, list[Neighbour]]:
    """
    Every bond, keeping the periodic image each one goes to.

    The image is the whole point. Two atoms in a small cell are often
    bonded several times over, once through each of several
    translations -- silicon's two-atom cell has one atom bonded to the
    other four times. Collapsing those into a single edge makes a
    diamond lattice look like a dumbbell that comes apart when you cut
    it, which is how a crystal ends up torn in half.
    """

    from ase.neighborlist import NeighborList

    from qvista.visualization.radii import covalent_radius

    cutoffs = [scale * covalent_radius(atom.symbol) for atom in atoms]

    # skin=0. ASE defaults it to 0.3 A, which exists so a neighbour list
    # can be reused as atoms drift during dynamics. Here it is not a
    # tolerance but an extra 0.6 A on every cutoff, and that is the
    # difference between water being O-H, O-H and water being a triangle
    # with an H-H bond in it.
    neighbours = NeighborList(
        cutoffs=cutoffs,
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )

    neighbours.update(atoms)

    edges: dict[int, list[Neighbour]] = {
        index: [] for index in range(len(atoms))
    }

    for index in range(len(atoms)):

        others, offsets = neighbours.get_neighbors(index)

        for other, offset in zip(others, offsets):

            edges[index].append(
                (int(other), tuple(int(value) for value in offset))
            )

    return edges


def bond_graph(atoms, scale: float = BOND_SCALE) -> dict[int, set[int]]:
    """
    Which atoms are bonded to which, ignoring the images.

    The plain adjacency, for anything that just wants to know whether
    two atoms are neighbours. Deciding what may move needs
    :func:`bond_edges` instead.
    """

    return {
        index: {other for other, _offset in neighbours}
        for index, neighbours in bond_edges(atoms, scale).items()
    }


def attached_group(
    atoms,
    anchor: int,
    moving: int,
    offset: tuple[int, int, int] = HOME,
    scale: float = BOND_SCALE,
) -> set[int] | None:
    """
    Everything rigidly attached to ``moving`` when one bond is cut.

    ``offset`` says which image of ``moving`` the bond runs to, relative
    to ``anchor`` in its home cell; the search walks outwards from
    there, cutting that one bond and no other.

    Returns ``None`` when the fragment is not a fragment: when the
    anchor turns out to be reachable another way, or when the walk
    arrives back at an atom it has already placed but in a different
    periodic image -- which is what an endless chain looks like from the
    inside. Either way there is no side to move, and moving one anyway
    would tear the structure apart.
    """

    edges = bond_edges(atoms, scale)

    cut = {
        ((moving, offset), (anchor, HOME)),
        ((anchor, HOME), (moving, offset)),
    }

    #: Site index -> the one image of it this fragment contains.
    reached: dict[int, tuple[int, int, int]] = {moving: offset}

    stack = [(moving, offset)]

    while stack:

        site, here = stack.pop()

        for other, step in edges.get(site, ()):

            there = (
                here[0] + step[0],
                here[1] + step[1],
                here[2] + step[2],
            )

            if ((site, here), (other, there)) in cut:
                continue

            if other == anchor:
                # Round the ring, or round the lattice.
                return None

            if other in reached:

                if reached[other] != there:
                    # The same atom in two places at once: the fragment
                    # repeats forever.
                    return None

                continue

            reached[other] = there

            stack.append((other, there))

    return set(reached)


# ----------------------------------------------------------------
# Editing
# ----------------------------------------------------------------


class GeometryError(Exception):
    """A change that cannot be made, with a reason a user can act on."""


@dataclass(frozen=True)
class Shift:
    """
    What one edit does: a displacement per atom index.

    Returned rather than applied so the caller can decide what to do with
    it -- put it on an undo stack, apply it to the home cell rather than
    to the periodic image that was clicked, or show it before committing.
    """

    #: Atom index -> the vector to add to its position, in Angstrom.
    moves: dict[int, np.ndarray]

    #: What changed, and from what to what. For the console.
    label: str

    def applied_to(self, positions) -> np.ndarray:
        """A fresh copy of ``positions`` with the shifts added."""

        moved = np.array(positions, dtype=float, copy=True)

        for index, delta in self.moves.items():
            moved[index] = moved[index] + delta

        return moved


def set_distance(
    first_position,
    second_position,
    target: float,
    first_index: int,
    second_index: int,
    move: str = MOVE_SECOND,
    first_group: set[int] | None = None,
    second_group: set[int] | None = None,
) -> Shift:
    """
    Move atoms so the two are ``target`` Angstrom apart.

    The pair keeps its direction: everything slides along the line
    joining them, so no angle anywhere else changes. ``*_group`` are the
    atoms that travel with each end; ``None`` means that end moves alone.
    """

    start = np.asarray(first_position, dtype=float)
    end = np.asarray(second_position, dtype=float)

    current = float(np.linalg.norm(end - start))

    if current < MIN_SEPARATION:
        raise GeometryError(
            "These two atoms are at the same position, so there is no "
            "direction to move them along."
        )

    if target <= 0.0:
        raise GeometryError("A bond length has to be greater than zero.")

    unit = (end - start) / current

    change = target - current

    if move == MOVE_SECOND:
        deltas = {second_index: change * unit}
        groups = {second_index: second_group}

    elif move == MOVE_FIRST:
        deltas = {first_index: -change * unit}
        groups = {first_index: first_group}

    elif move == MOVE_BOTH:
        deltas = {
            first_index: -0.5 * change * unit,
            second_index: 0.5 * change * unit,
        }
        groups = {first_index: first_group, second_index: second_group}

    else:
        raise GeometryError(f"Unknown move mode: {move}")

    moves: dict[int, np.ndarray] = {}

    for index, delta in deltas.items():

        for member in (groups.get(index) or {index}):
            moves[member] = np.array(delta, dtype=float)

    return Shift(
        moves=moves,
        label=f"{current:.4f} -> {target:.4f} A",
    )


def set_angle(
    first_position,
    vertex_position,
    last_position,
    target: float,
    first_index: int,
    last_index: int,
    move: str = MOVE_SECOND,
    first_group: set[int] | None = None,
    last_group: set[int] | None = None,
    positions=None,
) -> Shift:
    """
    Rotate about the vertex so the angle becomes ``target`` degrees.

    The rotation is in the plane the three atoms already define, about
    an axis through the vertex, so both bond lengths are preserved
    exactly -- the thing that makes this an angle change rather than a
    displacement.

    ``move`` names the end that swings: :data:`MOVE_SECOND` is the last
    atom, :data:`MOVE_FIRST` the first, :data:`MOVE_BOTH` splits the
    change between them.

    A group swings *with* the atom rather than sliding after it, so
    ``positions`` -- every atom's position -- is required whenever a
    group is given. Translating a group by the pivot's displacement
    would keep its internal geometry and lose its orientation, which is
    what makes a rigid rotation different from a shift.
    """

    vertex = np.asarray(vertex_position, dtype=float)

    a = np.asarray(first_position, dtype=float) - vertex
    c = np.asarray(last_position, dtype=float) - vertex

    if (
        float(np.linalg.norm(a)) < MIN_SEPARATION
        or float(np.linalg.norm(c)) < MIN_SEPARATION
    ):
        raise GeometryError(
            "One of the atoms sits on the vertex, so there is no angle "
            "to change."
        )

    if not 0.0 < target < 180.0:
        raise GeometryError(
            "A bond angle has to be between 0 and 180 degrees."
        )

    current = angle(first_position, vertex_position, last_position)

    axis = np.cross(a, c)

    if float(np.linalg.norm(axis)) < MIN_NORMAL:
        # Straight through the vertex: every plane containing the three
        # atoms is equally valid, so any perpendicular will do. Bending
        # a linear fragment has to pick a direction somehow.
        axis = _any_perpendicular(a)

    axis = axis / float(np.linalg.norm(axis))

    change = np.radians(target - current)

    # Rotating the first atom about a x c by +theta carries it onto the
    # last one, so the last atom swings away from the first on +change
    # and the first swings away on -change.
    if move == MOVE_SECOND:
        turns = [(last_index, c, change, last_group)]

    elif move == MOVE_FIRST:
        turns = [(first_index, a, -change, first_group)]

    elif move == MOVE_BOTH:
        turns = [
            (first_index, a, -0.5 * change, first_group),
            (last_index, c, 0.5 * change, last_group),
        ]

    else:
        raise GeometryError(f"Unknown move mode: {move}")

    moves: dict[int, np.ndarray] = {}

    for index, arm, theta, group in turns:

        if not group or group == {index}:
            moves[index] = _rotate(arm, axis, theta) - arm
            continue

        if positions is None:
            raise GeometryError(
                "Rotating a group needs every atom's position."
            )

        every = np.asarray(positions, dtype=float)

        for member in group:

            spoke = every[member] - vertex

            moves[member] = _rotate(spoke, axis, theta) - spoke

    return Shift(
        moves=moves,
        label=f"{current:.3f} -> {target:.3f} deg",
    )


def _rotate(vector, axis, theta: float):
    """Rodrigues' rotation of ``vector`` about a unit ``axis``."""

    vector = np.asarray(vector, dtype=float)
    axis = np.asarray(axis, dtype=float)

    cos = float(np.cos(theta))
    sin = float(np.sin(theta))

    return (
        vector * cos
        + np.cross(axis, vector) * sin
        + axis * float(np.dot(axis, vector)) * (1.0 - cos)
    )


def _any_perpendicular(vector):
    """Some unit vector at right angles to ``vector``."""

    vector = np.asarray(vector, dtype=float)

    # Cross with whichever axis the vector is least aligned to, so the
    # result is never the zero vector.
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(vector)))] = 1.0

    perpendicular = np.cross(vector, axis)

    return perpendicular / float(np.linalg.norm(perpendicular))

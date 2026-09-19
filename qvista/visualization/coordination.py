"""
Coordination shells, using the viewer's own idea of a bond.

Shared by everything that asks "what is this atom bonded to": the
electronegativity panel and the shift map both have to agree with each
other and with the sticks drawn on screen, so the rule lives here once.

The subtlety is the display boundary. Visible atoms are clipped to the
box the user set, so an atom near its edge appears to have lost half its
neighbours -- an octahedral site reads as four- or five-coordinate and
every derived number is then wrong. The shell is therefore searched in a
*padded* pool that reaches beyond the box, which is what the scene's
generator produces for the polyhedron builder for the same reason.
"""

from __future__ import annotations

from qvista.visualization.radii import covalent_radius
from qvista.visualization.selection import atom_key


def neighbour_pool(scene, structure, boundary) -> list:
    """Visible atoms plus those just outside the display boundary."""

    return scene.periodic_generator.generate_padded(structure, boundary)


def boundary_key(structure, boundary) -> tuple:
    """
    Identity of a pool, for caching.

    Rebuilding the padded pool per selection change is wasteful; it only
    changes when the structure or the display boundary does.
    """

    return (
        id(structure),
        boundary.x_minus,
        boundary.x_plus,
        boundary.y_minus,
        boundary.y_plus,
        boundary.z_minus,
        boundary.z_plus,
    )


def coordination_shell(atom, pool, scale: float) -> list:
    """
    What ``atom`` is bonded to, as ``(symbol, distance, vector)``.

    ``vector`` points from the centre to the neighbour and is not
    normalised -- callers that need directions normalise them
    themselves. ``scale`` is the bond-length factor the viewer uses, so
    the shell is exactly the set of sticks it would draw.
    """

    key = atom_key(atom)

    radius = covalent_radius(atom.symbol)

    shell = []

    for other in pool:

        if atom_key(other) == key:
            continue

        offset = other.position - atom.position

        distance = float((offset @ offset) ** 0.5)

        if distance <= scale * (radius + covalent_radius(other.symbol)):

            shell.append((other.symbol, distance, tuple(offset)))

    return shell

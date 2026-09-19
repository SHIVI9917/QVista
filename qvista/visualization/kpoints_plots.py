"""
K-point mesh figure.

The k-points themselves, where they actually sit: fractional coordinates
carried through the reciprocal lattice into inverse angstrom, folded into
the first Brillouin zone, and drawn inside the zone's wireframe.

This used to be a lattice of dots on the integer axes ``i, j, k`` -- a
schematic of *how many* points there are along each direction, with the
zone drawn nowhere near it. It could not answer the questions the figure
is opened for: whether the mesh is denser along the short axis than the
long one (it should be, and on the index axes every mesh looks like a
box), where Gamma is, or what a Monkhorst-Pack mesh does differently from
a Gamma-centred one. All three are read straight off this.

The zone comes from :mod:`qvista.visualization.brillouin_plots`, so the
polyhedron here and the one drawn under a k-path are the same object
built the same way.
"""

from __future__ import annotations

from qvista.analyses.kpoints import GAMMA_SCHEME, mesh_points

#: Past this many k-points the scatter is a solid block of colour that
#: takes longer to draw than it does to read. Above it the mesh is drawn
#: thinned, and the title says so.
MAX_DRAWN = 6000

GAMMA_SYMBOL = "Γ"

POINT_COLOUR = "#2980b9"
GAMMA_COLOUR = "#c0392b"
ZONE_COLOUR = "#95a5a6"


def draw_kpoint_grid(
    figure,
    nx: int,
    ny: int,
    nz: int,
    lattice=None,
    scheme: str = GAMMA_SCHEME,
    title: str = "",
) -> None:
    """
    Draw the ``nx`` x ``ny`` x ``nz`` mesh in reciprocal space.

    ``lattice`` is the real-space lattice the mesh belongs to -- a
    pymatgen ``Lattice``, or anything carrying ``reciprocal_lattice``.
    Without it there is no reciprocal space to place the points in, and
    the mesh is drawn on its fractional axes instead, which is the
    honest fallback rather than a zone invented from nothing.
    """

    import numpy as np

    figure.clear()

    axes = figure.add_subplot(111, projection="3d")

    grid = (max(1, int(nx)), max(1, int(ny)), max(1, int(nz)))

    fractional = mesh_points(grid, scheme)

    total = int(np.prod(grid))

    basis = _reciprocal_basis(lattice)

    if basis is None:
        points = fractional
        names = ("$k_1$", "$k_2$", "$k_3$")
        unit = "fractional"
    else:
        points = _fold_into_zone(fractional @ basis, basis)
        names = ("$k_x$", "$k_y$", "$k_z$")
        unit = "1/Å"

        _draw_zone(axes, basis)

    # ----------------------------------------------------------
    # The k-points
    # ----------------------------------------------------------

    shown = points

    thinned = len(points) > MAX_DRAWN

    if thinned:
        # An even stride rather than a random sample, so what is drawn is
        # still a regular mesh and still reads as one.
        stride = int(np.ceil(len(points) / MAX_DRAWN))
        shown = points[::stride]

    axes.scatter(
        shown[:, 0],
        shown[:, 1],
        shown[:, 2],
        s=14 if len(shown) > 400 else 26,
        color=POINT_COLOUR,
        depthshade=False,
        alpha=0.85 if len(shown) > 400 else 1.0,
    )

    # ----------------------------------------------------------
    # Gamma
    #
    # Whether the mesh has a point on the origin is the difference
    # between the two schemes, so the origin is always marked and the
    # eye can check it against the dots.
    # ----------------------------------------------------------

    axes.scatter(
        [0.0], [0.0], [0.0],
        s=55,
        color=GAMMA_COLOUR,
        marker="x",
        depthshade=False,
    )

    axes.text(0.0, 0.0, 0.0, f"  {GAMMA_SYMBOL}", color=GAMMA_COLOUR, fontsize=9)

    # ----------------------------------------------------------
    # Framing
    # ----------------------------------------------------------

    _equalise(axes, points, basis)

    axes.set_xlabel(f"{names[0]} ({unit})")
    axes.set_ylabel(f"{names[1]} ({unit})")
    axes.set_zlabel(f"{names[2]} ({unit})")

    heading = title or f"{grid[0]} x {grid[1]} x {grid[2]} mesh"

    if thinned:
        heading = (
            f"{heading}\n{len(shown)} of {total} k-points drawn"
        )

    axes.set_title(heading)

    figure.tight_layout()


# ------------------------------------------------------------


def _reciprocal_basis(lattice):
    """
    The reciprocal basis as rows, in inverse angstrom, or ``None``.

    Accepts a lattice or a structure, because both are to hand at the
    call sites and asking each one to reach for the other is a line of
    ceremony at every call.
    """

    if lattice is None:
        return None

    import numpy as np

    # A structure carries its lattice; a lattice is its own.
    lattice = getattr(lattice, "lattice", lattice)

    reciprocal = getattr(lattice, "reciprocal_lattice", None)

    if reciprocal is None:
        return None

    return np.asarray(reciprocal.matrix, dtype=float)


def _fold_into_zone(points, basis):
    """
    Move each k-point to the copy of it nearest the origin.

    Folding the fractional coordinates into ``[-0.5, 0.5)`` puts the
    mesh in the reciprocal *unit cell* -- a parallelepiped. The zone
    drawn around it is the Wigner-Seitz cell, and for anything but a
    cubic lattice the two are different shapes of the same volume. Drawn
    together and unfolded, the mesh spills out of the hexagon on one
    side and leaves its corners empty on the other, which reads as the
    figure being wrong rather than as the two regions being equivalent.

    A k-point and that k-point plus any reciprocal lattice vector are
    the same state, so the nearest copy to the origin is the one inside
    the zone. Searching the 27 vectors of the first shell is enough for
    any cell whose Wigner-Seitz cell is reached by its own neighbours,
    which is every cell a Brillouin zone is drawn for here.
    """

    import numpy as np

    points = np.asarray(points, dtype=float)
    basis = np.asarray(basis, dtype=float)

    if points.size == 0:
        return points

    shifts = np.array(
        [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
        ],
        dtype=float,
    ) @ basis

    # (points, shifts, 3) -> the distance of every candidate copy.
    candidates = points[:, None, :] - shifts[None, :, :]

    nearest = np.argmin((candidates ** 2).sum(axis=2), axis=1)

    return candidates[np.arange(len(points)), nearest]


def _draw_zone(axes, basis) -> None:
    """Put the first Brillouin zone's wireframe around the mesh."""

    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    from qvista.visualization.brillouin_plots import brillouin_facets

    try:
        facets = brillouin_facets(basis)

    except Exception:
        # A degenerate cell has no Wigner-Seitz cell to speak of. The
        # points are still worth drawing without it.
        return

    edges = []

    for facet in facets:

        points = [np.asarray(point, dtype=float) for point in facet]

        for start, end in zip(points, points[1:] + points[:1]):
            edges.append((start, end))

    axes.add_collection3d(
        Line3DCollection(edges, colors=ZONE_COLOUR, linewidths=0.9)
    )


def _equalise(axes, points, basis=None) -> None:
    """
    One scale on all three axes, framing the points and the zone.

    Reciprocal space is where anisotropy matters most -- a slab has a
    long axis in real space and so a short one here -- and stretched
    axes would hide exactly the thing the figure is being read for.

    The zone is measured as well as the points, and not inferred from
    them: a line collection does not contribute to autoscaling, so a
    zone framed only by the k-points inside it would have its corners
    cut off.
    """

    import numpy as np

    points = np.asarray(points, dtype=float)

    reach = [float(np.abs(points).max())] if points.size else []

    if basis is not None:
        # The Wigner-Seitz cell reaches at most half way along any sum
        # of the basis vectors.
        reach.append(
            0.5 * float(np.abs(np.asarray(basis, dtype=float)).sum(axis=0).max())
        )

    limit = 1.15 * (max(reach) if reach else 1.0)

    if limit <= 0.0:
        limit = 1.0

    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_zlim(-limit, limit)

    try:
        axes.set_box_aspect((1.0, 1.0, 1.0))

    except Exception:
        # Older matplotlib without set_box_aspect on 3D axes.
        pass

"""
Tests for the k-point mesh figure.

The figure used to be a lattice of dots on the integer axes ``i, j, k``,
with the Brillouin zone drawn nowhere near it. On those axes every mesh
is a box: a 12 x 12 x 2 slab mesh and a 12 x 12 x 12 bulk one differ only
in how far the box goes back, and neither says anything about reciprocal
space. The questions the figure is opened for -- is the sampling denser
along the short axis, is there a point on Gamma, does the mesh fill the
zone -- could none of them be answered from it.

So what is asserted here is that the points are in reciprocal space: at
the reciprocal lattice's own scale, anisotropic where the cell is, and
surrounded by the zone they belong to.
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pymatgen.core import Lattice, Structure

from qvista.analyses import kpoints as kp
from qvista.visualization import kpoints_plots as kpp


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------


def cubic(a: float = 5.0):

    return Structure(Lattice.cubic(a), ["Si"], [[0.0, 0.0, 0.0]])


def slab(a: float = 4.0, c: float = 25.0):

    return Structure(
        Lattice.from_parameters(a, a, c, 90.0, 90.0, 90.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )


def drawn(axes) -> np.ndarray:
    """
    The k-points, as the scatter holds them.

    The first collection is the mesh; the second is the single cross on
    the origin. A line collection is the zone, and carries no offsets.
    """

    for collection in axes.collections:

        if isinstance(collection, Line3DCollection):
            continue

        offsets = np.asarray(collection._offsets3d, dtype=float).T

        if len(offsets) > 1:
            return offsets

    raise AssertionError("no k-points were drawn")


def zone_segments(axes) -> list:
    """
    The Brillouin zone's edges, in data coordinates.

    Read from ``_segments3d`` because there is no public getter for
    them: ``get_segments`` returns the *projected* 2D segments, which
    are empty until the figure is rendered and in screen coordinates
    afterwards -- so a test written against it either sees nothing or
    sees the wrong numbers.
    """

    for collection in axes.collections:

        if isinstance(collection, Line3DCollection):
            return list(getattr(collection, "_segments3d", []))

    return []


# ------------------------------------------------------------
# Where the points are
# ------------------------------------------------------------


def test_the_points_are_in_reciprocal_space() -> None:
    """
    At the reciprocal lattice's scale, not on the integer axes.

    A 4 x 4 x 4 mesh of a 5 angstrom cell reaches half of ``|b|``, about
    0.63 -- not 3, which is where index axes would put it.
    """

    structure = cubic(5.0)

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 4, 4, 4, lattice=structure)

    points = drawn(figure.axes[0])

    reach = float(np.abs(points).max())

    half = 0.5 * structure.lattice.reciprocal_lattice.abc[0]

    assert reach == pytest.approx(half, rel=0.05)


def test_the_mesh_is_anisotropic_where_the_cell_is() -> None:
    """
    The question the old figure could not answer.

    A slab has a long real-space axis and so a short reciprocal one, and
    the points along it have to be visibly closer together.
    """

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 6, 6, 2, lattice=slab())

    points = drawn(figure.axes[0])

    in_plane = float(np.abs(points[:, 0]).max())
    out_of_plane = float(np.abs(points[:, 2]).max())

    assert out_of_plane < in_plane


def test_the_points_match_the_mesh_the_generator_produced() -> None:
    """The figure draws the mesh, not something else shaped like it."""

    structure = cubic(6.0)

    grid = (3, 4, 2)

    figure = Figure()

    kpp.draw_kpoint_grid(
        figure, *grid, lattice=structure, scheme=kp.GAMMA_SCHEME
    )

    points = drawn(figure.axes[0])

    basis = np.asarray(
        structure.lattice.reciprocal_lattice.matrix, dtype=float
    )

    # Taken back to fractional coordinates and compared modulo one,
    # because the points are folded into the zone on the way out and a
    # fold is a whole reciprocal lattice vector. Comparing there is
    # what makes this a check on the mesh rather than on the folding.
    back = np.linalg.solve(basis.T, points.T).T % 1.0

    expected = kp.mesh_points(grid, kp.GAMMA_SCHEME) % 1.0

    assert sorted(map(tuple, back.round(6))) == sorted(
        map(tuple, expected.round(6))
    )


def test_every_subdivision_is_drawn() -> None:

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 5, 4, 3, lattice=cubic())

    assert len(drawn(figure.axes[0])) == 60


# ------------------------------------------------------------
# The zone around them
# ------------------------------------------------------------


def test_the_brillouin_zone_is_drawn() -> None:
    """
    The mesh is meaningless without the zone it samples: "is this
    enough" is a question about how the points fill it.
    """

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 4, 4, 4, lattice=cubic())

    segments = zone_segments(figure.axes[0])

    # A cube's Wigner-Seitz cell is a cube: six faces, four edges each.
    assert len(segments) == 24


def test_the_zone_is_inside_the_frame() -> None:
    """
    A line collection does not autoscale, so the limits have to be set
    against the zone as well as the points -- or its corners are cut
    off, which is what a Wigner-Seitz cell is recognised by.
    """

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 2, 2, 2, lattice=cubic(5.0))

    axes = figure.axes[0]

    corners = np.asarray(
        [
            point
            for segment in zone_segments(axes)
            for point in segment
        ],
        dtype=float,
    )

    assert corners.size

    for limits, column in zip(
        (axes.get_xlim(), axes.get_ylim(), axes.get_zlim()),
        corners.T,
    ):
        assert limits[0] <= column.min()
        assert limits[1] >= column.max()


def test_the_mesh_is_folded_into_the_zone_not_the_unit_cell() -> None:
    """
    A hexagonal cell is where the two come apart.

    Folding the fractional coordinates into ``[-0.5, 0.5)`` fills the
    reciprocal unit cell, a parallelepiped. The zone drawn around it is
    the Wigner-Seitz cell, a hexagonal prism of the same volume and a
    different shape -- so the unfolded mesh spills out one side and
    leaves the corners of the hexagon empty, which reads as the figure
    being wrong.

    Every point must be at least as close to the origin as to any other
    reciprocal lattice point, which is what being inside the zone means.
    """

    structure = Structure(
        Lattice.from_parameters(3.2, 3.2, 5.2, 90.0, 90.0, 120.0),
        ["Zn"],
        [[0.0, 0.0, 0.0]],
    )

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 6, 6, 4, lattice=structure)

    points = drawn(figure.axes[0])

    basis = np.asarray(
        structure.lattice.reciprocal_lattice.matrix, dtype=float
    )

    neighbours = np.array(
        [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)
        ],
        dtype=float,
    ) @ basis

    own = (points ** 2).sum(axis=1)

    for lattice_point in neighbours:

        other = ((points - lattice_point) ** 2).sum(axis=1)

        assert np.all(own <= other + 1e-9)


def test_folding_moves_points_without_losing_them() -> None:
    """A fold is a relabelling: the same number of distinct k-points."""

    structure = Structure(
        Lattice.from_parameters(3.2, 3.2, 5.2, 90.0, 90.0, 120.0),
        ["Zn"],
        [[0.0, 0.0, 0.0]],
    )

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 5, 5, 3, lattice=structure)

    points = drawn(figure.axes[0])

    assert len(points) == 75
    assert len(np.unique(points.round(6), axis=0)) == 75


def test_the_axes_are_equally_scaled() -> None:
    """A cubic zone has to look cubic, or the geometry is unreadable."""

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 4, 4, 1, lattice=slab())

    axes = figure.axes[0]

    spans = [
        limits[1] - limits[0]
        for limits in (axes.get_xlim(), axes.get_ylim(), axes.get_zlim())
    ]

    assert max(spans) == pytest.approx(min(spans))


# ------------------------------------------------------------
# Schemes, size and the fallback
# ------------------------------------------------------------


def test_the_two_schemes_are_drawn_differently() -> None:
    """
    Whether a point sits on Gamma is the difference between them, and
    the only way to see it is to draw the points.
    """

    figure = Figure()
    kpp.draw_kpoint_grid(figure, 4, 4, 4, lattice=cubic(), scheme=kp.GAMMA_SCHEME)
    centred = drawn(figure.axes[0])

    figure = Figure()
    kpp.draw_kpoint_grid(figure, 4, 4, 4, lattice=cubic(), scheme=kp.MP_SCHEME)
    straddling = drawn(figure.axes[0])

    assert (np.abs(centred).sum(axis=1) < 1e-9).any()
    assert not (np.abs(straddling).sum(axis=1) < 1e-9).any()


def test_a_huge_mesh_is_thinned_and_says_so() -> None:
    """
    A hundred thousand points is a solid block of colour that takes
    longer to draw than to read. Thinning it is fine; doing so silently
    is not, because the figure would misreport the mesh.
    """

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 40, 40, 40, lattice=cubic())

    axes = figure.axes[0]

    assert len(drawn(axes)) <= kpp.MAX_DRAWN
    assert "64000" in axes.get_title()


def test_a_mesh_under_the_cap_is_drawn_whole() -> None:

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 10, 10, 10, lattice=cubic())

    assert len(drawn(figure.axes[0])) == 1000
    assert "drawn" not in figure.axes[0].get_title()


def test_without_a_lattice_the_points_are_fractional() -> None:
    """
    The honest fallback.

    There is no reciprocal space to place the points in and no zone to
    draw, so the axes say so rather than inventing one.
    """

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 3, 3, 3, lattice=None)

    axes = figure.axes[0]

    assert "fractional" in axes.get_xlabel()

    assert not zone_segments(axes)

    assert float(np.abs(drawn(axes)).max()) <= 0.5


def test_a_lattice_may_be_given_instead_of_a_structure() -> None:
    """Both are to hand at the call sites; neither should need unwrapping."""

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 3, 3, 3, lattice=Lattice.cubic(5.0))

    assert len(drawn(figure.axes[0])) == 27


def test_a_one_by_one_by_one_mesh_is_just_gamma() -> None:
    """The degenerate case still has to draw, and frame, something."""

    figure = Figure()

    kpp.draw_kpoint_grid(figure, 1, 1, 1, lattice=cubic())

    axes = figure.axes[0]

    assert axes.get_xlim()[1] > 0.0

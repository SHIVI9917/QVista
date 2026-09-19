"""
Tests for k-point mesh generation.

The whole of this module turns on one number, ``2*pi``, and on the fact
that "k-spacing 0.02" names two different things depending on who is
saying it.

**VASP's KSPACING tag** is in inverse angstrom with the ``2*pi`` in it.
VASP's own default is 0.5 and pymatgen warns above it, so a KSPACING of
0.02 would be twenty-five times finer than VASP's default -- which on any
real cell is a mesh nobody asked for and nothing can afford.

**A k-point resolution** is in units of ``2*pi/angstrom``. 0.06 is coarse
and 0.01 is fine, and the spacing it asks for is ``2*pi`` times the
number quoted.

Reading one as the other is a factor of ``2*pi`` in the subdivisions
along every axis, and so a factor of about 250 in the number of
k-points. That is the failure guarded here: the two rules are tested
separately, against each other, and against the physical spacing each
one claims to deliver.
"""

from __future__ import annotations

import math

import pytest

from pymatgen.core import Lattice, Structure

from qvista.analyses import kpoints as kp


# ------------------------------------------------------------
# Cells
# ------------------------------------------------------------


def cubic(a: float = 5.0):

    return Structure(Lattice.cubic(a), ["Si"], [[0.0, 0.0, 0.0]])


def slab(a: float = 4.0, c: float = 25.0):
    """A slab: two short axes and one long one, with vacuum along c."""

    return Structure(
        Lattice.from_parameters(a, a, c, 90.0, 90.0, 90.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )


# ------------------------------------------------------------
# The resolution rule
# ------------------------------------------------------------


def test_the_resolution_is_in_units_of_two_pi() -> None:
    """
    Asking for 0.02 asks for k-points ``2*pi * 0.02`` apart.

    Stated against a cell whose answer is exact -- 10 angstrom cubic
    gives ``|b| = 2*pi/10``, and ``|b| / (2*pi * 0.02)`` is 5 with
    nothing to round -- so the assertion is about the convention and not
    about rounding.
    """

    structure = cubic(10.0)

    assert kp.grid_from_resolution(structure, 0.02) == (5, 5, 5)

    spacing = kp.achieved_spacing(structure, (5, 5, 5))

    assert spacing[0] == pytest.approx(kp.TWO_PI * 0.02)


def test_a_bigger_cell_needs_fewer_k_points() -> None:
    """
    The reason the bug was noticed: large cells came out dense.

    Reciprocal space shrinks as the cell grows, so doubling every axis
    halves the subdivisions. Anything that returns a large mesh for a
    large cell has the convention inverted somewhere.
    """

    small = kp.grid_from_resolution(cubic(5.0), 0.02)
    large = kp.grid_from_resolution(cubic(20.0), 0.02)

    assert small > large

    for a in (5.0, 10.0, 20.0, 40.0):
        # Never absurd: 40 angstrom at a fine resolution is still a
        # handful of points, not thousands.
        assert max(kp.grid_from_resolution(cubic(a), 0.02)) <= 12


def test_a_finer_resolution_is_a_denser_mesh() -> None:

    structure = cubic(8.0)

    counts = [
        kp.grid_from_resolution(structure, value)[0]
        for value in kp.RESOLUTION_PRESETS
    ]

    assert counts == sorted(counts)


def test_the_mesh_follows_the_shape_of_the_cell() -> None:
    """
    A slab is sampled densely in-plane and hardly at all along the
    vacuum. On the index axes every mesh looks like a box, which is
    what made the old figure useless; the numbers say otherwise.
    """

    nx, ny, nz = kp.grid_from_resolution(slab(4.0, 25.0), 0.03)

    assert nx == ny
    assert nz < nx


def test_a_long_enough_axis_needs_one_point() -> None:
    """Gamma alone along an axis is a real answer, not a failure."""

    assert kp.grid_from_resolution(slab(4.0, 60.0), 0.04)[2] == 1


# ------------------------------------------------------------
# VASP's KSPACING rule
# ------------------------------------------------------------


def test_kspacing_is_vasps_own_rule() -> None:
    """``N_i = ceil(|b_i| / KSPACING)``, with ``|b_i|`` carrying 2*pi."""

    structure = cubic(10.0)

    length = structure.lattice.reciprocal_lattice.abc[0]

    for kspacing in (0.5, 0.3, 0.15):

        expected = max(1, math.ceil(length / kspacing))

        assert kp.grid_from_kspacing(structure, kspacing) == (
            expected, expected, expected
        )


def test_kspacing_never_returns_a_coarser_mesh_than_asked_for() -> None:
    """
    Ceiling and not rounding.

    Rounding down would deliver a mesh coarser than the number asks
    for, which is the one direction a convergence parameter must not
    fail in.
    """

    for a in (3.0, 5.5, 7.3, 12.0):
        for kspacing in (0.5, 0.35, 0.2):

            structure = cubic(a)
            grid = kp.grid_from_kspacing(structure, kspacing)

            for got in kp.achieved_spacing(structure, grid):
                assert got <= kspacing + 1e-12


def test_vasps_default_is_a_coarse_mesh() -> None:
    """
    A sanity check on the units.

    KSPACING 0.5 is VASP's default and pymatgen warns above it, so it
    has to come out as a handful of points for an ordinary cell. If
    this ever returns a large mesh the 2*pi has moved.
    """

    assert max(kp.grid_from_kspacing(cubic(5.0), 0.5)) <= 4


# ------------------------------------------------------------
# The two against each other
# ------------------------------------------------------------


def test_the_conventions_differ_by_exactly_two_pi() -> None:
    """
    The same physical mesh, asked for two ways.

    A resolution of ``R`` and a KSPACING of ``2*pi * R`` mean the same
    spacing, so they must agree up to the rounding each one does.
    """

    for a in (4.0, 6.0, 9.0):
        for resolution in (0.04, 0.02):

            structure = cubic(a)

            by_resolution = kp.grid_from_resolution(structure, resolution)
            by_kspacing = kp.grid_from_kspacing(
                structure, kp.TWO_PI * resolution
            )

            # Rounding against ceiling, so at most one subdivision.
            for left, right in zip(by_resolution, by_kspacing):
                assert abs(left - right) <= 1


def test_reading_a_resolution_as_a_kspacing_is_the_old_bug() -> None:
    """
    A regression guard with the failure written into it.

    0.02 read as a KSPACING gave 31 subdivisions on a 10 angstrom cell
    -- nearly thirty thousand k-points, and the report that started
    this. The two rules must not agree on that number.
    """

    structure = cubic(10.0)

    assert kp.grid_from_resolution(structure, 0.02) == (5, 5, 5)
    assert kp.grid_from_kspacing(structure, 0.02)[0] > 30


def test_the_dispatcher_picks_the_named_rule() -> None:

    structure = cubic(7.0)

    assert kp.grid_for(
        structure, 0.03, kp.KP_RESOLUTION
    ) == kp.grid_from_resolution(structure, 0.03)

    assert kp.grid_for(
        structure, 0.3, kp.VASP_KSPACING
    ) == kp.grid_from_kspacing(structure, 0.3)


def test_an_unnamed_convention_is_refused() -> None:
    """
    Not defaulted quietly.

    A silent fallback would put the factor of 2*pi back in the hands of
    whichever caller forgot to say which it meant.
    """

    with pytest.raises(ValueError):
        kp.grid_for(cubic(), 0.02, "whatever")


@pytest.mark.parametrize("spacing", [0.0, -0.1])
def test_a_spacing_of_zero_or_less_is_refused(spacing: float) -> None:
    """It would divide by zero, or return a negative subdivision."""

    with pytest.raises(ValueError):
        kp.grid_from_resolution(cubic(), spacing)

    with pytest.raises(ValueError):
        kp.grid_from_kspacing(cubic(), spacing)


# ------------------------------------------------------------
# What the mesh delivers
# ------------------------------------------------------------


def test_the_achieved_spacing_is_the_axis_divided_by_the_count() -> None:

    structure = slab()

    lengths = kp.reciprocal_lengths(structure)
    spacing = kp.achieved_spacing(structure, (4, 4, 1))

    assert spacing[0] == pytest.approx(lengths[0] / 4)
    assert spacing[2] == pytest.approx(lengths[2])


def test_the_reciprocal_lengths_carry_two_pi() -> None:
    """
    The convention every rule here is written against.

    A cubic cell of side ``a`` has ``|b| = 2*pi/a``. Getting ``1/a``
    instead is precisely the bug this module exists to have fixed.
    """

    assert kp.reciprocal_lengths(cubic(10.0))[0] == pytest.approx(
        kp.TWO_PI / 10.0
    )


# ------------------------------------------------------------
# The points themselves
# ------------------------------------------------------------


def test_a_gamma_centred_mesh_has_a_point_on_gamma() -> None:
    """That is the whole of what the scheme name means."""

    import numpy as np

    points = kp.mesh_points((4, 4, 4), kp.GAMMA_SCHEME)

    assert (np.abs(points).sum(axis=1) < 1e-12).any()


def test_an_even_monkhorst_pack_mesh_straddles_gamma() -> None:
    """
    Which is why the two schemes give different answers on the same
    grid, and why the figure has to draw the points rather than count
    them.
    """

    import numpy as np

    points = kp.mesh_points((4, 4, 4), kp.MP_SCHEME)

    assert not (np.abs(points).sum(axis=1) < 1e-12).any()


def test_an_odd_monkhorst_pack_mesh_includes_gamma() -> None:
    """The odd case is Gamma-centred, which is the usual reason to pick it."""

    import numpy as np

    points = kp.mesh_points((3, 3, 3), kp.MP_SCHEME)

    assert (np.abs(points).sum(axis=1) < 1e-12).any()


@pytest.mark.parametrize("scheme", [kp.GAMMA_SCHEME, kp.MP_SCHEME])
def test_every_point_is_in_the_first_zone(scheme: str) -> None:
    """
    Folded, so the mesh sits around the origin rather than running off
    into the next cell -- which is what lets it be drawn inside the
    Brillouin zone.
    """

    import numpy as np

    points = kp.mesh_points((5, 4, 3), scheme)

    assert np.all(points >= -0.5 - 1e-12)
    assert np.all(points < 0.5 + 1e-12)


@pytest.mark.parametrize("scheme", [kp.GAMMA_SCHEME, kp.MP_SCHEME])
def test_there_is_one_point_per_subdivision(scheme: str) -> None:
    """Folding must not merge two points into one."""

    import numpy as np

    grid = (5, 4, 3)

    points = kp.mesh_points(grid, scheme)

    assert len(points) == 5 * 4 * 3
    assert len(np.unique(points.round(9), axis=0)) == len(points)


# ------------------------------------------------------------
# The file
# ------------------------------------------------------------


def test_the_kpoints_file_is_an_explicit_zero_shift_mesh() -> None:

    text = kp.kpoints_content(4, 4, 2, kp.GAMMA_SCHEME)

    lines = text.splitlines()

    assert lines[1] == "0"
    assert lines[2] == kp.GAMMA_SCHEME
    assert lines[3] == "4 4 2"
    assert lines[4] == "0 0 0"

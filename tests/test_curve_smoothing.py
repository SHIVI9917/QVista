"""
Tests for the B-spline every QVista curve is drawn through.

This replaced the Gaussian smearing the COHP window used to offer, and
the two are not the same kind of operation at all. Convolution *moves*
the data: it lowers peaks, fills valleys, and merges two nearby states
into one when sigma is set past the gap between them. Interpolation does
not -- the spline passes through every calculated point, and all that
changes is how many points the line is drawn with in between.

So the properties worth pinning down are the ones that say the figure
still describes the calculation:

**Every calculated point survives, exactly.** A peak keeps its height
and its position, which is what makes it safe to read a barrier or a
band edge off a splined line.

**Area is conserved.** The integral of a COHP over the occupied states
*is* the bond strength quoted in the table beside it, and a curve drawn
with a different area than the one in the table is a figure disagreeing
with its own caption.

**A straight line stays straight.** A cubic through collinear points is
that line, so nothing is invented where there was nothing to smooth.

**It refuses rather than guesses.** Too few points, a repeated abscissa
or a non-finite value comes back untouched, because a band structure's
distances repeat at every corner of the k-path and interpolating across
one would join two branches that are not connected.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.visualization.smoothing import (
    DEGREE,
    DENSITY,
    MAX_POINTS,
    refine,
    runs,
    spline,
    spline_runs,
)


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------

STEP = 0.005


@pytest.fixture
def energies() -> np.ndarray:
    """A LOBSTER-like energy grid, finer than the drawn figure."""

    return np.arange(-10.0, 6.0 + STEP, STEP)


def spike(grid: np.ndarray, at: float = -2.0) -> np.ndarray:
    """One resonance, three points wide -- the case that shows corners."""

    values = np.zeros_like(grid)

    values[np.argmin(np.abs(grid - at))] = 1.0

    return values


def coarse() -> tuple[np.ndarray, np.ndarray]:
    """A grid coarse enough that the spline actually does something."""

    x = np.linspace(0.0, 10.0, 21)

    return x, np.sin(x)


# ------------------------------------------------------------
# The calculated points survive
# ------------------------------------------------------------


def test_every_calculated_point_is_still_on_the_curve() -> None:
    """
    The whole difference from a broadening.

    The dense grid is built by subdividing each interval, so the
    original abscissa is a subset of it and the values there are the
    values that were calculated.
    """

    x, y = coarse()

    grid, values = spline(x, y)

    for point, value in zip(x, y):

        index = int(np.argmin(np.abs(grid - point)))

        assert grid[index] == pytest.approx(point)
        assert values[index] == pytest.approx(value)


def test_a_peak_keeps_its_height_and_its_place() -> None:
    """
    Which is what a Gaussian would not have done.

    Convolving a three-point spike lowers it and widens it; that is the
    point of convolving. Here the maximum has to come back untouched.
    """

    x = np.linspace(0.0, 10.0, 41)

    y = np.exp(-((x - 4.0) ** 2) / 0.5)

    grid, values = spline(x, y)

    assert values.max() == pytest.approx(y.max())

    assert grid[int(np.argmax(values))] == pytest.approx(
        x[int(np.argmax(y))]
    )


def test_the_curve_gets_denser(energies) -> None:
    """Otherwise nothing has been smoothed."""

    x = np.linspace(0.0, 1.0, 30)

    grid, values = spline(x, np.sin(x * 6.0))

    assert grid.size > x.size
    assert values.size == grid.size


def test_the_density_is_the_stated_one() -> None:
    """One calculated gap becomes ``DENSITY`` drawn ones."""

    x = np.linspace(0.0, 1.0, 30)

    grid, _ = spline(x, np.sin(x))

    assert grid.size == (x.size - 1) * DENSITY + 1


# ------------------------------------------------------------
# What it must not change
# ------------------------------------------------------------


def test_area_is_conserved() -> None:
    """
    A COHP's integral below the Fermi level is the bond strength.

    Drawing a curve whose area disagrees with the number in the table
    beside it would be the figure contradicting its own caption.
    """

    x = np.linspace(-10.0, 6.0, 200)

    y = np.exp(-((x + 2.0) ** 2) / 0.4) - 0.5 * np.exp(
        -((x - 1.0) ** 2) / 0.3
    )

    grid, values = spline(x, y)

    assert float(np.trapezoid(values, grid)) == pytest.approx(
        float(np.trapezoid(y, x)), rel=1e-4
    )


def test_a_straight_line_stays_straight() -> None:
    """A cubic through collinear points is that line, not a wiggle."""

    x = np.linspace(0.0, 5.0, 12)

    grid, values = spline(x, 3.0 * x - 1.0)

    assert np.allclose(values, 3.0 * grid - 1.0)


def test_a_constant_stays_constant() -> None:
    """
    Which is what keeps a fat band's widths honest.

    A series weighted the same everywhere has to be drawn at one width,
    not at a width that ripples along the path.
    """

    x = np.linspace(0.0, 1.0, 40)

    _, values = spline(x, np.full_like(x, 0.15))

    assert np.allclose(values, 0.15)


def test_the_ends_are_not_dragged_anywhere() -> None:
    """
    A window is a view onto a longer curve.

    The old convolution had to pad the ends to stop them sagging towards
    the axis. Interpolation has no such problem: the first and last
    calculated points are endpoints of the spline.
    """

    x, y = coarse()

    grid, values = spline(x, y)

    assert grid[0] == pytest.approx(x[0])
    assert grid[-1] == pytest.approx(x[-1])

    assert values[0] == pytest.approx(y[0])
    assert values[-1] == pytest.approx(y[-1])


# ------------------------------------------------------------
# When it refuses
# ------------------------------------------------------------


def test_too_few_points_come_back_untouched() -> None:
    """A cubic needs four; three points are not a curve."""

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])

    grid, values = spline(x, y)

    assert np.array_equal(grid, x)
    assert np.array_equal(values, y)


def test_a_repeated_abscissa_is_refused() -> None:
    """
    Which is exactly a corner in a k-path.

    pymatgen writes the same distance twice where the path turns, and a
    spline across that would draw a line between two branches that are
    not connected.
    """

    x = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 5.0, 6.0, 7.0, 8.0])

    grid, values = spline(x, y)

    assert np.array_equal(grid, x)
    assert np.array_equal(values, y)


def test_a_non_finite_value_is_refused() -> None:
    """A NaN would propagate across the whole spline, not one point."""

    x = np.linspace(0.0, 1.0, 10)

    y = np.ones_like(x)
    y[4] = np.nan

    grid, values = spline(x, y)

    assert np.array_equal(grid, x)
    assert np.array_equal(values, y, equal_nan=True)


def test_an_already_dense_grid_is_left_alone(energies) -> None:
    """
    A DOSCAR does not need eight times as many points to look smooth.

    Refining it anyway would make every figure slow to redraw for no
    visible difference.
    """

    grid, values = spline(energies, spike(energies))

    assert grid.size == energies.size

    assert np.array_equal(values, spike(energies))


def test_the_result_is_capped() -> None:
    """However coarse the input, the output stays a drawable size."""

    x = np.linspace(0.0, 1.0, 900)

    grid, _ = spline(x, np.sin(x))

    assert grid.size <= MAX_POINTS


# ------------------------------------------------------------
# Descending, two dimensional, and split
# ------------------------------------------------------------


def test_a_descending_abscissa_works_and_stays_descending() -> None:
    """Some curves are handed over from high energy down."""

    x = np.linspace(5.0, 0.0, 20)

    grid, values = spline(x, x**2)

    assert grid[0] > grid[-1]

    assert np.allclose(values, grid**2, atol=1e-6)


def test_several_curves_share_one_grid() -> None:
    """
    Which is what keeps a fat band in step.

    Its energies and its weights are splined together, so the width at
    a point belongs to the energy at that same point.
    """

    x = np.linspace(0.0, 1.0, 20)

    stacked = np.column_stack([np.sin(x), np.cos(x)])

    grid, values = spline(x, stacked)

    assert values.shape == (grid.size, 2)

    assert np.allclose(values[:, 0], np.sin(grid), atol=1e-5)
    assert np.allclose(values[:, 1], np.cos(grid), atol=1e-5)


def test_a_path_is_split_at_its_corners() -> None:
    """One stretch per straight run of the abscissa."""

    x = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])

    assert runs(x) == [(0, 3), (3, 6)]


def test_an_unbroken_path_is_one_stretch() -> None:

    assert runs(np.linspace(0.0, 1.0, 10)) == [(0, 10)]


def test_splining_a_path_yields_one_curve_per_stretch() -> None:

    x = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    y = np.arange(6.0)

    pieces = list(spline_runs(x, y))

    assert len(pieces) == 2

    for grid, _ in pieces:
        assert np.all(np.diff(grid) > 0.0)


def test_an_empty_curve_yields_nothing() -> None:

    assert runs(np.array([])) == []


# ------------------------------------------------------------
# The dense abscissa itself
# ------------------------------------------------------------


def test_refining_keeps_an_uneven_grid_uneven() -> None:
    """
    A DOS written more finely near the Fermi level stays that way.

    Spreading points evenly over the whole range instead would level it
    out, and sample the interesting region no better than the rest.
    """

    x = np.array([0.0, 0.1, 0.2, 1.0, 5.0])

    grid = refine(x, density=4, max_points=1000)

    # Every original point is still there.
    for point in x:
        assert np.isclose(grid, point).any()

    first = np.diff(grid[:4]).mean()
    last = np.diff(grid[-4:]).mean()

    assert last > first


def test_refining_backs_off_rather_than_truncating() -> None:
    """The whole range is always covered, however tight the budget."""

    x = np.linspace(0.0, 1.0, 300)

    grid = refine(x, density=DENSITY, max_points=600)

    assert grid[0] == pytest.approx(x[0])
    assert grid[-1] == pytest.approx(x[-1])

    assert grid.size <= 600


def test_a_cubic_is_what_is_used() -> None:
    """The lowest order with a continuous second derivative."""

    assert DEGREE == 3

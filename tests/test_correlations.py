"""
Tests for the pair and time correlations.

Each of these has a value it is *required* to take, which is what makes
them testable at all: an RDF tends to one at large r by construction, a
Van Hove self part integrates to one at every lag, and a survival
correlation starts at exactly one. A curve of plausible shape that
misses those is normalised wrongly, and nothing about the picture says
so.

The other thing guarded here is the minimum image. Every distance is
taken across periodic boundaries, and the usual shortcut -- rounding
each Cartesian component against the cell edge -- is correct only for an
orthorhombic cell and silently wrong for anything else. A hexagonal cell
is used deliberately in the test that checks it.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services import correlations as co
from qvista.services.msd import Trajectory, TrajectoryError


# ------------------------------------------------------------
# Trajectories
# ------------------------------------------------------------


def gas(
    frames: int = 200,
    n_mobile: int = 16,
    n_fixed: int = 32,
    edge: float = 12.0,
    step: float = 0.10,
    seed: int = 0,
):
    """
    A random-walking species inside a rattling framework.

    Not a real material, but it has the two properties the tests need:
    the mobile species genuinely diffuses, and the density is uniform,
    so g(r) must approach one.
    """

    generator = np.random.default_rng(seed)

    lattice = np.eye(3) * edge

    fixed = generator.random((n_fixed, 3)) * edge
    mobile = generator.random((n_mobile, 3)) * edge

    positions = np.zeros((frames, n_mobile + n_fixed, 3))

    for frame in range(frames):

        mobile = mobile + generator.normal(0.0, step, (n_mobile, 3))

        positions[frame, :n_mobile] = mobile
        positions[frame, n_mobile:] = fixed + generator.normal(
            0.0, 0.03, (n_fixed, 3)
        )

    return Trajectory(
        positions=positions,
        symbols=("Li",) * n_mobile + ("O",) * n_fixed,
        lattice=lattice,
        timestep=0.002,
        temperature=900.0,
    )


def frozen(frames: int = 60, edge: float = 12.0):
    """
    A crystal that does not move at all.

    Three atoms in a line, 2.5 A apart. Every pair separation -- 2.5,
    2.5 and 5.0 A -- sits inside the 6 A half-cell cutoff, so all of
    them are counted and each atom has exactly two partners. Spread
    them further and the pairs fall outside the range, which makes the
    coordination number zero for a reason that has nothing to do with
    the code.
    """

    lattice = np.eye(3) * edge

    base = np.array(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [5.0, 0.0, 0.0]]
    )

    positions = np.repeat(base[None, :, :], frames, axis=0)

    return Trajectory(
        positions=positions,
        symbols=("Na",) * 3,
        lattice=lattice,
        timestep=0.001,
        temperature=300.0,
    )


# ------------------------------------------------------------
# Minimum image
# ------------------------------------------------------------


def test_a_displacement_across_the_boundary_is_short() -> None:

    lattice = np.eye(3) * 10.0

    wrapped = co.minimum_image(np.array([[9.0, 0.0, 0.0]]), lattice)

    assert np.allclose(wrapped, [[-1.0, 0.0, 0.0]])


def test_the_minimum_image_is_right_for_a_skewed_cell() -> None:
    """
    The test the usual shortcut fails.

    Rounding each Cartesian component against its own cell edge is
    correct only when the axes are orthogonal. In a hexagonal cell it
    gives an answer that is neither the shortest image nor a lattice
    translation away from the original.
    """

    from pymatgen.core import Lattice

    lattice = np.asarray(
        Lattice.from_parameters(
            6.0, 6.0, 6.0, 90.0, 90.0, 120.0
        ).matrix,
        dtype=float,
    )

    # A displacement of exactly one lattice vector must reduce to zero.
    for row in lattice:

        assert np.allclose(
            co.minimum_image(row[None, :], lattice), 0.0, atol=1e-9
        )


def test_no_reduced_displacement_is_longer_than_the_original() -> None:

    generator = np.random.default_rng(0)

    lattice = np.array(
        [[6.0, 0.0, 0.0], [-3.0, 5.196, 0.0], [0.0, 0.0, 9.0]]
    )

    delta = generator.normal(0.0, 8.0, (200, 3))

    reduced = co.minimum_image(delta, lattice)

    assert np.all(
        np.linalg.norm(reduced, axis=1)
        <= np.linalg.norm(delta, axis=1) + 1e-9
    )


# ------------------------------------------------------------
# RDF
# ------------------------------------------------------------


def test_g_of_r_tends_to_one() -> None:
    """
    The property that makes an RDF an RDF.

    It is normalised against a uniform gas of the same density, so far
    from any atom it has to be one. A curve of the right shape sitting
    at 0.3 or 3 is normalised wrongly, and the peaks are then in the
    right places with the wrong heights.
    """

    result = co.rdf(gas())

    for name, curve in result.curves.items():

        tail = curve[-30:].mean()

        assert 0.85 < tail < 1.15, f"{name} tail is {tail:.3f}"


def test_every_pair_is_computed_by_default() -> None:

    result = co.rdf(gas())

    assert set(result.curves) == {"Li-Li", "Li-O", "O-O"}


def test_a_named_pair_is_the_only_one_computed() -> None:

    result = co.rdf(gas(), pairs=[("Li", "O")])

    assert list(result.curves) == ["Li-O"]


def test_a_pair_that_is_not_there_is_refused() -> None:

    with pytest.raises(TrajectoryError, match="None of the requested"):
        co.rdf(gas(), pairs=[("Fe", "S")])


def test_the_range_stops_at_half_the_cell() -> None:
    """
    Beyond that an atom meets its own periodic image.

    The sampling sphere no longer fits inside the box, so the shell
    volumes are wrong and g(r) drifts away from one -- gradually, which
    is what makes it easy to miss.
    """

    trajectory = gas(edge=12.0)

    result = co.rdf(trajectory, r_max=100.0)

    assert result.r[-1] <= 6.0 + 1e-9


def test_a_shorter_range_is_honoured() -> None:

    result = co.rdf(gas(), r_max=4.0)

    assert result.r[-1] <= 4.0


def test_a_trajectory_with_no_cell_is_refused() -> None:

    trajectory = gas()
    trajectory.lattice = None

    with pytest.raises(TrajectoryError, match="needs the cell"):
        co.rdf(trajectory)


def test_the_coordination_number_counts_neighbours() -> None:
    """
    A frozen structure has an exactly known first shell.

    Three atoms in a cubic cell: each has two partners, so the
    coordination integrated past the last peak is two.
    """

    result = co.rdf(frozen(), pairs=[("Na", "Na")], skip=0.0)

    assert result.coordination["Na-Na"][-1] == pytest.approx(
        2.0, abs=0.05
    )


def test_skipping_everything_is_refused() -> None:

    with pytest.raises(TrajectoryError, match="leaves nothing"):
        co.rdf(gas(frames=10), skip=1.0)


def test_a_short_run_is_called_out() -> None:

    result = co.rdf(gas(frames=20))

    assert any("thin average" in note for note in result.notes)


# ------------------------------------------------------------
# Van Hove
# ------------------------------------------------------------


def test_the_self_part_is_a_normalised_distribution() -> None:
    """
    ``4 pi r^2 Gs`` integrates to one at every lag past zero.

    That is what makes Gs a probability density rather than a count,
    and it is the thing that goes wrong if the shell volumes are
    dropped from the normalisation -- which leaves a curve of exactly
    the right shape scaled by the number of atoms.
    """

    result = co.van_hove(gas(), "Li", lags=6)

    for index in range(1, len(result.t)):

        integral = np.trapezoid(
            4.0 * np.pi * result.r**2 * result.self_part[index],
            result.r,
        )

        assert integral == pytest.approx(1.0, abs=0.05)


def test_the_distinct_part_tends_to_one() -> None:
    """It is the RDF with time in it, and inherits its normalisation."""

    result = co.van_hove(gas(), "Li", lags=4)

    assert result.distinct[0][-30:].mean() == pytest.approx(
        1.0, abs=0.2
    )


def test_the_self_part_spreads_with_time() -> None:
    """
    The whole content of the figure.

    An atom is at the origin at t = 0 and further away later, so the
    mean of the radial distribution has to grow.
    """

    result = co.van_hove(gas(), "Li", lags=8)

    def mean_radius(row):
        weight = 4.0 * np.pi * result.r**2 * row
        return float(
            np.trapezoid(weight * result.r, result.r)
            / max(np.trapezoid(weight, result.r), 1e-12)
        )

    first = mean_radius(result.self_part[1])
    last = mean_radius(result.self_part[-1])

    assert last > first


def test_a_frozen_structure_does_not_spread() -> None:
    """The control: nothing moves, so Gs stays at the origin."""

    result = co.van_hove(frozen(), "Na", lags=4, skip=0.0)

    peak = result.r[np.argmax(result.self_part[-1])]

    assert peak < 0.5


def test_the_distinct_part_can_be_switched_off() -> None:
    """It is the expensive half: every pair at every lag."""

    result = co.van_hove(gas(), "Li", lags=4, distinct=False)

    assert result.self_part.size

    assert not result.has_distinct


def test_an_element_that_is_not_there_says_what_is() -> None:

    with pytest.raises(TrajectoryError, match="no Fe in this"):
        co.van_hove(gas(), "Fe")


def test_the_default_element_is_the_first() -> None:

    assert co.van_hove(gas(), lags=3).element == "Li"


# ------------------------------------------------------------
# Residence time
# ------------------------------------------------------------


def test_the_correlation_starts_at_one() -> None:
    """
    At zero lag every ion is exactly where it was.

    Anything else means the lag indexing is off by one, which shifts
    the whole curve and the fitted tau with it.
    """

    result = co.residence_time(gas(), "Li", cutoff=1.0)

    assert result.correlation[0] == pytest.approx(1.0)


def test_the_correlation_decays_for_a_mobile_ion() -> None:

    result = co.residence_time(gas(), "Li", cutoff=1.0)

    assert result.correlation[-1] < result.correlation[0]

    assert result.decayed


def test_a_residence_time_comes_back() -> None:

    result = co.residence_time(gas(), "Li", cutoff=1.0)

    assert np.isfinite(result.tau)
    assert result.tau > 0.0

    assert result.rate == pytest.approx(1.0 / result.tau)


def test_a_frozen_ion_never_leaves() -> None:
    """
    The control, and the case that has to be reported rather than fitted.

    Nothing moves, so the correlation stays at one; there is no
    residence time to measure and the result says so instead of
    returning a number from a fit to a flat line.
    """

    result = co.residence_time(frozen(), "Na", cutoff=1.0, skip=0.0)

    assert result.correlation[-1] == pytest.approx(1.0)

    assert not result.decayed

    assert any("never fell to 1/e" in note for note in result.notes)


def test_a_bigger_cutoff_gives_a_longer_residence() -> None:
    """More room to wander in before it counts as having left."""

    trajectory = gas()

    short = co.residence_time(trajectory, "Li", cutoff=0.6)
    long = co.residence_time(trajectory, "Li", cutoff=1.6)

    assert long.tau > short.tau


def test_a_non_positive_cutoff_is_refused() -> None:

    with pytest.raises(TrajectoryError, match="must be positive"):
        co.residence_time(gas(), "Li", cutoff=0.0)


def test_a_run_too_short_to_correlate_is_refused() -> None:

    with pytest.raises(TrajectoryError, match="needs more than"):
        co.residence_time(gas(frames=4), "Li", skip=0.0)


def test_the_cutoff_is_always_explained() -> None:
    """It is the definition of "still there", and it is a choice."""

    result = co.residence_time(gas(), "Li")

    assert any("counts as having stayed" in n for n in result.notes)

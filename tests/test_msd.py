"""
Tests for mean squared displacement and diffusion.

Built on random walks rather than on a real trajectory, because a random
walk has an exactly known diffusion coefficient: ``MSD = 2 d D t`` holds
by construction, so the pipeline can be asked to recover a number that
is known in advance rather than merely to produce a plausible one.

The estimator is unbiased but noisy, and the noise falls as one over the
square root of the number of walkers.  Measured over two dozen seeds: at
40 walkers a single run lands within about 10% of the truth, at 200
within about 3%.  The fixtures therefore use enough walkers that a tight
tolerance is a real assertion rather than a coin flip.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services import msd

#: Enough walkers that one realisation is within a few percent.
WALKERS = 200

FRAMES = 1200
TIMESTEP = 0.02
BOX = 25.0


def random_walk(
    diffusivity: float = 0.25,
    walkers: int = WALKERS,
    frames: int = FRAMES,
    timestep: float = TIMESTEP,
    seed: int = 3,
) -> np.ndarray:
    """
    Brownian motion with a diffusion coefficient set in advance.

    Each axis takes a gaussian step of variance ``2 D dt``, which is
    exactly the Einstein relation in one dimension; over three axes the
    MSD is then ``6 D t``.
    """

    generator = np.random.default_rng(seed)

    width = np.sqrt(2.0 * diffusivity * timestep)

    steps = generator.normal(0.0, width, size=(frames - 1, walkers, 3))

    return np.concatenate(
        [np.zeros((1, walkers, 3)), np.cumsum(steps, axis=0)]
    )


def trajectory_of(positions, symbols=None, **kwargs) -> msd.Trajectory:

    count = positions.shape[1]

    return msd.Trajectory(
        positions=positions,
        symbols=tuple(symbols or ("Li",) * count),
        lattice=np.eye(3) * BOX,
        timestep=kwargs.pop("timestep", TIMESTEP),
        temperature=kwargs.pop("temperature", 600.0),
        **kwargs,
    )


# ------------------------------------------------------------
# The Einstein relation
# ------------------------------------------------------------


def test_a_known_diffusion_coefficient_comes_back() -> None:
    """The end-to-end check: in 0.25, out 0.25."""

    walk = random_walk(diffusivity=0.25)

    result = msd.analyse(trajectory_of(walk), drift=False)

    fit = result.fit_for("Li")

    assert fit is not None
    assert fit.diffusivity_a2_ps == pytest.approx(0.25, rel=0.05)


def test_doubling_the_diffusivity_doubles_the_answer() -> None:
    """Scaling has to be linear, whatever the absolute accuracy."""

    slow = msd.analyse(
        trajectory_of(random_walk(diffusivity=0.1)), drift=False
    ).fit_for("Li")

    fast = msd.analyse(
        trajectory_of(random_walk(diffusivity=0.2)), drift=False
    ).fit_for("Li")

    assert fast.diffusivity_a2_ps / slow.diffusivity_a2_ps == pytest.approx(
        2.0, rel=0.08
    )


def test_a_walk_is_a_straight_line() -> None:

    result = msd.analyse(trajectory_of(random_walk()), drift=False)

    fit = result.fit_for("Li")

    assert fit.r_squared > 0.99
    assert fit.trustworthy


def test_atoms_that_only_vibrate_do_not_diffuse() -> None:
    """
    A framework atom rattles about a fixed site.

    Its MSD rises and then flattens at the size of the cage, which is
    not a straight line -- so the fit must come back untrustworthy
    rather than quoting the small residual slope as a diffusivity.
    """

    generator = np.random.default_rng(5)

    jitter = generator.normal(
        0.0, 0.1, size=(FRAMES, WALKERS, 3)
    )

    result = msd.analyse(trajectory_of(jitter), drift=False)

    fit = result.fit_for("Li")

    assert fit.diffusivity_a2_ps == pytest.approx(0.0, abs=1e-3)
    assert not fit.trustworthy


def test_the_diffusivity_units_agree_with_the_reference_route() -> None:
    """
    ``D`` in cm^2/s from a slope in A^2/ps.

    pymatgen does the same conversion from a slope in A^2/fs by dividing
    by 60 -- six for ``2 d`` and ten for the units. Landing on the same
    number from a different starting unit is the check that neither has
    a stray factor.
    """

    fit = msd.Diffusion(
        label="x", count=1, slope=3.0, intercept=0.0,
        r_squared=1.0, stderr=0.0, fit_from=0.0, fit_to=1.0,
    )

    assert fit.diffusivity_a2_ps == pytest.approx(0.5)

    slope_per_fs = 3.0 / 1000.0

    assert fit.diffusivity == pytest.approx(slope_per_fs / 60.0)


# ------------------------------------------------------------
# Unwrapping
# ------------------------------------------------------------


def test_unwrapping_undoes_the_fold() -> None:
    """
    An atom walked steadily across several cells and folded back.

    Recovering the straight line it actually travelled is the whole job.
    """

    lattice = np.eye(3) * 10.0

    steps = 300

    true = np.zeros((steps, 1, 3))
    true[:, 0, 0] = np.linspace(0.0, 45.0, steps)

    folded = (true @ np.linalg.inv(lattice)) % 1.0

    recovered = msd.unwrap(folded, lattice)

    assert np.abs((recovered - recovered[0]) - true).max() < 1e-9


def test_folded_coordinates_ruin_the_msd() -> None:
    """
    Not a small error: every boundary crossing adds a whole cell.

    The resulting curve still looks straight, which is why this has to
    be handled rather than eyeballed.
    """

    lattice = np.eye(3) * 8.0

    walk = random_walk(diffusivity=0.5, walkers=40, frames=400)

    folded = (walk @ np.linalg.inv(lattice)) % 1.0 @ lattice

    honest = msd.msd_curve(walk, timestep=TIMESTEP)[1]
    naive = msd.msd_curve(folded, timestep=TIMESTEP)[1]

    assert naive.max() > honest.max() * 2.0


def test_unwrapping_a_run_that_did_not_move_changes_nothing() -> None:

    lattice = np.eye(3) * 10.0

    still = np.tile(np.array([[[0.2, 0.3, 0.4]]]), (50, 4, 1))

    recovered = msd.unwrap(still, lattice)

    assert np.abs(recovered - recovered[0]).max() < 1e-12


def test_unwrap_refuses_the_wrong_shape() -> None:

    with pytest.raises(msd.TrajectoryError, match="frames, atoms, 3"):
        msd.unwrap(np.zeros((10, 3)), np.eye(3))


# ------------------------------------------------------------
# Drift
# ------------------------------------------------------------


def test_cell_drift_looks_like_diffusion_until_it_is_removed() -> None:
    """
    Drift enters every atom equally, so it survives the averaging.

    A framework that is not moving will show a diffusion coefficient
    purely from the cell creeping, which is the commonest way to report
    diffusion in a material that has none.
    """

    walk = random_walk(diffusivity=0.25)

    creep = np.linspace(0.0, 20.0, walk.shape[0])[:, None, None]

    drifting = trajectory_of(walk + creep)

    uncorrected = msd.analyse(drifting, drift=False).fit_for("Li")
    corrected = msd.analyse(drifting, drift=True).fit_for("Li")

    assert uncorrected.diffusivity_a2_ps > 0.25 * 1.5

    assert corrected.diffusivity_a2_ps == pytest.approx(0.25, rel=0.05)


def test_removing_drift_leaves_a_still_cell_still() -> None:

    positions = np.zeros((20, 5, 3))

    assert np.abs(msd.remove_drift(positions)).max() < 1e-12


# ------------------------------------------------------------
# The curve
# ------------------------------------------------------------


def test_the_curve_starts_at_zero() -> None:

    lags, curve, origins = msd.msd_curve(random_walk(), timestep=TIMESTEP)

    assert lags[0] == 0.0
    assert curve[0] == 0.0


def test_time_origins_fall_away_with_lag() -> None:
    """
    The number that says how far the curve is worth believing.

    Each longer lag has one fewer place in the trajectory it can start
    from, so the average behind the far end is over very little.
    """

    _, _, origins = msd.msd_curve(random_walk(frames=200), timestep=TIMESTEP)

    assert origins[1] > origins[-1]
    assert np.all(np.diff(origins[1:]) < 0)


def test_the_lag_range_can_be_capped() -> None:

    lags, _, _ = msd.msd_curve(
        random_walk(frames=500), timestep=TIMESTEP, max_lag=1.0
    )

    assert lags[-1] == pytest.approx(1.0, abs=TIMESTEP)


def test_a_curve_needs_more_than_one_frame() -> None:

    with pytest.raises(msd.TrajectoryError, match="two frames"):
        msd.msd_curve(np.zeros((1, 4, 3)))


def test_a_run_too_short_for_the_origins_is_refused() -> None:

    with pytest.raises(msd.TrajectoryError, match="time origins"):
        msd.msd_curve(np.zeros((4, 2, 3)), min_origins=10)


def test_the_timestep_must_be_positive() -> None:

    with pytest.raises(msd.TrajectoryError):
        msd.msd_curve(np.zeros((10, 2, 3)), timestep=0.0)


# ------------------------------------------------------------
# Fitting
# ------------------------------------------------------------


def test_the_fit_window_is_reported_in_picoseconds() -> None:

    lags = np.linspace(0.0, 10.0, 101)
    curve = 3.0 * lags

    fit = msd.fit_diffusion(lags, curve, start=0.2, end=0.6)

    assert fit.fit_from == pytest.approx(2.0, abs=0.1)
    assert fit.fit_to == pytest.approx(6.0, abs=0.1)


def test_a_perfect_line_fits_perfectly() -> None:

    lags = np.linspace(0.0, 10.0, 101)

    fit = msd.fit_diffusion(lags, 6.0 * lags)

    assert fit.slope == pytest.approx(6.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.diffusivity_a2_ps == pytest.approx(1.0)


def test_an_empty_fit_window_is_refused() -> None:

    lags = np.linspace(0.0, 10.0, 101)

    with pytest.raises(msd.TrajectoryError):
        msd.fit_diffusion(lags, lags, start=0.7, end=0.3)


def test_too_few_points_in_the_window_is_refused() -> None:

    lags = np.linspace(0.0, 1.0, 6)

    with pytest.raises(msd.TrajectoryError, match="three points"):
        msd.fit_diffusion(lags, lags, start=0.0, end=0.05)


def test_mismatched_lengths_are_refused() -> None:

    with pytest.raises(msd.TrajectoryError):
        msd.fit_diffusion(np.zeros(10), np.zeros(9))


# ------------------------------------------------------------
# Conductivity
# ------------------------------------------------------------


def test_nernst_einstein_matches_the_reference_formula() -> None:
    """
    Checked against pymatgen's ``get_conversion_factor``.

    sigma [mS/cm] = 1000 n / (V N_A) z^2 (N_A e)^2 D / (R T)
    """

    volume, count, charge, temperature = 1000.0, 20, 1.0, 300.0
    diffusivity = 1e-7

    expected = (
        1000.0
        * count
        / (volume * 1e-24 * msd.AVOGADRO)
        * charge**2
        * (msd.AVOGADRO * msd.ELEMENTARY_CHARGE) ** 2
        * diffusivity
        / (msd.GAS_CONSTANT * temperature)
    )

    got = msd.conductivity(
        diffusivity, count, charge, volume, temperature
    )

    assert got == pytest.approx(expected)


def test_a_superionic_diffusivity_gives_a_superionic_conductivity() -> None:
    """A sanity anchor: 1e-7 cm^2/s of lithium is a few mS/cm."""

    got = msd.conductivity(1e-7, 20, 1.0, 1000.0, 300.0)

    assert 1.0 < got < 100.0


def test_conductivity_scales_with_the_square_of_the_charge() -> None:

    single = msd.conductivity(1e-7, 10, 1.0, 500.0, 300.0)
    double = msd.conductivity(1e-7, 10, 2.0, 500.0, 300.0)

    assert double / single == pytest.approx(4.0)


def test_conductivity_needs_a_temperature_and_a_cell() -> None:

    with pytest.raises(msd.TrajectoryError):
        msd.conductivity(1e-7, 10, 1.0, 500.0, 0.0)

    with pytest.raises(msd.TrajectoryError, match="volume"):
        msd.conductivity(1e-7, 10, 1.0, 0.0, 300.0)


# ------------------------------------------------------------
# Trajectories
# ------------------------------------------------------------


def test_species_get_their_own_curve() -> None:

    walk = random_walk(walkers=20, frames=400)

    symbols = ("Li",) * 10 + ("Co",) * 10

    result = msd.analyse(
        trajectory_of(walk, symbols=symbols), drift=False
    )

    assert set(result.curves) == {"Li", "Co"}

    assert result.fit_for("Li") is not None
    assert result.fit_for("Co") is not None
    assert result.fit_for("all") is not None


def test_a_short_run_says_so() -> None:

    result = msd.analyse(
        trajectory_of(random_walk(frames=200, walkers=20)), drift=False
    )

    assert any("order of magnitude" in note for note in result.notes)


def test_a_trajectory_knows_its_own_shape() -> None:

    trajectory = trajectory_of(random_walk(walkers=8, frames=50))

    assert trajectory.frames == 50
    assert trajectory.natoms == 8
    assert trajectory.duration == pytest.approx(49 * TIMESTEP)
    assert trajectory.volume == pytest.approx(BOX**3)


def test_frames_and_symbols_have_to_agree() -> None:

    with pytest.raises(msd.TrajectoryError, match="symbols"):
        msd.from_frames(np.zeros((5, 4, 3)), ["Li", "Co"])


def test_folded_frames_need_a_cell_to_be_unwrapped() -> None:

    with pytest.raises(msd.TrajectoryError, match="cell"):
        msd.from_frames(
            np.zeros((5, 2, 3)), ["Li", "Co"], wrapped=True
        )


# ------------------------------------------------------------
# XDATCAR
# ------------------------------------------------------------


def write_xdatcar(path, frames, lattice, symbols) -> None:

    inverse = np.linalg.inv(lattice)

    lines = ["test", "           1"]

    for row in lattice:
        lines.append("  " + "  ".join(f"{value:12.6f}" for value in row))

    ordered = list(dict.fromkeys(symbols))

    lines.append("   " + "  ".join(ordered))
    lines.append("   " + "  ".join(str(symbols.count(s)) for s in ordered))

    for index, frame in enumerate(frames, start=1):

        lines.append(f"Direct configuration=  {index:5d}")

        for position in frame @ inverse:
            lines.append(
                "  " + "  ".join(f"{value:12.8f}" for value in position)
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def test_an_xdatcar_round_trips(tmp_path) -> None:

    lattice = np.eye(3) * 12.0

    walk = random_walk(walkers=6, frames=120, diffusivity=0.4)

    folded = (walk @ np.linalg.inv(lattice)) % 1.0 @ lattice

    path = tmp_path / "XDATCAR"

    write_xdatcar(path, folded, lattice, ["Li"] * 6)

    trajectory = msd.read_xdatcar(path, timestep=TIMESTEP)

    assert trajectory.frames == 120
    assert trajectory.natoms == 6
    assert trajectory.symbols == ("Li",) * 6

    # Read back and unwrapped, it is the walk it started as.
    moved = trajectory.positions - trajectory.positions[0]

    assert np.abs(moved - (walk - walk[0])).max() < 1e-4


def test_an_xdatcar_names_its_species(tmp_path) -> None:

    lattice = np.eye(3) * 12.0

    frames = np.zeros((3, 5, 3))

    path = tmp_path / "XDATCAR"

    write_xdatcar(path, frames, lattice, ["Li", "Li", "O", "O", "O"])

    trajectory = msd.read_xdatcar(path)

    assert trajectory.symbols == ("Li", "Li", "O", "O", "O")
    assert trajectory.elements() == ["Li", "O"]
    assert trajectory.indices_of("O") == [2, 3, 4]


def test_a_relaxation_is_not_a_trajectory(tmp_path) -> None:
    """
    One configuration is a static run, not dynamics.

    Worth its own message: an XDATCAR from a relaxation looks exactly
    like one from dynamics until you count the frames.
    """

    lattice = np.eye(3) * 12.0

    path = tmp_path / "XDATCAR"

    write_xdatcar(path, np.zeros((1, 4, 3)), lattice, ["Li"] * 4)

    with pytest.raises(msd.TrajectoryError, match="IBRION"):
        msd.read_xdatcar(path)


def test_a_missing_xdatcar_says_so(tmp_path) -> None:

    with pytest.raises(msd.TrajectoryError, match="No XDATCAR"):
        msd.read_xdatcar(tmp_path / "nowhere")


def test_a_truncated_xdatcar_is_refused(tmp_path) -> None:

    path = tmp_path / "XDATCAR"
    path.write_text("test\n1\n", encoding="utf-8")

    with pytest.raises(msd.TrajectoryError, match="too short"):
        msd.read_xdatcar(path)

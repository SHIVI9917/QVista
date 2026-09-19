"""
Tests for the Arrhenius fit and the ionic conductivity.

The Arrhenius tests are recovery tests: a set of diffusivities is built
from a known ``Ea`` and ``D0``, and the fit is asked to give those back.
That is possible here in a way it is not for most analyses, and it is
worth using -- a fit that returns a plausible barrier from data with no
barrier in it would pass any test that only checked the shape.

The conductivity tests pin the thing that is easy to get wrong and
impossible to see afterwards: the direction of the Nernst-Einstein
temperature dependence.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services import arrhenius as ar
from qvista.services import conductivity as cond
from qvista.services.msd import TrajectoryError


# ------------------------------------------------------------
# Building points with a known answer
# ------------------------------------------------------------


def ideal(
    activation: float = 0.35,
    prefactor: float = 1.2e-3,
    temperatures=(600.0, 700.0, 800.0, 900.0, 1000.0),
    noise: float = 0.0,
    seed: int = 0,
):
    """Diffusivities generated from a known Ea and D0."""

    generator = np.random.default_rng(seed)

    points = []

    for temperature in temperatures:

        value = prefactor * np.exp(
            -activation / (ar.BOLTZMANN_EV * temperature)
        )

        if noise:
            value *= generator.normal(1.0, noise)

        points.append(ar.Point(temperature, float(value)))

    return points


# ------------------------------------------------------------
# The fit
# ------------------------------------------------------------


def test_a_clean_set_is_recovered_exactly() -> None:

    fit = ar.fit(ideal())

    assert fit.activation_energy == pytest.approx(0.35, abs=1e-6)
    assert fit.prefactor == pytest.approx(1.2e-3, rel=1e-6)

    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_a_noisy_set_is_recovered_within_its_error() -> None:
    """
    Five points with 6% scatter on D.

    The recovered barrier should sit within a few times its own quoted
    uncertainty of the truth -- if it does not, the uncertainty is
    wrong even when the value is right.
    """

    fit = ar.fit(ideal(noise=0.06))

    assert abs(fit.activation_energy - 0.35) < 0.03

    assert np.isfinite(fit.activation_error)

    assert abs(fit.activation_energy - 0.35) < 5.0 * fit.activation_error


def test_the_barrier_is_reported_in_both_units() -> None:

    fit = ar.fit(ideal(activation=0.5))

    assert fit.activation_kj == pytest.approx(0.5 * 96.485, rel=1e-4)


def test_two_points_are_refused() -> None:
    """
    A line through two points is not a fit.

    It goes exactly through both whatever they are, so R-squared is one
    and says nothing at all.
    """

    with pytest.raises(TrajectoryError, match="at least 3"):
        ar.fit(ideal(temperatures=(600.0, 900.0)))


def test_a_non_positive_diffusivity_is_dropped() -> None:
    """A logarithm needs it above zero; the drop is reported."""

    points = ideal() + [ar.Point(1100.0, 0.0)]

    fit = ar.fit(points)

    assert len(fit.points) == 5

    assert any("were dropped" in note for note in fit.notes)


def test_scattered_points_are_called_out() -> None:
    """
    A poor R-squared means a single barrier does not describe them.

    Usually the mechanism changed across the range, which is exactly
    the case where extrapolating down is worst.
    """

    generator = np.random.default_rng(3)

    points = [
        ar.Point(temperature, float(generator.uniform(1e-6, 1e-4)))
        for temperature in (600.0, 700.0, 800.0, 900.0, 1000.0)
    ]

    fit = ar.fit(points)

    assert not fit.trustworthy

    assert any("do not lie on a line" in note for note in fit.notes)


def test_a_negative_barrier_is_called_out() -> None:
    """D falling as T rises is not diffusion behaving."""

    fit = ar.fit(ideal(activation=-0.2))

    assert fit.activation_energy < 0.0

    assert any("negative" in note for note in fit.notes)


# ------------------------------------------------------------
# Extrapolating
# ------------------------------------------------------------


def test_the_fit_reproduces_its_own_points() -> None:

    fit = ar.fit(ideal())

    for point in fit.points:

        assert fit.at(point.temperature) == pytest.approx(
            point.diffusivity, rel=1e-6
        )


def test_extrapolating_below_the_range_is_flagged() -> None:
    """
    The number everyone wants is the one furthest from the data.

    AIMD is run hot because nothing happens at 300 K inside a
    picosecond; the result is then extrapolated across a factor of
    three in 1/T, which holds only if the mechanism does not change.
    """

    fit = ar.fit(ideal())

    assert fit.extrapolating

    assert any("extrapolation" in note for note in fit.notes)


def test_a_range_that_includes_room_temperature_is_not() -> None:

    fit = ar.fit(
        ideal(temperatures=(250.0, 300.0, 400.0, 500.0))
    )

    assert not fit.extrapolating


def test_diffusivity_at_absolute_zero_is_refused() -> None:

    with pytest.raises(TrajectoryError, match="absolute zero"):
        ar.fit(ideal()).at(0.0)


# ------------------------------------------------------------
# Ionic conductivity
# ------------------------------------------------------------


def test_nernst_einstein_scales_with_the_diffusivity() -> None:

    first = cond.nernst_einstein(1e-6, 6, 1.0, 400.0, 300.0)
    second = cond.nernst_einstein(2e-6, 6, 1.0, 400.0, 300.0)

    assert second == pytest.approx(2.0 * first)


def test_it_scales_with_the_square_of_the_charge() -> None:
    """
    ``z^2``, not ``z``.

    A divalent carrier gives four times the conductivity at the same
    diffusivity, and writing ``z`` instead is a factor-of-two error
    that looks entirely plausible.
    """

    single = cond.nernst_einstein(1e-6, 6, 1.0, 400.0, 300.0)
    double = cond.nernst_einstein(1e-6, 6, 2.0, 400.0, 300.0)

    assert double == pytest.approx(4.0 * single)


def test_it_falls_with_temperature_at_fixed_d() -> None:
    """
    The explicit ``1/T`` in front.

    Real conductivity rises with temperature because D rises far
    faster; at fixed D the prefactor alone falls, and getting that
    backwards inverts the whole Arrhenius plot.
    """

    hot = cond.nernst_einstein(1e-6, 6, 1.0, 400.0, 900.0)
    cold = cond.nernst_einstein(1e-6, 6, 1.0, 400.0, 300.0)

    assert cold > hot


def test_absolute_zero_is_refused() -> None:

    with pytest.raises(cond.ConductivityError, match="absolute zero"):
        cond.nernst_einstein(1e-6, 6, 1.0, 400.0, 0.0)


def test_no_mobile_ions_is_refused() -> None:

    with pytest.raises(cond.ConductivityError, match="no mobile ions"):
        cond.nernst_einstein(1e-6, 0, 1.0, 400.0, 300.0)


def test_the_ionic_result_carries_the_diffusion_barrier() -> None:
    """
    Nernst-Einstein multiplies D by a prefactor in 1/T.

    That shifts the intercept and not the slope, so the conductivity
    has the same activation energy as the diffusivity.
    """

    result = cond.ionic(ideal(activation=0.42), "Li", 1.0, 6, 400.0)

    assert result.activation_energy == pytest.approx(0.42, abs=1e-6)


def test_the_ionic_result_converts_every_point() -> None:

    result = cond.ionic(ideal(), "Li", 1.0, 6, 400.0)

    assert len(result.measured) == 5

    assert all(sigma > 0.0 for _t, sigma in result.measured)


def test_the_haven_ratio_caveat_is_always_stated() -> None:
    """Every number here is an upper estimate, and should say so."""

    result = cond.ionic(ideal(), "Li", 1.0, 6, 400.0)

    assert any("Haven ratio" in note for note in result.notes)

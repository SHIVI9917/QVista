"""
Tests for the molecular-dynamics traces figure.

Two panels: potential energy on top, temperature below.

Only the potential energy is drawn. Kinetic and total live on the same
axis but not on the same scale -- on a cell of any size they are two or
three orders of magnitude apart, so a shared axis spends its whole
height on the gap between them and none of it on the structure in the
curve being read. Both are still in the CSV, and the energy drift they
answer for is in the summary rows.

Each panel carries one dotted black reference: the mean potential energy
on top, the thermostat target below. Neither is labelled -- the axis
already carries the number, and a text annotation sits wherever the
value happens to fall, which on a fluctuating trace is usually on top
of it.
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from qvista.analyses.md import ENSEMBLE_NVE, ENSEMBLE_NVT, MDResult
from qvista.visualization import dynamics_plots as dp


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------


def run(steps: int = 50, ensemble: str = ENSEMBLE_NVT, target=300.0):
    """A short run with distinguishable energy channels."""

    generator = np.random.default_rng(0)

    times = np.arange(1, steps + 1, dtype=float)

    potential = -300.0 + generator.normal(0.0, 0.05, steps)
    kinetic = 1.5 + generator.normal(0.0, 0.05, steps)

    return MDResult(
        times=list(times),
        potential_energies=list(potential),
        kinetic_energies=list(kinetic),
        total_energies=list(potential + kinetic),
        temperatures=list(target + generator.normal(0.0, 40.0, steps)),
        ensemble=ensemble,
        temperature=target,
        natoms=8,
        steps=steps,
        force_field="CHGNet",
    )


def drawn(result=None) -> Figure:

    figure = Figure()

    dp.draw_md_traces(figure, result if result is not None else run())

    return figure


def data_lines(axes):
    """The traces, as against the dotted references."""

    return [
        line
        for line in axes.lines
        if line.get_linestyle() not in (":", "dotted")
    ]


def references(axes):
    """The dotted reference lines."""

    return [
        line
        for line in axes.lines
        if line.get_linestyle() in (":", "dotted")
    ]


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------


def test_there_are_two_panels() -> None:

    assert len(drawn().axes) == 2


def test_an_empty_run_draws_nothing_rather_than_failing() -> None:
    """A cancelled run has no steps in it."""

    figure = Figure()

    dp.draw_md_traces(figure, MDResult())

    assert not figure.axes


# ------------------------------------------------------------
# Only the potential energy
# ------------------------------------------------------------


def test_the_energy_panel_draws_one_trace() -> None:

    top = drawn().axes[0]

    assert len(data_lines(top)) == 1


def test_that_trace_is_the_potential_energy() -> None:
    """
    Not the total, which is the one it would be confused with.

    The two differ only by the kinetic energy, so a figure showing the
    wrong one still looks entirely reasonable -- and the question the
    panel is read for, whether the structure has settled, is answered
    by the potential.
    """

    result = run()

    top = drawn(result).axes[0]

    drawn_values = data_lines(top)[0].get_ydata()

    assert np.allclose(drawn_values, result.potential_energies)

    assert not np.allclose(drawn_values, result.total_energies)


def test_there_is_no_legend_on_the_energy_panel() -> None:
    """One curve needs no key."""

    assert drawn().axes[0].get_legend() is None


def test_the_axis_says_which_energy() -> None:
    """Since the legend that used to say so is gone."""

    assert "Potential" in drawn().axes[0].get_ylabel()


def test_the_other_channels_are_still_in_the_csv() -> None:
    """
    Dropped from the figure, not from the data.

    The kinetic and total energies are what the NVE drift diagnostic is
    computed from, so losing them would cost more than a curve.
    """

    text = run().to_csv()

    assert "kinetic_ev" in text
    assert "total_ev" in text


# ------------------------------------------------------------
# The mean
# ------------------------------------------------------------


def test_the_energy_panel_marks_the_mean() -> None:

    result = run()

    top = drawn(result).axes[0]

    marks = references(top)

    assert len(marks) == 1

    assert marks[0].get_ydata()[0] == pytest.approx(
        float(np.mean(result.potential_energies))
    )


def test_the_mean_is_dotted_and_black() -> None:

    mark = references(drawn().axes[0])[0]

    assert mark.get_linestyle() in (":", "dotted")

    assert mark.get_color() == "black"


def test_the_mean_is_not_a_legend_entry() -> None:
    """It is a reference, not a series."""

    label = references(drawn().axes[0])[0].get_label()

    assert not label or label.startswith("_")


# ------------------------------------------------------------
# The temperature panel
# ------------------------------------------------------------


def test_the_target_is_marked_under_a_thermostat() -> None:

    bottom = drawn(run(ensemble=ENSEMBLE_NVT, target=750.0)).axes[1]

    marks = references(bottom)

    assert len(marks) == 1

    assert marks[0].get_ydata()[0] == pytest.approx(750.0)


def test_there_is_no_target_without_a_thermostat() -> None:
    """
    NVE has no target temperature.

    Drawing one would invite reading a drift away from it as a failure,
    when in NVE the temperature is free to go wherever the dynamics
    take it.
    """

    bottom = drawn(run(ensemble=ENSEMBLE_NVE)).axes[1]

    assert not references(bottom)


def test_the_target_line_carries_no_text() -> None:
    """
    The axis already gives the number.

    The label used to sit at the target's own height, which on a
    fluctuating trace is the middle of the data.
    """

    bottom = drawn().axes[1]

    assert not bottom.texts


def test_the_target_is_dotted_and_black_too() -> None:
    """The two references match, because they are the same kind of thing."""

    top = references(drawn().axes[0])[0]
    bottom = references(drawn().axes[1])[0]

    assert top.get_linestyle() == bottom.get_linestyle()
    assert top.get_color() == bottom.get_color()


# ------------------------------------------------------------
# What is left on the figure
# ------------------------------------------------------------


def test_the_force_field_is_still_stamped_on() -> None:
    """So a screenshot can never be attributed to the wrong method."""

    figure = drawn()

    assert any("CHGNet" in text.get_text() for text in figure.texts)


def test_the_panels_share_the_time_axis() -> None:

    figure = drawn()

    # Ticks are not generated until something asks for a render.
    figure.canvas.draw()

    top, bottom = figure.axes

    assert bottom.get_xlabel() == "Time (fs)"

    # Only the lower panel is labelled: the two share an axis, so
    # repeating the numbers on both would be the same row twice.
    assert not top.get_xlabel()

    assert all(
        not label.get_visible() for label in top.get_xticklabels()
    )

    assert any(
        label.get_visible() for label in bottom.get_xticklabels()
    )

"""
Tests for how the branches of a phonon dispersion are coloured.

A dispersion is ``3N`` lines. Drawn in one colour it says nothing about
which line is which, and a legend is the only thing that can say so --
but a legend naming forty-eight lines names nothing at all.

So there are three schemes and two rules that hold across them.

**A branch is named once.** A branch runs through every segment of the
path, and every segment is a separate call to ``plot``. Labelling each
one would give a legend that repeats "acoustic" once per high-symmetry
line.

**A group is one colour.** The Design panel hands its palettes out per
group of same-coloured artists, and only to groups a legend names. The
grouping done here is therefore what that panel later recolours, which
is why the two acoustic-scheme colours have to be exactly two.
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from qvista.visualization import phonon_plots as pp
from qvista.visualization import plot_style as ps


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------


def dispersion(branches: int = 12, segments: int = 2, points: int = 30):
    """A band dict of the shape phonopy hands back."""

    distances = [
        np.linspace(index, index + 1.0, points)
        for index in range(segments)
    ]

    frequencies = [
        np.sort(
            np.outer(np.ones(points), np.arange(branches, dtype=float))
            + np.cos(np.linspace(0.0, np.pi, points))[:, None],
            axis=1,
        )
        for _ in range(segments)
    ]

    return {
        "distances": distances,
        "frequencies": frequencies,
        "labels": ["GAMMA", "X", "M"][: segments + 1],
        "path_connections": [True] * segments,
    }


def density(points: int = 40):

    return {
        "frequency_points": np.linspace(0.0, 12.0, points),
        "total_dos": np.abs(np.sin(np.linspace(0.0, 6.0, points))),
    }


def entries(axes) -> list[str]:
    """What the legend would name, in order."""

    return axes.get_legend_handles_labels()[1]


def band_colours(axes) -> list[str]:
    """
    The colour of every branch, in the order drawn.

    The zero line and the high-symmetry guides are drawn with a blended
    transform, so they are decoration rather than data and are left out
    -- the same rule the Design panel uses.
    """

    return [
        line.get_color()
        for line in axes.lines
        if ps.is_data(line, axes)
    ]


# ------------------------------------------------------------
# Acoustic and optical
# ------------------------------------------------------------


def test_the_three_lowest_branches_are_the_acoustic_ones() -> None:
    """
    Three, always: one per translation of the whole crystal.

    They are the branches that go to zero at Gamma and carry the sound
    velocities, which is the division a reader is asking about.
    """

    figure = Figure()

    pp.draw_band(figure, dispersion(branches=9), scheme=pp.BAND_ACOUSTIC)

    colours = band_colours(figure.axes[0])

    # Nine branches over two segments, lowest first within each.
    assert colours[:3] == [pp.ACOUSTIC_COLOUR] * 3
    assert colours[3:9] == [pp.OPTICAL_COLOUR] * 6

    assert set(colours) == {pp.ACOUSTIC_COLOUR, pp.OPTICAL_COLOUR}


def test_the_legend_names_each_group_once() -> None:
    """Not once per branch, and not once per segment of the path."""

    figure = Figure()

    pp.draw_band(
        figure, dispersion(branches=12, segments=3), scheme=pp.BAND_ACOUSTIC
    )

    assert entries(figure.axes[0]) == ["acoustic", "optical"]


def test_a_cell_with_no_optical_branches_says_so_by_omission() -> None:
    """
    One atom in the cell is three branches, all of them acoustic.

    An "optical" entry there would name a group with nothing in it.
    """

    figure = Figure()

    pp.draw_band(figure, dispersion(branches=3), scheme=pp.BAND_ACOUSTIC)

    assert entries(figure.axes[0]) == ["acoustic"]


# ------------------------------------------------------------
# One colour per branch
# ------------------------------------------------------------


def test_neighbouring_branches_are_told_apart() -> None:
    """
    The scheme exists for following one branch through a crossing,
    which needs the branches either side of it to differ.
    """

    figure = Figure()

    pp.draw_band(figure, dispersion(branches=6), scheme=pp.BAND_BRANCH)

    colours = band_colours(figure.axes[0])[:6]

    assert len(set(colours)) == 6


def test_the_same_branch_keeps_its_colour_across_the_path() -> None:
    """
    A colour that changed at every high-symmetry line would be worse
    than no colour: the eye would follow it into the wrong branch.
    """

    figure = Figure()

    pp.draw_band(
        figure, dispersion(branches=4, segments=3), scheme=pp.BAND_BRANCH
    )

    colours = band_colours(figure.axes[0])

    assert colours[0:4] == colours[4:8] == colours[8:12]


def test_the_legend_is_dropped_when_it_would_name_too_many() -> None:
    """
    Past a dozen entries a legend fills the axes and identifies
    nothing. The colours stay, because they still separate the lines.
    """

    figure = Figure()

    pp.draw_band(
        figure,
        dispersion(branches=pp.MAX_BRANCH_LEGEND + 1),
        scheme=pp.BAND_BRANCH,
    )

    axes = figure.axes[0]

    assert entries(axes) == []
    assert len(set(band_colours(axes))) > 1


def test_the_legend_is_kept_at_the_limit() -> None:

    figure = Figure()

    pp.draw_band(
        figure,
        dispersion(branches=pp.MAX_BRANCH_LEGEND),
        scheme=pp.BAND_BRANCH,
    )

    assert len(entries(figure.axes[0])) == pp.MAX_BRANCH_LEGEND


# ------------------------------------------------------------
# One colour
# ------------------------------------------------------------


def test_the_plain_scheme_is_what_it_always_was() -> None:
    """
    One colour, no legend: the figure as it was drawn before there was
    a choice, so nothing already published changes underfoot.
    """

    figure = Figure()

    pp.draw_band(figure, dispersion(), scheme=pp.BAND_SINGLE)

    axes = figure.axes[0]

    assert set(band_colours(axes)) == {pp.DEFAULT_BAND_COLOUR}
    assert axes.get_legend() is None


def test_the_plain_scheme_takes_the_colour_it_is_given() -> None:

    figure = Figure()

    pp.draw_band(
        figure, dispersion(), scheme=pp.BAND_SINGLE, colour="#ff0000"
    )

    assert set(band_colours(figure.axes[0])) == {"#ff0000"}


def test_the_default_colour_parses_as_a_colour_qt_understands() -> None:
    """
    It is put on a swatch as well as on the lines.

    matplotlib's own names -- ``tab:blue`` -- mean nothing to Qt, and
    the button would silently lose its swatch.
    """

    assert pp.DEFAULT_BAND_COLOUR.startswith("#")
    assert len(pp.DEFAULT_BAND_COLOUR) == 7

    int(pp.DEFAULT_BAND_COLOUR[1:], 16)


# ------------------------------------------------------------
# Across the schemes
# ------------------------------------------------------------


@pytest.mark.parametrize("scheme", list(pp.BAND_SCHEMES))
def test_every_branch_is_drawn_whatever_the_scheme(scheme: str) -> None:
    """Colouring must never be a filter."""

    figure = Figure()

    pp.draw_band(figure, dispersion(branches=8, segments=2), scheme=scheme)

    assert len(band_colours(figure.axes[0])) == 16


@pytest.mark.parametrize("scheme", list(pp.BAND_SCHEMES))
def test_the_legend_can_be_turned_off(scheme: str) -> None:

    figure = Figure()

    pp.draw_band(figure, dispersion(), scheme=scheme, legend=False)

    assert figure.axes[0].get_legend() is None


@pytest.mark.parametrize("scheme", list(pp.BAND_SCHEMES))
def test_the_combined_figure_colours_its_band_panel_the_same(
    scheme: str,
) -> None:
    """
    Band-with-DOS is the same dispersion beside a density of states.

    It has to take the same scheme, or the choice would silently stop
    applying the moment a DOS was asked for alongside.
    """

    figure = Figure()

    pp.draw_band_with_dos(
        figure, dispersion(branches=6), density(), scheme=scheme
    )

    alone = Figure()

    pp.draw_band(alone, dispersion(branches=6), scheme=scheme)

    assert band_colours(figure.axes[0]) == band_colours(alone.axes[0])


def test_a_single_branch_dispersion_does_not_fall_over() -> None:
    """
    One branch arrives as a flat array rather than a column of one,
    and indexing it as two-dimensional would raise mid-repaint.
    """

    band = dispersion(branches=1)

    band["frequencies"] = [
        np.asarray(segment).ravel() for segment in band["frequencies"]
    ]

    figure = Figure()

    pp.draw_band(figure, band, scheme=pp.BAND_ACOUSTIC)

    assert band_colours(figure.axes[0]) == [pp.ACOUSTIC_COLOUR] * 2


# ------------------------------------------------------------
# What the Design panel makes of it
# ------------------------------------------------------------


def test_the_design_palette_now_reaches_the_branches() -> None:
    """
    The reason for labelling them at all.

    Recolouring is handed out per group of same-coloured artists, and
    only to groups a legend names. Unlabelled, a whole dispersion was
    one group and took one colour; grouped, acoustic and optical take
    two -- and the legend beside them says which is which.
    """

    figure = Figure()

    pp.draw_band(figure, dispersion(branches=6), scheme=pp.BAND_ACOUSTIC)

    ps.apply(figure, ps.PlotStyle(palette="Bright"))

    colours = band_colours(figure.axes[0])

    bright = ps.PALETTES["Bright"]

    assert colours[:3] == [bright[0]] * 3
    assert colours[3:6] == [bright[1]] * 3


def test_a_restyled_dispersion_goes_back_as_it_was_drawn() -> None:

    figure = Figure()

    pp.draw_band(figure, dispersion(branches=6), scheme=pp.BAND_ACOUSTIC)

    before = band_colours(figure.axes[0])

    ps.apply(figure, ps.PlotStyle(palette="Greyscale"))
    ps.reset(figure)

    assert band_colours(figure.axes[0]) == before

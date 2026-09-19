"""
Tests for how a band structure is drawn.

A projection on a dispersion is a *fat band*: one line per branch whose
width follows the weight along the path, thick where that character
dominates and tapering to a hairline where it does not.

Markers encode the same number and are worse at showing it. A scatter
breaks each branch into a cloud of separate dots, and the eye stops
following the band -- which is the thing the figure is drawn on top of.

The rendering therefore has to be a
:class:`~matplotlib.collections.LineCollection`: ``plot`` takes one
linewidth for a whole line, so it cannot vary along it at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from qvista.services import electronic as el
from qvista.visualization import electronic_plots as ep


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------


def path_result(weights, nk: int = 40, nb: int = 4):
    """A k-path result carrying the given per-series weights."""

    x = np.linspace(0.0, 1.0, nk)

    bands = np.vstack([np.cos(x * (i + 1)) - 1.0 + i for i in range(nb)])

    return el.BandResult(
        distances=[x],
        energies={"up": [bands]},
        weights=weights,
        mode=el.PATH_MODE,
    )


def mesh_result(weights, nk: int = 4, nb: int = 4):
    """The same, on a sampling mesh rather than a path."""

    return el.BandResult(
        distances=[np.arange(1.0, nk + 1.0)],
        energies={"up": [np.zeros((nb, nk))]},
        weights=weights,
        mode=el.SCF_MODE,
    )


def bands_on(figure) -> list[LineCollection]:
    """The fat bands drawn on a figure."""

    return [
        artist
        for artist in figure.axes[0].collections
        if isinstance(artist, LineCollection)
    ]


def legend_labels(figure) -> list[str]:
    """What the legend actually says."""

    legend = figure.axes[0].get_legend()

    if legend is None:
        return []

    return [text.get_text() for text in legend.get_texts()]


def widest(figure, first: int, count: int) -> float:
    """The largest width across a slice of the drawn bands."""

    return max(
        float(max(item.get_linewidths()))
        for item in bands_on(figure)[first:first + count]
    )


def drawn(weights, **kwargs) -> Figure:

    figure = Figure()

    ep.draw_band(figure, path_result(weights), **kwargs)

    return figure


# ------------------------------------------------------------
# The rendering
# ------------------------------------------------------------


def test_a_projection_on_a_path_is_drawn_as_a_fat_band() -> None:

    assert bands_on(drawn({"d": np.ones((4, 40))}))


def test_the_width_follows_the_weight_along_the_band() -> None:
    """
    The whole point: one line, many widths.

    ``plot`` cannot do this -- its linewidth is one number for the
    entire line -- which is why this goes through a LineCollection.
    """

    ramp = {"d": np.tile(np.linspace(0.0, 1.0, 40), (4, 1))}

    widths = bands_on(drawn(ramp))[0].get_linewidths()

    assert len(widths) > 1

    assert widths[0] < widths[-1]


def test_the_thinnest_a_band_gets_is_not_nothing() -> None:
    """
    A series that is present but weak should still trace its branch.

    Zero width would make it disappear where it is small, which reads
    as the band ending rather than as the character fading out.
    """

    weights = np.zeros((4, 40))
    weights[:, 20] = 1.0

    widths = bands_on(drawn({"d": weights}))[0].get_linewidths()

    assert min(widths) == pytest.approx(ep.FAT_BASE_WIDTH)

    assert max(widths) == pytest.approx(
        ep.FAT_BASE_WIDTH + ep.FAT_SCALE
    )


def test_the_segments_are_round_ended() -> None:
    """
    Otherwise the band reads as a chain of tapered sticks.

    Each segment is drawn separately with its own width, so without
    round caps and joins the joints between them show.
    """

    collection = bands_on(drawn({"d": np.ones((4, 40))}))[0]

    assert collection.get_capstyle() == "round"
    assert collection.get_joinstyle() == "round"


def test_the_band_is_segmented_across_every_gap_between_k_points() -> None:
    """
    One segment per gap, subdivided by the spline that smooths it.

    The branch is resampled through a B-spline before it is drawn, so
    each calculated gap becomes ``DENSITY`` segments rather than one.
    What has to hold is that the segments still tile the whole path and
    nothing has been dropped or joined across a corner.
    """

    from qvista.visualization import smoothing

    collection = bands_on(drawn({"d": np.ones((4, 40))}))[0]

    assert len(collection.get_segments()) == 39 * smoothing.DENSITY


def test_a_single_k_point_draws_nothing_rather_than_failing() -> None:
    """One point has no segments; it must not raise."""

    figure = Figure()

    ep.draw_band(figure, path_result({"d": np.ones((4, 1))}, nk=1))


# ------------------------------------------------------------
# A mesh is not a path
# ------------------------------------------------------------


def test_a_mesh_is_drawn_with_markers() -> None:
    """
    There are no adjacent k-points on a mesh to join.

    A segment between two of them crosses the Brillouin zone between
    unrelated states, and would look exactly like a band.
    """

    figure = Figure()

    ep.draw_band(figure, mesh_result({"d": np.ones((4, 4))}))

    assert not bands_on(figure)

    assert figure.axes[0].collections, "nothing was drawn at all"


def test_the_legend_key_is_a_dot_either_way() -> None:
    """
    A fat band has no single width to put in a key.

    Showing one would mean picking a weight to represent the series by,
    and any choice there is arbitrary. Colour is what the reader is
    matching on, so a plain filled dot carries it in both renderings.
    """

    path = drawn({"d": np.ones((4, 40))})

    mesh = Figure()
    ep.draw_band(mesh, mesh_result({"d": np.ones((4, 4))}))

    def marker_of(figure):
        return figure.axes[0].get_legend().legend_handles[0].get_marker()

    assert marker_of(path) == "o"
    assert marker_of(mesh) == "o"


# ------------------------------------------------------------
# How a weight becomes a width
# ------------------------------------------------------------


def weak_and_strong():
    """One series reaching unity, one never exceeding 0.15."""

    return {
        "strong": np.ones((4, 40)),
        "weak": np.full((4, 40), 0.15),
    }


def test_per_series_lets_a_weak_series_reach_full_width() -> None:
    """
    Each series scaled against its own largest weight.

    What a fat-band figure is normally read for is *where* along the
    path a character sits. A series that never takes more than a fifth
    of any state answers that just as well as a strong one -- but not
    if it is drawn as a permanent hairline.
    """

    figure = drawn(weak_and_strong(), weight_scale=ep.PER_SERIES)

    full = ep.FAT_BASE_WIDTH + ep.FAT_SCALE

    assert widest(figure, 0, 4) == pytest.approx(full)
    assert widest(figure, 4, 4) == pytest.approx(full)


def test_per_state_keeps_the_series_comparable() -> None:
    """
    The other reading: how much, rather than where.

    Here the weak series stays thin, which is the honest answer to how
    much of each state sits on it.
    """

    figure = drawn(weak_and_strong(), weight_scale=ep.PER_STATE)

    assert widest(figure, 0, 4) > widest(figure, 4, 4)

    assert widest(figure, 4, 4) == pytest.approx(
        ep.FAT_BASE_WIDTH + ep.FAT_SCALE * 0.15
    )


def test_both_scalings_are_offered() -> None:

    assert set(ep.WEIGHT_SCALES) == {ep.PER_SERIES, ep.PER_STATE}


def test_an_all_zero_series_does_not_divide_by_zero() -> None:
    """A selection with no weight anywhere is a normal thing to ask."""

    figure = drawn(
        {"none": np.zeros((4, 40))}, weight_scale=ep.PER_SERIES
    )

    assert any("below" in label for label in legend_labels(figure))


# ------------------------------------------------------------
# The dispersion underneath
# ------------------------------------------------------------


def test_the_dispersion_goes_under_the_fat_bands() -> None:
    """
    Drawn first is not enough; matplotlib sorts by zorder.

    A grey dispersion painted over the projection would hide the thing
    the figure exists to show.
    """

    figure = drawn({"d": np.ones((4, 40))})

    background = [
        line
        for line in figure.axes[0].lines
        if line.get_color() == ep.BACKGROUND_COLOUR
    ]

    assert background

    assert all(
        line.get_zorder() < bands_on(figure)[0].get_zorder()
        for line in background
    )


def test_an_unprojected_band_keeps_its_own_colour() -> None:
    """The grey is there to recede behind a projection, not on its own."""

    colours = {
        line.get_color() for line in drawn({}).axes[0].lines
    }

    assert ep.BACKGROUND_COLOUR not in colours


def test_every_series_is_named() -> None:

    figure = drawn(
        {
            "dz2": np.ones((4, 40)),
            "dxy": np.ones((4, 40)),
            "dxz": np.ones((4, 40)),
        }
    )

    assert legend_labels(figure) == ["dz2", "dxy", "dxz"]


# ------------------------------------------------------------
# Layout
#
# A band figure is read the way the published ones are: panels divided
# by solid separators, the high-symmetry letters serving as the x axis,
# and the key outside the frame.
# ------------------------------------------------------------


def with_ticks():
    """A result carrying a k-path with named high-symmetry points."""

    result = path_result({"Os (d)": np.ones((4, 40))})

    result.ticks = [0.0, 0.5, 1.0]
    result.labels = ["GAMMA", "X", "M"]

    return result


def test_the_separators_are_solid_and_dark() -> None:
    """
    They are panel edges, not grid lines.

    Each stretch between two of them is a straight line in reciprocal
    space; a band meeting one is turning a corner, not continuing. A
    faint hairline reads as decoration and loses that.
    """

    figure = Figure()

    ep.draw_band(figure, with_ticks())

    separators = [
        line
        for line in figure.axes[0].lines
        if line.get_color() == ep.SEPARATOR_COLOUR
    ]

    assert len(separators) == 3

    assert all(line.get_linestyle() == "-" for line in separators)


def test_the_separators_go_over_the_bands() -> None:

    figure = Figure()

    ep.draw_band(figure, with_ticks())

    separators = [
        line
        for line in figure.axes[0].lines
        if line.get_color() == ep.SEPARATOR_COLOUR
    ]

    assert all(
        line.get_zorder() > bands_on(figure)[0].get_zorder()
        for line in separators
    )


def test_gamma_is_written_as_the_letter() -> None:

    figure = Figure()

    ep.draw_band(figure, with_ticks())

    assert figure.axes[0].get_xticklabels()[0].get_text() == "Γ"


def test_there_is_no_x_label_under_the_symmetry_letters() -> None:
    """They are the axis; a caption under them says nothing further."""

    figure = Figure()

    ep.draw_band(figure, with_ticks())

    assert figure.axes[0].get_xlabel() == ""


def test_a_mesh_keeps_its_x_label() -> None:
    """There are no letters there, so the axis has to say what it is."""

    figure = Figure()

    ep.draw_band(figure, mesh_result({"d": np.ones((4, 4))}))

    assert "k-point index" in figure.axes[0].get_xlabel()


def test_the_legend_sits_outside_the_frame() -> None:
    """
    A band figure fills its frame edge to edge.

    That is what the tight x limits are for, so a legend inside it
    necessarily covers bands.
    """

    figure = Figure()

    ep.draw_band(figure, with_ticks())

    legend = figure.axes[0].get_legend()

    assert legend.get_bbox_to_anchor() is not None

    assert not legend.get_frame_on()


def test_the_x_axis_is_tight_to_the_path() -> None:
    """No margin: the first and last k-point are the frame."""

    figure = Figure()

    ep.draw_band(figure, with_ticks())

    assert figure.axes[0].get_xlim() == (0.0, 1.0)


def test_the_first_three_colours_stay_separable() -> None:
    """
    Blue, amber, green.

    Most figures are carried by three series, and these three are the
    set that survives greyscale and the common forms of colour
    blindness -- which a red-and-green pair does not.
    """

    assert len(set(ep.SERIES_COLOURS[:3])) == 3

    def luminance(hexcode: str) -> float:
        text = hexcode.lstrip("#")
        red, green, blue = (
            int(text[i:i + 2], 16) / 255 for i in (0, 2, 4)
        )
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    lums = sorted(luminance(c) for c in ep.SERIES_COLOURS[:3])

    assert lums[1] - lums[0] > 0.1
    assert lums[2] - lums[1] > 0.1

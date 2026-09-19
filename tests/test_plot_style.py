"""
Tests for the figure design layer.

This code styles a figure it did not draw, so almost everything here is
about telling one artist from another. Two rules carry it, and both fail
silently when wrong.

**Decoration is not data.** A Fermi level, a zero line and a
high-symmetry separator are ``Line2D`` objects exactly like the curves.
Widening them along with the data produces a figure that is merely ugly;
*recolouring* them produces one where the reference line is the same
colour as a series and reads as one.

**Reset has to be exact.** The design window applies from the original
every time a control moves, so a reset that leaves anything behind makes
dragging a slider back and forth drift away from where it started.
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from qvista.visualization import plot_style as ps


# ------------------------------------------------------------
# A figure shaped like the ones QVista draws
# ------------------------------------------------------------


def figure_with(labels: bool = True):
    """Two curves, one unlabelled trace, and two decorations."""

    figure = Figure()

    axes = figure.add_subplot(111)

    x = np.linspace(-5.0, 5.0, 50)

    axes.plot(
        x, np.sin(x), linewidth=0.7, color="#111111",
        label="alpha" if labels else None,
    )
    axes.plot(
        x, np.cos(x), linewidth=0.7, color="#222222",
        label="beta" if labels else None,
    )

    # Data, but not a legend entry.
    axes.plot(x, 0.3 * x, linewidth=0.7, color="#333333")

    # Decoration: pinned to the frame in one direction.
    axes.axhline(0.0, color="grey", linewidth=0.9, linestyle="--")
    axes.axvline(0.0, color="0.8", linewidth=0.6)

    axes.set_title("original")
    axes.set_xlabel("original x")
    axes.set_ylabel("original y")

    if labels:
        axes.legend()

    return figure, axes


# ------------------------------------------------------------
# Doing nothing
# ------------------------------------------------------------


def test_a_fresh_style_changes_nothing() -> None:
    """
    The panel applies a style to every figure, opened or not.

    A default that touched anything would restyle every plot in the
    application whether or not the user ever pressed Design.
    """

    assert not ps.PlotStyle().touches_anything


def test_applying_a_fresh_style_leaves_the_figure_alone() -> None:

    figure, axes = figure_with()

    before = [
        (line.get_linewidth(), line.get_color(), line.get_linestyle())
        for line in axes.lines
    ]

    limits = (axes.get_xlim(), axes.get_ylim())

    ps.apply(figure, ps.PlotStyle())

    after = [
        (line.get_linewidth(), line.get_color(), line.get_linestyle())
        for line in axes.lines
    ]

    assert before == after
    assert (axes.get_xlim(), axes.get_ylim()) == limits


def test_an_empty_figure_is_survivable() -> None:
    """Panels are styled before anything has been drawn into them."""

    ps.apply(Figure(), ps.PlotStyle(line_width=3.0))

    ps.reset(Figure())


def test_a_missing_style_is_survivable() -> None:

    ps.apply(Figure(), None)


# ------------------------------------------------------------
# Data against decoration
# ------------------------------------------------------------


def test_decoration_is_recognised_by_its_transform() -> None:

    _figure, axes = figure_with()

    assert [ps.is_data(line, axes) for line in axes.lines] == [
        True, True, True, False, False
    ]


def test_line_width_reaches_the_data_only() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(line_width=3.0))

    assert [line.get_linewidth() for line in axes.lines] == [
        3.0, 3.0, 3.0, 0.9, 0.6
    ]


def test_line_style_reaches_the_data_only() -> None:
    """
    The Fermi line is dashed on purpose.

    Setting every line solid would remove the one visual cue that says
    it is a reference rather than a result.
    """

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(line_style="Dotted"))

    assert axes.lines[0].get_linestyle() == ":"
    assert axes.lines[3].get_linestyle() == "--"


def test_colour_never_reaches_decoration() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(palette="Colourblind safe"))

    assert axes.lines[3].get_color() == "grey"
    assert axes.lines[4].get_color() == "0.8"


# ------------------------------------------------------------
# Colour
# ------------------------------------------------------------


def test_the_palette_is_handed_out_along_the_series() -> None:
    """In legend order, because that is what a reader matches against."""

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(palette="Colourblind safe"))

    colours = ps.PALETTES["Colourblind safe"]

    assert axes.lines[0].get_color() == colours[0]
    assert axes.lines[1].get_color() == colours[1]


def test_an_unlabelled_trace_keeps_its_colour_beside_labelled_ones() -> None:
    """
    A background trace is not a series.

    Where a figure has both, the unlabelled one is a reference the
    drawing code coloured deliberately -- the grey dispersion under a
    fat band, for instance.
    """

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(palette="Bright"))

    assert axes.lines[2].get_color() == "#333333"


def test_a_band_structure_becomes_one_colour() -> None:
    """
    Hundreds of anonymous lines, all drawn the same colour.

    They are one series -- that is what drawing them identically says --
    so they take one palette colour between them, not three hundred.
    Leaving them alone instead would make the palette do nothing on the
    figure most likely to want recolouring.
    """

    figure = Figure()
    axes = figure.add_subplot(111)

    for index in range(50):
        axes.plot([0, 1], [index, index + 1], color="#2980b9")

    axes.axhline(0.0, color="grey", linestyle="--")

    ps.apply(figure, ps.PlotStyle(palette="Bright"))

    first = ps.PALETTES["Bright"][0]

    assert {line.get_color() for line in axes.lines[:50]} == {first}

    # Still not the decoration.
    assert axes.lines[50].get_color() == "grey"


def test_a_spin_pair_keeps_one_colour() -> None:
    """
    A spin-polarised DOS draws each species twice and labels it once.

    The two halves are one series, and the drawing code says so by
    giving them the same colour. Recolouring only what the legend names
    would leave spin down on the old palette while spin up moved to the
    new one, so the mirrored curve stops matching the one above it.
    """

    figure = Figure()
    axes = figure.add_subplot(111)

    x = np.linspace(-1.0, 1.0, 10)

    # Up is labelled, down is not; both are the species' colour.
    axes.plot(x, x, color="#aa0000", label="Co")
    axes.plot(x, -x, color="#aa0000")

    axes.plot(x, 2 * x, color="#0000aa", label="O")
    axes.plot(x, -2 * x, color="#0000aa")

    ps.apply(figure, ps.PlotStyle(palette="Colourblind safe"))

    colours = ps.PALETTES["Colourblind safe"]

    assert axes.lines[0].get_color() == colours[0]
    assert axes.lines[1].get_color() == colours[0]

    assert axes.lines[2].get_color() == colours[1]
    assert axes.lines[3].get_color() == colours[1]


def test_an_unlabelled_group_of_its_own_is_left_alone() -> None:
    """
    The grey dispersion drawn under a fat band.

    It shares its colour with nothing labelled, so no legend entry
    points at it and it keeps the neutral colour it was given.
    """

    figure = Figure()
    axes = figure.add_subplot(111)

    axes.plot([0, 1], [0, 1], color="0.55")
    axes.scatter([0, 1], [0, 1], color="#aa0000", label="Co d")

    ps.apply(figure, ps.PlotStyle(palette="Bright"))

    assert axes.lines[0].get_color() == "0.55"


def test_a_palette_shorter_than_the_series_wraps() -> None:

    figure = Figure()
    axes = figure.add_subplot(111)

    for index in range(12):
        axes.plot([0, 1], [index, index], label=f"s{index}")

    ps.apply(figure, ps.PlotStyle(palette="QVista"))

    colours = ps.PALETTES["QVista"]

    assert axes.lines[0].get_color() == colours[0]
    assert axes.lines[len(colours)].get_color() == colours[0]


def test_as_drawn_leaves_every_colour_alone() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(palette=ps.AS_DRAWN))

    assert axes.lines[0].get_color() == "#111111"


# ------------------------------------------------------------
# Ranges
# ------------------------------------------------------------


def test_both_ends_of_a_range_can_be_set() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(x_min=-2.0, x_max=2.0))

    assert axes.get_xlim() == (-2.0, 2.0)


def test_one_end_can_be_set_alone() -> None:
    """
    "Start the y axis at zero, keep the top wherever it is" is the
    common request, and needs the other end to stay automatic.
    """

    figure, axes = figure_with()

    top = axes.get_ylim()[1]

    ps.apply(figure, ps.PlotStyle(y_min=0.0))

    assert axes.get_ylim() == (0.0, top)


def test_no_range_leaves_the_limits_alone() -> None:

    figure, axes = figure_with()

    before = axes.get_xlim()

    ps.apply(figure, ps.PlotStyle(line_width=2.0))

    assert axes.get_xlim() == before


# ------------------------------------------------------------
# Text and legend
# ------------------------------------------------------------


def test_empty_text_keeps_the_plots_own() -> None:
    """
    The drawing functions write informative titles.

    An override that applied on empty would blank every one of them the
    moment the design window opened.
    """

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(line_width=2.0))

    assert axes.get_title() == "original"
    assert axes.get_xlabel() == "original x"


def test_text_can_be_overridden() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(title="mine", y_label="mine too"))

    assert axes.get_title() == "mine"
    assert axes.get_ylabel() == "mine too"

    # The one not named is untouched.
    assert axes.get_xlabel() == "original x"


def test_the_legend_can_be_hidden() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(legend="Hide"))

    assert not axes.get_legend().get_visible()


def test_the_legend_can_be_shown_where_there_was_none() -> None:

    figure = Figure()
    axes = figure.add_subplot(111)
    axes.plot([0, 1], [0, 1], label="only")

    assert axes.get_legend() is None

    ps.apply(figure, ps.PlotStyle(legend="Show"))

    assert axes.get_legend() is not None


def test_showing_a_legend_with_nothing_to_name_is_survivable() -> None:
    """An unlabelled figure has no handles; asking for one must not fail."""

    figure = Figure()
    axes = figure.add_subplot(111)
    axes.plot([0, 1], [0, 1])

    ps.apply(figure, ps.PlotStyle(legend="Show"))

    assert axes.get_legend() is None


# ------------------------------------------------------------
# Markers
# ------------------------------------------------------------


def test_marker_size_is_a_radius_not_an_area() -> None:
    """
    A scatter takes an area, which is not what a size control means.

    Passing the number straight through would make doubling it
    quadruple the apparent size.
    """

    figure = Figure()
    axes = figure.add_subplot(111)

    axes.scatter([0, 1], [0, 1], s=[4.0, 4.0])

    ps.apply(figure, ps.PlotStyle(marker_size=3.0))

    assert list(axes.collections[0].get_sizes()) == [9.0, 9.0]


# ------------------------------------------------------------
# Reset
# ------------------------------------------------------------


def snapshot(axes):
    """Everything the style layer is allowed to touch."""

    return (
        [
            (
                line.get_linewidth(),
                line.get_color(),
                line.get_linestyle(),
                line.get_alpha(),
            )
            for line in axes.lines
        ],
        axes.get_title(),
        axes.get_xlabel(),
        axes.get_ylabel(),
        axes.get_xlim(),
        axes.get_ylim(),
    )


def test_reset_puts_everything_back() -> None:

    figure, axes = figure_with()

    before = snapshot(axes)

    ps.apply(
        figure,
        ps.PlotStyle(
            line_width=3.0,
            line_style="Dotted",
            alpha=0.4,
            palette="Bright",
            x_min=-1.0,
            x_max=1.0,
            y_min=-0.5,
            y_max=0.5,
            title="changed",
            x_label="changed",
            y_label="changed",
        ),
    )

    assert snapshot(axes) != before

    ps.reset(figure)

    assert snapshot(axes) == before


def test_applying_twice_does_not_compound() -> None:
    """
    The panel resets before every re-apply, but the layer must be safe
    on its own: a remembered original is written once, not overwritten
    by the value the last apply put there.
    """

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(line_width=2.0))
    ps.apply(figure, ps.PlotStyle(line_width=5.0))

    ps.reset(figure)

    assert axes.lines[0].get_linewidth() == 0.7


def test_reset_on_an_untouched_figure_is_harmless() -> None:

    figure, axes = figure_with()

    before = snapshot(axes)

    ps.reset(figure)

    assert snapshot(axes) == before


# ------------------------------------------------------------
# What it says about itself
# ------------------------------------------------------------


def test_a_fresh_style_summarises_to_the_export_setting_only() -> None:

    rows = dict(ps.summary(ps.PlotStyle()))

    assert list(rows) == ["Export dpi"]


def test_the_summary_names_what_is_active() -> None:

    rows = dict(
        ps.summary(
            ps.PlotStyle(line_width=2.0, palette="Bright", y_min=0.0)
        )
    )

    assert rows["Line width"] == "2"
    assert rows["Colours"] == "Bright"
    assert rows["Y range"] == "0 to auto"


# ------------------------------------------------------------
# Choices
# ------------------------------------------------------------


def test_every_palette_has_colours_except_as_drawn() -> None:

    for name, colours in ps.PALETTES.items():

        if name == ps.AS_DRAWN:
            assert colours == ()
        else:
            assert len(colours) >= 6


def test_the_colourblind_palette_is_okabe_ito() -> None:
    """Pinned, because a substitution here would be invisible."""

    assert ps.PALETTES["Colourblind safe"][:3] == (
        "#0072b2", "#e69f00", "#009e73"
    )


def test_every_line_style_is_one_matplotlib_knows() -> None:

    for name, dashes in ps.LINE_STYLES.items():

        if name == ps.AS_DRAWN:
            assert dashes == ""
        else:
            assert dashes in ("-", "--", "-.", ":")


def test_a_scatter_counts_as_data() -> None:
    """
    A fat band is a scatter, and scatters hide their positions.

    ``get_transform`` on a scatter describes the marker shape, not where
    the markers are; the positions live in ``get_offset_transform``.
    Checking only the first makes every projected band structure look
    like decoration and exempts it from restyling entirely.
    """

    figure = Figure()
    axes = figure.add_subplot(111)

    scatter = axes.scatter([0, 1], [0, 1], label="Co d")

    assert ps.is_data(scatter, axes)

    assert ps.series_of(figure) == [scatter]


def test_a_filled_region_counts_as_data() -> None:
    """stackplot and fill_between, used by the DOS figures."""

    figure = Figure()
    axes = figure.add_subplot(111)

    filled = axes.fill_between([0, 1], [0, 1], label="O p")

    assert ps.is_data(filled, axes)


def test_a_shaded_span_is_still_decoration() -> None:
    """
    ``axvspan`` marks the imaginary-frequency region on a phonon DOS.

    It is pinned to the frame vertically, so it must not be swept into
    the palette alongside the curves.
    """

    figure = Figure()
    axes = figure.add_subplot(111)

    axes.plot([0, 1], [0, 1], color="#aa0000", label="curve")

    span = axes.axvspan(0.2, 0.4, color="#dddddd")

    assert not ps.is_data(span, axes)

    ps.apply(figure, ps.PlotStyle(palette="Bright"))

    # A patch, so get_facecolor is one RGBA tuple rather than a list.
    assert span.get_facecolor()[:3] == pytest.approx(
        (0xDD / 255,) * 3, abs=0.01
    )


# ------------------------------------------------------------
# Subscripts and superscripts
#
# matplotlib will typeset them, but only inside $...$, and asking
# somebody to write $\mathrm{Li}_2$ to label an axis is not reasonable.
# ------------------------------------------------------------


def test_a_formula_subscripts_each_number_separately() -> None:
    """
    ``Li_2CO_3`` is Li two, C, O three.

    Taking the whole run up to a space instead -- which is the obvious
    implementation -- gives one subscript reading "2CO_3", which is
    both wrong and renders without complaint.
    """

    assert ps.to_mathtext("Li_2CO_3") == "Li$_{2}$CO$_{3}$"


def test_one_character_unless_braced() -> None:
    """LaTeX's own rule, so it is the one people already know."""

    assert ps.to_mathtext("Li_{12}Mn_2") == "Li$_{12}$Mn$_{2}$"


def test_a_superscript_works_the_same_way() -> None:

    assert ps.to_mathtext("eV^{-1}") == "eV$^{-1}$"

    assert ps.to_mathtext("x^2") == "x$^{2}$"


def test_the_fermi_level_reads_naturally() -> None:

    assert ps.to_mathtext("E - E_F (eV)") == "E - E$_{F}$ (eV)"


def test_plain_text_is_untouched() -> None:

    assert ps.to_mathtext("Density of states") == "Density of states"


def test_existing_mathtext_is_left_alone() -> None:
    """
    Anyone who wrote it by hand meant it.

    Re-processing would double-wrap and produce a parse error, turning
    a working label into a crash.
    """

    original = r"E - E$_F$  (eV)"

    assert ps.to_mathtext(original) == original


def test_an_underscore_can_be_escaped() -> None:
    """For a label that genuinely contains one."""

    assert ps.to_mathtext(r"file\_name") == "file_name"


def test_a_trailing_marker_is_harmless() -> None:
    """Half-typed text is the normal state of a text box."""

    assert ps.to_mathtext("trailing_") == "trailing_"

    assert ps.to_mathtext("_") == "_"


def test_an_unbalanced_brace_is_left_alone() -> None:
    """Rather than guessing where it was meant to close."""

    assert ps.to_mathtext("Li_{2") == "Li_{2"


def test_every_conversion_renders() -> None:
    """
    The real test: matplotlib has to accept what comes out.

    A mathtext parse error raises at draw time, long after the label was
    typed, and takes the whole figure with it.
    """

    figure = Figure()
    axes = figure.add_subplot(111)
    axes.plot([0, 1], [0, 1])

    for text in (
        "Li_2CO_3", "E_F", "eV^{-1}", "a_b c_d", "sigma_{xx}",
        "LiCoO_2", "trailing_", "Li_{2", r"file\_name",
    ):
        axes.set_xlabel(ps.to_mathtext(text))

        figure.canvas.draw()


def test_labels_are_converted_when_applied() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(title="LiCoO_2"))

    assert axes.get_title() == "LiCoO$_{2}$"


def test_conversion_can_be_switched_off() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(title="LiCoO_2", mathtext=False))

    assert axes.get_title() == "LiCoO_2"


# ------------------------------------------------------------
# Weight
# ------------------------------------------------------------


def test_bold_reaches_the_title_and_labels() -> None:

    figure, axes = figure_with()

    ps.apply(
        figure,
        ps.PlotStyle(title_weight="bold", label_weight="bold"),
    )

    assert axes.title.get_fontweight() == "bold"
    assert axes.xaxis.label.get_fontweight() == "bold"
    assert axes.yaxis.label.get_fontweight() == "bold"


def test_bold_reaches_the_tick_labels() -> None:
    """
    ``tick_params`` has no weight, so these are set one by one.

    Assuming it did would make the tick box do nothing at all.
    """

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(tick_weight="bold"))

    assert axes.get_xticklabels()[0].get_fontweight() == "bold"


def test_each_piece_of_text_bolds_on_its_own() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(title_weight="bold"))

    assert axes.title.get_fontweight() == "bold"
    assert axes.xaxis.label.get_fontweight() != "bold"


def test_weight_is_restored_by_reset() -> None:

    figure, axes = figure_with()

    before = axes.title.get_fontweight()

    ps.apply(figure, ps.PlotStyle(title_weight="bold"))
    ps.reset(figure)

    assert axes.title.get_fontweight() == before


def test_bold_appears_in_the_summary() -> None:

    rows = dict(ps.summary(ps.PlotStyle(title_weight="bold")))

    assert rows["Bold"] == "title"


# ------------------------------------------------------------
# Colours typed by hand
# ------------------------------------------------------------


def test_a_hex_colour_is_recognised() -> None:

    for text in ("#ff0000", "#F00", "#ff0000ff", "  #00aa00  "):
        assert ps.is_hex(text)


def test_anything_else_is_not() -> None:

    for text in ("red", "#gg0000", "#ff00", "", "0072b2ff00"):
        assert not ps.is_hex(text)


def test_colours_split_on_commas_or_spaces() -> None:

    good, bad = ps.parse_colours("#ff0000, #00ff00 #0000ff")

    assert good == ["#ff0000", "#00ff00", "#0000ff"]
    assert bad == []


def test_a_missing_hash_is_added() -> None:
    """
    Copying a code out of a paper very often loses the leading #.

    Refusing that is pedantry rather than validation.
    """

    good, _bad = ps.parse_colours("0072b2, e69f00")

    assert good == ["#0072b2", "#e69f00"]


def test_a_bad_colour_is_reported_not_dropped() -> None:
    """
    Silently skipping one shifts the palette by a place.

    Every series after it then changes colour, with nothing on screen
    saying why.
    """

    good, bad = ps.parse_colours("#ff0000, notacolour, #0000ff")

    assert good == ["#ff0000", "#0000ff"]
    assert bad == ["notacolour"]


def test_empty_text_gives_nothing() -> None:

    assert ps.parse_colours("   ") == ([], [])


def test_the_custom_palette_uses_the_typed_colours() -> None:

    style = ps.PlotStyle(
        palette=ps.CUSTOM, custom_colours=("#ff0000", "#00ff00")
    )

    assert style.colours() == ("#ff0000", "#00ff00")


def test_typed_colours_are_ignored_unless_custom_is_chosen() -> None:
    """So switching back to a named palette does what it says."""

    style = ps.PlotStyle(
        palette="Bright", custom_colours=("#ff0000",)
    )

    assert style.colours() == ps.PALETTES["Bright"]


def test_custom_with_nothing_typed_leaves_colours_alone() -> None:

    figure, axes = figure_with()

    ps.apply(figure, ps.PlotStyle(palette=ps.CUSTOM))

    assert axes.lines[0].get_color() == "#111111"


def test_typed_colours_reach_the_figure() -> None:

    figure, axes = figure_with()

    ps.apply(
        figure,
        ps.PlotStyle(
            palette=ps.CUSTOM,
            custom_colours=("#ff0000", "#00ff00"),
        ),
    )

    assert axes.lines[0].get_color() == "#ff0000"
    assert axes.lines[1].get_color() == "#00ff00"

    # And still not the decoration.
    assert axes.lines[3].get_color() == "grey"


def test_the_summary_counts_the_typed_colours() -> None:

    rows = dict(
        ps.summary(
            ps.PlotStyle(
                palette=ps.CUSTOM, custom_colours=("#ff0000", "#00ff00")
            )
        )
    )

    assert rows["Colours"] == "2 custom"


# ----------------------------------------------------------------
# The legend follows the series it names
# ----------------------------------------------------------------


def legend_colours(axes) -> list[str]:
    """The colour of each key drawn in the legend."""

    legend = axes.get_legend()

    return [
        ps._as_text(handle.get_color())
        for handle in ps._legend_handles(legend)
    ]


def data_colours(axes) -> list[str]:

    return [ps._as_text(line.get_color()) for line in axes.lines]


def proxy_legend_figure():
    """
    A figure built the way the band plots build one.

    Two data lines, and a legend made from *proxy* artists rather than
    from the lines themselves. The band figures do this deliberately: a
    fat band's width is its weight, so a swatch cut from one would be a
    hairline rather than a colour anyone could match.
    """

    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    figure = Figure()

    axes = figure.add_subplot(111)

    axes.plot([0.0, 1.0], [0.0, 1.0], color="#4C72B0")
    axes.plot([0.0, 1.0], [1.0, 0.0], color="#DD8452")

    axes.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", color="#4C72B0",
                label="Ni d",
            ),
            Line2D(
                [], [], marker="o", linestyle="", color="#DD8452",
                label="O p",
            ),
        ]
    )

    return figure, axes


def recoloured():

    return ps.PlotStyle(
        palette=ps.CUSTOM, custom_colours=("#ff0000", "#00ff00")
    )


def test_a_recoloured_figure_recolours_its_legend():
    """
    The bug this fixes. A legend is not a live view of the plot --
    matplotlib builds its keys once, from the handles it was given -- so
    recolouring the data left every band figure naming the right series
    in the wrong colours.
    """

    figure, axes = proxy_legend_figure()

    ps.apply(figure, recoloured())

    assert legend_colours(axes) == ["#ff0000", "#00ff00"]


def test_the_legend_matches_the_lines_it_names():

    figure, axes = proxy_legend_figure()

    ps.apply(figure, recoloured())

    assert legend_colours(axes) == data_colours(axes)


def test_proxy_keys_are_reached_at_all():
    """
    Proxies are not in ``axes.lines``, so nothing that walks the axes
    ever finds them however carefully it looks.
    """

    figure, axes = proxy_legend_figure()

    assert len(axes.lines) == 2
    assert len(ps._legend_handles(axes.get_legend())) == 2

    ps.apply(figure, recoloured())

    assert "#4C72B0" not in legend_colours(axes)


def test_resetting_puts_the_legend_back():

    figure, axes = proxy_legend_figure()

    ps.apply(figure, recoloured())
    ps.reset(figure)

    # Lower case: colour keys are canonicalised, so that a collection's
    # RGBA array and a proxy's hex string name the same series.
    assert legend_colours(axes) == ["#4c72b0", "#dd8452"]


def test_a_legend_built_from_the_lines_themselves_also_follows():
    """
    The other way of legending a figure. Both have to work, since a DOS
    labels its own curves while a band structure uses proxies.
    """

    from matplotlib.figure import Figure

    figure = Figure()

    axes = figure.add_subplot(111)

    axes.plot([0.0, 1.0], [0.0, 1.0], color="#4C72B0", label="up")
    axes.plot([0.0, 1.0], [1.0, 0.0], color="#DD8452", label="down")

    axes.legend()

    ps.apply(figure, recoloured())

    assert legend_colours(axes) == data_colours(axes)


def test_a_figure_with_no_legend_is_untouched():
    """A plain band structure is hundreds of anonymous lines."""

    from matplotlib.figure import Figure

    figure = Figure()

    axes = figure.add_subplot(111)

    axes.plot([0.0, 1.0], [0.0, 1.0], color="#4C72B0")

    ps.apply(figure, recoloured())

    assert axes.get_legend() is None


def test_the_handles_are_found_whatever_matplotlib_calls_them():
    """
    Renamed between versions, and the older spelling is gone in the
    newer ones.
    """

    _, axes = proxy_legend_figure()

    assert ps._legend_handles(axes.get_legend())


def test_nothing_breaks_on_a_legend_with_no_handles():

    class Empty:
        pass

    assert ps._legend_handles(Empty()) == []


# ----------------------------------------------------------------
# Fat bands: collections named by Line2D proxies
# ----------------------------------------------------------------


def fatband_figure(count: int = 3):
    """
    A figure built the way the projected band plots build one.

    The bands are ``LineCollection`` objects -- that is what a fat band
    is -- and the legend is made from ``Line2D`` proxies. The two hand
    their colours back in different forms, which is the whole point of
    this fixture.
    """

    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    palette = ("#2b4b9b", "#f5a623", "#3aa757")

    figure = Figure()

    axes = figure.add_subplot(111)

    proxies = []

    for index in range(count):

        colour = palette[index]

        # Several bands per series, as a real projection has.
        for band in range(3):

            points = np.column_stack(
                [np.linspace(0.0, 3.0, 20), np.full(20, float(band))]
            )

            axes.add_collection(
                LineCollection([points], colors=colour, alpha=0.6)
            )

        proxies.append(
            Line2D(
                [], [], marker="o", linestyle="", markersize=8,
                color=colour, label=f"series {index}",
            )
        )

    axes.legend(handles=proxies)

    axes.autoscale()

    return figure, axes


def collection_colours(axes) -> list[str]:

    return [ps._as_text(item.get_color()) for item in axes.collections]


def test_a_collection_and_its_proxy_share_one_key():
    """
    The bug behind the bug. A fat band hands back an RGBA array and its
    legend proxy hands back the hex string it was given; keyed literally
    those are different series, so the legend never followed the bands.
    """

    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    band = LineCollection(
        [np.array([[0.0, 0.0], [1.0, 1.0]])], colors="#4C72B0", alpha=0.6
    )

    proxy = Line2D([], [], color="#4C72B0")

    assert ps._original_colour(band) == ps._original_colour(proxy)


def test_colour_keys_are_canonical_hex():

    assert ps._as_text("#4C72B0") == "#4c72b0"
    assert ps._as_text((0.298, 0.4471, 0.6902, 1.0)) == "#4c72b0"


def test_alpha_does_not_split_a_series():
    """A fat band is translucent and its legend proxy is not."""

    assert ps._as_text((0.1, 0.2, 0.3, 0.5)) == ps._as_text(
        (0.1, 0.2, 0.3, 1.0)
    )


def test_an_unrecognisable_colour_does_not_raise():

    assert ps._as_text("not a colour")


def test_greyscaling_a_projected_band_greyscales_its_legend():
    """
    The reported case: the bands went grey and the key stayed in
    colour.
    """

    figure, axes = fatband_figure()

    ps.apply(
        figure,
        ps.PlotStyle(
            palette=ps.CUSTOM,
            custom_colours=("#111111", "#333333", "#555555"),
        ),
    )

    assert legend_colours(axes) == ["#111111", "#333333", "#555555"]


def test_every_band_of_a_series_moves_with_its_key(count=3):

    figure, axes = fatband_figure()

    ps.apply(
        figure,
        ps.PlotStyle(
            palette=ps.CUSTOM,
            custom_colours=("#111111", "#333333", "#555555"),
        ),
    )

    drawn = collection_colours(axes)

    # Three bands per series, and each series is one colour.
    for index, colour in enumerate(("#111111", "#333333", "#555555")):
        assert drawn[index * 3 : index * 3 + 3] == [colour] * 3


def test_resetting_a_projected_band_restores_both():

    figure, axes = fatband_figure()

    before_bands = collection_colours(axes)
    before_keys = legend_colours(axes)

    ps.apply(
        figure,
        ps.PlotStyle(
            palette=ps.CUSTOM,
            custom_colours=("#111111", "#333333", "#555555"),
        ),
    )

    ps.reset(figure)

    assert collection_colours(axes) == before_bands
    assert legend_colours(axes) == before_keys

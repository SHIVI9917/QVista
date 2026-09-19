"""
Figures for the electronic structure tools.

Pure matplotlib onto a figure the caller owns, matching the phonon and
dynamics plotting modules, so the same code serves the embedded canvas
and a headless export.

Spin-polarised runs are drawn with the minority channel mirrored below
the axis.  That is the convention in the literature and it is not
decoration: two channels drawn on top of each other are unreadable, and
the asymmetry between them is usually the thing being looked for.

Every curve goes through :mod:`qvista.visualization.smoothing` on its
way to the canvas, so a DOSCAR grid does not put a visible corner on the
top of a peak and a dispersion is drawn curving rather than faceted.
That is interpolation and not broadening -- the spline passes through
every calculated point -- so nothing on these figures has moved from
where the calculation put it.
"""

from __future__ import annotations

from typing import Any, Sequence

from matplotlib.lines import Line2D

from qvista.visualization.smoothing import spline, spline_runs

#: Cycled across whatever is being resolved.
#:
#: Blue, amber, green first. Those three carry most figures -- a metal
#: d band, an oxygen p band and whatever else is being asked about --
#: and they are the set that stays separable in greyscale and under the
#: common forms of colour blindness, which a red-first cycle does not.
SERIES_COLOURS = (
    "#2b4b9b",
    "#f5a623",
    "#3aa757",
    "#8e44ad",
    "#c0392b",
    "#16a085",
    "#d35400",
    "#7f8c8d",
)

TOTAL_COLOUR = "#4a5568"
FERMI_COLOUR = "#6b7280"


#: How a curve is drawn between its calculated points.
CURVE_SPLINE = "spline"
CURVE_LINE = "line"

#: In the order they are offered. The first is the default.
CURVE_STYLES: dict[str, str] = {
    CURVE_SPLINE: "Beta-spline (smooth)",
    CURVE_LINE: "Line (as computed)",
}


def _curve(energies, values, curve: str = CURVE_SPLINE):
    """
    The points to draw, given the chosen curve style.

    The spline passes through every calculated point rather than
    approximating them, so a peak keeps its height and its position and
    the area under it is unchanged -- which matters, because that area
    is a number the table beside the figure reports.

    The plain line is the honest rendering when the grid is what is
    being judged. A DOSCAR grid is coarse enough that a sharp peak is
    drawn as a visible corner, and smoothing it away can hide that the
    NEDOS was too low. Anyone checking convergence wants to see the
    corners.
    """

    import numpy as np

    energies = np.asarray(energies, dtype=float)
    values = np.asarray(values, dtype=float)

    if curve == CURVE_LINE:
        return energies, values

    return spline(energies, values)


def _style(axes) -> None:
    """Ticks pointing inward, matching the other QVista figures."""

    axes.tick_params(direction="in", top=True, right=True)


def _fermi_line(axes, vertical: bool = True) -> None:
    """The zero of energy, which everything here is measured from."""

    if vertical:
        axes.axvline(
            0.0, color=FERMI_COLOUR, linewidth=0.9, linestyle="--"
        )
    else:
        axes.axhline(
            0.0, color=FERMI_COLOUR, linewidth=0.9, linestyle="--"
        )


def draw_dos(
    figure,
    result,
    title: str = "",
    with_total: bool = False,
    fill: bool = True,
    window: tuple[float, float] | None = None,
    curve_style: str = CURVE_SPLINE,
) -> None:
    """
    Density of states against energy, referenced to the Fermi level.

    One line per curve, with the minority spin channel mirrored below
    the axis when the run is spin polarised.

    ``with_total`` is off by default. The total used to be drawn on top
    of every projected figure, which is a different quantity on a
    different scale -- on a forty-atom cell it is an order of magnitude
    larger than any projection and flattens all of them. It has a mode
    of its own now.
    """

    import numpy as np

    figure.clear()

    axes = figure.add_subplot(111)

    labels = list(dict.fromkeys(curve.label for curve in result.curves))

    colours = {
        label: SERIES_COLOURS[index % len(SERIES_COLOURS)]
        for index, label in enumerate(labels)
    }

    seen: set[str] = set()

    for curve in result.curves:

        values = np.asarray(curve.values, dtype=float)

        # Minority channel below the axis: the usual convention, and
        # the only way two channels stay legible on one panel.
        sign = -1.0 if curve.spin == "down" else 1.0

        energies, values = _curve(
            curve.energies, sign * values, curve_style
        )

        axes.plot(
            energies,
            values,
            linewidth=1.1,
            color=colours[curve.label],
            label=curve.label if curve.label not in seen else None,
        )

        if fill:
            axes.fill_between(
                energies,
                0.0,
                values,
                color=colours[curve.label],
                alpha=0.18,
            )

        seen.add(curve.label)

    if with_total and result.totals:

        # Per spin, and mirrored the same way the projections are, so
        # the two halves of the figure stay comparable.
        for index, total in enumerate(result.totals):

            sign = -1.0 if total.spin == "down" else 1.0

            grid, values = _curve(
                total.energies,
                sign * np.asarray(total.values, dtype=float),
                curve_style,
            )

            axes.plot(
                grid,
                values,
                linewidth=0.9,
                color=TOTAL_COLOUR,
                linestyle="--",
                label="total" if index == 0 else None,
            )

    _fermi_line(axes)

    axes.set_xlabel(r"E - E$_F$  (eV)")
    axes.set_ylabel("DOS (states/eV)")

    if window is not None:
        axes.set_xlim(*window)

    _style(axes)

    if axes.get_legend_handles_labels()[0]:
        axes.legend(fontsize=7, loc="upper right", framealpha=0.85)

    if title:
        axes.set_title(title, fontsize=10)

    figure.tight_layout()


#: How the eigenvalues themselves are drawn.
#:
#: ``"line"`` joins consecutive k-points into branches, which is what a
#: dispersion is. ``"points"`` marks each state on its own, which is the
#: only honest rendering of a sampling mesh whose k-points are in no
#: particular order.
LINE_STYLE = "line"
POINT_STYLE = "points"

BAND_STYLES: dict[str, str] = {
    LINE_STYLE: "Join the k-points into branches.",
    POINT_STYLE: "One marker per state, nothing joined.",
}

#: Width of a fat band where the weight is zero, in points. Not zero:
#: a series that is present but weak should still trace its branch.
FAT_BASE_WIDTH = 0.5

#: Extra width at full weight. With the base above, a fully weighted
#: band is 6.5 points across.
FAT_SCALE = 6.0

#: The dispersion under the fat bands.
BACKGROUND_COLOUR = "#c8c8c8"
BACKGROUND_WIDTH = 0.7

#: The high-symmetry separators.
#:
#: Solid black and full height, drawn over the bands rather than under
#: them. They are the panel edges of the figure: each stretch between
#: two of them is a straight line in reciprocal space, and a band
#: meeting one is turning a corner, not continuing. A faint grey
#: hairline reads as a grid line, which is the opposite of what they
#: are.
SEPARATOR_COLOUR = "#000000"
SEPARATOR_WIDTH = 1.0

#: Font sizes on a band figure. Larger than matplotlib's defaults,
#: because the high-symmetry letters *are* the x axis and have to stay
#: readable at whatever size the figure ends up in a document.
TICK_FONT = 13
LABEL_FONT = 13
LEGEND_FONT = 11

#: How a projected weight is turned into a width.
#:
#: ``PER_SERIES`` divides each series by its own largest value, so every
#: one reaches full width somewhere and a weak series is still legible.
#: That is what a fat-band figure is normally read for -- *where* along
#: the path a character sits, not how much of it there is.
#:
#: ``PER_STATE`` leaves the weights as the fraction of each state on the
#: selection, so widths compare between series and against unity. It is
#: the honest one for "how much", and it makes a series that never
#: exceeds a fifth of any state permanently thin.
PER_SERIES = "series"
PER_STATE = "state"

WEIGHT_SCALES: dict[str, str] = {
    PER_SERIES: "Scale each series to its own maximum.",
    PER_STATE: "Leave as the fraction of each state.",
}


def _fatband(axes, x, y, weight, colour) -> None:
    """
    One band drawn as a line whose width follows the weight along it.

    Built as a :class:`~matplotlib.collections.LineCollection` of one
    segment per adjacent pair of k-points. A single ``plot`` call cannot
    do this: ``linewidth`` there is one number for the whole line.

    The width of segment *i* comes from the weight at its left-hand
    point, which is why ``weight`` is sliced one short. Round caps and
    joins are what stop the result reading as a chain of tapered sticks
    rather than as one band.

    The dispersion and the weight are splined together onto one grid, so
    a band that curves is drawn curving and its width follows it point
    for point. Splining them apart would put the two out of step. The
    spline passes through every calculated k-point, so a weight of one
    is still drawn at full width there; between them it can overshoot
    slightly, and a negative width is clipped away rather than drawn.
    """

    import numpy as np

    from matplotlib.collections import LineCollection

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)

    if x.size < 2:
        return

    stacked = np.column_stack([y, weight])

    # One collection per straight stretch of the path: joining across a
    # high-symmetry corner would interpolate between two branches.
    for grid, pair in spline_runs(x, stacked):

        if grid.size < 2:
            continue

        values = pair[:, 0]
        widths = np.clip(pair[:, 1], 0.0, None)

        points = np.array([grid, values]).T.reshape(-1, 1, 2)

        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        axes.add_collection(
            LineCollection(
                segments,
                colors=colour,
                linewidths=FAT_BASE_WIDTH + FAT_SCALE * widths[:-1],
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
        )


def draw_band(
    figure,
    result,
    title: str = "",
    window: tuple[float, float] = (-6.0, 6.0),
    marker_scale: float = 55.0,
    threshold: float = 0.02,
    style: str | None = None,
    weight_scale: str = PER_SERIES,
) -> None:
    """
    Band structure, with the projection drawn as a fat band.

    The dispersion goes down first in light grey, and each projected
    series is drawn over it as a line whose *width* follows the weight
    along the path -- thick where that character dominates, tapering to
    a hairline where it does not.

    Why a varying-width line rather than markers
    --------------------------------------------
    Both encode the weight, but a scatter breaks the band into a cloud
    of separate dots and the eye stops following the branch. The width
    of a continuous line keeps the dispersion readable *and* carries the
    weight on top of it, which is the whole point of the figure.

    The line is built from one segment per pair of adjacent k-points,
    each given its own width, with round caps and joins so the segments
    read as one band rather than as a chain of tapered sticks.

    Markers are still used for a self-consistent mesh, where there are
    no adjacent k-points to join and a segment between two of them would
    be a line across the Brillouin zone.

    ``style`` picks between the two. Left unset it follows the result.
    ``weight_scale`` picks how a weight becomes a width; see
    :data:`WEIGHT_SCALES`.
    """

    import numpy as np

    figure.clear()

    axes = figure.add_subplot(111)

    if style is None:
        style = (
            LINE_STYLE
            if getattr(result, "is_path", True)
            else POINT_STYLE
        )

    plain = "#2980b9"

    for spin, segments in result.energies.items():

        dashes = "-" if spin == "up" else "--"

        for bands in segments:

            values = np.asarray(bands, dtype=float)

            colour = BACKGROUND_COLOUR if result.weights else plain

            if style == POINT_STYLE:

                distances = np.asarray(result.distances[0], dtype=float)

                # One scatter for the whole block rather than one per
                # band: a mesh run can carry three hundred bands, and
                # three hundred calls is seconds of drawing time.
                axes.scatter(
                    np.tile(distances, values.shape[0]),
                    values.ravel(),
                    s=4.0,
                    marker="_" if spin == "up" else "|",
                    color=colour,
                    linewidths=0.8,
                    alpha=0.75,
                )

                continue

            for row in values:

                # Splined per straight stretch of the path, so the
                # curvature of a branch is drawn rather than faceted,
                # and no line is invented across a corner.
                for grid, smooth in spline_runs(result.distances[0], row):

                    axes.plot(
                        grid,
                        smooth,
                        linewidth=(
                            BACKGROUND_WIDTH if result.weights else 0.7
                        ),
                        color=colour,
                        linestyle=dashes,
                        # Under the fat bands, always.
                        zorder=1,
                    )

    keys: list[Any] = []

    for index, (label, weights) in enumerate(result.weights.items()):

        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]

        values = np.asarray(
            next(iter(result.energies.values()))[0], dtype=float
        )

        share = np.asarray(weights, dtype=float)

        distances = np.asarray(result.distances[0], dtype=float)

        if weight_scale == PER_SERIES:

            # Each series against its own largest weight, so a species
            # that never takes more than a fifth of any state is still
            # visible where it does sit.
            peak = float(np.abs(share).max()) if share.size else 0.0

            scaled = share / peak if peak > 0.0 else share

        else:
            scaled = share

        drew = bool(scaled.size) and float(scaled.max()) > threshold

        if style == POINT_STYLE:

            # A mesh has no adjacent k-points, so there is nothing to
            # join; markers are the only honest rendering.
            for band in range(min(values.shape[0], scaled.shape[0])):

                keep = scaled[band] > threshold

                if not keep.any():
                    continue

                axes.scatter(
                    distances[keep],
                    values[band][keep],
                    s=scaled[band][keep] * marker_scale,
                    color=colour,
                    alpha=0.6,
                    edgecolors="none",
                    zorder=2,
                )

        else:

            for band in range(min(values.shape[0], scaled.shape[0])):

                if float(scaled[band].max()) <= threshold:
                    continue

                _fatband(
                    axes,
                    distances,
                    values[band],
                    scaled[band],
                    colour,
                )

        # The key is built from proxies rather than from the drawn
        # artists.
        #
        # A fat band's width *is* the weight, so a swatch taken from one
        # would be whatever that band happened to weigh -- often a
        # hairline. And the label used to ride on band 0, the deepest,
        # most core-like state, whose weight is under the threshold for
        # almost any selection: it was skipped, and the series lost its
        # only legend entry. Between them those left every atom-wise and
        # orbital-resolved band figure with no usable legend at all.
        keys.append(
            Line2D(
                [],
                [],
                # A filled dot, whichever way the bands themselves are
                # drawn. A fat band has no single width to show in a
                # key, and colour is what the reader matches on.
                marker="o",
                linestyle="",
                markersize=8,
                color=colour,
                label=label if drew else f"{label} (below {threshold:g})",
            )
        )

    _fermi_line(axes, vertical=False)

    for tick in result.ticks:
        axes.axvline(
            tick,
            color=SEPARATOR_COLOUR,
            linewidth=SEPARATOR_WIDTH,
            zorder=3,
        )

    if result.ticks:
        axes.set_xticks(result.ticks)
        axes.set_xticklabels(
            [
                "Γ" if label.upper() in ("GAMMA", "\\GAMMA") else label
                for label in result.labels
            ],
            fontsize=TICK_FONT,
        )

    axes.set_ylabel(r"E - E$_F$  (eV)", fontsize=LABEL_FONT)
    axes.set_ylim(*window)

    axes.tick_params(axis="y", labelsize=TICK_FONT - 2)

    # No x label where there are high-symmetry letters: they *are* the
    # axis, and "Wave vector" underneath them says nothing further.
    axes.set_xlabel(
        "" if result.ticks else getattr(result, "xlabel", ""),
        fontsize=9,
    )

    distances = np.asarray(result.distances[0], dtype=float)

    if distances.size:

        if style == POINT_STYLE:
            # A little air either side, so the first and last k-point are
            # not drawn on the frame.
            pad = max(0.5, 0.02 * float(distances[-1] - distances[0]))

            axes.set_xlim(
                float(distances[0]) - pad, float(distances[-1]) + pad
            )

            # k-point indices are integers; matplotlib would otherwise
            # offer halves.
            if not result.ticks:
                from matplotlib.ticker import MaxNLocator

                axes.xaxis.set_major_locator(
                    MaxNLocator(integer=True, nbins=10)
                )

        elif distances[-1] > distances[0]:
            # Guarded: a run with one k-point has no span, and asking
            # matplotlib for a zero-width axis warns and then invents
            # one anyway.
            axes.set_xlim(float(distances[0]), float(distances[-1]))

    _style(axes)

    if keys or axes.get_legend_handles_labels()[0]:

        # Outside, to the right. A band structure fills its frame edge
        # to edge -- that is what the tight x limits are for -- so a
        # legend inside it necessarily covers bands. Out here it covers
        # nothing and stays legible however the figure is scaled.
        axes.legend(
            handles=keys or None,
            fontsize=LEGEND_FONT,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            handletextpad=0.4,
            borderaxespad=0.0,
        )

    if title:
        axes.set_title(title, fontsize=10)

    figure.tight_layout()


#: How solid one selection's cloud is drawn.
#:
#: Low enough that three overlapping selections are still individually
#: readable, and high enough that one on its own is not a ghost. The
#: overlaps are the content of this figure, so this is the number that
#: decides whether it works.
CLOUD_ALPHA = 0.28


#: The two ways a band centre is drawn.
PLOT_PDOS = "pdos"
PLOT_SCHEMATIC = "schematic"

#: In the order they are offered. The first is the default.
CENTRE_PLOTS: dict[str, str] = {
    PLOT_PDOS: "Actual DOSCAR PDOS",
    PLOT_SCHEMATIC: "Norskov Schematic",
}

#: One colour per series, cycled in the order the series were asked
#: for. A series is an element *and* a contribution: keyed on the
#: contribution alone, Co p-band and B p-band came out the same green,
#: so a figure comparing two elements' p states had two curves and one
#: colour and nothing to tell them apart by.
#:
#: Chosen to stay distinguishable in the common forms of colour
#: blindness, and to survive being printed in grey.
CENTRE_PALETTE: tuple[str, ...] = (
    "#4c72b0",
    "#dd8452",
    "#55a868",
    "#c44e52",
    "#8172b3",
    "#937860",
    "#da8bc3",
    "#8c8c8c",
    "#ccb974",
    "#64b5cd",
)


def centre_colours(curves) -> dict[tuple[str, str], str]:
    """
    A colour for each series, in the order they were asked for.

    Keyed by ``(element, contribution)`` so that the same series keeps
    its colour between the two diagrams and between redraws, and so
    that two elements sharing a contribution do not share a colour.
    """

    seen: list[tuple[str, str]] = []

    for curve in curves:

        key = (curve.element, curve.contribution)

        if key not in seen:
            seen.append(key)

    return {
        key: CENTRE_PALETTE[index % len(CENTRE_PALETTE)]
        for index, key in enumerate(seen)
    }

#: Savitzky-Golay defaults. A DOSCAR grid is noisy enough that the
#: balance point is obvious in the number and invisible in the curve
#: without some smoothing; the filter is used rather than a running
#: mean because it flattens noise without moving peaks.
SMOOTH_WINDOW = 21
SMOOTH_ORDER = 3

#: How far a normalised PDOS curve reaches across the panel.
PDOS_SCALE = 0.7

#: How solid the area under a real curve is drawn.
CENTRE_FILL_ALPHA = 0.30

#: How far under its centre line a label sits, in points. Enough to
#: clear the rule without drifting far enough to be read against the
#: wrong one.
LABEL_DROP = 3

#: The schematic's Gaussian width, in eV, and how far it reaches. The
#: schematic is a drawing of the numbers, not of the states, so its
#: width is a setting rather than the second moment.
SCHEMATIC_WIDTH = 0.30
SCHEMATIC_REACH = 0.85
SCHEMATIC_ALPHA = 0.55


def _centre_colour(curve, colours) -> str:
    """The colour assigned to one series."""

    return colours.get(
        (curve.element, curve.contribution), CENTRE_PALETTE[0]
    )


def _smoothed(values, window: int, order: int):
    """
    A Savitzky-Golay pass, with the window forced into legality.

    The filter needs an odd window, longer than the polynomial order and
    no longer than the data. Any of those violated raises, so they are
    fixed here rather than left for the user to discover by choosing a
    slider value the data cannot take.
    """

    import numpy as np
    from scipy.signal import savgol_filter

    values = np.asarray(values, dtype=float)

    if values.size < 3:
        return values

    window = int(window)

    if window >= values.size:
        window = values.size - 1

    if window % 2 == 0:
        window -= 1

    smallest = order + 2

    if smallest % 2 == 0:
        smallest += 1

    if window < smallest:
        return values

    return savgol_filter(values, window, order)


def _fermi_mark(axes) -> None:
    """The Fermi level, the origin the curves rise from, and the label."""

    axes.axhline(0.0, color="black", linewidth=1.8)
    axes.axvline(0.0, color="black", linewidth=1.2)

    axes.text(
        0.02,
        0.0,
        r"$E_F=0$",
        transform=axes.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _missing_note(axes, curves) -> None:
    """Name the selections that turned out to have no states."""

    missing = [
        f"{curve.element} {curve.contribution}"
        for curve in curves
        if not curve.defined
    ]

    if not missing:
        return

    axes.annotate(
        "no states: " + ", ".join(missing),
        xy=(0.0, -0.09),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        color="0.45",
    )


def draw_band_centre(
    figure,
    curves,
    plot: str = PLOT_PDOS,
    window: tuple[float, float] | None = None,
    smooth_window: int = SMOOTH_WINDOW,
    smooth_order: int = SMOOTH_ORDER,
    scale: float = PDOS_SCALE,
    band_width: float = SCHEMATIC_WIDTH,
) -> None:
    """
    A band-centre diagram, either from the real states or as a schematic.

    Two drawings of the same numbers, and both are worth having.

    The PDOS diagram draws each contribution's projected density and
    puts its centre across it, so the centre is visibly the balance
    point of the states beside it -- and a tail dragging it away from
    the peak is visible too, which is the usual reason a d-band centre
    is not where somebody expected.

    The schematic throws the states away and draws every band as a
    Gaussian of the same set width at its centre. Nothing is left but
    the energies, which is what makes it the figure to compare four
    materials with: the real densities differ in shape for reasons that
    have nothing to do with where the band sits.
    """

    if plot == PLOT_SCHEMATIC:
        _draw_schematic(figure, curves, window, band_width)
        return

    _draw_pdos(figure, curves, window, smooth_window, smooth_order, scale)


def _draw_pdos(
    figure, curves, window, smooth_window, smooth_order, scale
) -> None:
    """The real projected states, normalised, with centres across them."""

    import numpy as np

    figure.clear()

    axes = figure.add_subplot(111)

    origin = 0.0

    drawn = [curve for curve in curves if curve.defined]

    colours = centre_colours(curves)

    for curve in drawn:

        colour = _centre_colour(curve, colours)

        values = np.nan_to_num(
            np.asarray(curve.dos, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        values = np.maximum(values, 0.0)

        smooth = np.maximum(
            _smoothed(values, smooth_window, smooth_order), 0.0
        )

        peak = float(np.max(smooth)) if smooth.size else 0.0

        if peak <= 0.0:
            continue

        # Each curve to its own maximum. The absolute height of a
        # projected DOS depends on how many atoms were summed and on the
        # PAW potential, neither of which is being compared here.
        across = origin + (smooth / peak) * scale

        axes.fill_betweenx(
            curve.energies,
            origin,
            across,
            color=colour,
            alpha=CENTRE_FILL_ALPHA,
        )

        axes.plot(across, curve.energies, color=colour, linewidth=2.0)

        axes.axhline(
            curve.centre, color=colour, linestyle="--", linewidth=1.2
        )

        # Just under the line rather than across it. Centred on the
        # centre, the text sat on the very line it names, and two
        # selections close in energy put one label through the other's
        # rule. Offset in points, so it stays clear at any zoom.
        axes.annotate(
            f"{curve.element} {curve.contribution}",
            xy=(scale + 0.04, curve.centre),
            xytext=(0, -LABEL_DROP),
            textcoords="offset points",
            fontsize=10,
            ha="left",
            va="top",
            color=colour,
        )

    _fermi_mark(axes)

    if window is not None:
        axes.set_ylim(*window)

    elif drawn:
        axes.set_ylim(
            min(float(np.min(c.energies)) for c in drawn),
            max(float(np.max(c.energies)) for c in drawn),
        )

    axes.set_xlim(-0.03, scale + 0.65)

    axes.set_ylabel(r"$E-E_F$ (eV)", fontsize=14)
    axes.set_xlabel("Normalized PDOS", fontsize=12)

    axes.set_title(
        "Band Center Diagram - DOSCAR", fontsize=15, fontweight="bold"
    )

    axes.grid(axis="y", linestyle=":", alpha=0.18)

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    _missing_note(axes, curves)

    figure.tight_layout()


def _draw_schematic(figure, curves, window, band_width) -> None:
    """
    Every band as a Gaussian of the same width at its own centre.

    All of them start from the same horizontal reference, so the only
    thing that varies down the page is the energy -- which is the point
    of the schematic. It says nothing about the shape of the states and
    is not meant to.
    """

    import numpy as np

    figure.clear()

    axes = figure.add_subplot(111)

    origin = 0.0

    low, high = window if window is not None else (-6.0, 1.0)

    energy = np.linspace(low, high, 1500)

    colours = centre_colours(curves)

    for curve in curves:

        if not curve.defined:
            continue

        colour = _centre_colour(curve, colours)

        profile = np.exp(
            -0.5 * ((energy - curve.centre) / band_width) ** 2
        )

        peak = float(np.max(profile))

        if peak > 0.0:
            profile = profile / peak

        across = origin + SCHEMATIC_REACH * profile

        axes.fill_betweenx(
            energy, origin, across, color=colour, alpha=SCHEMATIC_ALPHA
        )

        axes.plot(across, energy, color=colour, linewidth=1.5)

        # The centre, drawn across the band rather than the whole panel:
        # a level, and only as wide as the band it belongs to.
        axes.plot(
            [origin, origin + SCHEMATIC_REACH],
            [curve.centre, curve.centre],
            color=colour,
            linewidth=1.5,
        )

        axes.annotate(
            f"{curve.element} {curve.contribution}",
            xy=(SCHEMATIC_REACH + 0.08, curve.centre),
            xytext=(0, -LABEL_DROP),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="top",
            color=colour,
        )

    axes.axhline(0.0, color="black", linewidth=1.5)
    axes.axvline(origin, color="black", linewidth=1.0)

    axes.text(
        0.02,
        0.0,
        r"$E_F=0$",
        transform=axes.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    axes.set_ylim(low, high)
    axes.set_xlim(-0.08, 1.55)

    axes.set_ylabel(r"$E-E_F$ (eV)", fontsize=14)

    # Nothing is measured across a schematic.
    axes.set_xticks([])
    axes.set_xlabel("")

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["bottom"].set_visible(False)

    axes.grid(axis="y", linestyle=":", alpha=0.20)

    axes.set_title(
        "Norskov Band-Center Diagram", fontsize=14, fontweight="bold"
    )

    _missing_note(axes, curves)

    figure.tight_layout()

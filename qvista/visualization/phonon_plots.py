"""
Phonon band structure and DOS figures.

Pure matplotlib: these functions draw onto a ``Figure`` that the caller
owns, so they work identically in the embedded Qt canvas and in a headless
export or test.

Every figure is annotated with the force field used, so a screenshot cannot
be mistaken for a DFT result.

Colouring a dispersion
----------------------
A phonon band structure is ``3N`` lines and, drawn in one colour, says
nothing about which line is which. Two colourings are offered besides
the plain one, and both answer a question a reader actually asks.

*Acoustic / optical* separates the three lowest branches from the rest.
That is the division the physics turns on -- the acoustic branches go
to zero at Gamma and carry the sound velocities, the optical ones do
not -- and it costs a two-entry legend.

*Per branch* gives every branch its own colour, for following one
across a crossing. The legend is dropped once there are more branches
than a legend can usefully name; the colours still separate them.

The scheme is a parameter rather than a fixed choice because the same
figure is read for different things, and because a legend that names
forty-eight lines is worse than no legend at all.
"""

from __future__ import annotations

from typing import Any, Sequence

from qvista.visualization.smoothing import spline

#: Rendered instead of the literal label "GAMMA".
GAMMA_SYMBOL = "Γ"

#: Export default matching publication use.
EXPORT_DPI = 600

FREQUENCY_LABEL = "Frequency (THz)"

# ------------------------------------------------------------
# How the branches are coloured
# ------------------------------------------------------------

BAND_ACOUSTIC = "acoustic"
BAND_BRANCH = "branch"
BAND_SINGLE = "single"

#: In the order they are offered. The first is the default.
BAND_SCHEMES: dict[str, str] = {
    BAND_ACOUSTIC: "Acoustic / optical",
    BAND_BRANCH: "One colour per branch",
    BAND_SINGLE: "One colour",
}

#: Three of them, in any crystal: one per translation of the whole
#: solid. They are the branches that go to zero frequency at Gamma.
ACOUSTIC_BRANCHES = 3

ACOUSTIC_COLOUR = "#c0392b"
OPTICAL_COLOUR = "#2980b9"

#: What the plain scheme draws with, and what every figure used before
#: there was a choice. Written as hex rather than as ``tab:blue`` --
#: the same colour -- because Qt has to parse it too, for the swatch on
#: the colour button, and matplotlib's own names mean nothing to it.
DEFAULT_BAND_COLOUR = "#1f77b4"

#: Cycled across branch index in the per-branch scheme.
BRANCH_COLOURS = (
    "#2980b9",
    "#c0392b",
    "#27ae60",
    "#8e44ad",
    "#d68910",
    "#16a085",
    "#d35400",
    "#2c3e50",
)

#: Past this many branches the legend is dropped: naming every line in
#: a forty-eight branch dispersion fills the axes and identifies
#: nothing. The colours stay, since following one branch through a
#: crossing is what they are for.
MAX_BRANCH_LEGEND = 12

#: matplotlib's own way of saying "not in the legend". Every branch
#: after the first that would carry a given entry gets this, so the
#: legend names each group once rather than once per segment.
NO_LEGEND = "_nolegend_"


def _branch_style(
    branch: int,
    count: int,
    scheme: str,
    colour: str,
) -> tuple[str, str]:
    """
    The colour and legend entry for one branch.

    The label is :data:`NO_LEGEND` for every branch that should not add
    an entry of its own -- the second and third acoustic branches share
    the first one's, and nothing is named at all in the plain scheme.
    """

    if scheme == BAND_ACOUSTIC:

        if branch < ACOUSTIC_BRANCHES:
            return (
                ACOUSTIC_COLOUR,
                "acoustic" if branch == 0 else NO_LEGEND,
            )

        return (
            OPTICAL_COLOUR,
            "optical" if branch == ACOUSTIC_BRANCHES else NO_LEGEND,
        )

    if scheme == BAND_BRANCH:

        return (
            BRANCH_COLOURS[branch % len(BRANCH_COLOURS)],
            f"branch {branch + 1}"
            if count <= MAX_BRANCH_LEGEND
            else NO_LEGEND,
        )

    return colour or DEFAULT_BAND_COLOUR, NO_LEGEND


def _pretty(label: str) -> str:
    """Render point labels, turning GAMMA into the Greek letter."""

    return "|".join(
        GAMMA_SYMBOL if part == "GAMMA" else part
        for part in label.split("|")
    )


def band_ticks(
    distances: Sequence[Any],
    labels: Sequence[str],
    connections: Sequence[bool],
) -> tuple[list[float], list[str]]:
    """
    Work out where the high-symmetry labels sit along the band axis.

    Where the path breaks, the two labels either side of the break are
    joined with a bar, which is the usual way of showing a discontinuity.
    """

    if not distances:
        return [], []

    positions = [float(distances[0][0])]

    out = [_pretty(labels[0])] if labels else [""]

    index = 1

    for segment_index, segment in enumerate(distances):

        positions.append(float(segment[-1]))

        if index >= len(labels):
            out.append("")
            continue

        current = labels[index]
        index += 1

        connected = (
            connections[segment_index]
            if segment_index < len(connections)
            else True
        )

        if not connected and index < len(labels):
            current = f"{current}|{labels[index]}"
            index += 1

        out.append(_pretty(current))

    return positions, out


def _style_axes(axes) -> None:
    """Ticks pointing inward, as in the reference figures."""

    axes.tick_params(direction="in", top=True, right=True)


def draw_band(
    figure,
    band: dict[str, Any],
    force_field: str = "",
    title: str = "",
    scheme: str = BAND_ACOUSTIC,
    colour: str = DEFAULT_BAND_COLOUR,
    legend: bool = True,
) -> None:
    """Draw a phonon band structure onto ``figure``."""

    figure.clear()

    axes = figure.add_subplot(111)

    _plot_band_panel(axes, band, scheme, colour, legend)

    axes.set_ylabel(FREQUENCY_LABEL)

    if title:
        axes.set_title(title)

    _annotate(figure, force_field)

    figure.tight_layout()


def draw_dos(
    figure,
    dos: dict[str, Any],
    force_field: str = "",
    title: str = "",
) -> None:
    """Draw a total phonon density of states onto ``figure``."""

    figure.clear()

    axes = figure.add_subplot(111)

    frequencies, values = spline(
        dos["frequency_points"], dos["total_dos"]
    )

    axes.plot(frequencies, values, linewidth=1.2)

    axes.axvline(0.0, color="black", linewidth=0.8, linestyle="-")

    axes.set_xlabel(FREQUENCY_LABEL)
    axes.set_ylabel("DOS (states/THz)")

    _style_axes(axes)

    if title:
        axes.set_title(title)

    _annotate(figure, force_field)

    figure.tight_layout()


def draw_band_with_dos(
    figure,
    band: dict[str, Any],
    dos: dict[str, Any],
    force_field: str = "",
    title: str = "",
    scheme: str = BAND_ACOUSTIC,
    colour: str = DEFAULT_BAND_COLOUR,
    legend: bool = True,
) -> None:
    """
    Draw the band structure beside the DOS, sharing the frequency axis.

    The shared y-range is the point of this figure: features in the DOS line
    up with the flat bands that produce them.
    """

    figure.clear()

    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[3, 1],
        wspace=0.05,
    )

    band_axes = figure.add_subplot(grid[0])
    dos_axes = figure.add_subplot(grid[1], sharey=band_axes)

    _plot_band_panel(band_axes, band, scheme, colour, legend)

    band_axes.set_ylabel(FREQUENCY_LABEL)

    # ----------------------------------------------------------
    # DOS panel, rotated so frequency runs vertically
    # ----------------------------------------------------------

    dos_frequencies, dos_values = spline(
        dos["frequency_points"], dos["total_dos"]
    )

    dos_axes.plot(
        dos_values,
        dos_frequencies,
        linewidth=1.2,
    )

    dos_axes.axhline(0.0, color="black", linewidth=0.8, linestyle="-")

    dos_axes.set_xlabel("DOS")

    dos_axes.tick_params(labelleft=False)

    _style_axes(dos_axes)

    if title:
        figure.suptitle(title)

    _annotate(figure, force_field)


#: Cycled across elements in the projected band plot.
ELEMENT_COLOURS = (
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#8e44ad",
    "#d68910",
    "#16a085",
)

#: Largest marker area, in points squared, for a fully projected mode.
MAX_MARKER_AREA = 55.0


def draw_projected_band(
    figure,
    band: dict[str, Any],
    force_field: str = "",
    title: str = "",
    threshold: float = 0.02,
) -> None:
    """
    Draw a phonon band structure with element-projected weights.

    Each mode is drawn as markers whose area is proportional to how much of
    that mode belongs to a given element, over a faint line showing the
    dispersion itself. A mode dominated by the heavy sublattice and one
    dominated by the light sublattice become visually distinct, which is
    the point of projecting.

    Markers below ``threshold`` are skipped so the figure does not fill
    with dots representing almost nothing.
    """

    import numpy as np

    figure.clear()

    projections = band.get("projections") or {}

    axes = figure.add_subplot(111)

    # Dispersion underneath, so the bands stay readable where no element
    # dominates.
    for segment_distances, segment_frequencies in zip(
        band["distances"],
        band["frequencies"],
    ):
        smooth_x, smooth_y = spline(
            segment_distances, segment_frequencies
        )

        axes.plot(
            smooth_x,
            smooth_y,
            color="#b0b0b0",
            linewidth=0.6,
            zorder=1,
        )

    # ----------------------------------------------------------
    # Projected markers
    # ----------------------------------------------------------

    distances = np.concatenate(
        [np.asarray(segment) for segment in band["distances"]]
    )

    frequencies = np.concatenate(
        [np.asarray(segment) for segment in band["frequencies"]]
    )

    for index, (element, weights) in enumerate(sorted(projections.items())):

        weights = np.asarray(weights)

        if weights.shape != frequencies.shape:
            continue

        colour = ELEMENT_COLOURS[index % len(ELEMENT_COLOURS)]

        mask = weights > threshold

        if not mask.any():
            continue

        # np.repeat gives every band the same x, matching frequencies.
        x = np.repeat(
            distances[:, None],
            frequencies.shape[1],
            axis=1,
        )

        axes.scatter(
            x[mask],
            frequencies[mask],
            s=weights[mask] * MAX_MARKER_AREA,
            color=colour,
            alpha=0.65,
            edgecolors="none",
            label=element,
            zorder=2,
        )

    axes.axhline(0.0, color="black", linewidth=0.8)

    positions, tick_labels = band_ticks(
        band["distances"],
        band.get("labels", []),
        band.get("path_connections", []),
    )

    if positions:
        axes.set_xticks(positions)
        axes.set_xticklabels(tick_labels)

        for position in positions[1:-1]:
            axes.axvline(
                position,
                color="grey",
                linewidth=0.6,
            )

        axes.set_xlim(positions[0], positions[-1])

    axes.set_ylabel(FREQUENCY_LABEL)

    if projections:
        axes.legend(
            loc="upper right",
            frameon=True,
            fontsize=8,
            markerscale=1.4,
        )

    _style_axes(axes)

    if title:
        axes.set_title(title)

    _annotate(figure, force_field)

    figure.tight_layout()


def _plot_band_panel(
    axes,
    band: dict[str, Any],
    scheme: str = BAND_ACOUSTIC,
    colour: str = DEFAULT_BAND_COLOUR,
    legend: bool = True,
) -> None:
    """Shared band drawing for the standalone and combined figures."""

    import numpy as np

    distances = band["distances"]
    frequencies = band["frequencies"]

    labels = band.get("labels", [])
    connections = band.get("path_connections", [])

    # Drawn branch by branch rather than as one array per segment,
    # because a branch is the thing being coloured and the thing a
    # legend entry would name.
    for segment, (segment_distances, segment_frequencies) in enumerate(
        zip(distances, frequencies)
    ):
        x = np.asarray(segment_distances, dtype=float)
        y = np.asarray(segment_frequencies, dtype=float)

        if y.ndim == 1:
            y = y[:, None]

        # Splined once for the whole segment: every branch shares the
        # same q-points, so one grid serves all of them and the widths
        # stay in step.
        smooth_x, smooth_y = spline(x, y)

        if smooth_y.ndim == 1:
            smooth_y = smooth_y[:, None]

        for branch in range(y.shape[1]):

            branch_colour, entry = _branch_style(
                branch, y.shape[1], scheme, colour
            )

            axes.plot(
                smooth_x,
                smooth_y[:, branch],
                color=branch_colour,
                linewidth=1.0,
                # A branch runs through every segment of the path, so
                # only the first may carry the entry; otherwise the
                # legend repeats it once per high-symmetry line.
                label=entry if segment == 0 else NO_LEGEND,
            )

    # Zero line: everything below it is an imaginary mode.
    axes.axhline(0.0, color="black", linewidth=0.8, linestyle="-")

    positions, tick_labels = band_ticks(distances, labels, connections)

    if positions:

        axes.set_xticks(positions)
        axes.set_xticklabels(tick_labels)

        # Vertical guide at every high-symmetry point.
        for position in positions[1:-1]:
            axes.axvline(
                position,
                color="grey",
                linewidth=0.6,
                linestyle="-",
            )

        axes.set_xlim(positions[0], positions[-1])

    if legend and axes.get_legend_handles_labels()[0]:
        axes.legend(fontsize=7, loc="upper right", framealpha=0.85)

    _style_axes(axes)


def draw_dos_result(
    figure,
    result,
    force_field: str = "",
    title: str = "",
    stacked: bool = True,
    with_total: bool = True,
) -> None:
    """
    Draw a :class:`~qvista.services.phonon_dos.DosResult`.

    Projections are stacked by default, because they partition the same
    ``3N`` modes: stacking makes the total the outline of the stack, and
    the share each species holds is the thickness of its band.  Overlaid
    curves are offered too, since two curves that overlap heavily are
    easier to compare side by side than as slices of a stack.

    Imaginary modes come out of phonopy as negative frequencies.  They
    are shaded rather than hidden -- a DOS with weight below zero is the
    single most important thing the figure can say.
    """

    import numpy as np

    figure.clear()

    axes = figure.add_subplot(111)

    frequencies = np.asarray(result.frequencies, dtype=float)
    total = np.asarray(result.total, dtype=float)

    if result.curves and stacked:

        labels = list(result.curves)

        # All of them on one splined grid, so the layers of the stack
        # still add up to the same outline. A cubic can dip a hair below
        # zero between two points that are both zero; a negative layer
        # would be drawn under the stack, so it is clipped away.
        grid, stack = spline(
            frequencies,
            np.column_stack(
                [np.asarray(result.curves[label]) for label in labels]
            ),
        )

        axes.stackplot(
            grid,
            *np.clip(np.atleast_2d(stack.T), 0.0, None),
            labels=labels,
            colors=[
                ELEMENT_COLOURS[index % len(ELEMENT_COLOURS)]
                for index in range(len(labels))
            ],
            alpha=0.75,
        )

    elif result.curves:

        for index, (label, values) in enumerate(result.curves.items()):

            grid, smooth = spline(frequencies, np.asarray(values))

            axes.plot(
                grid,
                smooth,
                linewidth=1.1,
                color=ELEMENT_COLOURS[index % len(ELEMENT_COLOURS)],
                label=label,
            )

    if with_total or not result.curves:

        grid, smooth = spline(frequencies, total)

        axes.plot(
            grid,
            smooth,
            linewidth=1.2,
            color="black" if result.curves else None,
            label="total",
        )

    # ----------------------------------------------------------
    # Imaginary modes
    # ----------------------------------------------------------

    # Only where there are genuinely imaginary modes. phonopy's default
    # frequency range runs a little below the lowest mode, so a stable
    # structure also has negative points on its axis -- shading those
    # would put a red band on every figure.
    if getattr(result, "has_imaginary", False) and frequencies.size:

        axes.axvspan(
            float(frequencies.min()),
            0.0,
            color="#ef6076",
            alpha=0.13,
            zorder=0,
        )

    axes.axvline(0.0, color="black", linewidth=0.8)

    axes.set_xlabel(f"Frequency ({result.unit_label})")
    axes.set_ylabel(f"DOS (states/{result.unit_label})")

    axes.set_ylim(bottom=0.0)

    _style_axes(axes)

    if result.curves or with_total:
        axes.legend(fontsize=7, loc="upper right", framealpha=0.85)

    if title:
        axes.set_title(title)

    _annotate(figure, force_field)

    figure.tight_layout()


def draw_thermal(
    figure,
    thermal,
    force_field: str = "",
    title: str = "",
) -> None:
    """
    Vibrational free energy, entropy and heat capacity against temperature.

    Three panels rather than three curves on one: they are in two
    different units and differ by an order of magnitude, so a shared
    axis would flatten two of them into the baseline.  Cv carries its
    Dulong-Petit limit as a guide, which is the one line that says
    whether the mesh and the range are behaving.

    The first panel is the *vibrational* Helmholtz free energy and not
    the free energy of the material. It is what the phonons contribute
    and nothing else:

        F_vib(T) = E_ZPE + k_B T sum_i ln(1 - exp(-h w_i / k_B T))

    The static electronic energy is absent, because no total-energy
    calculation happens here -- these panels are integrated from force
    constants, and the DFT energy of the relaxed cell never enters. A
    free energy to compare against another phase is

        F(T) = E_DFT + F_vib(T)

    with E_DFT taken from the relaxation. Labelling this "free energy"
    invited exactly that number being quoted on its own, which is wrong
    by tens of eV per cell and wrong in a direction that looks
    plausible. Its value at T = 0 is the zero-point energy, which is the
    quickest way to confirm what is being plotted.
    """

    import numpy as np

    figure.clear()

    axes = figure.subplots(1, 3)

    temperatures = np.asarray(thermal.temperatures, dtype=float)

    panels = (
        (thermal.free_energy, "F$_{vib}$ (kJ/mol)", "#2980b9"),
        (thermal.entropy, "Entropy (J/K/mol)", "#27ae60"),
        (thermal.heat_capacity, "C$_V$ (J/K/mol)", "#c0392b"),
    )

    for panel, (values, label, colour) in zip(axes, panels):

        grid, smooth = spline(temperatures, np.asarray(values))

        panel.plot(grid, smooth, linewidth=1.3, color=colour)

        panel.set_xlabel("Temperature (K)")
        panel.set_ylabel(label)

        _style_axes(panel)

    limit = thermal.dulong_petit

    axes[2].axhline(
        limit,
        color="0.45",
        linewidth=0.9,
        linestyle="--",
    )

    axes[2].annotate(
        f"3NR = {limit:.1f}",
        xy=(0.97, limit),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.45",
    )

    if title:
        figure.suptitle(title, fontsize=10)

    _annotate(figure, force_field)

    figure.tight_layout()


def _annotate(figure, force_field: str) -> None:
    """Stamp the figure with how the forces were obtained."""

    if not force_field:
        return

    figure.text(
        0.995,
        0.005,
        force_field,
        ha="right",
        va="bottom",
        fontsize=6,
        color="grey",
    )

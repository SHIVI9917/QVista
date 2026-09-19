"""
Figures for NEB paths and molecular dynamics runs.

Pure matplotlib, so they work identically in the embedded canvas, in a
headless export, and in tests.
"""

from __future__ import annotations

BARRIER_COLOUR = "#c0392b"
PATH_COLOUR = "#2980b9"

POTENTIAL_COLOUR = "#2980b9"
TEMPERATURE_COLOUR = "#c0392b"

#: The mean, and the thermostat target. Both are references rather than
#: data, so both are the same understated dotted black.
REFERENCE_COLOUR = "black"
REFERENCE_STYLE = ":"
REFERENCE_WIDTH = 1.0


def _style(axes) -> None:
    """Ticks pointing inward, matching the other QVista figures."""

    axes.tick_params(direction="in", top=True, right=True)


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


# ------------------------------------------------------------
# NEB
# ------------------------------------------------------------


def draw_neb_profile(figure, result, title: str = "") -> None:
    """
    Draw the minimum energy path.

    The barrier is marked explicitly: it is the number the calculation
    exists to produce, so it should not have to be read off the axis.
    """

    import numpy as np

    figure.clear()

    axes = figure.add_subplot(111)

    energies = np.asarray(result.energies, dtype=float)

    if energies.size == 0:
        return

    # The reaction coordinate, in Angstrom, when the read gave one.
    # A NEB profile against image index is a plot against a label; the
    # distance along the path is the quantity the barrier belongs to,
    # and unevenly spaced images look unevenly spaced.
    dat = list(getattr(result, "dat", []) or [])

    if dat:
        indices = np.asarray([row[1] for row in dat], dtype=float)
        energies = np.asarray([row[2] for row in dat], dtype=float)
        axis_label = "Reaction coordinate (Å)"

    else:
        indices = np.arange(energies.size)
        axis_label = "Image"

    # A smooth guide through the images, with the images themselves
    # marked. The guide is a B-spline through the calculated energies:
    # a minimum energy path is a continuous function of the reaction
    # coordinate, and joining five images with straight lines draws the
    # saddle as a corner that the calculation never found. The spline
    # passes through every image, so the barrier read off the curve is
    # still the barrier in the table.
    spline_data = getattr(result, "spline", ()) or ()

    if len(spline_data) == 2 and len(spline_data[0]):

        # The force-informed spline: a cubic through each interval using
        # the energy *and* the gradient at both ends, which is what
        # VTST's nebspline draws. It knows where the saddle is between
        # two images; a spline through the energies alone only knows
        # where the images are.
        guide_x, guide_y = spline_data

    else:

        from qvista.visualization.smoothing import spline

        guide_x, guide_y = spline(indices, energies)

    axes.plot(
        guide_x,
        guide_y,
        "-",
        color=PATH_COLOUR,
        linewidth=1.4,
    )

    axes.plot(
        indices,
        energies,
        "o",
        color=PATH_COLOUR,
        markersize=5,
        linestyle="none",
    )

    saddle = int(result.saddle_index)

    if dat:
        # saddle_index counts image folders; the drawn arrays skip any
        # image that had no energy, so it has to be looked up rather
        # than used directly.
        wanted = [row[0] for row in dat]

        saddle = (
            wanted.index(saddle)
            if saddle in wanted
            else int(np.argmax(energies))
        )

    if 0 <= saddle < energies.size:

        # The x coordinate, not the array position. They were the same
        # thing while the axis was the image index and are not now that
        # it is a distance, which put the saddle marker between two
        # images rather than on one.
        at = float(indices[saddle])

        axes.plot(
            [at],
            [energies[saddle]],
            "o",
            color=BARRIER_COLOUR,
            markersize=9,
            zorder=3,
        )

        # The number goes in the key, not across the curve. Written into
        # the axes with an arrow it sat over the path it was measuring,
        # and it repeats what the title already says -- so it belongs
        # where a reader looks for a value rather than where the data is.
        from matplotlib.lines import Line2D

        axes.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=BARRIER_COLOUR,
                    marker="o",
                    linestyle="none",
                    markersize=8,
                    label=f"barrier {result.barrier_forward:.3f} eV",
                )
            ],
            loc="upper right",
            fontsize=9,
            frameon=True,
            framealpha=0.85,
        )

        axes.vlines(
            at,
            0.0,
            energies[saddle],
            color=BARRIER_COLOUR,
            linestyle=":",
            linewidth=0.9,
        )

    axes.axhline(0.0, color="black", linewidth=0.8)

    axes.set_xlabel(axis_label)
    axes.set_ylabel("Energy (eV)")

    axes.set_xticks(indices)

    _style(axes)

    axes.set_title(title or "Minimum energy path")

    _annotate(figure, getattr(result, "force_field", ""))

    figure.tight_layout()


# ------------------------------------------------------------
# Molecular dynamics
# ------------------------------------------------------------


def draw_md_traces(figure, result, title: str = "") -> None:
    """
    Draw potential energy and temperature against time.

    Two stacked panels sharing the time axis, because the questions
    asked of a run are read off one each: has the potential energy
    settled, and did the thermostat hold.

    Only the potential energy is drawn. The kinetic and total used to be
    on the same axis, and on a cell of any size they are separated by
    two or three orders of magnitude -- the total sits on top of the
    potential and the kinetic on the floor, so the panel spends its
    whole height on a gap and none of it on the structure in the curve
    that is actually being read. The other two are in the CSV, and the
    energy drift they answer for is in the summary rows.

    A dotted line marks the mean, which is what turns the curve from a
    wandering line into "equilibrated about here, with this much
    scatter".
    """

    import numpy as np

    figure.clear()

    times = np.asarray(result.times, dtype=float)

    if times.size == 0:
        return

    grid = figure.add_gridspec(2, 1, hspace=0.08, height_ratios=[3, 2])

    energy_axes = figure.add_subplot(grid[0])
    temperature_axes = figure.add_subplot(grid[1], sharex=energy_axes)

    # ----------------------------------------------------------
    # Potential energy
    # ----------------------------------------------------------

    potential = np.asarray(result.potential_energies, dtype=float)

    energy_axes.plot(
        times,
        potential,
        color=POTENTIAL_COLOUR,
        linewidth=1.1,
    )

    if potential.size:

        energy_axes.axhline(
            float(potential.mean()),
            color=REFERENCE_COLOUR,
            linestyle=REFERENCE_STYLE,
            linewidth=REFERENCE_WIDTH,
        )

    energy_axes.set_ylabel("Potential energy (eV)")
    energy_axes.tick_params(labelbottom=False)

    _style(energy_axes)

    # ----------------------------------------------------------
    # Temperature
    # ----------------------------------------------------------

    temperature_axes.plot(
        times,
        result.temperatures,
        color=TEMPERATURE_COLOUR,
        linewidth=1.1,
    )

    # The target, so a thermostat that is not holding is obvious. No
    # label on it: the axis already carries the number, and the text sat
    # over the trace at whatever height the target happened to be.
    if getattr(result, "ensemble", "") == "nvt":

        temperature_axes.axhline(
            result.temperature,
            color=REFERENCE_COLOUR,
            linestyle=REFERENCE_STYLE,
            linewidth=REFERENCE_WIDTH,
        )

    temperature_axes.set_xlabel("Time (fs)")
    temperature_axes.set_ylabel("T (K)")

    _style(temperature_axes)

    if title:
        energy_axes.set_title(title)

    _annotate(figure, getattr(result, "force_field", ""))


# ------------------------------------------------------------
# Mean squared displacement
# ------------------------------------------------------------

#: Cycled across the species curves.
SPECIES_COLOURS = (
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#8e44ad",
    "#d68910",
    "#16a085",
)

TOTAL_MSD_COLOUR = "#4a5568"


def draw_msd(
    figure,
    result,
    force_field: str = "",
    title: str = "",
    log_scale: bool = False,
    with_total: bool = True,
) -> None:
    """
    Mean squared displacement against lag time, with the fits drawn on.

    The fitted line is drawn only over the window it was fitted to, and
    extended to the origin as a dashed guide.  Showing the fit that way
    is the point of the figure: a straight line through the whole curve
    would look convincing whether or not the run ever reached the
    diffusive regime, and seeing where the fit starts and stops is what
    tells the reader it did.

    ``log_scale`` puts both axes on a log scale, where diffusion is a
    slope of one.  Anything shallower is an ion still rattling in its
    cage rather than moving through the structure, which is much easier
    to see here than on linear axes.
    """

    import numpy as np

    figure.clear()

    axes = figure.add_subplot(111)

    lags = np.asarray(result.lags, dtype=float)

    if lags.size == 0:
        return

    curves = list(result.curves.items())

    if with_total and not curves:
        curves = [("all", result.total)]

    for index, (label, values) in enumerate(curves):

        colour = SPECIES_COLOURS[index % len(SPECIES_COLOURS)]

        axes.plot(
            lags,
            np.asarray(values, dtype=float),
            linewidth=1.3,
            color=colour,
            label=label,
        )

        fit = result.fit_for(label)

        if fit is None:
            continue

        inside = (lags >= fit.fit_from) & (lags <= fit.fit_to)

        axes.plot(
            lags[inside],
            fit.slope * lags[inside] + fit.intercept,
            linewidth=2.2,
            color=colour,
            alpha=0.45,
            solid_capstyle="butt",
        )

    if with_total and result.curves:

        axes.plot(
            lags,
            np.asarray(result.total, dtype=float),
            linewidth=1.0,
            color=TOTAL_MSD_COLOUR,
            linestyle="--",
            label="all atoms",
        )

    if log_scale:

        axes.set_xscale("log")
        axes.set_yscale("log")

        # The reference the eye needs on log axes: pure diffusion is a
        # slope of one, whatever the diffusion coefficient.
        positive = lags[lags > 0]

        if positive.size:

            reference = np.asarray(result.total, dtype=float)[lags > 0]

            if reference.size and reference[reference > 0].size:
                anchor = float(reference[reference > 0][0])
                axes.plot(
                    positive,
                    anchor * positive / positive[0],
                    color="0.55",
                    linewidth=0.9,
                    linestyle=":",
                    label="slope 1",
                )

    axes.set_xlabel("Lag time (ps)")
    axes.set_ylabel(r"MSD ($\mathrm{\AA}^2$)")

    _style(axes)

    axes.legend(fontsize=7, loc="upper left", framealpha=0.85)

    if title:
        axes.set_title(title)

    _annotate(figure, force_field)

    figure.tight_layout()


def draw_msd_origins(figure, result, title: str = "") -> None:
    """
    How many time origins stand behind each point of the curve.

    Worth its own small figure because it is the thing that decides how
    far along the MSD is worth believing: the count falls off as the lag
    grows, and where it gets small the curve stops being an average of
    anything.
    """

    import numpy as np

    figure.clear()

    if result.origins is None:
        return

    axes = figure.add_subplot(111)

    axes.plot(
        np.asarray(result.lags, dtype=float),
        np.asarray(result.origins, dtype=float),
        linewidth=1.2,
        color=PATH_COLOUR,
    )

    axes.set_xlabel("Lag time (ps)")
    axes.set_ylabel("Time origins averaged")

    axes.set_ylim(bottom=0.0)

    _style(axes)

    if title:
        axes.set_title(title)

    figure.tight_layout()


def draw_relaxation(figure, result, title: str = "") -> None:
    """
    Energy and largest force against optimiser step.

    Two stacked panels sharing the step axis, because the two questions
    asked of a relaxation are read off one each: did the energy settle,
    and did the forces actually reach the target.

    The force panel is logarithmic. A relaxation drops the forces by two
    or three orders of magnitude, and on a linear axis the whole
    interesting part -- the approach to the target -- is squashed into
    the bottom pixel row. The target itself is drawn as a horizontal
    line, so "did it get there" is a glance rather than a calculation.
    """

    import numpy as np

    figure.clear()

    if not result.energies:
        return

    grid = figure.add_gridspec(2, 1, hspace=0.08, height_ratios=[3, 2])

    energy_axes = figure.add_subplot(grid[0])
    force_axes = figure.add_subplot(grid[1], sharex=energy_axes)

    steps = np.arange(len(result.energies))

    energies = np.asarray(result.energies, dtype=float)

    # Relative to where it started: the absolute total energy of a
    # forty-atom cell is a large number whose leading digits never
    # change, and plotting it hides the entire relaxation in the noise
    # of the axis labels.
    energy_axes.plot(
        steps,
        energies - energies[0],
        linewidth=1.2,
        color="#2b4b9b",
        marker="o",
        markersize=2.5,
    )

    energy_axes.set_ylabel("E - E$_0$  (eV)")
    energy_axes.tick_params(labelbottom=False)

    _style(energy_axes)

    forces = np.asarray(result.forces, dtype=float)

    force_axes.semilogy(
        steps[: len(forces)],
        np.maximum(forces, 1e-6),
        linewidth=1.2,
        color="#c0392b",
        marker="o",
        markersize=2.5,
    )

    target = float(getattr(result, "fmax", 0.0) or 0.0)

    if target > 0.0:
        force_axes.axhline(
            target,
            color="#3aa757",
            linestyle="--",
            linewidth=1.0,
        )

        force_axes.text(
            0.99,
            target,
            f" target {target:g}",
            transform=force_axes.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color="#3aa757",
        )

    force_axes.set_ylabel("max |F|  (eV/Å)")
    force_axes.set_xlabel("Optimiser step")

    _style(force_axes)

    _annotate(figure, getattr(result, "force_field", ""))

    if title:
        energy_axes.set_title(title, fontsize=10)

    # No tight_layout: the two panels share an axis through an explicit
    # gridspec, which tight_layout cannot lay out and warns about. The
    # gridspec's own spacing is what positions them, exactly as in
    # draw_md_traces above.

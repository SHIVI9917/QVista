"""
Figures for the transport analyses.

RDF, Van Hove, residence time, Arrhenius and conductivity. Pure
matplotlib onto a figure the caller owns, matching the other plotting
modules here, so the same code serves the embedded canvas and a
headless export.
"""

from __future__ import annotations

from qvista.visualization.smoothing import spline

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

FIT_COLOUR = "#c0392b"
GUIDE_COLOUR = "#6b7280"


def _style(axes) -> None:
    """Ticks pointing inward, matching the other QVista figures."""

    axes.tick_params(direction="in", top=True, right=True)


def _annotate(figure, text: str) -> None:
    """Stamp the figure with where the numbers came from."""

    if not text:
        return

    figure.text(
        0.995, 0.005, text, ha="right", va="bottom",
        fontsize=6, color="grey",
    )


# ------------------------------------------------------------
# Radial distribution function
# ------------------------------------------------------------


def draw_rdf(
    figure,
    result,
    title: str = "",
    with_coordination: bool = True,
) -> None:
    """
    ``g(r)`` for each pair, with the running coordination number.

    Two panels rather than two y-axes: the coordination number is a
    different quantity on a different scale, and overlaying it on a
    shared frame makes it look like part of the same curve.

    The line at ``g = 1`` is the uniform-density reference every RDF is
    measured against, and the first minimum is marked because that is
    where the coordination shell is conventionally cut.
    """

    import numpy as np

    figure.clear()

    if not result.curves:
        return

    if with_coordination:

        grid = figure.add_gridspec(
            2, 1, hspace=0.08, height_ratios=[3, 2]
        )

        axes = figure.add_subplot(grid[0])
        lower = figure.add_subplot(grid[1], sharex=axes)

    else:
        axes = figure.add_subplot(111)
        lower = None

    for index, (name, curve) in enumerate(result.curves.items()):

        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]

        grid, smooth = spline(result.r, curve)

        axes.plot(grid, smooth, linewidth=1.3, color=colour, label=name)

        minimum = result.first_minimum.get(name)

        if minimum is not None:
            axes.axvline(
                minimum, color=colour, linestyle=":", linewidth=0.8
            )

        if lower is not None:

            grid, smooth = spline(result.r, result.coordination[name])

            lower.plot(
                grid,
                smooth,
                linewidth=1.2,
                color=colour,
            )

    # The uniform-density reference.
    axes.axhline(
        1.0, color=GUIDE_COLOUR, linestyle="--", linewidth=0.8
    )

    axes.set_ylabel("g(r)")
    axes.legend(fontsize=8, loc="upper right", framealpha=0.85)

    _style(axes)

    if lower is not None:

        axes.tick_params(labelbottom=False)

        lower.set_ylabel("n(r)")
        lower.set_xlabel("r  (Å)")

        _style(lower)

    else:
        axes.set_xlabel("r  (Å)")

    axes.set_xlim(0.0, float(result.r[-1]))

    if title:
        axes.set_title(title, fontsize=10)

    _annotate(
        figure,
        f"{result.frames} frames"
        + (f", {result.temperature:.0f} K" if result.temperature else ""),
    )


# ------------------------------------------------------------
# Van Hove
# ------------------------------------------------------------


def draw_van_hove(
    figure,
    result,
    title: str = "",
    part: str = "self",
) -> None:
    """
    ``G(r, t)`` as a heat map, r across and t up.

    A heat map rather than a stack of lines because the thing being
    looked for is a *feature moving*: a second peak in the self part
    appearing at the hop distance as time goes on, which reads
    immediately as a bright arm leaving the origin and is nearly
    invisible in twenty overlaid curves.

    The self part is drawn as ``4*pi*r^2*Gs``, which is the radial
    probability density -- the quantity that integrates to one and whose
    peak is where the atom actually is, rather than ``Gs`` itself which
    always peaks at the origin.
    """

    import numpy as np

    figure.clear()

    data = (
        result.self_part if part == "self" else result.distinct
    )

    if data is None or not getattr(data, "size", 0):
        return

    axes = figure.add_subplot(111)

    values = np.asarray(data, dtype=float)

    if part == "self":

        # Radial probability, so the peak follows the atom.
        values = values * 4.0 * np.pi * result.r[None, :] ** 2

        label = r"4$\pi r^2 G_s(r,t)$"

    else:
        label = r"$G_d(r,t)$"

    image = axes.pcolormesh(
        result.r,
        result.t,
        values,
        shading="auto",
        cmap="viridis",
    )

    bar = figure.colorbar(image, ax=axes)
    bar.set_label(label, fontsize=9)

    axes.set_xlabel("r  (Å)")
    axes.set_ylabel("t  (ps)")

    _style(axes)

    if title:
        axes.set_title(title, fontsize=10)

    _annotate(
        figure,
        f"{result.element}, {result.frames} frames"
        + (f", {result.temperature:.0f} K" if result.temperature else ""),
    )

    figure.tight_layout()


def draw_van_hove_slices(
    figure, result, title: str = "", count: int = 6
) -> None:
    """
    A few time slices of the self part, as lines.

    The heat map shows the whole surface; this shows what a slice
    actually looks like, which is what a peak position is read off.
    """

    import numpy as np

    figure.clear()

    if not getattr(result.self_part, "size", 0):
        return

    axes = figure.add_subplot(111)

    chosen = np.unique(
        np.linspace(0, len(result.t) - 1, count).astype(int)
    )

    for position, index in enumerate(chosen):

        colour = SERIES_COLOURS[position % len(SERIES_COLOURS)]

        grid, smooth = spline(
            result.r,
            result.self_part[index] * 4.0 * np.pi * result.r**2,
        )

        axes.plot(
            grid,
            smooth,
            linewidth=1.2,
            color=colour,
            label=f"t = {result.t[index]:.3f} ps",
        )

    axes.set_xlabel("r  (Å)")
    axes.set_ylabel(r"4$\pi r^2 G_s(r,t)$")

    axes.legend(fontsize=7, loc="upper right", framealpha=0.85)

    _style(axes)

    if title:
        axes.set_title(title, fontsize=10)

    figure.tight_layout()


# ------------------------------------------------------------
# Residence time
# ------------------------------------------------------------


def draw_residence(figure, result, title: str = "") -> None:
    """
    The survival correlation, with the fitted exponential over it.

    Logarithmic, because an exponential is a straight line there and
    the eye is far better at judging straightness than at judging
    whether a curve decays exponentially. The 1/e line is drawn because
    that is where the residence time is read off.
    """

    import numpy as np

    figure.clear()

    if not getattr(result.correlation, "size", 0):
        return

    axes = figure.add_subplot(111)

    axes.semilogy(
        result.t,
        np.maximum(result.correlation, 1e-4),
        linewidth=1.3,
        color=SERIES_COLOURS[0],
        marker="o",
        markersize=2.5,
        label=f"{result.element} survival",
    )

    if getattr(result.fitted, "size", 0):

        axes.semilogy(
            result.t,
            np.maximum(result.fitted, 1e-4),
            linewidth=1.1,
            linestyle="--",
            color=FIT_COLOUR,
            label=(
                rf"exp(-t / $\tau$),  $\tau$ = {result.tau:.3f} ps"
            ),
        )

    axes.axhline(
        np.exp(-1.0),
        color=GUIDE_COLOUR,
        linestyle=":",
        linewidth=0.9,
    )

    axes.set_xlabel("t  (ps)")
    axes.set_ylabel("C(t)")

    axes.set_ylim(1e-2, 1.5)

    axes.legend(fontsize=8, loc="upper right", framealpha=0.85)

    _style(axes)

    if title:
        axes.set_title(title, fontsize=10)

    _annotate(
        figure,
        f"cutoff {result.cutoff:g} Å"
        + (f", {result.temperature:.0f} K" if result.temperature else ""),
    )

    figure.tight_layout()


# ------------------------------------------------------------
# Arrhenius
# ------------------------------------------------------------


def draw_arrhenius(
    figure,
    result,
    title: str = "",
    room_temperature: float = 300.0,
) -> None:
    """
    ``ln D`` against ``1000 / T``, with the fitted line.

    Plotted against ``1000/T`` rather than ``1/T`` so the axis carries
    readable numbers, and with the temperatures themselves along the
    top -- ``1/T`` is what makes it a line, but nobody thinks in
    reciprocal kelvin.

    The extrapolation down to room temperature is drawn dashed, and the
    measured range is shaded, because the difference between those two
    regions is the whole question of whether the number can be used.
    """

    import numpy as np

    figure.clear()

    if not result.points:
        return

    axes = figure.add_subplot(111)

    temperatures = np.array(result.temperatures, dtype=float)

    inverse = 1000.0 / temperatures

    values = np.array(
        [point.diffusivity for point in result.points], dtype=float
    )

    axes.semilogy(
        inverse,
        values,
        "o",
        color=SERIES_COLOURS[0],
        markersize=7,
        label="measured",
        zorder=3,
    )

    if np.isfinite(result.activation_energy):

        low = min(room_temperature, temperatures.min()) * 0.95
        high = temperatures.max() * 1.05

        span = np.linspace(low, high, 200)

        fitted = np.array([result.at(value) for value in span])

        order = np.argsort(1000.0 / span)

        axes.semilogy(
            (1000.0 / span)[order],
            fitted[order],
            linewidth=1.2,
            linestyle="--",
            color=FIT_COLOUR,
            label=(
                rf"$E_a$ = {result.activation_energy:.3f} eV,  "
                rf"$D_0$ = {result.prefactor:.2e}"
            ),
        )

        # Where the measurements actually are.
        axes.axvspan(
            1000.0 / temperatures.max(),
            1000.0 / temperatures.min(),
            color=SERIES_COLOURS[0],
            alpha=0.08,
            zorder=0,
        )

        if result.extrapolating:

            axes.axvline(
                1000.0 / room_temperature,
                color=GUIDE_COLOUR,
                linestyle=":",
                linewidth=0.9,
            )

            axes.annotate(
                f"{room_temperature:.0f} K",
                xy=(1000.0 / room_temperature, values.min()),
                fontsize=8,
                color=GUIDE_COLOUR,
                ha="right",
                va="bottom",
                rotation=90,
            )

    axes.set_xlabel("1000 / T  (1/K)")
    axes.set_ylabel("D  (cm$^2$/s)")

    axes.legend(fontsize=8, loc="upper right", framealpha=0.85)

    _style(axes)

    # The temperatures themselves along the top.
    upper = axes.secondary_xaxis(
        "top",
        functions=(
            lambda value: np.divide(
                1000.0, value, out=np.full_like(
                    np.asarray(value, dtype=float), np.nan
                ), where=np.asarray(value) != 0
            ),
            lambda value: np.divide(
                1000.0, value, out=np.full_like(
                    np.asarray(value, dtype=float), np.nan
                ), where=np.asarray(value) != 0
            ),
        ),
    )

    upper.set_xlabel("T  (K)", fontsize=9)

    if title:
        axes.set_title(title, fontsize=10, pad=28)

    figure.tight_layout()


def draw_conductivity(
    figure,
    result,
    title: str = "",
    room_temperature: float = 300.0,
) -> None:
    """
    Ionic conductivity against ``1000 / T``.

    The same figure as the Arrhenius one, in the units a battery paper
    quotes: mS/cm, and with the room-temperature value marked, since
    that single number is what a material is judged on.
    """

    import numpy as np

    figure.clear()

    if not result.measured:
        return

    axes = figure.add_subplot(111)

    temperatures = np.array(
        [item[0] for item in result.measured], dtype=float
    )

    sigmas = np.array(
        [item[1] for item in result.measured], dtype=float
    )

    axes.semilogy(
        1000.0 / temperatures,
        sigmas,
        "o",
        color=SERIES_COLOURS[2],
        markersize=7,
        label="measured",
        zorder=3,
    )

    if result.arrhenius is not None and np.isfinite(
        result.activation_energy
    ):

        span = np.linspace(
            min(room_temperature, temperatures.min()) * 0.95,
            temperatures.max() * 1.05,
            200,
        )

        fitted = np.array([result.at(value) for value in span])

        order = np.argsort(1000.0 / span)

        axes.semilogy(
            (1000.0 / span)[order],
            fitted[order],
            linewidth=1.2,
            linestyle="--",
            color=FIT_COLOUR,
            label=(
                rf"$E_a$ = {result.activation_energy:.3f} eV"
            ),
        )

        room = result.at(room_temperature)

        if np.isfinite(room):

            axes.plot(
                [1000.0 / room_temperature],
                [room],
                "*",
                color=FIT_COLOUR,
                markersize=13,
                zorder=4,
                label=f"{room:.3g} mS/cm at {room_temperature:.0f} K",
            )

        axes.axvspan(
            1000.0 / temperatures.max(),
            1000.0 / temperatures.min(),
            color=SERIES_COLOURS[2],
            alpha=0.08,
            zorder=0,
        )

    axes.set_xlabel("1000 / T  (1/K)")
    axes.set_ylabel(r"$\sigma$  (mS/cm)")

    axes.legend(fontsize=8, loc="upper right", framealpha=0.85)

    _style(axes)

    if title:
        axes.set_title(title, fontsize=10)

    _annotate(
        figure,
        f"Nernst-Einstein, {result.count} {result.element} "
        f"({result.charge:+g}), Haven ratio taken as 1",
    )

    figure.tight_layout()

"""
Presentation settings applied to a finished matplotlib figure.

Every figure in QVista is drawn by a function in this package and shown
in a :class:`~qvista.gui.widgets.plot_panel.PlotPanel`. This module sits
between the two: the drawing functions decide what the figure *means*,
and this decides what it *looks like*.

Why it works on the figure rather than through the drawing code
--------------------------------------------------------------
A style parameter threaded through every ``draw_*`` function would mean
ten signatures to keep in step, and any new figure would start out
unstyled. Walking the finished figure instead means one implementation
covers every plot in the application -- bands, DOS, phonons, MSD, NEB,
COHP, Brillouin zones -- including ones not written yet.

The cost is that this has to work out what it is looking at, which is
what the two rules below are for.

Decoration is not data
----------------------
A Fermi level marker, a zero line, a high-symmetry tick: all of them are
``Line2D`` objects exactly like the curves, and none of them should take
the line width chosen for the data. They are told apart by their
transform. Anything drawn with ``axvline``, ``axhline`` or ``axspan``
uses a blended transform -- data in one direction, axes fraction in the
other -- so it is pinned to the frame rather than to the numbers.
``transData`` therefore identifies the artists that carry the result,
and only those are restyled.

A series is something the legend would name
-------------------------------------------
Recolouring is applied to artists carrying a real label, in the order
they were drawn, because that is what a reader matches against the
legend. Where a figure has no labelled artists at all -- a plain band
structure is three hundred anonymous lines -- they are treated as one
series and take a single colour, which is the only thing recolouring
could sensibly mean there.

Originals are remembered on the artist itself the first time it is
touched, so :func:`reset` puts a figure back exactly as its drawing
function left it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

#: Where the untouched value of a property is stashed.
ORIGINAL = "_qvista_style_original"

#: Leave whatever the drawing function chose.
AS_DRAWN = "As drawn"


# ------------------------------------------------------------
# Choices
# ------------------------------------------------------------

#: Colour sets offered for the series.
#:
#: "Colourblind safe" is the Okabe-Ito palette, which stays
#: distinguishable under the three common forms of colour blindness and
#: also survives being printed in greyscale.
PALETTES: dict[str, tuple[str, ...]] = {
    AS_DRAWN: (),
    "QVista": (
        "#2980b9", "#e67e22", "#27ae60", "#8e44ad",
        "#c0392b", "#16a085", "#d35400", "#2c3e50",
    ),
    "Colourblind safe": (
        "#0072b2", "#e69f00", "#009e73", "#cc79a7",
        "#d55e00", "#56b4e9", "#f0e442", "#000000",
    ),
    "Bright": (
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ),
    "Muted": (
        "#4c72b0", "#dd8452", "#55a868", "#c44e52",
        "#8172b3", "#937860", "#da8bc3", "#8c8c8c",
    ),
    "Greyscale": (
        "#000000", "#404040", "#6b6b6b", "#8f8f8f",
        "#a8a8a8", "#c0c0c0", "#d4d4d4", "#e8e8e8",
    ),
}

#: Line styles, by the name a user would look for.
LINE_STYLES: dict[str, str] = {
    AS_DRAWN: "",
    "Solid": "-",
    "Dashed": "--",
    "Dash-dot": "-.",
    "Dotted": ":",
}

#: What a grid may cover.
GRID_CHOICES: dict[str, str] = {
    "Off": "",
    "Both axes": "both",
    "X only": "x",
    "Y only": "y",
}

#: Which way tick marks point.
TICK_DIRECTIONS: tuple[str, ...] = (AS_DRAWN, "in", "out", "inout")

#: Where a legend may sit. matplotlib's own names.
LEGEND_LOCATIONS: tuple[str, ...] = (
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center left",
    "center right",
    "upper center",
    "lower center",
    "center",
)

LEGEND_CHOICES: tuple[str, ...] = (AS_DRAWN, "Show", "Hide")

#: Font weights offered for the figure's text.
WEIGHTS: tuple[str, ...] = (AS_DRAWN, "normal", "bold")

#: The palette name that means "use the colours typed in by hand".
CUSTOM = "Custom"

#: Bounds on the spin boxes, so a typo cannot produce an unusable figure.
MIN_WIDTH = 0.1
MAX_WIDTH = 10.0

MIN_FONT = 4.0
MAX_FONT = 32.0

MIN_DPI = 50
MAX_DPI = 1200

MIN_INCHES = 2.0
MAX_INCHES = 30.0


# ------------------------------------------------------------
# The settings
# ------------------------------------------------------------


@dataclass
class PlotStyle:
    """
    Everything the design window can change about a figure.

    Every field has an "leave it alone" value -- ``None``, ``""`` or
    :data:`AS_DRAWN` -- and that is the default, so a fresh style is
    exactly no change at all. That matters: the panel is applied to
    every figure whether or not anyone has opened it.
    """

    # -- lines and markers --------------------------------------
    line_width: float | None = None
    line_style: str = AS_DRAWN
    alpha: float | None = None
    marker_size: float | None = None

    # -- colour -------------------------------------------------
    palette: str = AS_DRAWN

    custom_colours: tuple[str, ...] = ()
    """Hex codes typed by hand. Used when ``palette`` is :data:`CUSTOM`."""

    # -- axes ---------------------------------------------------
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None

    grid: str = "Off"
    grid_alpha: float = 0.3

    tick_direction: str = AS_DRAWN

    # -- text ---------------------------------------------------
    title: str = ""
    x_label: str = ""
    y_label: str = ""

    title_size: float | None = None
    label_size: float | None = None
    tick_size: float | None = None
    legend_size: float | None = None

    title_weight: str = AS_DRAWN
    label_weight: str = AS_DRAWN
    tick_weight: str = AS_DRAWN

    #: Read ``_`` and ``^`` in the label fields as sub- and superscripts.
    #: On by default: somebody typing ``Li_2CO_3`` means the subscript.
    mathtext: bool = True

    # -- legend -------------------------------------------------
    legend: str = AS_DRAWN
    legend_location: str = "best"
    legend_frame: bool = True
    legend_columns: int = 1

    # -- the figure itself --------------------------------------
    width: float | None = None
    height: float | None = None
    export_dpi: int = 600
    tight: bool = True

    # ------------------------------------------------------------

    @property
    def touches_anything(self) -> bool:
        """False when this style would leave a figure exactly as drawn."""

        return self != PlotStyle()

    def colours(self) -> tuple[str, ...]:
        """The palette in force, or empty when colours are left alone."""

        if self.palette == CUSTOM:
            return tuple(self.custom_colours)

        return PALETTES.get(self.palette, ())

    def copy(self) -> "PlotStyle":

        return replace(self)


# ------------------------------------------------------------
# Colours typed by hand
# ------------------------------------------------------------

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def is_hex(text: str) -> bool:
    """
    Whether ``text`` is a hex colour matplotlib will accept.

    Three, six or eight digits -- the last being RGBA. Anything else is
    refused here rather than at draw time, where an invalid colour
    raises out of the middle of a repaint.
    """

    return bool(_HEX.match(text.strip()))


def parse_colours(text: str) -> tuple[list[str], list[str]]:
    """
    Split a typed list of colours into the good ones and the bad ones.

    Accepts commas, spaces or newlines between them, and adds a missing
    leading ``#`` -- somebody copying a hex code out of a paper or a
    style guide very often leaves it off, and refusing that is pedantry
    rather than validation.

    Returns ``(accepted, rejected)`` so the window can show both: a
    silently dropped colour is worse than a named one, because the
    palette then quietly shifts by one and every series changes colour.
    """

    good: list[str] = []
    bad: list[str] = []

    for piece in re.split(r"[,\s]+", text.strip()):

        if not piece:
            continue

        candidate = piece if piece.startswith("#") else f"#{piece}"

        if is_hex(candidate):
            good.append(candidate.lower())
        else:
            bad.append(piece)

    return good, bad


# ------------------------------------------------------------
# Subscripts and superscripts in a label
# ------------------------------------------------------------

#: Characters that end a run of subscripted text when no braces are used.
_BREAK = " \t()[]{},;:/*+=-"


def to_mathtext(text: str) -> str:
    """
    Turn plain ``Li_2CO_3`` or ``E_F`` into something matplotlib renders.

    matplotlib will typeset subscripts and superscripts, but only inside
    ``$...$``, and asking a user to write ``$\\mathrm{Li}_2$`` to label an
    axis is not a reasonable thing to ask. This accepts the notation
    people already type -- ``_`` for a subscript, ``^`` for a
    superscript, braces for more than one character -- and wraps each run
    so matplotlib picks it up.

    Text that already contains ``$`` is returned untouched, so anyone who
    does want to write mathtext by hand still can.

    ``\\_`` escapes a literal underscore, for a label that genuinely
    contains one.
    """

    if not text or "$" in text:
        return text

    if "_" not in text and "^" not in text:
        return text

    out: list[str] = []

    index = 0
    length = len(text)

    while index < length:

        character = text[index]

        # An escaped marker is a literal one.
        if character == "\\" and index + 1 < length:

            if text[index + 1] in "_^":
                out.append(text[index + 1])
                index += 2
                continue

        if character not in "_^":
            out.append(character)
            index += 1
            continue

        index += 1

        if index >= length:
            # A trailing marker is just that character.
            out.append(character)
            break

        if text[index] == "{":

            close = text.find("}", index)

            if close == -1:
                # Unbalanced: leave it alone rather than guess.
                out.append(character)
                continue

            body = text[index + 1:close]
            index = close + 1

        else:

            # One character, as LaTeX does it. Taking the whole run up
            # to a space instead would turn ``Li_2CO_3`` into a single
            # subscript reading "2CO_3" rather than the two the writer
            # meant; braces are there for the times more is wanted.
            body = text[index]
            index += 1

        if not body:
            out.append(character)
            continue

        out.append(f"${character}{{{_escape(body)}}}$")

    return "".join(out)


def _escape(body: str) -> str:
    """Protect the few characters mathtext treats specially."""

    for character in ("\\", "$", "{", "}"):
        body = body.replace(character, f"\\{character}")

    return body


# ------------------------------------------------------------
# Telling data from decoration
# ------------------------------------------------------------


def is_data(artist, axes) -> bool:
    """
    Whether an artist carries the result rather than framing it.

    True when it is positioned in data coordinates. A Fermi line, a zero
    line and a high-symmetry separator are all drawn with one axis
    pinned to the frame, so they fail this and keep the width and style
    their drawing function chose.

    Two transforms have to be consulted, not one. A ``Line2D`` and a
    filled region carry their geometry in ``get_transform``, but a
    scatter carries fixed-size markers there and puts the positions in
    ``get_offset_transform`` -- so a fat band, which is a scatter, looks
    like decoration if only the first is checked.
    """

    try:
        if artist.get_transform() is axes.transData:
            return True

    except AttributeError:  # pragma: no cover - not all artists have one
        pass

    offsets = getattr(artist, "get_offset_transform", None)

    if offsets is None:
        return False

    try:
        return offsets() is axes.transData

    except Exception:  # pragma: no cover - artist-specific
        return False


def is_series(artist) -> bool:
    """Whether the legend would name this artist."""

    try:
        label = artist.get_label()

    except AttributeError:  # pragma: no cover
        return False

    return bool(label) and not str(label).startswith("_")


def series_of(figure) -> list[Any]:
    """
    Every labelled artist in the figure, in the order drawn.

    The order matters: a palette is handed out along it, and a reader
    matches colours against the legend, which is built in the same
    order.
    """

    found: list[Any] = []

    for axes in figure.axes:

        for artist in list(axes.lines) + list(axes.collections):

            if is_series(artist):
                found.append(artist)

    return found


# ------------------------------------------------------------
# Remembering what was there first
# ------------------------------------------------------------


def _remember(artist, name: str, value) -> None:
    """Stash the untouched value, once."""

    store = getattr(artist, ORIGINAL, None)

    if store is None:
        store = {}
        setattr(artist, ORIGINAL, store)

    store.setdefault(name, value)


def _restore(artist) -> None:
    """Put every remembered property back."""

    store = getattr(artist, ORIGINAL, None)

    if not store:
        return

    for name, value in store.items():

        setter = getattr(artist, f"set_{name}", None)

        if setter is None:
            continue

        try:
            setter(value)

        except Exception:  # pragma: no cover - artist-specific
            continue

    setattr(artist, ORIGINAL, {})


# ------------------------------------------------------------
# Applying
# ------------------------------------------------------------


def apply(figure, style: PlotStyle | None) -> None:
    """
    Impose ``style`` on a finished figure.

    Safe to call on any figure, including an empty one, and safe to call
    repeatedly: each artist remembers what it looked like before the
    first change, so re-applying a different style does not compound.
    """

    if figure is None or style is None:
        return

    if not figure.axes:
        _apply_figure(figure, style)
        return

    for axes in figure.axes:

        _apply_lines(axes, style)
        _apply_limits(axes, style)
        _apply_grid(axes, style)
        _apply_text(axes, style)
        _apply_ticks(axes, style)

    _apply_colours(figure, style)

    for axes in figure.axes:
        _apply_legend(axes, style)

    _apply_figure(figure, style)


def reset(figure) -> None:
    """Undo everything :func:`apply` did, artist by artist."""

    if figure is None:
        return

    for axes in figure.axes:

        for artist in list(axes.lines) + list(axes.collections):
            _restore(artist)

        legend = axes.get_legend()

        if legend is not None:

            for handle in _legend_handles(legend):
                _restore(handle)

        _restore(axes.title)
        _restore(axes.xaxis.label)
        _restore(axes.yaxis.label)

        store = getattr(axes, ORIGINAL, None)

        if store:

            if "xlim" in store:
                axes.set_xlim(store["xlim"])

            if "ylim" in store:
                axes.set_ylim(store["ylim"])

            setattr(axes, ORIGINAL, {})

        axes.grid(False)


# ------------------------------------------------------------


def _apply_lines(axes, style: PlotStyle) -> None:
    """Width, dash pattern and transparency, on the data only."""

    dashes = LINE_STYLES.get(style.line_style, "")

    for line in axes.lines:

        if not is_data(line, axes):
            continue

        if style.line_width is not None:
            _remember(line, "linewidth", line.get_linewidth())
            line.set_linewidth(style.line_width)

        if dashes:
            _remember(line, "linestyle", line.get_linestyle())
            line.set_linestyle(dashes)

        if style.alpha is not None:
            _remember(line, "alpha", line.get_alpha())
            line.set_alpha(style.alpha)

        if style.marker_size is not None and line.get_marker() not in (
            None, "None", "",
        ):
            _remember(line, "markersize", line.get_markersize())
            line.set_markersize(style.marker_size)

    if style.marker_size is not None:

        for collection in axes.collections:

            if not hasattr(collection, "get_sizes"):
                continue

            sizes = collection.get_sizes()

            if not len(sizes):
                continue

            _remember(collection, "sizes", sizes.copy())

            # A scatter takes an area; the control is a radius, so that
            # doubling the number doubles the apparent size rather than
            # quadrupling it.
            collection.set_sizes(
                [style.marker_size ** 2] * len(sizes)
            )


def _original_colour(artist) -> str:
    """
    What the drawing function coloured this artist, before any restyle.

    Read from the remembered value when there is one, so that grouping
    stays stable if this is called on an already-styled figure.
    """

    store = getattr(artist, ORIGINAL, None) or {}

    if "color" in store:
        return _as_text(store["color"])

    if "facecolor" in store:
        return _as_text(store["facecolor"])

    for name in ("get_color", "get_facecolor"):

        getter = getattr(artist, name, None)

        if getter is None:
            continue

        try:
            return _as_text(getter())

        except Exception:  # pragma: no cover - artist-specific
            continue

    return ""


def _as_text(value) -> str:
    """
    A colour in whatever form matplotlib returned it, as a canonical key.

    Always lower-case hex, because this key is what decides that two
    artists are the same series -- and the same colour reaches here in
    several forms. A fat band is a ``LineCollection`` and hands back an
    RGBA array; the proxy that names it in the legend is a ``Line2D``
    and hands back the string it was given. Keyed literally, ``#4C72B0``
    and ``0.2980,0.4471,0.6902,1.0000`` are different series, so the
    legend never followed the bands it named.

    Alpha is dropped. A fat band is drawn translucent and its legend
    proxy is not, and they are still one series.
    """

    import numpy as np
    from matplotlib.colors import to_hex

    # A collection carries one colour per segment; they are all the same
    # colour here, and the first is the series.
    if not isinstance(value, str):

        array = np.asarray(value, dtype=float).ravel()

        if array.size == 0:
            return ""

        value = tuple(float(item) for item in array[:4])

    try:
        return to_hex(value, keep_alpha=False).lower()

    except (ValueError, TypeError):
        # Not a colour matplotlib recognises. Fall back to something
        # stable so grouping still works, even if it groups oddly.
        return str(value)


def _apply_colours(figure, style: PlotStyle) -> None:
    """
    Hand the palette out, one colour per series.

    A *series* is not one artist. A spin-polarised DOS draws each
    species twice -- up above the axis and down mirrored below it -- and
    labels only the first, because a legend naming everything twice is
    useless. The two are nonetheless one series, and the drawing code
    says so by giving them the same colour.

    So artists are grouped by the colour they were drawn in, and a group
    is recoloured when any member of it carries a label. That keeps a
    spin pair together, and leaves alone the deliberately neutral
    artists that no legend names -- the grey dispersion drawn under a
    fat band, for instance.
    """

    colours = style.colours()

    if not colours:
        return

    groups: dict[str, list[Any]] = {}
    named: set[str] = set()
    order: list[str] = []

    for axes in figure.axes:

        for artist in list(axes.lines) + list(axes.collections):

            if not is_data(artist, axes):
                continue

            key = _original_colour(artist)

            if key not in groups:
                groups[key] = []
                order.append(key)

            groups[key].append(artist)

            if is_series(artist):
                named.add(key)

    if not groups:
        return

    # Where nothing is labelled at all -- a plain band structure is
    # hundreds of anonymous lines -- every group is fair game, since
    # there is no legend for them to contradict.
    wanted = [key for key in order if key in named] or order

    mapping: dict[str, str] = {}

    for index, key in enumerate(wanted):

        colour = colours[index % len(colours)]

        mapping[key] = colour

        for artist in groups[key]:
            _set_colour(artist, colour)

    _recolour_legends(figure, mapping)


def _recolour_legends(figure, mapping: dict[str, str]) -> None:
    """
    Bring every legend key into line with the series it names.

    A legend is not a live view of the plot. matplotlib builds its keys
    once, from the handles it was given, and recolouring the data
    afterwards leaves those keys as they were -- so a restyled figure
    ends up with a legend that names the right series in the wrong
    colours, which is worse than no legend.

    Worse still for the band figures. Their keys are proxy artists,
    built by the drawing code precisely because a fat band's width *is*
    its weight and a swatch cut from one would be a hairline. Proxies
    are not in ``axes.lines``, so nothing in the pass above ever reaches
    them however carefully it walks the axes.

    Matched by the colour each key was drawn in, which is the same key
    the data groups use -- so a legend entry follows the series it
    belongs to without needing to know which one that is.
    """

    for axes in figure.axes:

        legend = axes.get_legend()

        if legend is None:
            continue

        for handle in _legend_handles(legend):

            colour = mapping.get(_original_colour(handle))

            if colour is not None:
                _set_colour(handle, colour)


def _legend_handles(legend) -> list[Any]:
    """
    The artists drawn in a legend's keys.

    matplotlib renamed this between versions and the older spelling is
    gone in the newer ones, so both are asked for.
    """

    for name in ("legend_handles", "legendHandles"):

        handles = getattr(legend, name, None)

        if handles:
            return list(handles)

    return []


def _set_colour(artist, colour: str) -> None:
    """Recolour one artist, whatever kind it is."""

    if hasattr(artist, "get_color") and hasattr(artist, "set_color"):

        try:
            _remember(artist, "color", artist.get_color())
            artist.set_color(colour)
            return

        except Exception:  # pragma: no cover - artist-specific
            pass

    # Collections colour their faces and edges separately.
    if hasattr(artist, "set_facecolor"):

        try:
            _remember(artist, "facecolor", artist.get_facecolor())
            artist.set_facecolor(colour)

        except Exception:  # pragma: no cover
            pass


def _apply_limits(axes, style: PlotStyle) -> None:
    """Range on each axis, one end at a time."""

    store = getattr(axes, ORIGINAL, None)

    if store is None:
        store = {}
        setattr(axes, ORIGINAL, store)

    if style.x_min is not None or style.x_max is not None:

        store.setdefault("xlim", axes.get_xlim())

        low, high = axes.get_xlim()

        axes.set_xlim(
            low if style.x_min is None else style.x_min,
            high if style.x_max is None else style.x_max,
        )

    if style.y_min is not None or style.y_max is not None:

        store.setdefault("ylim", axes.get_ylim())

        low, high = axes.get_ylim()

        axes.set_ylim(
            low if style.y_min is None else style.y_min,
            high if style.y_max is None else style.y_max,
        )


def _apply_grid(axes, style: PlotStyle) -> None:

    which = GRID_CHOICES.get(style.grid, "")

    if not which:
        axes.grid(False)
        return

    axes.grid(
        True,
        axis=which,
        alpha=style.grid_alpha,
        linewidth=0.5,
        linestyle=":",
    )

    # Behind the data, always.
    axes.set_axisbelow(True)


def _apply_text(axes, style: PlotStyle) -> None:
    """Titles, axis labels, their sizes and their weight."""

    render = to_mathtext if style.mathtext else (lambda text: text)

    for wanted, artist, getter, setter in (
        (style.title, axes.title, axes.get_title, axes.set_title),
        (style.x_label, axes.xaxis.label, axes.get_xlabel, axes.set_xlabel),
        (style.y_label, axes.yaxis.label, axes.get_ylabel, axes.set_ylabel),
    ):
        if wanted:
            _remember(artist, "text", getter())
            setter(render(wanted))

    if style.title_size is not None:
        _remember(axes.title, "fontsize", axes.title.get_fontsize())
        axes.title.set_fontsize(style.title_size)

    if style.label_size is not None:

        for label in (axes.xaxis.label, axes.yaxis.label):
            _remember(label, "fontsize", label.get_fontsize())
            label.set_fontsize(style.label_size)

    if style.title_weight != AS_DRAWN:
        _remember(axes.title, "fontweight", axes.title.get_fontweight())
        axes.title.set_fontweight(style.title_weight)

    if style.label_weight != AS_DRAWN:

        for label in (axes.xaxis.label, axes.yaxis.label):
            _remember(label, "fontweight", label.get_fontweight())
            label.set_fontweight(style.label_weight)


def _apply_ticks(axes, style: PlotStyle) -> None:

    settings: dict[str, Any] = {}

    if style.tick_direction != AS_DRAWN:
        settings["direction"] = style.tick_direction

    if style.tick_size is not None:
        settings["labelsize"] = style.tick_size

    if settings:
        axes.tick_params(which="both", **settings)

    if style.tick_weight != AS_DRAWN:

        # tick_params has no weight, so the labels are set directly.
        for label in axes.get_xticklabels() + axes.get_yticklabels():
            _remember(label, "fontweight", label.get_fontweight())
            label.set_fontweight(style.tick_weight)


def _apply_legend(axes, style: PlotStyle) -> None:
    """Show, hide or move the legend."""

    existing = axes.get_legend()

    if style.legend == "Hide":

        if existing is not None:
            existing.set_visible(False)

        return

    handles, labels = axes.get_legend_handles_labels()

    wanted = style.legend == "Show" or (
        style.legend == AS_DRAWN and existing is not None
    )

    if not wanted or not handles:
        return

    legend = axes.legend(
        handles,
        labels,
        loc=style.legend_location,
        frameon=style.legend_frame,
        ncol=max(1, style.legend_columns),
        fontsize=style.legend_size,
        framealpha=0.85,
    )

    legend.set_visible(True)


def _apply_figure(figure, style: PlotStyle) -> None:
    """Size, and the layout pass that has to come last."""

    if style.width is not None and style.height is not None:

        try:
            figure.set_size_inches(style.width, style.height)

        except Exception:  # pragma: no cover - backend-specific
            pass

    if style.tight and figure.axes:

        try:
            figure.tight_layout()

        except Exception:  # pragma: no cover - some layouts refuse
            pass


# ------------------------------------------------------------
# Describing one
# ------------------------------------------------------------


def summary(style: PlotStyle) -> list[tuple[str, str]]:
    """The settings that are doing something, for a status line."""

    rows: list[tuple[str, str]] = []

    if style.line_width is not None:
        rows.append(("Line width", f"{style.line_width:g}"))

    if style.line_style != AS_DRAWN:
        rows.append(("Line style", style.line_style))

    if style.palette != AS_DRAWN:

        rows.append(
            (
                "Colours",
                f"{len(style.colours())} custom"
                if style.palette == CUSTOM
                else style.palette,
            )
        )

    for name, low, high in (
        ("X range", style.x_min, style.x_max),
        ("Y range", style.y_min, style.y_max),
    ):
        if low is not None or high is not None:
            rows.append(
                (
                    name,
                    f"{'auto' if low is None else f'{low:g}'} to "
                    f"{'auto' if high is None else f'{high:g}'}",
                )
            )

    if style.grid != "Off":
        rows.append(("Grid", style.grid))

    if style.legend != AS_DRAWN:
        rows.append(("Legend", style.legend))

    bold = [
        name
        for name, value in (
            ("title", style.title_weight),
            ("labels", style.label_weight),
            ("ticks", style.tick_weight),
        )
        if value == "bold"
    ]

    if bold:
        rows.append(("Bold", ", ".join(bold)))

    if style.width is not None and style.height is not None:
        rows.append(
            ("Figure size", f"{style.width:g} x {style.height:g} in")
        )

    rows.append(("Export dpi", str(style.export_dpi)))

    return rows

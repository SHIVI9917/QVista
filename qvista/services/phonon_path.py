"""
Choosing which high-symmetry points a phonon dispersion runs through.

Distinct from the k-path built under Crystallography, which describes a
structure and feeds KPOINTS files. This one describes a *figure*: the
force constants are already in hand, and changing the path re-evaluates
the dynamical matrix along a different line through the same Brillouin
zone. That is seconds of work against the hours the force constants
took, which is what makes it worth offering on the plot rather than as
something to decide before running anything.

The points come from seekpath applied to the primitive cell phonopy
derived, not to whatever structure happens to be open. That matters:
post-processing is meant to work on a folder someone else ran, where
there may be no open structure at all, and a path expressed in a
different cell's reciprocal basis would be silently wrong rather than
rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

#: q-points per segment. phonopy's own default for ``--band auto``.
DEFAULT_NPOINTS = 51

#: Below this a segment is a straight line between two labels and the
#: dispersion's structure is invisible.
MIN_NPOINTS = 11
MAX_NPOINTS = 401


class PhononPathError(RuntimeError):
    """Raised when a path cannot be built for this lattice."""


@dataclass(frozen=True)
class PathOptions:
    """Everything a chooser needs to offer a path for one cell."""

    #: Label to fractional coordinate, in the primitive reciprocal basis.
    points: dict[str, tuple[float, float, float]]

    #: seekpath's recommended path, as consecutive label pairs.
    recommended: list[tuple[str, str]]

    def labels(self) -> list[str]:
        """
        Every point available, Gamma first and the rest alphabetical.

        Gamma first because a path essentially always starts there and
        it is the one label a reader looks for.
        """

        rest = sorted(name for name in self.points if name != "GAMMA")

        return (["GAMMA"] if "GAMMA" in self.points else []) + rest


def options_for(primitive) -> PathOptions:
    """
    The high-symmetry points of a phonopy primitive cell.

    ``primitive`` is a phonopy ``Primitive`` -- the cell the dynamical
    matrix is actually built on, so the q-points cannot disagree with
    the frequencies.
    """

    import numpy as np
    import seekpath

    try:
        cell = (
            np.asarray(primitive.cell, dtype=float),
            np.asarray(primitive.scaled_positions, dtype=float),
            [int(n) for n in primitive.numbers],
        )

        found = seekpath.get_path(cell)

    except Exception as exc:
        raise PhononPathError(
            f"Could not work out the high-symmetry points for this "
            f"cell: {type(exc).__name__}: {exc}"
        ) from exc

    points = {
        str(name): tuple(float(v) for v in coord)
        for name, coord in found["point_coords"].items()
    }

    recommended = [
        (str(start), str(end)) for start, end in found["path"]
    ]

    if not points:
        raise PhononPathError(
            "seekpath found no high-symmetry points for this lattice."
        )

    return PathOptions(points=points, recommended=recommended)


def group_segments(segments) -> list[list[str]]:
    """
    Turn ``[(A,B),(B,C),(D,E)]`` into ``[[A,B,C],[D,E]]``.

    Each continuous stretch becomes one phonopy band path, so a jump in
    the path is drawn as a break rather than as a spurious segment
    joining two unrelated points.
    """

    stretches: list[list[str]] = []

    for start, end in segments:

        if stretches and stretches[-1][-1] == start:
            stretches[-1].append(end)

        else:
            stretches.append([start, end])

    return stretches


def build_qpoints(
    points: dict[str, tuple[float, float, float]],
    segments,
    npoints: int = DEFAULT_NPOINTS,
):
    """
    q-points, labels and connections for a chosen path.

    The three go together: phonopy needs the connections to know where
    the path breaks, and the labels to write the axis. Deriving all
    three here is what stops the figure's axis and its data disagreeing.
    """

    import numpy as np
    from phonopy.phonon.band_structure import (
        get_band_qpoints_and_path_connections,
    )

    stretches = group_segments(list(segments))

    if not stretches:
        raise PhononPathError(
            "The path is empty, so there is nothing to plot along."
        )

    missing = [
        name
        for stretch in stretches
        for name in stretch
        if name not in points
    ]

    if missing:
        raise PhononPathError(
            "This lattice has no high-symmetry point called "
            + ", ".join(sorted(set(missing)))
            + "."
        )

    band_paths = [
        [np.asarray(points[name], dtype=float) for name in stretch]
        for stretch in stretches
    ]

    qpoints, connections = get_band_qpoints_and_path_connections(
        band_paths, npoints=clamp_npoints(npoints)
    )

    # phonopy wants one flat list of labels with the breaks doubled:
    # a stretch ending at X followed by one starting at K is written
    # "... X K ...", which is what the "X|K" tick comes from.
    labels: list[str] = []

    for stretch in stretches:
        labels.extend(stretch)

    return qpoints, labels, connections


def clamp_npoints(npoints: int) -> int:
    """Hold the sampling inside what draws a readable dispersion."""

    try:
        value = int(npoints)

    except (TypeError, ValueError):
        return DEFAULT_NPOINTS

    return max(MIN_NPOINTS, min(MAX_NPOINTS, value))


def run_band(
    phonon,
    points: dict[str, tuple[float, float, float]],
    segments,
    npoints: int = DEFAULT_NPOINTS,
    with_eigenvectors: bool = True,
):
    """
    Re-evaluate the dispersion along a chosen path.

    Returns the same dictionary shape the analyses produce, which is
    what every figure here draws from. Built by hand rather than taken
    from ``get_band_structure_dict()``: that one carries the numbers but
    neither the labels nor the path connections, so a figure drawn from
    it would have a dispersion with no ticks on its axis and no idea
    where the path breaks.

    Eigenvectors are asked for by default because they cost nothing
    extra -- the diagonalisation happens either way -- and they are what
    the element-projected figure is built from. Without them, switching
    to the projected view after changing the path would need a second
    pass over the whole path.
    """

    qpoints, labels, connections = build_qpoints(
        points, segments, npoints
    )

    try:
        phonon.run_band_structure(
            qpoints,
            path_connections=connections,
            labels=labels,
            with_eigenvectors=with_eigenvectors,
        )

        structure = phonon.band_structure

    except Exception as exc:
        raise PhononPathError(
            f"The band structure failed along this path: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    band = {
        "qpoints": structure.qpoints,
        "distances": structure.distances,
        "frequencies": structure.frequencies,
        "labels": list(labels),
        "path_connections": list(connections),
        "symbols": [str(symbol) for symbol in phonon.primitive.symbols],
    }

    if with_eigenvectors:
        band["eigenvectors"] = structure.eigenvectors

    return band


def path_label(segments) -> str:
    """
    A path written the way the figure's axis writes it.

    Consecutive segments join with a dash; a jump between them becomes a
    vertical bar, which is the mark phonopy puts on the axis at a break.
    """

    if not segments:
        return "(empty)"

    parts: list[str] = []

    previous = None

    for start, end in segments:

        if previous is None:
            parts.append(start)

        elif start != previous:
            parts.append(f" | {start}")

        parts.append(f"-{end}")

        previous = end

    return "".join(parts)

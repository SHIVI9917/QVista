"""
B-spline resampling, shared by every curve QVista draws.

Why every line goes through here
--------------------------------
A calculated curve arrives on whatever grid the code that produced it
used: a DOSCAR on a few hundred energies, a COHP on a LOBSTER grid, a
phonon band on however many q-points were asked for.  Drawn as straight
segments between those points, a resonance shows a visible corner at its
top and a dispersion shows facets on its curvature -- both artefacts of
the grid, not of the physics.

A cubic B-spline through the same points, evaluated on a denser grid,
removes them.  This matters more than it sounds: it is *interpolating*,
not smoothing.  The curve passes through every calculated point exactly,
so no peak moves, no area changes and nothing is averaged away.  All
that changes is how many points the line is drawn with between them.

That is the difference from a Gaussian convolution, which is what the
COHP window used to offer.  Convolution is a physical statement -- the
calculation does not resolve structure finer than sigma -- and it moves
the data: peaks shrink, shoulders merge, and two nearby states can be
drawn as one.  Rounding the corners of a peak was never the reason to
reach for it, and a spline does that without touching the values.

What it refuses to do
---------------------
Interpolation needs a strictly monotonic abscissa, and a band structure
does not have one: pymatgen repeats a distance where the path turns a
corner at a high-symmetry point.  Splining across such a junction would
draw a curve joining two branches that are not connected.  So the input
is split at those repeats by :func:`runs` and each stretch is splined on
its own, and anything the spline cannot honestly handle -- too few
points, a flat or non-finite abscissa -- comes back unchanged rather
than approximated.
"""

from __future__ import annotations

import numpy as np

#: Drawn points per calculated point.  Eight is where the corner on a
#: sharp DOS peak stops being visible at ordinary figure sizes; past
#: about sixteen nothing changes on screen and the arrays get large.
DENSITY = 8

#: Ceiling on the resampled length.  A dense DOS already has thousands
#: of points and does not need eight times as many to look smooth; this
#: keeps a figure with fifty curves on it from becoming slow to redraw.
MAX_POINTS = 4000

#: Above this the curve is left exactly as it came.
#:
#: A figure is a couple of thousand pixels across at most, so a curve
#: already carrying this many points has more of them than the canvas
#: has columns to draw them in.  Subdividing further cannot change a
#: pixel and only costs memory and redraw time.
DENSE_ENOUGH = MAX_POINTS // 2

#: Cubic.  The lowest order whose second derivative is continuous, which
#: is what "no visible corner" actually requires.
DEGREE = 3


def refine(x, density: int = DENSITY, max_points: int = MAX_POINTS):
    """
    A denser abscissa that still contains every original point.

    Built by subdividing each interval rather than by spreading points
    evenly over the whole range, which matters for two reasons.  The
    calculated points survive exactly, so a peak keeps the height it was
    calculated with instead of being sampled somewhere near it.  And an
    uneven grid stays uneven -- a DOS written more finely around the
    Fermi level is not silently levelled out.

    Comes back unchanged when there is no room left under
    ``max_points``, which is what a grid that is already dense wants.
    """

    x = np.asarray(x, dtype=float)

    if x.size < 2 or x.size >= DENSE_ENOUGH:
        return x

    per = max(1, int(density))

    # Back off until the result fits, rather than truncating it.
    while per > 1 and (x.size - 1) * per + 1 > int(max_points):
        per -= 1

    if per <= 1:
        return x

    steps = np.linspace(0.0, 1.0, per, endpoint=False)

    inside = x[:-1, None] + steps[None, :] * np.diff(x)[:, None]

    return np.concatenate([inside.ravel(), x[-1:]])


def spline(
    x,
    y,
    density: int = DENSITY,
    degree: int = DEGREE,
    max_points: int = MAX_POINTS,
):
    """
    Resample ``(x, y)`` onto a dense grid through a B-spline.

    ``y`` may be one curve or a stack of them as columns, which is how a
    fat band gets its energies and its weights resampled together onto
    exactly the same abscissa.

    Returns ``(x, y)`` unchanged whenever the interpolation would not be
    trustworthy, so a caller never has to check first.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.shape[0] != x.shape[0]:
        return x, y

    # A cubic needs four points, and a two-point line has no curvature
    # to recover anyway.
    if x.size <= degree or x.size < 2:
        return x, y

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return x, y

    # Decreasing is as good as increasing; the spline only needs the
    # abscissa sorted, and putting it back the way it came keeps the
    # drawing order the caller chose.
    flipped = bool(x[0] > x[-1])

    ordered_x = x[::-1] if flipped else x
    ordered_y = y[::-1] if flipped else y

    if not np.all(np.diff(ordered_x) > 0.0):
        # Repeats or a reversal: for a band structure that is a corner
        # in the path, and runs() is the way to handle it.
        return x, y

    grid = refine(ordered_x, density=density, max_points=max_points)

    if grid.size <= ordered_x.size:
        return x, y

    try:
        from scipy.interpolate import make_interp_spline

        curve = make_interp_spline(ordered_x, ordered_y, k=degree)

        values = np.asarray(curve(grid), dtype=float)

    except Exception:  # noqa: BLE001 - a figure is never worth a crash
        return x, y

    if not np.all(np.isfinite(values)):
        return x, y

    if flipped:
        return grid[::-1], values[::-1]

    return grid, values


def runs(x) -> list[tuple[int, int]]:
    """
    Split an abscissa into its strictly increasing stretches.

    ``[(start, stop), ...]`` as half-open slices covering the whole
    array.  A band structure's distances repeat where the k-path turns a
    corner, and each stretch between two repeats is one straight line in
    reciprocal space -- the only span over which interpolating means
    anything.
    """

    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return []

    if x.size == 1:
        return [(0, 1)]

    # A break wherever the abscissa fails to advance.
    breaks = np.flatnonzero(np.diff(x) <= 0.0) + 1

    edges = [0, *breaks.tolist(), x.size]

    return [
        (int(start), int(stop))
        for start, stop in zip(edges[:-1], edges[1:])
        if stop > start
    ]


def spline_runs(x, y, **kwargs):
    """
    Spline each monotonic stretch of ``x`` separately.

    Yields ``(x, y)`` per stretch, so a caller draws one line per
    stretch instead of one across the whole path.  A curve with no
    repeats yields exactly one pair, which is the common case.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    for start, stop in runs(x):
        yield spline(x[start:stop], y[start:stop], **kwargs)

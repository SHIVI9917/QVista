"""
Structural and dynamical correlations from a molecular-dynamics run.

Three things read off the same trajectory, each answering a different
question about how the ions move.

**RDF** -- ``g(r)``, the radial distribution function. How likely an
atom of one species is to be found a distance ``r`` from an atom of
another, relative to a uniform gas of the same density. Its first peak
is the bond length, the area under that peak is the coordination
number, and its long-range value is one by construction. A liquid or a
melted sublattice shows a first peak and little beyond it; a crystal
shows sharp shells all the way out.

**Van Hove correlation** -- ``G(r, t)``, which is the RDF with time in
it. The *self* part ``Gs`` follows one atom away from where it started
and is the distribution whose second moment is the mean squared
displacement. The *distinct* part ``Gd`` follows every *other* atom
relative to where this one was, and is the RDF at ``t = 0``. For a
solid electrolyte the pair says whether the mobile ion hops between
well-defined sites -- ``Gs`` develops a second peak at the hop distance
while ``Gd`` keeps its crystalline shells -- or whether the sublattice
has simply melted, in which case both wash out.

**Residence time** -- how long an ion stays put before it hops.
Measured from the autocorrelation of "is this ion still within
``cutoff`` of where it was", which decays from one and whose decay
constant is the residence time.

Everything here works on a :class:`~qvista.services.msd.Trajectory`, so
the same XDATCAR reading, unwrapping and drift removal serve all of it.

A note on minimum images
------------------------
Every distance here is taken across periodic boundaries, so a pair on
opposite faces of the cell is a near neighbour and not a cell-width
apart. The trajectory positions are *unwrapped* -- they have to be, or
an ion crossing the boundary registers as a jump in the MSD -- so the
wrapping has to be redone here rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from qvista.services.msd import Trajectory, TrajectoryError

#: Bins across the r axis. Enough to resolve a first peak without
#: making every bin a single count.
DEFAULT_BINS = 200

#: How far out g(r) is computed, as a fraction of the smallest cell
#: dimension. Beyond half the cell an atom starts meeting its own
#: periodic image and the sampling sphere no longer fits inside the box,
#: so the normalisation stops being right.
MAX_RANGE_FRACTION = 0.5

#: Frames skipped at the start, as a fraction, before anything is
#: averaged. A run that began from a relaxed structure spends its first
#: picosecond equilibrating and those frames are not the ensemble.
DEFAULT_SKIP = 0.1

#: Default cutoff for "still in the same place", in angstrom. About
#: half a typical hop.
DEFAULT_RESIDENCE_CUTOFF = 1.5

#: Below this the autocorrelation is too noisy to fit.
MIN_CORRELATION_POINTS = 5


# ------------------------------------------------------------
# Shared geometry
# ------------------------------------------------------------


def minimum_image(delta: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Reduce Cartesian displacements to the shortest periodic image.

    Done by going into fractional coordinates, wrapping, and coming
    back. That is exact for any cell; the common shortcut of
    subtracting ``round(delta / L) * L`` per axis is only right for an
    orthorhombic one, and silently wrong for everything else.
    """

    lattice = np.asarray(lattice, dtype=float)

    inverse = np.linalg.inv(lattice)

    fractional = delta @ inverse

    fractional -= np.rint(fractional)

    return fractional @ lattice


def _usable_range(trajectory: Trajectory, requested: float) -> float:
    """How far out a pair correlation can honestly be taken."""

    if trajectory.lattice is None:
        raise TrajectoryError(
            "A pair correlation needs the cell, and this trajectory "
            "carries none."
        )

    lengths = np.linalg.norm(
        np.asarray(trajectory.lattice, dtype=float), axis=1
    )

    limit = MAX_RANGE_FRACTION * float(lengths.min())

    if requested <= 0.0:
        return limit

    return min(requested, limit)


def _frames_to_use(trajectory: Trajectory, skip: float) -> np.ndarray:
    """The frames left after discarding the equilibration."""

    total = trajectory.frames

    start = int(round(skip * total))

    if total - start < 1:
        raise TrajectoryError(
            f"Skipping {skip:.0%} of {total} frames leaves nothing to "
            f"average over."
        )

    return np.arange(start, total)


def _wrapped(trajectory: Trajectory, frame: int) -> np.ndarray:
    """
    Positions of one frame, folded back into the cell.

    The stored positions are unwrapped, which is what the MSD needs and
    the opposite of what a pair correlation needs.
    """

    lattice = np.asarray(trajectory.lattice, dtype=float)

    fractional = trajectory.positions[frame] @ np.linalg.inv(lattice)

    return (fractional % 1.0) @ lattice


# ------------------------------------------------------------
# Radial distribution function
# ------------------------------------------------------------


@dataclass
class RdfResult:
    """``g(r)`` for one or more species pairs."""

    r: np.ndarray = field(default_factory=lambda: np.zeros(0))

    curves: dict[str, np.ndarray] = field(default_factory=dict)
    """``"Li-O" -> g(r)``."""

    coordination: dict[str, np.ndarray] = field(default_factory=dict)
    """Running coordination number, integrated to each r."""

    first_peak: dict[str, float] = field(default_factory=dict)
    first_minimum: dict[str, float] = field(default_factory=dict)

    shell: dict[str, float] = field(default_factory=dict)
    """Coordination number out to the first minimum."""

    frames: int = 0
    temperature: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_csv(self) -> str:
        """The curves as CSV, one column per pair."""

        labels = list(self.curves)

        lines = [
            "# QVista radial distribution function",
            f"# frames averaged,{self.frames}",
            f"# temperature K,{self.temperature}",
            "",
            "r_ang," + ",".join(f"g_{name}" for name in labels),
        ]

        for index, radius in enumerate(self.r):

            row = [f"{radius:.5f}"]

            row.extend(
                f"{self.curves[name][index]:.6f}" for name in labels
            )

            lines.append(",".join(row))

        return "\n".join(lines) + "\n"


def rdf(
    trajectory: Trajectory,
    pairs: Sequence[tuple[str, str]] | None = None,
    bins: int = DEFAULT_BINS,
    r_max: float = 0.0,
    skip: float = DEFAULT_SKIP,
) -> RdfResult:
    """
    Radial distribution function, averaged over the trajectory.

    ``pairs`` names the species pairs; left empty, every pair present is
    computed. ``r_max`` defaults to half the smallest cell dimension,
    which is as far as the normalisation is valid.
    """

    limit = _usable_range(trajectory, r_max)

    frames = _frames_to_use(trajectory, skip)

    edges = np.linspace(0.0, limit, bins + 1)

    centres = 0.5 * (edges[:-1] + edges[1:])

    shell_volume = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    lattice = np.asarray(trajectory.lattice, dtype=float)

    volume = float(abs(np.linalg.det(lattice)))

    elements = trajectory.elements()

    if pairs is None:
        pairs = [
            (first, second)
            for index, first in enumerate(elements)
            for second in elements[index:]
        ]

    result = RdfResult(
        r=centres,
        frames=len(frames),
        temperature=trajectory.temperature,
    )

    for first, second in pairs:

        left = np.asarray(trajectory.indices_of(first), dtype=int)
        right = np.asarray(trajectory.indices_of(second), dtype=int)

        if left.size == 0 or right.size == 0:
            continue

        same = first == second

        counts = np.zeros(bins, dtype=float)

        for frame in frames:

            positions = _wrapped(trajectory, int(frame))

            delta = (
                positions[left][:, None, :] - positions[right][None, :, :]
            )

            distances = np.linalg.norm(
                minimum_image(delta.reshape(-1, 3), lattice), axis=1
            )

            if same:
                # Drop the self pairs, which sit at exactly zero.
                distances = distances.reshape(left.size, right.size)

                distances = distances[
                    ~np.eye(left.size, dtype=bool)
                ].ravel()

            counts += np.histogram(distances, bins=edges)[0]

        counts /= len(frames)

        # Ideal-gas count in each shell: the density of the second
        # species times the shell volume, per atom of the first.
        density = right.size / volume

        ideal = shell_volume * density * left.size

        # A same-species pair has one fewer partner available.
        if same:
            ideal *= (right.size - 1) / right.size

        with np.errstate(divide="ignore", invalid="ignore"):
            curve = np.where(ideal > 0.0, counts / ideal, 0.0)

        name = f"{first}-{second}"

        result.curves[name] = curve

        # Running coordination: the number of neighbours inside r,
        # which is the integral of the *count*, not of g.
        result.coordination[name] = np.cumsum(counts) / left.size

        _describe_peaks(result, name, centres, curve)

    if not result.curves:
        raise TrajectoryError(
            "None of the requested pairs is present in this trajectory."
        )

    result.notes = _rdf_notes(result, trajectory, limit, skip)

    return result


def _describe_peaks(
    result: RdfResult, name: str, r: np.ndarray, curve: np.ndarray
) -> None:
    """Find the first peak and the minimum after it."""

    # Ignore the region below any real bond length, where a single
    # spurious count makes an enormous g.
    usable = r > 0.5

    if not usable.any() or curve[usable].max() <= 0.0:
        return

    peak = int(np.argmax(np.where(usable, curve, 0.0)))

    result.first_peak[name] = float(r[peak])

    after = curve[peak:]

    if after.size < 3:
        return

    # The first place the curve stops falling.
    falling = np.diff(after) >= 0.0

    if not falling.any():
        return

    minimum = peak + int(np.argmax(falling))

    result.first_minimum[name] = float(r[minimum])

    result.shell[name] = float(result.coordination[name][minimum])


def _rdf_notes(
    result: RdfResult,
    trajectory: Trajectory,
    limit: float,
    skip: float,
) -> list[str]:
    """What is worth saying about an RDF before it is read."""

    notes: list[str] = []

    notes.append(
        f"Averaged over {result.frames} frames, discarding the first "
        f"{skip:.0%} as equilibration."
    )

    notes.append(
        f"Computed to {limit:.2f} A, half the smallest cell dimension. "
        f"Beyond that an atom meets its own periodic image and the "
        f"normalisation stops being right."
    )

    if trajectory.frames < 100:
        notes.append(
            f"WARNING: {trajectory.frames} frames is a thin average for "
            f"a pair correlation. The peaks will be noisy and the "
            f"coordination numbers uncertain."
        )

    return notes


# ------------------------------------------------------------
# Van Hove correlation
# ------------------------------------------------------------


@dataclass
class VanHoveResult:
    """``G(r, t)``, self and distinct parts."""

    r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Lag time, in picoseconds."""

    self_part: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0))
    )
    """``(time, r)``, normalised so 4*pi*r^2*Gs integrates to one."""

    distinct: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0))
    )
    """``(time, r)``, tending to one at large r."""

    element: str = ""
    frames: int = 0
    temperature: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def has_distinct(self) -> bool:

        return bool(self.distinct.size)


def van_hove(
    trajectory: Trajectory,
    element: str = "",
    bins: int = DEFAULT_BINS,
    r_max: float = 0.0,
    lags: int = 20,
    skip: float = DEFAULT_SKIP,
    distinct: bool = True,
) -> VanHoveResult:
    """
    Van Hove correlation for one species.

    ``Gs(r, t)`` is where an atom is at time ``t`` relative to where it
    was at time zero. ``Gd(r, t)`` is where every *other* atom is,
    relative to the same origin -- so ``Gd(r, 0)`` is the RDF and its
    filling-in over time is the sublattice rearranging.

    Averaged over every available time origin, which is what makes a
    short run usable: a trajectory of ``n`` frames gives ``n - k``
    independent origins for lag ``k``.
    """

    limit = _usable_range(trajectory, r_max)

    usable = _frames_to_use(trajectory, skip)

    if usable.size < 2:
        raise TrajectoryError(
            "A Van Hove correlation needs at least two frames."
        )

    chosen = element or trajectory.elements()[0]

    indices = np.asarray(trajectory.indices_of(chosen), dtype=int)

    if indices.size == 0:
        raise TrajectoryError(
            f"There is no {chosen} in this trajectory. It has "
            f"{', '.join(trajectory.elements())}."
        )

    lattice = np.asarray(trajectory.lattice, dtype=float)

    volume = float(abs(np.linalg.det(lattice)))

    edges = np.linspace(0.0, limit, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    shell_volume = (
        (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    )

    # Lags spread over the usable half of the run, so every one of them
    # still has a decent number of origins behind it.
    span = usable.size // 2

    steps = np.unique(
        np.linspace(0, max(span, 1), min(lags, max(span, 1)) + 1)
        .astype(int)
    )

    self_rows: list[np.ndarray] = []
    distinct_rows: list[np.ndarray] = []

    for step in steps:

        origins = usable[: usable.size - step]

        if origins.size == 0:
            continue

        self_counts = np.zeros(bins, dtype=float)
        other_counts = np.zeros(bins, dtype=float)

        for origin in origins:

            start = _wrapped(trajectory, int(origin))
            later = _wrapped(trajectory, int(origin) + int(step))

            # Self: each atom against its own earlier position.
            delta = later[indices] - start[indices]

            self_counts += np.histogram(
                np.linalg.norm(minimum_image(delta, lattice), axis=1),
                bins=edges,
            )[0]

            if not distinct:
                continue

            # Distinct: every other atom of the species against this
            # one's earlier position.
            pairs = (
                later[indices][:, None, :] - start[indices][None, :, :]
            )

            separations = np.linalg.norm(
                minimum_image(pairs.reshape(-1, 3), lattice), axis=1
            ).reshape(indices.size, indices.size)

            other_counts += np.histogram(
                separations[~np.eye(indices.size, dtype=bool)].ravel(),
                bins=edges,
            )[0]

        origins_used = float(origins.size)

        # Gs is a probability density in three dimensions: dividing by
        # the shell volume is what makes 4*pi*r^2*Gs integrate to one
        # rather than to the number of atoms.
        self_rows.append(
            self_counts
            / (origins_used * indices.size * shell_volume)
        )

        if distinct:

            density = indices.size / volume

            ideal = shell_volume * density * indices.size

            distinct_rows.append(
                other_counts / (origins_used * ideal)
                if ideal.all()
                else np.zeros(bins)
            )

    result = VanHoveResult(
        r=centres,
        t=steps * trajectory.timestep,
        self_part=np.vstack(self_rows) if self_rows else np.zeros((0, 0)),
        distinct=(
            np.vstack(distinct_rows)
            if distinct_rows
            else np.zeros((0, 0))
        ),
        element=chosen,
        frames=usable.size,
        temperature=trajectory.temperature,
    )

    result.notes = _van_hove_notes(result, trajectory, limit)

    return result


def _van_hove_notes(
    result: VanHoveResult, trajectory: Trajectory, limit: float
) -> list[str]:

    notes = [
        f"Self part follows one {result.element} away from where it "
        f"started; the distinct part follows every other one relative "
        f"to the same origin, so Gd at t = 0 is the RDF.",
        f"Computed to {limit:.2f} A and to half the run length, so "
        f"every lag still has origins behind it to average over.",
        "At t = 0 the self part is a delta function at the origin, "
        "which no histogram of spherical shells can represent: that "
        "row alone does not integrate to one. Every later row does. "
        "The t = 0 row is kept because the distinct part beside it is "
        "the RDF, which is worth seeing.",
    ]

    if trajectory.frames < 100:
        notes.append(
            f"WARNING: {trajectory.frames} frames. A Van Hove "
            f"correlation is a two-dimensional average and needs far "
            f"more than an MSD does before its features are real."
        )

    return notes


# ------------------------------------------------------------
# Residence time
# ------------------------------------------------------------


@dataclass
class ResidenceResult:
    """How long an ion stays put before it moves."""

    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    correlation: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    """``C(t)``, from one down towards zero."""

    tau: float = float("nan")
    """Residence time, in picoseconds."""

    fitted: np.ndarray = field(default_factory=lambda: np.zeros(0))

    element: str = ""
    cutoff: float = DEFAULT_RESIDENCE_CUTOFF
    temperature: float = 0.0

    decayed: bool = False
    """True when C(t) actually fell far enough to fit."""

    notes: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        """Hop rate, in inverse picoseconds."""

        return 1.0 / self.tau if self.tau > 0.0 else float("nan")


def residence_time(
    trajectory: Trajectory,
    element: str = "",
    cutoff: float = DEFAULT_RESIDENCE_CUTOFF,
    skip: float = DEFAULT_SKIP,
) -> ResidenceResult:
    """
    Residence time from the survival autocorrelation.

    ``C(t)`` is the fraction of ions still within ``cutoff`` of where
    they were ``t`` ago, averaged over every time origin. It starts at
    one and decays; the decay constant of ``exp(-t / tau)`` is the
    residence time, and its inverse is the hop rate.

    The cutoff is what defines "still there". It should be well below
    the hop distance and well above the vibrational amplitude -- the
    gap between those two is what makes the measurement possible at
    all, and if a material has no such gap the correlation decays
    smoothly and the fitted tau means nothing.
    """

    usable = _frames_to_use(trajectory, skip)

    if usable.size < MIN_CORRELATION_POINTS:
        raise TrajectoryError(
            f"A residence time needs more than {usable.size} frames."
        )

    chosen = element or trajectory.elements()[0]

    indices = np.asarray(trajectory.indices_of(chosen), dtype=int)

    if indices.size == 0:
        raise TrajectoryError(
            f"There is no {chosen} in this trajectory."
        )

    if cutoff <= 0.0:
        raise TrajectoryError("The cutoff must be positive.")

    # Unwrapped positions: an ion that leaves has left, and folding it
    # back would make a hop across the boundary look like staying.
    positions = np.asarray(trajectory.positions, dtype=float)[
        usable[:, None], indices[None, :]
    ]

    span = usable.size // 2

    correlation = np.zeros(span + 1, dtype=float)

    for lag in range(span + 1):

        displacement = positions[lag:] - positions[: positions.shape[0] - lag]

        inside = (
            np.linalg.norm(displacement, axis=2) <= cutoff
        )

        correlation[lag] = float(inside.mean())

    times = np.arange(span + 1) * trajectory.timestep

    result = ResidenceResult(
        t=times,
        correlation=correlation,
        element=chosen,
        cutoff=cutoff,
        temperature=trajectory.temperature,
    )

    _fit_decay(result)

    result.notes = _residence_notes(result, trajectory)

    return result


def _fit_decay(result: ResidenceResult) -> None:
    """
    Fit ``exp(-t / tau)`` to the part that has actually decayed.

    Fitted as a straight line through ``log C``, over the range where
    C is between 1/e and a floor: below the floor the values are a
    handful of counts and their logarithm is dominated by noise, and
    above 1/e almost nothing has happened yet.
    """

    times = result.t
    curve = result.correlation

    floor = 0.05

    usable = (curve > floor) & (curve < 1.0) & (times > 0.0)

    result.decayed = bool(curve.min() < np.exp(-1.0))

    if usable.sum() < MIN_CORRELATION_POINTS:
        return

    slope, intercept = np.polyfit(
        times[usable], np.log(curve[usable]), 1
    )

    if slope >= 0.0:
        return

    result.tau = float(-1.0 / slope)

    result.fitted = np.exp(intercept + slope * times)


def _residence_notes(
    result: ResidenceResult, trajectory: Trajectory
) -> list[str]:

    notes: list[str] = []

    if not result.decayed:
        notes.append(
            f"WARNING: the correlation never fell to 1/e, so nothing "
            f"like a residence time has been observed. Either the "
            f"{result.element} does not move on this timescale, or the "
            f"run is far too short. Any tau below is extrapolation."
        )

    if np.isfinite(result.tau) and result.tau > 0.5 * trajectory.duration:
        notes.append(
            f"WARNING: the fitted residence time ({result.tau:.2f} ps) "
            f"is a large fraction of the {trajectory.duration:.2f} ps "
            f"run. A time constant longer than the observation is not "
            f"measured, it is guessed."
        )

    notes.append(
        f"An ion counts as having stayed while it is within "
        f"{result.cutoff:g} A of where it was. That cutoff should sit "
        f"between the vibrational amplitude and the hop distance; if "
        f"the material has no gap between those, this number means "
        f"little."
    )

    return notes

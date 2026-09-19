"""
Mean squared displacement and diffusion from a trajectory.

The chain from a set of recorded positions to a diffusion coefficient is
short but every link has a way of going quietly wrong, so each one is a
separate function here with its own test.

    unwrap          Positions written by an MD code are folded back into
                    the cell.  An atom that crosses a face therefore
                    appears to jump the width of the cell, and an MSD
                    taken on folded coordinates saturates at roughly the
                    cell size instead of growing.  Undoing the folding is
                    the single most important step and the easiest to
                    forget.

    msd_curve       The mean squared displacement at lag ``t``, averaged
                    over every atom and every time origin.  Using many
                    origins is what makes a short run usable; the price
                    is that long lags have few origins left and are
                    correspondingly noisy.

    fit_diffusion   Einstein's relation, ``MSD = 2 d D t``.  Fitted over
                    a window rather than the whole curve: the first part
                    is ballistic -- atoms rattling in their cages, not
                    diffusing -- and the last part is the noise above.

    conductivity    Nernst-Einstein, for a mobile ion.

What the numbers do not say
---------------------------
The MSD at one lag and the MSD at the next are computed from largely the
same trajectory, so they are strongly correlated.  A least-squares fit
that treats them as independent reports a standard error far smaller
than the real uncertainty -- comparisons against methods that handle the
correlation properly find the naive error understated by two orders of
magnitude.  :attr:`Diffusion.stderr` is therefore reported as what it is,
a lower bound on the uncertainty, and the honest way to get an error bar
is to run the dynamics again from different starting velocities.

Nernst-Einstein assumes the hops are uncorrelated, that is a Haven ratio
of one.  Real solid electrolytes have correlated hops and a Haven ratio
below one, so the conductivity here is an upper estimate.

References
----------
pymatgen-analysis-diffusion, the reference implementation of the same
    conversion factors
    https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion
kinisi, on why the naive fit understates the uncertainty
    https://kinisi.readthedocs.io/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


class TrajectoryError(RuntimeError):
    """Raised when a trajectory cannot be read or analysed."""


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

#: Avogadro's number, per mole.
AVOGADRO = 6.02214076e23

#: Elementary charge, in coulombs.
ELEMENTARY_CHARGE = 1.602176634e-19

#: Gas constant, in J/(mol K).
GAS_CONSTANT = 8.314462618

#: One angstrom squared per picosecond, in cm^2/s.
#:
#: 1 A^2 = 1e-16 cm^2 and 1 ps = 1e-12 s, so the ratio is 1e-4.
A2_PS_TO_CM2_S = 1e-4

#: Spatial dimensions the diffusion is averaged over.
DIMENSIONS = 3

#: How far out to take the curve, as a fraction of the trajectory.
#:
#: A lag of half the run leaves only half as many time origins to
#: average over, and the curve there wanders badly. Measured against a
#: random walk of known D: taking the curve to a quarter of the run and
#: fitting it recovers D to about 2%, where going to the full length
#: gives 16%. It is also the difference between an O(T^2) pass over the
#: whole trajectory and a sixteenth of one.
DEFAULT_MAX_LAG_FRACTION = 0.25

#: Fraction of the computed curve skipped at each end when fitting.
#: The head is ballistic motion rather than diffusion; the tail has too
#: few time origins left to be worth anything.
DEFAULT_FIT_START = 0.1
DEFAULT_FIT_END = 0.5

#: Below this many time origins a lag is not reported at all.
MIN_ORIGINS = 4

#: An R^2 under this means the curve is not a straight line, so calling
#: its slope a diffusion coefficient is not justified.
GOOD_FIT = 0.95


# ------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------


@dataclass
class Trajectory:
    """Positions over time, in angstrom, already unwrapped."""

    positions: np.ndarray
    """``(frames, atoms, 3)`` in cartesian angstrom."""

    symbols: tuple[str, ...]
    lattice: np.ndarray | None = None
    timestep: float = 1.0
    """Picoseconds between the frames stored here."""

    temperature: float = 0.0
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def frames(self) -> int:

        return int(self.positions.shape[0])

    @property
    def natoms(self) -> int:

        return int(self.positions.shape[1])

    @property
    def duration(self) -> float:
        """Total time spanned, in picoseconds."""

        return (self.frames - 1) * self.timestep

    @property
    def volume(self) -> float:
        """Cell volume in cubic angstrom, or zero if there is no cell."""

        if self.lattice is None:
            return 0.0

        return float(abs(np.linalg.det(np.asarray(self.lattice, float))))

    def elements(self) -> list[str]:
        """The species, in the order they first appear."""

        return list(dict.fromkeys(self.symbols))

    def indices_of(self, element: str) -> list[int]:

        return [
            index
            for index, symbol in enumerate(self.symbols)
            if symbol == element
        ]


@dataclass(frozen=True)
class Diffusion:
    """A straight line fitted to one MSD curve."""

    label: str
    count: int
    """Atoms averaged over."""

    slope: float
    """Angstrom squared per picosecond."""

    intercept: float
    r_squared: float
    stderr: float
    """Standard error of the slope. A lower bound, not an uncertainty."""

    fit_from: float
    fit_to: float
    """The lag window fitted, in picoseconds."""

    @property
    def diffusivity(self) -> float:
        """``D`` in cm^2/s, from ``MSD = 2 d D t``."""

        return self.slope / (2.0 * DIMENSIONS) * A2_PS_TO_CM2_S

    @property
    def diffusivity_a2_ps(self) -> float:
        """The same thing in angstrom squared per picosecond."""

        return self.slope / (2.0 * DIMENSIONS)

    @property
    def trustworthy(self) -> bool:

        return self.r_squared >= GOOD_FIT and self.slope > 0.0


@dataclass
class MsdResult:
    """Everything one pass over a trajectory produced."""

    lags: np.ndarray
    """Lag times in picoseconds."""

    total: np.ndarray
    """MSD over every atom, in angstrom squared."""

    curves: dict[str, np.ndarray] = field(default_factory=dict)
    """Per element."""

    origins: np.ndarray | None = None
    """Time origins averaged at each lag."""

    fits: list[Diffusion] = field(default_factory=list)
    trajectory: Trajectory | None = None
    notes: list[str] = field(default_factory=list)

    def fit_for(self, label: str) -> Diffusion | None:

        for entry in self.fits:
            if entry.label == label:
                return entry

        return None


# ------------------------------------------------------------
# Unwrapping
# ------------------------------------------------------------


def unwrap(
    fractional: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """
    Undo the folding of coordinates back into the cell.

    Between consecutive frames an atom moves a small fraction of the
    cell, so a jump larger than half the cell is the periodic image
    rather than the atom: the step is taken modulo one, into
    ``[-0.5, 0.5)``, and the true displacements are accumulated from
    there.

    ``fractional`` is ``(frames, atoms, 3)``.  Returns cartesian
    angstrom of the same shape.

    This assumes no atom really moves more than half a cell between two
    stored frames, which is what "store often enough" means for a
    trajectory; a run written every thousand steps cannot be unwrapped
    by any method.
    """

    coordinates = np.asarray(fractional, dtype=float)

    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise TrajectoryError(
            f"Expected (frames, atoms, 3) fractional coordinates, got "
            f"shape {coordinates.shape}."
        )

    steps = np.diff(coordinates, axis=0)

    # The whole of the correction, in one line: take each step to its
    # nearest image.
    steps -= np.rint(steps)

    unwrapped = np.empty_like(coordinates)
    unwrapped[0] = coordinates[0]
    unwrapped[1:] = coordinates[0] + np.cumsum(steps, axis=0)

    return unwrapped @ np.asarray(lattice, dtype=float)


def remove_drift(positions: np.ndarray) -> np.ndarray:
    """
    Subtract the centre-of-mass motion of the whole cell.

    A thermostat, or plain accumulated rounding, can leave the cell as a
    whole creeping in some direction.  That drift enters every atom's
    displacement equally and shows up as diffusion that is not there,
    which matters most for the framework species that should not be
    moving at all.
    """

    values = np.asarray(positions, dtype=float)

    centre = values.mean(axis=1, keepdims=True)

    return values - centre + centre[0]


# ------------------------------------------------------------
# The curve
# ------------------------------------------------------------


def msd_curve(
    positions: np.ndarray,
    timestep: float = 1.0,
    max_lag: float | None = None,
    min_origins: int = MIN_ORIGINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean squared displacement against lag time.

    Averaged over every atom and every available time origin, which is
    what makes a run of a few hundred frames say anything at all: a
    single origin would give one noisy trace, while a hundred origins
    give a hundred estimates of the same quantity.

    Returns ``(lags, msd, origins)`` -- the lag times in picoseconds,
    the MSD in angstrom squared, and how many origins went into each
    point.  The last is worth keeping: a curve is only as good as the
    number of origins behind it, and that number falls to one at the
    longest lag.
    """

    values = np.asarray(positions, dtype=float)

    if values.ndim != 3 or values.shape[2] != 3:
        raise TrajectoryError(
            f"Expected (frames, atoms, 3) positions, got {values.shape}."
        )

    frames = values.shape[0]

    if frames < 2:
        raise TrajectoryError(
            "A mean squared displacement needs at least two frames."
        )

    if timestep <= 0.0:
        raise TrajectoryError("The timestep has to be positive.")

    longest = frames - 1

    if max_lag is not None:
        longest = min(longest, max(1, int(round(max_lag / timestep))))

    # A lag with too few origins behind it is noise, not signal.
    longest = min(longest, frames - min_origins)

    if longest < 1:
        raise TrajectoryError(
            f"{frames} frames is not enough to average over "
            f"{min_origins} time origins. Record more frames, or lower "
            f"the minimum."
        )

    lags = np.arange(1, longest + 1)

    msd = np.empty(longest + 1, dtype=float)
    origins = np.empty(longest + 1, dtype=float)

    msd[0] = 0.0
    origins[0] = float(frames)

    for lag in lags:

        # Every pair of frames this far apart, at once.
        displacement = values[lag:] - values[:-lag]

        msd[lag] = float(np.mean(np.sum(displacement**2, axis=2)))
        origins[lag] = float(displacement.shape[0])

    return (
        np.arange(longest + 1, dtype=float) * timestep,
        msd,
        origins,
    )


# ------------------------------------------------------------
# Fitting
# ------------------------------------------------------------


def fit_diffusion(
    lags: np.ndarray,
    msd: np.ndarray,
    label: str = "all",
    count: int = 0,
    start: float = DEFAULT_FIT_START,
    end: float = DEFAULT_FIT_END,
) -> Diffusion:
    """
    Fit ``MSD = 2 d D t`` over the middle of the curve.

    ``start`` and ``end`` are fractions of the lag range.  Neither end
    of an MSD curve is diffusive: the first part is atoms rattling
    inside their cages, which rises much faster than linearly, and the
    last part is averaged over so few time origins that it wanders.
    Fitting the whole curve mixes all three regimes and gives a number
    that is not a diffusion coefficient.
    """

    times = np.asarray(lags, dtype=float)
    values = np.asarray(msd, dtype=float)

    if times.size != values.size:
        raise TrajectoryError(
            f"{times.size} lags against {values.size} MSD values."
        )

    if not 0.0 <= start < end <= 1.0:
        raise TrajectoryError(
            f"The fit window runs {start:g} to {end:g} of the curve, "
            f"which is empty."
        )

    span = times[-1] if times.size else 0.0

    low, high = start * span, end * span

    inside = (times >= low) & (times <= high)

    if int(inside.sum()) < 3:
        raise TrajectoryError(
            "Fewer than three points inside the fit window. Widen it, "
            "or record more frames."
        )

    x = times[inside]
    y = values[inside]

    slope, intercept = np.polyfit(x, y, 1)

    predicted = slope * x + intercept

    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))

    r_squared = 1.0 - residual / total if total > 0 else 0.0

    # The textbook standard error of a slope. Reported, but see the
    # module docstring: successive MSD points are not independent, so
    # this is a floor on the uncertainty rather than the uncertainty.
    n = x.size

    if n > 2 and total > 0:
        variance = residual / (n - 2)
        stderr = float(np.sqrt(variance / np.sum((x - x.mean()) ** 2)))
    else:
        stderr = 0.0

    return Diffusion(
        label=label,
        count=count,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        stderr=stderr,
        fit_from=float(x[0]),
        fit_to=float(x[-1]),
    )


# ------------------------------------------------------------
# Conductivity
# ------------------------------------------------------------


def conductivity(
    diffusivity: float,
    count: int,
    charge: float,
    volume: float,
    temperature: float,
) -> float:
    """
    Nernst-Einstein ionic conductivity, in mS/cm.

    ``diffusivity`` in cm^2/s, ``count`` mobile ions in a cell of
    ``volume`` cubic angstrom, ``charge`` their formal charge, and
    ``temperature`` in kelvin.

    Assumes the hops are uncorrelated.  They are not, in any real solid
    electrolyte, so this is an upper estimate: the correction is the
    Haven ratio, which is below one and is not obtainable from a single
    trajectory.
    """

    if temperature <= 0.0:
        raise TrajectoryError(
            "Conductivity needs a temperature above absolute zero."
        )

    if volume <= 0.0:
        raise TrajectoryError(
            "Conductivity needs the cell volume, and this trajectory "
            "carries no cell."
        )

    # Cubic angstrom to cubic centimetre.
    volume_cm3 = volume * 1e-24

    concentration = count / (volume_cm3 * AVOGADRO)

    faraday = AVOGADRO * ELEMENTARY_CHARGE

    sigma = (
        concentration
        * charge**2
        * faraday**2
        * diffusivity
        / (GAS_CONSTANT * temperature)
    )

    # Siemens per centimetre to millisiemens per centimetre.
    return float(sigma * 1000.0)


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


def read_xdatcar(
    path: Path | str,
    timestep: float = 1.0,
    temperature: float = 0.0,
) -> Trajectory:
    """
    Read a VASP ``XDATCAR``.

    ``timestep`` is the time between the frames *in the file*, which is
    ``POTIM * NBLOCK`` femtoseconds, not ``POTIM``: VASP writes every
    NBLOCK-th step. Given here in picoseconds.
    """

    handle = Path(path)

    if not handle.is_file():
        raise TrajectoryError(f"No XDATCAR at {handle}.")

    lines = handle.read_text(errors="ignore").splitlines()

    if len(lines) < 8:
        raise TrajectoryError(f"{handle.name} is too short to be an XDATCAR.")

    try:
        scale = float(lines[1].split()[0])

        lattice = (
            np.array(
                [[float(v) for v in lines[row].split()[:3]] for row in (2, 3, 4)],
                dtype=float,
            )
            * scale
        )

    except (ValueError, IndexError) as exc:
        raise TrajectoryError(
            f"Could not read the cell from {handle.name}: {exc}"
        ) from exc

    species = lines[5].split()
    counts = [int(value) for value in lines[6].split()]

    symbols: list[str] = []

    for element, number in zip(species, counts):
        symbols.extend([element] * number)

    natoms = len(symbols)

    if natoms == 0:
        raise TrajectoryError(f"{handle.name} names no atoms.")

    frames: list[list[list[float]]] = []

    index = 7

    while index < len(lines):

        header = lines[index].strip()

        if not header:
            index += 1
            continue

        if not header.lower().startswith(("direct", "cartesian")):
            index += 1
            continue

        block = lines[index + 1 : index + 1 + natoms]

        if len(block) < natoms:
            break

        try:
            frames.append(
                [[float(v) for v in row.split()[:3]] for row in block]
            )

        except ValueError:
            break

        index += natoms + 1

    if len(frames) < 2:
        raise TrajectoryError(
            f"{handle.name} holds {len(frames)} configuration(s). A "
            f"diffusion analysis needs a trajectory, which is what VASP "
            f"writes when IBRION = 0."
        )

    positions = unwrap(np.asarray(frames, dtype=float), lattice)

    return Trajectory(
        positions=positions,
        symbols=tuple(symbols),
        lattice=lattice,
        timestep=timestep,
        temperature=temperature,
        source=str(handle),
    )


def from_frames(
    frames: Sequence[np.ndarray],
    symbols: Sequence[str],
    lattice: np.ndarray | None = None,
    timestep: float = 1.0,
    temperature: float = 0.0,
    wrapped: bool = False,
) -> Trajectory:
    """
    Build a trajectory from cartesian frames, as an MD run records them.

    ``wrapped`` says whether the positions have been folded back into
    the cell.  ASE's integrators do not fold, so a run made inside
    QVista comes through as it is; anything read back from a file
    usually has been folded and must say so.
    """

    positions = np.asarray(frames, dtype=float)

    if positions.ndim != 3:
        raise TrajectoryError(
            f"Expected a list of (atoms, 3) frames, got shape "
            f"{positions.shape}."
        )

    if positions.shape[1] != len(symbols):
        raise TrajectoryError(
            f"{positions.shape[1]} positions per frame against "
            f"{len(symbols)} symbols."
        )

    if wrapped:

        if lattice is None:
            raise TrajectoryError(
                "Folded positions cannot be unwrapped without the cell."
            )

        inverse = np.linalg.inv(np.asarray(lattice, dtype=float))

        positions = unwrap(positions @ inverse, lattice)

    return Trajectory(
        positions=positions,
        symbols=tuple(str(s) for s in symbols),
        lattice=None if lattice is None else np.asarray(lattice, float),
        timestep=timestep,
        temperature=temperature,
        source="this session's MD run",
    )


# ------------------------------------------------------------
# The whole pass
# ------------------------------------------------------------


def analyse(
    trajectory: Trajectory,
    elements: Sequence[str] | None = None,
    drift: bool = True,
    max_lag: float | None = None,
    start: float = DEFAULT_FIT_START,
    end: float = DEFAULT_FIT_END,
    progress: Callable[[str], None] | None = None,
) -> MsdResult:
    """
    MSD for the whole cell and for each element, with a fit on each.

    ``elements`` limits which species get their own curve; the default
    is all of them, because which one is mobile is often the question
    rather than the assumption.
    """

    positions = trajectory.positions

    notes: list[str] = list(trajectory.notes)

    if drift:
        positions = remove_drift(positions)

    if max_lag is None:
        max_lag = trajectory.duration * DEFAULT_MAX_LAG_FRACTION

    if progress:
        progress("Computing the mean squared displacement...")

    lags, total, origins = msd_curve(
        positions,
        timestep=trajectory.timestep,
        max_lag=max_lag,
    )

    curves: dict[str, np.ndarray] = {}

    wanted = list(elements) if elements else trajectory.elements()

    for element in wanted:

        rows = trajectory.indices_of(element)

        if not rows:
            continue

        _, values, _ = msd_curve(
            positions[:, rows, :],
            timestep=trajectory.timestep,
            max_lag=max_lag,
        )

        curves[element] = values

    if progress:
        progress("Fitting the diffusive regime...")

    fits: list[Diffusion] = []

    for label, values in (("all", total), *curves.items()):

        count = (
            trajectory.natoms
            if label == "all"
            else len(trajectory.indices_of(label))
        )

        try:
            fits.append(
                fit_diffusion(
                    lags, values, label=label, count=count,
                    start=start, end=end,
                )
            )

        except TrajectoryError as exc:
            notes.append(f"No fit for {label}: {exc}")

    # ----------------------------------------------------------
    # What the run says about itself
    # ----------------------------------------------------------

    if trajectory.duration < 10.0:
        notes.append(
            f"The trajectory spans {trajectory.duration:.2f} ps. A "
            f"diffusion coefficient from a run this short is an order "
            f"of magnitude estimate: tens of picoseconds is the usual "
            f"minimum, and a slow ion needs nanoseconds."
        )

    for entry in fits:

        if entry.label == "all":
            continue

        if not entry.trustworthy and entry.slope > 0:
            notes.append(
                f"{entry.label} fits a straight line poorly "
                f"(R^2 = {entry.r_squared:.3f}), so its slope is not a "
                f"diffusion coefficient. Either it is not diffusing, or "
                f"the run is too short to have reached the diffusive "
                f"regime."
            )

    return MsdResult(
        lags=lags,
        total=total,
        curves=curves,
        origins=origins,
        fits=fits,
        trajectory=trajectory,
        notes=notes,
    )

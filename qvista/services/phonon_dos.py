"""
Phonon density of states from a set of force constants.

The DOS is the count of vibrational modes per unit frequency, sampled
over a q-point mesh.  Its integral over all frequencies is fixed at
``3N`` for a primitive cell of ``N`` atoms -- three modes per atom -- and
that identity is the sharpest available check that the mesh was dense
enough, so it is computed and reported rather than assumed.

Three things it can be resolved by:

    Total       every mode, however it moves.

    Element     the modes grouped by species, which is what says whether
                a band is the light atom rattling or the heavy one.

    Atom        one curve per atom of the primitive cell, for when two
                sites of the same element are inequivalent.

The projection is phonopy's, taken from the squared eigenvector
amplitude on each atom, so the projections sum back to the total exactly.

Two ways of turning discrete mesh frequencies into a smooth curve:

    Tetrahedron The linear tetrahedron method (Bloechl 1994).  No free
                parameter and no artificial broadening, so it is the
                default and the right choice.

    Smearing    Each mode replaced by a Gaussian of width sigma.  Useful
                only to make a coarse mesh presentable, and it always
                costs some of the sum rule to the tails.

Thermal properties come off the same mesh: the harmonic free energy,
entropy and heat capacity as functions of temperature.  They are only
meaningful when the structure is dynamically stable, so an imaginary
mode anywhere on the mesh is reported as a warning against them rather
than left for the reader to notice.

References
----------
Phonopy setting tags and the Python API
    https://phonopy.github.io/phonopy/setting-tags.html
    https://phonopy.github.io/phonopy/phonopy-module.html
Togo & Tanaka, Scr. Mater. 108, 1 (2015)
    https://doi.org/10.1016/j.scriptamat.2015.07.021
Bloechl, Jepsen & Andersen, Phys. Rev. B 49, 16223 (1994)
    the linear tetrahedron method with corrections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


class PhononDosError(RuntimeError):
    """Raised when a density of states cannot be produced."""


# ------------------------------------------------------------
# Units
# ------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One frequency unit and what it takes to get there from THz."""

    key: str
    label: str
    factor: float
    """Multiply a frequency in THz by this."""

    decimals: int = 3


#: phonopy works in THz throughout; the other two are conversions of it.
#: The factors are phonopy's own constants, read at import so this module
#: and phonopy can never disagree about them.
def _conversions() -> tuple[float, float]:

    try:
        from phonopy.physical_units import get_physical_units

        units = get_physical_units()

        return float(units.THzToCm), float(units.THzToEv) * 1000.0

    except Exception:  # noqa: BLE001 - older phonopy, or none at all
        # CODATA values, matching phonopy's.
        return 33.3564095198152, 4.13566733


_THZ_TO_CM, _THZ_TO_MEV = _conversions()

UNITS: dict[str, Unit] = {
    "THz": Unit("THz", "THz", 1.0, 3),
    "cm^-1": Unit("cm^-1", "cm$^{-1}$", _THZ_TO_CM, 1),
    "meV": Unit("meV", "meV", _THZ_TO_MEV, 2),
}

DEFAULT_UNIT = "THz"


# ------------------------------------------------------------
# Choices
# ------------------------------------------------------------

#: How the DOS is resolved.
PROJECTIONS: dict[str, str] = {
    "total": "Total -- every mode together.",
    "element": "Element wise -- one curve per species.",
    "atom": "Atom wise -- one curve per atom of the primitive cell.",
}

#: How mesh frequencies become a curve.
METHODS: dict[str, str] = {
    "tetrahedron": (
        "Linear tetrahedron. No broadening parameter, and it keeps the "
        "sum rule. The default."
    ),
    "smearing": (
        "Gaussian of width sigma on each mode. Only for making a coarse "
        "mesh presentable; it loses weight to the tails."
    ),
}

DEFAULT_MESH = (20, 20, 20)
DEFAULT_SIGMA = 0.1

#: Anything below this on the mesh is an imaginary mode worth warning
#: about.  Not exactly zero: the acoustic branches at Gamma come out a
#: touch negative from finite differences on any real force constants.
IMAGINARY_TOLERANCE = -0.01

#: Fractional error in the ``3N`` sum rule that is worth reporting.
SUM_RULE_TOLERANCE = 0.02

#: Temperature grid for the thermal properties, in kelvin.
DEFAULT_TEMPERATURES = (0.0, 1000.0, 10.0)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------


@dataclass
class DosResult:
    """One density of states, with everything needed to redraw it."""

    frequencies: np.ndarray
    """In the unit named by :attr:`unit`."""

    total: np.ndarray
    """States per unit frequency."""

    curves: dict[str, np.ndarray] = field(default_factory=dict)
    """Projections, empty for a total-only run."""

    unit: str = DEFAULT_UNIT
    projection: str = "total"
    method: str = "tetrahedron"
    sigma: float | None = None
    mesh: tuple[int, int, int] = DEFAULT_MESH
    qpoints: int = 0

    natoms: int = 0
    symbols: tuple[str, ...] = ()

    integral: float = 0.0
    """What the total DOS actually integrates to; should be ``3 * natoms``."""

    lowest: float = 0.0
    """The lowest frequency anywhere on the mesh, in THz."""

    notes: list[str] = field(default_factory=list)

    @property
    def expected_integral(self) -> float:

        return 3.0 * self.natoms

    @property
    def has_imaginary(self) -> bool:

        return self.lowest < IMAGINARY_TOLERANCE

    @property
    def unit_label(self) -> str:

        return UNITS[self.unit].label


@dataclass
class ThermalResult:
    """Harmonic thermodynamics on the same mesh."""

    temperatures: np.ndarray
    free_energy: np.ndarray
    """kJ/mol."""

    entropy: np.ndarray
    """J/K/mol."""

    heat_capacity: np.ndarray
    """J/K/mol, at constant volume."""

    zero_point: float = 0.0
    """kJ/mol -- the free energy at 0 K."""

    natoms: int = 0
    trustworthy: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def dulong_petit(self) -> float:
        """The classical high-temperature limit of Cv, in J/K/mol."""

        return 3.0 * self.natoms * 8.314462618

    def at(self, temperature: float) -> tuple[float, float, float]:
        """Free energy, entropy and heat capacity at one temperature."""

        if self.temperatures.size == 0:
            raise PhononDosError("No temperatures were computed.")

        index = int(np.argmin(np.abs(self.temperatures - temperature)))

        return (
            float(self.free_energy[index]),
            float(self.entropy[index]),
            float(self.heat_capacity[index]),
        )


# ------------------------------------------------------------
# Conversion
# ------------------------------------------------------------


def convert(
    frequencies: np.ndarray,
    dos: np.ndarray,
    unit: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Move a DOS from THz onto another frequency axis.

    The density is per unit frequency, so the curve has to be divided by
    the same factor the axis is multiplied by.  Skipping that is the
    usual way a converted DOS stops integrating to ``3N``: the axis
    stretches by 33 and the area goes with it.
    """

    if unit not in UNITS:
        raise PhononDosError(
            f"{unit!r} is not a frequency unit. "
            f"Try one of {', '.join(UNITS)}."
        )

    factor = UNITS[unit].factor

    return (
        np.asarray(frequencies, dtype=float) * factor,
        np.asarray(dos, dtype=float) / factor,
    )


# ------------------------------------------------------------
# Grouping
# ------------------------------------------------------------


def group_projections(
    projected: np.ndarray,
    symbols: Sequence[str],
    projection: str,
) -> dict[str, np.ndarray]:
    """
    Turn phonopy's per-atom projections into the curves being asked for.

    ``projected`` is ``(natoms, nfrequencies)``.  Element wise sums the
    atoms of each species; atom wise keeps them apart and labels them
    the way the rest of QVista numbers atoms, from one.
    """

    values = np.asarray(projected, dtype=float)

    if values.ndim != 2:
        raise PhononDosError(
            f"Expected a 2D projection, got shape {values.shape}."
        )

    if values.shape[0] != len(symbols):
        raise PhononDosError(
            f"phonopy returned {values.shape[0]} projections for "
            f"{len(symbols)} atoms."
        )

    if projection == "atom":
        return {
            f"{index + 1} {symbol}": values[index]
            for index, symbol in enumerate(symbols)
        }

    if projection != "element":
        raise PhononDosError(
            f"{projection!r} is not a projection. "
            f"Try one of {', '.join(PROJECTIONS)}."
        )

    curves: dict[str, np.ndarray] = {}

    # dict.fromkeys keeps the order the species first appear in, which
    # is the order the structure lists them.
    for element in dict.fromkeys(symbols):

        rows = [
            index
            for index, symbol in enumerate(symbols)
            if symbol == element
        ]

        curves[element] = values[rows].sum(axis=0)

    return curves


# ------------------------------------------------------------
# Running
# ------------------------------------------------------------


def _mesh_frequencies(phonon) -> float:
    """The lowest frequency anywhere on the current mesh, in THz."""

    mesh = getattr(phonon, "mesh", None)

    if mesh is None:
        return 0.0

    frequencies = getattr(mesh, "frequencies", None)

    if frequencies is None:
        return 0.0

    values = np.asarray(frequencies, dtype=float)

    return float(values.min()) if values.size else 0.0


def run_dos(
    phonon,
    mesh: Sequence[int] = DEFAULT_MESH,
    projection: str = "total",
    method: str = "tetrahedron",
    sigma: float = DEFAULT_SIGMA,
    freq_min: float | None = None,
    freq_max: float | None = None,
    freq_pitch: float | None = None,
    unit: str = DEFAULT_UNIT,
    progress: Callable[[str], None] | None = None,
) -> DosResult:
    """
    Sample the Brillouin zone and accumulate the density of states.

    ``phonon`` is a Phonopy object that already has force constants.
    Everything here is post-processing, so it is cheap next to the
    forces that produced them -- with one exception, noted below.
    """

    if projection not in PROJECTIONS:
        raise PhononDosError(
            f"{projection!r} is not a projection. "
            f"Try one of {', '.join(PROJECTIONS)}."
        )

    if method not in METHODS:
        raise PhononDosError(
            f"{method!r} is not a broadening method. "
            f"Try one of {', '.join(METHODS)}."
        )

    if unit not in UNITS:
        raise PhononDosError(f"{unit!r} is not a frequency unit.")

    grid = tuple(int(value) for value in mesh)

    if len(grid) != 3 or any(value < 1 for value in grid):
        raise PhononDosError(
            f"The mesh must be three positive numbers, not {mesh!r}."
        )

    if method == "smearing" and sigma <= 0.0:
        raise PhononDosError(
            "A smearing width has to be positive; sigma is the standard "
            "deviation of the Gaussian put on each mode."
        )

    notes: list[str] = []

    wanted = projection != "total"

    if progress:
        progress(f"Sampling a {grid[0]}x{grid[1]}x{grid[2]} mesh...")

    try:
        # A projection needs the eigenvectors, and phonopy refuses to
        # give them from a symmetry-reduced mesh -- it raises rather
        # than quietly projecting onto the wrong atoms. The whole zone
        # is therefore sampled, which for a 20^3 mesh is 8000 q-points
        # against 256.
        phonon.run_mesh(
            list(grid),
            is_mesh_symmetry=not wanted,
            is_gamma_center=True,
            with_eigenvectors=wanted,
        )

    except Exception as exc:
        raise PhononDosError(
            f"The mesh sampling failed: {type(exc).__name__}: {exc}"
        ) from exc

    qpoints = len(np.asarray(phonon.mesh.qpoints))

    if wanted:
        notes.append(
            f"A projection needs the eigenvectors, so the whole zone was "
            f"sampled: {qpoints} q-points rather than the symmetry-reduced "
            f"set a total DOS would have used."
        )

    settings: dict[str, Any] = {
        "freq_min": freq_min,
        "freq_max": freq_max,
        "freq_pitch": freq_pitch,
        "use_tetrahedron_method": method == "tetrahedron",
    }

    if method == "smearing":
        settings["sigma"] = sigma

    if progress:
        progress("Accumulating the density of states...")

    try:
        total_dos = phonon.run_total_dos(**settings)

    except Exception as exc:
        raise PhononDosError(
            f"The density of states failed: {type(exc).__name__}: {exc}"
        ) from exc

    frequencies = np.asarray(total_dos.frequency_points, dtype=float)
    total = np.asarray(total_dos.dos, dtype=float)

    curves: dict[str, np.ndarray] = {}

    symbols = tuple(str(s) for s in phonon.primitive.symbols)

    if wanted:

        if progress:
            progress("Projecting onto the atoms...")

        try:
            projected = phonon.run_projected_dos(**settings)

        except Exception as exc:
            raise PhononDosError(
                f"The projection failed: {type(exc).__name__}: {exc}"
            ) from exc

        curves = group_projections(
            np.asarray(projected.projected_dos, dtype=float),
            symbols,
            projection,
        )

    natoms = len(symbols)

    integral = float(np.trapezoid(total, frequencies))

    lowest = _mesh_frequencies(phonon)

    # ----------------------------------------------------------
    # What the numbers say about themselves
    # ----------------------------------------------------------

    expected = 3.0 * natoms

    if expected > 0 and abs(integral - expected) > expected * SUM_RULE_TOLERANCE:
        notes.append(
            f"The DOS integrates to {integral:.3f} where three modes per "
            f"atom would give {expected:.0f}. The mesh is too coarse, or "
            f"the frequency range is clipping the curve."
        )

    if method == "smearing":
        notes.append(
            f"Gaussian smearing of {sigma:g} THz. The tetrahedron method "
            f"needs no width and keeps the sum rule; use smearing only to "
            f"tidy a coarse mesh."
        )

    if lowest < IMAGINARY_TOLERANCE:
        notes.append(
            f"Imaginary modes down to {lowest:.3f}i THz, drawn at negative "
            f"frequency. The structure is not at a minimum of the energy, "
            f"so thermal properties from this DOS are meaningless."
        )

    # The integral is worked out in THz, above, and the conversion
    # preserves it by construction -- so it stays 3N whichever axis the
    # curve ends up on.
    if unit != DEFAULT_UNIT:

        curves = {
            label: convert(frequencies, values, unit)[1]
            for label, values in curves.items()
        }

        frequencies, total = convert(frequencies, total, unit)

    return DosResult(
        frequencies=frequencies,
        total=total,
        curves=curves,
        unit=unit,
        projection=projection,
        method=method,
        sigma=sigma if method == "smearing" else None,
        mesh=grid,
        qpoints=qpoints,
        natoms=natoms,
        symbols=symbols,
        integral=integral,
        lowest=lowest,
        notes=notes,
    )


def run_thermal(
    phonon,
    t_min: float = DEFAULT_TEMPERATURES[0],
    t_max: float = DEFAULT_TEMPERATURES[1],
    t_step: float = DEFAULT_TEMPERATURES[2],
    progress: Callable[[str], None] | None = None,
) -> ThermalResult:
    """
    Harmonic free energy, entropy and heat capacity against temperature.

    Uses whichever mesh was last run, so this follows :func:`run_dos`
    rather than sampling again.
    """

    if t_max <= t_min:
        raise PhononDosError(
            f"The temperature range runs {t_min:g} to {t_max:g} K, which "
            f"is empty."
        )

    if t_step <= 0.0:
        raise PhononDosError("The temperature step has to be positive.")

    if progress:
        progress("Integrating the thermal properties...")

    try:
        thermal = phonon.run_thermal_properties(
            t_min=t_min, t_max=t_max, t_step=t_step
        )

    except Exception as exc:
        raise PhononDosError(
            f"The thermal properties failed: {type(exc).__name__}: {exc}"
        ) from exc

    temperatures = np.asarray(thermal.temperatures, dtype=float)

    lowest = _mesh_frequencies(phonon)

    notes: list[str] = []

    trustworthy = lowest >= IMAGINARY_TOLERANCE

    if not trustworthy:
        notes.append(
            f"The mesh carries imaginary modes down to {lowest:.3f}i THz. "
            f"Harmonic thermodynamics assumes every mode is a real "
            f"oscillator, so these numbers do not mean anything until the "
            f"structure is relaxed to a true minimum."
        )

    free_energy = np.asarray(thermal.free_energy, dtype=float)

    return ThermalResult(
        temperatures=temperatures,
        free_energy=free_energy,
        entropy=np.asarray(thermal.entropy, dtype=float),
        heat_capacity=np.asarray(thermal.heat_capacity, dtype=float),
        zero_point=float(free_energy[0]) if free_energy.size else 0.0,
        natoms=len(phonon.primitive.symbols),
        trustworthy=trustworthy,
        notes=notes,
    )


# ------------------------------------------------------------
# Reading a result
# ------------------------------------------------------------


def peak_frequencies(
    result: DosResult,
    count: int = 5,
    minimum_separation: float | None = None,
) -> list[tuple[float, float]]:
    """
    The tallest local maxima of the total DOS, strongest first.

    Returned as ``(frequency, height)``.  A phonon DOS is read by where
    its peaks are, and picking them off a plot by eye is exactly the
    step a table should be doing instead.
    """

    frequencies = result.frequencies
    values = result.total

    if frequencies.size < 3:
        return []

    if minimum_separation is None:
        span = float(frequencies.max() - frequencies.min())
        minimum_separation = span * 0.02

    interior = np.arange(1, values.size - 1)

    peaks = interior[
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ]

    ordered = sorted(peaks, key=lambda index: -values[index])

    chosen: list[tuple[float, float]] = []

    for index in ordered:

        frequency = float(frequencies[index])

        if any(
            abs(frequency - taken) < minimum_separation
            for taken, _ in chosen
        ):
            continue

        chosen.append((frequency, float(values[index])))

        if len(chosen) >= count:
            break

    return chosen


def band_weights(result: DosResult) -> list[tuple[str, float]]:
    """
    How many of the ``3N`` modes each projected curve accounts for.

    An element with a third of the atoms holds a third of the modes, so
    a departure from that is not a result -- it is the mesh being too
    coarse or the range clipping.  Quoted so it can be seen at a glance.
    """

    if not result.curves:
        return []

    return [
        (label, float(np.trapezoid(values, result.frequencies)))
        for label, values in result.curves.items()
    ]

"""
Diffusion kinetics: how the diffusivity depends on temperature.

One trajectory gives one diffusivity at one temperature, and that number
on its own says very little. The useful quantities come from several
temperatures at once, through the Arrhenius relation

    D(T) = D0 * exp(-Ea / (kB * T))

which is a straight line when ``ln D`` is plotted against ``1 / T``. Its
slope is ``-Ea / kB`` and its intercept is ``ln D0``.

**Ea**, the activation energy, is the barrier the ion crosses to hop.
It is the number that transfers between materials and the one a
screening study is usually after.

**D0**, the prefactor, collects the attempt frequency, the hop distance
and the number of available destinations. It is far less transferable
and is mostly used to extrapolate.

Why extrapolating down is the whole point, and the whole danger
---------------------------------------------------------------
Ab-initio MD is run hot -- six hundred to twelve hundred kelvin --
because at room temperature a solid electrolyte hops perhaps once per
nanosecond and nothing happens inside a picosecond trajectory. The
result is then extrapolated down to 300 K, often across a factor of
three in ``1/T``.

That is only valid if the mechanism does not change on the way down,
and it frequently does: a sublattice that has melted at 1000 K is not
the material anyone will use. Nothing here can detect that, which is
why the extrapolation is reported with the range it came from attached,
and why a fit through fewer than three temperatures is refused rather
than quietly drawn through two points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from qvista.services.msd import TrajectoryError

#: Boltzmann constant in eV/K, so activation energies come out in the
#: unit the field quotes them in.
BOLTZMANN_EV = 8.617333262e-5

#: Gas constant, J/(mol K), for the kJ/mol form.
GAS_CONSTANT = 8.314462618

#: eV per particle to kJ per mole.
EV_TO_KJ_PER_MOL = 96.48533212

#: Below this many temperatures a line is not a fit.
MIN_POINTS = 3

#: Where an extrapolation is usually wanted.
ROOM_TEMPERATURE = 300.0

#: An R-squared below this means the points are not on a line, so the
#: single-mechanism assumption behind Arrhenius has failed.
GOOD_FIT = 0.95


@dataclass(frozen=True)
class Point:
    """One temperature, and the diffusivity measured at it."""

    temperature: float
    """Kelvin."""

    diffusivity: float
    """cm^2/s."""

    label: str = ""

    error: float = 0.0
    """Standard error on D, when the source provided one."""


@dataclass
class ArrheniusResult:
    """A line through ``ln D`` against ``1 / T``."""

    points: list[Point] = field(default_factory=list)

    activation_energy: float = float("nan")
    """eV."""

    activation_error: float = float("nan")

    prefactor: float = float("nan")
    """D0, in cm^2/s."""

    r_squared: float = float("nan")

    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------

    @property
    def activation_kj(self) -> float:
        """The barrier in kJ/mol, which is the other common unit."""

        return self.activation_energy * EV_TO_KJ_PER_MOL

    @property
    def temperatures(self) -> list[float]:

        return [point.temperature for point in self.points]

    @property
    def trustworthy(self) -> bool:
        """Whether the points actually lie on a line."""

        return (
            len(self.points) >= MIN_POINTS
            and np.isfinite(self.r_squared)
            and self.r_squared >= GOOD_FIT
        )

    def at(self, temperature: float) -> float:
        """
        The fitted diffusivity at one temperature, in cm^2/s.

        Extrapolating below the measured range is the usual reason for
        doing this at all, and is also where it goes wrong; the range
        is reported alongside so the distance is visible.
        """

        if not np.isfinite(self.activation_energy):
            return float("nan")

        if temperature <= 0.0:
            raise TrajectoryError(
                "Diffusivity is not defined at or below absolute zero."
            )

        return float(
            self.prefactor
            * np.exp(
                -self.activation_energy
                / (BOLTZMANN_EV * temperature)
            )
        )

    @property
    def extrapolating(self) -> bool:
        """True when room temperature is outside the measured range."""

        if not self.points:
            return False

        return ROOM_TEMPERATURE < min(self.temperatures)

    def as_rows(self) -> list[tuple[str, str]]:
        """Console and table rows."""

        rows = [
            ("Temperatures", str(len(self.points))),
            (
                "Range",
                f"{min(self.temperatures):.0f} - "
                f"{max(self.temperatures):.0f} K"
                if self.points
                else "n/a",
            ),
            (
                "Activation energy Ea",
                f"{self.activation_energy:.4f} eV "
                f"({self.activation_kj:.1f} kJ/mol)",
            ),
        ]

        if np.isfinite(self.activation_error):
            rows.append(
                ("Ea uncertainty", f"+/- {self.activation_error:.4f} eV")
            )

        rows.extend(
            [
                ("Prefactor D0", f"{self.prefactor:.4e} cm^2/s"),
                ("R squared", f"{self.r_squared:.4f}"),
                (
                    f"D at {ROOM_TEMPERATURE:.0f} K",
                    f"{self.at(ROOM_TEMPERATURE):.4e} cm^2/s",
                ),
            ]
        )

        return rows

    def to_csv(self) -> str:

        lines = [
            "# QVista Arrhenius fit",
            f"# Ea eV,{self.activation_energy:.6f}",
            f"# D0 cm2 per s,{self.prefactor:.6e}",
            f"# R squared,{self.r_squared:.6f}",
            "",
            "temperature_k,inverse_t,diffusivity_cm2_s,ln_d,label",
        ]

        for point in self.points:

            lines.append(
                f"{point.temperature:.3f},"
                f"{1.0 / point.temperature:.8f},"
                f"{point.diffusivity:.6e},"
                f"{np.log(point.diffusivity):.6f},"
                f"{point.label}"
            )

        return "\n".join(lines) + "\n"


def fit(points: Sequence[Point]) -> ArrheniusResult:
    """
    Fit ``ln D = ln D0 - Ea / (kB T)`` by least squares.

    The fit is done on the logarithm, which is what makes it a straight
    line, and which also means the *relative* error on each D is what
    weights it -- appropriate here, since a diffusivity is uncertain by
    a factor rather than by an amount.
    """

    usable = [
        point
        for point in points
        if point.temperature > 0.0 and point.diffusivity > 0.0
    ]

    dropped = len(points) - len(usable)

    if len(usable) < MIN_POINTS:
        raise TrajectoryError(
            f"An Arrhenius fit needs at least {MIN_POINTS} "
            f"temperatures; there "
            f"{'is' if len(usable) == 1 else 'are'} {len(usable)}.\n\n"
            f"Two points define a line through any two numbers and say "
            f"nothing about whether the relation holds."
        )

    usable = sorted(usable, key=lambda point: point.temperature)

    inverse = np.array(
        [1.0 / point.temperature for point in usable], dtype=float
    )

    logs = np.array(
        [np.log(point.diffusivity) for point in usable], dtype=float
    )

    slope, intercept = np.polyfit(inverse, logs, 1)

    predicted = slope * inverse + intercept

    residual = float(np.sum((logs - predicted) ** 2))
    total = float(np.sum((logs - logs.mean()) ** 2))

    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")

    # Standard error on the slope, and from it on Ea.
    spread = float(np.sum((inverse - inverse.mean()) ** 2))

    if len(usable) > 2 and spread > 0.0:

        variance = residual / (len(usable) - 2)

        slope_error = float(np.sqrt(variance / spread))

    else:
        slope_error = float("nan")

    result = ArrheniusResult(
        points=usable,
        activation_energy=float(-slope * BOLTZMANN_EV),
        activation_error=(
            float(slope_error * BOLTZMANN_EV)
            if np.isfinite(slope_error)
            else float("nan")
        ),
        prefactor=float(np.exp(intercept)),
        r_squared=r_squared,
    )

    result.notes = _notes(result, dropped)

    return result


def _notes(result: ArrheniusResult, dropped: int) -> list[str]:
    """Everything that decides whether the numbers can be used."""

    notes: list[str] = []

    if dropped:
        notes.append(
            f"{dropped} point(s) had a non-positive temperature or "
            f"diffusivity and were dropped; a logarithm needs both to "
            f"be above zero."
        )

    if result.activation_energy < 0.0:
        notes.append(
            "WARNING: the fitted activation energy is negative, which "
            "means D fell as the temperature rose. That is not "
            "diffusion behaving; check whether the hot runs melted, or "
            "whether the fitting windows on the individual MSDs were "
            "wrong."
        )

    if not result.trustworthy and len(result.points) >= MIN_POINTS:
        notes.append(
            f"WARNING: R squared is {result.r_squared:.3f}. The points "
            f"do not lie on a line, so a single activation energy does "
            f"not describe them -- usually because the mechanism "
            f"changes across the range, or because one of the "
            f"individual diffusivities is not converged."
        )

    if result.extrapolating:

        low = min(result.temperatures)

        notes.append(
            f"D at {ROOM_TEMPERATURE:.0f} K is an extrapolation: the "
            f"lowest temperature measured is {low:.0f} K, and the "
            f"extrapolation spans a factor of "
            f"{low / ROOM_TEMPERATURE:.1f} in 1/T. It holds only if "
            f"the mechanism is the same at both ends, which for a "
            f"sublattice that has melted at the top of the range it "
            f"is not."
        )

    if len(result.points) < 4:
        notes.append(
            f"{len(result.points)} temperatures is the minimum that "
            f"can show curvature at all. Four or five make the "
            f"straightness of the line something you have checked "
            f"rather than assumed."
        )

    notes.append(
        "Ea is the barrier to a hop; D0 collects the attempt "
        "frequency, hop distance and number of destinations. Ea is the "
        "transferable one."
    )

    return notes

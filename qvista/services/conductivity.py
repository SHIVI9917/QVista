"""
Ionic conductivity.

Ions carrying charge as they diffuse. The Nernst-Einstein relation
turns a diffusivity into a conductivity,

    sigma = n z^2 F^2 D / (R T)

with ``n`` the concentration of mobile ions. Combined with an Arrhenius
fit over several temperatures this gives ``sigma(T)``, and the same
``D0`` and ``Ea`` carry through -- the conductivity has the same
activation energy as the diffusivity, plus the explicit ``1/T`` in
front.

Nernst-Einstein assumes the hops are uncorrelated. In any real solid
electrolyte they are not, and the correction is the Haven ratio, which
is below one and cannot be obtained from a single trajectory. Every
ionic conductivity here is therefore an upper estimate, and says so.

References
----------
Nernst, Z. Phys. Chem. 4, 129 (1889); Einstein, Ann. Phys. 322, 549
(1905)
    the relation between diffusivity and mobility
Murch, Solid State Ionics 7, 177 (1982)
    the Haven ratio, and why an uncorrelated estimate is an upper bound
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from qvista.services.arrhenius import (
    ArrheniusResult,
    Point,
    ROOM_TEMPERATURE,
)


class ConductivityError(RuntimeError):
    """Raised when a conductivity cannot be computed, with a reason."""


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

AVOGADRO = 6.02214076e23
ELEMENTARY_CHARGE = 1.602176634e-19
GAS_CONSTANT = 8.314462618


def nernst_einstein(
    diffusivity: float,
    count: int,
    charge: float,
    volume: float,
    temperature: float,
) -> float:
    """
    Ionic conductivity from a diffusivity, in mS/cm.

    ``diffusivity`` in cm^2/s, ``count`` mobile ions in a cell of
    ``volume`` cubic angstrom, ``charge`` their formal charge, and
    ``temperature`` in kelvin.
    """

    if temperature <= 0.0:
        raise ConductivityError(
            "Conductivity needs a temperature above absolute zero."
        )

    if volume <= 0.0:
        raise ConductivityError(
            "Conductivity needs the cell volume."
        )

    if count <= 0:
        raise ConductivityError(
            "There are no mobile ions to carry a current."
        )

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

    # S/cm to mS/cm.
    return sigma * 1000.0


@dataclass
class IonicResult:
    """Ionic conductivity across a temperature range."""

    arrhenius: ArrheniusResult | None = None

    element: str = ""
    charge: float = 1.0
    count: int = 0
    volume: float = 0.0

    #: ``(temperature, sigma in mS/cm)`` at each measured point.
    measured: list[tuple[float, float]] = field(default_factory=list)

    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------

    def at(self, temperature: float) -> float:
        """Fitted conductivity at one temperature, in mS/cm."""

        if self.arrhenius is None:
            return float("nan")

        diffusivity = self.arrhenius.at(temperature)

        if not np.isfinite(diffusivity):
            return float("nan")

        return nernst_einstein(
            diffusivity,
            self.count,
            self.charge,
            self.volume,
            temperature,
        )

    @property
    def activation_energy(self) -> float:
        """
        The barrier, in eV.

        The same one the diffusivity has: Nernst-Einstein multiplies D
        by a temperature-dependent prefactor, which shifts the
        intercept and not the slope.
        """

        return (
            self.arrhenius.activation_energy
            if self.arrhenius is not None
            else float("nan")
        )

    @property
    def prefactor(self) -> float:
        """D0 in cm^2/s, carried through from the Arrhenius fit."""

        return (
            self.arrhenius.prefactor
            if self.arrhenius is not None
            else float("nan")
        )

    def as_rows(self) -> list[tuple[str, str]]:

        rows: list[tuple[str, str]] = [
            ("Mobile ion", f"{self.element} ({self.charge:+g})"),
            ("Ions in the cell", str(self.count)),
            ("Cell volume", f"{self.volume:.2f} A^3"),
        ]

        if self.arrhenius is not None:
            rows.extend(self.arrhenius.as_rows())

        rows.append(
            (
                f"Conductivity at {ROOM_TEMPERATURE:.0f} K",
                f"{self.at(ROOM_TEMPERATURE):.4g} mS/cm",
            )
        )

        return rows


def ionic(
    points: Sequence[Point],
    element: str,
    charge: float,
    count: int,
    volume: float,
) -> IonicResult:
    """
    Ionic conductivity from diffusivities at several temperatures.

    Each point is converted through Nernst-Einstein at its own
    temperature, and the Arrhenius fit on the diffusivities carries
    ``D0`` and ``Ea`` through.
    """

    from qvista.services import arrhenius as ar

    fit = ar.fit(points)

    result = IonicResult(
        arrhenius=fit,
        element=element,
        charge=charge,
        count=count,
        volume=volume,
        measured=[
            (
                point.temperature,
                nernst_einstein(
                    point.diffusivity,
                    count,
                    charge,
                    volume,
                    point.temperature,
                ),
            )
            for point in points
            if point.temperature > 0.0 and point.diffusivity > 0.0
        ],
    )

    result.notes = list(fit.notes)

    result.notes.append(
        "Nernst-Einstein assumes the hops are uncorrelated. In a real "
        "solid electrolyte they are not, and the correction -- the "
        "Haven ratio -- is below one. Every conductivity here is "
        "therefore an upper estimate."
    )

    return result

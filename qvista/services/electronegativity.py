"""
Electronegativity of atoms, pairs of atoms and whole cells.

Three quantities, all from tabulated element data -- nothing here needs a
calculation to have been run:

* **Δχ between two atoms.** The difference in electronegativity across a
  bond is the usual first estimate of how ionic it is. Pauling's own
  relation turns it into a percentage, and the familiar 0.4 / 1.7
  thresholds split the range into nonpolar covalent, polar covalent and
  ionic.

* **χ of the cell.** Sanderson's electronegativity equalization principle:
  when atoms combine, their electronegativities equalize to the geometric
  mean over the atoms present. Comparing one atom against that mean says
  which way charge flows at that site.

Scales are the awkward part. Pauling's scale is the one the thresholds are
calibrated for, but it is undefined for helium, neon and argon -- they form
too few compounds to fit -- and for the superheavy elements. Where an
element in the comparison has no Pauling value, everything in that
comparison switches to the **Mulliken** scale, χ = (IE + EA)/2 in eV, which
is defined for the noble gases. The two scales are never mixed inside one
number, and a Mulliken result suppresses the percentage and the labels,
which do not carry over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pymatgen.core import Element

#: Δχ below which a bond is conventionally called nonpolar covalent.
NONPOLAR_MAX = 0.4

#: Δχ at or above which a bond is conventionally called ionic. Pauling's
#: relation puts this at just over 50% ionic character.
IONIC_MIN = 1.7


class Scale(StrEnum):
    """Which electronegativity scale a number is on."""

    PAULING = "Pauling"
    MULLIKEN = "Mulliken"

    @property
    def unit(self) -> str:
        """Mulliken values are energies; Pauling's scale is unitless."""

        return "eV" if self is Scale.MULLIKEN else ""


def pauling(symbol: str) -> float | None:
    """
    Pauling electronegativity, or ``None`` where it is undefined.

    pymatgen reports NaN rather than raising for the elements that have no
    value, so the check is for NaN and not for absence.
    """

    try:
        value = Element(symbol).X

    except (ValueError, KeyError, AttributeError):
        return None

    if value is None or math.isnan(value):
        return None

    return float(value)


def mulliken(symbol: str) -> float | None:
    """
    Mulliken electronegativity ``(IE + EA) / 2`` in eV, or ``None``.

    The electron affinity is used as tabulated, including where it is
    negative -- an atom that does not bind an extra electron. That is
    Mulliken's definition; some tabulations clamp it to zero instead,
    which raises the noble gases.
    """

    try:
        element = Element(symbol)
        ionization = element.ionization_energy
        affinity = element.electron_affinity

    except (ValueError, KeyError, AttributeError):
        return None

    if ionization is None or affinity is None:
        return None

    return (float(ionization) + float(affinity)) / 2.0


#: Lookup per scale, so the scale can be chosen and then applied uniformly.
SCALES = {
    Scale.PAULING: pauling,
    Scale.MULLIKEN: mulliken,
}


def chi(symbol: str, scale: Scale = Scale.PAULING) -> float | None:
    """Electronegativity of one element on ``scale``."""

    return SCALES[scale](symbol)


def common_scale(symbols) -> Scale | None:
    """
    The best scale that covers every one of ``symbols``.

    Pauling when it has a value for all of them, otherwise Mulliken, and
    ``None`` when neither scale is complete -- comparing across scales
    would produce a difference that means nothing.
    """

    unique = list(dict.fromkeys(symbols))

    for scale in (Scale.PAULING, Scale.MULLIKEN):

        if all(chi(symbol, scale) is not None for symbol in unique):
            return scale

    return None


# ----------------------------------------------------------------
# A pair of atoms
# ----------------------------------------------------------------


def ionic_character(delta: float) -> float:
    """
    Fraction of ionic character from Δχ, by Pauling's relation.

    ``1 - exp(-Δχ² / 4)``. Only meaningful for Pauling differences.
    """

    return 1.0 - math.exp(-(delta**2) / 4.0)


def classify(delta: float) -> str:
    """Conventional name for a bond with this Pauling Δχ."""

    if delta < NONPOLAR_MAX:
        return "nonpolar covalent"

    if delta < IONIC_MIN:
        return "polar covalent"

    return "ionic"


@dataclass(frozen=True)
class PairResult:
    """Electronegativity comparison between two atoms."""

    symbol_a: str
    symbol_b: str

    chi_a: float | None = None
    chi_b: float | None = None

    scale: Scale | None = None

    #: Separation in Angstrom, minimum-image. ``None`` when not supplied.
    distance: float | None = None

    #: Whether the two are bonded according to the viewer's own bonding
    #: rules. ``None`` when the caller did not say.
    bonded: bool | None = None

    @property
    def defined(self) -> bool:
        """True when both atoms have a value on a shared scale."""

        return self.chi_a is not None and self.chi_b is not None

    @property
    def delta(self) -> float | None:
        """|Δχ|, on :attr:`scale`."""

        if not self.defined:
            return None

        return abs(self.chi_a - self.chi_b)

    @property
    def more_electronegative(self) -> str | None:
        """
        Which element pulls the electron density, or ``None`` if tied.

        The direction is the useful half of the answer and the magnitude
        alone throws it away.
        """

        if not self.defined or self.chi_a == self.chi_b:
            return None

        return self.symbol_a if self.chi_a > self.chi_b else self.symbol_b

    @property
    def ionic_percent(self) -> float | None:
        """
        Percent ionic character, Pauling only.

        Withheld on the Mulliken scale: the relation is fitted to Pauling
        differences and an eV difference put through it is a number with
        no meaning.
        """

        if self.delta is None or self.scale is not Scale.PAULING:
            return None

        return 100.0 * ionic_character(self.delta)

    @property
    def character(self) -> str | None:
        """Bond character label, Pauling only, for the same reason."""

        if self.delta is None or self.scale is not Scale.PAULING:
            return None

        return classify(self.delta)


def compare(
    symbol_a: str,
    symbol_b: str,
    distance: float | None = None,
    bonded: bool | None = None,
) -> PairResult:
    """Compare two elements on whichever scale covers both."""

    scale = common_scale((symbol_a, symbol_b))

    if scale is None:
        return PairResult(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            distance=distance,
            bonded=bonded,
        )

    return PairResult(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        chi_a=chi(symbol_a, scale),
        chi_b=chi(symbol_b, scale),
        scale=scale,
        distance=distance,
        bonded=bonded,
    )


# ----------------------------------------------------------------
# The coordination shell around one atom
# ----------------------------------------------------------------

#: Resultant/total ratio below which a coordination shell is called
#: balanced -- the pulls cancel, as they do for a regular octahedron of
#: identical ligands.
BALANCED_MAX = 0.10

#: Ratio above which the shell pulls decidedly one way.
LOPSIDED_MIN = 0.40


@dataclass(frozen=True)
class Ligand:
    """One atom of a coordination shell, seen from the centre."""

    symbol: str

    #: Bond length in Angstrom.
    distance: float

    #: χ(ligand) − χ(centre). Positive means this neighbour pulls
    #: electron density away from the central atom.
    delta: float | None

    #: Unit vector from the centre towards this ligand.
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _normalise(vector) -> tuple[float, float, float]:
    """Unit vector, or the zero vector when there is no direction."""

    x, y, z = (float(component) for component in vector)

    length = math.sqrt(x * x + y * y + z * z)

    if length == 0.0:
        return (0.0, 0.0, 0.0)

    return (x / length, y / length, z / length)


@dataclass(frozen=True)
class PolyhedronResult:
    """
    The electronegativity an atom faces from its whole coordination shell.

    The per-bond numbers say what each neighbour does; this says what
    they add up to. Two separate things matter, and a single Δχ hides
    both:

    * **How electronegative the shell is on average** -- the geometric
      mean over the ligands, against the centre.
    * **Whether it is balanced.** Six identical ligands around a
      centrosymmetric site pull equally in opposite directions and
      cancel; the same six arranged to one side do not. Treating each
      bond as a dipole along the bond direction with magnitude Δχ and
      summing them gives that, and it is what distinguishes a regular
      octahedron from a distorted or mixed one.
    """

    centre: str

    chi_centre: float | None = None

    scale: Scale | None = None

    #: Nearest first.
    ligands: tuple[Ligand, ...] = ()

    @property
    def coordination(self) -> int:
        """How many atoms are bonded to the centre."""

        return len(self.ligands)

    @property
    def defined(self) -> bool:
        """True when the centre and every ligand have a value."""

        return (
            self.chi_centre is not None
            and self.coordination > 0
            and all(ligand.delta is not None for ligand in self.ligands)
        )

    @property
    def composition(self) -> str:
        """The shell's formula, e.g. ``O6`` or ``O4 Li2``."""

        counts: dict[str, int] = {}

        for ligand in self.ligands:
            counts[ligand.symbol] = counts.get(ligand.symbol, 0) + 1

        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

        return " ".join(
            symbol if count == 1 else f"{symbol}{count}"
            for symbol, count in ordered
        )

    @property
    def mean_ligand_chi(self) -> float | None:
        """
        Geometric mean of the ligands' electronegativity.

        Geometric rather than arithmetic to stay consistent with the
        Sanderson mean used for the cell, so the two numbers can be read
        against each other.
        """

        if not self.defined:
            return None

        values = [
            self.chi_centre + ligand.delta for ligand in self.ligands
        ]

        if any(value <= 0.0 for value in values):
            return None

        total = sum(math.log(value) for value in values)

        return math.exp(total / len(values))

    @property
    def net_delta(self) -> float | None:
        """
        Mean ligand χ minus the centre's.

        Positive: the shell as a whole pulls electron density off the
        central atom.
        """

        mean = self.mean_ligand_chi

        if mean is None:
            return None

        return mean - self.chi_centre

    @property
    def resultant(self) -> float | None:
        """
        Magnitude of the summed bond dipoles, in Δχ units.

        Each bond contributes ``Δχ`` along its own direction. Bond length
        is deliberately left out: this is the composition and geometry of
        the shell, not an attempt at a real dipole moment in debye.
        """

        if not self.defined:
            return None

        total = [0.0, 0.0, 0.0]

        for ligand in self.ligands:

            for axis in range(3):
                total[axis] += ligand.delta * ligand.direction[axis]

        return math.sqrt(sum(component**2 for component in total))

    @property
    def scalar_total(self) -> float | None:
        """Sum of |Δχ| over the shell, ignoring direction."""

        if not self.defined:
            return None

        return sum(abs(ligand.delta) for ligand in self.ligands)

    @property
    def asymmetry(self) -> float | None:
        """
        Resultant as a fraction of what it would be if all pulls aligned.

        0 means the shell cancels itself out; 1 means every neighbour
        pulls the same way.
        """

        total = self.scalar_total

        if total is None or total == 0.0:
            return None

        return self.resultant / total

    def attracting(self) -> "PolyhedronResult":
        """
        The same shell with only the neighbours that pull density away.

        The bonding rule that draws the sticks also joins cation to
        cation -- Mn to Mn at 2.87 Å in a layered oxide -- and those
        contacts drag the mean towards the centre's own value while
        contributing nothing to the resultant. Dropping them leaves the
        coordination polyhedron in the usual sense: the MnO6 of an
        oxide, not its fourteen nearest neighbours.
        """

        return PolyhedronResult(
            centre=self.centre,
            chi_centre=self.chi_centre,
            scale=self.scale,
            ligands=tuple(
                ligand
                for ligand in self.ligands
                if ligand.delta is not None and ligand.delta > 0.0
            ),
        )

    @property
    def balance(self) -> str | None:
        """Plain-language reading of :attr:`asymmetry`."""

        ratio = self.asymmetry

        if ratio is None:
            return None

        if ratio < BALANCED_MAX:
            return "balanced -- the pulls cancel"

        if ratio < LOPSIDED_MIN:
            return "slightly one-sided"

        return "one-sided -- the shell pulls one way"


def polyhedron(centre: str, neighbours) -> PolyhedronResult:
    """
    Describe the coordination shell around one atom.

    ``neighbours`` is an iterable of ``(symbol, distance, direction)``,
    where ``direction`` points from the centre to the neighbour and need
    not be normalised. Coordination is whatever the caller counts as
    bonded -- here that is the viewer's own bond graph, so the shell is
    exactly the sticks drawn on screen.
    """

    neighbours = [
        (symbol, float(distance), _normalise(direction))
        for symbol, distance, direction in neighbours
    ]

    symbols = [centre] + [symbol for symbol, _, _ in neighbours]

    scale = common_scale(symbols)

    chi_centre = chi(centre, scale) if scale is not None else None

    ligands = tuple(
        Ligand(
            symbol=symbol,
            distance=distance,
            delta=(
                None
                if scale is None or chi_centre is None
                else chi(symbol, scale) - chi_centre
            ),
            direction=direction,
        )
        for symbol, distance, direction in sorted(
            neighbours, key=lambda item: item[1]
        )
    )

    return PolyhedronResult(
        centre=centre,
        chi_centre=chi_centre,
        scale=scale,
        ligands=ligands,
    )


# ----------------------------------------------------------------
# The cell as a whole
# ----------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """Sanderson electronegativity of a whole cell."""

    #: Every atom in the cell, in order, so the mean is stoichiometric.
    symbols: tuple[str, ...] = ()

    #: Geometric mean of the atoms' electronegativities.
    equalized: float | None = None

    scale: Scale | None = None

    @property
    def defined(self) -> bool:
        """True when a mean could be formed."""

        return self.equalized is not None

    def deviation(self, symbol: str) -> float | None:
        """
        χ(atom) − χ(cell) for one element.

        Positive means the site is more electronegative than the cell it
        sits in, so charge flows towards it.
        """

        if not self.defined:
            return None

        value = chi(symbol, self.scale)

        if value is None:
            return None

        return value - self.equalized


#: Half-width used when every atom has the same shift, so a colour bar
#: over a single value still has a range to draw.
FLAT_SPAN = 0.05


def symmetric_limits(values, minimum: float = FLAT_SPAN) -> tuple[float, float]:
    """
    Colour limits centred on zero.

    A shift is signed, and the sign is the point: negative gives electron
    density up, positive pulls it in. Letting the colour scale run from
    the smallest value to the largest would put zero at whatever colour
    happened to fall in the middle, so "no shift" would read as blue in
    one structure and orange in the next. Anchoring the range on zero
    keeps the midpoint meaning the same thing everywhere.
    """

    magnitudes = [abs(value) for value in values if value is not None]

    largest = max(magnitudes) if magnitudes else 0.0

    largest = max(largest, minimum)

    return (-largest, largest)


def cell_electronegativity(symbols) -> CellResult:
    """
    Sanderson's equalized electronegativity for a cell.

    The geometric mean over *every atom*, not over the distinct elements:
    equalization is a property of the assembled compound, so an atom that
    appears eight times pulls the mean eight times.
    """

    symbols = tuple(symbols)

    if not symbols:
        return CellResult()

    scale = common_scale(symbols)

    if scale is None:
        return CellResult(symbols=symbols)

    values = [chi(symbol, scale) for symbol in symbols]

    # A non-positive value has no geometric mean. Mulliken values can go
    # negative in principle, for an element whose electron affinity is
    # more negative than its ionization energy is large.
    if any(value <= 0.0 for value in values):
        return CellResult(symbols=symbols)

    log_mean = sum(math.log(value) for value in values) / len(values)

    return CellResult(
        symbols=symbols,
        equalized=math.exp(log_mean),
        scale=scale,
    )

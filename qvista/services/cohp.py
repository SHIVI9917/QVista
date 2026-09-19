"""
Bonding analysis from a LOBSTER run: COHP, COOP and COBI.

LOBSTER projects a plane-wave calculation back onto a local basis and
reports, for every pair of atoms it found, how much each energy
contributes to the interaction between them.  Three flavours of that
number exist and this module reads all three, because they answer
different questions and a run usually writes them together.

The sign conventions differ, which is the one thing worth getting right
before reading any of it.

``COHP``
    Crystal Orbital Hamilton Population, in eV.  *Negative is bonding*
    and positive is antibonding, so it is universally plotted as
    ``-COHP`` with bonding to the right.  The integral up to the Fermi
    level, ICOHP, is the bond strength; it too is negative for a bond,
    so strength is quoted as ``-ICOHP`` in eV and a bigger number is a
    stronger bond.

``COOP``
    Crystal Orbital Overlap Population, dimensionless.  *Positive is
    bonding*, the opposite convention, and it is plotted as it comes.

``COBI``
    Crystal Orbital Bond Index.  Positive, and close in spirit to a bond
    order.

:data:`KINDS` carries which is which, so nothing downstream has to
remember, and :attr:`Bond.strength` is always positive for a bonding
interaction whichever one is loaded.

Where the numbers come from
---------------------------
``ICOHPLIST.lobster``
    The table: one row per interaction, with the two atoms, the distance
    between them, the periodic image, and the integrated value per spin
    channel.  This is the exact integral and is what the tables show.

``COHPCAR.lobster``
    The curves, on an energy grid already shifted so the Fermi level is
    at zero.  Its running integral agrees with ICOHPLIST to about the
    grid spacing, so the curves are used for plotting and the list for
    numbers.

The ``COOP`` and ``COBI`` equivalents have the same two files under
their own names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class CohpError(RuntimeError):
    """A LOBSTER run could not be read."""


# ------------------------------------------------------------
# The three flavours
# ------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    """One of the three quantities, and how to handle it."""

    name: str
    curves: str
    table: str
    flip: bool
    """Whether bonding is negative, and so the reported value is negated."""

    unit: str
    strength_label: str
    curve_label: str


KINDS: dict[str, Kind] = {
    "COHP": Kind(
        name="COHP",
        curves="COHPCAR.lobster",
        table="ICOHPLIST.lobster",
        flip=True,
        unit="eV",
        strength_label="-ICOHP",
        curve_label="-COHP",
    ),
    "COOP": Kind(
        name="COOP",
        curves="COOPCAR.lobster",
        table="ICOOPLIST.lobster",
        flip=False,
        unit="",
        strength_label="ICOOP",
        curve_label="COOP",
    ),
    "COBI": Kind(
        name="COBI",
        curves="COBICAR.lobster",
        table="ICOBILIST.lobster",
        flip=False,
        unit="",
        strength_label="ICOBI",
        curve_label="COBI",
    ),
}

#: Structure files to try, in order.  LOBSTER writes the first one
#: itself, which makes it the one guaranteed to match the site numbering
#: in its own output.
STRUCTURES = ("POSCAR.lobster.vasp", "POSCAR.lobster", "CONTCAR", "POSCAR")

#: Rows of ``ICOHPLIST.lobster``:
#: ``1  Li1  Li6  2.82962  0 0 0  -0.05635  -0.05533``
_ROW = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]{1,3}\d+)\s+([A-Za-z]{1,3}\d+)\s+"
    r"([-\d.Ee+]+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(.*)$"
)

#: ``Li1`` -> ``Li``
_ELEMENT = re.compile(r"^([A-Za-z]{1,3})")


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------


@dataclass
class Bond:
    """One interaction between two atoms."""

    index: str
    a_label: str
    b_label: str
    length: float
    translation: tuple[int, int, int]
    values: list[float]
    """The integrated value per spin channel, as LOBSTER wrote it."""

    kind: Kind
    a_site: int | None = None
    b_site: int | None = None
    energies: np.ndarray | None = None
    curve: np.ndarray | None = None
    """The curve summed over spin channels, as LOBSTER wrote it."""

    integrated: np.ndarray | None = None

    @property
    def a_element(self) -> str:

        found = _ELEMENT.match(self.a_label)

        return found.group(1) if found else self.a_label

    @property
    def b_element(self) -> str:

        found = _ELEMENT.match(self.b_label)

        return found.group(1) if found else self.b_label

    @property
    def pair(self) -> str:
        """``Co-O``, in a fixed order so one kind of bond is one row."""

        return "-".join(sorted((self.a_element, self.b_element)))

    @property
    def raw(self) -> float:
        """The integrated value, summed over spin channels."""

        return float(sum(self.values))

    @property
    def strength(self) -> float:
        """
        Bond strength, positive when the interaction is bonding.

        This is the only place the sign convention lives: ``-ICOHP`` for
        COHP, and the value itself for COOP and COBI.
        """

        return -self.raw if self.kind.flip else self.raw

    @property
    def signed_curve(self) -> np.ndarray | None:
        """The curve as it should be drawn."""

        if self.curve is None:
            return None

        return -self.curve if self.kind.flip else self.curve

    @property
    def signed_integrated(self) -> np.ndarray | None:

        if self.integrated is None:
            return None

        return -self.integrated if self.kind.flip else self.integrated

    def label(self) -> str:

        inside = self.translation == (0, 0, 0)

        return (
            f"{self.a_label}-{self.b_label}"
            + ("" if inside else "  {},{},{}".format(*self.translation))
        )


@dataclass(frozen=True)
class PairSummary:
    """Every interaction between the same two elements, together."""

    pair: str
    count: int
    strength_mean: float
    strength_min: float
    strength_max: float
    strength_total: float
    length_mean: float
    length_min: float
    length_max: float


@dataclass(frozen=True)
class AtomSummary:
    """One atom's bonding, summed over the interactions it takes part in."""

    label: str
    element: str
    site: int | None
    count: int
    strength_total: float
    strength_mean: float
    length_mean: float


@dataclass
class CohpResult:
    """Everything read out of one LOBSTER run, for one quantity."""

    kind: Kind
    bonds: list[Bond]
    structure: object | None
    folder: Path
    energies: np.ndarray | None = None
    average: np.ndarray | None = None
    average_integrated: np.ndarray | None = None
    efermi: float | None = None
    spin_polarized: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def has_curves(self) -> bool:

        return self.energies is not None

    def elements(self) -> list[str]:

        seen: list[str] = []

        for bond in self.bonds:
            for element in (bond.a_element, bond.b_element):
                if element not in seen:
                    seen.append(element)

        return seen

    def pairs(self) -> list[str]:

        seen: list[str] = []

        for bond in self.bonds:
            if bond.pair not in seen:
                seen.append(bond.pair)

        return sorted(seen)

    @property
    def signed_average(self) -> np.ndarray | None:

        if self.average is None:
            return None

        return -self.average if self.kind.flip else self.average


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


def available(folder: Path | str) -> list[str]:
    """Which quantities this folder has files for."""

    folder = Path(folder)

    found = []

    for name, kind in KINDS.items():

        if (folder / kind.table).is_file() or (folder / kind.curves).is_file():
            found.append(name)

    return found


def _structure(folder: Path):
    """The geometry LOBSTER analysed, if one of its usual names is there."""

    from pymatgen.core import Structure

    for name in STRUCTURES:

        path = folder / name

        if not path.is_file() or not path.stat().st_size:
            continue

        try:
            return Structure.from_file(str(path))

        except Exception:  # noqa: BLE001 - try the next candidate
            continue

    return None


def _read_table(path: Path, kind: Kind) -> tuple[list[Bond], bool]:
    """
    Parse ``ICOHPLIST.lobster`` and friends.

    Read directly rather than through pymatgen, which drops the atom
    labels on the floor -- and the labels are what make a table of a
    hundred and sixty bonds readable.
    """

    bonds: list[Bond] = []
    spin_polarized = False

    with open(path, "r", errors="ignore") as handle:

        for line in handle:

            found = _ROW.match(line)

            if not found:
                continue

            numbers = [
                float(piece)
                for piece in found.group(8).split()
                if _looks_numeric(piece)
            ]

            if not numbers:
                continue

            # Two trailing numbers means two spin channels. Newer LOBSTER
            # versions append a bond count after them, which is an
            # integer and never the value being read.
            values = numbers[:2] if len(numbers) >= 2 else numbers[:1]

            if len(values) == 2:
                spin_polarized = True

            bonds.append(
                Bond(
                    index=found.group(1),
                    a_label=found.group(2),
                    b_label=found.group(3),
                    length=float(found.group(4)),
                    translation=(
                        int(found.group(5)),
                        int(found.group(6)),
                        int(found.group(7)),
                    ),
                    values=values,
                    kind=kind,
                )
            )

    if not bonds:
        raise CohpError(f"No interactions could be read from {path.name}.")

    return bonds, spin_polarized


def _looks_numeric(piece: str) -> bool:

    try:
        float(piece)

    except ValueError:
        return False

    return True


def _read_curves(path: Path, kind: Kind):
    """The curves, through pymatgen, which knows the header layout."""

    from pymatgen.io.lobster import Cohpcar

    return Cohpcar(
        filename=str(path),
        are_coops=kind.name == "COOP",
        are_cobis=kind.name == "COBI",
    )


#: pymatgen keys every curve it reads as ``COHP`` and ``ICOHP``, whether
#: the file was a COHPCAR, a COOPCAR or a COBICAR.  The quantity is
#: carried by the reader's flags, not by the key.
_CURVE_KEY = "COHP"
_INTEGRAL_KEY = "ICOHP"


def _summed(block: dict) -> np.ndarray:
    """Add the spin channels of one curve together."""

    return np.sum([np.asarray(values) for values in block.values()], axis=0)


def read(folder: Path | str, quantity: str = "COHP") -> CohpResult:
    """
    Read one quantity out of a LOBSTER run.

    The table is required; the curves are optional, so a run that wrote
    only the integrated values still gives a usable table.
    """

    if quantity not in KINDS:
        raise CohpError(f"Unknown quantity {quantity!r}.")

    kind = KINDS[quantity]

    folder = Path(folder)

    if not folder.is_dir():
        raise CohpError(f"{folder} is not a folder.")

    table = folder / kind.table
    curves = folder / kind.curves

    if not table.is_file():
        raise CohpError(
            f"{kind.table} is not in {folder}. LOBSTER writes it when the "
            f"lobsterin asks for {kind.name}."
        )

    notes: list[str] = []

    bonds, spin_polarized = _read_table(table, kind)

    structure = _structure(folder)

    if structure is None:
        notes.append(
            "No structure file in the folder, so the bonds cannot be drawn. "
            "LOBSTER writes POSCAR.lobster.vasp beside its output."
        )

    energies = average = average_integrated = None
    efermi = None

    if curves.is_file():

        data = _read_curves(curves, kind)

        energies = np.asarray(data.energies, dtype=float)
        efermi = float(data.efermi)

        block = data.cohp_data.get("average")

        if block:
            average = _summed(block[_CURVE_KEY])
            average_integrated = _summed(block[_INTEGRAL_KEY])

        matched = 0

        for bond in bonds:

            entry = data.cohp_data.get(bond.index)

            if entry is None:
                continue

            bond.energies = energies
            bond.curve = _summed(entry[_CURVE_KEY])
            bond.integrated = _summed(entry[_INTEGRAL_KEY])

            sites = entry.get("sites")

            if sites is not None and len(sites) == 2:
                bond.a_site, bond.b_site = int(sites[0]), int(sites[1])

            # The two files are written from the same interaction list, so
            # a disagreement on a bond length means they are not from the
            # same run and nothing downstream can be trusted.
            length = entry.get("length")

            if length is not None and abs(length - bond.length) > 1e-3:
                raise CohpError(
                    f"{kind.table} and {kind.curves} disagree about bond "
                    f"{bond.index}: {bond.length:.5f} against "
                    f"{float(length):.5f} A. They are from different runs."
                )

            matched += 1

        if matched != len(bonds):
            notes.append(
                f"{len(bonds) - matched} of {len(bonds)} interactions have "
                f"no curve in {kind.curves}; their numbers are still exact."
            )

    else:
        notes.append(
            f"No {kind.curves} in the folder, so there is nothing to plot. "
            "The table below is complete."
        )

    return CohpResult(
        kind=kind,
        bonds=bonds,
        structure=structure,
        folder=folder,
        energies=energies,
        average=average,
        average_integrated=average_integrated,
        efermi=efermi,
        spin_polarized=spin_polarized,
        notes=notes,
    )


# ------------------------------------------------------------
# Selecting
# ------------------------------------------------------------


def select(
    bonds: Iterable[Bond],
    max_length: float | None = None,
    min_strength: float | None = None,
    pairs: Sequence[str] | None = None,
) -> list[Bond]:
    """
    Narrow a bond list, strongest first.

    LOBSTER reports every interaction inside its cutoff, which for a
    dense structure includes a long tail of second and third neighbours
    contributing almost nothing.
    """

    kept = list(bonds)

    if max_length is not None:
        kept = [bond for bond in kept if bond.length <= max_length]

    if min_strength is not None:
        kept = [bond for bond in kept if bond.strength >= min_strength]

    if pairs:
        wanted = set(pairs)
        kept = [bond for bond in kept if bond.pair in wanted]

    return sorted(kept, key=lambda bond: -bond.strength)


def by_pair(bonds: Iterable[Bond]) -> list[PairSummary]:
    """One row per pair of elements."""

    grouped: dict[str, list[Bond]] = {}

    for bond in bonds:
        grouped.setdefault(bond.pair, []).append(bond)

    rows: list[PairSummary] = []

    for pair, members in grouped.items():

        strengths = np.array([bond.strength for bond in members])
        lengths = np.array([bond.length for bond in members])

        rows.append(
            PairSummary(
                pair=pair,
                count=len(members),
                strength_mean=float(strengths.mean()),
                strength_min=float(strengths.min()),
                strength_max=float(strengths.max()),
                strength_total=float(strengths.sum()),
                length_mean=float(lengths.mean()),
                length_min=float(lengths.min()),
                length_max=float(lengths.max()),
            )
        )

    return sorted(rows, key=lambda row: -row.strength_mean)


def by_atom(bonds: Iterable[Bond]) -> list[AtomSummary]:
    """
    One row per atom, over the bonds it takes part in.

    An atom appears in a bond from either end, so both are counted; the
    total is that atom's whole bonding, which is what separates an
    under-coordinated site from a well bound one.
    """

    grouped: dict[str, list[tuple[Bond, int | None]]] = {}

    for bond in bonds:

        grouped.setdefault(bond.a_label, []).append((bond, bond.a_site))
        grouped.setdefault(bond.b_label, []).append((bond, bond.b_site))

    rows: list[AtomSummary] = []

    for label, members in grouped.items():

        strengths = np.array([bond.strength for bond, _ in members])
        lengths = np.array([bond.length for bond, _ in members])

        site = next((site for _, site in members if site is not None), None)

        found = _ELEMENT.match(label)

        rows.append(
            AtomSummary(
                label=label,
                element=found.group(1) if found else label,
                site=site,
                count=len(members),
                strength_total=float(strengths.sum()),
                strength_mean=float(strengths.mean()),
                length_mean=float(lengths.mean()),
            )
        )

    return sorted(rows, key=lambda row: -row.strength_total)


def summed_curve(bonds: Sequence[Bond]) -> tuple[np.ndarray, np.ndarray] | None:
    """
    The curves of several bonds added together, ready to draw.

    Adding is the meaningful operation: the COHP of a coordination shell
    is the sum over its bonds, not their average.
    """

    usable = [bond for bond in bonds if bond.signed_curve is not None]

    if not usable:
        return None

    energies = usable[0].energies

    curve = np.sum([bond.signed_curve for bond in usable], axis=0)
    integrated = np.sum(
        [
            bond.signed_integrated
            for bond in usable
            if bond.signed_integrated is not None
        ],
        axis=0,
    )

    return energies, np.vstack([curve, integrated])


def strength_at_fermi(bond: Bond) -> float | None:
    """
    The bond strength read off the curve rather than the table.

    Only useful as a check on the two files agreeing; the table carries
    the exact integral and the curve is on a grid.
    """

    if bond.signed_integrated is None or bond.energies is None:
        return None

    return float(bond.signed_integrated[int(np.argmin(np.abs(bond.energies)))])

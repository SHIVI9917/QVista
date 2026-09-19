"""
DDEC6 charges and bond orders from a Chargemol run.

DDEC6 partitions the electron density between atoms by an iterative
stockholder scheme rather than by a dividing surface: every atom gets a
weighted share of the density at every point, with the weights taken
from conditioned reference ion densities [1, 2].  It also gives bond
orders, from a dressed exchange hole built out of the electron and spin
densities [3].

Unlike the Bader tool, none of that is computed here.  DDEC6 needs a
library of precomputed reference ion densities and a fixed sequence of
seven charge partitioning steps, and the reference implementation is the
Chargemol program [4], which is a separate download.  This module reads
what Chargemol wrote, and can run it when the user says where it is.

What it adds
------------
Chargemol reports each bond as a pair of atom numbers plus the periodic
image the second one sits in, and leaves it at that.  The bond *length*
is what makes a bond order mean something -- a 0.4 bond order at 1.9 A
and at 3.1 A are different claims -- so this module reconstructs it from
the image vector and the lattice, which is the one number the output
file does not carry.

Files it reads
--------------
``DDEC6_even_tempered_net_atomic_charges.xyz``
    Required.  Carries the net charges, the lattice, and the atomic
    dipoles and quadrupoles.
``DDEC6_even_tempered_bond_orders.xyz``
    Bond orders and the sum of bond orders per atom.
``DDEC6_even_tempered_atomic_spin_moments.xyz``
    Only written for a spin polarised run.
``VASP_DDEC_analysis.output``
    CM5 charges, when Chargemol computed them.

The ``DDEC3`` equivalents are read as a fallback, since a run made with
the older method has the same file layout under a different name.

References
----------
[1] T A Manz and N Gabaldon Limas, *Introducing DDEC6 atomic population
    analysis: part 1*, RSC Adv. **6**, 47771 (2016).
[2] N Gabaldon Limas and T A Manz, *part 2*, RSC Adv. **6**, 45727 (2016).
[3] T A Manz, *part 3: comprehensive method to compute bond orders*,
    RSC Adv. **7**, 45552 (2017).
[4] Chargemol, https://sourceforge.net/projects/ddec/
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


class DdecError(RuntimeError):
    """A DDEC6 analysis could not be read or run."""


# ------------------------------------------------------------
# File names
# ------------------------------------------------------------

#: The output files, newest method first.  Each entry is what the file
#: holds and the names to look for, in order of preference.
FILES: dict[str, tuple[str, ...]] = {
    "charges": (
        "DDEC6_even_tempered_net_atomic_charges.xyz",
        "DDEC3_net_atomic_charges.xyz",
    ),
    "bonds": (
        "DDEC6_even_tempered_bond_orders.xyz",
        "DDEC3_bond_orders.xyz",
    ),
    "spins": (
        "DDEC6_even_tempered_atomic_spin_moments.xyz",
        "DDEC3_atomic_spin_moments.xyz",
    ),
    "analysis": (
        "VASP_DDEC_analysis.output",
        "DDEC_analysis.output",
    ),
}

#: Where a Chargemol install is remembered between sessions.
EXE_KEY = "ddec/executable"
DENSITIES_KEY = "ddec/atomic_densities"

#: Bond orders below this are noise; Chargemol itself prints down to
#: 0.001 and says so in the file.
DEFAULT_MIN_ORDER = 0.05


# ------------------------------------------------------------
# Line formats
# ------------------------------------------------------------

#: ``Printing BOs for ATOM #      1 ( Na ) in the reference unit cell.``
_ATOM_BLOCK = re.compile(
    r"Printing BOs for ATOM\s*#\s*(\d+)\s*\(\s*([A-Za-z]+)\s*\)"
)

#: ``Bonded to the ( -1,  -1,   0) translated image of atom number
#: 2 ( Cl ) with bond order =     0.0882    The average spin
#: polarization of this bonding =     0.0000``
#:
#: Matched with a regular expression rather than by splitting on
#: whitespace and counting: Chargemol pads its columns to fit the
#: numbers, so ``( -1,`` becomes ``(-1,`` once an index needs the space,
#: and every field after it shifts by one.
_BONDED = re.compile(
    r"Bonded to the\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*"
    r"translated image of atom number\s*(\d+)\s*\(\s*([A-Za-z]+)\s*\)\s*"
    r"with bond order\s*=\s*([-+0-9.EeDd]+)"
)

#: ``The average spin polarization of this bonding =     0.0000``
_SPIN_POLARIZATION = re.compile(
    r"spin polarization of this bonding\s*=\s*([-+0-9.EeDd]+)"
)

#: ``The sum of bond orders for this atom is SBO =        0.539920``
_SBO = re.compile(
    r"sum of bond orders for this atom is SBO\s*=\s*([-+0-9.EeDd]+)"
)

#: The lattice, written into the jmol script on the second line.
_UNITCELL = re.compile(r"\{([^}]*)\}")


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    """One atom's DDEC6 numbers."""

    index: int
    """Counting from 1, as Chargemol prints them."""

    symbol: str
    coords: np.ndarray
    charge: float
    """Net atomic charge: positive when electrons were given away."""

    spin: float | None = None
    dipole: float | None = None
    sbo: float | None = None
    """Sum of bond orders -- roughly, the atom's total valence."""

    cm5: float | None = None

    def label(self) -> str:

        return f"{self.index} {self.symbol}"


@dataclass(frozen=True)
class Bond:
    """One bond, between an atom in the cell and a periodic image."""

    a: int
    a_symbol: str
    b: int
    b_symbol: str
    image: tuple[int, int, int]
    order: float
    length: float
    spin_polarization: float = 0.0

    @property
    def pair(self) -> str:
        """``Li-O``, with the two elements in a fixed order so that the
        same kind of bond always lands in the same row."""

        return "-".join(sorted((self.a_symbol, self.b_symbol)))

    def label(self) -> str:

        inside = self.image == (0, 0, 0)

        return (
            f"{self.a} {self.a_symbol} - {self.b} {self.b_symbol}"
            + ("" if inside else f"  {self.image}")
        )


@dataclass(frozen=True)
class GroupCharge:
    """A set of atoms taken together."""

    label: str
    members: tuple[int, ...]
    charge: float
    spin: float | None
    sbo: float | None

    @property
    def count(self) -> int:

        return len(self.members)

    @property
    def charge_per_atom(self) -> float:

        return self.charge / self.count if self.count else 0.0


@dataclass(frozen=True)
class BondType:
    """Every bond between the same pair of elements, summarised."""

    pair: str
    count: int
    order_mean: float
    order_min: float
    order_max: float
    length_mean: float
    length_min: float
    length_max: float


@dataclass
class DdecResult:
    """Everything read out of one Chargemol run."""

    atoms: list[Atom]
    bonds: list[Bond]
    lattice: np.ndarray | None
    folder: Path
    method: str = "DDEC6"
    notes: list[str] = field(default_factory=list)

    @property
    def periodic(self) -> bool:

        return self.lattice is not None

    @property
    def total_charge(self) -> float:

        return sum(atom.charge for atom in self.atoms)

    def elements(self) -> list[str]:
        """The species, in the order they first appear."""

        seen: list[str] = []

        for atom in self.atoms:
            if atom.symbol not in seen:
                seen.append(atom.symbol)

        return seen

    def structure(self):
        """
        A pymatgen structure, for the viewer to draw.

        Built from the coordinates in the charges file rather than from a
        POSCAR, so the picture is of the geometry Chargemol actually
        analysed and cannot drift from it.
        """

        from pymatgen.core import Lattice, Molecule, Structure

        symbols = [atom.symbol for atom in self.atoms]
        coords = [atom.coords for atom in self.atoms]

        if self.lattice is None:
            return Molecule(symbols, coords)

        return Structure(
            Lattice(self.lattice),
            symbols,
            coords,
            coords_are_cartesian=True,
        )


# ------------------------------------------------------------
# Finding the files
# ------------------------------------------------------------


def _find(folder: Path, kind: str) -> Path | None:
    """The first of a kind's candidate names that is actually there."""

    for name in FILES[kind]:

        path = folder / name

        if path.is_file() and path.stat().st_size:
            return path

    return None


def available(folder: Path | str) -> dict[str, Path]:
    """Which of the output files this folder has."""

    folder = Path(folder)

    found = {}

    for kind in FILES:

        path = _find(folder, kind)

        if path is not None:
            found[kind] = path

    return found


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


def _lattice_from_header(line: str) -> np.ndarray | None:
    """
    The cell, out of the jmol script Chargemol writes on line two.

    A non-periodic run says ``Nonperiodic system`` there instead, which
    is not a failure -- a molecule has no lattice and its bonds need no
    image vectors.
    """

    if "unitcell" not in line:
        return None

    # The line opens with a {1 1 1} supercell block before it gets to the
    # cell itself, so only what follows "unitcell" is the lattice.
    tail = line.split("unitcell", 1)[1]

    vectors = [
        [float(value) for value in block.split()]
        for block in _UNITCELL.findall(tail)
    ]

    if len(vectors) != 3 or any(len(vector) != 3 for vector in vectors):
        raise DdecError(
            "The lattice on line 2 of the charges file could not be read."
        )

    return np.array(vectors, dtype=float)


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray, list[float], np.ndarray | None]:
    """
    The xyz block every Chargemol file opens with.

    The layout is the same in all of them: a count, a comment line
    carrying the lattice, then one line per atom ending in whichever
    quantity that file is about.
    """

    with open(path, "r", errors="ignore") as handle:
        lines = handle.readlines()

    if len(lines) < 2:
        raise DdecError(f"{path.name} is too short to be a Chargemol file.")

    try:
        count = int(lines[0].split()[0])

    except (IndexError, ValueError) as exc:
        raise DdecError(
            f"{path.name} does not start with an atom count."
        ) from exc

    lattice = _lattice_from_header(lines[1])

    symbols: list[str] = []
    coords: list[list[float]] = []
    values: list[float] = []

    for line in lines[2 : 2 + count]:

        # A blank line ends the block. Reaching one early means the count
        # on line 1 and the atoms below it disagree, which the check
        # after this loop reports properly.
        if not line.strip():
            break

        parts = line.split()

        if len(parts) < 5:
            raise DdecError(
                f"{path.name} has a short atom line: {line.strip()!r}"
            )

        symbols.append(parts[0])
        coords.append([float(value) for value in parts[1:4]])
        values.append(float(parts[4]))

    if len(symbols) != count:
        raise DdecError(
            f"{path.name} promises {count} atoms and carries {len(symbols)}."
        )

    return symbols, np.array(coords, dtype=float), values, lattice


def _read_dipoles(path: Path, count: int) -> list[float] | None:
    """
    Dipole magnitudes, from the wide table below the xyz block.

    Column 10 of the row that starts with the atom number.  Absent from
    a DDEC3 file, which is why this returns None rather than raising.
    """

    magnitudes: list[float] = []

    with open(path, "r", errors="ignore") as handle:

        started = False

        for line in handle:

            if "dipole_mag" in line:
                started = True
                continue

            if not started:
                continue

            parts = line.split()

            if len(parts) < 10 or not parts[0].isdigit():
                break

            magnitudes.append(float(parts[9]))

    return magnitudes if len(magnitudes) == count else None


def _read_cm5(path: Path, count: int) -> list[float] | None:
    """CM5 charges, when the analysis output carries them."""

    values: list[float] = []

    with open(path, "r", errors="ignore") as handle:

        started = False

        for line in handle:

            if "computed CM5" in line:
                started = True
                continue

            if started and "Hirshfeld and CM5" in line:
                break

            if started:
                for piece in line.split():
                    try:
                        values.append(float(piece))
                    except ValueError:
                        return None

    return values if len(values) == count else None


def bond_length(
    a: np.ndarray,
    b: np.ndarray,
    image: Sequence[int],
    lattice: np.ndarray | None,
) -> float:
    """
    How far apart two bonded atoms actually are.

    Chargemol names the second atom by its number and the image it sits
    in, so the bond it means is to ``b`` shifted by that image -- not to
    ``b`` in the reference cell, which for a bond across a face is a
    different and much longer distance.
    """

    offset = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)

    if lattice is not None:
        offset = offset + np.asarray(image, dtype=float) @ np.asarray(
            lattice, dtype=float
        )

    return float(np.linalg.norm(offset))


def _read_bonds(
    path: Path,
    coords: np.ndarray,
    symbols: Sequence[str],
    lattice: np.ndarray | None,
) -> tuple[list[Bond], list[float]]:
    """Every printed bond, with its length worked out."""

    bonds: list[Bond] = []
    sums: list[float] = [0.0] * len(symbols)

    current = 0

    with open(path, "r", errors="ignore") as handle:

        for line in handle:

            block = _ATOM_BLOCK.search(line)

            if block:
                current = int(block.group(1))
                continue

            total = _SBO.search(line)

            if total and 1 <= current <= len(sums):
                sums[current - 1] = float(total.group(1))
                continue

            found = _BONDED.search(line)

            if not found or not current:
                continue

            image = (
                int(found.group(1)),
                int(found.group(2)),
                int(found.group(3)),
            )

            other = int(found.group(4))

            if not (1 <= current <= len(symbols) and 1 <= other <= len(symbols)):
                raise DdecError(
                    f"{path.name} names atom {other}, but the run has "
                    f"{len(symbols)} atoms."
                )

            polarization = _SPIN_POLARIZATION.search(line)

            bonds.append(
                Bond(
                    a=current,
                    a_symbol=symbols[current - 1],
                    b=other,
                    b_symbol=symbols[other - 1],
                    image=image,
                    order=float(found.group(6)),
                    length=bond_length(
                        coords[current - 1], coords[other - 1], image, lattice
                    ),
                    spin_polarization=(
                        float(polarization.group(1)) if polarization else 0.0
                    ),
                )
            )

    return bonds, sums


def read(folder: Path | str) -> DdecResult:
    """
    Read a finished Chargemol run.

    Only the charges file is required; everything else fills in what it
    can and says in the notes what was missing.
    """

    folder = Path(folder)

    if not folder.is_dir():
        raise DdecError(f"{folder} is not a folder.")

    found = available(folder)

    if "charges" not in found:
        raise DdecError(
            "No DDEC net atomic charges file in that folder. Chargemol "
            "writes DDEC6_even_tempered_net_atomic_charges.xyz beside the "
            "density it was run on."
        )

    charges_path = found["charges"]

    symbols, coords, charges, lattice = _read_xyz(charges_path)

    count = len(symbols)

    method = "DDEC3" if "DDEC3" in charges_path.name else "DDEC6"

    notes: list[str] = []

    dipoles = _read_dipoles(charges_path, count)

    spins: list[float] | None = None

    if "spins" in found:

        spin_symbols, _, spin_values, _ = _read_xyz(found["spins"])

        if len(spin_values) == count and list(spin_symbols) == list(symbols):
            spins = spin_values
        else:
            notes.append(
                "The spin moments file does not match the charges file "
                "atom for atom, so it was ignored."
            )

    cm5 = _read_cm5(found["analysis"], count) if "analysis" in found else None

    bonds: list[Bond] = []
    sums: list[float] | None = None

    if "bonds" in found:

        bond_symbols, bond_coords, _, bond_lattice = _read_xyz(found["bonds"])

        if list(bond_symbols) != list(symbols):
            notes.append(
                "The bond orders file describes different atoms from the "
                "charges file, so no bonds were read."
            )
        else:
            bonds, sums = _read_bonds(
                found["bonds"],
                bond_coords,
                symbols,
                bond_lattice if bond_lattice is not None else lattice,
            )
    else:
        notes.append(
            "No bond orders file. Chargemol writes one when its "
            "job_control.txt asks for bond orders."
        )

    atoms = [
        Atom(
            index=position + 1,
            symbol=symbols[position],
            coords=coords[position],
            charge=charges[position],
            spin=spins[position] if spins else None,
            dipole=dipoles[position] if dipoles else None,
            sbo=sums[position] if sums else None,
            cm5=cm5[position] if cm5 else None,
        )
        for position in range(count)
    ]

    total = sum(charges)

    if abs(total) > 0.05:
        notes.append(
            f"The net charges add up to {total:+.4f} rather than zero. "
            "That is the run's own net charge if it was charged, and a "
            "sign the partition did not converge if it was not."
        )

    return DdecResult(
        atoms=atoms,
        bonds=bonds,
        lattice=lattice,
        folder=folder,
        method=method,
        notes=notes,
    )


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------


def by_element(result: DdecResult) -> list[GroupCharge]:
    """One row per species."""

    rows: list[GroupCharge] = []

    for symbol in result.elements():

        members = [atom for atom in result.atoms if atom.symbol == symbol]

        spins = [atom.spin for atom in members if atom.spin is not None]
        sums = [atom.sbo for atom in members if atom.sbo is not None]

        rows.append(
            GroupCharge(
                label=symbol,
                members=tuple(atom.index for atom in members),
                charge=sum(atom.charge for atom in members),
                spin=sum(spins) if spins else None,
                sbo=sum(sums) if sums else None,
            )
        )

    return rows


def bonds_above(result: DdecResult, minimum: float) -> list[Bond]:
    """
    The bonds worth showing, strongest first.

    Chargemol prints everything down to 0.001, which for a metal is
    hundreds of entries per atom; almost all of them are the tail of the
    exchange hole rather than anything anyone would call a bond.
    """

    return sorted(
        (bond for bond in result.bonds if bond.order >= minimum),
        key=lambda bond: -bond.order,
    )


def unique_bonds(bonds: Iterable[Bond]) -> list[Bond]:
    """
    Each bond once, rather than once from each end.

    Chargemol prints the A-B bond under A and again under B, so a plain
    listing double counts every one of them.
    """

    seen: set[tuple] = set()
    kept: list[Bond] = []

    for bond in bonds:

        # The same bond seen from the other end has the atoms swapped
        # and the image negated.
        forward = (bond.a, bond.b, bond.image)
        backward = (
            bond.b,
            bond.a,
            tuple(-value for value in bond.image),
        )

        key = min(forward, backward)

        if key in seen:
            continue

        seen.add(key)
        kept.append(bond)

    return kept


def by_bond_type(bonds: Iterable[Bond]) -> list[BondType]:
    """One row per pair of elements."""

    grouped: dict[str, list[Bond]] = {}

    for bond in bonds:
        grouped.setdefault(bond.pair, []).append(bond)

    rows: list[BondType] = []

    for pair, members in grouped.items():

        orders = np.array([bond.order for bond in members])
        lengths = np.array([bond.length for bond in members])

        rows.append(
            BondType(
                pair=pair,
                count=len(members),
                order_mean=float(orders.mean()),
                order_min=float(orders.min()),
                order_max=float(orders.max()),
                length_mean=float(lengths.mean()),
                length_min=float(lengths.min()),
                length_max=float(lengths.max()),
            )
        )

    return sorted(rows, key=lambda row: -row.order_mean)


# ------------------------------------------------------------
# Running Chargemol
# ------------------------------------------------------------


def find_executable(hint: str | None = None) -> str | None:
    """Chargemol, if it is on the path or where the user said."""

    if hint:

        path = Path(hint)

        if path.is_file():
            return str(path)

    for name in (
        "Chargemol_09_26_2017_linux_parallel",
        "Chargemol_09_26_2017_linux_serial",
        "chargemol",
        "Chargemol",
    ):
        found = shutil.which(name)

        if found:
            return found

    return None


def write_job_control(
    folder: Path | str,
    densities: Path | str,
    net_charge: float = 0.0,
    periodic: Sequence[bool] = (True, True, True),
    bond_orders: bool = True,
) -> Path:
    """
    Write the ``job_control.txt`` Chargemol reads its settings from.

    Getting this file right is most of the difficulty of running
    Chargemol by hand, so it is written here rather than left to the
    user.
    """

    folder = Path(folder)
    densities = str(densities)

    # Chargemol wants a trailing separator on this path and fails
    # obscurely without one.
    if not densities.endswith(("/", "\\")):
        densities += "\\" if "\\" in densities else "/"

    body = [
        "<net charge>",
        f"{float(net_charge):.6f}",
        "</net charge>",
        "",
        "<periodicity along A, B, and C vectors>",
        *[".true." if flag else ".false." for flag in periodic],
        "</periodicity along A, B, and C vectors>",
        "",
        "<atomic densities directory complete path>",
        densities,
        "</atomic densities directory complete path>",
        "",
        "<charge type>",
        "DDEC6",
        "</charge type>",
        "",
        "<compute BOs>",
        ".true." if bond_orders else ".false.",
        "</compute BOs>",
        "",
    ]

    path = folder / "job_control.txt"

    path.write_text("\n".join(body), encoding="utf-8")

    return path


def run(
    folder: Path | str,
    executable: str,
    densities: Path | str,
    net_charge: float = 0.0,
    periodic: Sequence[bool] = (True, True, True),
    bond_orders: bool = True,
    timeout: int = 7200,
    progress: Callable[[str, int], None] | None = None,
) -> DdecResult:
    """
    Run Chargemol in ``folder`` and read what it wrote.

    ``folder`` must already hold the ``AECCAR0``, ``AECCAR2`` and
    ``CHGCAR`` that Chargemol reads, and the POTCAR it takes the core
    electron counts from.
    """

    folder = Path(folder)

    def say(message: str, percent: int) -> None:
        if progress is not None:
            progress(message, percent)

    if not Path(executable).is_file() and shutil.which(executable) is None:
        raise DdecError(f"Chargemol is not at {executable}.")

    if not Path(densities).is_dir():
        raise DdecError(
            f"{densities} is not a folder. Chargemol needs its atomic "
            "densities directory, which ships with the program."
        )

    for name in ("CHGCAR", "AECCAR0", "AECCAR2", "POTCAR"):

        if not (folder / name).is_file():
            raise DdecError(
                f"{name} is not in {folder}. Chargemol reads all of "
                "CHGCAR, AECCAR0, AECCAR2 and POTCAR."
            )

    say("Writing job_control.txt...", 5)

    write_job_control(folder, densities, net_charge, periodic, bond_orders)

    say("Running Chargemol. This takes minutes, not seconds...", 10)

    try:
        finished = subprocess.run(
            [str(executable)],
            cwd=str(folder),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        raise DdecError(
            f"Chargemol did not finish within {timeout // 60} minutes."
        ) from exc

    except OSError as exc:
        raise DdecError(f"Chargemol would not start: {exc}") from exc

    if finished.returncode != 0 and "charges" not in available(folder):

        tail = (finished.stderr or finished.stdout or "").strip().splitlines()

        raise DdecError(
            "Chargemol failed"
            + (f": {tail[-1]}" if tail else f" with code {finished.returncode}.")
        )

    say("Reading the output...", 90)

    result = read(folder)

    say("Done.", 100)

    return result

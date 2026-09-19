"""
Elastic constants from a finished VASP run.

The machine-learning route in :mod:`qvista.analyses.elastic` strains the
cell itself and fits a tensor from the stresses it measures. This module
does no calculation at all: VASP has already done it, and what is left is
to read the answer out of ``OUTCAR`` correctly.

Nothing here launches VASP. Write the input with VASP Input >
INCAR > Elastic Constant, run it wherever the licence lives, and
point this at the folder that comes back.

What VASP writes
----------------
``IBRION = 6`` with ``ISIF >= 3`` displaces the ions and strains the cell
by finite differences, and prints three 6x6 blocks:

    SYMMETRIZED ELASTIC MODULI       the clamped-ion tensor: the response
                                     with the ions held where they are

    ELASTIC MODULI CONTR FROM        what the ions relaxing adds, always
    IONIC RELAXATION                 negative -- letting them move can
                                     only make the crystal softer

    TOTAL ELASTIC MODULI             the sum of the two, and the one that
                                     compares with experiment

The total is the physical answer. The other two are kept because their
ratio says how much of the elasticity is the ions rearranging rather than
the bonds stretching, and because their sum is a free check that the
block was read correctly -- see :meth:`VaspElastic.sum_error`.

Two traps
---------
**Units.** VASP prints kBar. Everything else in QVista is GPa, and
1 kBar = 0.1 GPa. Skipping this makes every modulus ten times too large,
which is not obviously wrong for a stiff oxide.

**Ordering.** VASP labels its columns ``XX YY ZZ XY YZ ZX``. The Voigt
convention every textbook formula assumes -- and the one
:func:`~qvista.analyses.elastic.derive_moduli` is written against -- is
``xx yy zz yz xz xy``. The last three are a *cyclic rotation* apart, so a
tensor read without reordering stays symmetric, stays positive definite,
and still passes the Born criteria. It simply reports C44 where C66
belongs. For a cubic crystal the three shear constants are equal and the
error is invisible; for anything less symmetric it silently swaps them.
:data:`VOIGT_FROM_VASP` is that permutation, applied to rows and columns
both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class ElasticVaspError(Exception):
    """Raised when a run cannot be read, with a reason a user can act on."""


# ------------------------------------------------------------
# Conventions
# ------------------------------------------------------------

#: VASP reports elastic moduli in kBar; QVista works in GPa.
KBAR_TO_GPA = 0.1

#: The column order VASP prints.
VASP_ORDER = ("XX", "YY", "ZZ", "XY", "YZ", "ZX")

#: The column order every formula assumes.
VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")

#: Where each Voigt slot is found in a VASP block. Voigt 3 (yz) is VASP's
#: YZ at index 4, Voigt 4 (xz) is VASP's ZX at index 5, and Voigt 5 (xy)
#: is VASP's XY at index 3.
VOIGT_FROM_VASP = (0, 1, 2, 4, 5, 3)

#: The blocks read out of OUTCAR, and what each one is.
BLOCKS: dict[str, tuple[str, str]] = {
    "total": (
        "TOTAL ELASTIC MODULI",
        "the full response, ions free to relax",
    ),
    "clamped": (
        "SYMMETRIZED ELASTIC MODULI",
        "the clamped-ion response, ions held fixed",
    ),
    "ionic": (
        "ELASTIC MODULI CONTR FROM IONIC RELAXATION",
        "what the ions relaxing contributes",
    ),
}

#: Tolerance on ``total == clamped + ionic``, as a fraction of the largest
#: entry. VASP prints four decimals, so anything above rounding means a
#: block was mismatched.
SUM_TOLERANCE = 1e-3

#: Below this the tensor is treated as not filled in at all.
EMPTY = 1e-8


# ------------------------------------------------------------
# Reordering
# ------------------------------------------------------------


def to_voigt(matrix: Any) -> np.ndarray:
    """
    Put a VASP-ordered 6x6 block into Voigt order.

    The permutation acts on rows *and* columns, because both index the
    same six strain components.
    """

    array = np.asarray(matrix, dtype=float)

    if array.shape != (6, 6):
        raise ElasticVaspError(
            f"An elastic block should be 6x6, but this one is "
            f"{'x'.join(str(n) for n in array.shape)}."
        )

    index = np.asarray(VOIGT_FROM_VASP)

    return array[np.ix_(index, index)]


# ------------------------------------------------------------
# Result
# ------------------------------------------------------------


@dataclass
class VaspElastic:
    """Every elastic block found in one OUTCAR, in Voigt order and GPa."""

    tensors: dict[str, np.ndarray] = field(default_factory=dict)

    source: Path | None = None

    #: Tags read back from the run, for checking it was the right kind.
    ibrion: int | None = None
    isif: int | None = None
    nfree: int | None = None
    potim: float | None = None

    natoms: int = 0
    energy: float = 0.0

    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------

    @property
    def total(self) -> np.ndarray | None:
        """The tensor to quote, or ``None`` if the run did not print one."""

        return self.tensors.get("total")

    def available(self) -> list[str]:
        """Which blocks this run actually has."""

        return [key for key in BLOCKS if key in self.tensors]

    @property
    def sum_error(self) -> float:
        """
        How far ``clamped + ionic`` is from ``total``, relative.

        VASP computes the total as that sum, so a nonzero answer here
        means the blocks were mismatched while parsing rather than that
        anything is wrong with the physics. ``nan`` when a block is
        missing and the check cannot be made.
        """

        if not {"total", "clamped", "ionic"} <= set(self.tensors):
            return float("nan")

        difference = (
            self.tensors["total"]
            - self.tensors["clamped"]
            - self.tensors["ionic"]
        )

        scale = float(np.abs(self.tensors["total"]).max())

        if scale < EMPTY:
            return float("nan")

        return float(np.abs(difference).max() / scale)

    @property
    def ionic_fraction(self) -> float:
        """
        How much of the bulk modulus comes from the ions relaxing.

        Large values mean the clamped-ion tensor is a poor stand-in, which
        is worth knowing before anyone quotes one.
        """

        if not {"total", "clamped"} <= set(self.tensors):
            return float("nan")

        total = _bulk_estimate(self.tensors["total"])
        clamped = _bulk_estimate(self.tensors["clamped"])

        if abs(clamped) < EMPTY:
            return float("nan")

        return float((clamped - total) / clamped)


def _bulk_estimate(tensor: np.ndarray) -> float:
    """The Voigt bulk modulus, used only for comparing two tensors."""

    c = np.asarray(tensor, dtype=float)

    return float(
        (
            (c[0, 0] + c[1, 1] + c[2, 2])
            + 2.0 * (c[0, 1] + c[0, 2] + c[1, 2])
        )
        / 9.0
    )


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------

#: A row of a moduli block: a direction label then six numbers.
_ROW = re.compile(
    r"^\s*(XX|YY|ZZ|XY|YZ|ZX)\s+((?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+){5}"
    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)

_TAG = re.compile(r"\b(IBRION|ISIF|NFREE)\s*=\s*(-?\d+)")
_POTIM = re.compile(r"\bPOTIM\s*=\s*([-+]?\d*\.?\d+)")


def find_outcar(folder: Path | str) -> Path:
    """
    Locate the OUTCAR for a run.

    Accepts the file itself or the directory holding it, because a user
    who has just run VASP thinks in terms of the folder.
    """

    path = Path(folder)

    if path.is_file():
        return path

    if not path.is_dir():
        raise ElasticVaspError(f"There is no such folder: {path}")

    outcar = path / "OUTCAR"

    if outcar.is_file():
        return outcar

    raise ElasticVaspError(
        f"No OUTCAR in {path.name}.\n\nThis reads a finished VASP elastic "
        f"run. Write the input with VASP Input > INCAR > "
        f"Elastic Constant, run it, and point this at the folder it "
        f"produced."
    )


def read_outcar(folder: Path | str) -> VaspElastic:
    """
    Read every elastic block out of a VASP run.

    Later blocks win: VASP prints the moduli once per ionic step in some
    setups, and the last one is the converged geometry.
    """

    path = find_outcar(folder)

    try:
        text = path.read_text(errors="replace")

    except OSError as exc:
        raise ElasticVaspError(f"Could not read {path.name}: {exc}") from exc

    result = VaspElastic(source=path)

    _read_tags(text, result)

    for key, (heading, _) in BLOCKS.items():

        block = _read_block(text, heading)

        if block is not None:
            result.tensors[key] = to_voigt(block) * KBAR_TO_GPA

    if not result.tensors:
        raise ElasticVaspError(
            f"{path.name} has no elastic moduli in it.\n\nThe run needs "
            f"IBRION = 6 with ISIF >= 3 for VASP to compute them; this "
            f"one had "
            + (
                f"IBRION = {result.ibrion}."
                if result.ibrion is not None
                else "no IBRION recorded."
            )
        )

    result.notes = _notes(result)

    return result


def _read_block(text: str, heading: str) -> np.ndarray | None:
    """The last 6x6 block printed under ``heading``, or ``None``."""

    found: np.ndarray | None = None

    for match in re.finditer(re.escape(heading), text):

        rows: dict[str, list[float]] = {}

        # The six data rows follow within a couple of lines of the
        # heading; stop at the first line that is not one of them once
        # collecting has begun.
        for line in text[match.end():].splitlines()[:12]:

            row = _ROW.match(line)

            if row is None:
                if rows:
                    break
                continue

            rows[row.group(1)] = [
                float(value) for value in row.group(2).split()
            ]

        if len(rows) == 6:
            found = np.asarray(
                [rows[name] for name in VASP_ORDER], dtype=float
            )

    return found


def _read_tags(text: str, result: VaspElastic) -> None:
    """Pull back the tags that say what kind of run this was."""

    for name, value in _TAG.findall(text):

        if getattr(result, name.lower()) is None:
            setattr(result, name.lower(), int(value))

    potim = _POTIM.search(text)

    if potim is not None:
        result.potim = float(potim.group(1))

    atoms = re.search(r"NIONS\s*=\s*(\d+)", text)

    if atoms is not None:
        result.natoms = int(atoms.group(1))

    energies = re.findall(
        r"free  energy   TOTEN\s*=\s*([-+]?\d*\.?\d+)", text
    )

    if energies:
        result.energy = float(energies[-1])


def _notes(result: VaspElastic) -> list[str]:
    """Everything worth saying about how the run was set up."""

    notes: list[str] = []

    # ----------------------------------------------------------
    # Was this the right kind of run?
    # ----------------------------------------------------------

    if result.ibrion is not None and result.ibrion != 6:
        notes.append(
            f"WARNING: this run used IBRION = {result.ibrion}, not 6. The "
            f"moduli below were still printed, but the finite-difference "
            f"elastic calculation is IBRION = 6."
        )

    if result.isif is not None and result.isif < 3:
        notes.append(
            f"WARNING: ISIF = {result.isif} does not vary the cell, so the "
            f"elastic tensor cannot be complete. Elastic constants need "
            f"ISIF >= 3."
        )

    if "total" not in result.tensors:
        notes.append(
            "WARNING: this run printed no TOTAL ELASTIC MODULI, so the "
            "ions were never allowed to relax against the strain. The "
            "clamped-ion tensor overestimates every modulus."
        )

    # ----------------------------------------------------------
    # Did the blocks read back consistently?
    # ----------------------------------------------------------

    error = result.sum_error

    if np.isfinite(error) and error > SUM_TOLERANCE:
        notes.append(
            f"WARNING: the total tensor is not the sum of the clamped-ion "
            f"and ionic blocks (off by {error:.1%}). The blocks may have "
            f"been read from different ionic steps."
        )

    fraction = result.ionic_fraction

    if np.isfinite(fraction) and fraction > 0.25:
        notes.append(
            f"Ionic relaxation accounts for {fraction:.0%} of the bulk "
            f"modulus, so the clamped-ion numbers are far from the "
            f"physical ones for this structure."
        )

    notes.append(
        "Read from a VASP run, not computed here. The numbers are as good "
        "as that run's convergence: check ENCUT and the k-mesh were "
        "converged for stress, which needs more of both than energy does."
    )

    return notes


# ------------------------------------------------------------
# Into the shared result type
# ------------------------------------------------------------


def to_result(
    elastic: VaspElastic,
    which: str = "total",
    structure=None,
    symprec: float | None = None,
):
    """
    Derive moduli and stability, packaged like the force-field route.

    Returns a :class:`~qvista.analyses.elastic.ElasticResult`, so the same
    panel, the same tables and the same CSV export serve both routes and
    neither can drift away from the other.

    ``structure`` is optional and only used to name the space group; the
    tensor itself comes entirely from the OUTCAR.
    """

    from qvista.analyses.base import AnalysisError
    from qvista.analyses.elastic import (
        ElasticResult,
        born_stability,
        derive_moduli,
    )
    from qvista.services.structure_analyzer import DEFAULT_SYMPREC

    if which not in BLOCKS:
        raise ElasticVaspError(f"Unknown block: {which}")

    tensor = elastic.tensors.get(which)

    if tensor is None:
        raise ElasticVaspError(
            f"This run has no {BLOCKS[which][0]} block. It has: "
            + ", ".join(BLOCKS[key][0] for key in elastic.available())
        )

    try:
        moduli = derive_moduli(tensor)

    except AnalysisError as exc:
        raise ElasticVaspError(str(exc)) from exc

    stability = born_stability(tensor)

    if structure is None:
        structure = read_structure(elastic)

    before = after = ("", 0)

    if structure is not None:
        before = after = _spacegroup(
            structure, symprec or DEFAULT_SYMPREC
        )

    result = ElasticResult(
        tensor=tensor,
        moduli=moduli,
        stability=stability,
        spacegroup_before=before[0],
        spacegroup_number_before=before[1],
        spacegroup_after=after[0],
        spacegroup_number_after=after[1],
        # VASP works in the orientation of the POSCAR it was given, and
        # nothing was refined here, so the caveat the force-field route
        # raises about arbitrary axes does not apply the same way.
        orientation_is_arbitrary=False,
        energy=elastic.energy,
        symprec=symprec or DEFAULT_SYMPREC,
        force_field=describe(elastic, which),
    )

    result.rows = _rows(elastic, result, which)
    result.notes = _result_notes(elastic, result)

    return result


def read_structure(elastic: VaspElastic):
    """
    The structure the run was done on, from the folder holding OUTCAR.

    Only used to name the space group. CONTCAR first, because that is
    the geometry the moduli actually describe; a run with ISIF = 3 moved
    away from its POSCAR before it measured anything.
    """

    if elastic.source is None:
        return None

    folder = elastic.source.parent

    from pymatgen.core import Structure

    for name in ("CONTCAR", "POSCAR"):

        path = folder / name

        if not path.is_file() or path.stat().st_size == 0:
            continue

        try:
            return Structure.from_file(str(path))

        except Exception:  # pragma: no cover - depends on the file
            continue

    return None


def describe(elastic: VaspElastic, which: str = "total") -> str:
    """One line naming what produced these numbers."""

    parts = ["VASP"]

    if elastic.ibrion is not None:
        parts.append(f"IBRION = {elastic.ibrion}")

    if elastic.nfree is not None:
        parts.append(f"NFREE = {elastic.nfree}")

    return f"{', '.join(parts)} ({BLOCKS[which][1]})"


def _spacegroup(structure, symprec: float) -> tuple[str, int]:
    """Name the space group, or say nothing if it cannot be determined."""

    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)

        return (
            str(analyzer.get_space_group_symbol()),
            int(analyzer.get_space_group_number()),
        )

    except Exception:  # pragma: no cover - depends on the structure
        return ("unknown", 0)


def _rows(
    elastic: VaspElastic, result, which: str
) -> list[tuple[str, str]]:
    """Console rows."""

    rows = [
        ("Source", str(elastic.source)),
        ("Forces from", result.force_field),
        ("Atoms", str(elastic.natoms) if elastic.natoms else "?"),
    ]

    if elastic.energy:
        rows.append(("Final energy", f"{elastic.energy:.6f} eV"))

    rows.append(
        ("Blocks read", ", ".join(elastic.available()))
    )

    fraction = elastic.ionic_fraction

    if np.isfinite(fraction):
        rows.append(
            ("Ionic contribution to K", f"{fraction:.1%}")
        )

    if result.spacegroup_after:
        rows.append(
            (
                "Space group",
                f"{result.spacegroup_after} "
                f"(#{result.spacegroup_number_after})",
            )
        )

    if result.moduli is not None:
        rows.extend(result.moduli.as_rows())

    if result.stability is not None:
        rows.extend(result.stability.as_rows())

    return rows


def _result_notes(elastic: VaspElastic, result) -> list[str]:
    """The tensor itself, then the warnings collected while reading."""

    notes: list[str] = []

    if result.tensor is not None:

        lines = ["Elastic tensor C (GPa), Voigt order xx yy zz yz xz xy"]

        for row in result.tensor:
            lines.append(
                "  " + " ".join(f"{float(value):9.2f}" for value in row)
            )

        notes.append("\n".join(lines))

    notes.extend(elastic.notes)

    if result.stability is not None:
        for failure in result.stability.failures:
            notes.append(f"WARNING: {failure}")

    return notes

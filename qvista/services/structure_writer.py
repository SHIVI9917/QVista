"""
Write a structure out in any supported format.

Input format and output format are independent: a CIF can be saved as a
POSCAR, a POSCAR as XYZ, and so on. Structures are obtained through
:class:`~qvista.services.structure_analyzer.StructureAnalyzer`, so the same
parsing rules apply as everywhere else -- in particular an extended XYZ
keeps its lattice rather than silently becoming a molecule.

Periodic output formats need a lattice. A plain XYZ has none, so converting
one to POSCAR or CIF is refused with an explanation rather than writing a
file with a made-up cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Extended XYZ, so a lattice survives the round trip.
XYZ_FORMAT = "extxyz"


class StructureWriteError(Exception):
    """Raised when a structure cannot be written in the requested format."""


@dataclass(frozen=True)
class OutputFormat:
    """One format a structure can be saved as."""

    key: str
    label: str
    default_name: str
    file_filter: str

    #: False for formats that cannot represent a molecule.
    supports_molecules: bool

    #: True for formats that require a lattice.
    requires_lattice: bool


OUTPUT_FORMATS: tuple[OutputFormat, ...] = (
    OutputFormat(
        key="poscar",
        label="VASP POSCAR",
        default_name="POSCAR",
        file_filter="VASP POSCAR (POSCAR *.vasp);;All Files (*)",
        supports_molecules=False,
        requires_lattice=True,
    ),
    OutputFormat(
        key="cif",
        label="CIF",
        default_name="structure.cif",
        file_filter="CIF File (*.cif);;All Files (*)",
        supports_molecules=False,
        requires_lattice=True,
    ),
    OutputFormat(
        key="xyz",
        label="XYZ",
        default_name="structure.xyz",
        file_filter="XYZ File (*.xyz);;All Files (*)",
        supports_molecules=True,
        requires_lattice=False,
    ),
)


#: Names that identify a VASP structure file without an extension.
VASP_NAMES = {"POSCAR", "CONTCAR"}


def ensure_readable_name(target: Path, fmt: OutputFormat) -> Path:
    """
    Give ``target`` an extension QVista can detect when reopening it.

    Format detection works from the filename, so a CIF saved as ``mystruct``
    is written correctly but cannot be read back. Rather than let the user
    create a file the application itself cannot open, the canonical suffix
    is appended when it is missing.

    POSCAR is the exception: the convention is a bare ``POSCAR`` or
    ``CONTCAR``, which is already detectable, so those are left alone and
    anything else gains ``.vasp``.
    """

    suffix = target.suffix.lower()

    if fmt.key == "cif":
        return target if suffix == ".cif" else target.with_suffix(".cif")

    if fmt.key == "xyz":
        return target if suffix == ".xyz" else target.with_suffix(".xyz")

    if fmt.key == "poscar":

        if target.name.upper() in VASP_NAMES:
            return target

        if suffix in {".vasp", ".poscar", ".contcar"}:
            return target

        return target.with_suffix(".vasp")

    return target


def formats_for(periodic: bool) -> list[OutputFormat]:
    """Formats a structure can actually be written as."""

    if periodic:
        return list(OUTPUT_FORMATS)

    return [fmt for fmt in OUTPUT_FORMATS if fmt.supports_molecules]


def find_format(key: str) -> OutputFormat:
    """Look up an output format by key."""

    for fmt in OUTPUT_FORMATS:
        if fmt.key == key:
            return fmt

    raise StructureWriteError(f"Unknown output format: {key}")


def convert(
    source: str | Path,
    target: str | Path,
    fmt: str,
) -> Path:
    """
    Read ``source`` in whatever format it is, write ``target`` as ``fmt``.

    Returns the path written.

    Raises
    ------
    StructureWriteError
        If the structure cannot be represented in the requested format, or
        the file could not be written.
    """

    from qvista.services.structure_analyzer import (
        StructureAnalysisError,
        StructureAnalyzer,
    )

    output = find_format(fmt)

    try:
        parsed = StructureAnalyzer.parse(source)

    except StructureAnalysisError as exc:
        raise StructureWriteError(str(exc)) from exc

    if output.requires_lattice and not parsed.periodic:
        raise StructureWriteError(
            f"{Path(source).name} was read as a molecule, with no lattice.\n\n"
            f"{output.label} describes a periodic crystal and cannot be "
            f"written without a unit cell. Save it as XYZ instead, or "
            f"supply an input file that carries a lattice."
        )

    return write(parsed.structure, target, fmt, periodic=parsed.periodic)


def write(
    structure: Any,
    target: str | Path,
    fmt: str,
    periodic: bool = True,
) -> Path:
    """
    Write an already-parsed pymatgen object to ``target``.

    ``structure`` is a pymatgen ``Structure`` when periodic, or a
    ``Molecule`` otherwise.
    """

    output = find_format(fmt)

    target = ensure_readable_name(Path(target), output)

    if output.requires_lattice and not periodic:
        raise StructureWriteError(
            f"{output.label} cannot represent a structure without a lattice."
        )

    try:
        if output.key == "xyz":
            _write_xyz(structure, target, periodic)

        elif output.key == "poscar":
            structure.to(filename=str(target), fmt="poscar")

        elif output.key == "cif":
            structure.to(filename=str(target), fmt="cif")

        else:  # pragma: no cover - guarded by find_format
            raise StructureWriteError(
                f"Unsupported output format: {output.key}"
            )

    except StructureWriteError:
        raise

    except Exception as exc:
        raise StructureWriteError(
            f"Could not write {target.name}: {type(exc).__name__}: {exc}"
        ) from exc

    return target


def _write_xyz(structure: Any, target: Path, periodic: bool) -> None:
    """
    Write XYZ, keeping the lattice when there is one.

    Extended XYZ is used for periodic structures so the cell is preserved
    and QVista can read the file back as a crystal rather than a molecule.
    """

    from ase.io import write as ase_write
    from pymatgen.io.ase import AseAtomsAdaptor

    if periodic:
        atoms = AseAtomsAdaptor.get_atoms(structure)

        ase_write(str(target), atoms, format=XYZ_FORMAT)
        return

    # A molecule has no cell; a plain XYZ is the correct representation.
    lines = [str(len(structure)), target.stem]

    for site in structure:
        x, y, z = site.coords

        lines.append(
            f"{site.specie.symbol:<3} "
            f"{x:14.8f} {y:14.8f} {z:14.8f}"
        )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

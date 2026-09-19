"""
Extract structural properties from a structure file.

Handles CIF, POSCAR, CONTCAR and XYZ without the caller stating the format.

pymatgen is imported lazily inside the functions below -- it is a heavy
import and the application should not pay for it at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qvista.models.structure_info import StructureInfo

#: Symmetry tolerance in Angstrom. Relaxed structures may need a looser
#: value; pass ``symprec`` to :meth:`StructureAnalyzer.analyze` to override.
DEFAULT_SYMPREC = 0.01

#: Filenames that are VASP structure files despite having no extension.
VASP_NAMES = {"POSCAR", "CONTCAR"}


class StructureAnalysisError(Exception):
    """Raised when a structure file cannot be read or understood."""


@dataclass
class ParsedStructure:
    """
    The outcome of parsing a structure file.

    ``structure`` is a pymatgen ``Structure`` when the file carries a
    lattice, and a pymatgen ``Molecule`` when it does not.
    """

    structure: Any
    periodic: bool
    source_format: str


class StructureAnalyzer:
    """Reads structural properties from a file."""

    # ------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------

    @staticmethod
    def analyze(
        path: str | Path,
        symprec: float = DEFAULT_SYMPREC,
    ) -> StructureInfo:
        """
        Extract the six structural properties from ``path``.

        Raises
        ------
        StructureAnalysisError
            If the file is missing or cannot be parsed. The message is
            written for a user, not a developer.
        """

        parsed = StructureAnalyzer.parse(path)

        if not parsed.periodic:
            # A molecule has no lattice, so only composition is defined.
            return StructureInfo(
                formula=parsed.structure.composition.reduced_formula,
                natoms=len(parsed.structure),
                periodic=False,
                source_format=parsed.source_format,
            )

        return StructureAnalyzer._from_structure(
            parsed.structure,
            parsed.source_format,
            symprec,
        )

    # ------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------

    @staticmethod
    def parse(path: str | Path) -> ParsedStructure:
        """
        Parse ``path`` into a pymatgen object.

        This is the only supported way for the rest of the application to
        obtain a structure. Calling ``Structure.from_file`` directly is
        wrong: it rejects ``.xyz`` outright, and the obvious
        ``Molecule.from_file`` fallback silently discards the lattice of an
        extended XYZ that actually has one.
        """

        path = Path(path)

        if not path.exists():
            raise StructureAnalysisError(
                f"File not found: {path}"
            )

        fmt = StructureAnalyzer.detect_format(path)

        try:
            if fmt == "XYZ":
                return StructureAnalyzer._parse_xyz(path)

            from pymatgen.core import Structure

            return ParsedStructure(
                structure=Structure.from_file(str(path)),
                periodic=True,
                source_format=fmt,
            )

        except StructureAnalysisError:
            raise

        except Exception as exc:
            raise StructureAnalysisError(
                f"Could not parse {path.name} as {fmt}: {exc}"
            ) from exc

    # ------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------

    @staticmethod
    def detect_format(path: Path) -> str:
        """Identify the file format from its name and extension."""

        suffix = path.suffix.lower()

        if suffix == ".cif":
            return "CIF"

        if suffix == ".xyz":
            return "XYZ"

        stem = path.stem.upper()

        if path.name.upper() in VASP_NAMES or stem in VASP_NAMES:
            return stem if stem in VASP_NAMES else path.name.upper()

        if suffix in {".vasp", ".poscar", ".contcar"}:
            return "POSCAR"

        # Unknown extension: let pymatgen's own detection decide.
        return "structure file"

    # ------------------------------------------------------------
    # XYZ
    # ------------------------------------------------------------

    @staticmethod
    def _parse_xyz(path: Path) -> ParsedStructure:
        """
        Parse an XYZ file.

        Extended (ASE-style) XYZ carries a lattice in its comment line; when
        one is present the file describes a periodic structure. A plain XYZ
        has no cell and is therefore a molecule.
        """

        structure = StructureAnalyzer._try_extended_xyz(path)

        if structure is not None:
            return ParsedStructure(
                structure=structure,
                periodic=True,
                source_format="extended XYZ",
            )

        # Plain XYZ -- a molecule. Only composition is defined.
        from pymatgen.core import Molecule

        return ParsedStructure(
            structure=Molecule.from_file(str(path)),
            periodic=False,
            source_format="XYZ",
        )

    @staticmethod
    def _try_extended_xyz(path: Path):
        """
        Return a pymatgen ``Structure`` if the XYZ has lattice information.

        Returns ``None`` when the file has no usable cell, so the caller can
        fall back to a plain molecule parse.
        """

        try:
            from ase.io import read
            from pymatgen.io.ase import AseAtomsAdaptor

            # The format is stated explicitly rather than letting ASE guess
            # from the filename. ASE's detection matches "poscar" and
            # "contcar" anywhere in the basename, so a file legitimately
            # called something like "my_poscar_conversion.xyz" is read with
            # the VASP parser, which fails -- and the failure used to be
            # swallowed below, silently turning a periodic structure into a
            # molecule and dropping its lattice.
            #
            # extxyz is a superset of plain XYZ, so this reads both.
            atoms = read(str(path), format="extxyz")

            cell = atoms.get_cell()

            # A plain XYZ yields an all-zero cell with zero volume.
            if cell.rank < 3 or cell.volume <= 0:
                return None

            return AseAtomsAdaptor.get_structure(atoms)

        except Exception:
            # Not extended XYZ, or ASE could not read it -- fall back.
            return None

    # ------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------

    @staticmethod
    def _from_structure(
        structure,
        fmt: str,
        symprec: float,
    ) -> StructureInfo:
        """Build a :class:`StructureInfo` from a pymatgen ``Structure``."""

        info = StructureInfo(
            formula=structure.composition.reduced_formula,
            natoms=len(structure),
            volume=structure.volume,
            periodic=True,
            source_format=fmt,
        )

        symbol, number, system = StructureAnalyzer._symmetry(
            structure,
            symprec,
        )

        info.spacegroup_symbol = symbol
        info.spacegroup_number = number
        info.crystal_system = system

        return info

    @staticmethod
    def _symmetry(
        structure,
        symprec: float,
    ) -> tuple[str | None, int | None, str | None]:
        """
        Determine space group and crystal system.

        Symmetry detection can fail on a badly formed or heavily distorted
        cell. That is not fatal -- the remaining properties are still
        useful -- so this returns ``None`` rather than raising.
        """

        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            analyzer = SpacegroupAnalyzer(structure, symprec=symprec)

            return (
                analyzer.get_space_group_symbol(),
                analyzer.get_space_group_number(),
                analyzer.get_crystal_system().capitalize(),
            )

        except Exception:
            return None, None, None

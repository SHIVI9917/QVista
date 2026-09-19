"""
Structural properties extracted from an input file.
"""

from __future__ import annotations

from dataclasses import dataclass

NOT_AVAILABLE = "N/A"


@dataclass
class StructureInfo:
    """
    Six structural properties read from a structure file.

    Fields that cannot be determined -- typically the periodic ones for a
    plain XYZ file, which carries no lattice -- are ``None`` and render as
    ``N/A``.
    """

    formula: str
    natoms: int

    spacegroup_symbol: str | None = None
    spacegroup_number: int | None = None
    crystal_system: str | None = None
    volume: float | None = None

    #: False for a molecule (no lattice), True for a periodic structure.
    periodic: bool = True

    #: Source format, for display ("CIF", "POSCAR", "XYZ", ...).
    source_format: str = ""

    # ------------------------------------------------------------
    # Display
    # ------------------------------------------------------------

    def as_rows(self) -> list[tuple[str, str]]:
        """
        Return the properties as ``(label, value)`` pairs, ready to print.

        Undefined values are rendered as ``N/A``.
        """

        if self.volume is None:
            volume = NOT_AVAILABLE
        else:
            volume = f"{self.volume:.4f} Å³"

        return [
            ("Formula", self.formula or NOT_AVAILABLE),
            ("Number of atoms", str(self.natoms)),
            ("Space group", self.spacegroup_symbol or NOT_AVAILABLE),
            (
                "Space group number",
                NOT_AVAILABLE
                if self.spacegroup_number is None
                else str(self.spacegroup_number),
            ),
            ("Crystal system", self.crystal_system or NOT_AVAILABLE),
            ("Volume", volume),
        ]

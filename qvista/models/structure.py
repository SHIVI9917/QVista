"""
Structure model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ase import Atoms


@dataclass(slots=True)
class Structure:
    """Represents an atomic structure."""

    atoms: Atoms

    @property
    def formula(self) -> str:
        return self.atoms.get_chemical_formula()

    @property
    def natoms(self) -> int:
        return len(self.atoms)

    @property
    def cell(self):
        return self.atoms.cell

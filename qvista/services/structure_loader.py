"""
Structure loader.
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read

from qvista.models.structure import Structure


class StructureLoader:
    """Loads structures from VASP files."""

    @staticmethod
    def load(poscar: str | Path) -> Structure:
        """Load a structure from a POSCAR/CONTCAR file."""

        atoms = read(poscar)

        return Structure(atoms)

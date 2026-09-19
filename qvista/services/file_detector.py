"""
Detect VASP files inside a project.
"""

from __future__ import annotations

from pathlib import Path

REQUIRED_FILES = [
    "POSCAR",
    "CONTCAR",
    "OUTCAR",
    "CHGCAR",
    "AECCAR0",
    "AECCAR2",
    "vasprun.xml",
    "DOSCAR",
    "EIGENVAL",
]


class FileDetector:
    """Detect VASP files."""

    @staticmethod
    def detect(directory: Path) -> tuple[dict[str, Path], list[str]]:
        found = {}
        missing = []

        for filename in REQUIRED_FILES:

            path = directory / filename

            if path.exists():
                found[filename] = path
            else:
                missing.append(filename)

        return found, missing

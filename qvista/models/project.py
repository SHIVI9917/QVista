"""
Project model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qvista.models.project_info import ProjectInfo
from qvista.models.structure import Structure
from qvista.models.structure_info import StructureInfo
from qvista.models.results import Results


@dataclass(slots=True)
class Project:
    """Represents a VASP project."""

    info: ProjectInfo

    structure: Structure | None = None

    #: Structural properties of the loaded structure.
    structure_info: StructureInfo | None = None

    #: Why the structure could not be analysed, if it could not be.
    structure_error: str | None = None

    results: Results = field(default_factory=Results)

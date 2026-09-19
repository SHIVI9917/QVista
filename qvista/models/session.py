from dataclasses import dataclass

from qvista.models.project import Project


@dataclass(slots=True)
class Session:
    """Current QVista session."""

    current_project: Project | None = None

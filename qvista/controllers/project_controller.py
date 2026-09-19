"""
Project controller.
"""

from __future__ import annotations

from pathlib import Path

from qvista.models.project import Project
from qvista.models.session import Session
from qvista.services.project_service import ProjectService


class ProjectController:
    """Controller responsible for managing projects."""

    def __init__(self) -> None:
        self.session = Session()

    def open_project(self, path: str | Path) -> Project:
        """
        Load a project and make it the active project.
        """

        project = ProjectService.load(path)

        self.session.current_project = project

        return project

    def has_project(self) -> bool:
        """Return True if a project is currently open."""

        return self.session.current_project is not None

    def current_project(self) -> Project | None:
        """Return the active project."""

        return self.session.current_project

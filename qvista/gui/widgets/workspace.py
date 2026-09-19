"""
Central workspace widget.
"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from qvista.models.structure import Structure
from qvista.visualization.viewer import CrystalViewer


class Workspace(QWidget):
    """Central workspace."""

    def __init__(self) -> None:
        super().__init__()

        layout = QVBoxLayout(self)

        # Remove margins so the viewer fills the workspace
        layout.setContentsMargins(0, 0, 0, 0)

        # Embedded crystal viewer
        self.viewer = CrystalViewer(self)

        layout.addWidget(self.viewer)

    def load_structure(self, structure: Structure) -> None:
        """
        Display a structure in the crystal viewer.
        """

        self.viewer.load_structure(structure)

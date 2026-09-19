"""
Properties panel.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from qvista.models.project import Project


class PropertiesPanel(QWidget):
    """Displays project properties."""

    def __init__(self) -> None:
        super().__init__()

        self.layout = QVBoxLayout(self)

        self.title = QLabel("Properties")
        self.layout.addWidget(self.title)

        self.text = QTextEdit()
        self.text.setReadOnly(True)

        self.layout.addWidget(self.text)

        self.show_empty()

    # ---------------------------------------------------------
    # Empty state
    # ---------------------------------------------------------

    def show_empty(self) -> None:
        """Display empty panel."""

        self.text.setPlainText(
            "No project loaded."
        )

    # ---------------------------------------------------------
    # Load project
    # ---------------------------------------------------------

    def load_project(self, project: Project) -> None:
        """Display project information."""

        info = project.info

        text = []

        text.append("Project")
        text.append("------------------------------")
        text.append(f"Name      : {info.name}")
        text.append(f"Directory : {info.directory}")
        text.append("")

        text.append("Files")
        text.append("------------------------------")
        text.append(f"Detected : {len(info.files)}")
        text.append(f"Missing  : {len(info.missing)}")
        text.append("")

        if project.structure is not None:

            atoms = project.structure.atoms

            formula = atoms.get_chemical_formula()

            symbols = sorted(set(atoms.get_chemical_symbols()))

            text.append("Structure")
            text.append("------------------------------")
            text.append(f"Formula  : {formula}")
            text.append(f"Atoms    : {len(atoms)}")
            text.append(f"Elements : {' '.join(symbols)}")
            text.append("")

            cell = atoms.cell.lengths()

            text.append("Cell (Å)")
            text.append("------------------------------")
            text.append(f"a = {cell[0]:8.3f}")
            text.append(f"b = {cell[1]:8.3f}")
            text.append(f"c = {cell[2]:8.3f}")

        self.text.setPlainText("\n".join(text))

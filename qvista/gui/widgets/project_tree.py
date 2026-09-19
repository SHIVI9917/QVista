"""
Project Explorer widget.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from qvista.models.project import Project


class ProjectTree(QTreeWidget):
    """Project Explorer panel."""

    def __init__(self) -> None:
        super().__init__()

        self.setHeaderHidden(True)

        self.show_empty()

    # ---------------------------------------------------------
    # Empty State
    # ---------------------------------------------------------

    def show_empty(self) -> None:
        """Display an empty project tree."""

        self.clear()

        root = QTreeWidgetItem(["No Project"])

        QTreeWidgetItem(root, ["Structures"])
        QTreeWidgetItem(root, ["Analyses"])
        QTreeWidgetItem(root, ["Results"])

        root.setExpanded(True)

        self.addTopLevelItem(root)

    # ---------------------------------------------------------
    # Load Project
    # ---------------------------------------------------------

    def load_project(self, project: Project) -> None:
        """Populate the Project Explorer."""

        self.clear()

        root = QTreeWidgetItem([project.info.name])

        # -----------------------------------------------------
        # Input Files
        # -----------------------------------------------------

        input_item = QTreeWidgetItem(root, ["Input"])

        input_files = [
            "POSCAR",
            "INCAR",
            "KPOINTS",
            "POTCAR",
        ]

        # -----------------------------------------------------
        # Output Files
        # -----------------------------------------------------

        output_item = QTreeWidgetItem(root, ["Output"])

        output_files = [
            "CONTCAR",
            "OUTCAR",
            "CHGCAR",
            "AECCAR0",
            "AECCAR2",
            "DOSCAR",
            "EIGENVAL",
            "vasprun.xml",
        ]

        # -----------------------------------------------------
        # Missing Files
        # -----------------------------------------------------

        missing_item = QTreeWidgetItem(root, ["Missing"])

        # Input files

        for filename in input_files:
            if filename in project.info.files:
                QTreeWidgetItem(input_item, [filename])

        # Output files

        for filename in output_files:
            if filename in project.info.files:
                QTreeWidgetItem(output_item, [filename])

        # Missing files

        for filename in sorted(project.info.missing):
            QTreeWidgetItem(missing_item, [filename])

        root.setExpanded(True)
        input_item.setExpanded(True)
        output_item.setExpanded(True)
        missing_item.setExpanded(True)

        self.addTopLevelItem(root)

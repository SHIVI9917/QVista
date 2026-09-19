"""
Project service.
"""

from __future__ import annotations

from pathlib import Path

from qvista.models.project import Project
from qvista.models.project_info import ProjectInfo
from qvista.services.file_detector import FileDetector
from qvista.services.structure_analyzer import (
    DEFAULT_SYMPREC,
    StructureAnalysisError,
    StructureAnalyzer,
)
from qvista.services.structure_loader import StructureLoader

#: Extensions that identify a file the user can open directly as a structure.
STRUCTURE_SUFFIXES = {".cif", ".xyz", ".vasp", ".poscar", ".contcar"}

#: Extension-less filenames that are structure files.
STRUCTURE_NAMES = {"POSCAR", "CONTCAR"}


def is_structure_file(path: Path) -> bool:
    """Return True if ``path`` looks like a structure file we can read."""

    return (
        path.suffix.lower() in STRUCTURE_SUFFIXES
        or path.name.upper() in STRUCTURE_NAMES
    )


class ProjectService:
    """Service for loading VASP projects."""

    @staticmethod
    def load(
        path: str | Path,
        symprec: float = DEFAULT_SYMPREC,
    ) -> Project:
        """
        Load a VASP project from either a directory or a project file.

        Parameters
        ----------
        path
            Project directory or a file inside the project
            (POSCAR, CONTCAR, vasprun.xml, CIF, XYZ, ...).
        symprec
            Symmetry tolerance used for space group detection. Relaxed
            structures may need a looser value than the default.

        Notes
        -----
        This never raises. Everything that can go wrong with the structure
        is recorded on ``Project.structure_error`` as a sentence the user
        can read, because the caller is a window that has to stay open and
        say something rather than a script that can die. A project whose
        structure could not be loaded therefore comes back with
        ``structure`` unset and ``structure_error`` set -- never with both
        unset, which would leave nothing to report.
        """

        path = Path(path)

        # If the user selected a file, use its parent directory.
        if path.is_file():
            directory = path.parent
            source_file = path
        else:
            directory = path
            source_file = None

        files, missing = FileDetector.detect(directory)

        info = ProjectInfo(
            name=directory.name,
            directory=directory,
            files=files,
            missing=missing,
            source_file=source_file,
        )

        project = Project(info=info)

        # ----------------------------------------------------------
        # A path that is not there at all.
        #
        # ``is_file()`` above is False for it, so it arrives here having
        # been taken for a project directory: FileDetector finds nothing
        # in a directory that does not exist, no structure is selected,
        # and the project is returned with ``structure`` and
        # ``structure_error`` both unset. Nothing fails loudly, so a
        # mistyped path is only noticed later, from whichever analysis
        # is tried next, as "No structure is loaded. Open a structure
        # file first." -- which sends the user to the File menu instead
        # of to the name they got wrong.
        #
        # Reported here in the wording StructureAnalyzer.parse uses for
        # the same condition, so a missing file reads the same whether
        # it was missing or unreadable.
        # ----------------------------------------------------------

        if not path.exists():
            project.structure_error = f"File not found: {path}"
            return project

        # ----------------------------------------------------------
        # Pick the structure to load.
        #
        # A file the user chose explicitly wins over a POSCAR that
        # happens to sit in the same directory.
        # ----------------------------------------------------------

        if source_file is not None and is_structure_file(source_file):
            structure_path = source_file

        elif "POSCAR" in files:
            structure_path = files["POSCAR"]

        else:
            structure_path = None

        if structure_path is None:
            return project

        # ----------------------------------------------------------
        # Structural properties.
        #
        # Analysed before loading because the analyser reports parse
        # failures in plain language, while the loader tends to surface
        # bare exception types.
        # ----------------------------------------------------------

        try:
            project.structure_info = StructureAnalyzer.analyze(
                structure_path,
                symprec=symprec,
            )

        except StructureAnalysisError as exc:
            project.structure_error = str(exc)

        # ----------------------------------------------------------
        # Geometry for the 3D viewer
        # ----------------------------------------------------------

        try:
            project.structure = StructureLoader.load(structure_path)

        except Exception as exc:
            if project.structure_error is None:
                # Some readers raise with an empty message; fall back to
                # the type so the console never shows a bare colon.
                detail = str(exc).strip() or type(exc).__name__

                project.structure_error = (
                    f"Could not read {structure_path.name}: {detail}"
                )

        return project

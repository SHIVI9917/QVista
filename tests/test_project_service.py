"""
Tests for the project loader.

What is guarded here is the reporting, not the reading. ProjectService.load
never raises -- it is called from a window that has to stay open and say
something -- so every failure has to arrive as a sentence on
``Project.structure_error``. A project that comes back with ``structure``
and ``structure_error`` both unset has failed without saying so, and the
user finds out much later and somewhere else.

A path that does not exist used to do exactly that. ``Path.is_file()`` is
False for it, so it was taken for a project directory, no VASP files were
found in a directory that was not there, and the load returned quietly with
nothing loaded and nothing to report. The mistyped name only surfaced from
whichever analysis was tried next, as "No structure is loaded. Open a
structure file first." -- which is true, and points at the wrong thing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qvista.services.project_service import ProjectService


@pytest.fixture
def poscar(tmp_path: Path) -> Path:
    """Rocksalt LiF, written where nothing else can collide with it."""

    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(4.03),
        ["Li", "F"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    directory = tmp_path / "lif"
    directory.mkdir()

    path = directory / "POSCAR"
    structure.to(filename=str(path), fmt="poscar")

    return path


# ------------------------------------------------------------
# A path that is not there
# ------------------------------------------------------------


def test_a_missing_path_is_reported_rather_than_loaded_empty(
    tmp_path: Path,
) -> None:
    """The failure is recorded, not swallowed."""

    project = ProjectService.load(tmp_path / "does_not_exist_at_all.vasp")

    assert project.structure is None
    assert project.structure_info is None

    # The point of the test: not None. Both unset is the state that made
    # this fail silently.
    assert project.structure_error is not None


def test_the_report_names_the_path_that_was_not_found(
    tmp_path: Path,
) -> None:
    """
    The message has to carry the name.

    A user who mistypes a filename needs to see which name failed; a
    message that only says a structure is missing sends them looking at
    the viewer instead of at what they typed.
    """

    missing = tmp_path / "does_not_exist_at_all.vasp"

    project = ProjectService.load(missing)

    assert "does_not_exist_at_all.vasp" in project.structure_error


def test_a_missing_path_does_not_raise(tmp_path: Path) -> None:
    """
    Reported, never thrown.

    The caller is the main window, which opens whatever the file dialog
    hands it. An exception here would take the window down over a name
    that no longer exists on disk.
    """

    ProjectService.load(tmp_path / "gone.vasp")


def test_a_missing_directory_is_reported_too(tmp_path: Path) -> None:
    """A directory that is not there is the same mistake as a file."""

    project = ProjectService.load(tmp_path / "no_such_project")

    assert project.structure is None
    assert project.structure_error is not None


# ------------------------------------------------------------
# The guard does not fire on paths that are there
# ------------------------------------------------------------


def test_an_existing_structure_still_loads_with_nothing_to_report(
    poscar: Path,
) -> None:
    """The new check must not stand between a real file and the viewer."""

    project = ProjectService.load(poscar)

    assert project.structure is not None
    assert project.structure_error is None
    assert project.structure_info.formula == "LiF"


def test_an_existing_directory_with_no_structure_is_left_alone(
    tmp_path: Path,
) -> None:
    """
    An empty project directory is not a missing path.

    It exists, the user may be about to put a POSCAR in it, and the
    window already says "No structure file found in this project." for
    it. Reporting a not-found error here would be wrong.
    """

    directory = tmp_path / "empty_project"
    directory.mkdir()

    project = ProjectService.load(directory)

    assert project.structure is None
    assert project.structure_error is None

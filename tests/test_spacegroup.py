"""
Tests for the space group analyzer.

Covers known structures with known answers, the tolerance dependence that
makes symprec worth exposing, and the guard behaviour for non-periodic
input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qvista.analyses.base import AnalysisError
from qvista.analyses.spacegroup import SpaceGroupAnalysis
from qvista.services.project_service import ProjectService


def _poscar(tmp_path: Path, name: str) -> Path:
    """A POSCAR in its own directory, so fixtures cannot collide."""

    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)

    return directory / "POSCAR"


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------


@pytest.fixture
def silicon_primitive(tmp_path: Path) -> Path:
    """Diamond silicon, primitive cell: Fd-3m, #227."""

    from pymatgen.core import Lattice, Structure

    half = 5.43 / 2.0

    structure = Structure(
        Lattice([[0.0, half, half], [half, 0.0, half], [half, half, 0.0]]),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )

    path = _poscar(tmp_path, "silicon")
    structure.to(filename=str(path), fmt="poscar")

    return path


@pytest.fixture
def rocksalt(tmp_path: Path) -> Path:
    """NaCl, conventional cell: Fm-3m, #225, two distinct orbits."""

    from pymatgen.core import Lattice, Structure

    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]

    species = ["Na"] * 4 + ["Cl"] * 4

    positions = coords + [
        [x + 0.5 if i == 0 else x for i, x in enumerate(c)] for c in coords
    ]

    structure = Structure(Lattice.cubic(5.64), species, positions)

    path = _poscar(tmp_path, "rocksalt")
    structure.to(filename=str(path), fmt="poscar")

    return path


@pytest.fixture
def noisy_silicon(tmp_path: Path) -> Path:
    """Silicon with small displacements -- symmetry depends on tolerance."""

    from pymatgen.core import Lattice, Structure

    half = 5.43 / 2.0

    structure = Structure(
        Lattice([[0.0, half, half], [half, 0.0, half], [half, half, 0.0]]),
        ["Si", "Si"],
        [[0.004, -0.003, 0.002], [0.25, 0.25, 0.25]],
    )

    path = _poscar(tmp_path, "noisy")
    structure.to(filename=str(path), fmt="poscar")

    return path


@pytest.fixture
def triclinic_p1(tmp_path: Path) -> Path:
    """
    Genuinely P1.

    Three distinct species in general positions on a triclinic lattice: no
    symmetry operation can permute them, and with only one atom of each
    species an inversion centre is impossible.
    """

    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.from_parameters(4.0, 4.5, 5.0, 71.0, 82.0, 95.0),
        ["Si", "O", "N"],
        [[0.01, 0.02, 0.03], [0.31, 0.27, 0.23], [0.62, 0.11, 0.55]],
    )

    path = _poscar(tmp_path, "triclinic")
    structure.to(filename=str(path), fmt="poscar")

    return path


@pytest.fixture
def plain_xyz(tmp_path: Path) -> Path:
    """A molecule -- no lattice, so no space group."""

    path = tmp_path / "water.xyz"

    path.write_text(
        "3\nwater\n"
        "O  0.000  0.000  0.000\n"
        "H  0.758  0.586  0.000\n"
        "H -0.758  0.586  0.000\n"
    )

    return path


def _run(path: Path, symprec: float = 0.01):
    """Run the analysis the way the GUI does."""

    return SpaceGroupAnalysis(ProjectService.load(path), symprec).run()


# ------------------------------------------------------------
# Known answers
# ------------------------------------------------------------


def test_silicon_symmetry(silicon_primitive: Path) -> None:
    """Diamond silicon has a well known symmetry description."""

    result = _run(silicon_primitive)

    assert result.symbol == "Fd-3m"
    assert result.number == 227
    assert result.crystal_system == "Cubic"
    assert result.lattice_type == "Cubic"
    assert result.point_group == "m-3m"

    assert result.hall
    assert result.n_operations > 1

    # m-3m contains inversion.
    assert result.centrosymmetric

    assert not result.is_p1


def test_silicon_has_one_orbit(silicon_primitive: Path) -> None:
    """Both silicon atoms are symmetry equivalent."""

    result = _run(silicon_primitive)

    assert len(result.sites) == 1

    site = result.sites[0]

    assert site.species == "Si"
    assert site.multiplicity == 2
    assert site.label == f"2{site.wyckoff_letter}"


def test_rocksalt_has_two_orbits(rocksalt: Path) -> None:
    """Na and Cl sit on different Wyckoff positions."""

    result = _run(rocksalt)

    assert result.number == 225
    assert result.symbol == "Fm-3m"

    assert len(result.sites) == 2

    species = {site.species for site in result.sites}
    assert species == {"Na", "Cl"}


def test_orbits_account_for_every_atom(rocksalt: Path) -> None:
    """Wyckoff multiplicities must sum to the atom count."""

    result = _run(rocksalt)

    total = sum(site.multiplicity for site in result.sites)

    assert total == result.n_input


def test_primitive_is_smaller_than_conventional(rocksalt: Path) -> None:
    """A face-centred cell reduces by a factor of four."""

    result = _run(rocksalt)

    assert result.n_primitive > 0
    assert result.n_primitive < result.n_input

    assert result.n_input == 8
    assert result.n_primitive == 2


# ------------------------------------------------------------
# Tolerance dependence
# ------------------------------------------------------------


def test_symprec_changes_the_answer(noisy_silicon: Path) -> None:
    """
    The reason symprec is exposed in the UI.

    A slightly displaced structure reads as low symmetry at a tight
    tolerance and recovers its true symmetry at a looser one.
    """

    tight = _run(noisy_silicon, symprec=1e-5)
    loose = _run(noisy_silicon, symprec=0.1)

    assert tight.number < loose.number

    assert loose.number == 227


def test_p1_is_warned_about(triclinic_p1: Path) -> None:
    """A P1 result must surface a visible warning."""

    result = _run(triclinic_p1)

    assert result.is_p1
    assert result.number == 1
    assert not result.centrosymmetric

    warnings = " ".join(result.notes).lower()

    assert "warning" in warnings
    assert "p1" in warnings
    assert "tolerance" in warnings


def test_symprec_is_reported(silicon_primitive: Path) -> None:
    """The tolerance used is part of the result, not hidden."""

    result = _run(silicon_primitive, symprec=0.05)

    assert result.symprec == 0.05

    assert any("0.05" in value for _, value in result.as_rows())


# ------------------------------------------------------------
# Guards
# ------------------------------------------------------------


def test_plain_xyz_is_refused(plain_xyz: Path) -> None:
    """A molecule has no space group; refuse with a readable message."""

    analysis = SpaceGroupAnalysis(ProjectService.load(plain_xyz))

    with pytest.raises(AnalysisError) as excinfo:
        analysis.run()

    message = str(excinfo.value)

    assert "lattice" in message.lower()
    assert "Traceback" not in message


# ------------------------------------------------------------
# Presentation
# ------------------------------------------------------------


def test_rows_cover_the_full_description(silicon_primitive: Path) -> None:
    """The console summary includes every headline field."""

    result = _run(silicon_primitive)

    labels = {label for label, _ in result.as_rows()}

    for expected in (
        "Space group",
        "Space group number",
        "Hall symbol",
        "Point group",
        "Crystal system",
        "Lattice type",
        "Symmetry operations",
        "Centrosymmetric",
    ):
        assert expected in labels


def test_wyckoff_table_is_rendered(rocksalt: Path) -> None:
    """The Wyckoff orbits appear as a table in the console notes."""

    result = _run(rocksalt)

    table = "\n".join(result.notes)

    assert "Wyckoff positions" in table

    for site in result.sites:
        assert site.label in table

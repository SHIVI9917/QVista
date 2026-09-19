"""
Tests for the high-symmetry k-path workflow.

The important case is the conventional cell: seekpath expresses the path in
the reciprocal basis of its own standardised primitive cell, so the result
must carry that cell rather than the cell the user opened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qvista.analyses.base import AnalysisError
from qvista.analyses.kpath import (
    GAMMA,
    KPathAnalysis,
    parse_path_text,
    segments_from_labels,
    validate_path_labels,
)
from qvista.services.project_service import ProjectService


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------


def _project(path: Path):
    """Load a project the same way the GUI does."""

    return ProjectService.load(path)


#: Diamond silicon, conventional cubic cell. Written out explicitly rather
#: than via ``from_spacegroup("Fd-3m", ...)``, which uses an origin setting
#: that duplicates every site and yields 16 atoms instead of 8.
SILICON_A = 5.43

SILICON_CONVENTIONAL_COORDS = [
    [0.00, 0.00, 0.00],
    [0.00, 0.50, 0.50],
    [0.50, 0.00, 0.50],
    [0.50, 0.50, 0.00],
    [0.25, 0.25, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
    [0.75, 0.75, 0.25],
]


@pytest.fixture
def silicon_primitive(tmp_path: Path) -> Path:
    """Two-atom primitive silicon, written as a POSCAR."""

    from pymatgen.core import Lattice, Structure

    half = SILICON_A / 2.0

    lattice = Lattice(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]]
    )

    structure = Structure(
        lattice,
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )

    path = tmp_path / "POSCAR"
    structure.to(filename=str(path), fmt="poscar")

    return path


@pytest.fixture
def silicon_conventional_cif(tmp_path: Path) -> Path:
    """Conventional cubic silicon as a CIF -- the case that exposes the bug."""

    from pymatgen.core import Lattice, Structure

    conventional = Structure(
        Lattice.cubic(SILICON_A),
        ["Si"] * 8,
        SILICON_CONVENTIONAL_COORDS,
    )

    path = tmp_path / "silicon_conventional.cif"
    conventional.to(filename=str(path))

    return path


@pytest.fixture
def triclinic_p1(tmp_path: Path) -> Path:
    """
    A deliberately unsymmetric cell, so symmetry analysis returns P1.

    Three distinct species are used on purpose: any two-atom cell has an
    inversion centre at the midpoint of the pair and comes out as P-1 (#2)
    rather than P1 (#1).
    """

    from pymatgen.core import Lattice, Structure

    lattice = Lattice.from_parameters(4.0, 4.5, 5.0, 71.0, 82.0, 95.0)

    structure = Structure(
        lattice,
        ["Si", "O", "N"],
        [
            [0.01, 0.02, 0.03],
            [0.31, 0.27, 0.23],
            [0.62, 0.11, 0.55],
        ],
    )

    path = tmp_path / "POSCAR"
    structure.to(filename=str(path), fmt="poscar")

    return path


@pytest.fixture
def plain_xyz(tmp_path: Path) -> Path:
    """A molecule -- no lattice, so no reciprocal space."""

    path = tmp_path / "water.xyz"

    path.write_text(
        "3\nwater\n"
        "O  0.000  0.000  0.000\n"
        "H  0.758  0.586  0.000\n"
        "H -0.758  0.586  0.000\n"
    )

    return path


# ------------------------------------------------------------
# Clean symmetric structure
# ------------------------------------------------------------


def test_full_path_on_symmetric_structure(silicon_primitive: Path) -> None:
    """A clean structure yields the recommended path and its points."""

    result = KPathAnalysis(_project(silicon_primitive)).run()

    assert result.spacegroup_number == 227
    assert result.segments
    assert not result.is_custom

    # Every label used by the path must be a point that was reported.
    for start, end in result.segments:
        assert start in result.points
        assert end in result.points

    assert GAMMA in result.points

    # A cubic diamond lattice has these by name.
    for expected in ("X", "L", "W", "K"):
        assert expected in result.points


def test_primitive_cell_is_carried(silicon_primitive: Path) -> None:
    """The result always exposes the cell the path is expressed in."""

    result = KPathAnalysis(_project(silicon_primitive)).run()

    assert result.primitive is not None
    assert result.primitive.natoms == 2

    structure = result.primitive.to_pymatgen()

    assert len(structure) == 2


# ------------------------------------------------------------
# Conventional cell -- the ported bug
# ------------------------------------------------------------


def test_conventional_cif_reports_primitive_cell(
    silicon_conventional_cif: Path,
) -> None:
    """
    Opening a conventional cell must not leave the path in the input basis.

    seekpath reduces the 8-atom conventional cubic cell to a 2-atom
    primitive cell, and the path is expressed against the latter.
    """

    project = _project(silicon_conventional_cif)

    result = KPathAnalysis(project).run()

    input_natoms = project.structure_info.natoms

    assert result.primitive is not None

    # The primitive cell really is smaller than what the user opened.
    assert result.primitive.natoms < input_natoms

    # ...and the volume ratio says by how much.
    assert result.volume_ratio > 1.0

    assert result.volume_ratio == pytest.approx(
        input_natoms / result.primitive.natoms,
        rel=1e-6,
    )


def test_reciprocal_lattice_matches_primitive_cell(
    silicon_conventional_cif: Path,
) -> None:
    """
    The stored reciprocal lattice must belong to the stored primitive cell.

    This is the invariant that makes the path usable downstream: if these
    two disagree, q-points and force constants end up in different bases.

    Also pins the unit convention. seekpath returns the reciprocal lattice
    *with* the 2*pi factor, and silently mixing that with a 2*pi-free
    convention rescales every q-point.
    """

    import numpy as np

    result = KPathAnalysis(_project(silicon_conventional_cif)).run()

    primitive = result.primitive

    direct = np.asarray(primitive.lattice, dtype=float)
    reciprocal = np.asarray(primitive.reciprocal_lattice, dtype=float)

    assert np.allclose(
        reciprocal,
        2.0 * np.pi * np.linalg.inv(direct).T,
        atol=1e-8,
    )

    # And it is definitely not the 2*pi-free convention.
    assert not np.allclose(reciprocal, np.linalg.inv(direct).T, atol=1e-6)


def test_conventional_basis_would_have_been_wrong(
    silicon_conventional_cif: Path,
) -> None:
    """
    Guard against a regression to the reference implementation.

    Interpreting the returned fractional k-points in the *input* cell's
    basis gives a materially different vector, which is exactly the silent
    error this workflow exists to avoid.
    """

    import numpy as np

    project = _project(silicon_conventional_cif)

    result = KPathAnalysis(project).run()

    from qvista.services.structure_analyzer import StructureAnalyzer

    input_structure = StructureAnalyzer.parse(
        project.info.source_file
    ).structure

    point = np.asarray(result.points["X"], dtype=float)

    via_primitive = point @ np.asarray(
        result.primitive.reciprocal_lattice,
        dtype=float,
    )

    # Same 2*pi convention on both sides, so the only difference under test
    # is the basis itself rather than a unit factor.
    via_input = point @ (
        2.0
        * np.pi
        * np.linalg.inv(
            np.asarray(input_structure.lattice.matrix, dtype=float)
        ).T
    )

    assert not np.allclose(via_primitive, via_input, atol=1e-3)

    # The error is on the scale of the Brillouin zone, not a rounding wobble.
    assert np.linalg.norm(via_primitive - via_input) > 0.1


# ------------------------------------------------------------
# P1
# ------------------------------------------------------------


def test_p1_structure_warns(triclinic_p1: Path) -> None:
    """P1 must be visible as a warning, not buried."""

    result = KPathAnalysis(_project(triclinic_p1), symprec=0.01).run()

    assert result.spacegroup_number == 1

    warnings = " ".join(result.notes).lower()

    assert "p1" in warnings
    assert "warning" in warnings


# ------------------------------------------------------------
# Plain XYZ
# ------------------------------------------------------------


def test_plain_xyz_is_refused(plain_xyz: Path) -> None:
    """A molecule has no reciprocal space; the refusal must be readable."""

    analysis = KPathAnalysis(_project(plain_xyz))

    with pytest.raises(AnalysisError) as excinfo:
        analysis.run()

    message = str(excinfo.value)

    assert "lattice" in message.lower()

    # A message for a user, not a dump for a developer.
    assert "Traceback" not in message
    assert len(message) > 30


# ------------------------------------------------------------
# Path parsing and validation
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("GAMMA-X-W", ["GAMMA", "X", "W"]),
        ("GAMMA X W", ["GAMMA", "X", "W"]),
        ("GAMMA,X,W", ["GAMMA", "X", "W"]),
        ("GAMMA->X->W", ["GAMMA", "X", "W"]),
        ("  gamma - x  ", ["GAMMA", "X"]),
        ("GAMMA-W_2-GAMMA", ["GAMMA", "W_2", "GAMMA"]),
        ("", []),
    ],
)
def test_parse_path_text(text: str, expected: list[str]) -> None:
    """Any reasonable separator should work."""

    assert parse_path_text(text) == expected


def test_segments_from_labels() -> None:
    """Labels become consecutive pairs."""

    assert segments_from_labels(["GAMMA", "X", "GAMMA"]) == [
        ("GAMMA", "X"),
        ("X", "GAMMA"),
    ]


def test_valid_path_accepted() -> None:
    """A well-formed path produces no message."""

    points = {"GAMMA": [0, 0, 0], "X": [0.5, 0, 0.5], "W": [0.5, 0.25, 0.75]}

    assert validate_path_labels(["GAMMA", "X", "W", "GAMMA"], points) is None


@pytest.mark.parametrize(
    "labels,fragment",
    [
        (["X", "GAMMA"], "start"),
        (["GAMMA", "X"], "end"),
        (["GAMMA", "Q", "GAMMA"], "high-symmetry point"),
        (["GAMMA"], "at least two"),
        (["GAMMA", "X", "X", "GAMMA"], "repeated"),
    ],
)
def test_invalid_paths_explain_themselves(
    labels: list[str],
    fragment: str,
) -> None:
    """Every rejection must say what is wrong, in words."""

    points = {"GAMMA": [0, 0, 0], "X": [0.5, 0, 0.5]}

    message = validate_path_labels(labels, points)

    assert message is not None
    assert fragment in message


def test_custom_path_rejected_by_analysis(silicon_primitive: Path) -> None:
    """An invalid custom path fails as AnalysisError, never as a crash."""

    analysis = KPathAnalysis(
        _project(silicon_primitive),
        custom_labels=["X", "GAMMA"],
    )

    with pytest.raises(AnalysisError) as excinfo:
        analysis.run()

    assert "start" in str(excinfo.value)


def test_custom_path_accepted_by_analysis(silicon_primitive: Path) -> None:
    """A valid custom path is used instead of seekpath's."""

    analysis = KPathAnalysis(
        _project(silicon_primitive),
        custom_labels=["GAMMA", "X", "GAMMA"],
    )

    result = analysis.run()

    assert result.is_custom
    assert result.segments == [("GAMMA", "X"), ("X", "GAMMA")]

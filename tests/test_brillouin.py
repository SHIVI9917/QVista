"""
Tests for the Brillouin zone figure and the KPOINTS export.

The zone must be built from the same primitive cell the k-path came from;
if it is not, the high-symmetry points do not sit on the zone they belong
to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvista.analyses.kpath import KPathAnalysis, group_path_segments
from qvista.services.project_service import ProjectService
from qvista.visualization.brillouin_plots import (
    brillouin_facets,
    draw_brillouin_zone,
    pretty_label,
)


def _poscar(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)

    return directory / "POSCAR"


@pytest.fixture
def silicon_primitive(tmp_path: Path) -> Path:
    """FCC silicon: its Brillouin zone is a truncated octahedron."""

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
def kpath(silicon_primitive: Path):
    """A computed k-path, as the GUI would obtain it."""

    return KPathAnalysis(ProjectService.load(silicon_primitive)).run()


def _figure():
    import matplotlib

    matplotlib.use("Agg")

    from matplotlib.figure import Figure

    return Figure(figsize=(6.0, 5.0))


# ------------------------------------------------------------
# Zone geometry
# ------------------------------------------------------------


def test_fcc_zone_is_a_truncated_octahedron(kpath) -> None:
    """
    The FCC Brillouin zone has 14 faces: 8 hexagons and 6 squares.

    Getting this right is the check that the zone comes from the
    reciprocal lattice of the correct cell.
    """

    facets = brillouin_facets(kpath.primitive.reciprocal_lattice)

    assert len(facets) == 14

    sizes = sorted(len(facet) for facet in facets)

    assert sizes.count(6) == 8
    assert sizes.count(4) == 6


def test_zone_is_centred_on_gamma(kpath) -> None:
    """The zone is the Wigner-Seitz cell about the origin."""

    facets = brillouin_facets(kpath.primitive.reciprocal_lattice)

    vertices = np.asarray(
        [point for facet in facets for point in facet],
        dtype=float,
    )

    assert np.allclose(vertices.mean(axis=0), np.zeros(3), atol=1e-8)


# ------------------------------------------------------------
# Points sit where they should
# ------------------------------------------------------------


def test_gamma_is_at_the_origin(kpath) -> None:
    """Gamma is the zone centre by definition."""

    cartesian = kpath.cartesian_points()

    assert np.allclose(cartesian["GAMMA"], np.zeros(3), atol=1e-12)


def test_high_symmetry_points_lie_within_the_zone(kpath) -> None:
    """
    Every high-symmetry point must sit inside or on the zone boundary.

    This is the test that fails if the zone and the k-points come from
    different cells: the points scatter outside the polyhedron.
    """

    facets = brillouin_facets(kpath.primitive.reciprocal_lattice)

    vertices = np.asarray(
        [point for facet in facets for point in facet],
        dtype=float,
    )

    radius = np.linalg.norm(vertices, axis=1).max()

    for label, point in kpath.cartesian_points().items():

        distance = float(np.linalg.norm(point))

        assert distance <= radius + 1e-6, (
            f"{label} lies outside the Brillouin zone"
        )


def test_x_point_sits_on_a_zone_face(kpath) -> None:
    """
    X is a face-centre of the FCC zone, so it is on the boundary.

    Its distance from Gamma should match a square face centre, which for
    a cubic lattice constant a is 2*pi/a.
    """

    point = kpath.cartesian_points()["X"]

    assert float(np.linalg.norm(point)) == pytest.approx(
        2.0 * np.pi / 5.43,
        rel=1e-6,
    )


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------


def test_figure_is_drawn(kpath) -> None:
    """The figure renders without error and produces 3D content."""

    figure = _figure()

    draw_brillouin_zone(figure, kpath)

    assert figure.axes

    axes = figure.axes[0]

    assert axes.get_xlabel()
    assert axes.collections or axes.lines


def test_figure_exports(kpath, tmp_path: Path) -> None:
    """The figure survives a real save, as the export button does it."""

    figure = _figure()

    draw_brillouin_zone(figure, kpath)

    target = tmp_path / "bz.png"

    figure.savefig(target, dpi=100, bbox_inches="tight")

    assert target.stat().st_size > 5000


def test_drawing_without_a_primitive_cell_is_refused() -> None:
    """A path with no reference cell cannot be drawn."""

    from qvista.analyses.kpath import KPathResult

    with pytest.raises(ValueError) as excinfo:
        draw_brillouin_zone(_figure(), KPathResult())

    assert "primitive" in str(excinfo.value).lower()


def test_pretty_labels() -> None:
    """GAMMA renders as the Greek letter; subscripts are converted."""

    assert pretty_label("GAMMA") == "Γ"
    assert pretty_label("X") == "X"
    assert pretty_label("W_2") == "W₂"


# ------------------------------------------------------------
# KPOINTS export
# ------------------------------------------------------------


def test_kpoints_header(kpath) -> None:
    """A VASP line-mode KPOINTS file has a fixed four-line header."""

    lines = kpath.to_kpoints(npoints=20).splitlines()

    assert lines[1].strip() == "20"
    assert lines[2].strip().lower().startswith("line")
    assert lines[3].strip().lower().startswith("reciprocal")


def test_kpoints_contains_every_segment(kpath) -> None:
    """Each segment contributes a start/end coordinate pair."""

    text = kpath.to_kpoints()

    pairs = sum(len(branch) - 1 for branch in kpath.branches())

    # Every coordinate line carries a "! <label>" comment; two per segment.
    coordinate_lines = [
        line for line in text.splitlines() if "!" in line
    ]

    assert len(coordinate_lines) == 2 * pairs


def test_kpoints_writes_gamma_by_name(kpath) -> None:
    """VASP cannot read the Greek letter, so Gamma is spelled out."""

    text = kpath.to_kpoints()

    assert "! Gamma" in text
    assert "Γ" not in text


def test_branches_match_the_shared_grouping(kpath) -> None:
    """
    The zone plot and the phonon band split the path identically.

    Both go through group_path_segments, so a break drawn in the zone is
    the same break the band structure leaves open.
    """

    from qvista.analyses.phonon import group_segments

    assert kpath.branches() == group_path_segments(kpath.segments)
    assert kpath.branches() == group_segments(kpath.segments)

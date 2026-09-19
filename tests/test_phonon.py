"""
Tests for the phonon workflow.

Forces come from a trivial analytic calculator rather than MatterSim, so the
whole pipeline -- displacements, forces, force constants, band, DOS -- runs
end to end in seconds and never needs torch installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qvista.analyses.base import AnalysisError
from qvista.analyses.guards import detect_vacuum
from qvista.analyses.kpath import KPathAnalysis
from qvista.analyses.phonon import (
    MODE_BAND,
    MODE_BOTH,
    MODE_DOS,
    PhononAnalysis,
    count_displacements,
    group_segments,
    suggest_supercell,
)
from qvista.services.project_service import ProjectService
from qvista.services.structure_analyzer import StructureAnalyzer


# ------------------------------------------------------------
# A cheap stand-in for the force field
# ------------------------------------------------------------


def analytic_calculator():
    """
    A cheap, real interatomic potential standing in for the force field.

    Lennard-Jones rather than a hand-rolled on-site spring: a spring that
    pulls each atom back to its own lattice site couples nothing to
    anything, so every mode comes out degenerate, the DOS frequency range
    collapses to zero width and phonopy divides by it. A genuine pair
    potential produces real dispersion, which is what the pipeline needs to
    be exercised against.

    Sigma is set near silicon's nearest-neighbour distance so the atoms are
    actually inside the cutoff and forces are non-trivial.
    """

    from ase.calculators.lj import LennardJones

    return LennardJones(sigma=2.35, epsilon=1.0, rc=6.0)


@pytest.fixture
def analytic_forces(monkeypatch):
    """Make PhononAnalysis use the analytic potential, never MatterSim."""

    monkeypatch.setattr(
        PhononAnalysis,
        "_get_calculator",
        lambda self: analytic_calculator(),
    )


# ------------------------------------------------------------
# Structures
# ------------------------------------------------------------


def _poscar(tmp_path: Path, name: str) -> Path:
    """
    A POSCAR in its own directory.

    The filename must stay "POSCAR" for format detection, and pytest hands
    every fixture in a test the same tmp_path, so each structure needs its
    own subdirectory or the fixtures overwrite one another.
    """

    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)

    return directory / "POSCAR"


@pytest.fixture
def silicon_primitive(tmp_path: Path) -> Path:
    """Two-atom primitive silicon -- small and highly symmetric."""

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
def triclinic_p1(tmp_path: Path) -> Path:
    """Three species in general positions -- genuinely P1."""

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
def slab(tmp_path: Path) -> Path:
    """A thin film with a long, mostly empty c axis."""

    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.from_parameters(4.0, 4.0, 30.0, 90.0, 90.0, 90.0),
        ["Si", "Si", "Si", "Si"],
        [
            [0.0, 0.0, 0.10],
            [0.5, 0.5, 0.10],
            [0.0, 0.5, 0.17],
            [0.5, 0.0, 0.17],
        ],
    )

    path = _poscar(tmp_path, "slab")
    structure.to(filename=str(path), fmt="poscar")

    return path


@pytest.fixture
def plain_xyz(tmp_path: Path) -> Path:
    """A molecule -- no lattice."""

    path = tmp_path / "water.xyz"

    path.write_text(
        "3\nwater\n"
        "O  0.000  0.000  0.000\n"
        "H  0.758  0.586  0.000\n"
        "H -0.758  0.586  0.000\n"
    )

    return path


def _kpath(path: Path):
    """Project plus its k-path, as the GUI would obtain them."""

    project = ProjectService.load(path)

    return project, KPathAnalysis(project).run()


# ------------------------------------------------------------
# End to end
# ------------------------------------------------------------


def test_band_and_dos_end_to_end(
    silicon_primitive: Path,
    analytic_forces,
) -> None:
    """The whole pipeline produces a band structure and a DOS."""

    project, kpath = _kpath(silicon_primitive)

    result = PhononAnalysis(
        project,
        kpath,
        supercell=(1, 1, 1),
        mode=MODE_BOTH,
        mesh=(4, 4, 4),
    ).run()

    # --- band ---
    assert result.band is not None
    assert result.band["frequencies"]
    assert result.band["distances"]
    assert len(result.band["frequencies"]) == len(result.band["distances"])

    n_bands = result.band["frequencies"][0].shape[1]
    assert n_bands == 3 * kpath.primitive.natoms

    # --- dos ---
    assert result.dos is not None
    assert len(result.dos["frequency_points"]) > 0
    assert len(result.dos["total_dos"]) == len(result.dos["frequency_points"])


def test_band_uses_public_api_not_private_attribute(
    silicon_primitive: Path,
    analytic_forces,
) -> None:
    """
    The reference read ``phonon._band_structure`` without ever calling
    ``run_band_structure()``. Confirm the public accessor is populated.
    """

    project, kpath = _kpath(silicon_primitive)

    result = PhononAnalysis(
        project,
        kpath,
        supercell=(1, 1, 1),
        mode=MODE_BAND,
    ).run()

    phonon = result.phonon_data.phonon

    # The public property is populated, which only happens after the public
    # run_band_structure() has been called. Reading the private
    # _band_structure, as the reference did, would have found nothing.
    assert phonon.band_structure is not None
    assert phonon.band_structure.frequencies


def test_phonon_is_built_on_the_kpath_primitive_cell(
    silicon_primitive: Path,
    analytic_forces,
) -> None:
    """
    The phonon cell and the k-path must share a basis.

    Building Phonopy from the raw input while taking the path from
    seekpath's primitive cell is the silent-wrongness bug being avoided.
    """

    import numpy as np

    project, kpath = _kpath(silicon_primitive)

    result = PhononAnalysis(
        project,
        kpath,
        supercell=(1, 1, 1),
        mode=MODE_BAND,
    ).run()

    phonon = result.phonon_data.phonon

    expected = np.asarray(kpath.primitive.lattice, dtype=float)

    assert np.allclose(
        np.asarray(phonon.unitcell.cell, dtype=float),
        expected,
        atol=1e-8,
    )

    # The band structure is evaluated against phonon.primitive, not
    # phonon.unitcell. phonopy 4 defaults primitive_matrix to "auto", which
    # silently re-reduces the cell to a different basis; this asserts it was
    # pinned to the identity so the q-points and the dynamical matrix agree.
    assert np.allclose(
        np.asarray(phonon.primitive.cell, dtype=float),
        expected,
        atol=1e-8,
    )

    assert np.allclose(
        np.asarray(phonon.primitive_matrix, dtype=float),
        np.eye(3),
        atol=1e-12,
    )

    assert result.phonon_data.primitive is kpath.primitive


def test_force_constants_are_cached_not_recomputed(
    silicon_primitive: Path,
    analytic_forces,
    monkeypatch,
) -> None:
    """A second menu item must reuse force constants, not rebuild them."""

    project, kpath = _kpath(silicon_primitive)

    first = PhononAnalysis(
        project,
        kpath,
        supercell=(1, 1, 1),
        mode=MODE_BAND,
    ).run()

    cached = first.phonon_data

    # Any attempt to rebuild would call this; make it fail loudly.
    def explode(self):
        raise AssertionError("force constants were recomputed")

    monkeypatch.setattr(PhononAnalysis, "_build_force_constants", explode)

    second = PhononAnalysis(
        project,
        kpath,
        supercell=(1, 1, 1),
        mode=MODE_DOS,
        cached=cached,
        mesh=(4, 4, 4),
    ).run()

    assert second.reused_cache
    assert second.dos is not None
    assert second.phonon_data is cached


def test_cache_rejected_when_supercell_differs(
    silicon_primitive: Path,
    analytic_forces,
) -> None:
    """A cache for a different supercell must not be reused."""

    project, kpath = _kpath(silicon_primitive)

    first = PhononAnalysis(
        project,
        kpath,
        supercell=(1, 1, 1),
        mode=MODE_BAND,
    ).run()

    assert not first.phonon_data.matches(
        first.phonon_data.source,
        (2, 2, 2),
        first.phonon_data.symprec,
        first.phonon_data.distance,
    )


def test_forces_are_cancellable(
    silicon_primitive: Path,
    analytic_forces,
) -> None:
    """Cancelling stops the force loop rather than running to completion."""

    from qvista.analyses.base import AnalysisCancelled

    project, kpath = _kpath(silicon_primitive)

    analysis = PhononAnalysis(
        project,
        kpath,
        supercell=(2, 2, 2),
        mode=MODE_BAND,
    )

    seen = []

    def progress(message, percent):
        seen.append(message)
        analysis.cancel()

    with pytest.raises(AnalysisCancelled):
        analysis.run(progress=progress)

    # It stopped early rather than completing every displacement.
    assert len(seen) < 50


# ------------------------------------------------------------
# Guards
# ------------------------------------------------------------


def test_plain_xyz_is_refused(plain_xyz: Path) -> None:
    """A molecule has no phonons; the refusal must be readable."""

    project = ProjectService.load(plain_xyz)

    from qvista.analyses.kpath import KPathResult

    analysis = PhononAnalysis(project, KPathResult())

    with pytest.raises(AnalysisError) as excinfo:
        analysis.run()

    message = str(excinfo.value)

    assert "lattice" in message.lower()
    assert "Traceback" not in message


def test_p1_displacement_count_explodes(
    triclinic_p1: Path,
    silicon_primitive: Path,
) -> None:
    """
    P1 gives no symmetry reduction, so displacements scale with atoms.

    Silicon, at 227, needs far fewer displacements per atom.
    """

    _, p1_path = _kpath(triclinic_p1)
    _, si_path = _kpath(silicon_primitive)

    p1_count, p1_atoms = count_displacements(p1_path.primitive, (1, 1, 1))
    si_count, si_atoms = count_displacements(si_path.primitive, (1, 1, 1))

    assert p1_path.spacegroup_number == 1

    # Six displacements per atom is the unreduced worst case (+/- x, y, z).
    assert p1_count >= 3 * p1_atoms

    assert (p1_count / p1_atoms) > (si_count / si_atoms)


def test_slab_supercell_suggestion_keeps_vacuum_axis_at_one(
    slab: Path,
) -> None:
    """A vacuum direction must never be replicated."""

    structure = StructureAnalyzer.parse(slab).structure

    vacuum = detect_vacuum(structure)

    assert vacuum.has_vacuum

    axes = [d.axis for d in vacuum.directions]
    assert 2 in axes  # the long c axis

    suggestion = suggest_supercell(len(structure), vacuum)

    assert suggestion[2] == 1
    assert suggestion[0] > 1

    # And it is emphatically not a blind 3x3x3.
    assert suggestion != (3, 3, 3)


def test_bulk_supercell_suggestion_is_isotropic(
    silicon_primitive: Path,
) -> None:
    """A clean bulk cell gets an ordinary supercell."""

    structure = StructureAnalyzer.parse(silicon_primitive).structure

    suggestion = suggest_supercell(len(structure), detect_vacuum(structure))

    assert suggestion == (3, 3, 3)


def test_slab_cost_is_reported_before_running(slab: Path) -> None:
    """
    The 3x3x3 cost on a slab must be knowable up front.

    This is the number the dialog shows and refuses to run past without
    explicit confirmation.
    """

    _, kpath = _kpath(slab)

    count, atoms = count_displacements(kpath.primitive, (3, 3, 3))

    assert count > 0

    # 27 copies of the primitive cell.
    assert atoms == 27 * kpath.primitive.natoms


# ------------------------------------------------------------
# Path grouping
# ------------------------------------------------------------


def test_group_segments_splits_at_breaks() -> None:
    """Disconnected stretches become separate phonopy band paths."""

    segments = [("G", "X"), ("X", "U"), ("K", "G"), ("G", "L")]

    assert group_segments(segments) == [["G", "X", "U"], ["K", "G", "L"]]


def test_group_segments_handles_single_stretch() -> None:
    """A fully connected path stays as one stretch."""

    segments = [("G", "X"), ("X", "W"), ("W", "G")]

    assert group_segments(segments) == [["G", "X", "W", "G"]]


def test_group_segments_empty() -> None:
    """No segments, no stretches."""

    assert group_segments([]) == []

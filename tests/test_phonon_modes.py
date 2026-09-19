"""
Tests for element projection and phonon mode animation.

Uses a cheap analytic potential, so nothing here needs a machine-learning
force field installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvista.analyses.kpath import KPathAnalysis
from qvista.analyses.phonon import (
    MODE_PROJECTED,
    PhononAnalysis,
    project_eigenvectors,
)
from qvista.analyses.phonon_modes import (
    animation_frames,
    mode_displacements,
    modes_at_qpoint,
    to_xyz_trajectory,
    travelling_frames,
)
from qvista.services.project_service import ProjectService


def _poscar(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)

    return directory / "POSCAR"


@pytest.fixture
def analytic_forces(monkeypatch):
    """Lennard-Jones instead of a machine-learning potential."""

    from ase.calculators.lj import LennardJones

    monkeypatch.setattr(
        PhononAnalysis,
        "_get_calculator",
        lambda self: LennardJones(sigma=2.35, epsilon=1.0, rc=6.0),
    )


@pytest.fixture
def binary_crystal(tmp_path: Path) -> Path:
    """
    Rocksalt-like NaCl: two elements with very different masses.

    A binary is essential here -- projection onto a single-element crystal
    is trivially 1.0 everywhere and would test nothing.
    """

    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    path = _poscar(tmp_path, "nacl")
    structure.to(filename=str(path), fmt="poscar")

    return path


@pytest.fixture
def phonon_result(binary_crystal: Path, analytic_forces):
    """A projected band calculation on the binary crystal."""

    project = ProjectService.load(binary_crystal)
    kpath = KPathAnalysis(project).run()

    return PhononAnalysis(
        project,
        kpath,
        supercell=(1, 1, 1),
        mode=MODE_PROJECTED,
    ).run()


# ------------------------------------------------------------
# Projection
# ------------------------------------------------------------


def test_projection_is_produced(phonon_result) -> None:
    """A projected run carries per-element weights."""

    projections = phonon_result.band["projections"]

    assert set(projections) == {"Na", "Cl"}

    for weights in projections.values():
        assert weights.ndim == 2


def test_projection_weights_sum_to_one(phonon_result) -> None:
    """
    Every mode is fully accounted for across the elements.

    Eigenvectors are normalised, so the shares must sum to 1 at every
    q-point and band. This is the check that catches a wrong reshape.
    """

    projections = phonon_result.band["projections"]

    total = sum(np.asarray(w) for w in projections.values())

    assert np.allclose(total, 1.0, atol=1e-8)


def test_projection_weights_are_physical(phonon_result) -> None:
    """Weights are probabilities, so they live in [0, 1]."""

    for weights in phonon_result.band["projections"].values():

        assert np.asarray(weights).min() >= -1e-12
        assert np.asarray(weights).max() <= 1.0 + 1e-12


def test_projection_matches_band_shape(phonon_result) -> None:
    """Weights line up with the frequencies they annotate."""

    frequencies = np.concatenate(
        [np.asarray(s) for s in phonon_result.band["frequencies"]]
    )

    for weights in phonon_result.band["projections"].values():
        assert np.asarray(weights).shape == frequencies.shape


def test_projection_reshape_separates_atoms_from_axes() -> None:
    """
    A hand-built case: mode 0 sits entirely on atom 0.

    Two atoms, so 6 components. Putting all the weight on the first atom's
    x component must give element A weight 1 and element B weight 0 -- the
    assertion that fails if atom and Cartesian indices are transposed.
    """

    eigenvectors = [np.zeros((1, 6, 6), dtype=complex)]

    eigenvectors[0][0, 0, 0] = 1.0  # atom 0, x, mode 0
    eigenvectors[0][0, 4, 1] = 1.0  # atom 1, y, mode 1

    weights = project_eigenvectors(eigenvectors, ["A", "B"])

    assert weights["A"][0, 0] == pytest.approx(1.0)
    assert weights["B"][0, 0] == pytest.approx(0.0)

    assert weights["A"][0, 1] == pytest.approx(0.0)
    assert weights["B"][0, 1] == pytest.approx(1.0)


# ------------------------------------------------------------
# Mode animation
# ------------------------------------------------------------


def test_modes_at_gamma(phonon_result) -> None:
    """Diagonalising at Gamma gives 3N modes."""

    phonon = phonon_result.phonon_data.phonon

    modes = modes_at_qpoint(phonon, [0.0, 0.0, 0.0], label="GAMMA")

    assert len(modes) == 3 * len(phonon.primitive)

    for mode in modes:
        assert mode.label == "GAMMA"
        assert mode.eigenvector.shape == (3 * len(phonon.primitive),)


def test_acoustic_modes_are_near_zero_at_gamma(phonon_result) -> None:
    """The three acoustic branches vanish at the zone centre."""

    phonon = phonon_result.phonon_data.phonon

    modes = modes_at_qpoint(phonon, [0.0, 0.0, 0.0])

    lowest = sorted(abs(mode.frequency) for mode in modes)[:3]

    assert max(lowest) < 0.2


def test_heavy_atoms_move_less(phonon_result) -> None:
    """
    Mass weighting must be undone.

    phonopy's eigenvectors are mass-weighted; dividing by sqrt(m) recovers
    real displacements. Without it a heavy atom would be drawn moving as
    far as a light one, which is backwards.
    """

    phonon = phonon_result.phonon_data.phonon

    masses = np.asarray(phonon.primitive.masses)

    # An eigenvector spread evenly over both atoms in mass-weighted space.
    eigenvector = np.ones(3 * len(masses), dtype=complex)

    displacement = np.abs(mode_displacements(eigenvector, masses))

    heavy = int(np.argmax(masses))
    light = int(np.argmin(masses))

    assert displacement[heavy].max() < displacement[light].max()


def test_animation_frames_oscillate(phonon_result) -> None:
    """Frames sweep one period and return to the start."""

    phonon = phonon_result.phonon_data.phonon
    primitive = phonon.primitive

    modes = modes_at_qpoint(phonon, [0.0, 0.0, 0.0])

    positions = np.asarray(primitive.positions)

    frames = animation_frames(
        positions,
        modes[-1].eigenvector,
        primitive.masses,
        frames=8,
        amplitude=0.3,
    )

    assert len(frames) == 8

    for frame in frames:
        assert np.asarray(frame).shape == positions.shape

    # Half a period later the displacement is inverted.
    first = np.asarray(frames[0]) - positions
    half = np.asarray(frames[4]) - positions

    assert np.allclose(first, -half, atol=1e-8)


def test_animation_amplitude_is_respected(phonon_result) -> None:
    """The largest excursion matches the requested amplitude."""

    phonon = phonon_result.phonon_data.phonon
    primitive = phonon.primitive

    modes = modes_at_qpoint(phonon, [0.0, 0.0, 0.0])

    positions = np.asarray(primitive.positions)

    frames = animation_frames(
        positions,
        modes[-1].eigenvector,
        primitive.masses,
        frames=24,
        amplitude=0.25,
    )

    largest = max(
        float(np.abs(np.asarray(frame) - positions).max())
        for frame in frames
    )

    assert largest == pytest.approx(0.25, rel=0.05)


def test_xyz_trajectory_round_trips(phonon_result, tmp_path: Path) -> None:
    """The exported trajectory is readable by ASE, frame for frame."""

    from ase.io import read

    phonon = phonon_result.phonon_data.phonon
    primitive = phonon.primitive

    modes = modes_at_qpoint(phonon, [0.0, 0.0, 0.0])

    frames = animation_frames(
        np.asarray(primitive.positions),
        modes[-1].eigenvector,
        primitive.masses,
        frames=6,
    )

    symbols = [str(s) for s in primitive.symbols]

    text = to_xyz_trajectory(
        symbols,
        frames,
        cell=primitive.cell,
        comment="test",
    )

    target = tmp_path / "mode.xyz"
    target.write_text(text, encoding="utf-8")

    images = read(str(target), index=":")

    assert len(images) == 6
    assert len(images[0]) == len(symbols)

    # The lattice survived, so the frames are periodic structures.
    assert images[0].cell.volume > 0


# ------------------------------------------------------------
# The repeated cell
#
# What separates a phonon from a rigid shuffle is the phase between one
# cell and the next. In a single cell that phase never appears, so every
# mode looks the same regardless of where in the zone it came from.
# ------------------------------------------------------------


def test_repeating_the_cell_repeats_the_atoms(phonon_result) -> None:

    phonon = phonon_result.phonon_data.phonon
    primitive = phonon.primitive

    modes = modes_at_qpoint(phonon, [0.0, 0.0, 0.0])

    frames, source, lattice, equilibrium = travelling_frames(
        np.asarray(primitive.cell),
        np.asarray(primitive.positions),
        primitive.masses,
        modes[-1].eigenvector,
        (0.0, 0.0, 0.0),
        repeat=(2, 1, 3),
        frames=4,
    )

    natoms = len(primitive) * 2 * 1 * 3

    assert len(frames) == 4
    assert frames[0].shape == (natoms, 3)
    assert len(source) == natoms
    assert equilibrium.shape == (natoms, 3)

    # Every repeated atom points back at one of the originals.
    assert set(source.tolist()) == set(range(len(primitive)))

    # ...and the lattice grew by exactly the repetition.
    assert np.allclose(
        np.asarray(lattice),
        (np.asarray(primitive.cell).T * np.array([2.0, 1.0, 3.0])).T,
    )


def test_at_gamma_every_cell_moves_together(phonon_result) -> None:
    """
    q = 0 means no phase between cells, so the copies are identical.

    This is the case where repeating buys nothing, and it is the one
    that proves the phase factor is being applied rather than assumed.
    """

    phonon = phonon_result.phonon_data.phonon
    primitive = phonon.primitive

    modes = modes_at_qpoint(phonon, [0.0, 0.0, 0.0])

    frames, _source, _lattice, equilibrium = travelling_frames(
        np.asarray(primitive.cell),
        np.asarray(primitive.positions),
        primitive.masses,
        modes[-1].eigenvector,
        (0.0, 0.0, 0.0),
        repeat=(2, 1, 1),
        frames=8,
    )

    natoms = len(primitive)

    for positions in frames:

        first = positions[:natoms] - equilibrium[:natoms]
        second = positions[natoms:] - equilibrium[natoms:]

        assert np.allclose(first, second, atol=1e-12)


def test_at_the_zone_boundary_neighbouring_cells_oppose(
    phonon_result,
) -> None:
    """
    q = 1/2 puts exp(i pi) = -1 between one cell and the next.

    Neighbouring cells move exactly against each other -- which is the
    whole reason for repeating, and is invisible in a single cell.
    """

    phonon = phonon_result.phonon_data.phonon
    primitive = phonon.primitive

    modes = modes_at_qpoint(phonon, [0.5, 0.0, 0.0])

    frames, _source, _lattice, equilibrium = travelling_frames(
        np.asarray(primitive.cell),
        np.asarray(primitive.positions),
        primitive.masses,
        modes[-1].eigenvector,
        (0.5, 0.0, 0.0),
        repeat=(2, 1, 1),
        frames=8,
    )

    natoms = len(primitive)

    moved = False

    for positions in frames:

        first = positions[:natoms] - equilibrium[:natoms]
        second = positions[natoms:] - equilibrium[natoms:]

        assert np.allclose(first, -second, atol=1e-12)

        if np.abs(first).max() > 1e-6:
            moved = True

    assert moved, "the mode has to actually displace something"


def test_the_amplitude_is_the_largest_excursion(phonon_result) -> None:
    """
    Set in Angstrom, and honoured whatever the eigenvector's norm was.

    phonopy normalises eigenvectors to one, not to anything with a
    length, so without this the drawn amplitude would depend on the
    number of atoms in the cell.
    """

    phonon = phonon_result.phonon_data.phonon
    primitive = phonon.primitive

    modes = modes_at_qpoint(phonon, [0.0, 0.0, 0.0])

    frames, _source, _lattice, equilibrium = travelling_frames(
        np.asarray(primitive.cell),
        np.asarray(primitive.positions),
        primitive.masses,
        modes[-1].eigenvector,
        (0.0, 0.0, 0.0),
        repeat=(1, 1, 1),
        frames=48,
        amplitude=0.25,
    )

    largest = max(
        float(np.abs(positions - equilibrium).max())
        for positions in frames
    )

    assert largest == pytest.approx(0.25, rel=0.05)


def test_equilibrium_is_not_the_first_frame(phonon_result) -> None:
    """
    Frame zero is already displaced, at phase zero of the oscillation.

    Anything measuring a displacement -- the arrows in the viewer, for
    one -- has to measure it from the rest positions, not from whichever
    frame happened to be drawn first.
    """

    phonon = phonon_result.phonon_data.phonon
    primitive = phonon.primitive

    modes = modes_at_qpoint(phonon, [0.0, 0.0, 0.0])

    frames, _source, _lattice, equilibrium = travelling_frames(
        np.asarray(primitive.cell),
        np.asarray(primitive.positions),
        primitive.masses,
        modes[-1].eigenvector,
        (0.0, 0.0, 0.0),
        repeat=(1, 1, 1),
        frames=8,
        amplitude=0.3,
    )

    assert np.allclose(
        equilibrium, np.asarray(primitive.positions), atol=1e-12
    )

    assert not np.allclose(frames[0], equilibrium, atol=1e-6)

"""
Tests for NEB and molecular dynamics.

Both use a cheap analytic potential rather than a machine-learning model,
so the whole pipeline runs in seconds and needs no force field installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvista.analyses.base import AnalysisCancelled, AnalysisError
from qvista.analyses.md import (
    ENSEMBLE_NVE,
    ENSEMBLE_NVT,
    MDAnalysis,
)
from qvista.analyses.neb import NEBAnalysis
from qvista.services.project_service import ProjectService


def analytic_calculator():
    """A real pair potential, instant to evaluate."""

    from ase.calculators.lj import LennardJones

    return LennardJones(sigma=2.5, epsilon=0.05, rc=7.0)


@pytest.fixture
def analytic_forces(monkeypatch):
    """Point both workflows at the analytic potential."""

    for cls in (NEBAnalysis, MDAnalysis):
        monkeypatch.setattr(
            cls,
            "_get_calculator",
            lambda self: analytic_calculator(),
        )


def _poscar(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)

    return directory / "POSCAR"


def _cell(tmp_path: Path, name: str, shift: float = 0.0) -> Path:
    """
    A small cell with one atom optionally displaced.

    Two of these, with different shifts, are the endpoints of a hop.
    """

    from pymatgen.core import Lattice, Structure

    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]

    coords[0] = [shift, 0.0, 0.0]

    structure = Structure(Lattice.cubic(4.2), ["Ar"] * 4, coords)

    path = _poscar(tmp_path, name)
    structure.to(filename=str(path), fmt="poscar")

    return path


# ------------------------------------------------------------
# NEB
# ------------------------------------------------------------


def test_neb_produces_a_path(tmp_path: Path, analytic_forces) -> None:
    """A NEB run returns an energy for every image."""

    start = _cell(tmp_path, "start", shift=0.0)
    end = _cell(tmp_path, "end", shift=0.5)

    result = NEBAnalysis(
        ProjectService.load(start),
        end,
        n_images=3,
        steps=5,
    ).run()

    # endpoints plus intermediates
    assert result.n_images == 5
    assert len(result.energies) == 5

    # Energies are relative to the start, so the first is zero.
    assert result.energies[0] == pytest.approx(0.0, abs=1e-9)

    assert len(result.coordinates) == 5
    assert result.coordinates[0] == 0.0

    # The path coordinate only ever increases.
    assert all(
        b >= a
        for a, b in zip(result.coordinates[:-1], result.coordinates[1:])
    )


def test_neb_barrier_is_consistent(tmp_path: Path, analytic_forces) -> None:
    """The reported barriers follow from the profile."""

    start = _cell(tmp_path, "start", shift=0.0)
    end = _cell(tmp_path, "end", shift=0.5)

    result = NEBAnalysis(
        ProjectService.load(start),
        end,
        n_images=3,
        steps=5,
    ).run()

    assert result.barrier_forward == pytest.approx(max(result.energies))

    assert result.reaction_energy == pytest.approx(result.energies[-1])

    assert result.barrier_reverse == pytest.approx(
        result.barrier_forward - result.reaction_energy
    )

    assert result.saddle_index == int(np.argmax(result.energies))


def test_neb_rejects_mismatched_atom_counts(
    tmp_path: Path,
    analytic_forces,
) -> None:
    """Endpoints must contain the same atoms."""

    from pymatgen.core import Lattice, Structure

    start = _cell(tmp_path, "start")

    other = _poscar(tmp_path, "other")
    Structure(
        Lattice.cubic(4.2),
        ["Ar"] * 3,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
    ).to(filename=str(other), fmt="poscar")

    analysis = NEBAnalysis(ProjectService.load(start), other, n_images=2)

    with pytest.raises(AnalysisError) as excinfo:
        analysis.run()

    message = str(excinfo.value)

    assert "atom counts" in message
    assert "Traceback" not in message


def test_neb_rejects_different_composition(
    tmp_path: Path,
    analytic_forces,
) -> None:
    """A path needs the same elements at both ends."""

    from pymatgen.core import Lattice, Structure

    start = _cell(tmp_path, "start")

    other = _poscar(tmp_path, "other")
    Structure(
        Lattice.cubic(4.2),
        ["Ar", "Ar", "Ar", "Kr"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    ).to(filename=str(other), fmt="poscar")

    with pytest.raises(AnalysisError) as excinfo:
        NEBAnalysis(ProjectService.load(start), other, n_images=2).run()

    assert "composition" in str(excinfo.value)


def test_neb_needs_at_least_one_image(
    tmp_path: Path,
    analytic_forces,
) -> None:
    """Zero images is not a path."""

    start = _cell(tmp_path, "start")
    end = _cell(tmp_path, "end", shift=0.5)

    with pytest.raises(AnalysisError):
        NEBAnalysis(ProjectService.load(start), end, n_images=0).run()


def test_neb_is_cancellable(tmp_path: Path, analytic_forces) -> None:
    """Cancelling stops the relaxation part way."""

    start = _cell(tmp_path, "start")
    end = _cell(tmp_path, "end", shift=0.5)

    analysis = NEBAnalysis(
        ProjectService.load(start),
        end,
        n_images=3,
        steps=200,
    )

    seen = []

    def progress(message, percent):
        seen.append(message)
        analysis.cancel()

    with pytest.raises(AnalysisCancelled):
        analysis.run(progress=progress)

    assert len(seen) < 50


def test_neb_csv_contains_the_profile(
    tmp_path: Path,
    analytic_forces,
) -> None:
    """Export carries the numbers, not just a summary."""

    start = _cell(tmp_path, "start")
    end = _cell(tmp_path, "end", shift=0.5)

    result = NEBAnalysis(
        ProjectService.load(start),
        end,
        n_images=3,
        steps=3,
    ).run()

    csv = result.to_csv()

    assert "forward barrier" in csv
    assert "image,reaction_coordinate_ang,energy_ev" in csv

    rows = [
        line
        for line in csv.splitlines()
        if line and not line.startswith("#") and line.count(",") == 2
    ]

    # header plus one row per image
    assert len(rows) == result.n_images + 1


# ------------------------------------------------------------
# Molecular dynamics
# ------------------------------------------------------------


def test_md_records_a_time_series(tmp_path: Path, analytic_forces) -> None:
    """Every step contributes one point to each trace."""

    source = _cell(tmp_path, "md")

    result = MDAnalysis(
        ProjectService.load(source),
        temperature=300.0,
        steps=20,
        ensemble=ENSEMBLE_NVT,
    ).run()

    for series in (
        result.times,
        result.potential_energies,
        result.kinetic_energies,
        result.total_energies,
        result.temperatures,
    ):
        assert len(series) == 20

    # Time advances by the timestep.
    assert result.times[0] == pytest.approx(result.timestep)
    assert result.times[-1] == pytest.approx(20 * result.timestep)


def test_md_total_energy_is_the_sum(
    tmp_path: Path,
    analytic_forces,
) -> None:
    """Total energy is potential plus kinetic, at every step."""

    source = _cell(tmp_path, "md")

    result = MDAnalysis(
        ProjectService.load(source),
        steps=10,
    ).run()

    for index in range(len(result.times)):
        assert result.total_energies[index] == pytest.approx(
            result.potential_energies[index]
            + result.kinetic_energies[index]
        )


def test_md_temperature_is_positive(
    tmp_path: Path,
    analytic_forces,
) -> None:
    """Atoms given velocities have a temperature above zero."""

    source = _cell(tmp_path, "md")

    result = MDAnalysis(
        ProjectService.load(source),
        temperature=400.0,
        steps=15,
    ).run()

    assert min(result.temperatures) > 0.0
    assert result.mean_temperature > 0.0


def test_nve_conserves_energy_better_than_a_long_timestep(
    tmp_path: Path,
    analytic_forces,
) -> None:
    """
    NVE energy drift grows with the timestep.

    This is the diagnostic the workflow reports: a short step conserves
    energy, a long one does not. If drift did not respond to the timestep
    the number would be meaningless.
    """

    source = _cell(tmp_path, "md")
    project = ProjectService.load(source)

    short = MDAnalysis(
        project,
        timestep=0.5,
        steps=30,
        ensemble=ENSEMBLE_NVE,
    ).run()

    long = MDAnalysis(
        project,
        timestep=5.0,
        steps=30,
        ensemble=ENSEMBLE_NVE,
    ).run()

    assert abs(short.energy_drift) < abs(long.energy_drift)


def test_md_rejects_bad_settings(tmp_path: Path, analytic_forces) -> None:
    """Nonsensical input is refused before any work is done."""

    source = _cell(tmp_path, "md")
    project = ProjectService.load(source)

    with pytest.raises(AnalysisError):
        MDAnalysis(project, steps=0).run()

    with pytest.raises(AnalysisError):
        MDAnalysis(project, timestep=0.0).run()

    with pytest.raises(AnalysisError) as excinfo:
        MDAnalysis(project, ensemble="npt").run()

    assert "ensemble" in str(excinfo.value).lower()


def test_md_is_cancellable(tmp_path: Path, analytic_forces) -> None:
    """A long run can be stopped."""

    source = _cell(tmp_path, "md")

    analysis = MDAnalysis(ProjectService.load(source), steps=500)

    seen = []

    def progress(message, percent):
        seen.append(message)
        analysis.cancel()

    with pytest.raises(AnalysisCancelled):
        analysis.run(progress=progress)

    assert len(seen) < 50


def test_md_trajectory_round_trips(
    tmp_path: Path,
    analytic_forces,
) -> None:
    """The exported trajectory is readable by ASE."""

    from ase.io import read

    source = _cell(tmp_path, "md")

    result = MDAnalysis(ProjectService.load(source), steps=12).run()

    assert result.frames

    target = tmp_path / "traj.xyz"
    target.write_text(result.to_xyz_trajectory(), encoding="utf-8")

    images = read(str(target), index=":")

    assert len(images) == len(result.frames)
    assert len(images[0]) == result.natoms
    assert images[0].cell.volume > 0


def test_md_csv_has_one_row_per_step(
    tmp_path: Path,
    analytic_forces,
) -> None:
    """Export carries the full time series."""

    source = _cell(tmp_path, "md")

    result = MDAnalysis(ProjectService.load(source), steps=8).run()

    csv = result.to_csv()

    rows = [
        line
        for line in csv.splitlines()
        if line and not line.startswith("#") and line.count(",") == 4
    ]

    # header plus one row per step
    assert len(rows) == 9


# ------------------------------------------------------------------
# Each image carries its own calculator
# ------------------------------------------------------------------


def test_neb_gives_every_image_its_own_calculator(
    tmp_path: Path, monkeypatch
) -> None:
    """
    No two images share a calculator.

    ASE caches a calculator's results against the last atoms it saw, so
    one calculator shared down the band is evicted on every image's turn
    and the endpoints are recomputed every step. Measured on a 47-atom
    cell that came to 11 force evaluations per optimiser step where 2
    were needed -- a run four times longer than it had to be.
    """

    handed = []

    def issue(self):
        calculator = analytic_calculator()
        handed.append(calculator)
        return calculator

    monkeypatch.setattr(NEBAnalysis, "_get_calculator", issue)

    start = _cell(tmp_path, "start", shift=0.0)
    end = _cell(tmp_path, "end", shift=0.5)

    analysis = NEBAnalysis(
        ProjectService.load(start), end, n_images=3, steps=2
    )

    captured = {}

    original = analysis._relax

    def spy(optimiser, band):
        captured["images"] = list(band.images)
        return original(optimiser, band)

    monkeypatch.setattr(analysis, "_relax", spy)

    analysis.run()

    images = captured["images"]

    assert len(images) == 5

    # Distinct objects, one per image.
    assert len({id(image.calc) for image in images}) == len(images)

    # And every one of them came from the hook rather than being
    # conjured somewhere else.
    assert all(image.calc in handed for image in images)


def test_neb_reports_progress_while_relaxing(
    tmp_path: Path, analytic_forces
) -> None:
    """
    The run says what it is doing, step by step.

    A force field on a CPU takes minutes, and a window that says nothing
    for minutes cannot be told apart from one that has hung. The step
    messages are what the console log and the status line are made of.
    """

    start = _cell(tmp_path, "start", shift=0.0)
    end = _cell(tmp_path, "end", shift=0.5)

    messages = []

    NEBAnalysis(
        ProjectService.load(start), end, n_images=2, steps=3
    ).run(progress=lambda message, percent=None: messages.append(message))

    # Something before the force field loads, so the wait for it is
    # not silent.
    assert any("force field" in item.lower() for item in messages)

    steps = [item for item in messages if item.startswith("[")]

    assert steps, "no per-step progress was reported"

    # Each carries the residual and the target it is heading for.
    assert "max|F|" in steps[0]
    assert "target" in steps[0]

    # And the rate, so the remaining wait can be estimated.
    assert "s/step" in steps[0]

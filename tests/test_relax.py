"""
Tests for structure relaxation.

Run against a Lennard-Jones potential rather than the force field: it is
instant, and it has a minimum whose position is known in closed form, so
the relaxation can be asked to find a number decided in advance rather
than merely to produce a plausible one.

What is actually being guarded
------------------------------
The optimiser is ASE's and needs no testing. What needs it is everything
around the optimiser: that "ions only" really does leave the cell alone,
that a relaxation which ran out of steps says so rather than reporting
success, that the displacement of an atom which crossed the periodic
boundary is measured the short way, and that a silent drop in symmetry
is reported. Each of those produces a result that looks entirely normal
when it is wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvista.analyses.base import AnalysisError
from qvista.analyses.relax import (
    CONSTANT_VOLUME,
    IONS_AND_CELL,
    IONS_ONLY,
    ISIF,
    MODES,
    MOVED,
    NEEDS_CELL,
    RelaxAnalysis,
    RelaxResult,
)
from qvista.services.project_service import ProjectService

#: The Lennard-Jones minimum sits at 2^(1/6) sigma.
SIGMA = 2.5
MINIMUM = SIGMA * 2.0 ** (1.0 / 6.0)


def analytic_calculator():
    """A real pair potential, instant to evaluate."""

    from ase.calculators.lj import LennardJones

    return LennardJones(sigma=SIGMA, epsilon=0.05, rc=8.0)


@pytest.fixture
def analytic_forces(monkeypatch):
    """Point the workflow at the analytic potential."""

    monkeypatch.setattr(
        RelaxAnalysis,
        "_get_calculator",
        lambda self: analytic_calculator(),
    )


# ------------------------------------------------------------
# Structures
# ------------------------------------------------------------


def write_cell(tmp_path: Path, name: str, structure) -> Path:
    """Put a structure on disk and open it as a project."""

    from pymatgen.io.vasp.inputs import Poscar

    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "POSCAR"

    Poscar(structure).write_file(str(path))

    return path


def squeezed_dimer(tmp_path, separation: float = 2.0):
    """
    Two atoms closer than they want to be, in a large box.

    The pair sits along x, so the relaxation has one obvious thing to do
    and the answer is the Lennard-Jones minimum.
    """

    from pymatgen.core import Lattice, Structure

    box = 14.0

    structure = Structure(
        Lattice.cubic(box),
        ["Ar", "Ar"],
        [
            [0.5 - separation / (2 * box), 0.5, 0.5],
            [0.5 + separation / (2 * box), 0.5, 0.5],
        ],
    )

    return write_cell(tmp_path, "dimer", structure)


def project_for(path: Path):

    return ProjectService.load(path)


def relax(path, **kwargs):
    """Run a relaxation and hand back the result."""

    analysis = RelaxAnalysis(project_for(path), **kwargs)

    return analysis.run()


# ------------------------------------------------------------
# It actually relaxes
# ------------------------------------------------------------


def test_a_squeezed_pair_finds_the_minimum(
    tmp_path, analytic_forces
) -> None:
    """
    In at 2.0 angstrom, out at 2^(1/6) sigma.

    The one test here that checks the physics rather than the
    bookkeeping around it.
    """

    result = relax(
        squeezed_dimer(tmp_path), mode=IONS_ONLY, fmax=0.005
    )

    positions = result.structure.cart_coords

    separation = float(np.linalg.norm(positions[1] - positions[0]))

    assert separation == pytest.approx(MINIMUM, abs=0.02)


def test_the_energy_goes_down(tmp_path, analytic_forces) -> None:

    result = relax(squeezed_dimer(tmp_path), mode=IONS_ONLY)

    assert result.final_energy < result.initial_energy

    assert result.energy_change < 0.0


def test_the_forces_come_down(tmp_path, analytic_forces) -> None:

    result = relax(
        squeezed_dimer(tmp_path), mode=IONS_ONLY, fmax=0.01
    )

    assert result.final_forces < result.initial_forces

    assert result.final_forces <= 0.01

    assert result.converged


def test_the_trace_has_one_point_per_step(
    tmp_path, analytic_forces
) -> None:

    result = relax(squeezed_dimer(tmp_path), mode=IONS_ONLY)

    assert len(result.energies) == result.steps
    assert len(result.forces) == result.steps

    assert result.steps > 1


# ------------------------------------------------------------
# What may move
# ------------------------------------------------------------


def test_ions_only_leaves_the_cell_exactly_alone(
    tmp_path, analytic_forces
) -> None:
    """
    Not approximately: exactly.

    A cell that drifts under "ions only" invalidates every lattice
    constant the user chose, and a per-cent change is invisible unless
    it is looked for.
    """

    path = squeezed_dimer(tmp_path)

    before = project_for(path).structure

    result = relax(path, mode=IONS_ONLY)

    assert result.final_volume == pytest.approx(result.initial_volume)

    assert np.allclose(
        result.structure.lattice.matrix,
        before.lattice.matrix if hasattr(before, "lattice") else
        result.structure.lattice.matrix,
    )


def test_relaxing_the_cell_is_allowed_to_change_the_volume(
    tmp_path, analytic_forces
) -> None:
    """
    Two atoms in a large box under a pair potential: the cell collapses
    towards them, which is the correct behaviour and worth seeing.
    """

    result = relax(
        squeezed_dimer(tmp_path), mode=IONS_AND_CELL, max_steps=40
    )

    assert result.final_volume != pytest.approx(result.initial_volume)

    assert result.volume_change != pytest.approx(0.0)


def test_a_large_volume_change_is_called_out(
    tmp_path, analytic_forces
) -> None:

    result = relax(
        squeezed_dimer(tmp_path), mode=IONS_AND_CELL, max_steps=40
    )

    if abs(result.volume_change) > 0.05:
        assert any("volume changed" in note for note in result.notes)


def test_an_unknown_mode_is_refused(tmp_path, analytic_forces) -> None:

    with pytest.raises(AnalysisError, match="Unknown relaxation mode"):
        relax(squeezed_dimer(tmp_path), mode="springs")


def test_a_negative_force_target_is_refused(
    tmp_path, analytic_forces
) -> None:

    with pytest.raises(AnalysisError, match="must be positive"):
        relax(squeezed_dimer(tmp_path), fmax=0.0)


def test_zero_steps_is_refused(tmp_path, analytic_forces) -> None:

    with pytest.raises(AnalysisError, match="at least one step"):
        relax(squeezed_dimer(tmp_path), max_steps=0)


# ------------------------------------------------------------
# Not converging
# ------------------------------------------------------------


def test_running_out_of_steps_is_not_success(
    tmp_path, analytic_forces
) -> None:
    """
    The failure that looks exactly like a success.

    A relaxation stopped by the step cap returns a structure, an energy
    and a trace like any other. Nothing about it says the forces are
    still large except this flag.
    """

    result = relax(
        squeezed_dimer(tmp_path, separation=1.6),
        mode=IONS_ONLY,
        fmax=1e-9,
        max_steps=2,
    )

    assert not result.converged

    assert any("stopped at" in note for note in result.notes)


def test_the_row_says_so_too(tmp_path, analytic_forces) -> None:
    """Console output has to carry it as well as the notes."""

    result = relax(
        squeezed_dimer(tmp_path, separation=1.6),
        mode=IONS_ONLY,
        fmax=1e-9,
        max_steps=2,
    )

    rows = dict(result.rows)

    assert rows["Converged"].startswith("NO")


# ------------------------------------------------------------
# Displacements
# ------------------------------------------------------------


def test_an_atom_crossing_the_boundary_moved_a_little(tmp_path) -> None:
    """
    Measured the short way round.

    An atom relaxing across the cell edge goes from 0.99 to 0.01 in
    fractional coordinates. Subtracting those directly says it crossed
    the whole cell, and the reported displacement becomes a lattice
    vector instead of a hundredth of an angstrom.
    """

    from pymatgen.core import Lattice, Structure

    before = Structure(
        Lattice.cubic(10.0), ["Ar"], [[0.99, 0.5, 0.5]]
    )
    after = Structure(
        Lattice.cubic(10.0), ["Ar"], [[0.01, 0.5, 0.5]]
    )

    analysis = RelaxAnalysis.__new__(RelaxAnalysis)
    analysis.project = None
    analysis.mode = IONS_ONLY

    # _periodic reads the project; stub it for this one measurement.
    analysis._periodic = lambda: True

    moved = analysis._displacements(before, after, None, None)

    assert moved[0] == pytest.approx(0.2, abs=1e-6)


def test_atoms_that_moved_are_counted(
    tmp_path, analytic_forces
) -> None:

    result = relax(squeezed_dimer(tmp_path), mode=IONS_ONLY)

    assert result.max_displacement > MOVED

    assert result.moved_atoms == 2


# ------------------------------------------------------------
# What the result says about itself
# ------------------------------------------------------------


def test_energy_per_atom_divides_by_the_atom_count() -> None:

    result = RelaxResult(
        initial_energy=-10.0, final_energy=-14.0, natoms=4
    )

    assert result.energy_change == pytest.approx(-4.0)
    assert result.energy_per_atom == pytest.approx(-1.0)


def test_energy_per_atom_survives_an_empty_result() -> None:

    assert RelaxResult().energy_per_atom == 0.0
    assert RelaxResult().volume_change == 0.0


def test_a_drop_in_symmetry_is_recognised() -> None:
    """
    An optimiser does not know about symmetry.

    Noise in the forces breaks it, and a structure that went in as R-3m
    can come out as P1 with every atom moved by a thousandth of an
    angstrom. Nothing looks wrong; the next calculation costs
    forty-eight times as much.
    """

    result = RelaxResult(
        spacegroup_number_before=166, spacegroup_number_after=1
    )

    assert result.lost_symmetry


def test_keeping_the_symmetry_is_not_a_drop() -> None:

    result = RelaxResult(
        spacegroup_number_before=166, spacegroup_number_after=166
    )

    assert not result.lost_symmetry


def test_a_p1_structure_cannot_lose_symmetry() -> None:
    """There is none to lose, so the warning would be noise."""

    result = RelaxResult(
        spacegroup_number_before=1, spacegroup_number_after=1
    )

    assert not result.lost_symmetry


def test_the_result_names_the_force_field(
    tmp_path, analytic_forces
) -> None:
    """So a geometry can never be attributed to the wrong method."""

    result = relax(squeezed_dimer(tmp_path), mode=IONS_ONLY)

    assert result.force_field

    assert any("not DFT" in note for note in result.notes)


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------


def test_the_relaxed_structure_writes_a_poscar(
    tmp_path, analytic_forces
) -> None:

    result = relax(squeezed_dimer(tmp_path), mode=IONS_ONLY)

    text = result.to_poscar()

    assert "Ar" in text

    # Header, scale, three lattice vectors, species, counts, mode, 2 atoms.
    assert len(text.splitlines()) >= 10


def test_an_empty_result_writes_no_poscar() -> None:

    assert RelaxResult().to_poscar() == ""


def test_the_trace_exports_as_csv(tmp_path, analytic_forces) -> None:

    result = relax(squeezed_dimer(tmp_path), mode=IONS_ONLY)

    text = result.to_csv()

    assert "max_force_ev_per_ang" in text

    assert text.count("\n") > result.steps


def test_the_path_downhill_exports_as_a_trajectory(
    tmp_path, analytic_forces
) -> None:

    result = relax(squeezed_dimer(tmp_path), mode=IONS_ONLY)

    assert result.to_xyz_trajectory().startswith("2\n")


# ------------------------------------------------------------
# ISIF 4: cell shape free, volume held
#
# The third mode, and the one that is easy to overlook. It is how a
# structure is relaxed at a fixed density -- along an equation of state,
# or at a volume chosen elsewhere -- and it differs from ISIF 3 by a
# single flag on the cell filter. Without that flag the two are the same
# calculation, and nothing about the result says so.
# ------------------------------------------------------------


def test_three_modes_are_offered() -> None:

    assert set(MODES) == {IONS_ONLY, IONS_AND_CELL, CONSTANT_VOLUME}


def test_they_map_onto_vasps_isif() -> None:
    """
    The same choice as in an INCAR, so it can be named the same way.

    2 holds the cell, 3 frees it, 4 frees the shape and holds the
    volume.
    """

    assert ISIF[IONS_ONLY] == 2
    assert ISIF[IONS_AND_CELL] == 3
    assert ISIF[CONSTANT_VOLUME] == 4


def test_only_the_cell_modes_need_a_lattice() -> None:

    assert set(NEEDS_CELL) == {IONS_AND_CELL, CONSTANT_VOLUME}

    assert IONS_ONLY not in NEEDS_CELL


def test_isif_four_holds_the_volume(tmp_path, analytic_forces) -> None:
    """
    The whole content of the mode.

    Two atoms in a large box under a pair potential pull the cell in
    hard when it is free, so this is a case where a missing constraint
    would be obvious in the number.
    """

    result = relax(
        squeezed_dimer(tmp_path),
        mode=CONSTANT_VOLUME,
        max_steps=40,
    )

    assert result.volume_change == pytest.approx(0.0, abs=1e-6)


def test_isif_three_does_not(tmp_path, analytic_forces) -> None:
    """The control: same structure, same steps, free volume."""

    result = relax(
        squeezed_dimer(tmp_path),
        mode=IONS_AND_CELL,
        max_steps=40,
    )

    assert abs(result.volume_change) > 1e-6


def test_isif_four_still_moves_the_atoms(tmp_path, analytic_forces) -> None:
    """A held volume is not a held structure."""

    result = relax(
        squeezed_dimer(tmp_path),
        mode=CONSTANT_VOLUME,
        max_steps=40,
    )

    assert result.max_displacement > 0.0

    assert result.final_forces < result.initial_forces


def test_a_volume_that_moves_under_isif_four_is_reported() -> None:
    """
    If the constraint fails, the number is the only sign.

    A result built by hand here, because making ASE break its own
    constraint is not something a test should be arranging.
    """

    result = RelaxResult(
        mode=CONSTANT_VOLUME,
        initial_volume=100.0,
        final_volume=103.0,
    )

    analysis = RelaxAnalysis.__new__(RelaxAnalysis)

    notes = analysis._notes(result)

    assert any("constraint did not hold" in note for note in notes)


def test_isif_four_on_a_molecule_is_refused(tmp_path) -> None:
    """There is no cell to hold the volume of."""

    path = tmp_path / "mol.xyz"

    path.write_text(
        "2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8"
    )

    analysis = RelaxAnalysis(
        project_for(path), mode=CONSTANT_VOLUME
    )

    with pytest.raises(AnalysisError, match="no cell to relax"):
        analysis.run()


def test_the_mode_is_named_in_the_rows(tmp_path, analytic_forces) -> None:
    """So a geometry can be traced back to how it was relaxed."""

    result = relax(squeezed_dimer(tmp_path), mode=CONSTANT_VOLUME)

    rows = dict(result.rows)

    assert rows["Equivalent to"] == "ISIF = 4"

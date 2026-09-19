"""
Runs that outlive the window they were made in.

Both force-field workflows used to keep their result in the session and
nowhere else: close the window and the only copy went with it, and the
analyses that open on a finished run had nothing to open on. These tests
pin the two round trips -- write a run out, read it back, get the same
numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvista.analyses.md import MDResult
from qvista.services import md_vasp


# ------------------------------------------------------------------
# Molecular dynamics
# ------------------------------------------------------------------


def _run(frames: int = 12, spacing: int = 10) -> MDResult:
    """A small run with deliberately interleaved species."""

    symbols = ["Li", "O", "Li", "O", "Co", "Li"]

    rng = np.random.default_rng(0)

    start = rng.random((len(symbols), 3)) * 10.0

    steps = frames * spacing

    return MDResult(
        frames=[start + 0.02 * index for index in range(frames)],
        symbols=symbols,
        cell=np.diag([10.0, 10.0, 10.0]),
        ensemble="nvt",
        temperature=600.0,
        timestep=2.0,
        steps=steps,
        record_every=spacing,
        natoms=len(symbols),
        force_field="CHGNet (machine-learning force field - not DFT)",
        times=[2.0 * (index + 1) for index in range(steps)],
        potential_energies=[-1.0] * steps,
        kinetic_energies=[0.5] * steps,
        total_energies=[-0.5] * steps,
        temperatures=[600.0] * steps,
        log=["[1/120] T =   600.0 K"],
    )


def test_a_saved_run_reads_back(tmp_path: Path) -> None:
    """The whole point: write it out, open it again, same trajectory."""

    result = _run()

    saved = md_vasp.save_run(result, tmp_path / "run", plots=False)

    assert saved.count

    reopened = md_vasp.read_run(saved.root, with_trajectory=True)

    assert len(reopened.frames) == len(result.frames)
    assert reopened.natoms == result.natoms


def test_the_time_axis_survives(tmp_path: Path) -> None:
    """
    A reopened run is as long as the one that was written.

    The frames are one per ``record_every`` steps, so the gap between
    them is that times the timestep. Read as if it were one timestep,
    every diffusion coefficient comes back too large by exactly that
    factor -- and nothing about the curve looks wrong.
    """

    result = _run(frames=12, spacing=10)

    saved = md_vasp.save_run(result, tmp_path / "run", plots=False)

    reopened = md_vasp.read_run(saved.root, with_trajectory=True)

    assert reopened.potim == pytest.approx(2.0)
    assert reopened.nblock == 10

    # 2 fs between steps, 10 steps between frames, in picoseconds.
    assert reopened.trajectory.timestep == pytest.approx(0.02)

    # 12 frames at 20 fs apart.
    assert reopened.simulated_time == pytest.approx(240.0)


def test_species_are_grouped_for_vasp(tmp_path: Path) -> None:
    """
    POSCAR names each species once, so the atoms must arrive grouped.

    An ASE trajectory is under no such obligation. Writing one out in
    its own order keeps the counts adding up while silently relabelling
    every atom, and the species-resolved analyses afterwards answer
    about the wrong element.
    """

    result = _run()

    saved = md_vasp.save_run(result, tmp_path / "run", plots=False)

    reopened = md_vasp.read_run(saved.root, with_trajectory=True)

    symbols = list(reopened.symbols)

    # Same composition as went in.
    assert sorted(symbols) == sorted(result.symbols)

    # And each species in one contiguous block.
    blocks = [symbols[0]]

    for symbol in symbols[1:]:
        if symbol != blocks[-1]:
            blocks.append(symbol)

    assert len(blocks) == len(set(blocks)), symbols


def test_positions_without_energies_still_open(tmp_path: Path) -> None:
    """
    No OSZICAR is not a reason to refuse the folder.

    A run relaxed here never involved VASP, so there is no OSZICAR to
    write and none to invent. Everything anyone opens this for --
    diffusion, RDF, Van Hove, residence -- reads positions.
    """

    result = _run()

    saved = md_vasp.save_run(result, tmp_path / "run", plots=False)

    assert not (saved.root / "OSZICAR").exists()
    assert not (saved.root / "vasprun.xml").exists()

    reopened = md_vasp.read_run(saved.root, with_trajectory=True)

    assert reopened.positions_only
    assert len(reopened.frames) == len(result.frames)

    # Said plainly, rather than left to be discovered.
    assert any("no energy traces" in note for note in reopened.notes)

    # And not called truncated, which it is not.
    assert not any("stopped after" in note for note in reopened.notes)


def test_nothing_claims_to_be_vasp_output(tmp_path: Path) -> None:
    """
    Inputs may be written; output may not be invented.

    An INCAR is a statement of the settings that were used, which is
    true. An OSZICAR would be a claim that VASP produced these numbers,
    which it did not.
    """

    saved = md_vasp.save_run(_run(), tmp_path / "run", plots=False)

    names = {path.name for path in saved.written}

    assert "INCAR" in names
    assert "XDATCAR" in names

    assert "OSZICAR" not in names
    assert "OUTCAR" not in names
    assert "vasprun.xml" not in names

    # And the file says so itself.
    readme = (saved.root / "README.txt").read_text(encoding="utf-8")

    assert "VASP did not run" in readme


def test_a_missing_piece_costs_only_that_piece(tmp_path: Path) -> None:
    """One unwritable thing does not lose the rest of the run."""

    result = _run()

    result.frames = []  # nothing to write a trajectory from

    saved = md_vasp.save_run(result, tmp_path / "run", plots=False)

    assert saved.skipped

    # The time series does not depend on the frames, so it survives.
    assert any(path.name == "md.csv" for path in saved.written)


def test_the_log_is_kept(tmp_path: Path) -> None:
    """How the run went is written beside what it produced."""

    saved = md_vasp.save_run(_run(), tmp_path / "run", plots=False)

    log = saved.root / "md.log"

    assert log.is_file()
    assert "600.0 K" in log.read_text(encoding="utf-8")


# ------------------------------------------------------------------
# NEB
# ------------------------------------------------------------------


@pytest.fixture
def analytic_forces(monkeypatch):
    """A real pair potential, instant to evaluate."""

    from ase.calculators.lj import LennardJones

    from qvista.analyses.neb import NEBAnalysis

    monkeypatch.setattr(
        NEBAnalysis,
        "_get_calculator",
        lambda self: LennardJones(sigma=2.5, epsilon=0.05, rc=7.0),
    )


def test_a_saved_neb_reopens_with_the_same_barrier(
    tmp_path: Path, analytic_forces
) -> None:
    """
    A relaxed band written out and read back gives the same barrier.

    The force-field route produces no OUTCARs, so the energies travel
    in neb.dat. Reading them back is what lets a run that took minutes
    be reopened through Post-Processing instead of repeated.
    """

    from qvista.services import neb_vasp
    from qvista.services.project_service import ProjectService

    from tests.test_dynamics import _cell

    start = _cell(tmp_path, "start", shift=0.0)
    end = _cell(tmp_path, "end", shift=0.5)

    from qvista.analyses.neb import NEBAnalysis

    result = NEBAnalysis(
        ProjectService.load(start), end, n_images=3, steps=5
    ).run()

    saved = neb_vasp.save_run(
        result, tmp_path / "saved", plots=False, trajectory=False
    )

    assert (saved.root / "neb.dat").is_file()

    reopened = neb_vasp.read_path(saved.root)

    assert reopened.barrier_forward == pytest.approx(
        result.barrier_forward, abs=1e-6
    )
    assert reopened.saddle_index == result.saddle_index


def test_neb_dat_is_only_used_when_there_is_nothing_else(
    tmp_path: Path,
) -> None:
    """
    Its energies are relative; an OUTCAR's are absolute.

    Filling a gap in one scale from the other would mix them, so the
    fallback is all-or-nothing.
    """

    from qvista.services import neb_vasp

    root = tmp_path / "mixed"

    rows = neb_vasp.read_neb_dat(root)

    assert rows == []

    root.mkdir()

    (root / "neb.dat").write_text(
        "# image distance energy force\n"
        "    0       0.000      0.000      0.000\n"
        "    1       0.500      0.250     -0.100\n"
        "    2       1.000      0.000      0.000\n",
        encoding="utf-8",
    )

    rows = neb_vasp.read_neb_dat(root)

    assert [row[0] for row in rows] == [0, 1, 2]
    assert rows[1][2] == pytest.approx(0.25)

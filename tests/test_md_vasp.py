"""
Tests for reading a finished VASP molecular-dynamics run.

The parsing itself is the easy part. What is pinned here is the handful
of places where a wrong answer still looks like a right one:

* ``NBLOCK``. XDATCAR holds every NBLOCK-th step, so the gap between
  frames is ``POTIM * NBLOCK``. Using POTIM alone scales every diffusion
  coefficient by NBLOCK and changes nothing about the shape of the curve,
  so no plot looks wrong.

* The **ensemble**. VASP records MDALGO and SMASS, not "NVE" or "NVT".
  The energy-drift diagnostic is only meaningful in NVE, so misreading
  this either hides a real integration error or invents a fake one.

* VASP's **float format**. ``-.295E+03`` has no leading digit, and a
  regex written against ``-0.295E+03`` silently matches nothing, leaving
  a run that reads as empty rather than as broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services import md_vasp


# ------------------------------------------------------------
# Building a run to read
# ------------------------------------------------------------


def write_oszicar(
    path,
    steps: int = 20,
    temperature: float = 300.0,
    potential: float = -300.0,
    natoms: int = 8,
    drift: float = 0.0,
    bare_exponent: bool = True,
    with_ek: bool = True,
):
    """
    Write an OSZICAR the way VASP writes one for an MD run.

    ``bare_exponent`` reproduces VASP's habit of dropping the leading
    zero, which is the format the parser actually meets.
    """

    def number(value: float) -> str:

        text = f"{value:.8E}"

        if bare_exponent and text.startswith("-0."):
            return "-." + text[3:]

        if bare_exponent and text.startswith("0."):
            return "." + text[2:]

        return text

    lines = []

    for index in range(1, steps + 1):

        kinetic = 1.5 * natoms * 8.617333262e-5 * temperature

        energy = potential + kinetic + drift * index

        lines.append(
            f"DAV:   1    {number(potential)}   {number(0.0)}"
        )

        row = (
            f"{index:6d} T= {temperature:8.1f} "
            f"E= {number(energy)} "
            f"F= {number(potential)} "
            f"E0= {number(potential)}"
        )

        if with_ek:
            row += f"  EK= {number(kinetic)} SP= 0.00E+00 SK= 0.00E+00"

        lines.append(row)

    (path / "OSZICAR").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_poscar(path, natoms: int = 8, cell: float = 10.0) -> None:
    """
    The starting structure.

    Present in every real run, and the cheapest place to learn how many
    atoms there are -- which the per-atom diagnostics need even when the
    trajectory is not being parsed.
    """

    lines = [
        "test",
        "   1.0",
        f"  {cell} 0.0 0.0",
        f"  0.0 {cell} 0.0",
        f"  0.0 0.0 {cell}",
        "   Li   O",
        f"    {natoms // 2}    {natoms - natoms // 2}",
        "Direct",
    ]

    lines.extend("  0.0  0.0  0.0" for _ in range(natoms))

    (path / "POSCAR").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_incar(path, **tags) -> None:
    """Write an INCAR, with a decoy value hidden inside a comment."""

    settings = {
        "IBRION": 0,
        "NSW": 20,
        "POTIM": 2.0,
        "TEBEG": 300,
        "MDALGO": 2,
        "SMASS": 0.5,
    }

    settings.update(tags)

    lines = [f"{key} = {value}" for key, value in settings.items()]

    # A tag that must NOT be picked up: it is commented out.
    lines.append("# TEEND = 9999")
    lines.append("ISYM = 0    ! NBLOCK = 777")

    (path / "INCAR").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_xdatcar(
    path, frames: int = 20, natoms: int = 8, cell: float = 10.0, step=0.01
):
    """A cubic cell with atoms taking a small random walk."""

    generator = np.random.default_rng(0)

    header = [
        "test",
        "   1.0",
        f"  {cell} 0.0 0.0",
        f"  0.0 {cell} 0.0",
        f"  0.0 0.0 {cell}",
        "   Li   O",
        f"    {natoms // 2}    {natoms - natoms // 2}",
    ]

    body = []

    positions = generator.random((natoms, 3))

    for index in range(1, frames + 1):

        positions = positions + generator.normal(0.0, step, (natoms, 3))

        body.append(f"Direct configuration=  {index:5d}")

        for row in positions % 1.0:
            body.append("  %.8f  %.8f  %.8f" % tuple(row))

    (path / "XDATCAR").write_text(
        "\n".join(header + body) + "\n", encoding="utf-8"
    )


def make_run(path, steps=20, frames=20, **tags):
    """A complete run: INCAR, OSZICAR and XDATCAR."""

    write_poscar(path)
    write_incar(path, **tags)
    write_oszicar(path, steps=steps)
    write_xdatcar(path, frames=frames)

    return path


# ------------------------------------------------------------
# Parsing OSZICAR
# ------------------------------------------------------------


def test_every_step_is_found(tmp_path) -> None:

    run = md_vasp.read_run(make_run(tmp_path, steps=25, frames=25))

    assert run.steps == 25


def test_vasps_bare_exponent_format_parses(tmp_path) -> None:
    """``-.295E+03``, which is what VASP actually writes."""

    write_incar(tmp_path)
    write_oszicar(tmp_path, steps=5, bare_exponent=True)

    run = md_vasp.read_run(tmp_path, with_trajectory=False)

    assert run.steps == 5

    assert run.potential_energies[0] == pytest.approx(-300.0)


def test_the_ordinary_format_parses_too(tmp_path) -> None:

    write_incar(tmp_path)
    write_oszicar(tmp_path, steps=5, bare_exponent=False)

    run = md_vasp.read_run(tmp_path, with_trajectory=False)

    assert run.steps == 5

    assert run.potential_energies[0] == pytest.approx(-300.0)


def test_kinetic_energy_falls_back_to_the_difference(tmp_path) -> None:
    """
    Not every VASP build prints EK.

    ``E - F`` is the same quantity by definition, so the traces stay
    complete rather than showing a flat zero.
    """

    write_incar(tmp_path)
    write_oszicar(tmp_path, steps=5, with_ek=False)

    run = md_vasp.read_run(tmp_path, with_trajectory=False)

    assert run.kinetic_energies[0] > 0.0

    assert run.kinetic_energies[0] == pytest.approx(
        run.total_energies[0] - run.potential_energies[0]
    )


def test_the_time_axis_uses_potim(tmp_path) -> None:

    run = md_vasp.read_run(
        make_run(tmp_path, steps=10, frames=10, POTIM=3.0),
        with_trajectory=False,
    )

    assert run.times[0] == pytest.approx(3.0)
    assert run.simulated_time == pytest.approx(30.0)


# ------------------------------------------------------------
# INCAR
# ------------------------------------------------------------


def test_a_commented_tag_is_not_read(tmp_path) -> None:
    """
    ``# TEEND = 9999`` must not become the ramp temperature.

    Reading commented tags is the classic INCAR-parsing bug, and here it
    would silently claim the run ramped when it held temperature.
    """

    run = md_vasp.read_run(make_run(tmp_path), with_trajectory=False)

    assert run.teend is None


def test_a_trailing_comment_is_stripped(tmp_path) -> None:
    """``ISYM = 0    ! NBLOCK = 777`` must not set NBLOCK."""

    run = md_vasp.read_run(make_run(tmp_path), with_trajectory=False)

    assert run.nblock == 1


def test_a_missing_incar_is_survivable(tmp_path) -> None:
    """The traces are still readable; only the labels are poorer."""

    write_oszicar(tmp_path, steps=6)

    run = md_vasp.read_run(tmp_path, with_trajectory=False)

    assert run.steps == 6
    assert run.potim == 1.0


# ------------------------------------------------------------
# Which ensemble ran
# ------------------------------------------------------------


def test_smass_minus_three_is_nve(tmp_path) -> None:

    run = md_vasp.read_run(
        make_run(tmp_path, SMASS=-3, MDALGO=0), with_trajectory=False
    )

    assert run.ensemble == "nve"
    assert "NVE" in run.ensemble_label


def test_nose_hoover_is_nvt(tmp_path) -> None:

    run = md_vasp.read_run(
        make_run(tmp_path, MDALGO=2, SMASS=0.5), with_trajectory=False
    )

    assert run.ensemble == "nvt"
    assert "Nose-Hoover" in run.ensemble_label


def test_langevin_is_nvt(tmp_path) -> None:

    run = md_vasp.read_run(
        make_run(tmp_path, MDALGO=3, SMASS=0.5), with_trajectory=False
    )

    assert run.ensemble == "nvt"
    assert "Langevin" in run.ensemble_label


def test_smass_beats_mdalgo(tmp_path) -> None:
    """``SMASS = -3`` is microcanonical whatever MDALGO says."""

    run = md_vasp.read_run(
        make_run(tmp_path, MDALGO=2, SMASS=-3), with_trajectory=False
    )

    assert run.ensemble == "nve"


def test_drift_is_only_a_warning_in_nve(tmp_path) -> None:
    """
    A thermostat is supposed to exchange energy.

    Warning about drift under NVT would train the user to ignore the
    warning that matters.
    """

    write_poscar(tmp_path, natoms=8)
    write_incar(tmp_path, MDALGO=2, SMASS=0.5)
    write_oszicar(tmp_path, steps=20, drift=0.05)

    thermostatted = md_vasp.read_run(tmp_path, with_trajectory=False)

    assert not any("drifted" in note for note in thermostatted.notes)

    write_incar(tmp_path, MDALGO=0, SMASS=-3)

    microcanonical = md_vasp.read_run(tmp_path, with_trajectory=False)

    assert any("drifted" in note for note in microcanonical.notes)


# ------------------------------------------------------------
# The trajectory, and NBLOCK
# ------------------------------------------------------------


def test_the_trajectory_is_read(tmp_path) -> None:

    run = md_vasp.read_run(make_run(tmp_path, frames=20))

    assert run.has_trajectory
    assert len(run.frames) == 20
    assert run.symbols[0] == "Li"


def test_nblock_stretches_the_frame_spacing(tmp_path) -> None:
    """
    The one that changes a diffusion coefficient without changing a plot.

    XDATCAR holds every NBLOCK-th step, so with NBLOCK = 5 and
    POTIM = 2 fs the frames are 10 fs apart, not 2.
    """

    make_run(tmp_path, NBLOCK=5, POTIM=2.0)

    run = md_vasp.read_run(tmp_path)

    # read_xdatcar works in picoseconds.
    assert run.trajectory.timestep == pytest.approx(0.010)


def test_nblock_defaults_to_one(tmp_path) -> None:

    make_run(tmp_path, POTIM=2.0)

    run = md_vasp.read_run(tmp_path)

    assert run.trajectory.timestep == pytest.approx(0.002)


def test_a_missing_xdatcar_is_reported_not_fatal(tmp_path) -> None:

    write_incar(tmp_path)
    write_oszicar(tmp_path, steps=10)

    run = md_vasp.read_run(tmp_path)

    assert not run.has_trajectory

    assert any("No XDATCAR" in note for note in run.notes)
    assert any("no diffusion" in note for note in run.notes)


# ------------------------------------------------------------
# Runs that are not what they claim
# ------------------------------------------------------------


def test_an_empty_folder_says_what_to_do(tmp_path) -> None:

    with pytest.raises(md_vasp.MdVaspError, match="No OSZICAR"):
        md_vasp.read_run(tmp_path)


def test_a_missing_folder_is_refused(tmp_path) -> None:

    with pytest.raises(md_vasp.MdVaspError, match="no such folder"):
        md_vasp.read_run(tmp_path / "nowhere")


def test_a_relaxation_is_not_dynamics(tmp_path) -> None:
    """
    One ionic step is a static run.

    This is exactly what the XDATCAR in the user's Analysis folder is,
    so the message has to say what is wrong rather than fail on an
    empty array later.
    """

    write_incar(tmp_path)
    write_oszicar(tmp_path, steps=1)

    with pytest.raises(md_vasp.MdVaspError, match="IBRION = 0"):
        md_vasp.read_run(tmp_path)


def test_a_run_cut_short_is_flagged(tmp_path) -> None:
    """The common case: the wall clock ended it."""

    write_incar(tmp_path, NSW=1000)
    write_oszicar(tmp_path, steps=40)

    run = md_vasp.read_run(tmp_path, with_trajectory=False)

    assert any("40 of the 1000" in note for note in run.notes)


def test_a_short_run_is_called_out(tmp_path) -> None:

    run = md_vasp.read_run(
        make_run(tmp_path, steps=10, frames=10), with_trajectory=False
    )

    assert any("too short" in note for note in run.notes)


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------


def test_the_expected_temperature_spread_follows_equipartition(
    tmp_path,
) -> None:
    """
    ``sigma_T = T * sqrt(2 / 3N)``.

    Worth pinning because it is the only thing that says whether a
    thermostat is over-damped, and it is easy to write with N where 3N
    belongs.
    """

    write_incar(tmp_path)
    write_oszicar(tmp_path, steps=10, temperature=600.0, natoms=8)
    write_xdatcar(tmp_path, frames=10, natoms=8)

    run = md_vasp.read_run(tmp_path)

    assert run.natoms == 8

    assert run.expected_spread == pytest.approx(
        600.0 * np.sqrt(2.0 / 24.0)
    )


def test_a_flat_temperature_is_called_over_damped(tmp_path) -> None:
    """Every step at exactly the target is not physics."""

    make_run(tmp_path)

    run = md_vasp.read_run(tmp_path)

    assert run.temperature_spread == pytest.approx(0.0)

    assert any("over-damped" in note for note in run.notes)


def test_energy_drift_is_per_atom(tmp_path) -> None:

    write_poscar(tmp_path, natoms=8)
    write_incar(tmp_path, MDALGO=0, SMASS=-3)
    write_oszicar(tmp_path, steps=11, natoms=8, drift=0.008)

    run = md_vasp.read_run(tmp_path, with_trajectory=False)

    # Ten intervals of 0.008 eV over 8 atoms.
    assert run.energy_drift == pytest.approx(0.08 / 8, rel=1e-6)


# ------------------------------------------------------------
# Into the shared result
# ------------------------------------------------------------


def test_the_result_is_an_md_result(tmp_path) -> None:
    """So the CHGNet figure and CSV serve this route unchanged."""

    from qvista.analyses.md import MDResult

    result = md_vasp.to_result(md_vasp.read_run(make_run(tmp_path)))

    assert isinstance(result, MDResult)

    assert len(result.times) == len(result.temperatures)
    assert len(result.times) == len(result.total_energies)

    assert result.natoms == 8


def test_the_result_says_it_came_from_vasp(tmp_path) -> None:

    result = md_vasp.to_result(md_vasp.read_run(make_run(tmp_path)))

    assert "VASP" in result.force_field


def test_the_result_exports_csv(tmp_path) -> None:

    result = md_vasp.to_result(md_vasp.read_run(make_run(tmp_path)))

    text = result.to_csv()

    assert "VASP" in text

    # A header line plus one row per step.
    assert text.count("\n") > result.steps


def test_the_result_exports_a_trajectory(tmp_path) -> None:

    result = md_vasp.to_result(md_vasp.read_run(make_run(tmp_path)))

    xyz = result.to_xyz_trajectory()

    assert xyz.startswith("8\n")


# ------------------------------------------------------------
# Where the run came from
#
# A VASP run read off disk becomes the same MDResult as one run inside
# QVista, and is kept in the same place. That sharing is deliberate --
# one figure, one CSV, one exporter for both -- but it leaves nothing to
# tell them apart, and the analyses under MD > CHGNet must not open on a
# VASP trajectory and label the figure with the wrong method.
# ------------------------------------------------------------


def test_a_vasp_run_says_it_came_from_vasp(tmp_path) -> None:

    from qvista.analyses.md import ORIGIN_VASP

    run = md_vasp.read_run(make_run(tmp_path, steps=25, frames=25))

    assert md_vasp.to_result(run).origin == ORIGIN_VASP


def test_a_run_made_here_says_so_by_default() -> None:
    """
    The default matters as much as the flag.

    MDResult is constructed in the force-field path without naming an
    origin, so a default of "vasp" would mislabel every CHGNet run.
    """

    from qvista.analyses.md import ORIGIN_FORCEFIELD, MDResult

    assert MDResult().origin == ORIGIN_FORCEFIELD


def test_the_two_origins_are_different() -> None:

    from qvista.analyses.md import ORIGIN_FORCEFIELD, ORIGIN_VASP

    assert ORIGIN_FORCEFIELD != ORIGIN_VASP


def test_the_origin_survives_storage(tmp_path) -> None:
    """It is read back out of the project, not off the result."""

    from qvista.analyses.md import ORIGIN_VASP

    result = md_vasp.to_result(md_vasp.read_run(make_run(tmp_path, steps=25, frames=25)))

    assert result.as_storage()["result"].origin == ORIGIN_VASP

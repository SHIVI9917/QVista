"""
Molecular dynamics from a finished VASP run.

:mod:`qvista.analyses.md` integrates the equations of motion itself, with
forces from a machine-learning potential, and answers in seconds. This
module reads a run that has already happened. Nothing here launches VASP:
write the input with VASP Input > INCAR > AIMD,
run it, and bring the folder back.

Where the numbers come from
---------------------------
``OSZICAR`` carries one line per ionic step, and that line has everything
the traces need::

    1 T=  300. E= -.295E+03 F= -.296E+03 E0= -.296E+03 EK= 0.10E+01 ...

    T    instantaneous temperature, K
    F    potential energy (the free energy of the electrons), eV
    EK   kinetic energy of the ions, eV
    E    F + EK, the conserved quantity in NVE

``XDATCAR`` carries the positions, and is what a diffusion coefficient is
computed from -- see :mod:`qvista.services.msd`. ``vasprun.xml`` has both
but is far slower to parse for a long run, so it is only consulted when
OSZICAR is missing.

Reading OSZICAR rather than vasprun.xml is deliberate: an AIMD run that
was killed by a wall clock -- which is most of them -- leaves a truncated
vasprun.xml that will not parse at all, while its OSZICAR is complete up
to the last step written. The common case is the one that has to work.

Which ensemble ran
------------------
VASP does not record "NVE" or "NVT" anywhere; it records ``MDALGO`` and
``SMASS``, and the ensemble follows from them. Getting this wrong matters
because the diagnostics differ: in NVE a drifting total energy means the
timestep is too long, whereas in NVT the thermostat is *supposed* to
exchange energy and drift says nothing at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class MdVaspError(Exception):
    """Raised when a run cannot be read, with a reason a user can act on."""


# ------------------------------------------------------------
# Ensembles
# ------------------------------------------------------------

#: ``MDALGO`` values VASP understands, and what each one samples.
MDALGO_LABELS = {
    0: ("nve", "NVE (no thermostat)"),
    1: ("nvt", "NVT (Andersen thermostat)"),
    2: ("nvt", "NVT (Nose-Hoover thermostat)"),
    3: ("nvt", "NVT (Langevin thermostat)"),
    4: ("nvt", "NVT (multiple Andersen)"),
    5: ("nvt", "NVT (Langevin, cell shape fixed)"),
}

#: ``SMASS = -3`` is VASP's way of asking for microcanonical dynamics.
SMASS_NVE = -3

#: Files looked for, in the order they are preferred.
OSZICAR = "OSZICAR"
XDATCAR = "XDATCAR"
VASPRUN = "vasprun.xml"
INCAR = "INCAR"

#: Below this many steps there is nothing to average over.
MIN_STEPS = 2

#: Energy drift above this, in meV per atom, is called out in NVE.
DRIFT_WARNING = 1.0


# ------------------------------------------------------------
# Parsing OSZICAR
# ------------------------------------------------------------

#: VASP writes floats as ``-.295E+03`` as often as ``-0.295E+03``, so the
#: leading digit cannot be assumed.
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"

_STEP = re.compile(
    r"^\s*(\d+)\s+T=\s*(" + _NUMBER + r")"
    r".*?\bE=\s*(" + _NUMBER + r")"
    r".*?\bF=\s*(" + _NUMBER + r")",
    re.IGNORECASE,
)

_KINETIC = re.compile(r"\bEK=\s*(" + _NUMBER + r")", re.IGNORECASE)

_INCAR_INT = r"\b{}\s*=\s*(-?\d+)"
_INCAR_FLOAT = r"\b{}\s*=\s*(" + _NUMBER + r")"


def _to_float(text: str) -> float:
    """Parse a VASP float, whose exponent may be written with ``D``."""

    return float(text.replace("D", "E").replace("d", "e"))


# ------------------------------------------------------------
# Result
# ------------------------------------------------------------


@dataclass
class VaspMd:
    """One finished VASP molecular-dynamics run, as read off disk."""

    directory: Path | None = None

    times: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    potential_energies: list[float] = field(default_factory=list)
    kinetic_energies: list[float] = field(default_factory=list)
    total_energies: list[float] = field(default_factory=list)

    #: Positions per recorded frame, in Angstrom, when XDATCAR was read.
    frames: list = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    cell: Any = None

    #: The parsed XDATCAR, kept whole so the diffusion window can be
    #: handed it directly rather than rebuilding it from ``frames``.
    trajectory: Any = None

    #: Tags read back from INCAR.
    potim: float = 1.0
    nsw: int = 0
    nblock: int = 1
    tebeg: float = 0.0
    teend: float | None = None
    mdalgo: int | None = None
    smass: float | None = None

    natoms: int = 0

    #: True when the folder had positions but nothing that timed them --
    #: a force-field run saved from here, rather than VASP output. The
    #: time axis then comes from the frames and the recording interval,
    #: and the diagnostics that count ionic steps do not apply.
    positions_only: bool = False

    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------

    @property
    def steps(self) -> int:
        """Ionic steps actually found, which is not always ``NSW``."""

        return len(self.times)

    @property
    def ensemble(self) -> str:
        """``"nve"`` or ``"nvt"``, worked out from MDALGO and SMASS."""

        if self.smass is not None and int(self.smass) == SMASS_NVE:
            return "nve"

        if self.mdalgo is not None:
            return MDALGO_LABELS.get(self.mdalgo, ("nvt", ""))[0]

        return "nvt"

    @property
    def ensemble_label(self) -> str:
        """The ensemble spelled out, naming the thermostat."""

        if self.smass is not None and int(self.smass) == SMASS_NVE:
            return "NVE (SMASS = -3, no thermostat)"

        if self.mdalgo is not None and self.mdalgo in MDALGO_LABELS:
            return MDALGO_LABELS[self.mdalgo][1]

        return "NVT (thermostat not identified)"

    @property
    def mean_temperature(self) -> float:
        """Average temperature over the run."""

        if not self.temperatures:
            return 0.0

        return float(np.mean(self.temperatures))

    @property
    def temperature_spread(self) -> float:
        """
        Standard deviation of the temperature.

        For N atoms this should be about ``T * sqrt(2 / (3N))`` in
        equilibrium, so a much smaller value means the thermostat is
        over-damped and the run is not sampling properly.
        """

        if len(self.temperatures) < 2:
            return 0.0

        return float(np.std(self.temperatures))

    @property
    def expected_spread(self) -> float:
        """The equilibrium temperature fluctuation for this many atoms."""

        if not self.natoms or not self.mean_temperature:
            return float("nan")

        return self.mean_temperature * float(
            np.sqrt(2.0 / (3.0 * self.natoms))
        )

    @property
    def energy_drift(self) -> float:
        """Change in total energy per atom over the run, in eV."""

        if len(self.total_energies) < 2 or not self.natoms:
            return 0.0

        return (
            self.total_energies[-1] - self.total_energies[0]
        ) / self.natoms

    @property
    def simulated_time(self) -> float:
        """Total simulated time, in femtoseconds."""

        return self.times[-1] if self.times else 0.0

    @property
    def has_trajectory(self) -> bool:
        """True when positions were read, so diffusion can be computed."""

        return len(self.frames) >= MIN_STEPS


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


def read_run(
    folder: Path | str,
    with_trajectory: bool = True,
) -> VaspMd:
    """
    Read a finished VASP molecular-dynamics run from a folder.

    ``with_trajectory`` also parses XDATCAR, which is what a diffusion
    coefficient needs but is the slow part for a long run.
    """

    path = Path(folder)

    if not path.is_dir():
        raise MdVaspError(f"There is no such folder: {path}")

    run = VaspMd(directory=path)

    _read_incar(path / INCAR, run)

    # Before the traces, because the energy-drift and temperature-spread
    # diagnostics are both per atom and both silently do nothing when
    # the count is zero. Reading a POSCAR header is cheap; waiting for
    # the trajectory would mean losing them whenever XDATCAR is absent
    # or the caller asked not to parse it.
    _read_natoms(path, run)

    oszicar = path / OSZICAR

    #: True when the folder has positions but nothing that timed them.
    positions_only = False

    if oszicar.is_file():
        _read_oszicar(oszicar, run)

    elif (path / VASPRUN).is_file():
        _read_vasprun(path / VASPRUN, run)

    elif (path / XDATCAR).is_file():

        positions_only = True

        # Positions but no energies. That is what a force-field run
        # saved from here looks like -- there was no VASP, so there is
        # no OSZICAR to invent -- and it is enough for everything that
        # reads a trajectory: MSD and diffusion, RDF, Van Hove,
        # residence. Only the energy traces are missing, and refusing
        # the whole folder over them would be refusing the part anyone
        # actually came for.
        run.notes.append(
            "No OSZICAR or vasprun.xml, so there are no energy traces. "
            "The trajectory was read, so diffusion and the correlation "
            "functions are all available."
        )

    else:
        raise MdVaspError(
            f"No OSZICAR, vasprun.xml or XDATCAR in {path.name}.\n\nThis "
            f"reads a finished molecular-dynamics run. Write the input "
            f"with VASP Input > INCAR > AIMD, run it, and point this at "
            f"the folder it produced -- or point it at a folder saved "
            f"from MD > CHGNet > Run Dynamics."
        )

    if positions_only:

        # Nothing here timed the run, so ``steps`` -- which counts
        # energies -- is zero however long it ran. The evidence that
        # this is dynamics rather than a relaxation is the number of
        # recorded frames instead, which means reading them first.
        run.positions_only = True

        _read_trajectory(path / XDATCAR, run)

        # times drives both ``steps`` and ``simulated_time``, and
        # without it a perfectly good 400 fs run reports itself as zero.
        spacing = run.potim * max(1, run.nblock)

        run.times = [
            spacing * (index + 1) for index in range(len(run.frames))
        ]

        if len(run.frames) < MIN_STEPS:
            raise MdVaspError(
                f"{path.name} has only {len(run.frames)} recorded "
                f"frame(s) in its XDATCAR.\n\nThat is a static "
                f"calculation or a relaxation, not dynamics."
            )

    elif run.steps < MIN_STEPS:
        raise MdVaspError(
            f"{path.name} has only {run.steps} molecular-dynamics "
            f"step(s).\n\nThat is a static calculation or a relaxation, "
            f"not dynamics. An MD run needs IBRION = 0 and NSW set to the "
            f"number of steps."
        )

    if with_trajectory and not positions_only:
        _read_trajectory(path / XDATCAR, run)

    run.notes = _notes(run)

    return run


def _read_oszicar(path: Path, run: VaspMd) -> None:
    """Pull the per-step traces out of OSZICAR."""

    try:
        text = path.read_text(errors="replace")

    except OSError as exc:
        raise MdVaspError(f"Could not read {path.name}: {exc}") from exc

    for line in text.splitlines():

        step = _STEP.match(line)

        if step is None:
            continue

        index = int(step.group(1))

        temperature = _to_float(step.group(2))
        total = _to_float(step.group(3))
        potential = _to_float(step.group(4))

        kinetic = _KINETIC.search(line)

        run.times.append(index * run.potim)
        run.temperatures.append(temperature)
        run.total_energies.append(total)
        run.potential_energies.append(potential)

        run.kinetic_energies.append(
            _to_float(kinetic.group(1))
            if kinetic is not None
            # EK is not printed by every VASP build; E - F is the same
            # quantity by definition.
            else total - potential
        )


def _read_vasprun(path: Path, run: VaspMd) -> None:
    """Fall back to vasprun.xml when there is no OSZICAR."""

    try:
        from pymatgen.io.vasp.outputs import Vasprun

        vasprun = Vasprun(
            str(path),
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            exception_on_bad_xml=False,
        )

    except Exception as exc:
        raise MdVaspError(
            f"Could not parse {path.name}: {type(exc).__name__}: {exc}\n\n"
            f"A run cut short by a wall clock leaves a truncated "
            f"vasprun.xml. Its OSZICAR is usually still readable -- keep "
            f"that file and point this at the folder again."
        ) from exc

    for index, step in enumerate(vasprun.ionic_steps, start=1):

        energy = float(step.get("e_fr_energy", step.get("e_0_energy", 0.0)))

        run.times.append(index * run.potim)
        run.potential_energies.append(energy)
        run.total_energies.append(energy)
        run.kinetic_energies.append(0.0)
        run.temperatures.append(float("nan"))

    run.notes.append(
        "Read from vasprun.xml because there was no OSZICAR. Temperatures "
        "and kinetic energies are not available from it."
    )


def _read_trajectory(path: Path, run: VaspMd) -> None:
    """Read the positions, if XDATCAR is there."""

    if not path.is_file():
        run.notes.append(
            "No XDATCAR, so there are no positions and no diffusion "
            "coefficient. The traces below come from OSZICAR alone."
        )
        return

    from qvista.services import msd

    try:
        # XDATCAR holds every NBLOCK-th step, so the gap between frames
        # in the file is POTIM * NBLOCK, not POTIM. read_xdatcar wants
        # that in picoseconds. Getting this wrong scales every diffusion
        # coefficient by NBLOCK without changing the shape of the curve,
        # so nothing about the plot would look wrong.
        run.trajectory = msd.read_xdatcar(
            path,
            timestep=run.potim * max(1, run.nblock) / 1000.0,
            temperature=run.mean_temperature or run.tebeg,
        )

    except msd.TrajectoryError as exc:
        run.notes.append(f"XDATCAR could not be read: {exc}")
        return

    run.frames = [
        np.asarray(frame, dtype=float)
        for frame in run.trajectory.positions
    ]

    run.symbols = list(run.trajectory.symbols)
    run.cell = run.trajectory.lattice

    if not run.natoms:
        run.natoms = len(run.symbols)


def _read_natoms(folder: Path, run: VaspMd) -> None:
    """
    How many atoms, from whichever file names them first.

    Every candidate carries the count on its seventh line (POSCAR
    format) or as ``NIONS`` (OUTCAR), so none of them has to be parsed
    in full.
    """

    for name in (XDATCAR, "POSCAR", "CONTCAR"):

        path = folder / name

        if not path.is_file():
            continue

        try:
            with path.open(errors="replace") as handle:
                head = [next(handle) for _ in range(7)]

        except (OSError, StopIteration):
            continue

        try:
            counts = [int(value) for value in head[6].split()]

        except ValueError:
            continue

        if counts:
            run.natoms = sum(counts)
            return

    outcar = folder / "OUTCAR"

    if outcar.is_file():

        try:
            found = re.search(
                r"NIONS\s*=\s*(\d+)", outcar.read_text(errors="replace")
            )

        except OSError:
            return

        if found:
            run.natoms = int(found.group(1))


def _read_incar(path: Path, run: VaspMd) -> None:
    """Read back the tags that say how the dynamics were set up."""

    if not path.is_file():
        return

    try:
        text = path.read_text(errors="replace")

    except OSError:
        return

    # Comments would otherwise supply values that were never set.
    text = "\n".join(
        line.split("#")[0].split("!")[0] for line in text.splitlines()
    )

    for tag, attribute in (
        ("NSW", "nsw"),
        ("NBLOCK", "nblock"),
        ("MDALGO", "mdalgo"),
    ):
        found = re.search(_INCAR_INT.format(tag), text, re.IGNORECASE)

        if found:
            setattr(run, attribute, int(found.group(1)))

    for tag, attribute in (
        ("POTIM", "potim"),
        ("TEBEG", "tebeg"),
        ("TEEND", "teend"),
        ("SMASS", "smass"),
    ):
        found = re.search(_INCAR_FLOAT.format(tag), text, re.IGNORECASE)

        if found:
            setattr(run, attribute, _to_float(found.group(1)))


def _notes(run: VaspMd) -> list[str]:
    """Diagnostics, in the order they matter."""

    notes = list(run.notes)

    # ----------------------------------------------------------
    # Did it finish?
    # ----------------------------------------------------------

    recorded = run.steps * (
        max(1, run.nblock) if run.positions_only else 1
    )

    if run.nsw and recorded < run.nsw:
        notes.append(
            f"WARNING: the run stopped after {run.steps} of the "
            f"{run.nsw} steps asked for. Everything below describes the "
            f"part that ran."
        )

    # ----------------------------------------------------------
    # Energy conservation, but only where it means something
    # ----------------------------------------------------------

    drift = abs(run.energy_drift) * 1000.0

    if run.ensemble == "nve" and drift > DRIFT_WARNING:
        notes.append(
            f"WARNING: the total energy drifted by {drift:.2f} meV per "
            f"atom. In NVE it should be conserved, so POTIM = {run.potim} "
            f"fs is too long for this system."
        )

    # ----------------------------------------------------------
    # Is the thermostat doing its job?
    # ----------------------------------------------------------

    if run.ensemble == "nvt" and run.tebeg and run.temperatures:

        difference = abs(run.mean_temperature - run.tebeg)

        if difference > max(50.0, 0.2 * run.tebeg):
            notes.append(
                f"WARNING: the mean temperature "
                f"({run.mean_temperature:.1f} K) is far from TEBEG "
                f"({run.tebeg:.1f} K). The run has probably not "
                f"equilibrated yet."
            )

        expected = run.expected_spread

        if np.isfinite(expected) and expected > 0.0:

            ratio = run.temperature_spread / expected

            if ratio < 0.5:
                notes.append(
                    f"The temperature fluctuates by "
                    f"{run.temperature_spread:.1f} K where "
                    f"{expected:.1f} K is expected for {run.natoms} atoms. "
                    f"The thermostat is over-damped, which suppresses the "
                    f"fluctuations that carry the physics."
                )

    # ----------------------------------------------------------
    # Is it long enough to mean anything?
    # ----------------------------------------------------------

    if run.simulated_time < 1000.0:
        notes.append(
            f"This run is {run.simulated_time:.0f} fs. That is enough to "
            f"see whether the setup is stable, and too short for any "
            f"equilibrium property. Diffusion in a solid normally needs "
            f"tens of picoseconds."
        )

    if not run.has_trajectory:
        notes.append(
            "Without XDATCAR there is no mean squared displacement and no "
            "diffusion coefficient."
        )

    return notes


# ------------------------------------------------------------
# Into the shared result type
# ------------------------------------------------------------


def to_result(run: VaspMd):
    """
    Package as an :class:`~qvista.analyses.md.MDResult`.

    The same result type as the force-field route, so the same traces
    figure, the same CSV and the same trajectory export serve both and
    neither can quietly lose a feature the other has.
    """

    from qvista.analyses.md import ORIGIN_VASP, MDResult

    result = MDResult(
        times=list(run.times),
        potential_energies=list(run.potential_energies),
        kinetic_energies=list(run.kinetic_energies),
        total_energies=list(run.total_energies),
        temperatures=list(run.temperatures),
        frames=list(run.frames),
        symbols=list(run.symbols),
        cell=run.cell,
        ensemble=run.ensemble,
        temperature=run.tebeg,
        timestep=run.potim,
        steps=run.steps,
        natoms=run.natoms,
        force_field=describe(run),
        # Marks the run as read rather than computed here, so the
        # CHGNet analyses do not open on it.
        origin=ORIGIN_VASP,
    )

    result.rows = _rows(run)
    result.notes = list(run.notes)

    return result


def describe(run: VaspMd) -> str:
    """One line naming what produced these numbers."""

    return f"VASP ab-initio MD, {run.ensemble_label}"


def _rows(run: VaspMd) -> list[tuple[str, str]]:
    """Console rows."""

    rows: list[tuple[str, str]] = [
        ("Source", str(run.directory)),
        ("Forces from", describe(run)),
        ("Atoms", str(run.natoms) if run.natoms else "?"),
        ("Steps", f"{run.steps}" + (f" of {run.nsw}" if run.nsw else "")),
        ("Timestep", f"{run.potim} fs"),
        ("Simulated time", f"{run.simulated_time:.1f} fs"),
    ]

    if run.tebeg:
        rows.append(("Target temperature", f"{run.tebeg:.1f} K"))

        if run.teend is not None and run.teend != run.tebeg:
            rows.append(("Ramped to", f"{run.teend:.1f} K"))

    if run.temperatures and np.isfinite(run.mean_temperature):
        rows.append(
            ("Mean temperature", f"{run.mean_temperature:.1f} K")
        )
        rows.append(
            (
                "Temperature spread",
                f"{run.temperature_spread:.1f} K"
                + (
                    f"  (expected {run.expected_spread:.1f})"
                    if np.isfinite(run.expected_spread)
                    else ""
                ),
            )
        )

    if run.total_energies:
        rows.append(
            ("Final total energy", f"{run.total_energies[-1]:.4f} eV")
        )
        rows.append(
            ("Energy drift", f"{run.energy_drift * 1000:.3f} meV/atom")
        )

    rows.append(
        (
            "Trajectory",
            f"{len(run.frames)} frames"
            if run.has_trajectory
            else "not available (no XDATCAR)",
        )
    )

    return rows


# ------------------------------------------------------------
# Writing a run out
# ------------------------------------------------------------

#: What a saved run can be made of. Each is independent -- a run whose
#: plot failed to write should still leave its trajectory behind.
SAVEABLE = (
    ("structures", "Structures (POSCAR, CONTCAR)"),
    ("trajectory", "Trajectory (XDATCAR, md.traj)"),
    ("log", "Run log (md.log)"),
    ("plots", "Traces plot (md_traces.png)"),
)


@dataclass
class SaveResult:
    """What a save actually managed to write."""

    root: Path

    written: list[Path] = field(default_factory=list)

    #: What was asked for and did not happen, and why.
    skipped: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.written)


def save_run(
    result,
    root: Path | str,
    structures: bool = True,
    trajectory: bool = True,
    log: bool = True,
    plots: bool = True,
    figure=None,
) -> SaveResult:
    """
    Write a finished run to disk in the layout the reader expects.

    Dynamics that took minutes used to live only in the session: close
    the window and everything under MD > CHGNet > Analysis had nothing
    to open on, and the run had to be done again. What goes out is an
    XDATCAR and an INCAR naming the timestep, which together are what
    :func:`read_run` needs -- so a saved run reopens through
    MD > VASP > Analysis and its diffusion coefficient comes back the
    same.

    The INCAR is genuinely the settings this run used, and the XDATCAR
    genuinely its positions. Nothing here fabricates VASP *output*: the
    energies go to md.csv rather than to an invented OSZICAR, because a
    file claiming to have come from VASP should have come from VASP.

    Each piece is written independently; a failure costs that piece and
    is reported, not the save.
    """

    base = Path(root)

    base.mkdir(parents=True, exist_ok=True)

    saved = SaveResult(root=base)

    frames = list(getattr(result, "frames", []) or [])
    symbols = list(getattr(result, "symbols", []) or [])

    order = _species_order(symbols)

    if structures:

        if not frames:
            saved.skipped.append("No recorded frames to write.")

        else:
            for name, frame in (
                ("POSCAR", frames[0]),
                ("CONTCAR", frames[-1]),
            ):
                try:
                    (base / name).write_text(
                        _poscar_text(result, frame, order, name),
                        encoding="utf-8",
                    )

                    saved.written.append(base / name)

                except Exception as exc:  # noqa: BLE001
                    saved.skipped.append(
                        f"{name}: {type(exc).__name__}: {exc}"
                    )

    if trajectory:

        if not frames:
            saved.skipped.append("No recorded frames to write.")

        else:
            try:
                (base / XDATCAR).write_text(
                    _xdatcar_text(result, order), encoding="utf-8"
                )

                saved.written.append(base / XDATCAR)

            except Exception as exc:  # noqa: BLE001
                saved.skipped.append(
                    f"{XDATCAR}: {type(exc).__name__}: {exc}"
                )

            # The settings the reader needs to put a time axis on those
            # frames. Without POTIM and NBLOCK every diffusion
            # coefficient read back is out by the recording interval.
            try:
                (base / "INCAR").write_text(
                    _incar_text(result), encoding="utf-8"
                )

                saved.written.append(base / "INCAR")

            except OSError as exc:
                saved.skipped.append(f"INCAR: {exc}")

            try:
                path = base / "md.traj"

                _write_ase_trajectory(result, path)

                saved.written.append(path)

            except Exception as exc:  # noqa: BLE001
                saved.skipped.append(
                    f"md.traj: {type(exc).__name__}: {exc}"
                )

    try:
        (base / "md.csv").write_text(result.to_csv(), encoding="utf-8")

        saved.written.append(base / "md.csv")

    except Exception as exc:  # noqa: BLE001
        saved.skipped.append(f"md.csv: {type(exc).__name__}: {exc}")

    if log:

        lines = list(getattr(result, "log", []) or [])

        if not lines:
            saved.skipped.append("The run kept no log.")

        else:
            try:
                path = base / "md.log"

                path.write_text("\n".join(lines) + "\n", encoding="utf-8")

                saved.written.append(path)

            except OSError as exc:
                saved.skipped.append(f"md.log: {exc}")

    if plots:

        if figure is None:
            saved.skipped.append("No figure to save.")

        else:
            try:
                path = base / "md_traces.png"

                figure.savefig(str(path), dpi=300, bbox_inches="tight")

                saved.written.append(path)

            except Exception as exc:  # noqa: BLE001
                saved.skipped.append(
                    f"md_traces.png: {type(exc).__name__}: {exc}"
                )

    try:
        (base / "README.txt").write_text(
            _saved_readme(result), encoding="utf-8"
        )

        saved.written.append(base / "README.txt")

    except OSError:
        pass

    return saved


def _species_order(symbols: Sequence[str]) -> list[int]:
    """
    Atom indices grouped by species, species in first-appearance order.

    POSCAR and XDATCAR name each species once and then count it, so the
    positions must arrive grouped. An ASE trajectory is under no such
    obligation, and writing one out in its own order silently relabels
    atoms -- the counts still add up, so nothing complains, and every
    species-resolved analysis afterwards is answering about the wrong
    element.
    """

    seen: list[str] = []

    for symbol in symbols:
        if symbol not in seen:
            seen.append(symbol)

    order: list[int] = []

    for species in seen:
        order += [
            index
            for index, symbol in enumerate(symbols)
            if symbol == species
        ]

    return order


def _species_counts(symbols: Sequence[str]) -> tuple[list[str], list[int]]:
    """The species line and the count line, in the same order."""

    names: list[str] = []
    counts: list[int] = []

    for symbol in symbols:

        if symbol not in names:
            names.append(symbol)
            counts.append(0)

        counts[names.index(symbol)] += 1

    return names, counts


def _fractional(result, frame) -> np.ndarray:
    """Cartesian positions as fractions of the cell."""

    cell = np.asarray(result.cell, dtype=float).reshape(3, 3)

    return np.asarray(frame, dtype=float) @ np.linalg.inv(cell)


def _lattice_lines(result) -> list[str]:
    """The scale and the three cell vectors."""

    cell = np.asarray(result.cell, dtype=float).reshape(3, 3)

    lines = ["   1.00000000000000"]

    for row in cell:
        lines.append("  " + "  ".join(f"{value:20.16f}" for value in row))

    return lines


def _header(result, order: Sequence[int], comment: str) -> list[str]:
    """The six lines every POSCAR and XDATCAR starts with."""

    symbols = [result.symbols[index] for index in order]

    names, counts = _species_counts(symbols)

    return (
        [comment]
        + _lattice_lines(result)
        + [
            "   " + "  ".join(f"{name:>4s}" for name in names),
            "   " + "  ".join(f"{count:>4d}" for count in counts),
        ]
    )


def _poscar_text(result, frame, order: Sequence[int], name: str) -> str:
    """One frame as a POSCAR."""

    which = "first" if name == "POSCAR" else "last"

    comment = (
        f"QVista MD {result.ensemble.upper()} "
        f"{result.temperature:.0f}K ({which} recorded frame)"
    )

    lines = _header(result, order, comment) + ["Direct"]

    coordinates = _fractional(result, frame)

    for index in order:
        lines.append(
            "  "
            + "  ".join(f"{value:19.16f}" for value in coordinates[index])
        )

    return "\n".join(lines) + "\n"


def _xdatcar_text(result, order: Sequence[int]) -> str:
    """Every recorded frame, in the format :func:`read_run` parses."""

    comment = (
        f"QVista MD {result.ensemble.upper()} "
        f"{result.temperature:.0f}K dt={result.timestep}fs"
    )

    lines = _header(result, order, comment)

    for number, frame in enumerate(result.frames, start=1):

        lines.append(f"Direct configuration={number:6d}")

        coordinates = _fractional(result, frame)

        for index in order:
            lines.append(
                "  "
                + "  ".join(
                    f"{value:19.16f}" for value in coordinates[index]
                )
            )

    return "\n".join(lines) + "\n"


def _incar_text(result) -> str:
    """
    The settings this run used, in INCAR spelling.

    Written so the trajectory has a time axis when it is read back.
    ``NBLOCK`` is the recording interval rather than 1, because the
    frames in XDATCAR are that far apart -- read as 1, every diffusion
    coefficient comes back too large by exactly that factor, and the
    shape of the curve gives no hint that anything is wrong.
    """

    return (
        "\n".join(
            [
                "# Written by QVista to describe a force-field MD run.",
                "# These are the settings that were used, in INCAR",
                "# spelling, so the trajectory beside them can be read",
                "# back with the right time axis. VASP did not run this.",
                "",
                "  IBRION =      0",
                f"  POTIM  = {result.timestep:6.3f}",
                f"  NSW    = {result.steps:6d}",
                f"  NBLOCK = {max(1, int(result.record_every)):6d}",
                f"  TEBEG  = {result.temperature:6.1f}",
                f"  TEEND  = {result.temperature:6.1f}",
            ]
        )
        + "\n"
    )


def _write_ase_trajectory(result, path: Path) -> None:
    """Every recorded frame as an ASE trajectory."""

    from ase import Atoms
    from ase.io import write as ase_write

    cell = np.asarray(result.cell, dtype=float).reshape(3, 3)

    images = [
        Atoms(
            symbols=list(result.symbols),
            positions=np.asarray(frame, dtype=float),
            cell=cell,
            pbc=True,
        )
        for frame in result.frames
    ]

    ase_write(str(path), images)


def _saved_readme(result) -> str:
    """What this folder is, and what produced it."""

    forces = getattr(result, "force_field", "") or "unknown"

    lines = [
        "QVista molecular dynamics run",
        "=" * 29,
        "",
        f"Forces: {forces}",
        "",
        "  POSCAR         first recorded frame",
        "  CONTCAR        last recorded frame",
        "  XDATCAR        every recorded frame",
        "  INCAR          the settings used, so the frames have a time axis",
        "  md.traj        the same frames as an ASE trajectory",
        "  md.csv         energy and temperature against time",
        "  md.log         the run, step by step",
        "  md_traces.png  the energy and temperature traces",
        "",
        "Reopen this folder through MD > VASP > Analysis to get the",
        "trajectory back -- MSD and diffusion, RDF, Van Hove and",
        "residence time all read it.",
        "",
        "There is no OSZICAR or OUTCAR here, because VASP did not run.",
        "The energies are in md.csv instead.",
    ]

    if "not DFT" in forces:
        lines += [
            "",
            "These are machine-learning forces, not DFT.",
        ]

    return "\n".join(lines) + "\n"

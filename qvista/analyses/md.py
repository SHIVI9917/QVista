"""
Molecular dynamics.

Gives the atoms velocities drawn from a temperature and lets them move
under the forces between them, recording energy, temperature and positions
as it goes. Useful for seeing whether a structure actually holds together
at operating temperature, and for watching how a mobile ion behaves.

Forces come from a machine-learning force field, **not** DFT. See
:mod:`qvista.services.forcefield`.

Two ensembles are offered:

* **NVE** -- constant energy. Nothing controls the temperature, so drift in
  the total energy is a direct measure of integration error: if it wanders,
  the timestep is too long.
* **NVT** -- constant temperature, via a Langevin thermostat. What you want
  for equilibrium properties at a given temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qvista.analyses.base import (
    INDETERMINATE,
    Analysis,
    AnalysisError,
    AnalysisResult,
)
from qvista.models.project import Project

#: Ensembles the user can pick.
ENSEMBLE_NVE = "nve"
ENSEMBLE_NVT = "nvt"

ENSEMBLE_LABELS = {
    ENSEMBLE_NVE: "NVE (constant energy)",
    ENSEMBLE_NVT: "NVT (constant temperature, Langevin)",
}

#: Where a run came from. A VASP run read off disk becomes the same
#: :class:`MDResult` as one run here, and is kept in the same place, so
#: without this there is nothing to tell them apart -- and the analyses
#: under MD > CHGNet would happily open on a VASP trajectory and label
#: the figure with the wrong method.
ORIGIN_FORCEFIELD = "forcefield"
ORIGIN_VASP = "vasp"

#: Femtoseconds. 1 fs is safe for most inorganic solids; hydrogen needs
#: less, because it moves fastest.
DEFAULT_TIMESTEP = 1.0

DEFAULT_TEMPERATURE = 300.0

DEFAULT_STEPS = 200

#: Langevin coupling strength, in inverse femtoseconds.
DEFAULT_FRICTION = 0.01

#: Frames recorded for the trajectory. Recording every step of a long run
#: would use a lot of memory for little extra information.
MAX_FRAMES = 200


# ------------------------------------------------------------
# Result
# ------------------------------------------------------------


@dataclass
class MDResult(AnalysisResult):
    """Time series recorded during the run."""

    times: list[float] = field(default_factory=list)

    potential_energies: list[float] = field(default_factory=list)
    kinetic_energies: list[float] = field(default_factory=list)
    total_energies: list[float] = field(default_factory=list)

    temperatures: list[float] = field(default_factory=list)

    #: Recorded positions, for the trajectory export.
    frames: list = field(default_factory=list)

    symbols: list[str] = field(default_factory=list)
    cell: Any = None

    #: Every progress message the run published, in order.
    log: list = field(default_factory=list)

    #: Steps between recorded frames. The gap between XDATCAR frames is
    #: this times the timestep, and a diffusion coefficient read back
    #: without it is wrong by exactly this factor.
    record_every: int = 1

    ensemble: str = ENSEMBLE_NVT
    temperature: float = DEFAULT_TEMPERATURE
    timestep: float = DEFAULT_TIMESTEP
    steps: int = DEFAULT_STEPS

    natoms: int = 0
    force_field: str = ""

    #: ``ORIGIN_FORCEFIELD`` or ``ORIGIN_VASP``. Defaults to the former
    #: because that is what this class runs; the VASP reader sets it.
    origin: str = ORIGIN_FORCEFIELD

    # ------------------------------------------------------------

    @property
    def mean_temperature(self) -> float:
        """Average temperature over the run."""

        if not self.temperatures:
            return 0.0

        return sum(self.temperatures) / len(self.temperatures)

    @property
    def energy_drift(self) -> float:
        """
        Total energy change per atom over the run, in eV.

        The diagnostic for NVE: it should be near zero, and grows when the
        timestep is too long.
        """

        if len(self.total_energies) < 2 or not self.natoms:
            return 0.0

        return (
            self.total_energies[-1] - self.total_energies[0]
        ) / self.natoms

    def as_storage(self) -> dict[str, Any]:
        """Package for :class:`~qvista.models.results.Results`."""

        return {
            "times": self.times,
            "temperatures": self.temperatures,
            "total_energies": self.total_energies,
            "result": self,
        }

    def to_csv(self) -> str:
        """Render the time series as CSV."""

        lines = [
            "# QVista molecular dynamics",
            f"# force field,{self.force_field}",
            f"# ensemble,{ENSEMBLE_LABELS.get(self.ensemble, self.ensemble)}",
            f"# target temperature K,{self.temperature}",
            f"# timestep fs,{self.timestep}",
            f"# atoms,{self.natoms}",
            "",
            "time_fs,potential_ev,kinetic_ev,total_ev,temperature_k",
        ]

        for index in range(len(self.times)):
            lines.append(
                f"{self.times[index]:.4f},"
                f"{self.potential_energies[index]:.6f},"
                f"{self.kinetic_energies[index]:.6f},"
                f"{self.total_energies[index]:.6f},"
                f"{self.temperatures[index]:.4f}"
            )

        return "\n".join(lines) + "\n"

    def to_xyz_trajectory(self) -> str:
        """Render the recorded frames as an extended XYZ trajectory."""

        from qvista.analyses.phonon_modes import to_xyz_trajectory

        if not self.frames:
            return ""

        return to_xyz_trajectory(
            self.symbols,
            self.frames,
            cell=self.cell,
            comment=(
                f"QVista MD {self.ensemble.upper()} "
                f"T={self.temperature}K dt={self.timestep}fs"
            ),
        )


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------


class MDAnalysis(Analysis):
    """Run molecular dynamics and record the trajectory."""

    name = "Molecular Dynamics"

    #: A molecule can be run in NVE just as a crystal can.
    requires_periodic = False

    def __init__(
        self,
        project: Project,
        temperature: float = DEFAULT_TEMPERATURE,
        timestep: float = DEFAULT_TIMESTEP,
        steps: int = DEFAULT_STEPS,
        ensemble: str = ENSEMBLE_NVT,
        friction: float = DEFAULT_FRICTION,
    ) -> None:

        super().__init__(project)

        self.temperature = float(temperature)
        self.timestep = float(timestep)
        self.steps = int(steps)
        self.ensemble = ensemble
        self.friction = friction

        self._log: list[str] = []

    def report(self, message: str, percent: int = INDETERMINATE) -> None:
        """
        Publish progress, and keep it.

        The step-by-step record of temperature and energy is how anyone
        judges whether a run equilibrated. It scrolls out of a console
        and dies with the window, so it is kept and written beside the
        trajectory.
        """

        self._log.append(message)

        super().report(message, percent)

    # ------------------------------------------------------------

    def _get_calculator(self):
        """Return the force calculator. Overridden in tests."""

        from qvista.services import forcefield

        try:
            return forcefield.get_calculator()

        except forcefield.ForceFieldUnavailable as exc:
            raise AnalysisError(str(exc)) from exc

    def validate(self) -> None:
        """Check the run is well posed."""

        super().validate()

        if self.steps < 1:
            raise AnalysisError("A run needs at least one step.")

        if self.timestep <= 0.0:
            raise AnalysisError("The timestep must be positive.")

        if self.ensemble not in ENSEMBLE_LABELS:
            raise AnalysisError(
                f"Unknown ensemble: {self.ensemble}"
            )

        if self.temperature < 0.0:
            raise AnalysisError(
                "Temperature cannot be negative."
            )

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    def execute(self) -> MDResult:
        """Integrate the equations of motion, recording as we go."""

        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from pymatgen.io.ase import AseAtomsAdaptor

        from qvista.services import forcefield
        from qvista.services.structure_analyzer import (
            StructureAnalysisError,
            StructureAnalyzer,
        )

        source = self.project.info.source_file

        if source is None:
            raise AnalysisError(
                "This project has no structure file to run."
            )

        try:
            parsed = StructureAnalyzer.parse(source)

            atoms = AseAtomsAdaptor.get_atoms(parsed.structure)

        except StructureAnalysisError as exc:
            raise AnalysisError(str(exc)) from exc

        except Exception as exc:
            raise AnalysisError(
                f"Could not prepare the structure for dynamics: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        atoms.calc = self._get_calculator()

        # ----------------------------------------------------------
        # Initial velocities
        #
        # Drawn from a Maxwell-Boltzmann distribution at twice the target,
        # because equipartition sends half the kinetic energy into
        # potential energy as the system settles.
        # ----------------------------------------------------------

        self.report(
            f"Initialising velocities at {self.temperature:.1f} K "
            f"using {forcefield.describe()}."
        )

        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=2.0 * self.temperature,
        )

        # ----------------------------------------------------------
        # Integrator
        # ----------------------------------------------------------

        try:
            if self.ensemble == ENSEMBLE_NVE:
                dynamics = VelocityVerlet(
                    atoms,
                    timestep=self.timestep * units.fs,
                    logfile=None,
                )

            else:
                dynamics = Langevin(
                    atoms,
                    timestep=self.timestep * units.fs,
                    temperature_K=self.temperature,
                    friction=self.friction / units.fs,
                    # fixcm=True does not sample the correct NVT
                    # distribution and is deprecated in ASE 3.28+. Centre
                    # of mass drift is a uniform translation, which is
                    # harmless here.
                    fixcm=False,
                    logfile=None,
                )

        except Exception as exc:
            raise AnalysisError(
                f"Could not set up the integrator: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # ----------------------------------------------------------
        # Run
        # ----------------------------------------------------------

        result = MDResult(
            symbols=[str(s) for s in atoms.get_chemical_symbols()],
            cell=atoms.get_cell(),
            ensemble=self.ensemble,
            temperature=self.temperature,
            timestep=self.timestep,
            steps=self.steps,
            natoms=len(atoms),
            force_field=forcefield.describe(),
        )

        record_every = max(1, self.steps // MAX_FRAMES)

        result.record_every = record_every

        for step in range(self.steps):

            self.raise_if_cancelled()

            try:
                # One step at a time, so progress is live and the run can
                # be stopped part way.
                dynamics.run(1)

                potential = float(atoms.get_potential_energy())
                kinetic = float(atoms.get_kinetic_energy())
                temperature = float(atoms.get_temperature())

            except Exception as exc:
                raise AnalysisError(
                    f"Dynamics failed at step {step + 1}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            result.times.append((step + 1) * self.timestep)
            result.potential_energies.append(potential)
            result.kinetic_energies.append(kinetic)
            result.total_energies.append(potential + kinetic)
            result.temperatures.append(temperature)

            if step % record_every == 0:
                result.frames.append(atoms.get_positions().copy())

            percent = int(round(100.0 * (step + 1) / self.steps))

            self.report(
                f"[{step + 1}/{self.steps}] "
                f"T = {temperature:7.1f} K   "
                f"E = {potential + kinetic:12.4f} eV",
                min(percent, 100),
            )

        result.rows = self._rows(result)
        result.notes = self._notes(result)

        result.log = list(self._log)

        return result

    # ------------------------------------------------------------
    # Console text
    # ------------------------------------------------------------

    def _rows(self, result: MDResult) -> list[tuple[str, str]]:
        """Summary rows."""

        import statistics

        spread = (
            statistics.pstdev(result.temperatures)
            if len(result.temperatures) > 1
            else 0.0
        )

        return [
            ("Force field", result.force_field),
            ("Ensemble", ENSEMBLE_LABELS.get(result.ensemble, "?")),
            ("Atoms", str(result.natoms)),
            ("Steps", str(result.steps)),
            ("Timestep", f"{result.timestep} fs"),
            (
                "Simulated time",
                f"{result.times[-1] if result.times else 0.0:.1f} fs",
            ),
            ("Target temperature", f"{result.temperature:.1f} K"),
            ("Mean temperature", f"{result.mean_temperature:.1f} K"),
            ("Temperature spread", f"{spread:.1f} K"),
            (
                "Final total energy",
                f"{result.total_energies[-1]:.4f} eV"
                if result.total_energies
                else "n/a",
            ),
            ("Energy drift", f"{result.energy_drift * 1000:.3f} meV/atom"),
        ]

    def _notes(self, result: MDResult) -> list[str]:
        """Diagnostics and warnings."""

        notes: list[str] = []

        # ----------------------------------------------------------
        # Energy conservation
        # ----------------------------------------------------------

        drift = abs(result.energy_drift) * 1000.0

        if result.ensemble == ENSEMBLE_NVE and drift > 1.0:
            notes.append(
                f"WARNING: the total energy drifted by {drift:.2f} meV per "
                f"atom. In NVE it should be conserved, so this means the "
                f"timestep of {result.timestep} fs is too long for this "
                f"system. Halve it and run again."
            )

        # ----------------------------------------------------------
        # Thermostat behaviour
        # ----------------------------------------------------------

        if result.ensemble == ENSEMBLE_NVT and result.temperatures:

            difference = abs(
                result.mean_temperature - result.temperature
            )

            if difference > max(50.0, 0.2 * result.temperature):
                notes.append(
                    f"WARNING: the mean temperature "
                    f"({result.mean_temperature:.1f} K) is far from the "
                    f"target ({result.temperature:.1f} K). The run is "
                    f"probably too short to have equilibrated, or the "
                    f"friction is too weak."
                )

        if result.steps < 100:
            notes.append(
                f"This run was only {result.steps} steps "
                f"({result.times[-1] if result.times else 0:.0f} fs). That "
                f"is enough to check the setup is stable, but far too "
                f"short for any equilibrium property."
            )

        notes.append(
            "Forces came from a machine-learning force field, not DFT."
        )

        return notes

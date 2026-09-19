"""
Structure relaxation with a machine-learning force field.

Moves the atoms downhill until the forces on them are small: the step
that comes before every other calculation, because a band structure, a
phonon spectrum or an elastic tensor computed on an unrelaxed cell
describes a structure nobody has.

Forces come from a machine-learning potential, **not** DFT. See
:mod:`qvista.services.forcefield`. That makes this fast enough to use
interactively and makes the result a starting point rather than an
answer: a geometry relaxed here is a good place to begin a VASP
relaxation, not a substitute for one.

What can move
-------------
**Ions only** keeps the cell exactly as given and lets the atoms find
their positions in it. This is what you want when the lattice constants
came from experiment, or from a calculation you trust more than this
one.

**Ions and cell** lets the lattice relax too, under a
``FrechetCellFilter`` -- the cell degrees of freedom are optimised
alongside the ionic ones, which is what VASP's ``ISIF = 3`` does.

**Ions and cell shape, volume fixed** is ``ISIF = 4``: the same filter
with ``constant_volume``, so the cell may change shape but not size.
That is how a structure is relaxed at a fixed density -- along an
equation of state, or at a volume somebody else has chosen -- and it is
the one that is easy to overlook.

A cell relaxation on a machine-learning potential will usually change
the volume by a few per cent. That is the model disagreeing with
whatever produced the input, and it is worth seeing rather than
accepting silently, which is why the volume change is reported and
flagged when it is large. Under ISIF 4 it should not change at all,
which makes that number a check rather than a result.

Why the symmetry is checked afterwards
--------------------------------------
An optimiser does not know about symmetry. Numerical noise in the forces
breaks it, so a structure that went in as R-3m can come out as P1 with
every atom displaced by a thousandth of an angstrom. Nothing looks
wrong, and the next calculation is on a triclinic cell with no
symmetry-reduced k-points and forty-eight times the work. The space
group is therefore measured before and after, and a drop is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qvista.analyses.base import Analysis, AnalysisError, AnalysisResult
from qvista.models.project import Project
from qvista.services.structure_analyzer import DEFAULT_SYMPREC

#: What is allowed to move, named after VASP's ISIF so the choice here
#: and the choice in an INCAR are the same choice.
#:
#:     ISIF   relax ions   cell shape   cell volume
#:       2       yes          no            no
#:       3       yes          yes           yes
#:       4       yes          yes           no
#:
#: ISIF 4 is the one that is easy to overlook and often the one wanted:
#: it lets the cell change *shape* while holding its volume, which is
#: how a structure is relaxed at a fixed density -- along an equation of
#: state, or under a pressure someone else has chosen.
IONS_ONLY = "ions"
IONS_AND_CELL = "cell"
CONSTANT_VOLUME = "shape"

MODES: dict[str, str] = {
    IONS_ONLY: "ISIF 2 - ions only, cell fixed",
    IONS_AND_CELL: "ISIF 3 - ions, cell shape and volume",
    CONSTANT_VOLUME: "ISIF 4 - ions and cell shape, volume fixed",
}

#: Which of them need a lattice at all.
NEEDS_CELL = (IONS_AND_CELL, CONSTANT_VOLUME)

#: The ISIF each one corresponds to, for the console and the notes.
ISIF: dict[str, int] = {
    IONS_ONLY: 2,
    IONS_AND_CELL: 3,
    CONSTANT_VOLUME: 4,
}

#: Force convergence, eV/Angstrom. Matches what the elastic workflow
#: relaxes to, and is a reasonable target for a surrogate model -- the
#: potential's own error is larger than this.
DEFAULT_FMAX = 0.02

#: Cap on optimiser steps. A relaxation that has not converged in this
#: many is not going to.
DEFAULT_STEPS = 300

#: Frames kept for the trajectory export. Every step of a long
#: relaxation is a lot of geometry for very little extra information.
MAX_FRAMES = 200

#: A volume change larger than this is called out, as a fraction.
VOLUME_WARNING = 0.05

#: Displacement above which an atom counts as having moved, in angstrom.
MOVED = 0.05


@dataclass
class RelaxResult(AnalysisResult):
    """The relaxed structure, and how far it travelled to get there."""

    structure: Any = None
    """The relaxed structure, as a pymatgen ``Structure``."""

    initial_energy: float = 0.0
    final_energy: float = 0.0

    initial_forces: float = 0.0
    final_forces: float = 0.0
    """Largest force on any atom, eV/Angstrom."""

    initial_volume: float = 0.0
    final_volume: float = 0.0

    spacegroup_before: str = ""
    spacegroup_number_before: int = 0

    spacegroup_after: str = ""
    spacegroup_number_after: int = 0

    steps: int = 0
    converged: bool = False

    mode: str = IONS_AND_CELL
    fmax: float = DEFAULT_FMAX
    symprec: float = DEFAULT_SYMPREC

    natoms: int = 0
    force_field: str = ""

    #: Energy and largest force at each step, for the figure.
    energies: list[float] = field(default_factory=list)
    forces: list[float] = field(default_factory=list)

    #: Recorded positions, for the trajectory export.
    frames: list = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    cell: Any = None

    #: Per-atom displacement from where it started, in angstrom.
    displacements: list[float] = field(default_factory=list)

    # ------------------------------------------------------------

    @property
    def energy_change(self) -> float:
        """How much energy the relaxation recovered, in eV. Negative."""

        return self.final_energy - self.initial_energy

    @property
    def energy_per_atom(self) -> float:

        if not self.natoms:
            return 0.0

        return self.energy_change / self.natoms

    @property
    def volume_change(self) -> float:
        """Fractional change in cell volume."""

        if not self.initial_volume:
            return 0.0

        return (
            self.final_volume - self.initial_volume
        ) / self.initial_volume

    @property
    def max_displacement(self) -> float:

        return max(self.displacements) if self.displacements else 0.0

    @property
    def moved_atoms(self) -> int:
        """How many atoms travelled far enough to matter."""

        return sum(1 for value in self.displacements if value > MOVED)

    @property
    def lost_symmetry(self) -> bool:
        """True when the relaxation dropped to a lower space group."""

        return (
            self.spacegroup_number_before > 1
            and self.spacegroup_number_after < self.spacegroup_number_before
        )

    def as_storage(self) -> dict[str, Any]:
        """Package for :class:`~qvista.models.results.Results`."""

        return {
            "structure": self.structure,
            "energy": self.final_energy,
            "converged": self.converged,
            "result": self,
        }

    def to_csv(self) -> str:
        """The convergence trace, as CSV."""

        lines = [
            "# QVista relaxation",
            f"# force field,{self.force_field}",
            f"# mode,{MODES.get(self.mode, self.mode)}",
            f"# fmax target eV/A,{self.fmax}",
            f"# converged,{self.converged}",
            "",
            "step,energy_ev,max_force_ev_per_ang",
        ]

        for index, energy in enumerate(self.energies):

            force = (
                self.forces[index] if index < len(self.forces) else 0.0
            )

            lines.append(f"{index},{energy:.6f},{force:.6f}")

        return "\n".join(lines) + "\n"

    def to_poscar(self) -> str:
        """The relaxed structure, as a VASP POSCAR."""

        if self.structure is None:
            return ""

        from pymatgen.io.vasp.inputs import Poscar

        return str(Poscar(self.structure))

    def to_xyz_trajectory(self) -> str:
        """The path downhill, as an extended XYZ trajectory."""

        from qvista.analyses.phonon_modes import to_xyz_trajectory

        if not self.frames:
            return ""

        return to_xyz_trajectory(
            self.symbols,
            self.frames,
            cell=self.cell,
            comment="QVista relaxation",
        )


class RelaxAnalysis(Analysis):
    """Relax a structure until the forces on it are small."""

    name = "Relaxation"

    #: A molecule relaxes as happily as a crystal, as long as the cell
    #: is not being asked to move.
    requires_periodic = False

    def __init__(
        self,
        project: Project,
        mode: str = IONS_AND_CELL,
        fmax: float = DEFAULT_FMAX,
        max_steps: int = DEFAULT_STEPS,
        symprec: float = DEFAULT_SYMPREC,
    ) -> None:

        super().__init__(project, symprec)

        self.mode = mode
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)

    # ------------------------------------------------------------

    def _get_calculator(self):
        """Return the force calculator. Overridden in tests."""

        from qvista.services import forcefield

        try:
            return forcefield.get_calculator()

        except forcefield.ForceFieldUnavailable as exc:
            raise AnalysisError(str(exc)) from exc

    def validate(self) -> None:

        super().validate()

        if self.mode not in MODES:
            raise AnalysisError(f"Unknown relaxation mode: {self.mode}")

        if self.fmax <= 0.0:
            raise AnalysisError(
                "The force target must be positive. It is the largest "
                "force allowed on any atom when the relaxation stops."
            )

        if self.max_steps < 1:
            raise AnalysisError("A relaxation needs at least one step.")

        if self.mode in NEEDS_CELL and not self._periodic():
            raise AnalysisError(
                f"There is no cell to relax: this structure is not "
                f"periodic. {MODES[self.mode]} needs one; choose "
                f"{MODES[IONS_ONLY]} instead."
            )

    def _periodic(self) -> bool:
        """
        Whether there is a cell to relax.

        Read from ``structure_info``, which is the one place that knows:
        QVista's own ``Structure`` wraps an ASE ``Atoms``, and an
        ``Atoms`` built from a plain XYZ still carries a cell -- a zero
        one. Asking it for a lattice therefore answers yes for a
        molecule, and the cell relaxation then divides by that zero
        volume somewhere inside ASE.
        """

        info = self.project.structure_info

        return bool(info is not None and info.periodic)

    # ------------------------------------------------------------

    def execute(self) -> RelaxResult:
        """Walk downhill, recording as we go."""

        import numpy as np

        from ase.optimize import BFGS
        from pymatgen.io.ase import AseAtomsAdaptor

        from qvista.services import forcefield

        structure = self.structure()

        atoms = AseAtomsAdaptor.get_atoms(structure)

        atoms.calc = self._get_calculator()

        start = atoms.get_positions().copy()

        result = RelaxResult(
            mode=self.mode,
            fmax=self.fmax,
            symprec=self.symprec,
            natoms=len(atoms),
            force_field=forcefield.describe(),
            symbols=[str(s) for s in atoms.get_chemical_symbols()],
            initial_volume=float(atoms.get_volume())
            if self._periodic()
            else 0.0,
        )

        before = self._spacegroup(structure)

        # ----------------------------------------------------------
        # What the optimiser is allowed to move
        # ----------------------------------------------------------

        if self.mode in NEEDS_CELL:

            from ase.filters import FrechetCellFilter

            # constant_volume holds the volume while letting the shape
            # change, which is exactly what ISIF 4 does. Without it the
            # two cell modes would be the same calculation.
            target = FrechetCellFilter(
                atoms,
                constant_volume=(self.mode == CONSTANT_VOLUME),
            )

        else:
            target = atoms

        self.report(
            f"Relaxing {len(atoms)} atoms "
            f"({MODES[self.mode].lower()}) with "
            f"{forcefield.describe()}, to {self.fmax} eV/A..."
        )

        try:
            # logfile=None: the optimiser otherwise scatters logs and
            # POSCARs through whatever directory the app was started in.
            optimiser = BFGS(target, logfile=None)

        except Exception as exc:
            raise AnalysisError(
                f"Could not set up the optimiser: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        record_every = max(1, self.max_steps // MAX_FRAMES)

        def observe() -> None:
            """Called by ASE after each step."""

            energy = float(atoms.get_potential_energy())

            forces = float(
                np.linalg.norm(atoms.get_forces(), axis=1).max()
            )

            result.energies.append(energy)
            result.forces.append(forces)

            step = len(result.energies) - 1

            if step % record_every == 0:
                result.frames.append(atoms.get_positions().copy())

            # The optimiser cannot say how far along it is -- that
            # depends on the landscape -- so progress is against the
            # step cap, which is at least an upper bound.
            percent = int(round(100.0 * (step + 1) / self.max_steps))

            self.report(
                f"[{step + 1}] E = {energy:14.6f} eV   "
                f"max|F| = {forces:8.4f} eV/A",
                min(percent, 99),
            )

            self.raise_if_cancelled()

        optimiser.attach(observe, interval=1)

        try:
            optimiser.run(fmax=self.fmax, steps=self.max_steps)

        except Exception as exc:

            # Cancellation travels through the observer, and is not a
            # failure.
            from qvista.analyses.base import AnalysisCancelled

            if isinstance(exc, AnalysisCancelled):
                raise

            raise AnalysisError(
                f"Relaxation failed: {type(exc).__name__}: {exc}"
            ) from exc

        # ----------------------------------------------------------
        # What came out
        # ----------------------------------------------------------

        relaxed = AseAtomsAdaptor.get_structure(atoms)

        after = self._spacegroup(relaxed)

        result.structure = relaxed
        result.steps = len(result.energies)

        result.initial_energy = (
            result.energies[0] if result.energies else 0.0
        )
        result.final_energy = (
            result.energies[-1] if result.energies else 0.0
        )

        result.initial_forces = (
            result.forces[0] if result.forces else 0.0
        )
        result.final_forces = (
            result.forces[-1] if result.forces else 0.0
        )

        result.final_volume = (
            float(atoms.get_volume()) if self._periodic() else 0.0
        )

        result.converged = bool(result.final_forces <= self.fmax)

        result.cell = atoms.get_cell()

        result.spacegroup_before, result.spacegroup_number_before = before
        result.spacegroup_after, result.spacegroup_number_after = after

        # Displacements against the minimum image, so an atom that
        # crossed the cell boundary is not reported as having flown
        # across the structure.
        result.displacements = self._displacements(
            structure, relaxed, start, atoms
        )

        result.rows = self._rows(result)
        result.notes = self._notes(result)

        return result

    # ------------------------------------------------------------

    def _displacements(self, before, after, start, atoms) -> list[float]:
        """How far each atom moved, in angstrom."""

        import numpy as np

        if not self._periodic():

            return [
                float(value)
                for value in np.linalg.norm(
                    atoms.get_positions() - start, axis=1
                )
            ]

        first = np.asarray(before.frac_coords, dtype=float)
        last = np.asarray(after.frac_coords, dtype=float)

        step = last - first
        step -= np.rint(step)

        cartesian = after.lattice.get_cartesian_coords(step)

        return [
            float(value)
            for value in np.linalg.norm(cartesian, axis=1)
        ]

    def _spacegroup(self, structure) -> tuple[str, int]:
        """Name the space group, or say nothing when it cannot be found."""

        if not self._periodic():
            return ("", 0)

        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            analyzer = SpacegroupAnalyzer(structure, symprec=self.symprec)

            return (
                str(analyzer.get_space_group_symbol()),
                int(analyzer.get_space_group_number()),
            )

        except Exception:  # pragma: no cover - depends on the structure
            return ("unknown", 0)

    # ------------------------------------------------------------

    def _rows(self, result: RelaxResult) -> list[tuple[str, str]]:
        """Summary rows."""

        rows = [
            ("Force field", result.force_field),
            ("Relaxed", MODES.get(result.mode, result.mode)),
            ("Equivalent to", f"ISIF = {ISIF.get(result.mode, '?')}"),
            ("Atoms", str(result.natoms)),
            ("Steps", str(result.steps)),
            (
                "Converged",
                "yes"
                if result.converged
                else f"NO - still {result.final_forces:.4f} eV/A",
            ),
            ("Force target", f"{result.fmax} eV/A"),
            ("Max force before", f"{result.initial_forces:.4f} eV/A"),
            ("Max force after", f"{result.final_forces:.4f} eV/A"),
            ("Energy before", f"{result.initial_energy:.6f} eV"),
            ("Energy after", f"{result.final_energy:.6f} eV"),
            (
                "Energy recovered",
                f"{result.energy_change:.6f} eV "
                f"({result.energy_per_atom * 1000:.2f} meV/atom)",
            ),
        ]

        if result.initial_volume:
            rows.extend(
                [
                    (
                        "Volume before",
                        f"{result.initial_volume:.3f} A^3",
                    ),
                    ("Volume after", f"{result.final_volume:.3f} A^3"),
                    (
                        "Volume change",
                        f"{result.volume_change * 100:+.2f}%",
                    ),
                ]
            )

        rows.extend(
            [
                (
                    "Largest displacement",
                    f"{result.max_displacement:.4f} A",
                ),
                (
                    "Atoms that moved",
                    f"{result.moved_atoms} of {result.natoms} "
                    f"(over {MOVED} A)",
                ),
            ]
        )

        if result.spacegroup_before:
            rows.extend(
                [
                    (
                        "Space group before",
                        f"{result.spacegroup_before} "
                        f"(#{result.spacegroup_number_before})",
                    ),
                    (
                        "Space group after",
                        f"{result.spacegroup_after} "
                        f"(#{result.spacegroup_number_after})",
                    ),
                ]
            )

        return rows

    def _notes(self, result: RelaxResult) -> list[str]:
        """Diagnostics, in the order they matter."""

        notes: list[str] = []

        if not result.converged:
            notes.append(
                f"WARNING: the relaxation stopped at "
                f"{result.final_forces:.4f} eV/A, above the target of "
                f"{result.fmax}. It ran out of steps rather than "
                f"finishing. Raise the step cap, or loosen the target."
            )

        if result.lost_symmetry:
            notes.append(
                f"WARNING: the space group dropped from "
                f"{result.spacegroup_before} "
                f"(#{result.spacegroup_number_before}) to "
                f"{result.spacegroup_after} "
                f"(#{result.spacegroup_number_after}). An optimiser does "
                f"not know about symmetry, and noise in the forces "
                f"breaks it. Every calculation after this one will use "
                f"the lower symmetry and cost far more. Re-symmetrise "
                f"before going on, or relax at a looser tolerance."
            )

        if (
            result.mode == CONSTANT_VOLUME
            and abs(result.volume_change) > 1e-4
        ):
            notes.append(
                f"WARNING: the volume changed by "
                f"{result.volume_change * 100:+.3f}% under a "
                f"constant-volume relaxation, so the constraint did "
                f"not hold. Treat the cell as unconstrained."
            )

        elif abs(result.volume_change) > VOLUME_WARNING:
            notes.append(
                f"The cell volume changed by "
                f"{result.volume_change * 100:+.1f}%. That is the force "
                f"field disagreeing with whatever produced the input "
                f"geometry, not a correction to it."
            )

        if result.energy_change > 0.0:
            notes.append(
                "WARNING: the energy went up. A relaxation should only "
                "go downhill, so this means the optimiser stepped past "
                "a minimum -- treat the result with suspicion."
            )

        notes.append(
            "Relaxed with a machine-learning force field, not DFT. This "
            "is a good starting geometry for a VASP relaxation, not a "
            "replacement for one: the bond lengths carry the model's "
            "error, typically a few hundredths of an angstrom."
        )

        return notes

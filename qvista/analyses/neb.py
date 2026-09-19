"""
Nudged elastic band: minimum energy paths and migration barriers.

Given a start and an end structure -- an ion in two neighbouring sites, say
-- NEB finds the lowest-energy way between them and reports the barrier.
For a cathode material that barrier is what governs how fast the ion can
move, so it is often the number the whole calculation exists to produce.

Forces come from a machine-learning force field, **not** DFT. See
:mod:`qvista.services.forcefield`. Barriers from a surrogate model are a
screening result: useful for comparing candidates, not for publication
numbers.

How it works
------------
A chain of intermediate images is interpolated between the endpoints and
connected by springs. Each image relaxes downhill, while the springs stop
them all sliding into the endpoints. With the climbing image switched on,
the highest image instead climbs *up* the path to sit exactly on the saddle
point, which is what makes the barrier accurate rather than merely an upper
bound from whichever image happened to land nearest the top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qvista.analyses.base import (
    INDETERMINATE,
    Analysis,
    AnalysisCancelled,
    AnalysisError,
    AnalysisResult,
)
from qvista.models.project import Project

#: Intermediate images between the two endpoints.
DEFAULT_IMAGES = 5

#: Force convergence, eV/Angstrom.
DEFAULT_FMAX = 0.05

#: Cap on optimiser steps.
DEFAULT_STEPS = 100

#: Spring constant between images, eV/Angstrom^2.
DEFAULT_SPRING = 0.1


# ------------------------------------------------------------
# Result
# ------------------------------------------------------------


@dataclass
class NEBResult(AnalysisResult):
    """The relaxed path, its energies, and the barriers they imply."""

    #: Energy of each image relative to the first, in eV.
    energies: list[float] = field(default_factory=list)

    #: Cumulative distance along the path, in Angstrom.
    coordinates: list[float] = field(default_factory=list)

    barrier_forward: float = 0.0

    #: The peak of the force-informed spline, in eV. Higher than the
    #: highest image whenever the saddle falls between two of them,
    #: which is usual unless a climbing image sat on it.
    barrier_spline: float = float("nan")

    barrier_reverse: float = 0.0

    #: (image, distance, energy, force) per image, the table VTST
    #: writes as neb.dat.
    dat: list = field(default_factory=list)

    #: (distance, energy) of the force-informed spline.
    spline: tuple = ()

    #: The atom the path is about, and what it is bonded to at the
    #: start, the saddle and the end.
    moving_atom: int | None = None
    environments: list = field(default_factory=list)

    #: One structure per image, kept so the coordination can be recounted
    #: at a different bond cutoff without reading the folder again.
    structures: list = field(default_factory=list)

    #: The cutoff the environments above were counted at, in Angstrom.
    neighbour_cutoff: float = 0.0
    reaction_energy: float = 0.0

    #: Index of the highest image.
    saddle_index: int = 0

    n_images: int = 0
    climbing: bool = True

    max_force: float = 0.0
    converged: bool = False

    #: The relaxed images, as ASE Atoms.
    images: list = field(default_factory=list)

    #: Every progress message the run published, in order.
    log: list = field(default_factory=list)

    force_field: str = ""

    def as_storage(self) -> dict[str, Any]:
        """Package for :class:`~qvista.models.results.Results`."""

        return {
            "energies": self.energies,
            "barrier_forward": self.barrier_forward,
            "barrier_reverse": self.barrier_reverse,
            "reaction_energy": self.reaction_energy,
            "result": self,
        }

    def to_csv(self) -> str:
        """Render the energy profile as CSV."""

        lines = [
            "# QVista NEB minimum energy path",
            f"# force field,{self.force_field}",
            f"# climbing image,{self.climbing}",
            f"# forward barrier eV,{self.barrier_forward:.6f}",
            f"# reverse barrier eV,{self.barrier_reverse:.6f}",
            f"# reaction energy eV,{self.reaction_energy:.6f}",
            "",
            "image,reaction_coordinate_ang,energy_ev",
        ]

        for index, energy in enumerate(self.energies):

            coordinate = (
                self.coordinates[index]
                if index < len(self.coordinates)
                else float(index)
            )

            lines.append(f"{index},{coordinate:.6f},{energy:.6f}")

        return "\n".join(lines) + "\n"

    def to_xyz_trajectory(self) -> str:
        """Render the path as an extended XYZ trajectory."""

        from qvista.analyses.phonon_modes import to_xyz_trajectory

        if not self.images:
            return ""

        first = self.images[0]

        return to_xyz_trajectory(
            [str(symbol) for symbol in first.get_chemical_symbols()],
            [image.get_positions() for image in self.images],
            cell=first.get_cell(),
            comment="QVista NEB path",
        )


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------


class NEBAnalysis(Analysis):
    """Relax a minimum energy path between two structures."""

    name = "NEB"

    #: NEB is defined for molecules as well as crystals.
    requires_periodic = False

    def __init__(
        self,
        project: Project,
        final_path: str | Path,
        n_images: int = DEFAULT_IMAGES,
        fmax: float = DEFAULT_FMAX,
        steps: int = DEFAULT_STEPS,
        climb: bool = True,
        spring: float = DEFAULT_SPRING,
    ) -> None:

        super().__init__(project)

        self.final_path = Path(final_path)
        self.n_images = int(n_images)
        self.fmax = fmax
        self.steps = int(steps)
        self.climb = climb
        self.spring = spring

        self._log: list[str] = []

    def report(self, message: str, percent: int = INDETERMINATE) -> None:
        """
        Publish progress, and keep it.

        The step-by-step record is the only account of how the band got
        where it did -- what the residual was doing, whether it stalled,
        how many steps it took. It scrolls out of a console and vanishes
        with the window, so it is kept here and written beside the
        result for anyone who has to defend the barrier later.
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

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def validate(self) -> None:
        """Check the two endpoints describe the same system."""

        super().validate()

        if self.n_images < 1:
            raise AnalysisError(
                "A NEB path needs at least one intermediate image."
            )

        initial, final = self.endpoints()

        if len(initial) != len(final):
            raise AnalysisError(
                f"The two structures have different atom counts "
                f"({len(initial)} and {len(final)}).\n\n"
                f"NEB moves the same atoms from one arrangement to "
                f"another, so both endpoints must contain the same atoms."
            )

        if initial.get_chemical_symbols() != final.get_chemical_symbols():

            if sorted(initial.get_chemical_symbols()) == sorted(
                final.get_chemical_symbols()
            ):
                raise AnalysisError(
                    "The endpoints contain the same elements but list them "
                    "in a different order.\n\n"
                    "NEB matches atoms by position in the file, so atom 1 "
                    "in the start must be the same atom as atom 1 in the "
                    "end. Reorder one of the files to match the other."
                )

            raise AnalysisError(
                "The two structures have different compositions, so there "
                "is no path between them."
            )

    def endpoints(self):
        """Return the start and end structures as ASE atoms."""

        if getattr(self, "_endpoints", None) is not None:
            return self._endpoints

        from pymatgen.io.ase import AseAtomsAdaptor

        from qvista.services.structure_analyzer import (
            StructureAnalysisError,
            StructureAnalyzer,
        )

        source = self.project.info.source_file

        if source is None:
            raise AnalysisError(
                "This project has no structure file to start from."
            )

        try:
            start = StructureAnalyzer.parse(source)
            end = StructureAnalyzer.parse(self.final_path)

        except StructureAnalysisError as exc:
            raise AnalysisError(str(exc)) from exc

        try:
            initial = AseAtomsAdaptor.get_atoms(start.structure)
            final = AseAtomsAdaptor.get_atoms(end.structure)

        except Exception as exc:
            raise AnalysisError(
                f"Could not convert the endpoints for NEB: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Both of these compare the two endpoints atom by atom, so
        # neither can run on a mismatched pair. Leaving that to the
        # check further down keeps the error the one that says what is
        # actually wrong, rather than a shape mismatch from in here.
        if len(initial) == len(final):

            self._share_cell(initial, final)
            self._shortest_path(initial, final)

        self._endpoints = (initial, final)

        return self._endpoints

    #: How far the two endpoint cells may differ before it is worth
    #: saying so, in Angstrom. Below this it is relaxation noise; above
    #: it, the two endpoints are arguably different structures.
    CELL_TOLERANCE = 0.01

    def _share_cell(self, initial, final) -> None:
        """
        Put both endpoints on one cell, keeping fractional positions.

        A NEB has a single cell. VASP enforces that by construction --
        one POSCAR per image, ISIF = 2 -- and ASE enforces it by
        refusing to build a band whose images disagree, to within 1e-8
        Angstrom. Two endpoints relaxed separately with the cell free
        never agree to that: this pair differed by 1.3e-3 Angstrom,
        which is ordinary relaxation noise and five orders of magnitude
        past what ASE accepts.

        The final endpoint is moved onto the initial cell by its
        *fractional* coordinates, not its Cartesian ones. The atom is in
        a crystallographic site, and the site is what the path runs
        between; carrying Cartesian positions across would shift every
        atom by the cell difference and count that as part of the
        migration.
        """

        import numpy as np

        start = np.asarray(initial.cell.array, dtype=float)
        end = np.asarray(final.cell.array, dtype=float)

        difference = float(np.abs(end - start).max())

        if difference <= 0.0:
            return

        if difference > self.CELL_TOLERANCE:
            self.report(
                f"WARNING: the endpoint cells differ by "
                f"{difference:.4f} Angstrom. NEB uses one cell for the "
                f"whole band, so the final endpoint has been placed on "
                f"the initial cell. A difference this large usually "
                f"means the two endpoints relaxed into different "
                f"structures rather than two sites of the same one."
            )

        else:
            self.report(
                f"The endpoint cells differ by {difference:.2e} "
                f"Angstrom. Using the initial cell for the whole band, "
                f"as NEB requires."
            )

        fractional = final.get_scaled_positions()

        final.set_cell(start)
        final.set_scaled_positions(fractional)

    def _shortest_path(self, initial, final) -> None:
        """
        Rewrap the final endpoint to the nearest image of the first.

        Without this, an atom whose fractional coordinate crosses a cell
        face is written at the far side -- 0.999 against 0.001 -- and a
        straight interpolation drags it the long way round, right across
        the cell, through every atom in between. The band then relaxes a
        path nobody asked about and the barrier is meaningless.

        It is not a hypothetical. On the pair this was first run on, one
        oxygen appeared to move 5.684 Angstrom -- exactly the *a*
        lattice parameter, which is the signature of a wrap rather than
        a migration. Minimum-imaged, it does not move at all, and the
        only atom going anywhere is the lithium, by 2.84 Angstrom.

        The same convention the VASP route interpolates with, so the two
        agree about which path they are describing.
        """

        import numpy as np

        start = initial.get_scaled_positions()
        end = final.get_scaled_positions()

        step = end - start

        # Nearest image of each displacement: whichever periodic copy of
        # the final position is closest to where the atom started.
        step -= np.round(step)

        final.set_scaled_positions(start + step)

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    def execute(self) -> NEBResult:
        """Interpolate, relax, and measure the barrier."""

        import numpy as np

        from ase.mep import NEB
        from ase.optimize import FIRE

        from qvista.services import forcefield

        initial, final = self.endpoints()

        # Before loading, not after: the first model load is several
        # seconds of silence otherwise, which reads as a hang.
        self.report("Loading the force field...")

        self._get_calculator()

        self.report(
            f"Building {self.n_images} intermediate images "
            f"using {forcefield.describe()}."
        )

        # ----------------------------------------------------------
        # Build and interpolate the chain
        # ----------------------------------------------------------

        images = [initial]
        images += [initial.copy() for _ in range(self.n_images)]
        images += [final]

        for image in images:
            # One calculator each, not one between them. ASE caches a
            # calculator's results against the last atoms it saw, so a
            # shared one is evicted on every image's turn and the
            # endpoints get recomputed every step -- measured at 11
            # force evaluations per step where 2 were needed. These
            # share the loaded weights, so the extra objects are
            # bookkeeping rather than a second copy of the network.
            image.calc = self._get_calculator()

        try:
            band = NEB(
                images,
                k=self.spring,
                climb=self.climb,
                # ASE's old default is unpublished and, in its own
                # words, "frequently results in very poor bands".
                method="improvedtangent",
            )

            band.interpolate()

        except Exception as exc:
            raise AnalysisError(
                f"Could not set up the NEB chain: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # Relax
        # ----------------------------------------------------------

        self.report("Relaxing the path...")

        optimiser = FIRE(band, logfile=None)

        converged = self._relax(optimiser, band)

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # Energies
        # ----------------------------------------------------------

        self.report("Collecting energies...")

        try:
            energies = [
                float(image.get_potential_energy()) for image in images
            ]

            forces = float(
                np.abs(
                    np.concatenate(
                        [image.get_forces() for image in images[1:-1]]
                    )
                ).max()
            )

        except Exception as exc:
            raise AnalysisError(
                f"Could not evaluate the relaxed path: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        reference = energies[0]

        relative = [energy - reference for energy in energies]

        result = NEBResult(
            energies=relative,
            coordinates=self._path_coordinates(images),
            barrier_forward=max(relative),
            barrier_reverse=max(relative) - relative[-1],
            reaction_energy=relative[-1],
            saddle_index=int(np.argmax(relative)),
            n_images=len(images),
            climbing=self.climb,
            max_force=forces,
            converged=converged,
            images=images,
            force_field=forcefield.describe(),
        )

        # The same fields the VASP route fills, so the same tables and
        # the same figure serve both.
        self._enrich(result, images)

        result.rows = self._rows(result)
        result.notes = self._notes(result)

        result.log = list(self._log)

        return result

    def _enrich(self, result, images) -> None:
        """
        Turn the relaxed band into what the post-processing views read.

        A failure here costs the coordination analysis and the spline,
        not the barrier -- the path is already relaxed and the energies
        are already in hand, and losing the extras should not lose that.
        """

        from pymatgen.io.ase import AseAtomsAdaptor

        from qvista.services import neb_vasp

        try:
            structures = [
                AseAtomsAdaptor.get_structure(image) for image in images
            ]

            forces = [image.get_forces() for image in images]

            neb_vasp.enrich(result, structures, forces)

        except Exception as exc:  # noqa: BLE001 - the barrier stands
            self.report(
                f"Path analysis skipped: {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------

    def _relax(self, optimiser, band) -> bool:
        """
        Run the optimiser, reporting progress and honouring cancellation.

        ``irun`` is a generator that yields after every step, so progress
        can be published and cancellation checked between iterations
        without taking over the optimiser's own convergence logic.

        The residual -- the largest force on any image -- comes from the
        band itself rather than the optimiser, which does not expose it.
        """

        import time

        step = 0

        started = time.perf_counter()

        try:
            for _ in optimiser.irun(fmax=self.fmax, steps=self.steps):

                self.raise_if_cancelled()

                step += 1

                force = float(band.get_residual())

                percent = int(round(100.0 * step / self.steps))

                elapsed = time.perf_counter() - started

                # The rate as well as the count: a machine-learning
                # potential on a CPU is slow enough that "is it moving"
                # is a fair question, and a seconds-per-step figure
                # answers it and lets the wait be estimated.
                self.report(
                    f"[{step}/{self.steps}] max|F| = {force:.4f} eV/Å "
                    f"(target {self.fmax:g}) "
                    f"-- {elapsed:.0f} s, {elapsed / step:.1f} s/step",
                    min(percent, 100),
                )

        except AnalysisError:
            raise

        except Exception as exc:
            if isinstance(exc, AnalysisCancelled):
                raise

            raise AnalysisError(
                f"NEB relaxation failed at step {step + 1}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        converged = float(band.get_residual()) < self.fmax

        if converged:
            self.report(
                f"Converged after {step} steps "
                f"(max|F| < {self.fmax} eV/Å)."
            )

        return converged

    def _path_coordinates(self, images) -> list[float]:
        """Cumulative distance along the path, in Angstrom."""

        import numpy as np

        coordinates = [0.0]

        for previous, current in zip(images[:-1], images[1:]):

            delta = current.get_positions() - previous.get_positions()

            coordinates.append(
                coordinates[-1] + float(np.linalg.norm(delta))
            )

        return coordinates

    # ------------------------------------------------------------
    # Console text
    # ------------------------------------------------------------

    def _rows(self, result: NEBResult) -> list[tuple[str, str]]:
        """
        Summary rows -- the same six the VASP route reports.

        A band relaxed here and a folder of VASP output answer the same
        question, so they get the same table rather than two that have
        to be read differently. What is dropped in the process is the
        convergence pair: a run that did not converge says so as a
        warning above the table, which is harder to miss than a row.
        """

        from qvista.services import neb_vasp

        return neb_vasp.summary_rows(result, result.n_images)

    def _notes(self, result: NEBResult) -> list[str]:
        """The profile, plus warnings."""

        notes: list[str] = []

        lines = ["Energy profile (eV relative to the start)"]

        for index, energy in enumerate(result.energies):

            marker = "  <- saddle" if index == result.saddle_index else ""

            lines.append(f"  image {index:>2}: {energy:9.4f}{marker}")

        notes.append("\n".join(lines))

        # ----------------------------------------------------------
        # Warnings
        # ----------------------------------------------------------

        if not result.converged:
            notes.append(
                f"WARNING: the path did not converge within {self.steps} "
                f"steps; the largest remaining force is "
                f"{result.max_force:.4f} eV/Å against a target of "
                f"{self.fmax}. The barrier below is not yet reliable -- "
                f"raise the step limit and run again."
            )

        if result.saddle_index in (0, result.n_images - 1):
            notes.append(
                "WARNING: the highest point of the path is an endpoint, so "
                "no barrier was found between them. Either the endpoints "
                "are not separated by one, or one of them is not a relaxed "
                "structure."
            )

        if not result.climbing:
            notes.append(
                "The climbing image was off, so the barrier is a lower "
                "bound: the true saddle point lies between images unless "
                "one happened to land exactly on it."
            )

        notes.append(
            "Energies came from a machine-learning force field, not DFT. "
            "Treat this barrier as a screening result."
        )

        return notes

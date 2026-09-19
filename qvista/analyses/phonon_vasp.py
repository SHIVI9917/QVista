"""
The back half of a VASP phonon calculation.

The front half -- laying out the run folders -- is
:mod:`qvista.services.phonopy_vasp`, driven straight from the dialog,
because it is file writing and finishes in a moment.
:class:`VaspPhononCollect` is the other end: days later, once VASP has
run somewhere else, it reads the same folder back and turns it into
force constants, a band structure and a density of states.

The two ends never talk to each other directly. Everything that has to
survive in between is on disk, in ``phonopy_disp.yaml``: the unit cell,
the supercell matrix, and every displacement phonopy chose. That is why
post-processing works on a folder someone else ran, on a different
machine, with QVista closed in between.

Results come back as the same :class:`~qvista.analyses.phonon.PhononData`
and :class:`~qvista.analyses.phonon.PhononResult` the machine-learning
route produces, so the plotting, the console output and the mode
animation are shared rather than reimplemented. Only
:attr:`~qvista.analyses.phonon.PhononData.force_field` differs, and it
says which of the two ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from qvista.analyses.base import Analysis, AnalysisError
from qvista.analyses.kpath import PrimitiveCell
from qvista.analyses.phonon import (
    BAND_NPOINTS,
    DEFAULT_MESH,
    MODE_BOTH,
    PhononData,
    PhononResult,
    project_eigenvectors,
)
from qvista.models.project import Project
from qvista.services import phonopy_vasp
from qvista.services.phonopy_vasp import (
    BAND_YAML,
    BORN_FILE,
    DFPT,
    FINITE_DISPLACEMENT,
    FORCE_CONSTANTS,
    FORCE_SETS,
    THERMAL_YAML,
    TOTAL_DOS,
)
from qvista.services.structure_analyzer import DEFAULT_SYMPREC

#: Stamped on every figure drawn from these force constants.
FORCE_LABEL = {
    FINITE_DISPLACEMENT: "VASP, finite displacement (first principles)",
    DFPT: "VASP, DFPT / IBRION = 8 (first principles)",
}

#: Below this, in THz, a frequency is taken as genuinely imaginary rather
#: than the numerical residue of the acoustic branches at Gamma.
IMAGINARY_THRESHOLD = -0.1


# ----------------------------------------------------------------
# Reading it back
# ----------------------------------------------------------------


@dataclass
class CollectResult(PhononResult):
    """
    Force constants from VASP, plus band and DOS.

    Subclasses the machine-learning route's result so the existing
    plotting and caching apply unchanged.
    """

    #: The folder this came out of.
    root: Path | None = None

    #: Files post-processing wrote into it.
    written: list[Path] = field(default_factory=list)

    #: Lowest frequency anywhere on the path, in THz. Negative means an
    #: imaginary mode.
    lowest: float | None = None

    #: True when the non-analytical correction was applied.
    nac: bool = False

    #: Harmonic thermodynamics on the same mesh as the DOS, when they
    #: were asked for.
    thermal: object | None = None


class VaspPhononCollect(Analysis):
    """
    Turn a finished configuration into force constants, band and DOS.

    The chain is the one phonopy documents: ``vasprun.xml`` from every
    run, then FORCE_SETS (or, for DFPT, force constants straight from the
    Hessian), then the dynamical matrix at every q-point on the path,
    then ``band.yaml`` and the mesh-sampled density of states. Each
    intermediate is written into the configuration folder next to the
    runs that produced it, so the result can be checked or re-plotted
    with phonopy's own tools.
    """

    name = "Phonon Post-Processing (VASP)"

    #: There is no project structure involved: everything comes from the
    #: folder, which is the point.
    requires_periodic = False

    def __init__(
        self,
        project: Project | None,
        root: Path,
        mesh: Sequence[int] = DEFAULT_MESH,
        npoints: int = BAND_NPOINTS,
        use_born: bool = True,
        symprec: float = DEFAULT_SYMPREC,
        thermal: bool = True,
        t_max: float = 1000.0,
        t_step: float = 10.0,
    ) -> None:

        super().__init__(project, symprec)

        self.root = Path(root)
        self.mesh = tuple(int(value) for value in mesh)
        self.npoints = int(npoints)
        self.use_born = use_born

        self.thermal = bool(thermal)
        self.t_max = float(t_max)
        self.t_step = float(t_step)

    def validate(self) -> None:
        """The folder stands in for the project here."""

        if not self.root.is_dir():
            raise AnalysisError(f"{self.root} is not a folder.")

    # ------------------------------------------------------------

    def execute(self) -> CollectResult:
        """Run the whole chain."""

        found = phonopy_vasp.discover(self.root)

        if found.problems:
            raise AnalysisError("\n\n".join(found.problems))

        # Discovery accepts the experiment folder, or the directory
        # holding it, and walks down to the configuration. Everything
        # below reads and writes beside the runs, so it has to follow --
        # otherwise phonopy_disp.yaml is looked for one level too high
        # and the outputs land outside the experiment.
        if found.root != self.root:

            self.report(f"Using the configuration in {found.root.name}.")

            self.root = found.root

        if found.siblings:
            self.report(
                f"{len(found.siblings)} configurations found; taking "
                f"{found.root.name}."
            )

        if found.missing:
            raise AnalysisError(
                f"{len(found.missing)} run folder(s) have no "
                f"vasprun.xml yet: {', '.join(found.missing[:10])}"
                + (" ..." if len(found.missing) > 10 else "")
                + "\n\nEvery displacement has to be finished before the "
                "force constants can be built -- phonopy has no way to "
                "fill in a missing one."
            )

        self.report(f"Reading {found.method} configuration...")

        phonon = self._load(found)

        self.raise_if_cancelled()

        written: list[Path] = []

        # ----------------------------------------------------------
        # Forces -> force constants
        # ----------------------------------------------------------

        if found.method == DFPT:
            written += self._force_constants_from_hessian(phonon, found)

        else:
            written += self._force_constants_from_forces(phonon, found)

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # Born charges
        # ----------------------------------------------------------

        nac = False

        if self.use_born and found.born is not None:
            nac, born_written = self._apply_born(phonon, found)
            written += born_written

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # Band and DOS
        # ----------------------------------------------------------

        band = self._run_band(phonon)

        written.append(self._write_band_yaml(phonon))

        self.raise_if_cancelled()

        dos = self._run_dos(phonon)

        written.append(self._write_total_dos(phonon))

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # Thermodynamics, off the same mesh
        # ----------------------------------------------------------

        thermal = None

        if self.thermal:

            thermal = self._run_thermal(phonon)

            if thermal is not None:
                written.append(self._write_thermal(phonon))

        # ----------------------------------------------------------

        data = PhononData(
            phonon=phonon,
            supercell=found.supercell,
            symprec=self.symprec,
            distance=_displacement_distance(phonon),
            primitive=_primitive_of(phonon),
            stretches=[],
            n_displacements=len(found.found),
            supercell_natoms=len(phonon.supercell),
            force_field=FORCE_LABEL.get(found.method, "VASP"),
            source=str(self.root),
        )

        result = CollectResult(
            phonon_data=data,
            band=band,
            dos=dos,
            mode=MODE_BOTH,
            root=self.root,
            written=[path for path in written if path is not None],
            nac=nac,
            thermal=thermal,
        )

        result.lowest = _minimum(band.get("frequencies"))

        result.rows = self._rows(result, found)
        result.notes = self._notes(result, found)

        return result

    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------

    def _load(self, found):
        """Rebuild the Phonopy object from ``phonopy_disp.yaml``."""

        from phonopy import load as phonopy_load

        try:
            return phonopy_load(
                self.root / phonopy_vasp.DISP_YAML,
                produce_fc=False,
                log_level=0,
            )

        except Exception as exc:
            raise AnalysisError(
                f"Could not read "
                f"{self.root / phonopy_vasp.DISP_YAML}: "
                f"{type(exc).__name__}: {exc}\n\n"
                f"That file is phonopy's record of the supercell and the "
                f"displacements. Without it the vasprun.xml files cannot "
                f"be interpreted, because nothing else says which atom "
                f"each one moved."
            ) from exc

    # ------------------------------------------------------------
    # Force constants
    # ------------------------------------------------------------

    def _force_constants_from_forces(self, phonon, found) -> list[Path]:
        """Finite displacement: forces from each run, then FORCE_SETS."""

        from phonopy.file_IO import write_FORCE_CONSTANTS, write_FORCE_SETS
        from phonopy.interface.vasp import parse_set_of_forces

        natoms = len(phonon.supercell)

        self.report(
            f"Reading forces from {len(found.found)} vasprun.xml "
            f"files ({natoms} atoms each)..."
        )

        try:
            dataset = parse_set_of_forces(
                natoms,
                found.found,
                verbose=False,
            )

        except Exception as exc:
            raise AnalysisError(
                f"Could not read the forces: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        forces = dataset.get("forces") if dataset else None

        if not forces:
            raise AnalysisError(
                "phonopy could not read forces from these vasprun.xml "
                "files.\n\nThe usual cause is a run that did not finish: "
                "an xml written by a job the queue killed is well-formed "
                "up to the point it stopped and has no forces in it. "
                "Check the tail of each OUTCAR."
            )

        if len(forces) != len(phonon.displacements):
            raise AnalysisError(
                f"{len(forces)} force sets were read but the "
                f"configuration has {len(phonon.displacements)} "
                f"displacements. The folder has been added to or taken "
                f"from since it was written."
            )

        self.raise_if_cancelled()

        self.report("Producing force constants...")

        try:
            phonon.forces = forces
            phonon.produce_force_constants()

        except Exception as exc:
            raise AnalysisError(
                f"Could not produce force constants: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        written: list[Path] = []

        try:
            write_FORCE_SETS(phonon.dataset, self.root / FORCE_SETS)
            written.append(self.root / FORCE_SETS)

            write_FORCE_CONSTANTS(
                phonon.force_constants,
                self.root / FORCE_CONSTANTS,
            )
            written.append(self.root / FORCE_CONSTANTS)

        except OSError as exc:
            self.report(f"Could not write into {self.root}: {exc}")

        return written

    def _force_constants_from_hessian(self, phonon, found) -> list[Path]:
        """DFPT: the Hessian is already in the one vasprun.xml."""

        import numpy as np

        from phonopy.file_IO import write_FORCE_CONSTANTS
        from phonopy.interface.vasp import parse_force_constants

        vasprun = found.found[0]

        self.report(f"Reading the Hessian from {vasprun.name}...")

        try:
            constants, _elements = parse_force_constants(vasprun)

        except Exception as exc:
            raise AnalysisError(
                f"Could not read force constants from {vasprun}: "
                f"{type(exc).__name__}: {exc}\n\n"
                f"A DFPT run writes them into a <dynmat> block at the "
                f"very end of vasprun.xml. If the job was killed before "
                f"it finished, the block is simply not there."
            ) from exc

        natoms = len(phonon.supercell)

        constants = np.asarray(constants, dtype=float)

        if constants.shape[:2] != (natoms, natoms):
            raise AnalysisError(
                f"The Hessian in {vasprun.name} is for "
                f"{constants.shape[0]} atoms, but the supercell in "
                f"{phonopy_vasp.DISP_YAML} has {natoms}.\n\n"
                f"The DFPT run has to be on SPOSCAR exactly as written."
            )

        phonon.force_constants = constants

        written: list[Path] = []

        try:
            write_FORCE_CONSTANTS(
                phonon.force_constants,
                self.root / FORCE_CONSTANTS,
            )
            written.append(self.root / FORCE_CONSTANTS)

        except OSError as exc:
            self.report(f"Could not write into {self.root}: {exc}")

        return written

    # ------------------------------------------------------------
    # Born effective charges
    # ------------------------------------------------------------

    def _apply_born(self, phonon, found) -> tuple[bool, list[Path]]:
        """
        Read the born/ run and switch the non-analytical correction on.

        Failure here is reported and stepped over rather than raised: the
        dispersion without the correction is still a result, and it is
        better than nothing on the screen.
        """

        import numpy as np

        from phonopy.file_IO import parse_BORN, write_BORN
        from phonopy.interface.vasp import get_born_vasprunxml

        self.report("Reading Born effective charges...")

        try:
            borns, epsilon, _indices = get_born_vasprunxml(
                found.born,
                # The born run is on the unit cell, so there is no
                # supercell to trace through and no further reduction.
                primitive_matrix=np.eye(3),
                supercell_matrix=np.eye(3),
                symprec=self.symprec,
            )

            write_BORN(
                phonon.primitive,
                borns,
                epsilon,
                filename=self.root / BORN_FILE,
            )

            phonon.nac_params = parse_BORN(
                phonon.primitive,
                symprec=self.symprec,
                filename=self.root / BORN_FILE,
            )

        except Exception as exc:
            self.report(
                f"Born charges could not be applied "
                f"({type(exc).__name__}: {exc}). Continuing without the "
                f"non-analytical correction."
            )
            return False, []

        return True, [self.root / BORN_FILE]

    # ------------------------------------------------------------
    # Band structure
    # ------------------------------------------------------------

    def _run_band(self, phonon):
        """
        Evaluate the dynamical matrix along phonopy's own path.

        ``get_band_qpoints_by_seekpath`` is what ``--band auto`` runs on
        the command line: it takes the primitive cell phonopy derived
        from the unit cell and returns the q-points, the labels and the
        breaks in one go. Using it rather than deriving a path here is
        the whole reason nothing needs standardising -- the q-points come
        out of the same cell the dynamical matrix is built on, so the
        two cannot disagree.
        """

        from phonopy.phonon.band_structure import (
            get_band_qpoints_by_seekpath,
        )

        self.report("Working out the high-symmetry path...")

        try:
            qpoints, labels, connections = get_band_qpoints_by_seekpath(
                phonon.primitive, self.npoints
            )

        except Exception as exc:
            raise AnalysisError(
                f"Could not determine the k-path: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not qpoints:
            raise AnalysisError(
                "seekpath returned no path for this lattice, so there "
                "is nothing to plot a band structure along."
            )

        self.report("Running band structure...")

        try:
            # Eigenvectors as well as frequencies. They are what the
            # element projection is built from, and they come out of a
            # diagonalisation that is happening either way -- so asking
            # for them here is what lets the projected figure be a
            # redraw of this result rather than a second pass over the
            # whole path.
            phonon.run_band_structure(
                qpoints,
                path_connections=connections,
                labels=labels,
                with_eigenvectors=True,
            )

            structure = phonon.band_structure

        except Exception as exc:
            raise AnalysisError(
                f"Band structure calculation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        symbols = [str(symbol) for symbol in phonon.primitive.symbols]

        band = {
            "qpoints": structure.qpoints,
            "distances": structure.distances,
            "frequencies": structure.frequencies,
            "labels": list(labels),
            "path_connections": list(connections),
            "symbols": symbols,
        }

        self.report("Projecting eigenvectors onto elements...")

        try:
            band["projections"] = project_eigenvectors(
                structure.eigenvectors, symbols
            )

        except Exception as exc:  # noqa: BLE001 - the bands still stand
            # A projection is an extra reading of the same dispersion,
            # not a precondition for it. Losing it should cost the
            # projected figure, not the band structure that is already
            # computed.
            self.report(f"Projection skipped: {type(exc).__name__}: {exc}")

        return band

    def _run_dos(self, phonon):
        """Sample the Brillouin zone and accumulate the total DOS."""

        self.report(
            f"Running {self.mesh[0]}x{self.mesh[1]}x{self.mesh[2]} "
            f"mesh for the density of states..."
        )

        try:
            # Gamma-centred, not phonopy's default half-shifted grid. An
            # even shifted mesh is not preserved by the point group, so
            # phonopy drops the symmetry reduction and samples the whole
            # zone -- slower, and it misses Gamma itself, where the
            # acoustic branches are.
            phonon.run_mesh(
                list(self.mesh),
                is_mesh_symmetry=True,
                is_gamma_center=True,
            )

            phonon.run_total_dos()

            total = phonon.total_dos

        except Exception as exc:
            raise AnalysisError(
                f"Density of states calculation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return {
            "frequency_points": total.frequency_points,
            "total_dos": total.dos,
            "mesh": self.mesh,
        }

    def _run_thermal(self, phonon):
        """
        Harmonic free energy, entropy and heat capacity.

        Off the mesh the DOS just sampled, so the two always describe
        the same Brillouin-zone sampling.  Shared with the
        machine-learning route rather than reimplemented: the physics is
        the same once there are force constants, and only the way they
        were obtained differs.
        """

        from qvista.services import phonon_dos

        self.report(
            f"Integrating thermal properties to {self.t_max:.0f} K..."
        )

        try:
            return phonon_dos.run_thermal(
                phonon, t_max=self.t_max, t_step=self.t_step
            )

        except Exception as exc:  # noqa: BLE001 - the band still stands
            self.report(f"Thermal properties failed: {exc}")
            return None

    # ------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------

    def _write_thermal(self, phonon) -> Path | None:
        """Write phonopy's own thermal_properties.yaml beside the runs."""

        target = self.root / THERMAL_YAML

        try:
            phonon.write_yaml_thermal_properties(filename=target)

        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            self.report(f"Could not write {target.name}: {exc}")
            return None

        return target

    def _write_band_yaml(self, phonon) -> Path | None:
        """Write band.yaml beside the runs."""

        target = self.root / BAND_YAML

        try:
            phonon.write_yaml_band_structure(filename=target)

        except Exception as exc:
            self.report(f"Could not write {target.name}: {exc}")
            return None

        return target

    def _write_total_dos(self, phonon) -> Path | None:
        """Write total_dos.dat beside the runs."""

        target = self.root / TOTAL_DOS

        try:
            phonon.write_total_dos(filename=target)

        except Exception as exc:
            self.report(f"Could not write {target.name}: {exc}")
            return None

        return target

    # ------------------------------------------------------------
    # Console text
    # ------------------------------------------------------------

    def _rows(self, result: CollectResult, found) -> list[tuple[str, str]]:
        """Summary rows for the console."""

        nx, ny, nz = found.supercell
        mx, my, mz = self.mesh

        rows = [
            ("Source", str(self.root)),
            ("Method", found.method),
            ("Forces", FORCE_LABEL.get(found.method, "VASP")),
            ("Supercell", f"{nx} x {ny} x {nz}"),
            ("Supercell atoms", str(found.supercell_natoms)),
        ]

        if found.method == FINITE_DISPLACEMENT:
            rows.append(("Displacements read", str(len(found.found))))

        rows += [
            ("LO-TO correction", "applied" if result.nac else "not applied"),
            ("DOS mesh", f"{mx} x {my} x {mz}"),
            ("Band points", str(self.npoints)),
        ]

        if result.lowest is not None:
            rows.append(
                ("Lowest frequency", f"{result.lowest:.4f} THz")
            )

        thermal = result.thermal

        if thermal is not None:

            rows.append(
                (
                    "Zero-point energy",
                    f"{thermal.zero_point:.4f} kJ/mol",
                )
            )

            try:
                free, entropy, capacity = thermal.at(300.0)

                rows += [
                    ("F at 300 K", f"{free:.4f} kJ/mol"),
                    ("S at 300 K", f"{entropy:.4f} J/K/mol"),
                    (
                        "Cv at 300 K",
                        f"{capacity:.4f} J/K/mol "
                        f"({capacity / thermal.dulong_petit * 100:.1f}% of 3NR)",
                    ),
                ]

            except Exception:  # noqa: BLE001 - the rest of the table stands
                pass

        rows.append(
            (
                "Written",
                ", ".join(path.name for path in result.written) or "nothing",
            )
        )

        return rows

    def _notes(self, result: CollectResult, found) -> list[str]:
        """The verdict, and anything that qualifies it."""

        notes: list[str] = []

        lowest = result.lowest

        if lowest is None:
            notes.append(
                "No frequencies were produced, so nothing can be said "
                "about stability."
            )

        elif lowest < IMAGINARY_THRESHOLD:
            notes.append(
                f"Dynamically UNSTABLE at this level of theory. The "
                f"lowest frequency on the path is {lowest:.3f} THz -- an "
                f"imaginary mode, drawn below zero. The structure sits "
                f"at a saddle point of the energy surface, not a "
                f"minimum: following that mode lowers the energy."
            )
            notes.append(
                "Before concluding that, rule out the cheap causes. A "
                "supercell too small to contain the unstable "
                "wavelength, a geometry not relaxed tightly enough "
                "(EDIFFG around 1E-3 eV/A or better), or a k-mesh too "
                "coarse for the supercell all produce imaginary modes "
                "in a structure that is really stable."
            )

        else:
            notes.append(
                f"Dynamically STABLE at this level of theory. Every "
                f"frequency on the path is real; the lowest is "
                f"{lowest:.3f} THz."
            )
            notes.append(
                "The three acoustic branches go to zero at Gamma by "
                "construction. Small negative values there -- a few "
                "hundredths of a THz -- are the numerical residue of "
                "that, not an instability."
            )

        if not result.nac:
            notes.append(
                "No non-analytical correction was applied. In a polar "
                "crystal that leaves the longitudinal optical branch "
                "degenerate with the transverse ones at Gamma."
            )

        notes.append(
            f"FORCE_SETS, FORCE_CONSTANTS, {BAND_YAML} and {TOTAL_DOS} "
            f"were written into {self.root}, so the same result can be "
            f"re-plotted with phonopy directly."
        )

        return notes


def _primitive_of(phonon) -> PrimitiveCell:
    """
    phonopy's primitive cell, as the rest of QVista describes one.

    This is the cell phonopy reduced the unit cell to and evaluated the
    dynamical matrix on -- not a standardisation applied beforehand. The
    reciprocal lattice carries the 2*pi factor, which is the convention
    :class:`~qvista.analyses.kpath.PrimitiveCell` documents.
    """

    import numpy as np

    lattice = np.asarray(phonon.primitive.cell, dtype=float)

    return PrimitiveCell(
        lattice=lattice,
        positions=np.asarray(
            phonon.primitive.scaled_positions, dtype=float
        ),
        types=[int(number) for number in phonon.primitive.numbers],
        reciprocal_lattice=2.0 * np.pi * np.linalg.inv(lattice).T,
    )


def _displacement_distance(phonon) -> float:
    """
    The finite-difference step this configuration used, in Angstrom.

    ``phonon.displacements`` is ``(n, 4)`` for the usual one-atom-at-a-
    time dataset, and the leading column is the atom *index* -- taking
    the maximum of the whole array would report atom 15 as a 15 Angstrom
    step. Only the three components after it are a displacement.
    """

    import numpy as np

    values = np.asarray(phonon.displacements, dtype=float)

    if values.size == 0:
        return 0.0

    if values.ndim == 2 and values.shape[1] == 4:
        values = values[:, 1:]

    return float(np.max(np.linalg.norm(values.reshape(-1, 3), axis=1)))


def _minimum(frequencies) -> float | None:
    """Smallest frequency across all band segments, or None."""

    if frequencies is None or len(frequencies) == 0:
        return None

    import numpy as np

    try:
        return float(
            min(np.asarray(segment).min() for segment in frequencies)
        )

    except (ValueError, TypeError):
        return None

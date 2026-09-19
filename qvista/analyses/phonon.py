"""
Phonon band structure and density of states.

One expensive pipeline -- displacements, forces, force constants -- feeds all
three Phonon Analysis menu items. The force constants are cached on the
project, so switching between band, DOS and band+DOS is presentation only and
never recomputes forces.

Forces come from a machine-learning force field -- whichever of the
supported potentials is installed, CHGNet by default -- and **not** from
DFT. See :mod:`qvista.services.forcefield`.

For the first-principles answer see :mod:`qvista.analyses.phonon_vasp`,
which drives the same phonopy pipeline with forces from VASP. The two
share :class:`PhononData` and :class:`PhononResult`, so the plotting, the
console output and the mode animation below are common to both; only
:attr:`PhononData.force_field` says which one ran, and it is stamped on
every figure.

Three corrections against the reference implementation
------------------------------------------------------
1. The reference read ``phonon._band_structure``, a private attribute, and
   never called ``run_band_structure()`` -- so it only worked by accident
   from leftover interpreter state. This calls the public
   ``run_band_structure()`` and reads ``get_band_structure_dict()``.

2. The reference built Phonopy from the raw input structure while taking the
   k-path from seekpath's standardised primitive cell. For a conventional
   cell those are different bases, which silently produces a wrong band
   structure. Here the Phonopy object is built from
   :attr:`~qvista.analyses.kpath.KPathResult.primitive` -- the same cell the
   path is expressed in.

3. The reference defaulted to a 3x3x3 supercell unconditionally. For a slab
   that replicates vacuum and treats it as periodic, and at P1 the
   displacement count explodes. Supercell choice is guarded and confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from qvista.analyses.base import Analysis, AnalysisError, AnalysisResult
from qvista.analyses.kpath import (
    KPathResult,
    PrimitiveCell,
    group_path_segments,
)
from qvista.models.project import Project
from qvista.services.structure_analyzer import DEFAULT_SYMPREC

#: Displacement amplitude in Angstrom, phonopy's usual finite-difference step.
DEFAULT_DISPLACEMENT = 0.01

#: Sampling mesh for the total DOS.
DEFAULT_MESH = (20, 20, 20)

#: q-points per band segment.
BAND_NPOINTS = 51

#: Supercell used for a well-behaved bulk cell.
BULK_SUPERCELL = (3, 3, 3)

#: Anything larger than this many atoms gets a smaller default supercell.
LARGE_CELL_ATOMS = 40

#: Above this many atoms per displaced supercell, the run is refused unless
#: the user overrides it. A 144-atom slab at 3x3x3 is 3,888 atoms.
SUPERCELL_ATOM_LIMIT = 2000

#: What the menu items ask for.
MODE_BAND = "band"
MODE_DOS = "dos"
MODE_BOTH = "both"

#: Band structure with eigenvectors, projected onto elements.
MODE_PROJECTED = "projected"

#: Free energy, entropy and heat capacity against temperature. Needs the
#: mesh the DOS runs on, so it implies a DOS calculation.
MODE_THERMAL = "thermal"


# ------------------------------------------------------------
# Supercell choice
# ------------------------------------------------------------


def suggest_supercell(
    natoms: int,
    vacuum_report=None,
) -> tuple[int, int, int]:
    """
    Propose a supercell that is not actively harmful.

    A vacuum direction is never replicated: repeating empty space makes the
    calculation larger and the physics meaningless. Large cells drop to
    2x2x2 so the displacement count stays tractable.
    """

    base = 2 if natoms > LARGE_CELL_ATOMS else BULK_SUPERCELL[0]

    supercell = [base, base, base]

    if vacuum_report is not None and vacuum_report.has_vacuum:

        for direction in vacuum_report.directions:
            supercell[direction.axis] = 1

    return tuple(supercell)


def supercell_atom_count(
    natoms: int,
    supercell: Sequence[int],
) -> int:
    """Atoms in one displaced supercell."""

    return int(natoms * supercell[0] * supercell[1] * supercell[2])


def build_phonopy(
    primitive: PrimitiveCell,
    supercell: Sequence[int],
    symprec: float = DEFAULT_SYMPREC,
    distance: float = DEFAULT_DISPLACEMENT,
):
    """
    Create a Phonopy object with displacements generated.

    Built from ``primitive`` -- seekpath's standardised cell, the one the
    k-path is expressed in -- so q-points and force constants share a basis.
    No forces are evaluated here, so this is cheap enough to call from a
    dialog.
    """

    import numpy as np

    from phonopy import Phonopy
    from pymatgen.io.phonopy import get_phonopy_structure

    unitcell = get_phonopy_structure(primitive.to_pymatgen())

    # primitive_matrix MUST be the identity here.
    #
    # phonopy 4 defaults it to "auto", which re-reduces the cell it is given
    # to phonopy's own primitive cell -- a different basis again. Band
    # structures are evaluated against phonon.primitive, so leaving this on
    # "auto" would reintroduce exactly the basis mismatch this module exists
    # to prevent: the q-points come from seekpath's primitive cell, and the
    # dynamical matrix would be built on phonopy's. The cell passed in is
    # already the standardised primitive cell, so it must be taken as-is.
    phonon = Phonopy(
        unitcell,
        supercell_matrix=np.diag([int(value) for value in supercell]),
        primitive_matrix=np.eye(3),
        symprec=symprec,
    )

    phonon.generate_displacements(distance=distance)

    return phonon


def count_displacements(
    primitive: PrimitiveCell,
    supercell: Sequence[int],
    symprec: float = DEFAULT_SYMPREC,
    distance: float = DEFAULT_DISPLACEMENT,
) -> tuple[int, int]:
    """
    Return phonopy's real ``(displacement count, atoms per supercell)``.

    Used to price a run before it starts. This is the actual number of force
    evaluations that will be performed, not an approximation from symmetry
    counting.
    """

    phonon = build_phonopy(primitive, supercell, symprec, distance)

    supercells = phonon.supercells_with_displacements

    if not supercells:
        return 0, 0

    return len(supercells), len(supercells[0])


# ------------------------------------------------------------
# Cached pipeline output
# ------------------------------------------------------------


@dataclass
class PhononData:
    """
    Force constants and everything needed to reuse them.

    Cached on the project so band, DOS and band+DOS share one calculation.
    """

    #: The phonopy object, carrying the force constants.
    phonon: Any

    supercell: tuple[int, int, int]
    symprec: float
    distance: float

    #: The cell the force constants and the k-path both live in.
    primitive: PrimitiveCell

    #: Path labels, grouped into connected stretches.
    stretches: list[list[str]] = field(default_factory=list)

    n_displacements: int = 0
    supercell_natoms: int = 0

    force_field: str = ""

    #: Identifies the structure these constants belong to.
    source: str = ""

    def matches(
        self,
        source: str,
        supercell: Sequence[int],
        symprec: float,
        distance: float,
    ) -> bool:
        """True when a cached result can be reused as-is."""

        return (
            self.source == source
            and tuple(self.supercell) == tuple(supercell)
            and abs(self.symprec - symprec) < 1e-12
            and abs(self.distance - distance) < 1e-12
        )


@dataclass
class PhononResult(AnalysisResult):
    """Band and/or DOS data, plus the cache entry that produced them."""

    phonon_data: PhononData | None = None

    #: ``get_band_structure_dict()`` output, when the band was requested.
    band: dict[str, Any] | None = None

    #: ``get_total_dos_dict()`` output, when the DOS was requested.
    dos: dict[str, Any] | None = None

    #: Free energy, entropy and heat capacity, when they were asked for.
    #: Integrated over the same mesh the DOS used, so the two always
    #: describe the same Brillouin-zone sampling.
    thermal: object | None = None

    mode: str = MODE_BOTH

    reused_cache: bool = False

    def as_storage(self) -> dict[str, Any]:
        """Package for :class:`~qvista.models.results.Results`."""

        return {
            "data": self.phonon_data,
            "band": self.band,
            "dos": self.dos,
            "thermal": self.thermal,
            "mode": self.mode,
            "result": self,
        }


# ------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------


def project_eigenvectors(
    eigenvectors,
    symbols: Sequence[str],
) -> dict[str, Any]:
    """
    Project band eigenvectors onto the elements present.

    phonopy returns mass-weighted eigenvectors as ``(nq, 3N, 3N)`` matrices
    per path segment, where column *j* is mode *j*. The squared modulus of
    the three components belonging to one atom is that atom's share of the
    mode, and the shares sum to one across the cell.

    Returns ``{element: array(nq, nbands)}`` with weights in ``[0, 1]``.
    """

    import numpy as np

    natoms = len(symbols)

    elements = list(dict.fromkeys(symbols))

    weights = {
        element: [] for element in elements
    }

    for segment in eigenvectors:

        vectors = np.asarray(segment)

        n_qpoints = vectors.shape[0]
        n_bands = vectors.shape[2]

        # |e|^2 reshaped so the atom index is separated from x, y, z.
        magnitude = np.abs(vectors) ** 2

        per_atom = magnitude.reshape(n_qpoints, natoms, 3, n_bands).sum(
            axis=2
        )

        for element in elements:

            mask = [
                index
                for index, symbol in enumerate(symbols)
                if symbol == element
            ]

            weights[element].append(per_atom[:, mask, :].sum(axis=1))

    return {
        element: np.concatenate(values, axis=0)
        if values
        else np.zeros((0, 0))
        for element, values in weights.items()
    }


def group_segments(
    segments: Sequence[tuple[str, str]],
) -> list[list[str]]:
    """
    Turn ``[(A,B),(B,C),(D,E)]`` into ``[[A,B,C],[D,E]]``.

    Each continuous stretch becomes one phonopy band path, so a break in
    the recommended path is preserved rather than drawn as a segment.
    Delegates to :func:`~qvista.analyses.kpath.group_path_segments` so the
    band structure and the Brillouin-zone plot split the path identically.
    """

    return group_path_segments(segments)


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------


class PhononAnalysis(Analysis):
    """
    Compute (or reuse) force constants, then band structure and/or DOS.

    ``mode`` selects what the caller wants plotted; the expensive part is
    identical for all three and is skipped entirely when ``cached`` already
    holds matching force constants.
    """

    name = "Phonon calculation"

    requires_periodic = True

    def __init__(
        self,
        project: Project,
        kpath: KPathResult,
        supercell: Sequence[int] = BULK_SUPERCELL,
        symprec: float = DEFAULT_SYMPREC,
        distance: float = DEFAULT_DISPLACEMENT,
        mesh: Sequence[int] = DEFAULT_MESH,
        mode: str = MODE_BOTH,
        cached: PhononData | None = None,
    ) -> None:

        super().__init__(project, symprec)

        self.kpath = kpath
        self.supercell = tuple(int(value) for value in supercell)
        self.distance = distance
        self.mesh = tuple(int(value) for value in mesh)
        self.mode = mode
        self.cached = cached

    # ------------------------------------------------------------

    @property
    def source_key(self) -> str:
        """Identifies the structure, for cache matching."""

        source = self.project.info.source_file

        return "" if source is None else str(source)

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    def execute(self) -> PhononResult:
        """Reuse or build force constants, then post-process."""

        reused = False

        if self.cached is not None and self.cached.matches(
            self.source_key,
            self.supercell,
            self.symprec,
            self.distance,
        ):
            self.report("Reusing cached force constants.")
            data = self.cached
            reused = True

        else:
            data = self._build_force_constants()

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # Post-processing
        # ----------------------------------------------------------

        band = None
        dos = None

        if self.mode in (MODE_BAND, MODE_BOTH, MODE_PROJECTED):
            band = self._run_band(
                data,
                with_eigenvectors=self.mode == MODE_PROJECTED,
            )

        self.raise_if_cancelled()

        if self.mode in (MODE_DOS, MODE_BOTH, MODE_THERMAL):
            dos = self._run_dos(data)

        self.raise_if_cancelled()

        thermal = None

        if self.mode == MODE_THERMAL:
            thermal = self._run_thermal(data)

        result = PhononResult(
            phonon_data=data,
            band=band,
            dos=dos,
            thermal=thermal,
            mode=self.mode,
            reused_cache=reused,
        )

        result.rows = self._rows(result)
        result.notes = self._notes(result)

        return result

    # ------------------------------------------------------------
    # Force constants
    # ------------------------------------------------------------

    def _build_force_constants(self) -> PhononData:
        """Displace the supercell, evaluate forces, build force constants."""

        import numpy as np

        from qvista.services import forcefield

        if self.kpath.primitive is None:
            raise AnalysisError(
                "The k-path did not carry a primitive cell, so the phonon "
                "calculation has no reference cell to build on."
            )

        # ----------------------------------------------------------
        # Build phonopy on seekpath's primitive cell -- the same basis the
        # k-path is expressed in.
        # ----------------------------------------------------------

        self.report("Building supercell...")

        try:
            phonon = build_phonopy(
                self.kpath.primitive,
                self.supercell,
                self.symprec,
                self.distance,
            )

            supercells = phonon.supercells_with_displacements

        except AnalysisError:
            raise

        except Exception as exc:
            raise AnalysisError(
                f"Could not set up the phonon calculation: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        total = len(supercells)

        if total == 0:
            raise AnalysisError(
                "phonopy generated no displacements for this structure."
            )

        natoms = len(supercells[0])

        # ----------------------------------------------------------
        # Forces
        # ----------------------------------------------------------

        calculator = self._get_calculator()

        self.report(
            f"Evaluating forces for {total} displaced supercells "
            f"of {natoms} atoms using {forcefield.describe()}."
        )

        forces = self._evaluate_forces(supercells, calculator)

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

        return PhononData(
            phonon=phonon,
            supercell=self.supercell,
            symprec=self.symprec,
            distance=self.distance,
            primitive=self.kpath.primitive,
            stretches=group_segments(self.kpath.segments),
            n_displacements=total,
            supercell_natoms=natoms,
            force_field=forcefield.describe(),
            source=self.source_key,
        )

    def _get_calculator(self):
        """
        Return the force calculator.

        Split out so tests can substitute a cheap analytic calculator and
        never touch the machine-learning model.
        """

        from qvista.services import forcefield

        try:
            return forcefield.get_calculator()

        except forcefield.ForceFieldUnavailable as exc:
            raise AnalysisError(str(exc)) from exc

    def _evaluate_forces(
        self,
        supercells,
        calculator,
    ) -> list:
        """
        Evaluate forces on every displaced supercell.

        Reports one line per structure so a long run is visibly making
        progress, and checks for cancellation between structures.
        """

        import numpy as np
        from ase import Atoms

        total = len(supercells)

        forces = []

        for index, cell in enumerate(supercells):

            self.raise_if_cancelled()

            atoms = Atoms(
                symbols=cell.symbols,
                scaled_positions=cell.scaled_positions,
                cell=cell.cell,
                pbc=True,
            )

            atoms.calc = calculator

            try:
                values = np.asarray(atoms.get_forces(), dtype=float)

            except Exception as exc:
                raise AnalysisError(
                    f"Force evaluation failed on displacement "
                    f"{index + 1} of {total}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            forces.append(values)

            percent = int(round(100.0 * (index + 1) / total))

            self.report(
                f"[{index + 1}/{total}] "
                f"Max|F| = {np.abs(values).max():.4f} eV/Å",
                min(percent, 100),
            )

        return forces

    # ------------------------------------------------------------
    # Band structure
    # ------------------------------------------------------------

    def _run_band(
        self,
        data: PhononData,
        with_eigenvectors: bool = False,
    ) -> dict[str, Any]:
        """
        Run the band structure along the k-path.

        Uses phonopy's public ``run_band_structure`` and public accessor --
        the reference read a private attribute that was never populated.
        """

        import numpy as np

        from phonopy.phonon.band_structure import (
            get_band_qpoints_and_path_connections,
        )

        self.report("Running band structure...")

        points = self.kpath.points

        stretches = data.stretches or group_segments(self.kpath.segments)

        if not stretches:
            raise AnalysisError(
                "The k-path has no segments, so there is nothing to plot."
            )

        try:
            band_paths = [
                [np.asarray(points[label], dtype=float) for label in stretch]
                for stretch in stretches
            ]

        except KeyError as exc:
            raise AnalysisError(
                f"The k-path refers to a point that this lattice does not "
                f"have: {exc}"
            ) from exc

        # phonopy uses this only to distribute q-points by segment length,
        # where any uniform scale cancels; pass the 2*pi-free reciprocal
        # lattice to stay in phonopy's own convention.
        reciprocal = (
            np.asarray(data.primitive.reciprocal_lattice, dtype=float)
            / (2.0 * np.pi)
        )

        qpoints, connections = get_band_qpoints_and_path_connections(
            band_paths,
            npoints=BAND_NPOINTS,
            rec_lattice=reciprocal,
        )

        labels = [label for stretch in stretches for label in stretch]

        try:
            data.phonon.run_band_structure(
                qpoints,
                path_connections=connections,
                labels=labels,
                with_eigenvectors=with_eigenvectors,
            )

            # The ``band_structure`` property, not get_band_structure_dict():
            # the dict accessor is deprecated in phonopy 4. Either way this
            # is the public result of the public run_band_structure() call,
            # never the private _band_structure the reference reached into.
            structure = data.phonon.band_structure

            band = {
                "qpoints": structure.qpoints,
                "distances": structure.distances,
                "frequencies": structure.frequencies,
            }

        except Exception as exc:
            raise AnalysisError(
                f"Band structure calculation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        band["labels"] = labels
        band["path_connections"] = connections
        band["stretches"] = stretches

        # ----------------------------------------------------------
        # Element projection
        # ----------------------------------------------------------

        if with_eigenvectors:

            self.report("Projecting eigenvectors onto elements...")

            symbols = [
                str(symbol) for symbol in data.phonon.primitive.symbols
            ]

            band["symbols"] = symbols

            band["projections"] = project_eigenvectors(
                structure.eigenvectors,
                symbols,
            )

        return band

    # ------------------------------------------------------------
    # Density of states
    # ------------------------------------------------------------

    def _run_thermal(self, data: PhononData):
        """
        Harmonic free energy, entropy and heat capacity.

        Off the mesh :meth:`_run_dos` has just sampled, so the two always
        describe the same Brillouin-zone sampling. Shared with the VASP
        route rather than reimplemented: once there are force constants
        the physics is identical and only their provenance differs.

        A failure here costs the thermal panel and nothing else, so it is
        reported and swallowed -- the force constants took minutes and
        are still good for a band and a DOS.
        """

        from qvista.services import phonon_dos

        self.report("Integrating the thermal properties...")

        try:
            return phonon_dos.run_thermal(data.phonon)

        except Exception as exc:  # noqa: BLE001 - the rest still stands
            self.report(f"Thermal properties failed: {exc}")
            return None

    def _run_dos(self, data: PhononData) -> dict[str, Any]:
        """Sample the Brillouin zone and accumulate the total DOS."""

        self.report(f"Running {self.mesh[0]}x{self.mesh[1]}x{self.mesh[2]} mesh...")

        try:
            # Gamma-centred, not phonopy's default half-shifted grid: an
            # even shifted mesh is not preserved by the point group, so
            # phonopy drops the symmetry reduction and samples the whole
            # zone -- slower, and it misses Gamma itself, where the
            # acoustic branches are.
            data.phonon.run_mesh(
                list(self.mesh),
                is_mesh_symmetry=True,
                is_gamma_center=True,
            )

            data.phonon.run_total_dos()

            # The ``total_dos`` property; get_total_dos_dict() is deprecated
            # in phonopy 4. Keys are kept so the plotting code is unchanged.
            total = data.phonon.total_dos

            dos = {
                "frequency_points": total.frequency_points,
                "total_dos": total.dos,
            }

        except Exception as exc:
            raise AnalysisError(
                f"Density of states calculation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        dos["mesh"] = self.mesh

        return dos

    # ------------------------------------------------------------
    # Console text
    # ------------------------------------------------------------

    def _rows(self, result: PhononResult) -> list[tuple[str, str]]:
        """Summary rows for the console."""

        data = result.phonon_data

        if data is None:
            return []

        nx, ny, nz = data.supercell

        rows = [
            ("Force field", data.force_field),
            ("Supercell", f"{nx} x {ny} x {nz}"),
            ("Atoms per supercell", str(data.supercell_natoms)),
            ("Displacements", str(data.n_displacements)),
            ("Displacement step", f"{data.distance} Å"),
            ("Primitive atoms", str(data.primitive.natoms)),
            ("Symmetry tolerance", f"{data.symprec} Å"),
            (
                "Force constants",
                "reused from cache" if result.reused_cache else "computed",
            ),
        ]

        if result.dos is not None:
            mesh = result.dos.get("mesh", self.mesh)
            rows.append(
                ("DOS mesh", f"{mesh[0]} x {mesh[1]} x {mesh[2]}")
            )

        return rows

    def _notes(self, result: PhononResult) -> list[str]:
        """Warnings and provenance."""

        notes = [
            "Forces were evaluated with a machine-learning force field, "
            "not DFT. Treat these frequencies as a fast screening result "
            "rather than a converged first-principles calculation."
        ]

        if result.band is not None:

            frequencies = result.band.get("frequencies")

            minimum = _minimum_frequency(frequencies)

            if minimum is not None and minimum < -0.1:
                notes.append(
                    f"Imaginary modes present: the lowest frequency is "
                    f"{minimum:.3f} THz. The structure is dynamically "
                    f"unstable at this level of theory, or the supercell "
                    f"is too small to capture the relevant force constants."
                )

        return notes


def _minimum_frequency(frequencies) -> float | None:
    """Smallest frequency across all band segments, or None."""

    if not frequencies:
        return None

    import numpy as np

    try:
        return float(min(np.asarray(seg).min() for seg in frequencies))

    except Exception:
        return None

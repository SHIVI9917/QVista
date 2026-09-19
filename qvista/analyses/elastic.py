"""
Elastic tensor and mechanical properties.

Relaxes the cell, refines its symmetry, applies a set of small strains, and
fits the 6x6 elastic tensor from the resulting stresses. Voigt-Reuss-Hill
averages and the usual derived moduli follow from that tensor.

Forces and stresses come from MatterSim, a machine-learning force field,
**not** from DFT. See :mod:`qvista.services.forcefield`. Moduli from a
surrogate model should be read as a screening result.

Why symmetry refinement is a correctness requirement
----------------------------------------------------
An elastic tensor is only meaningful in a standard crystal orientation --
C44, C55 and C66 are defined relative to the crystal axes. Refinement puts
the cell in that orientation. On a P1 input refinement is a no-op: the
structure passes through in whatever arbitrary orientation the file happened
to have, and the shear constants stop meaning what their names say while
still looking like plausible numbers. That case is detected and reported
rather than quietly returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from qvista.analyses.base import Analysis, AnalysisError, AnalysisResult
from qvista.models.project import Project
from qvista.services.structure_analyzer import DEFAULT_SYMPREC

#: Strain modes, in the order they are applied.
STRAIN_MODES = ("xx", "yy", "zz", "yz", "xz", "xy")

#: Applied amplitudes. Small and symmetric about zero, so the linear fit
#: has no constant-offset bias.
STRAIN_AMPLITUDES = (-0.01, -0.005, 0.005, 0.01)

#: Voigt ordering: 0..5 <-> xx, yy, zz, yz, xz, xy.
VOIGT_INDEX = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "xz": 4, "xy": 5}

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

#: Force convergence for the relaxations, eV/Angstrom.
FMAX = 0.01

#: Cap on optimiser steps for a single strained structure.
MAX_STEPS = 200


# ------------------------------------------------------------
# Strain geometry
# ------------------------------------------------------------


def deformation_gradient(mode: str, eps: float):
    """
    Build the deformation gradient for one strain mode.

    Normal modes add ``eps`` to a diagonal entry. Shear modes add ``eps/2``
    to *both* symmetric off-diagonal entries, which makes ``eps`` the
    engineering shear strain -- the quantity Voigt component 4, 5 or 6 is
    defined against, so ``C44 = d(sigma_4)/d(eps)`` comes out directly.
    """

    import numpy as np

    if mode not in VOIGT_INDEX:
        raise AnalysisError(f"Unknown strain mode: {mode}")

    gradient = np.eye(3)

    first = AXIS_INDEX[mode[0]]
    second = AXIS_INDEX[mode[1]]

    if first == second:
        gradient[first, first] += eps

    else:
        gradient[first, second] += eps / 2.0
        gradient[second, first] += eps / 2.0

    return gradient


def strain_lattice(matrix, gradient):
    """
    Apply a Cartesian deformation gradient to a lattice.

    Lattice matrices here store one lattice vector **per row**, as ASE and
    pymatgen both do. Deforming Cartesian space by ``F`` sends each vector
    ``a`` to ``F @ a``, which for row storage is ``M @ F.T``.

    Writing ``F @ M`` instead -- the natural-looking form, and what the
    reference used -- contracts ``F`` against the row index, mixing the
    three lattice vectors into each other rather than transforming each
    one. For a diagonal cell the two agree, which is why it survives on
    cubic and orthorhombic inputs; for anything non-diagonal (an FCC
    primitive cell, hexagonal, monoclinic, triclinic) it applies a
    different strain than intended and the shear constants come out wrong.
    """

    import numpy as np

    return (
        np.asarray(matrix, dtype=float)
        @ np.asarray(gradient, dtype=float).T
    )


# ------------------------------------------------------------
# Tensor fitting
# ------------------------------------------------------------


def fit_elastic_tensor(measurements: dict[str, Sequence[tuple]]):
    """
    Fit the 6x6 elastic tensor from stress-versus-strain measurements.

    ``measurements`` maps a strain mode to ``(strain, stress_voigt)`` pairs,
    with stress already referenced against the unstrained cell. Each stress
    component is fitted linearly against the strain, and the slope becomes
    one entry of the tensor's column for that mode.

    The result is symmetrised: ``C`` is symmetric for any real material, so
    averaging ``C`` with its transpose cancels part of the fitting noise.
    """

    import numpy as np

    tensor = np.zeros((6, 6))

    for mode, rows in measurements.items():

        if mode not in VOIGT_INDEX:
            raise AnalysisError(f"Unknown strain mode: {mode}")

        if len(rows) < 2:
            raise AnalysisError(
                f"Mode {mode} has fewer than two strain points, so no "
                f"slope can be fitted."
            )

        column = VOIGT_INDEX[mode]

        strains = np.asarray([row[0] for row in rows], dtype=float)
        stresses = np.asarray([row[1] for row in rows], dtype=float)

        for component in range(6):

            slope = np.polyfit(strains, stresses[:, component], 1)[0]

            tensor[component, column] = slope

    return 0.5 * (tensor + tensor.T)


# ------------------------------------------------------------
# Derived properties
# ------------------------------------------------------------


@dataclass
class ElasticModuli:
    """Voigt-Reuss-Hill averages and the moduli derived from them."""

    bulk_voigt: float
    bulk_reuss: float
    bulk_hill: float

    shear_voigt: float
    shear_reuss: float
    shear_hill: float

    youngs: float
    poisson: float
    pugh: float
    anisotropy: float

    def as_rows(self) -> list[tuple[str, str]]:
        """Console rows, with units."""

        return [
            ("Bulk K (Voigt)", f"{self.bulk_voigt:.2f} GPa"),
            ("Bulk K (Reuss)", f"{self.bulk_reuss:.2f} GPa"),
            ("Bulk K (Hill)", f"{self.bulk_hill:.2f} GPa"),
            ("Shear G (Voigt)", f"{self.shear_voigt:.2f} GPa"),
            ("Shear G (Reuss)", f"{self.shear_reuss:.2f} GPa"),
            ("Shear G (Hill)", f"{self.shear_hill:.2f} GPa"),
            ("Young E", f"{self.youngs:.2f} GPa"),
            ("Poisson ratio", f"{self.poisson:.4f}"),
            ("Pugh ratio K/G", f"{self.pugh:.4f}"),
            ("Anisotropy A_U", f"{self.anisotropy:.4f}"),
        ]

    def as_table(self) -> list[tuple[str, str, str, str, str]]:
        """
        Rows as ``(section, property, symbol, value, unit)``.

        The panel wants the symbol and the unit in columns of their own
        rather than run into the name and the number, so that a column
        of moduli lines up and can be read down.
        """

        return [
            ("Bulk modulus", "Voigt bound", "K_V",
             f"{self.bulk_voigt:.2f}", "GPa"),
            ("Bulk modulus", "Reuss bound", "K_R",
             f"{self.bulk_reuss:.2f}", "GPa"),
            ("Bulk modulus", "Hill average", "K_VRH",
             f"{self.bulk_hill:.2f}", "GPa"),
            ("Shear modulus", "Voigt bound", "G_V",
             f"{self.shear_voigt:.2f}", "GPa"),
            ("Shear modulus", "Reuss bound", "G_R",
             f"{self.shear_reuss:.2f}", "GPa"),
            ("Shear modulus", "Hill average", "G_VRH",
             f"{self.shear_hill:.2f}", "GPa"),
            ("Derived", "Young's modulus", "E",
             f"{self.youngs:.2f}", "GPa"),
            ("Derived", "Poisson ratio", "v",
             f"{self.poisson:.4f}", ""),
            ("Derived", "Pugh ratio", "K/G",
             f"{self.pugh:.4f}", ""),
            ("Derived", "Universal anisotropy", "A_U",
             f"{self.anisotropy:.4f}", ""),
        ]


def compliance_tensor(tensor):
    """
    The compliance tensor S, the inverse of the stiffness tensor C.

    Reported alongside C because the two answer different questions:
    C says what stress a given strain costs, S says what strain a given
    stress produces, and it is S whose diagonal gives the directional
    Young's moduli directly as ``1 / S_ii``.  In 1/GPa.
    """

    import numpy as np

    stiffness = np.asarray(tensor, dtype=float)

    try:
        return np.linalg.inv(stiffness)

    except np.linalg.LinAlgError as exc:
        raise AnalysisError(
            "The elastic tensor is singular and cannot be inverted, so no "
            "moduli can be derived. This normally means the relaxations "
            "did not converge."
        ) from exc


def derive_moduli(tensor) -> ElasticModuli:
    """
    Compute Voigt, Reuss and Hill averages and the derived moduli.

    ``tensor`` is the 6x6 elastic tensor in GPa.
    """

    import numpy as np

    stiffness = np.asarray(tensor, dtype=float)

    compliance = compliance_tensor(stiffness)

    # ----------------------------------------------------------
    # Voigt (uniform strain) averages
    # ----------------------------------------------------------

    c = stiffness

    bulk_voigt = (
        (c[0, 0] + c[1, 1] + c[2, 2])
        + 2.0 * (c[0, 1] + c[0, 2] + c[1, 2])
    ) / 9.0

    shear_voigt = (
        (c[0, 0] + c[1, 1] + c[2, 2])
        - (c[0, 1] + c[0, 2] + c[1, 2])
        + 3.0 * (c[3, 3] + c[4, 4] + c[5, 5])
    ) / 15.0

    # ----------------------------------------------------------
    # Reuss (uniform stress) averages
    # ----------------------------------------------------------

    s = compliance

    bulk_reuss_denominator = (
        (s[0, 0] + s[1, 1] + s[2, 2])
        + 2.0 * (s[0, 1] + s[0, 2] + s[1, 2])
    )

    shear_reuss_denominator = (
        4.0 * (s[0, 0] + s[1, 1] + s[2, 2])
        - 4.0 * (s[0, 1] + s[0, 2] + s[1, 2])
        + 3.0 * (s[3, 3] + s[4, 4] + s[5, 5])
    )

    if abs(bulk_reuss_denominator) < 1e-12:
        raise AnalysisError(
            "The Reuss bulk modulus is undefined for this tensor."
        )

    if abs(shear_reuss_denominator) < 1e-12:
        raise AnalysisError(
            "The Reuss shear modulus is undefined for this tensor."
        )

    bulk_reuss = 1.0 / bulk_reuss_denominator
    shear_reuss = 15.0 / shear_reuss_denominator

    # ----------------------------------------------------------
    # Hill averages and derived quantities
    # ----------------------------------------------------------

    bulk_hill = 0.5 * (bulk_voigt + bulk_reuss)
    shear_hill = 0.5 * (shear_voigt + shear_reuss)

    combined = 3.0 * bulk_hill + shear_hill

    if abs(combined) < 1e-12:
        raise AnalysisError(
            "Young's modulus is undefined for this tensor."
        )

    youngs = 9.0 * bulk_hill * shear_hill / combined
    poisson = (3.0 * bulk_hill - 2.0 * shear_hill) / (2.0 * combined)

    pugh = (
        float("nan")
        if abs(shear_hill) < 1e-12
        else bulk_hill / shear_hill
    )

    if abs(shear_reuss) < 1e-12 or abs(bulk_reuss) < 1e-12:
        anisotropy = float("nan")
    else:
        anisotropy = (
            5.0 * (shear_voigt / shear_reuss)
            + (bulk_voigt / bulk_reuss)
            - 6.0
        )

    return ElasticModuli(
        bulk_voigt=bulk_voigt,
        bulk_reuss=bulk_reuss,
        bulk_hill=bulk_hill,
        shear_voigt=shear_voigt,
        shear_reuss=shear_reuss,
        shear_hill=shear_hill,
        youngs=youngs,
        poisson=poisson,
        pugh=pugh,
        anisotropy=anisotropy,
    )


# ------------------------------------------------------------
# Mechanical stability
# ------------------------------------------------------------


@dataclass
class StabilityReport:
    """Whether the fitted tensor describes a mechanically stable crystal."""

    stable: bool
    eigenvalues: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def as_rows(self) -> list[tuple[str, str]]:
        """Console rows."""

        smallest = min(self.eigenvalues) if self.eigenvalues else float("nan")

        return [
            (
                "Born stability",
                "satisfied" if self.stable else "VIOLATED",
            ),
            ("Smallest eigenvalue", f"{smallest:.3f} GPa"),
        ]

    def as_table(self) -> list[tuple[str, str, str, str, str]]:
        """Rows as ``(section, property, symbol, value, unit)``."""

        smallest = min(self.eigenvalues) if self.eigenvalues else float("nan")

        return [
            (
                "Stability",
                "Born criteria",
                "",
                "satisfied" if self.stable else "VIOLATED",
                "",
            ),
            (
                "Stability",
                "Smallest eigenvalue of C",
                "min eig",
                f"{smallest:.3f}",
                "GPa",
            ),
        ]


def born_stability(tensor) -> StabilityReport:
    """
    Check the Born mechanical-stability criteria.

    In its general form, valid for every crystal class, the condition is
    that the elastic tensor be positive definite: all six eigenvalues
    positive. A violation almost always means an unconverged relaxation
    rather than a genuinely unstable material.
    """

    import numpy as np

    stiffness = np.asarray(tensor, dtype=float)

    symmetric = 0.5 * (stiffness + stiffness.T)

    eigenvalues = sorted(
        float(value) for value in np.linalg.eigvalsh(symmetric)
    )

    failures: list[str] = []

    if eigenvalues[0] <= 0.0:
        failures.append(
            f"The elastic tensor is not positive definite: its smallest "
            f"eigenvalue is {eigenvalues[0]:.3f} GPa. The Born criteria "
            f"are violated, which usually indicates that the cell or the "
            f"strained structures did not fully relax."
        )

    diagonal = np.diag(symmetric)

    if any(value <= 0.0 for value in diagonal):
        failures.append(
            "Some diagonal elastic constants are non-positive, which is "
            "unphysical for a stable crystal."
        )

    return StabilityReport(
        stable=not failures,
        eigenvalues=eigenvalues,
        failures=failures,
    )


# ------------------------------------------------------------
# Result
# ------------------------------------------------------------


@dataclass
class ElasticResult(AnalysisResult):
    """The fitted tensor, the derived moduli and the provenance."""

    tensor: Any = None

    moduli: ElasticModuli | None = None
    stability: StabilityReport | None = None

    spacegroup_before: str = ""
    spacegroup_number_before: int = 0

    spacegroup_after: str = ""
    spacegroup_number_after: int = 0

    #: True when refinement left the structure in P1, so the tensor
    #: orientation is arbitrary.
    orientation_is_arbitrary: bool = False

    energy: float = 0.0
    symprec: float = DEFAULT_SYMPREC
    force_field: str = ""

    def as_storage(self) -> dict[str, Any]:
        """Package for :class:`~qvista.models.results.Results`."""

        return {
            "tensor": self.tensor,
            "moduli": self.moduli,
            "stability": self.stability,
            "result": self,
        }

    def to_csv(self) -> str:
        """
        Render the tensor and moduli as CSV.

        Written only when the user asks for it, never automatically into the
        working directory.
        """

        lines = ["# QVista elastic tensor"]

        lines.append(f"# force field,{self.force_field}")
        lines.append(f"# space group (refined),{self.spacegroup_after}")
        lines.append(f"# symprec,{self.symprec}")

        if self.orientation_is_arbitrary:
            lines.append(
                "# WARNING,refined to P1 - tensor orientation is arbitrary"
            )

        lines.append("")
        lines.append("# elastic tensor C (GPa), Voigt order xx yy zz yz xz xy")

        if self.tensor is not None:
            for row in self.tensor:
                lines.append(
                    ",".join(f"{float(value):.6f}" for value in row)
                )

        lines.append("")
        lines.append("# derived property,value")

        if self.moduli is not None:
            for label, value in self.moduli.as_rows():
                lines.append(f"{label},{value}")

        if self.stability is not None:
            for label, value in self.stability.as_rows():
                lines.append(f"{label},{value}")

        return "\n".join(lines) + "\n"


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------


class ElasticAnalysis(Analysis):
    """
    Relax, strain, and fit the elastic tensor.

    Runs 1 + 1 + 24 optimisations, so it is long: the cell relaxation, the
    reference structure, and one ions-only relaxation per strained cell.
    """

    name = "Mechanical Properties"

    requires_periodic = True

    def __init__(
        self,
        project: Project,
        symprec: float = DEFAULT_SYMPREC,
        amplitudes: Sequence[float] = STRAIN_AMPLITUDES,
        fmax: float = FMAX,
        max_steps: int = MAX_STEPS,
    ) -> None:

        super().__init__(project, symprec)

        self.amplitudes = tuple(amplitudes)
        self.fmax = fmax
        self.max_steps = max_steps

    # ------------------------------------------------------------

    def _get_calculator(self):
        """
        Return the force calculator.

        Isolated so tests can substitute a cheap analytic potential.
        """

        from qvista.services import forcefield

        try:
            return forcefield.get_calculator()

        except forcefield.ForceFieldUnavailable as exc:
            raise AnalysisError(str(exc)) from exc

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    def execute(self) -> ElasticResult:
        """Relax, refine, strain, fit."""

        import numpy as np

        from qvista.services import forcefield

        calculator = self._get_calculator()

        structure = self.structure()

        # ----------------------------------------------------------
        # 1. Relax cell and ions
        # ----------------------------------------------------------

        self.report(
            f"Relaxing cell and ions with {forcefield.describe()}..."
        )

        relaxed, energy, stress = self._relax_cell(structure, calculator)

        self.report(
            f"Relaxed energy {energy:.6f} eV, "
            f"max |stress| {np.abs(stress).max():.4f} GPa"
        )

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # 2. Symmetry refinement
        # ----------------------------------------------------------

        before, after, refined = self._refine(relaxed)

        self.report(
            f"Symmetry before refinement: {before[0]} (#{before[1]}); "
            f"after: {after[0]} (#{after[1]})"
        )

        # ----------------------------------------------------------
        # 3. Reference stress on the unstrained refined cell
        # ----------------------------------------------------------

        self.report("Computing reference stress...")

        reference = self._relax_ions(refined, calculator)

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # 4. Strained structures
        # ----------------------------------------------------------

        measurements = self._run_strains(refined, calculator, reference)

        # ----------------------------------------------------------
        # 5-6. Fit and derive
        # ----------------------------------------------------------

        self.report("Fitting the elastic tensor...")

        tensor = fit_elastic_tensor(measurements)

        moduli = derive_moduli(tensor)
        stability = born_stability(tensor)

        result = ElasticResult(
            tensor=tensor,
            moduli=moduli,
            stability=stability,
            spacegroup_before=before[0],
            spacegroup_number_before=before[1],
            spacegroup_after=after[0],
            spacegroup_number_after=after[1],
            orientation_is_arbitrary=after[1] == 1,
            energy=energy,
            symprec=self.symprec,
            force_field=forcefield.describe(),
        )

        result.rows = self._rows(result)
        result.notes = self._notes(result)

        return result

    # ------------------------------------------------------------
    # Relaxation
    # ------------------------------------------------------------

    def _to_atoms(self, structure, calculator):
        """Build an ASE atoms object with the calculator attached."""

        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(structure)

        atoms.calc = calculator

        return atoms

    def _relax_cell(self, structure, calculator):
        """Relax cell and ions; return the structure, energy and stress."""

        from ase.filters import FrechetCellFilter
        from ase.optimize import BFGS
        from ase.units import GPa
        from pymatgen.io.ase import AseAtomsAdaptor

        import numpy as np

        atoms = self._to_atoms(structure, calculator)

        try:
            # logfile=None: the reference scattered optimiser logs and
            # POSCARs through the working directory; nothing is written here.
            optimiser = BFGS(FrechetCellFilter(atoms), logfile=None)

            optimiser.run(fmax=self.fmax, steps=self.max_steps)

            energy = float(atoms.get_potential_energy())

            stress = np.asarray(atoms.get_stress(voigt=True)) / GPa

        except Exception as exc:
            raise AnalysisError(
                f"Cell relaxation failed: {type(exc).__name__}: {exc}"
            ) from exc

        return AseAtomsAdaptor.get_structure(atoms), energy, stress

    def _relax_ions(self, structure, calculator):
        """Relax ions at fixed cell; return the stress in GPa."""

        from ase.optimize import BFGS
        from ase.units import GPa

        import numpy as np

        atoms = self._to_atoms(structure, calculator)

        try:
            optimiser = BFGS(atoms, logfile=None)

            optimiser.run(fmax=self.fmax, steps=self.max_steps)

            return np.asarray(atoms.get_stress(voigt=True)) / GPa

        except Exception as exc:
            raise AnalysisError(
                f"Ionic relaxation failed: {type(exc).__name__}: {exc}"
            ) from exc

    # ------------------------------------------------------------
    # Symmetry
    # ------------------------------------------------------------

    def _refine(self, structure):
        """Refine the relaxed cell and report symmetry before and after."""

        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=self.symprec)

            before = (
                analyzer.get_space_group_symbol(),
                int(analyzer.get_space_group_number()),
            )

            refined = analyzer.get_refined_structure()

            refined_analyzer = SpacegroupAnalyzer(
                refined,
                symprec=self.symprec,
            )

            after = (
                refined_analyzer.get_space_group_symbol(),
                int(refined_analyzer.get_space_group_number()),
            )

        except Exception as exc:
            raise AnalysisError(
                f"Symmetry refinement failed at symprec={self.symprec}: "
                f"{exc}"
            ) from exc

        return before, after, refined

    # ------------------------------------------------------------
    # Strains
    # ------------------------------------------------------------

    def _run_strains(self, refined, calculator, reference):
        """Apply every strain and collect the referenced stresses."""

        from pymatgen.core import Structure

        measurements: dict[str, list[tuple[float, Any]]] = {}

        total = len(STRAIN_MODES) * len(self.amplitudes)

        index = 0

        for mode in STRAIN_MODES:

            rows: list[tuple[float, Any]] = []

            for amplitude in self.amplitudes:

                self.raise_if_cancelled()

                gradient = deformation_gradient(mode, amplitude)

                strained = Structure(
                    strain_lattice(refined.lattice.matrix, gradient),
                    refined.species,
                    refined.frac_coords,
                    coords_are_cartesian=False,
                )

                stress = self._relax_ions(strained, calculator) - reference

                rows.append((amplitude, stress))

                index += 1

                percent = int(round(100.0 * index / total))

                self.report(
                    f"[{index}/{total}] {mode} "
                    f"eps={amplitude:+.4f}  "
                    f"max|sigma| = {max(abs(value) for value in stress):.4f} GPa",
                    min(percent, 100),
                )

            measurements[mode] = rows

        return measurements

    # ------------------------------------------------------------
    # Console text
    # ------------------------------------------------------------

    def _rows(self, result: ElasticResult) -> list[tuple[str, str]]:
        """Summary rows."""

        rows = [
            ("Force field", result.force_field),
            ("Relaxed energy", f"{result.energy:.6f} eV"),
            ("Symmetry tolerance", f"{result.symprec} Å"),
            (
                "Space group (input)",
                f"{result.spacegroup_before} "
                f"(#{result.spacegroup_number_before})",
            ),
            (
                "Space group (refined)",
                f"{result.spacegroup_after} "
                f"(#{result.spacegroup_number_after})",
            ),
        ]

        if result.moduli is not None:
            rows.extend(result.moduli.as_rows())

        if result.stability is not None:
            rows.extend(result.stability.as_rows())

        return rows

    def _notes(self, result: ElasticResult) -> list[str]:
        """The tensor itself, plus warnings."""

        notes: list[str] = []

        # ----------------------------------------------------------
        # The tensor
        # ----------------------------------------------------------

        if result.tensor is not None:

            lines = ["Elastic tensor C (GPa), Voigt order xx yy zz yz xz xy"]

            for row in result.tensor:
                lines.append(
                    "  "
                    + " ".join(f"{float(value):9.2f}" for value in row)
                )

            notes.append("\n".join(lines))

        # ----------------------------------------------------------
        # Orientation warning -- the guard the reference lacks
        # ----------------------------------------------------------

        if result.orientation_is_arbitrary:
            notes.append(
                "WARNING: symmetry refinement returned P1, so it did not "
                "reorient the cell. The crystal axes are whatever the input "
                "file happened to use, which means C44, C55 and C66 are not "
                "the shear constants they are labelled as and the derived "
                "moduli should not be trusted. Retry with a looser symmetry "
                "tolerance, or supply a structure in a standard setting."
            )

        # ----------------------------------------------------------
        # Stability
        # ----------------------------------------------------------

        if result.stability is not None:
            for failure in result.stability.failures:
                notes.append(f"WARNING: {failure}")

        notes.append(
            "Stresses were evaluated with a machine-learning force field, "
            "not DFT. Treat these moduli as a screening result rather than "
            "a converged first-principles calculation."
        )

        return notes

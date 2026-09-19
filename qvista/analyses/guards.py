"""
Precondition checks shared by every analysis workflow.

These run before a long calculation starts, so the user finds out that a
structure is unsuitable in a moment rather than an hour. Each helper either
raises :class:`~qvista.analyses.base.AnalysisError` with a user-readable
message, or returns a report object that renders as console rows.

pymatgen is imported inside the functions -- it is heavy and the application
should not pay for it at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qvista.analyses.base import AnalysisError
from qvista.models.project import Project
from qvista.services.structure_analyzer import DEFAULT_SYMPREC

#: A lattice direction is "mostly vacuum" when the largest gap between
#: atoms along it spans at least this fraction of the cell...
VACUUM_FRACTION_THRESHOLD = 0.30

#: ...and is at least this many Angstrom thick. Both conditions must hold,
#: so ordinary small cells are not mistaken for slabs.
VACUUM_MIN_THICKNESS = 5.0

AXIS_LABELS = ("a", "b", "c")

#: Very rough cost of one single-point force-field evaluation, per atom.
#: Only used to give the user an order of magnitude before a long run.
SECONDS_PER_ATOM = 0.02

#: Runs longer than this get an explicit caution in the estimate.
LONG_RUN_SECONDS = 1800.0


# ------------------------------------------------------------
# Periodicity
# ------------------------------------------------------------


def require_periodic(project: Project) -> None:
    """
    Refuse structures that have no lattice.

    k-paths, phonons, supercells and elastic tensors are all undefined for a
    molecule, so a plain XYZ cannot be used for any of them.
    """

    info = project.structure_info

    if info is None:
        raise AnalysisError(
            project.structure_error
            or "No structure is loaded. Open a structure file first."
        )

    if not info.periodic:
        raise AnalysisError(
            f"{info.formula} was read from a plain XYZ file, which carries "
            "no lattice. Periodic properties cannot be computed from it.\n\n"
            "Supply a POSCAR, CONTCAR, CIF, or an extended XYZ with a "
            "Lattice=\"...\" entry in its comment line."
        )


# ------------------------------------------------------------
# Symmetry
# ------------------------------------------------------------


@dataclass
class SymmetryPreview:
    """
    What symmetry analysis found, before committing to a long run.

    Comparing ``n_input`` with ``n_primitive`` shows whether the calculation
    can be made cheaper by reducing the cell first.
    """

    spacegroup_symbol: str
    spacegroup_number: int
    crystal_system: str

    n_input: int
    n_primitive: int
    n_conventional: int

    symprec: float = DEFAULT_SYMPREC

    warnings: list[str] = field(default_factory=list)

    @property
    def is_p1(self) -> bool:
        """True when no symmetry beyond the identity was detected."""

        return self.spacegroup_number == 1

    def as_rows(self) -> list[tuple[str, str]]:
        """Return the preview as console ``(label, value)`` pairs."""

        return [
            ("Space group", self.spacegroup_symbol),
            ("Space group number", str(self.spacegroup_number)),
            ("Crystal system", self.crystal_system),
            ("Symmetry tolerance", f"{self.symprec} Å"),
            ("Atoms (input cell)", str(self.n_input)),
            ("Atoms (primitive)", str(self.n_primitive)),
            ("Atoms (conventional)", str(self.n_conventional)),
        ]


def symmetry_preview(
    structure,
    symprec: float = DEFAULT_SYMPREC,
) -> SymmetryPreview:
    """
    Report the symmetry of ``structure`` and the size of its reduced cells.

    Shown to the user before a long calculation so they can see what was
    detected -- and, if it is not what they expected, adjust ``symprec``
    rather than waiting for a meaningless result.
    """

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)

        preview = SymmetryPreview(
            spacegroup_symbol=analyzer.get_space_group_symbol(),
            spacegroup_number=analyzer.get_space_group_number(),
            crystal_system=analyzer.get_crystal_system().capitalize(),
            n_input=len(structure),
            n_primitive=len(analyzer.get_primitive_standard_structure()),
            n_conventional=len(
                analyzer.get_conventional_standard_structure()
            ),
            symprec=symprec,
        )

    except Exception as exc:
        raise AnalysisError(
            f"Symmetry analysis failed at symprec={symprec}: {exc}\n\n"
            "A relaxed structure may need a looser tolerance."
        ) from exc

    # ----------------------------------------------------------
    # Warnings
    # ----------------------------------------------------------

    if preview.is_p1:
        preview.warnings.append(
            "No symmetry detected (P1, #1). Nothing can be reduced, so "
            "every atom must be displaced independently and the cost is "
            "the full unreduced calculation. If this structure should be "
            "symmetric, try a looser symmetry tolerance."
        )

    elif preview.n_primitive == preview.n_input:
        preview.warnings.append(
            "The input cell is already primitive, so it cannot be reduced "
            "further."
        )

    return preview


# ------------------------------------------------------------
# Vacuum / slab detection
# ------------------------------------------------------------


@dataclass
class VacuumDirection:
    """A single lattice direction found to be mostly vacuum."""

    axis: int
    label: str
    gap_fraction: float
    thickness: float


@dataclass
class VacuumReport:
    """Which lattice directions, if any, are mostly empty space."""

    directions: list[VacuumDirection] = field(default_factory=list)

    @property
    def has_vacuum(self) -> bool:
        """True when the input looks like a slab or a surface."""

        return bool(self.directions)

    def as_rows(self) -> list[tuple[str, str]]:
        """Return the report as console ``(label, value)`` pairs."""

        if not self.has_vacuum:
            return [("Vacuum", "none detected (bulk structure)")]

        return [
            (
                f"Vacuum along {d.label}",
                f"{d.thickness:.1f} Å "
                f"({d.gap_fraction * 100:.0f}% of the cell)",
            )
            for d in self.directions
        ]

    @property
    def warnings(self) -> list[str]:
        """User-facing cautions implied by this report."""

        if not self.has_vacuum:
            return []

        labels = ", ".join(d.label for d in self.directions)

        return [
            f"This looks like a slab or surface: lattice direction "
            f"{labels} is mostly vacuum. A default supercell would "
            f"replicate that empty space, which is both physically "
            f"meaningless and far more expensive than it looks. Set the "
            f"supercell to 1 along {labels}."
        ]


def detect_vacuum(
    structure,
    fraction_threshold: float = VACUUM_FRACTION_THRESHOLD,
    min_thickness: float = VACUUM_MIN_THICKNESS,
) -> VacuumReport:
    """
    Report which lattice directions are mostly vacuum.

    Works by sorting the fractional coordinates along each axis and finding
    the largest gap between consecutive atoms, wrapping around the periodic
    boundary. That gap is converted to a real thickness using the
    interplanar spacing, so the result stays correct for non-orthogonal
    cells.
    """

    import numpy as np

    lattice = getattr(structure, "lattice", None)

    if lattice is None or len(structure) == 0:
        return VacuumReport()

    # Perpendicular spacing between successive lattice planes, which is
    # what a fractional gap actually corresponds to in Angstrom.
    spacings = [
        1.0 / value
        for value in lattice.reciprocal_lattice_crystallographic.abc
    ]

    fracs = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)

    found: list[VacuumDirection] = []

    for axis in range(3):

        column = np.sort(fracs[:, axis])

        # Largest gap between neighbours, including across the boundary.
        interior = np.diff(column)

        wrap = (column[0] + 1.0) - column[-1]

        gap = float(max(interior.max() if interior.size else 0.0, wrap))

        thickness = gap * spacings[axis]

        if gap >= fraction_threshold and thickness >= min_thickness:

            found.append(
                VacuumDirection(
                    axis=axis,
                    label=AXIS_LABELS[axis],
                    gap_fraction=gap,
                    thickness=thickness,
                )
            )

    return VacuumReport(found)


# ------------------------------------------------------------
# Cost
# ------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Render a number of seconds as a short, readable duration."""

    if seconds < 90.0:
        return f"{seconds:.0f} s"

    minutes = seconds / 60.0

    if minutes < 90.0:
        return f"{minutes:.0f} min"

    hours = minutes / 60.0

    if hours < 48.0:

        whole = int(hours)
        remainder = int(round((hours - whole) * 60.0))

        if remainder:
            return f"{whole} h {remainder} min"

        return f"{whole} h"

    return f"{hours / 24.0:.1f} days"


def estimate_cost(
    n_structures: int,
    n_atoms: int,
    seconds_per_atom: float = SECONDS_PER_ATOM,
) -> str:
    """
    Describe the size of a pending run, for a confirm-before-run dialog.

    The time is a rough order of magnitude, not a promise -- it is quoted as
    a range for that reason.
    """

    total_evaluations = max(n_structures, 0) * max(n_atoms, 0)

    seconds = total_evaluations * seconds_per_atom

    lines = [
        f"Structures to evaluate : {n_structures:,}",
        f"Atoms per structure    : {n_atoms:,}",
        f"Total atom evaluations : {total_evaluations:,}",
        f"Estimated run time     : "
        f"{format_duration(seconds * 0.5)} - "
        f"{format_duration(seconds * 3.0)}",
    ]

    if seconds >= LONG_RUN_SECONDS:
        lines.append("")
        lines.append(
            "This is a long run. Reducing the structure to its primitive "
            "cell, or using a smaller supercell, would cut it "
            "substantially."
        )

    return "\n".join(lines)

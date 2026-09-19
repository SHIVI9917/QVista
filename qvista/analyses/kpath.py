"""
High-symmetry k-path through the Brillouin zone.

Wraps ``seekpath.get_path``, which returns both the path and the primitive
cell it is expressed in.

Why the primitive cell matters
------------------------------
seekpath standardises whatever cell it is given, and the ``point_coords``
and ``path`` it returns are in the reciprocal basis of *that standardised
primitive cell* -- not of the cell that was passed in. For a conventional
cell the two differ badly: for conventional-cell silicon the point ``X``
lands 0.23 1/Angstrom away depending on which basis is used, which is the
scale of the Brillouin zone itself.

Discarding the primitive cell is therefore a silent correctness bug: the
q-points and any force constants computed later end up in different bases
and the resulting band structure is wrong without ever raising an error.
:class:`KPathResult` keeps the primitive cell as the reference cell, and
downstream phonon work must build its supercell from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from qvista.analyses.base import Analysis, AnalysisError, AnalysisResult
from qvista.models.project import Project
from qvista.services.structure_analyzer import DEFAULT_SYMPREC

#: Every path must begin and end at the zone centre.
GAMMA = "GAMMA"

#: Readable names for seekpath's two-letter Bravais lattice codes.
BRAVAIS_NAMES = {
    "aP": "Triclinic Primitive",
    "mP": "Monoclinic Primitive",
    "mC": "Monoclinic Base-Centered",
    "oP": "Orthorhombic Primitive",
    "oC": "Orthorhombic Base-Centered (C)",
    "oA": "Orthorhombic Base-Centered (A)",
    "oF": "Orthorhombic Face-Centered",
    "oI": "Orthorhombic Body-Centered",
    "tP": "Tetragonal Primitive",
    "tI": "Tetragonal Body-Centered",
    "hP": "Hexagonal Primitive",
    "hR": "Rhombohedral",
    "cP": "Cubic Primitive",
    "cF": "Cubic Face-Centered",
    "cI": "Cubic Body-Centered",
}


def bravais_name(code: str) -> str:
    """
    Render a Bravais code as ``Cubic Face-Centered (cF)``.

    seekpath returns both a plain code ("cF") and an extended one ("cF2");
    the extended form carries a variant digit, so the lookup uses the first
    two characters.
    """

    if not code:
        return ""

    name = BRAVAIS_NAMES.get(code[:2], "Unknown")

    return f"{name} ({code})"

#: Accepted separators when a path is typed by hand.
_SEPARATORS = ("->", "→", ",", " ", "/")


# ------------------------------------------------------------
# Primitive cell
# ------------------------------------------------------------


@dataclass
class PrimitiveCell:
    """
    The standardised primitive cell seekpath used.

    This is the reference cell for the k-path: the fractional coordinates in
    :attr:`KPathResult.points` are relative to
    :attr:`reciprocal_lattice`. Any later calculation that consumes the path
    -- force constants, dynamical matrices -- must be built on this cell.
    """

    #: Direct lattice, rows are the cell vectors, in Angstrom.
    lattice: Any

    #: Fractional coordinates of the atoms.
    positions: Any

    #: Atomic numbers.
    types: Any

    #: Reciprocal lattice as seekpath returns it, which **includes the
    #: 2*pi factor**: ``B = 2 * pi * (A^-1).T``, in 1/Angstrom. Code that
    #: expects the crystallographic (2*pi-free) convention -- phonopy's
    #: q-point handling among others -- must divide by 2*pi first.
    reciprocal_lattice: Any

    @property
    def natoms(self) -> int:
        """Number of atoms in the primitive cell."""

        return len(self.types)

    @property
    def volume(self) -> float:
        """Cell volume in cubic Angstrom."""

        import numpy as np

        return float(abs(np.linalg.det(np.asarray(self.lattice, dtype=float))))

    def to_pymatgen(self):
        """Return this cell as a pymatgen ``Structure``."""

        from pymatgen.core import Structure

        return Structure(
            lattice=self.lattice,
            species=[int(z) for z in self.types],
            coords=self.positions,
            coords_are_cartesian=False,
        )


# ------------------------------------------------------------
# Result
# ------------------------------------------------------------


@dataclass
class KPathResult(AnalysisResult):
    """The high-symmetry points, the chosen path, and its reference cell."""

    #: Label -> fractional coordinate, in the primitive reciprocal basis.
    points: dict[str, Any] = field(default_factory=dict)

    #: Consecutive ``(start, end)`` label pairs.
    segments: list[tuple[str, str]] = field(default_factory=list)

    #: The cell the path is expressed in. Never discard this.
    primitive: PrimitiveCell | None = None

    spacegroup_symbol: str = ""
    spacegroup_number: int = 0
    bravais_lattice: str = ""

    symprec: float = DEFAULT_SYMPREC

    #: Volume of the input cell divided by the primitive volume. Greater
    #: than 1 means the input was not primitive.
    volume_ratio: float = 1.0

    #: True when the user supplied the path rather than taking seekpath's.
    is_custom: bool = False

    #: Points per segment, used by :meth:`to_kpoints`.
    npoints: int = 20

    # ------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------

    def branches(self) -> list[list[str]]:
        """The path split into continuous branches."""

        return group_path_segments(self.segments)

    def cartesian_points(self) -> dict[str, Any]:
        """
        High-symmetry points in Cartesian reciprocal space, 1/Angstrom.

        Uses the primitive reciprocal lattice, which carries the 2*pi
        factor, so these are directly plottable against a Brillouin zone
        built from the same lattice.
        """

        import numpy as np

        if self.primitive is None:
            return {}

        basis = np.asarray(self.primitive.reciprocal_lattice, dtype=float)

        return {
            label: np.asarray(coords, dtype=float) @ basis
            for label, coords in self.points.items()
        }

    def to_kpoints(self, npoints: int | None = None) -> str:
        """
        Render the path as a VASP line-mode KPOINTS file.

        Junction points are written twice, once closing a segment and once
        opening the next, which is what line mode expects. Defaults to
        :attr:`npoints`, the density chosen when the path was built.
        """

        if npoints is None:
            npoints = self.npoints

        lines = [
            "High-symmetry k-point path (QVista)",
            str(int(npoints)),
            "Line-mode",
            "reciprocal",
        ]

        for branch in self.branches():

            for start, end in zip(branch[:-1], branch[1:]):

                for label in (start, end):

                    coords = self.points.get(label)

                    if coords is None:
                        continue

                    name = "Gamma" if label == GAMMA else label

                    lines.append(
                        f"{float(coords[0]):.6f} "
                        f"{float(coords[1]):.6f} "
                        f"{float(coords[2]):.6f} ! {name}"
                    )

                lines.append("")

        return "\n".join(lines) + "\n"

    def path_string(self) -> str:
        """Render the path as ``GAMMA -> X -> W``, marking any breaks."""

        if not self.segments:
            return ""

        parts = [self.segments[0][0]]

        for index, (start, end) in enumerate(self.segments):

            if index > 0 and start != self.segments[index - 1][1]:
                # seekpath paths may jump; "|" is the usual mark for a
                # break between disconnected stretches of the path.
                parts.append(f"| {start}")

            parts.append(f"-> {end}")

        return " ".join(parts)

    def as_storage(self) -> dict[str, Any]:
        """
        Package the result for :class:`~qvista.models.results.Results`.

        Includes the result object itself so the phonon workflow can reuse
        the primitive cell without recomputing the path.
        """

        return {
            "points": self.points,
            "segments": self.segments,
            "primitive_lattice": None
            if self.primitive is None
            else self.primitive.lattice,
            "primitive_positions": None
            if self.primitive is None
            else self.primitive.positions,
            "primitive_types": None
            if self.primitive is None
            else self.primitive.types,
            "reciprocal_primitive_lattice": None
            if self.primitive is None
            else self.primitive.reciprocal_lattice,
            "spacegroup_symbol": self.spacegroup_symbol,
            "spacegroup_number": self.spacegroup_number,
            "bravais_lattice": self.bravais_lattice,
            "symprec": self.symprec,
            "is_custom": self.is_custom,
            "result": self,
        }


# ------------------------------------------------------------
# Path parsing and validation
# ------------------------------------------------------------


def parse_path_text(text: str) -> list[str]:
    """
    Split a typed path such as ``GAMMA-X-W-K-GAMMA`` into labels.

    Hyphens, arrows, commas, slashes and spaces all separate labels, so the
    user does not have to remember one syntax.
    """

    normalised = text.strip()

    for separator in _SEPARATORS:
        normalised = normalised.replace(separator, "-")

    return [
        part.strip().upper()
        for part in normalised.split("-")
        if part.strip()
    ]


def validate_path_labels(
    labels: Sequence[str],
    points: dict[str, Any],
) -> str | None:
    """
    Check a hand-built path against the points this lattice actually has.

    Returns ``None`` when the path is usable, otherwise a message written
    for the user. Returning rather than raising lets the dialog show the
    problem inline while it is being typed.
    """

    if len(labels) < 2:
        return "A path needs at least two points, for example GAMMA-X-GAMMA."

    if labels[0] != GAMMA:
        return (
            f"The path must start at {GAMMA}, but it starts at "
            f"{labels[0]}."
        )

    if labels[-1] != GAMMA:
        return (
            f"The path must end at {GAMMA}, but it ends at {labels[-1]}."
        )

    unknown = [label for label in labels if label not in points]

    if unknown:

        available = ", ".join(sorted(points))

        unique_unknown = ", ".join(dict.fromkeys(unknown))

        return (
            f"Not a high-symmetry point of this lattice: {unique_unknown}.\n"
            f"Available points: {available}"
        )

    for index in range(len(labels) - 1):

        if labels[index] == labels[index + 1]:
            return (
                f"{labels[index]} is repeated back to back at position "
                f"{index + 1}; a segment must join two different points."
            )

    return None


def validate_segment_list(
    segments: Sequence[tuple[str, str]],
    points: dict[str, Any],
) -> str | None:
    """
    Check a hand-assembled list of ``(start, end)`` segments.

    Unlike :func:`validate_path_labels`, this does not require the path to
    start or end at GAMMA or to form one continuous chain -- a custom
    selection may freely mix disconnected segments (``X-U``, ``Y-T``, ...),
    the same way the recommended path itself does. Returns ``None`` when
    the selection is usable, otherwise a message written for the user.
    """

    if not segments:
        return "Select at least one segment, or add a custom one."

    for start, end in segments:

        if start not in points or end not in points:

            unknown = [
                label
                for label in (start, end)
                if label not in points
            ]

            available = ", ".join(sorted(points))

            return (
                f"Not a high-symmetry point of this lattice: "
                f"{', '.join(dict.fromkeys(unknown))}.\n"
                f"Available points: {available}"
            )

        if start == end:
            return f"{start} is repeated: a segment must join two different points."

    return None


def segments_from_labels(
    labels: Sequence[str],
) -> list[tuple[str, str]]:
    """Turn ``[A, B, C]`` into ``[(A, B), (B, C)]``."""

    return [
        (labels[index], labels[index + 1])
        for index in range(len(labels) - 1)
    ]


def group_path_segments(
    segments: Sequence[tuple[str, str]],
) -> list[list[str]]:
    """
    Turn ``[(A,B),(B,C),(D,E)]`` into ``[[A,B,C],[D,E]]``.

    A recommended path may jump; each continuous stretch becomes its own
    branch so the break is preserved rather than drawn as a segment.
    """

    if not segments:
        return []

    branches: list[list[str]] = []

    current = [segments[0][0], segments[0][1]]

    for start, end in segments[1:]:

        if start == current[-1]:
            current.append(end)
        else:
            branches.append(current)
            current = [start, end]

    branches.append(current)

    return branches


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------


class KPathAnalysis(Analysis):
    """
    Determine the high-symmetry k-path for the loaded structure.

    With both ``custom_labels`` and ``custom_segments`` unset, the workflow
    takes seekpath's recommended path. ``custom_segments`` -- an ordered
    list of ``(start, end)`` pairs, possibly spanning several disconnected
    branches -- takes priority if given; otherwise ``custom_labels``, a
    single chain of point labels, is used after validating it.
    """

    name = "High Symmetry Pathway"

    requires_periodic = True

    def __init__(
        self,
        project: Project,
        symprec: float = DEFAULT_SYMPREC,
        custom_labels: Sequence[str] | None = None,
        custom_segments: Sequence[tuple[str, str]] | None = None,
        with_time_reversal: bool = True,
        npoints: int = 20,
    ) -> None:

        super().__init__(project, symprec)

        self.custom_labels = list(custom_labels) if custom_labels else None
        self.custom_segments = (
            list(custom_segments) if custom_segments else None
        )
        self.with_time_reversal = with_time_reversal
        self.npoints = npoints

        self._data: dict[str, Any] | None = None

    # ------------------------------------------------------------
    # seekpath
    # ------------------------------------------------------------

    def seekpath_data(self) -> dict[str, Any]:
        """
        Run seekpath and cache the result.

        Deliberately callable from the GUI thread: the dialogs need the
        available points before the user can choose a path, and seekpath
        takes milliseconds. :meth:`execute` reuses this cache, so the
        calculation is never repeated.
        """

        if self._data is not None:
            return self._data

        structure = self.structure()

        try:
            import seekpath
            from pymatgen.io.ase import AseAtomsAdaptor

            atoms = AseAtomsAdaptor.get_atoms(structure)

            cell = (
                atoms.cell,
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers(),
            )

            self._data = seekpath.get_path(
                cell,
                with_time_reversal=self.with_time_reversal,
                symprec=self.symprec,
            )

        except AnalysisError:
            raise

        except Exception as exc:
            raise AnalysisError(
                f"Could not determine the k-path: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return self._data

    def available_points(self) -> dict[str, Any]:
        """Return the high-symmetry points of this lattice."""

        return dict(self.seekpath_data()["point_coords"])

    def recommended_segments(self) -> list[tuple[str, str]]:
        """Return seekpath's recommended path as ``(start, end)`` pairs."""

        return [tuple(pair) for pair in self.seekpath_data()["path"]]

    def primitive_cell(self) -> PrimitiveCell:
        """Return the cell seekpath expressed the path in."""

        data = self.seekpath_data()

        return PrimitiveCell(
            lattice=data["primitive_lattice"],
            positions=data["primitive_positions"],
            types=data["primitive_types"],
            reciprocal_lattice=data["reciprocal_primitive_lattice"],
        )

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    def execute(self) -> KPathResult:
        """Build the k-path and describe it for the console."""

        self.report("Standardising cell and analysing symmetry...")

        data = self.seekpath_data()

        self.raise_if_cancelled()

        points = dict(data["point_coords"])

        # ----------------------------------------------------------
        # Path selection
        # ----------------------------------------------------------

        if self.custom_segments:

            self.report("Validating the requested path...")

            error = validate_segment_list(self.custom_segments, points)

            if error is not None:
                raise AnalysisError(error)

            segments = list(self.custom_segments)

        elif self.custom_labels:

            self.report("Validating the requested path...")

            error = validate_path_labels(self.custom_labels, points)

            if error is not None:
                raise AnalysisError(error)

            segments = segments_from_labels(self.custom_labels)

        else:

            self.report("Taking the recommended path...")

            segments = [tuple(pair) for pair in data["path"]]

        self.raise_if_cancelled()

        primitive = self.primitive_cell()

        # ----------------------------------------------------------
        # Result
        # ----------------------------------------------------------

        result = KPathResult(
            title="",
            points=points,
            segments=segments,
            primitive=primitive,
            spacegroup_symbol=data.get("spacegroup_international", ""),
            spacegroup_number=int(data.get("spacegroup_number", 0)),
            bravais_lattice=data.get("bravais_lattice_extended", ""),
            symprec=self.symprec,
            volume_ratio=float(data.get("volume_original_wrt_prim", 1.0)),
            is_custom=bool(self.custom_segments or self.custom_labels),
            npoints=self.npoints,
        )

        result.rows = self._rows(result)
        result.notes = self._notes(result)

        result.data = {
            "point_coords": points,
            "path": segments,
        }

        return result

    # ------------------------------------------------------------
    # Console text
    # ------------------------------------------------------------

    def _rows(self, result: KPathResult) -> list[tuple[str, str]]:
        """Summary rows, in the same shape as the structure properties."""

        primitive = result.primitive

        rows = [
            ("Space group", result.spacegroup_symbol),
            ("Space group number", str(result.spacegroup_number)),
            ("Bravais lattice", bravais_name(result.bravais_lattice)),
            ("Symmetry tolerance", f"{result.symprec} Å"),
            (
                "Path source",
                "user defined" if result.is_custom else "seekpath (HPKOT)",
            ),
            ("Symmetry points", str(len(result.points))),
            ("Path segments", str(len(result.segments))),
        ]

        if primitive is not None:
            rows.append(
                ("Primitive atoms", str(primitive.natoms))
            )
            rows.append(
                ("Input/primitive vol", f"{result.volume_ratio:.3g}")
            )

        return rows

    def _notes(self, result: KPathResult) -> list[str]:
        """The point list, the path, and any warnings."""

        notes: list[str] = []

        # ----------------------------------------------------------
        # Available points
        # ----------------------------------------------------------

        lines = [f"Available high-symmetry points ({len(result.points)})"]

        for label in sorted(result.points):

            coords = result.points[label]

            lines.append(
                f"  {label:<8} "
                f"{float(coords[0]):8.4f} "
                f"{float(coords[1]):8.4f} "
                f"{float(coords[2]):8.4f}"
            )

        notes.append("\n".join(lines))

        # ----------------------------------------------------------
        # Selected segments
        # ----------------------------------------------------------

        lines = [f"Path segments ({len(result.segments)})"]

        for index, (start, end) in enumerate(result.segments, start=1):
            lines.append(f"  {index:>2}. {start} → {end}")

        lines.append("")
        lines.append(f"  {result.path_string()}")

        notes.append("\n".join(lines))

        # ----------------------------------------------------------
        # Warnings
        # ----------------------------------------------------------

        if result.spacegroup_number == 1:
            notes.append(
                "WARNING: symmetry came out as P1 (#1). A triclinic path "
                "has almost no high-symmetry points, and this usually means "
                "an unrelaxed or noisy structure rather than genuinely "
                "triclinic material. Try a looser symmetry tolerance."
            )

        if result.volume_ratio > 1.0001:
            notes.append(
                f"Note: the input cell is {result.volume_ratio:.3g}x the "
                "primitive cell. The path above is expressed in the "
                "reciprocal basis of the primitive cell, which is the cell "
                "any later phonon calculation must also use."
            )

        return notes

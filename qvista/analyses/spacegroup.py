"""
Space group analysis.

Reports the full symmetry description of the loaded structure: space group
symbol and number, Hall symbol, point group, crystal system and lattice
type, the number of symmetry operations, and the Wyckoff orbit each atom
belongs to.

Everything here is derived from spglib through pymatgen's
``SpacegroupAnalyzer``, so the answer depends on the symmetry tolerance. The
tolerance is exposed rather than hidden: a relaxed or noisy structure will
read as P1 at a tight tolerance and recover its true symmetry at a looser
one, and the user needs to see and control that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qvista.analyses.base import Analysis, AnalysisError, AnalysisResult
from qvista.models.project import Project
from qvista.services.structure_analyzer import DEFAULT_SYMPREC


# ------------------------------------------------------------
# Wyckoff orbits
# ------------------------------------------------------------


@dataclass
class WyckoffSite:
    """One symmetry-distinct orbit of atoms."""

    species: str
    wyckoff_letter: str
    site_symmetry: str
    multiplicity: int

    #: Fractional coordinates of the orbit's representative atom.
    coords: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def label(self) -> str:
        """Conventional Wyckoff label, e.g. ``4a``."""

        return f"{self.multiplicity}{self.wyckoff_letter}"


# ------------------------------------------------------------
# Result
# ------------------------------------------------------------


@dataclass
class SpaceGroupResult(AnalysisResult):
    """The full symmetry description of a structure."""

    symbol: str = ""
    number: int = 0
    hall: str = ""
    point_group: str = ""
    crystal_system: str = ""
    lattice_type: str = ""

    n_operations: int = 0
    centrosymmetric: bool = False

    sites: list[WyckoffSite] = field(default_factory=list)

    n_input: int = 0
    n_primitive: int = 0
    n_conventional: int = 0

    symprec: float = DEFAULT_SYMPREC

    # ----------------------------------------------------------
    # Cell description
    # ----------------------------------------------------------

    full_formula: str = ""
    reduced_formula: str = ""

    lattice_abc: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lattice_angles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pbc: tuple[bool, bool, bool] = (True, True, True)
    volume: float = 0.0

    #: Every atom: ``(index, element, label, x, y, z)``.
    atom_listing: list[tuple] = field(default_factory=list)

    @property
    def is_p1(self) -> bool:
        """True when no symmetry beyond the identity was found."""

        return self.number == 1

    def as_storage(self) -> dict[str, Any]:
        """Package for :class:`~qvista.models.results.Results`."""

        return {
            "symbol": self.symbol,
            "number": self.number,
            "hall": self.hall,
            "point_group": self.point_group,
            "crystal_system": self.crystal_system,
            "lattice_type": self.lattice_type,
            "sites": self.sites,
            "symprec": self.symprec,
            "result": self,
        }


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------


class SpaceGroupAnalysis(Analysis):
    """Determine the space group and Wyckoff positions."""

    name = "Space Group Analyzer"

    requires_periodic = True

    def __init__(
        self,
        project: Project,
        symprec: float = DEFAULT_SYMPREC,
    ) -> None:

        super().__init__(project, symprec)

    # ------------------------------------------------------------

    def execute(self) -> SpaceGroupResult:
        """Run spglib and describe what it found."""

        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        self.report("Analysing symmetry...")

        structure = self.structure()

        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=self.symprec)

            result = SpaceGroupResult(
                symbol=analyzer.get_space_group_symbol(),
                number=int(analyzer.get_space_group_number()),
                hall=analyzer.get_hall(),
                point_group=analyzer.get_point_group_symbol(),
                crystal_system=analyzer.get_crystal_system().capitalize(),
                lattice_type=analyzer.get_lattice_type().capitalize(),
                n_operations=len(analyzer.get_symmetry_operations()),
                # The 11 Laue classes are exactly the centrosymmetric
                # point groups, so this is the inversion test.
                centrosymmetric=bool(analyzer.is_laue()),
                n_input=len(structure),
                symprec=self.symprec,
            )

        except Exception as exc:
            raise AnalysisError(
                f"Symmetry analysis failed at symprec={self.symprec}: "
                f"{exc}\n\n"
                f"A relaxed or noisy structure may need a looser tolerance."
            ) from exc

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # Cell description
        # ----------------------------------------------------------

        self._describe_cell(result, structure)

        # ----------------------------------------------------------
        # Reduced cells
        # ----------------------------------------------------------

        self.report("Reducing cell...")

        try:
            result.n_primitive = len(
                analyzer.get_primitive_standard_structure()
            )
            result.n_conventional = len(
                analyzer.get_conventional_standard_structure()
            )

        except Exception:
            # Cell reduction can fail on pathological inputs; the space
            # group itself is still worth reporting.
            result.n_primitive = 0
            result.n_conventional = 0

        self.raise_if_cancelled()

        # ----------------------------------------------------------
        # Wyckoff orbits
        # ----------------------------------------------------------

        self.report("Assigning Wyckoff positions...")

        result.sites = self._wyckoff_sites(analyzer, structure)

        result.rows = self._rows(result)
        result.notes = self._notes(result)

        return result

    # ------------------------------------------------------------

    def _describe_cell(self, result: SpaceGroupResult, structure) -> None:
        """
        Record the lattice parameters and the full atom listing.

        Atom labels are numbered per element (Mg1, Mg2, ...), which is how
        the coordinates are conventionally referred to.
        """

        from collections import defaultdict

        composition = structure.composition

        result.full_formula = composition.formula
        result.reduced_formula = composition.reduced_formula

        lattice = structure.lattice

        result.lattice_abc = tuple(float(value) for value in lattice.abc)
        result.lattice_angles = tuple(
            float(value) for value in lattice.angles
        )
        result.pbc = tuple(bool(value) for value in lattice.pbc)
        result.volume = float(lattice.volume)

        counter: dict[str, int] = defaultdict(int)

        listing: list[tuple] = []

        for index, site in enumerate(structure, start=1):

            symbol = site.specie.symbol

            counter[symbol] += 1

            x, y, z = site.frac_coords

            listing.append(
                (
                    index,
                    symbol,
                    f"{symbol}{counter[symbol]}",
                    float(x),
                    float(y),
                    float(z),
                )
            )

        result.atom_listing = listing

    def _wyckoff_sites(self, analyzer, structure) -> list[WyckoffSite]:
        """
        Group atoms into symmetry-distinct orbits.

        ``equivalent_atoms`` maps every atom to the index of its orbit
        representative, so grouping by that value gives the orbits directly
        and keeps the Wyckoff letters and site symmetries aligned with the
        original site ordering.
        """

        try:
            dataset = analyzer.get_symmetry_dataset()

        except Exception:
            return []

        if dataset is None:
            return []

        # These come back as numpy arrays, so `value or []` would raise
        # "truth value of an array is ambiguous" rather than defaulting.
        def as_list(name: str) -> list:
            value = getattr(dataset, name, None)

            return [] if value is None else list(value)

        equivalent = as_list("equivalent_atoms")
        letters = as_list("wyckoffs")
        symmetries = as_list("site_symmetry_symbols")

        if not equivalent:
            return []

        sites: list[WyckoffSite] = []

        for representative in dict.fromkeys(equivalent):

            multiplicity = sum(
                1 for value in equivalent if value == representative
            )

            index = int(representative)

            site = structure[index]

            sites.append(
                WyckoffSite(
                    species=site.species_string,
                    wyckoff_letter=(
                        letters[index] if index < len(letters) else "?"
                    ),
                    site_symmetry=(
                        symmetries[index]
                        if index < len(symmetries)
                        else "?"
                    ),
                    multiplicity=multiplicity,
                    coords=tuple(float(value) for value in site.frac_coords),
                )
            )

        return sites

    # ------------------------------------------------------------
    # Console text
    # ------------------------------------------------------------

    def _rows(self, result: SpaceGroupResult) -> list[tuple[str, str]]:
        """Summary rows."""

        rows = [
            ("Space group", result.symbol),
            ("Space group number", str(result.number)),
            ("Hall symbol", result.hall),
            ("Point group", result.point_group),
            ("Crystal system", result.crystal_system),
            ("Lattice type", result.lattice_type),
            ("Symmetry operations", str(result.n_operations)),
            (
                "Centrosymmetric",
                "yes" if result.centrosymmetric else "no",
            ),
            ("Symmetry tolerance", f"{result.symprec} Å"),
            ("Atoms (input cell)", str(result.n_input)),
        ]

        if result.n_primitive:
            rows.append(("Atoms (primitive)", str(result.n_primitive)))

        if result.n_conventional:
            rows.append(
                ("Atoms (conventional)", str(result.n_conventional))
            )

        rows.append(("Distinct orbits", str(len(result.sites))))

        return rows

    def _notes(self, result: SpaceGroupResult) -> list[str]:
        """The Wyckoff table, plus warnings."""

        notes: list[str] = []

        # ----------------------------------------------------------
        # Lattice
        # ----------------------------------------------------------

        a, b, c = result.lattice_abc
        alpha, beta, gamma = result.lattice_angles

        notes.append(
            "\n".join(
                [
                    "Lattice",
                    f"  Full formula   : {result.full_formula}",
                    f"  Reduced formula: {result.reduced_formula}",
                    f"  abc            : "
                    f"{a:10.6f} {b:10.6f} {c:10.6f}",
                    f"  angles         : "
                    f"{alpha:10.6f} {beta:10.6f} {gamma:10.6f}",
                    f"  pbc            : "
                    f"{str(result.pbc[0]):>10}"
                    f"{str(result.pbc[1]):>10}"
                    f"{str(result.pbc[2]):>10}",
                    f"  volume         : {result.volume:.4f} Å³",
                ]
            )
        )

        # ----------------------------------------------------------
        # Atomic coordinates
        # ----------------------------------------------------------

        if result.atom_listing:

            lines = [
                f"Sites ({len(result.atom_listing)})",
                f"  {'#':>4}  {'Element':<8} {'Label':<8}"
                f"{'x':>12}{'y':>12}{'z':>12}",
            ]

            for index, element, label, x, y, z in result.atom_listing:
                lines.append(
                    f"  {index:>4}  {element:<8} {label:<8}"
                    f"{x:12.6f}{y:12.6f}{z:12.6f}"
                )

            notes.append("\n".join(lines))

        # ----------------------------------------------------------
        # Wyckoff table
        # ----------------------------------------------------------

        if result.sites:

            lines = [
                f"Wyckoff positions ({len(result.sites)} distinct orbits)",
                f"  {'Site':<8} {'Species':<8} {'Symmetry':<10} "
                f"{'x':>9} {'y':>9} {'z':>9}",
            ]

            for site in result.sites:

                x, y, z = site.coords

                lines.append(
                    f"  {site.label:<8} {site.species:<8} "
                    f"{site.site_symmetry:<10} "
                    f"{x:9.5f} {y:9.5f} {z:9.5f}"
                )

            notes.append("\n".join(lines))

        # ----------------------------------------------------------
        # Warnings
        # ----------------------------------------------------------

        if result.is_p1:
            notes.append(
                "WARNING: no symmetry was found beyond the identity (P1, "
                "#1). For a structure that should be symmetric this "
                "usually means an unrelaxed or noisy input rather than "
                "genuinely triclinic material. Try a looser symmetry "
                "tolerance."
            )

        # Reported independently of the P1 warning: a P1 supercell triggers
        # both, and the multiplicities above make no sense without this.
        if result.n_primitive and result.n_primitive < result.n_input:

            factor = result.n_input / result.n_primitive

            notes.append(
                f"The input cell is not primitive: {result.n_input} atoms "
                f"reduce to {result.n_primitive}, a factor of "
                f"{factor:.3g}.\n"
                f"  - The Wyckoff multiplicities above are counted in the "
                f"input cell, so they include that many pure lattice "
                f"translations. In the primitive cell each orbit is "
                f"{factor:.3g}x smaller.\n"
                f"  - 'Symmetry operations' likewise counts the operations "
                f"of the input cell, including those translations.\n"
                f"  - Analyses that scale with cell size will be "
                f"substantially cheaper on the primitive cell."
            )

        return notes

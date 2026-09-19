"""
Phonons from VASP, by way of phonopy.

The machine-learning route in :mod:`qvista.analyses.phonon` evaluates
forces itself and answers in seconds. This module does the opposite: it
writes the VASP runs out and waits. Nothing here launches VASP, and
nothing here needs it installed -- QVista prepares the input, the user
runs it on whatever machine has the licence, and then brings the output
back.

Two routes to the same force constants
--------------------------------------
**Finite displacement.** phonopy displaces one atom at a time in the
supercell and asks VASP for the forces that result. Every displacement is
an ordinary static run, so it parallelises across a cluster trivially and
works with any functional VASP can do -- hybrids, +U, van der Waals. The
cost is the number of displacements, which symmetry sets.

**DFPT.** VASP computes the second derivatives directly with linear
response (``IBRION = 8``), in a single run on the undisplaced supercell.
One job instead of dozens, but it is restricted to the functionals VASP
implements linear response for, and it will not run with ``NCORE`` or
``NPAR`` set.

Both start from the same place -- a unit cell and a supercell size -- and
end at the same place, force constants that phonopy turns into a band
structure and a density of states.

What gets written
-----------------
A *configuration*: one directory per VASP run, each holding the four
files VASP reads, plus the phonopy bookkeeping at the top so the run can
be read back without the user having to remember anything about it. ::

    <root>/
        phonopy_disp.yaml       what phonopy did, and how to undo it
        POSCAR-unitcell         the cell everything is expressed in
        SPOSCAR                 the perfect supercell
        README.txt
        Finite Displacement/
            Config_1/  POSCAR INCAR KPOINTS POTCAR
            Config_2/  ...

DFPT writes a ``DFPT/`` directory beside that one instead, holding the
perfect supercell as its POSCAR. Both routes can be written into the
same root, so the two are directly comparable.

It happens in two passes, because on screen it is two decisions:
:func:`write_structures` lays out the folders with only their POSCARs,
and :func:`write_vasp_inputs` fills in INCAR, KPOINTS and POTCAR
afterwards. :func:`write_configuration` does both for callers that
already know everything.

``phonopy_disp.yaml`` is the important one. It records the unit cell, the
supercell matrix and every displacement, so post-processing reconstructs
the whole calculation from the folder alone and never has to trust that
the project still open is the structure that was run.

The cell the supercell is built from
------------------------------------
The one the user opened, taken as given. That is phonopy's own
workflow: the unit cell is whatever is in the POSCAR, the supercell is
that cell repeated, and ``primitive_matrix="auto"`` leaves phonopy to
find the primitive cell for itself. The band path is then generated
from *that* primitive cell, so the q-points and the dynamical matrix
share a basis by construction -- there is no standardisation step for
the caller to get wrong, and no cell substituted behind their back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from qvista.analyses.kpoints import (
    GAMMA_SCHEME,
    KSPACING_PRESETS as _KSPACING_PRESETS,
    kpoints_content,
)

# ----------------------------------------------------------------
# Names
# ----------------------------------------------------------------

#: The two ways of getting force constants out of VASP.
FINITE_DISPLACEMENT = "Finite Displacement"
DFPT = "DFPT"

#: Everything one phonon experiment needs lives under a folder of this
#: name.
#:
#: Pre-processing writes the configuration inside it and post-processing
#: looks inside it, so the two ends agree on where the work is without
#: the user having to remember. It also keeps a phonon experiment --
#: which is one folder per displacement and can be a hundred of them --
#: from being scattered through whatever directory was chosen.
EXPERIMENT_DIR = "phonopyexp"

#: phonopy's own record of the displacements. Its presence is what marks
#: a directory as a QVista phonon configuration.
DISP_YAML = "phonopy_disp.yaml"

#: The perfect supercell, and the cell it was built from.
SPOSCAR = "SPOSCAR"
UNITCELL = "POSCAR-unitcell"

#: Directory names inside a configuration. Each route gets one folder of
#: its own next to the other: ``DFPT`` holds the single linear-response
#: job directly, and ``Finite Displacement`` holds ``Config_1``,
#: ``Config_2``, ... one per displaced supercell.
DFPT_DIR = "DFPT"
FD_DIR = "Finite Displacement"
CONFIG_PREFIX = "Config"
BORN_DIR = "born"


def config_name(index: int) -> str:
    """The folder one displacement goes in. ``index`` counts from 1."""

    return f"{CONFIG_PREFIX}_{int(index)}"


def config_index(name: str) -> int | None:
    """
    The number in ``Config_7``, or ``None`` if it is not one of ours.

    Used to sort the folders. Sorting them as text puts ``Config_10``
    before ``Config_2``, which would hand phonopy the force sets in the
    wrong order -- and that is not an error, it is a wrong answer.
    """

    prefix = f"{CONFIG_PREFIX}_"

    if not name.lower().startswith(prefix.lower()):
        return None

    tail = name[len(prefix):]

    return int(tail) if tail.isdigit() else None

#: What VASP leaves behind that post-processing reads.
VASPRUN = "vasprun.xml"

#: ...and the compressed forms of it that survive being copied off a
#: cluster. phonopy reads all three.
VASPRUN_NAMES = ("vasprun.xml", "vasprun.xml.gz", "vasprun.xml.xz")

#: Written by post-processing, next to the runs that produced them.
FORCE_SETS = "FORCE_SETS"
FORCE_CONSTANTS = "FORCE_CONSTANTS"
BAND_YAML = "band.yaml"
TOTAL_DOS = "total_dos.dat"
THERMAL_YAML = "thermal_properties.yaml"
BORN_FILE = "BORN"

#: phonopy's usual finite-difference step, in Angstrom.
DEFAULT_DISTANCE = 0.01

#: Target k-point spacing for the supercell runs, in 1/Angstrom -- VASP's
#: own KSPACING units, with the 2*pi in them. Applied to the *supercell*
#: reciprocal lattice, so a bigger supercell automatically gets a coarser
#: mesh at the same physical resolution, which is what the VASP wiki asks
#: for.
#:
#: Re-exported from the k-point module rather than written out again:
#: the K-points menu offers a resolution in units of 2*pi/Angstrom, and
#: two lists of numbers that look alike but mean things a factor of 2*pi
#: apart must not be maintained in two places.
KSPACING_PRESETS = _KSPACING_PRESETS

#: Tight enough that a phonon run is not dominated by k-point noise,
#: loose enough to be affordable on a supercell.
DEFAULT_KSPACING = 0.3

#: A supercell mesh never drops below this. Gamma alone in a direction is
#: a real choice for a long axis, but it should be the user's.
MIN_SUBDIVISION = 1


# ----------------------------------------------------------------
# The two routes
# ----------------------------------------------------------------


@dataclass(frozen=True)
class Method:
    """One of the two routes, and what the dialog puts on its button."""

    #: Button title, and the name after SYSTEM in the INCAR.
    name: str

    #: The short line under it. Kept to a handful of words.
    tagline: str

    #: The one tag that makes this route what it is: no ionic update for
    #: finite displacement, linear response for DFPT.
    ibrion: str

    #: True when the whole calculation is one VASP run.
    single_run: bool


METHODS: tuple[Method, ...] = (
    Method(
        name=DFPT,
        tagline="One run on SPOSCAR",
        ibrion="8",
        single_run=True,
    ),
    Method(
        name=FINITE_DISPLACEMENT,
        tagline="One run per displacement",
        ibrion="-1",
        single_run=False,
    ),
)


METHODS_BY_NAME: dict[str, Method] = {
    method.name: method for method in METHODS
}


def method_by_name(name: str) -> Method | None:
    """Look up a method by its card title."""

    return METHODS_BY_NAME.get(name)


# ----------------------------------------------------------------
# The INCAR
#
# One file, the same for both routes but for two lines: SYSTEM names the
# route, and IBRION is what makes it that route. Nothing is composed,
# nothing is optional, and nothing is commented tag by tag -- a phonon
# force run wants a short file whose every line is doing something.
# ----------------------------------------------------------------

#: The rule between sections.
BANNER = "# " + "=" * 58

#: Column the values line up in. The longest tag in the file is ADDGRID.
TAG_WIDTH = 7

#: The file, section by section. ENCUT and IBRION are filled in per run;
#: everything else is fixed.
SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Electronic Structure",
        (
            ("ENCUT", "520"),
            ("PREC", "Accurate"),
            ("ALGO", "Normal"),
        ),
    ),
    (
        "SCF Convergence",
        (
            ("EDIFF", "1E-8"),
        ),
    ),
    (
        "Ionic / Force Calculation",
        (
            ("IBRION", "-1"),
            ("NSW", "0"),
            ("ISIF", "2"),
        ),
    ),
    (
        "Smearing",
        (
            ("ISMEAR", "0"),
            ("SIGMA", "0.05"),
        ),
    ),
    (
        "Spin",
        (
            ("ISPIN", "1"),
        ),
    ),
    (
        "Precision / Numerical Accuracy",
        (
            ("LREAL", ".FALSE."),
            ("ADDGRID", ".TRUE."),
        ),
    ),
    (
        "Output",
        (
            ("LWAVE", ".FALSE."),
            ("LCHARG", ".FALSE."),
        ),
    ),
)


def phonon_incar(method: Method, encut: int | None = None) -> str:
    """
    The INCAR for one route.

    ``encut`` overrides the cutoff when one has been derived from the
    POTCARs; left out, the file keeps its own value.
    """

    override = {"IBRION": method.ibrion}

    if encut is not None:
        override["ENCUT"] = str(int(encut))

    lines = [f"SYSTEM = Phonopy - {method.name}", ""]

    for title, tags in SECTIONS:

        lines += [BANNER, f"# {title}", BANNER]

        for tag, value in tags:
            lines.append(
                f"{tag:<{TAG_WIDTH}} = {override.get(tag, value)}"
            )

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------
# Supercells and displacements
# ----------------------------------------------------------------


def build_phonon(
    structure,
    supercell: Sequence[int],
    symprec: float = 1e-5,
    distance: float = DEFAULT_DISTANCE,
):
    """
    A Phonopy object for ``structure``, with displacements generated.

    ``structure`` is the cell the user opened, used exactly as given --
    phonopy's own workflow, where the unit cell is whatever is in the
    POSCAR and the supercell is built by repeating it. Nothing is
    standardised or reduced first.

    ``primitive_matrix="auto"`` is the other half of that workflow, and
    is what ``--pa auto`` does on the command line: phonopy finds the
    primitive cell of the given unit cell itself and evaluates the
    dynamical matrix on it. The band path is then taken from that same
    primitive cell, so the two share a basis by construction rather than
    by the caller having arranged it.

    No forces are evaluated, so this is cheap enough to call from a
    dialog while the user changes their mind about the supercell.
    """

    import warnings

    import numpy as np
    from phonopy import Phonopy
    from pymatgen.io.phonopy import get_phonopy_structure

    with warnings.catch_warnings():

        # phonopy 4 warns whenever "auto" resolves to something other
        # than the identity, because it changed from v3. Here it is the
        # chosen behaviour, not a surprise, and the warning would fire
        # on every keystroke in the supercell boxes.
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*primitive_matrix defaulted to 'auto'.*",
        )

        phonon = Phonopy(
            get_phonopy_structure(structure),
            supercell_matrix=np.diag([int(value) for value in supercell]),
            primitive_matrix="auto",
            symprec=symprec,
        )

    phonon.generate_displacements(distance=distance)

    return phonon


def to_pymatgen(cell) -> Any:
    """A phonopy cell as a pymatgen ``Structure``."""

    from pymatgen.core import Structure

    return Structure(
        lattice=cell.cell,
        species=[str(symbol) for symbol in cell.symbols],
        coords=cell.scaled_positions,
        coords_are_cartesian=False,
    )


def mesh_for(structure, kspacing: float) -> tuple[int, int, int]:
    """
    The k-mesh for one cell at a target spacing.

    VASP's own KSPACING rule, applied to whichever cell is passed --
    which is how a supercell ends up with a coarser mesh than the unit
    cell at the same physical resolution.
    """

    from qvista.analyses.kpoints import grid_from_kspacing

    grid = grid_from_kspacing(structure, kspacing)

    return tuple(max(MIN_SUBDIVISION, int(value)) for value in grid)


def species_of(cell) -> tuple[list[str], list[int]]:
    """
    ``(elements, counts)`` for a phonopy cell, in its POSCAR order.

    A supercell repeats the unit cell's species blocks rather than
    interleaving them, but the counts are multiplied -- so a MAGMOM
    written for the unit cell is the wrong length here, and this is what
    it has to be written against instead.
    """

    elements: list[str] = []
    counts: list[int] = []

    for symbol in (str(item) for item in cell.symbols):

        if elements and elements[-1] == symbol:
            counts[-1] += 1
            continue

        if symbol in elements:
            # Not contiguous: fall back to a plain tally, which is what
            # write_vasp will group the file by anyway.
            return _tally(cell)

        elements.append(symbol)
        counts.append(1)

    return elements, counts


def _tally(cell) -> tuple[list[str], list[int]]:
    """Species counts for a cell whose symbols are not grouped."""

    order: list[str] = []
    totals: dict[str, int] = {}

    for symbol in (str(item) for item in cell.symbols):

        if symbol not in totals:
            order.append(symbol)
            totals[symbol] = 0

        totals[symbol] += 1

    return order, [totals[symbol] for symbol in order]


# ----------------------------------------------------------------
# What a configuration is
# ----------------------------------------------------------------

#: A folder holding a force calculation...
RUN_FORCES = "forces"

#: ...and the one holding the Born-charge calculation.
RUN_BORN = "born"


@dataclass
class Run:
    """One directory holding one VASP job."""

    path: Path
    label: str
    kind: str = RUN_FORCES

    #: Atoms in the cell this run acts on.
    natoms: int = 0

    #: The mesh written into its KPOINTS.
    mesh: tuple[int, int, int] = (1, 1, 1)


@dataclass
class Configuration:
    """What :func:`write_configuration` produced."""

    root: Path
    method: str

    supercell: tuple[int, int, int]
    distance: float
    symprec: float

    runs: list[Run] = field(default_factory=list)

    #: Files written at the top level of the configuration.
    files: list[Path] = field(default_factory=list)

    unitcell_natoms: int = 0
    supercell_natoms: int = 0
    n_displacements: int = 0

    encut: int = 0
    kspacing: float = DEFAULT_KSPACING
    mesh: tuple[int, int, int] = (1, 1, 1)

    #: True when the POTCARs were resolved and written.
    has_potcar: bool = False

    warnings: list[str] = field(default_factory=list)

    @property
    def force_runs(self) -> list[Run]:
        """The runs whose forces the force constants are built from."""

        return [run for run in self.runs if run.kind == RUN_FORCES]

    def rows(self) -> list[tuple[str, str]]:
        """Summary pairs, for the console."""

        nx, ny, nz = self.supercell
        kx, ky, kz = self.mesh

        rows = [
            ("Method", self.method),
            ("Configuration", str(self.root)),
            ("Supercell", f"{nx} x {ny} x {nz}"),
            ("Unit cell atoms", str(self.unitcell_natoms)),
            ("Supercell atoms", str(self.supercell_natoms)),
            ("VASP runs", str(len(self.runs))),
            ("ENCUT", f"{self.encut} eV"),
            (
                "Supercell k-mesh",
                f"{kx} x {ky} x {kz}  (KSPACING {self.kspacing:g})",
            ),
            ("POTCAR", "written" if self.has_potcar else "not written"),
        ]

        if self.method == FINITE_DISPLACEMENT:
            rows.insert(
                5, ("Displacements", str(self.n_displacements))
            )
            rows.insert(
                6, ("Displacement step", f"{self.distance} A")
            )

        return rows


# ----------------------------------------------------------------
# Writing it out
# ----------------------------------------------------------------


def write_structures(
    root: Path,
    phonon,
    method: Method,
    progress=None,
) -> Configuration:
    """
    Step one: the cells, and nothing else.

    Writes phonopy's bookkeeping at the top, then one folder per VASP
    run holding only its POSCAR -- ``DFPT/POSCAR`` from the perfect
    supercell, or ``Config_1/POSCAR``, ``Config_2/POSCAR``, ... from the
    displaced ones.

    The VASP input files come afterwards, from
    :func:`write_vasp_inputs`, because choosing a route and generating
    the inputs for it are two separate decisions on screen.

    ``progress`` is called as ``progress(index, total, message)``.
    """

    from phonopy.interface.vasp import write_vasp

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    displaced = list(phonon.supercells_with_displacements)

    if not method.single_run and not displaced:
        raise ValueError(
            "phonopy generated no displacements for this cell, so there "
            "is nothing to run."
        )

    supercell = tuple(
        int(abs(phonon.supercell_matrix[axis][axis])) for axis in range(3)
    )

    config = Configuration(
        root=root,
        method=method.name,
        supercell=supercell,
        distance=_distance_of(phonon),
        symprec=float(phonon.symmetry.tolerance),
        unitcell_natoms=len(phonon.unitcell),
        supercell_natoms=len(phonon.supercell),
        n_displacements=len(displaced),
    )

    # ----------------------------------------------------------
    # Top level: what post-processing needs to read this back
    # ----------------------------------------------------------

    write_vasp(root / UNITCELL, phonon.unitcell)
    write_vasp(root / SPOSCAR, phonon.supercell)

    phonon.save(root / DISP_YAML)

    config.files += [
        root / DISP_YAML,
        root / UNITCELL,
        root / SPOSCAR,
    ]

    # ----------------------------------------------------------
    # One folder per run
    # ----------------------------------------------------------

    if method.single_run:
        cells = [(root / DFPT_DIR, DFPT_DIR, phonon.supercell)]

    else:
        cells = [
            (root / FD_DIR / config_name(index), config_name(index), cell)
            for index, cell in enumerate(displaced, start=1)
        ]

    for index, (folder, label, cell) in enumerate(cells):

        if progress is not None:
            progress(index, len(cells), label)

        folder.mkdir(parents=True, exist_ok=True)

        write_vasp(folder / "POSCAR", cell)

        config.runs.append(
            Run(
                path=folder,
                label=label,
                kind=RUN_FORCES,
                natoms=len(cell),
            )
        )

    return config


def write_vasp_inputs(
    config: Configuration,
    phonon,
    encut: int,
    kspacing: float = DEFAULT_KSPACING,
    potcar: str = "",
    scheme: str = GAMMA_SCHEME,
    incar: str = "",
    mesh: Sequence[int] | None = None,
    progress=None,
) -> list[Path]:
    """
    Step two: INCAR, KPOINTS and POTCAR into the folders already written.

    ``potcar`` is the concatenated POTCAR text for the species in POSCAR
    order, or empty to leave the file out; nothing here reads a
    pseudopotential library. A missing POTCAR is recorded in
    :attr:`Configuration.warnings` rather than raised -- the rest of the
    configuration is still worth having.

    ``incar`` and ``mesh`` override what would otherwise be derived, so
    a file edited by hand and a mesh chosen in the k-points dialog go in
    verbatim. Left out, both are worked out from the method and the
    supercell.

    Returns every path written. ``config`` is updated in place with the
    cutoff, the mesh and any warning, so the caller can report them.
    """

    method = method_by_name(config.method)

    if method is None:
        raise ValueError(f"Unknown method: {config.method}")

    super_elements, _super_counts = species_of(phonon.supercell)

    grid = (
        tuple(int(value) for value in mesh)
        if mesh is not None
        else mesh_for(to_pymatgen(phonon.supercell), kspacing)
    )

    config.encut = int(encut)
    config.kspacing = kspacing
    config.mesh = grid
    config.has_potcar = bool(potcar)

    if not potcar:
        config.warnings.append(
            "No POTCAR was written. Set the pseudopotential library "
            "under File > Set POTCAR Library, or copy one into each "
            f"run folder by hand; the species order is "
            f"{' '.join(super_elements)}."
        )

    if not incar:
        incar = phonon_incar(method, encut=encut)

    kpoints = kpoints_content(*grid, scheme)

    written: list[Path] = []

    runs = config.force_runs

    for index, run in enumerate(runs):

        if progress is not None:
            progress(index, len(runs), run.label)

        run.mesh = grid

        written += _write_inputs_into(
            run.path, incar, kpoints, potcar
        )

    # ----------------------------------------------------------

    readme = config.root / "README.txt"
    readme.write_text(readme_text(config), encoding="utf-8")

    if readme not in config.files:
        config.files.append(readme)

    written.append(readme)

    return written


def _write_inputs_into(
    folder: Path,
    incar: str,
    kpoints: str,
    potcar: str,
) -> list[Path]:
    """Drop the three input files into one run folder."""

    written = []

    for name, text in (
        ("INCAR", incar),
        ("KPOINTS", kpoints),
        ("POTCAR", potcar),
    ):
        if not text:
            continue

        (folder / name).write_text(text, encoding="utf-8")

        written.append(folder / name)

    return written


def write_configuration(
    root: Path,
    structure,
    method: Method,
    supercell: Sequence[int],
    encut: int,
    symprec: float = 1e-5,
    distance: float = DEFAULT_DISTANCE,
    kspacing: float = DEFAULT_KSPACING,
    potcar: str = "",
    scheme: str = GAMMA_SCHEME,
    progress=None,
) -> Configuration:
    """
    Both steps at once, for callers that have already decided everything.

    ``structure`` must already be the reference cell -- see the module
    docstring.
    """

    phonon = build_phonon(structure, supercell, symprec, distance)

    config = write_structures(root, phonon, method, progress=progress)

    write_vasp_inputs(
        config,
        phonon,
        encut=encut,
        kspacing=kspacing,
        potcar=potcar,
        scheme=scheme,
    )

    return config


def _distance_of(phonon) -> float:
    """The finite-difference step in a built phonopy object, in Angstrom."""

    import numpy as np

    values = np.asarray(phonon.displacements, dtype=float)

    if values.size == 0:
        return 0.0

    if values.ndim == 2 and values.shape[1] == 4:
        values = values[:, 1:]

    return float(np.max(np.linalg.norm(values.reshape(-1, 3), axis=1)))


def readme_text(config: Configuration) -> str:
    """
    The note left at the top of a configuration.

    Short on purpose, and the only prose in the whole workflow: the gap
    between writing these folders and having the output is measured in
    days, so the folder has to say what to do with itself.
    """

    nx, ny, nz = config.supercell
    kx, ky, kz = config.mesh

    runs = config.force_runs

    where = (
        f"{DFPT_DIR}/"
        if config.method == DFPT
        else f"{FD_DIR}/{config_name(1)} ... {config_name(len(runs))}"
    )

    lines = [
        f"QVista phonon configuration -- {config.method}",
        "",
        f"Supercell   : {nx} x {ny} x {nz}  ({config.supercell_natoms} atoms)",
        f"VASP runs   : {len(runs)}  in {where}",
        f"ENCUT       : {config.encut} eV",
        f"K-mesh      : {kx} x {ky} x {kz}  (KSPACING {config.kspacing:g} 1/A)",
        "",
        "POSCAR-unitcell is the cell this was built from, as supplied.",
        "phonopy reduces it to its own primitive cell and takes the band",
        "path from that, so the two always share a basis.",
        "",
    ]

    if config.method == DFPT:
        lines += [
            f"Run VASP once in {DFPT_DIR}/. Do not add NCORE or NPAR --",
            "VASP does not support them with IBRION = 8. Use KPAR.",
        ]

    else:
        lines += [
            f"Run VASP in each of the {len(runs)} {CONFIG_PREFIX}_*",
            f"folders under {FD_DIR}/. They are independent, so submit",
            "them all at once.",
        ]

    lines += [
        "",
        "Then keep every vasprun.xml in place and open",
        "Phonon Analysis > VASP > Post-Processing on this folder.",
    ]

    if config.warnings:
        lines += [""] + [f"NOTE: {text}" for text in config.warnings]

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------
# Reading a configuration back
# ----------------------------------------------------------------


@dataclass
class Discovery:
    """What was found in a folder handed to post-processing."""

    root: Path

    #: :data:`FINITE_DISPLACEMENT`, :data:`DFPT`, or empty when unknown.
    method: str = ""

    #: Run folders holding a readable vasprun.xml, in run order.
    found: list[Path] = field(default_factory=list)

    #: Run folders that exist but have no vasprun.xml yet.
    missing: list[str] = field(default_factory=list)

    #: The Born run's vasprun.xml, when there is one.
    born: Path | None = None

    supercell: tuple[int, int, int] = (1, 1, 1)

    #: How many force runs the configuration was written with.
    expected: int = 0

    unitcell_natoms: int = 0
    supercell_natoms: int = 0

    #: Why this folder cannot be used, if it cannot be.
    problems: list[str] = field(default_factory=list)

    #: Other configurations found beside the one chosen, when the folder
    #: handed over held more than one.
    siblings: list[Path] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when post-processing has everything it needs."""

        return not self.problems and bool(self.found) and not self.missing

    @property
    def complete(self) -> bool:
        """True when every expected run has produced output."""

        return bool(self.found) and not self.missing

    def summary(self) -> str:
        """One paragraph describing the folder, for the dialog."""

        if self.problems:
            return "\n".join(self.problems)

        nx, ny, nz = self.supercell

        lines = [
            f"{self.method} configuration, supercell {nx} x {ny} x {nz}, "
            f"{self.supercell_natoms} atoms.",
        ]

        if self.method == DFPT:
            lines.append(
                f"Force constants come from the Hessian in "
                f"{DFPT_DIR}/{VASPRUN}, which is "
                + ("there." if self.found else "not there yet.")
            )

        else:
            lines.append(
                f"{len(self.found)} of {self.expected} displacement runs "
                f"have a {VASPRUN}."
            )

        if self.missing:
            shown = ", ".join(self.missing[:8])
            more = (
                f" and {len(self.missing) - 8} more"
                if len(self.missing) > 8
                else ""
            )
            lines.append(f"Still waiting on: {shown}{more}.")

        if self.born is not None:
            lines.append(
                "A born/ run was found; the non-analytical correction "
                "will be applied."
            )

        return "\n".join(lines)


def is_configuration(folder: Path) -> bool:
    """True when this folder is a phonon configuration itself."""

    return (Path(folder) / DISP_YAML).is_file()


def configurations_under(root: Path) -> list[Path]:
    """
    Every configuration at or below ``root``, nearest first.

    Looked for in three places, in the order someone would think of
    them: the folder itself, inside its ``phonopyexp`` folder, and one
    level down.  Post-processing can then be handed the directory the
    experiment was saved in, the experiment folder, or the configuration
    -- all three name the same thing to a user and all three now work.
    """

    root = Path(root)

    if not root.is_dir():
        return []

    if is_configuration(root):
        return [root]

    found: list[Path] = []

    experiment = root / EXPERIMENT_DIR

    if experiment.is_dir():

        if is_configuration(experiment):
            found.append(experiment)

        found.extend(
            child
            for child in sorted(experiment.iterdir())
            if child.is_dir() and is_configuration(child)
        )

    found.extend(
        child
        for child in sorted(root.iterdir())
        if child.is_dir()
        and child.name != EXPERIMENT_DIR
        and is_configuration(child)
    )

    return found


def vasprun_in(folder: Path) -> Path | None:
    """The vasprun.xml in one run folder, compressed or not."""

    for name in VASPRUN_NAMES:

        candidate = folder / name

        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    return None


def discover(root: Path) -> Discovery:
    """
    Work out what a folder holds, without importing phonopy.

    Deliberately cheap and side-effect free: the post-processing dialog
    calls it every time the user browses to a different directory, and
    it has to be able to say "this is not one of ours" as readily as
    "this is ready".
    """

    chosen = Path(root)

    # Accept the experiment folder, or the directory holding it, and
    # walk down to the configuration rather than refusing: all three are
    # "the phonon folder" as far as anyone using this is concerned.
    candidates = configurations_under(chosen)

    root = candidates[0] if candidates else chosen

    found = Discovery(root=root)

    if len(candidates) > 1:
        found.siblings = candidates

    if not root.is_dir():
        found.problems.append(f"{root} is not a folder.")
        return found

    disp = root / DISP_YAML

    if not disp.is_file():
        found.problems.append(
            f"No {DISP_YAML} in {chosen}, nor in a {EXPERIMENT_DIR}/ "
            f"folder inside it.\n\n"
            f"Pick the folder holding {DISP_YAML}, {SPOSCAR} and the "
            f"{FD_DIR}/ or {DFPT_DIR}/ folders -- the {EXPERIMENT_DIR}/ "
            f"folder pre-processing wrote, or the directory it sits in "
            f"-- not one of the runs itself."
        )
        return found

    header = _read_yaml_header(disp)

    found.supercell = header.get("supercell", (1, 1, 1))
    found.unitcell_natoms = header.get("unitcell_natoms", 0)
    found.supercell_natoms = header.get("supercell_natoms", 0)

    # ----------------------------------------------------------
    # Which route was written
    # ----------------------------------------------------------

    dfpt = root / DFPT_DIR

    # The displacement runs live under their own folder, beside DFPT/.
    series = root / FD_DIR

    # Ordered by the number in the name, not by the name. Sorted as text,
    # Config_10 lands before Config_2, and phonopy would be handed the
    # force sets in an order that does not match its displacements --
    # which is not an error, it is a wrong dispersion.
    numbered = [
        (config_index(folder.name), folder)
        for folder in (series.iterdir() if series.is_dir() else ())
        if folder.is_dir()
    ]

    displacements = [
        folder
        for index, folder in sorted(
            (item for item in numbered if item[0] is not None),
            key=lambda item: item[0],
        )
    ]

    if dfpt.is_dir() and not displacements:

        found.method = DFPT
        found.expected = 1

        run = vasprun_in(dfpt)

        if run is None:
            found.missing.append(DFPT_DIR)
        else:
            found.found.append(run)

    elif displacements:

        found.method = FINITE_DISPLACEMENT
        found.expected = len(displacements)

        for folder in displacements:

            run = vasprun_in(folder)

            if run is None:
                found.missing.append(folder.name)
            else:
                found.found.append(run)

    else:
        found.problems.append(
            f"{root} holds a {DISP_YAML} but no run folders. Expected "
            f"{FD_DIR}/{config_name(1)}, {FD_DIR}/{config_name(2)}, ... "
            f"for a finite displacement set, or {DFPT_DIR}/ for a DFPT "
            f"run."
        )
        return found

    # Nothing run yet is not a problem with the folder: every run is
    # already in ``missing``, which is what the dialog shows and what
    # keeps ``usable`` False. Calling it a problem instead would hide
    # the listing behind an error message and say less.

    born = root / BORN_DIR

    if born.is_dir():
        found.born = vasprun_in(born)

    return found


def _read_yaml_header(path: Path) -> dict[str, Any]:
    """
    Pull the supercell matrix and the atom counts out of a disp yaml.

    Read line by line rather than with a YAML parser: this runs on every
    keystroke in a folder picker, and the interesting part is the first
    dozen lines of a file that can be megabytes long.
    """

    supercell = [1, 1, 1]
    counts: dict[str, int] = {}

    section = ""
    row = 0

    try:
        text = path.read_text(encoding="utf-8", errors="replace")

    except OSError:
        return {}

    for line in text.splitlines():

        stripped = line.lstrip()

        # A list item, not a heading -- phonopy writes the rows of a
        # matrix flush left, so the indentation test alone would read
        # "- [ 2, 0, 0 ]" as the start of a new section.
        item = stripped.startswith("-")

        if line and not line[0].isspace() and not item:
            section = line.split(":")[0].strip()
            row = 0
            continue

        if section == "supercell_matrix" and item:

            numbers = _numbers(line)

            if row < 3 and len(numbers) == 3:
                supercell[row] = int(abs(numbers[row]))

            row += 1
            continue

        if section in ("unit_cell", "supercell") and "symbol:" in line:
            counts[section] = counts.get(section, 0) + 1

    return {
        "supercell": tuple(supercell),
        "unitcell_natoms": counts.get("unit_cell", 0),
        "supercell_natoms": counts.get("supercell", 0),
    }


def _numbers(line: str) -> list[float]:
    """Every number on one yaml list line."""

    values: list[float] = []

    for token in line.replace("[", " ").replace("]", " ").replace(
        ",", " "
    ).split():

        try:
            values.append(float(token))

        except ValueError:
            continue

    return values

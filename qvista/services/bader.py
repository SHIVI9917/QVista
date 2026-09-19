"""
Bader (QTAIM) charge partitioning from a finished VASP run.

This is the calculation half of the Bader tool.  It reads the charge
density a run has already written and answers one question -- how many
electrons belong to each atom -- three ways: per atom, per element, and
per fragment.

Nothing here imports Qt.  Every function returns data; the dialog decides
how to show it.

What Bader analysis is
----------------------
Bader's idea is to divide space using the charge density itself rather
than using basis functions [1].  Each atom gets the volume surrounding
one maximum of the density, and the volumes are separated by *zero-flux
surfaces*: surfaces on which the density is a minimum in the direction
normal to them, so that ``grad(rho) . n = 0`` everywhere on them.  In
practice those surfaces fall in the bonding regions between atoms, and
each volume normally holds one nucleus.

Because the partition is built from the density -- an observable -- and
not from the basis set, it does not drift when the basis is changed, the
way Mulliken populations do.

How the volumes are found here
------------------------------
The density is only known on a grid, so the volumes are found by
following steepest-ascent paths from grid point to grid point [2, 3]:

``ongrid``
    From each point, step to whichever of the 26 neighbours maximises
    ``(rho(neighbour) - rho(point)) / |dr|``.  Every path ends at a
    density maximum, and every point on it belongs to that maximum's
    volume.  This is simple and fast but biased: paths are stuck to the
    26 lattice directions, so the surfaces come out faceted along the
    grid, and the bias does not go away as the grid is refined [3].

``neargrid``
    The same walk, but the step is chosen from the true density gradient
    and the difference between the true off-lattice path and the lattice
    step taken is carried along as a correction vector.  When any
    component of that vector exceeds half a grid spacing, an extra step
    is taken in that direction and the vector is reduced.  Points beside
    a surface are then re-walked until no more of them change hands.
    This removes the lattice bias [3] and is the default.

The difference is not cosmetic.  On the 42-atom run this was developed
against, ``ongrid`` gives Li +0.63 and ``neargrid`` gives Li +0.91; only
the second agrees with the published value.

Yu and Trinkle's weight method [4] is a third approach, which splits
boundary grid points fractionally between volumes instead of assigning
each one whole.  It is not implemented here.

Which density to use
--------------------
Two different densities are involved, and using one where the other
belongs is the most common way to get wrong numbers.

The *reference* draws the boundaries.  It should be the all-electron
density, ``AECCAR0`` (core) plus ``AECCAR2`` (valence), which VASP writes
when ``LAECHG = .TRUE.``.  The core peak is what puts a density maximum
on each nucleus.

The *integrand* is what gets counted inside those boundaries, and that is
``CHGCAR``, the valence pseudo-density, because the valence electrons are
the ones a charge is meant to describe.

``CHGCAR`` alone can be used as both, and is offered here for runs made
without ``LAECHG``, but its maxima are not reliably on the nuclei -- it
has no core peak -- so the boundaries it draws are worse.

A caveat worth reading
----------------------
The all-electron core density is sharp, and representing it on VASP's FFT
grid needs that grid to be fine.  When it is not, the density rings: it
goes negative between the nuclei and integrates to less than the number
of electrons in the system.  :func:`reference_quality` measures exactly
this, by integrating the reference and comparing it against the sum of
the atomic numbers, so a run whose grid is too coarse says so instead of
quietly returning plausible-looking numbers.  The remedy is to raise
``NGXF``, ``NGYF`` and ``NGZF`` and write the densities again [2].

References
----------
[1] R F W Bader, *Atoms in Molecules: A Quantum Theory*, Oxford (1990).
[2] Henkelman group, Bader Charge Analysis,
    https://theory.cm.utexas.edu/henkelman/code/bader/
[3] W Tang, E Sanville and G Henkelman, *A grid-based Bader analysis
    algorithm without lattice bias*, J. Phys.: Condens. Matter **21**,
    084204 (2009).
[4] M Yu and D R Trinkle, *Accurate and efficient algorithm for Bader
    charge integration*, J. Chem. Phys. **134**, 064111 (2011).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


# ------------------------------------------------------------
# Errors
# ------------------------------------------------------------


class BaderError(RuntimeError):
    """
    A Bader analysis could not run.

    The message is shown to the user, so it names the missing file or the
    missing INCAR tag rather than reporting a parse failure.
    """


# ------------------------------------------------------------
# Choices
# ------------------------------------------------------------

#: The two references, and what each one is.
REFERENCES: dict[str, str] = {
    "all-electron": (
        "AECCAR0 + AECCAR2 -- the all-electron density. The right choice: "
        "its core peaks put a density maximum on every nucleus. Needs "
        "LAECHG = .TRUE."
    ),
    "valence": (
        "CHGCAR alone -- the valence pseudo-density. Use only when the run "
        "has no AECCARs; it has no core peak, so its maxima are not "
        "reliably on the nuclei."
    ),
}

#: The partitioning algorithms, and what each one does.
METHODS: dict[str, str] = {
    "neargrid": (
        "Steepest ascent off the lattice, with a correction vector and "
        "edge refinement (Tang 2009). Removes the lattice bias."
    ),
    "ongrid": (
        "Steepest ascent along the 26 lattice directions (Henkelman 2006). "
        "Faster, but the surfaces are faceted along the grid."
    ),
}

#: Density below which a grid point is vacuum rather than part of an
#: atom, in electrons per cubic angstrom.  The value the Henkelman code
#: uses for ``-vac auto``.
VACUUM_TOLERANCE = 1e-3

#: How many refinement passes to allow.  Each pass re-walks the points
#: beside a surface; the count falls steeply and the charges stop moving
#: after about ten, but a handful of points can flip back and forth
#: forever, so the loop also stops when a pass makes no progress.
MAX_REFINEMENT_PASSES = 20

#: How many steps one ascent path may take before it is abandoned.
MAX_ASCENT_STEPS = 300

#: The 26 neighbours of a grid point, as index offsets.
OFFSETS: np.ndarray = np.array(
    [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)
    ],
    dtype=np.int64,
)


# ------------------------------------------------------------
# What the numbers mean
# ------------------------------------------------------------

#: Every quantity the tables show, and what it is.  Kept here rather than
#: in the dialog so the wording is testable and says the same thing
#: wherever it is shown.
DEFINITIONS: tuple[tuple[str, str], ...] = (
    (
        "Bader volume",
        "The region around one maximum of the charge density, bounded by "
        "a zero-flux surface -- a surface on which the density is a "
        "minimum in the direction normal to it. Normally one volume per "
        "nucleus.",
    ),
    (
        "Charge",
        "The number of electrons inside an atom's Bader volume. This is a "
        "population, not a charge: it counts electrons, so a more "
        "electronegative atom has a larger one. It is the CHARGE column "
        "of the reference code's ACF.dat.",
    ),
    (
        "ZVAL",
        "The number of valence electrons the POTCAR supplies for that "
        "element. Read from the POTCAR, or from the OUTCAR if there is no "
        "POTCAR in the folder.",
    ),
    (
        "Net charge",
        "ZVAL minus the charge. Positive means the atom has given "
        "electrons away, negative means it has taken them. This is the "
        "number usually quoted as 'the Bader charge'.",
    ),
    (
        "Net charge and oxidation state",
        "They track each other but are not the same. A Bader net charge "
        "is almost always well short of the formal oxidation state, and "
        "atoms in different formal states can differ by as little as "
        "0.06 electrons -- so it ranks atoms reliably and labels them "
        "only loosely.",
    ),
    (
        "Volume",
        "The volume of the Bader region, in cubic angstrom. The volumes "
        "of all atoms plus any vacuum add up to the cell.",
    ),
    (
        "Vacuum",
        "Grid points where the reference density is below "
        f"{VACUUM_TOLERANCE:g} electrons per cubic angstrom are set aside "
        "rather than given to an atom. This matters for a slab or a "
        "molecule in a box and does nothing in a dense solid.",
    ),
    (
        "Element rows",
        "One row per species, carrying the mean net charge over its "
        "atoms -- the number normally quoted for an element in a "
        "structure.",
    ),
    (
        "Fragment rows",
        "One row per group of atoms you name -- an adsorbate, a layer, a "
        "polyanion. Its net charge is the sum of ZVAL minus the sum of "
        "the charges over its atoms, which is the charge transferred into "
        "or out of the fragment as a whole.",
    ),
)

#: Where the definitions and the algorithms come from.
SOURCES: tuple[tuple[str, str], ...] = (
    (
        "Henkelman group, Bader Charge Analysis",
        "https://theory.cm.utexas.edu/henkelman/code/bader/",
    ),
    (
        "Tang, Sanville & Henkelman, J. Phys.: Condens. Matter 21, "
        "084204 (2009)",
        "https://doi.org/10.1088/0953-8984/21/8/084204",
    ),
    (
        "Yu & Trinkle, J. Chem. Phys. 134, 064111 (2011)",
        "https://doi.org/10.1063/1.3553716",
    ),
    (
        "Bader, Atoms in Molecules: A Quantum Theory, Oxford (1990)",
        "",
    ),
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------


@dataclass(frozen=True)
class AtomCharge:
    """One atom's share of the density."""

    index: int
    """Position in the structure, counting from 1 as VASP does."""

    symbol: str
    coords: np.ndarray
    charge: float
    """Electrons inside the Bader volume."""

    zval: float
    volume: float
    min_dist: float
    basins: int
    """How many density maxima were merged into this atom."""

    @property
    def population(self) -> float:
        """The charge column, named as what it is."""

        return self.charge

    @property
    def net(self) -> float:
        """ZVAL minus the population: positive when electrons were lost."""

        return self.zval - self.charge

    def label(self) -> str:

        return f"{self.index} {self.symbol}"


@dataclass(frozen=True)
class GroupCharge:
    """A set of atoms taken together -- one element, or one fragment."""

    label: str
    members: tuple[int, ...]
    charge: float
    zval: float
    volume: float

    @property
    def count(self) -> int:

        return len(self.members)

    @property
    def net(self) -> float:

        return self.zval - self.charge

    @property
    def charge_per_atom(self) -> float:

        return self.charge / self.count if self.count else 0.0

    @property
    def net_per_atom(self) -> float:

        return self.net / self.count if self.count else 0.0

    @property
    def volume_per_atom(self) -> float:

        return self.volume / self.count if self.count else 0.0


@dataclass
class BaderResult:
    """Everything one run of the analysis produced."""

    atoms: list[AtomCharge]
    structure: object
    labels: np.ndarray
    """Which atom owns each grid point; -1 for vacuum."""

    reference: str
    method: str
    grid: tuple[int, int, int]
    basins: int
    vacuum_charge: float = 0.0
    vacuum_volume: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def electrons(self) -> float:
        """Electrons accounted for, which must be NELECT."""

        return sum(atom.charge for atom in self.atoms) + self.vacuum_charge

    @property
    def volume(self) -> float:

        return sum(atom.volume for atom in self.atoms) + self.vacuum_volume

    def elements(self) -> list[str]:
        """The species, in the order they first appear."""

        seen: list[str] = []

        for atom in self.atoms:
            if atom.symbol not in seen:
                seen.append(atom.symbol)

        return seen


# ------------------------------------------------------------
# Reading a run
# ------------------------------------------------------------


def _density(folder: Path, name: str):
    """Read one VASP volumetric file, or say which one is missing."""

    path = Path(folder) / name

    if not path.is_file():
        raise BaderError(f"{name} is not in {folder}.")

    if path.stat().st_size == 0:
        raise BaderError(f"{name} is empty.")

    from pymatgen.io.vasp import Chgcar

    try:
        return Chgcar.from_file(str(path))

    except Exception as exc:  # noqa: BLE001 - re-raised with the file named
        raise BaderError(f"{name} could not be read: {exc}") from exc


def available_references(folder: Path | str) -> list[str]:
    """Which references this folder actually has the files for."""

    folder = Path(folder)

    found = []

    if (
        (folder / "AECCAR0").is_file()
        and (folder / "AECCAR2").is_file()
        and (folder / "CHGCAR").is_file()
    ):
        found.append("all-electron")

    if (folder / "CHGCAR").is_file():
        found.append("valence")

    return found


def load_densities(
    folder: Path | str,
    reference: str = "all-electron",
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    The reference and the integrand, on the same grid.

    Returns ``(reference, valence, structure)``, both arrays in VASP's own
    convention where the stored value is ``rho * V`` -- so a sum over the
    grid divided by the number of points is a number of electrons.
    """

    if reference not in REFERENCES:
        raise BaderError(f"Unknown reference {reference!r}.")

    folder = Path(folder)

    valence = _density(folder, "CHGCAR")
    structure = valence.structure

    counted = np.asarray(valence.data["total"], dtype=np.float64)

    if reference == "valence":
        return counted, counted, structure

    core = _density(folder, "AECCAR0")
    smooth = _density(folder, "AECCAR2")

    guide = np.asarray(core.data["total"], dtype=np.float64) + np.asarray(
        smooth.data["total"], dtype=np.float64
    )

    if guide.shape != counted.shape:
        raise BaderError(
            "AECCAR0/AECCAR2 and CHGCAR are on different grids "
            f"({guide.shape} against {counted.shape}). They have to come "
            "from the same run."
        )

    return guide, counted, structure


def load_zval(folder: Path | str, structure) -> dict[str, float]:
    """
    Valence electron counts per species, from POTCAR or OUTCAR.

    Without these there is no net charge, only a population, so a failure
    here says what to do about it rather than defaulting to zero.
    """

    folder = Path(folder)

    species = []

    for site in structure:
        if site.specie.symbol not in species:
            species.append(site.specie.symbol)

    potcar = folder / "POTCAR"

    if potcar.is_file() and potcar.stat().st_size:

        from pymatgen.io.vasp import Potcar

        try:
            entries = Potcar.from_file(str(potcar))

        except Exception:  # noqa: BLE001 - falls through to the OUTCAR
            entries = []

        if len(entries) == len(species):
            return {
                symbol: float(entry.ZVAL)
                for symbol, entry in zip(species, entries)
            }

        by_symbol = {
            entry.element: float(entry.ZVAL) for entry in entries
        }

        if all(symbol in by_symbol for symbol in species):
            return {symbol: by_symbol[symbol] for symbol in species}

    outcar = folder / "OUTCAR"

    if outcar.is_file():

        values = _zval_from_outcar(outcar)

        if len(values) == len(species):
            return dict(zip(species, values))

    raise BaderError(
        "No POTCAR or OUTCAR in the folder to read ZVAL from, so a net "
        "charge cannot be worked out. Copy the POTCAR in beside the "
        "densities, or the table can only show populations."
    )


def _zval_from_outcar(path: Path) -> list[float]:
    """The ZVAL line VASP prints once per species near the top."""

    values: list[float] = []

    pattern = re.compile(r"ZVAL\s*=\s*([-+0-9.Ee]+)")

    with open(path, "r", errors="ignore") as handle:

        for line in handle:

            if "POMASS" in line and "ZVAL" in line:

                match = pattern.search(line)

                if match:
                    values.append(float(match.group(1)))

            # Everything needed is in the header; the ionic loop is long.
            if "Iteration" in line:
                break

    return values


# ------------------------------------------------------------
# Grid geometry
# ------------------------------------------------------------


def step_lengths(shape: Sequence[int], lattice: np.ndarray) -> np.ndarray:
    """Cartesian length of each of the 26 grid steps."""

    fractional = OFFSETS / np.asarray(shape, dtype=float)

    return np.linalg.norm(fractional @ np.asarray(lattice, dtype=float), axis=1)


def gradient_matrix(
    shape: Sequence[int],
    lattice: np.ndarray,
) -> np.ndarray:
    """
    Turn a finite difference per grid index into a step in grid indices.

    The steepest ascent direction is the Cartesian gradient, but the walk
    happens in index space, and for anything other than a cubic cell the
    two do not point the same way.  Writing ``L`` for the lattice matrix
    and ``n`` for the grid shape, a Cartesian gradient corresponds to the
    index-space direction ``diag(n) (L L^T)^-1 diag(n) d``, where ``d``
    holds the central differences per index.  For an orthogonal cell this
    collapses to dividing each component by its grid spacing squared,
    which is what Tang 2009 equations 4 to 6 do.
    """

    counts = np.diag(np.asarray(shape, dtype=float))
    lattice = np.asarray(lattice, dtype=float)

    return counts @ np.linalg.inv(lattice @ lattice.T) @ counts


# ------------------------------------------------------------
# Partitioning
# ------------------------------------------------------------


def ascent_parents(rho: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    The steepest-ascent neighbour of every grid point, as a flat index.

    A point with no uphill neighbour is its own parent, which is what
    makes it a maximum.  All 26 neighbours are compared at once with an
    array shift each, so no Python loop ever touches a grid point.
    """

    shape = rho.shape

    flat = np.arange(rho.size, dtype=np.int64).reshape(shape)

    best = np.zeros(shape, dtype=np.float64)
    parent = flat.copy()

    for offset, length in zip(OFFSETS, step_lengths(shape, lattice)):

        shift = tuple(-int(step) for step in offset)

        score = (np.roll(rho, shift, axis=(0, 1, 2)) - rho) / length

        better = score > best

        best = np.where(better, score, best)
        parent = np.where(better, np.roll(flat, shift, axis=(0, 1, 2)), parent)

    return parent.ravel()


def resolve_maxima(parent: np.ndarray) -> np.ndarray:
    """
    Follow every parent chain to its end.

    Repeatedly replacing each entry by its parent's parent halves the
    remaining path length each time, so this finishes in a handful of
    passes however long the longest ascent path is.
    """

    while True:

        nxt = parent[parent]

        if np.array_equal(nxt, parent):
            return parent

        parent = nxt


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Points with a neighbour belonging to somebody else."""

    edge = np.zeros(labels.shape, dtype=bool)

    for offset in OFFSETS:

        shift = tuple(-int(step) for step in offset)

        edge |= np.roll(labels, shift, axis=(0, 1, 2)) != labels

    return edge


def nearest_atoms(
    maxima: np.ndarray,
    shape: Sequence[int],
    structure,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Which atom each density maximum belongs to, and how far away it is.

    A grid always finds more maxima than there are atoms -- the density
    is noisy between them, and a coarse all-electron core rings badly
    enough to create thousands.  Each one is given to the nearest
    nucleus, which is what the reference code does, and the returned
    distances let a caller notice any that are suspiciously far from one.
    """

    lattice = np.asarray(structure.lattice.matrix, dtype=float)

    grid = np.array(np.unravel_index(maxima, tuple(shape))).T
    fractional = grid / np.asarray(shape, dtype=float)

    offset = (
        fractional[:, None, :]
        - np.asarray(structure.frac_coords, dtype=float)[None, :, :]
    )
    offset -= np.round(offset)

    distance = np.linalg.norm(offset @ lattice, axis=2)

    owner = distance.argmin(axis=1)

    return owner, distance[np.arange(len(owner)), owner]


def refine_boundary(
    rho: np.ndarray,
    labels: np.ndarray,
    lattice: np.ndarray,
    is_max: np.ndarray,
    max_steps: int = MAX_ASCENT_STEPS,
) -> tuple[np.ndarray, int]:
    """
    Re-walk every point beside a surface, off the lattice this time.

    This is step (vii) of Tang 2009.  Each boundary point starts its own
    ascent, carrying a correction vector that remembers how far the real
    path has drifted from the grid points being stepped between; when a
    component of it passes half a spacing, an extra step is taken and the
    vector is reduced by it.  A path ends when it reaches a density
    maximum, or a point far enough inside one volume that all 26 of its
    neighbours agree -- and only the point it started from is reassigned.

    Every path is walked in step with all the others, one move for the
    whole set at a time, so this stays array work rather than a loop over
    several hundred thousand trajectories.
    """

    shape = rho.shape
    counts = np.asarray(shape, dtype=np.int64)

    matrix = gradient_matrix(shape, lattice)
    lengths = step_lengths(shape, lattice)

    edge = boundary_mask(labels)
    interior = ~edge

    start = np.flatnonzero(edge.ravel())

    if start.size == 0:
        return labels, 0

    position = np.array(
        np.unravel_index(start, shape), dtype=np.int64
    ).T
    drift = np.zeros((len(position), 3), dtype=np.float64)

    landed = labels.ravel()[start].copy()
    walking = np.arange(len(position))

    for _ in range(max_steps):

        here = position[walking]

        gradient = np.empty((len(here), 3), dtype=np.float64)

        for axis in range(3):

            ahead = here.copy()
            behind = here.copy()

            ahead[:, axis] += 1
            behind[:, axis] -= 1

            gradient[:, axis] = (
                rho[tuple((ahead % counts).T)]
                - rho[tuple((behind % counts).T)]
            ) / 2.0

        move = gradient @ matrix.T

        # Scaled so the largest component is one whole grid step: the
        # path advances by a point and never skips one.
        scale = np.abs(move).max(axis=1)
        flat = scale <= 0.0

        move = np.where(
            flat[:, None],
            0.0,
            move / np.where(flat, 1.0, scale)[:, None],
        )

        step = np.rint(move)

        drift[walking] += move - step

        correction = np.rint(drift[walking])

        step += correction
        drift[walking] -= correction

        # Where the density is flat the gradient says nothing, so fall
        # back to the on-grid rule rather than standing still.
        if flat.any():

            stuck = here[flat]
            level = rho[tuple((stuck % counts).T)]

            best = np.zeros(int(flat.sum()), dtype=np.float64)
            choice = np.zeros((int(flat.sum()), 3), dtype=np.float64)

            for offset, length in zip(OFFSETS, lengths):

                score = (
                    rho[tuple(((stuck + offset) % counts).T)] - level
                ) / length

                better = score > best

                best = np.where(better, score, best)
                choice[better] = offset

            step[flat] = choice
            drift[walking[flat]] = 0.0

        stalled = ~step.any(axis=1)

        position[walking] = (here + step.astype(np.int64)) % counts

        cell = tuple(position[walking].T)

        finished = is_max[cell] | interior[cell] | stalled

        if finished.any():

            landed[walking[finished]] = labels[cell][finished]
            walking = walking[~finished]

        if walking.size == 0:
            break

    moved = int((landed != labels.ravel()[start]).sum())

    refined = labels.copy()
    refined.ravel()[start] = landed

    return refined, moved


def partition(
    rho: np.ndarray,
    structure,
    method: str = "neargrid",
    progress: Callable[[str, int], None] | None = None,
) -> tuple[np.ndarray, int, np.ndarray, list[str]]:
    """
    Assign every grid point to an atom.

    Returns the label grid, how many density maxima were found, how many
    of them each atom absorbed, and any remarks worth passing on.
    """

    if method not in METHODS:
        raise BaderError(f"Unknown method {method!r}.")

    def say(message: str, percent: int) -> None:
        if progress is not None:
            progress(message, percent)

    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    shape = rho.shape

    say("Following the density uphill...", 10)

    roots = resolve_maxima(ascent_parents(rho, lattice))

    maxima, basin = np.unique(roots, return_inverse=True)

    is_max = np.zeros(rho.size, dtype=bool)
    is_max[maxima] = True
    is_max = is_max.reshape(shape)

    say(f"Assigning {len(maxima)} maxima to atoms...", 25)

    owner, distance = nearest_atoms(maxima, shape, structure)

    labels = owner[basin].reshape(shape).astype(np.int32)

    merged = np.bincount(owner, minlength=len(structure))

    notes: list[str] = []

    if len(maxima) > len(structure):
        notes.append(
            f"{len(maxima)} density maxima were found for "
            f"{len(structure)} atoms. Extra maxima are normal on a grid "
            "and each one is given to the nucleus nearest it; a large "
            "number of them means the reference density is noisy."
        )

    if len(distance) and float(distance.max()) > 1.0:
        notes.append(
            f"The furthest maximum sits {float(distance.max()):.2f} A from "
            "any nucleus. Charge that far out is being attributed to "
            "whichever atom happens to be closest."
        )

    if method == "ongrid":
        notes.append(
            "The on-grid method biases the surfaces along the grid "
            "directions, and refining the grid does not remove it. Use "
            "near-grid unless the comparison is the point."
        )
        return labels, len(maxima), merged, notes

    # Below this many reassignments a pass is rearranging a scatter of
    # points along the surfaces and no longer moves any charge in the
    # fifth decimal, so there is nothing to gain by running more.
    floor = max(8, rho.size // 20_000)

    previous = None

    for step in range(1, MAX_REFINEMENT_PASSES + 1):

        say(
            f"Refining the surfaces, pass {step}...",
            30 + int(55 * step / MAX_REFINEMENT_PASSES),
        )

        labels, moved = refine_boundary(rho, labels, lattice, is_max)

        if moved == 0:
            break

        # A pass that reassigns no fewer points than the one before has
        # stopped converging: a handful of points on a flat stretch of
        # surface will change hands forever.
        if moved < floor or (previous is not None and moved >= previous):
            notes.append(
                f"The surfaces settled after {step} refinement passes, "
                f"with {moved} grid points of "
                f"{rho.size} still changing hands. That is a rounding "
                "level disagreement and does not move the charges."
            )
            break

        previous = moved

    return labels, len(maxima), merged, notes


# ------------------------------------------------------------
# Integration
# ------------------------------------------------------------


def surface_distances(
    labels: np.ndarray,
    structure,
    count: int,
) -> np.ndarray:
    """
    Shortest distance from each nucleus to its own Bader surface.

    Only the points that actually sit on a surface are measured, and each
    is measured against its own atom, so this costs one distance per
    boundary point rather than one per atom per point.
    """

    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    shape = labels.shape

    edge = boundary_mask(labels) & (labels >= 0)

    where = np.flatnonzero(edge.ravel())

    shortest = np.full(count, np.inf)

    if where.size == 0:
        return np.full(count, np.nan)

    owner = labels.ravel()[where]

    fractional = (
        np.array(np.unravel_index(where, shape)).T
        / np.asarray(shape, dtype=float)
    )

    offset = fractional - np.asarray(structure.frac_coords, dtype=float)[owner]
    offset -= np.round(offset)

    distance = np.linalg.norm(offset @ lattice, axis=1)

    np.minimum.at(shortest, owner, distance)

    # An atom with no surface point of its own -- it was swallowed whole,
    # or the vacuum took it -- has no radius to report.
    return np.where(np.isfinite(shortest), shortest, np.nan)


def reference_quality(rho: np.ndarray, structure) -> tuple[float, float, str | None]:
    """
    Does the reference density integrate to the right number of electrons?

    The all-electron core is sharp, and an FFT grid too coarse to hold it
    rings: the density goes negative between the nuclei and integrates
    short.  Comparing the integral against the sum of atomic numbers is
    the cheapest honest check that the grid was fine enough.

    Returns ``(integrated, expected, warning)``.
    """

    from pymatgen.core.periodic_table import Element

    integrated = float(rho.sum() / rho.size)

    expected = float(
        sum(Element(site.specie.symbol).Z for site in structure)
    )

    negative = float((rho < 0).mean())

    if expected <= 0:
        return integrated, expected, None

    shortfall = abs(integrated - expected) / expected

    if shortfall < 0.005 and negative < 0.005:
        return integrated, expected, None

    return (
        integrated,
        expected,
        (
            f"The reference density integrates to {integrated:.1f} electrons "
            f"where the atoms hold {expected:.0f}, and {negative:.1%} of the "
            "grid is negative. The FFT grid is too coarse to represent the "
            "all-electron core, which rings. Raise NGXF, NGYF and NGZF and "
            "write the densities again if the boundaries need to be exact; "
            "the core error largely cancels, so the charges below are "
            "usually still usable."
        ),
    )


def integrate(
    labels: np.ndarray,
    valence: np.ndarray,
    structure,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Electrons and volume inside each atom's boundary.

    VASP stores ``rho * V`` on the grid, so a plain sum over a volume
    divided by the number of grid points is already a number of
    electrons, with no factor to remember.
    """

    count = len(structure)
    points = valence.size

    inside = labels.ravel() >= 0

    owner = labels.ravel()[inside]

    charge = (
        np.bincount(owner, weights=valence.ravel()[inside], minlength=count)
        / points
    )

    volume = (
        np.bincount(owner, minlength=count)
        * float(structure.volume)
        / points
    )

    return charge, volume


# ------------------------------------------------------------
# The whole thing
# ------------------------------------------------------------


def analyse(
    folder: Path | str,
    reference: str = "all-electron",
    method: str = "neargrid",
    vacuum: bool = False,
    progress: Callable[[str, int], None] | None = None,
) -> BaderResult:
    """
    Run a Bader analysis over a finished VASP run.

    ``folder`` holds the densities.  ``reference`` picks what draws the
    boundaries and ``method`` how they are found; see :data:`REFERENCES`
    and :data:`METHODS`.  ``vacuum`` sets aside the points below
    :data:`VACUUM_TOLERANCE` instead of giving them to an atom.
    """

    def say(message: str, percent: int) -> None:
        if progress is not None:
            progress(message, percent)

    say("Reading the densities...", 2)

    guide, counted, structure = load_densities(folder, reference)

    zval = load_zval(folder, structure)

    notes: list[str] = []

    # Only meaningful for the all-electron reference: the valence density
    # is not supposed to integrate to the atomic numbers.
    if reference == "all-electron":

        _, _, warning = reference_quality(guide, structure)

        if warning:
            notes.append(warning)

    labels, basins, merged, remarks = partition(
        guide, structure, method, progress
    )

    notes.extend(remarks)

    vacuum_charge = 0.0
    vacuum_volume = 0.0

    if vacuum:

        say("Setting the vacuum aside...", 88)

        density = guide / float(structure.volume)

        empty = density < VACUUM_TOLERANCE

        labels = np.where(empty, np.int32(-1), labels)

        vacuum_charge = float(counted[empty].sum() / counted.size)
        vacuum_volume = float(
            empty.sum() * structure.volume / counted.size
        )

        notes.append(
            f"{empty.sum()} grid points ({empty.mean():.1%} of the cell) "
            f"were below {VACUUM_TOLERANCE:g} e/A^3 and counted as vacuum."
        )

    say("Integrating...", 92)

    charge, volume = integrate(labels, counted, structure)

    say("Measuring the surfaces...", 96)

    radius = surface_distances(labels, structure, len(structure))

    atoms = [
        AtomCharge(
            index=position + 1,
            symbol=site.specie.symbol,
            coords=np.asarray(site.coords, dtype=float),
            charge=float(charge[position]),
            zval=float(zval[site.specie.symbol]),
            volume=float(volume[position]),
            min_dist=float(radius[position]),
            basins=int(merged[position]),
        )
        for position, site in enumerate(structure)
    ]

    electrons = sum(atom.charge for atom in atoms) + vacuum_charge

    notes.append(
        f"{electrons:.4f} electrons were accounted for over "
        f"{basins} density maxima on a "
        f"{guide.shape[0]}x{guide.shape[1]}x{guide.shape[2]} grid. "
        "That total is the number of valence electrons in the run, and a "
        "partition that loses any of it is broken."
    )

    say("Done.", 100)

    return BaderResult(
        atoms=atoms,
        structure=structure,
        labels=labels,
        reference=reference,
        method=method,
        grid=tuple(int(n) for n in guide.shape),
        basins=basins,
        vacuum_charge=vacuum_charge,
        vacuum_volume=vacuum_volume,
        notes=notes,
    )


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------


def by_element(result: BaderResult) -> list[GroupCharge]:
    """One row per species, in the order they appear in the structure."""

    rows: list[GroupCharge] = []

    for symbol in result.elements():

        members = [atom for atom in result.atoms if atom.symbol == symbol]

        rows.append(
            GroupCharge(
                label=symbol,
                members=tuple(atom.index for atom in members),
                charge=sum(atom.charge for atom in members),
                zval=sum(atom.zval for atom in members),
                volume=sum(atom.volume for atom in members),
            )
        )

    return rows


def by_fragment(
    result: BaderResult,
    fragments: dict[str, Sequence[int]],
) -> list[GroupCharge]:
    """
    One row per named group of atoms.

    Indices count from 1, matching the atom column of the per-atom table
    and VASP's own numbering.
    """

    known = {atom.index: atom for atom in result.atoms}

    rows: list[GroupCharge] = []

    for label, members in fragments.items():

        chosen = []

        for index in members:

            if index not in known:
                raise BaderError(
                    f"Fragment {label!r} names atom {index}, but the "
                    f"structure has {len(known)} atoms."
                )

            chosen.append(known[index])

        rows.append(
            GroupCharge(
                label=label,
                members=tuple(atom.index for atom in chosen),
                charge=sum(atom.charge for atom in chosen),
                zval=sum(atom.zval for atom in chosen),
                volume=sum(atom.volume for atom in chosen),
            )
        )

    return rows


def parse_selection(text: str, count: int) -> list[int]:
    """
    Read ``"1-6, 9, 12"`` as a list of atom indices, counting from 1.

    Written for the fragment box, where typing a range is much less
    tedious than clicking forty checkboxes.
    """

    chosen: list[int] = []

    for piece in re.split(r"[,\s]+", text.strip()):

        if not piece:
            continue

        match = re.fullmatch(r"(\d+)\s*(?:-|\.\.)\s*(\d+)", piece)

        if match:

            low, high = int(match.group(1)), int(match.group(2))

            if low > high:
                low, high = high, low

            span = range(low, high + 1)

        elif piece.isdigit():
            span = [int(piece)]

        else:
            raise BaderError(
                f"{piece!r} is not an atom number or a range like 1-6."
            )

        for index in span:

            if not 1 <= index <= count:
                raise BaderError(
                    f"Atom {index} is outside the structure, which has "
                    f"{count} atoms."
                )

            if index not in chosen:
                chosen.append(index)

    return chosen


#: The Cartesian axes a cutting plane can be normal to.
AXES: tuple[str, str, str] = ("X", "Y", "Z")


def axis_span(result: BaderResult, axis: int) -> tuple[float, float]:
    """
    How far the atoms reach along one Cartesian axis, in angstrom.

    A cutting plane is only choosable against the span it cuts: "5 A"
    means nothing until you know z runs from 0 to 14.
    """

    if not 0 <= axis <= 2:
        raise BaderError(f"{axis} is not one of the three axes.")

    values = [float(atom.coords[axis]) for atom in result.atoms]

    return min(values), max(values)


def split_on_axis(
    result: BaderResult,
    axis: int,
    cut: float,
) -> tuple[list[int], list[int]]:
    """
    Atom numbers below and above a plane normal to a Cartesian axis.

    The natural way to name a layer, a surface or a slab's two halves,
    where the fragment is a region of the cell rather than a list anyone
    would want to type.  Atoms exactly on the plane go with the upper
    half, so that the two lists always partition the structure.
    """

    if not 0 <= axis <= 2:
        raise BaderError(f"{axis} is not one of the three axes.")

    below: list[int] = []
    above: list[int] = []

    for atom in result.atoms:

        side = below if float(atom.coords[axis]) < cut else above

        side.append(atom.index)

    return below, above


def remaining(
    result: BaderResult,
    fragments: dict[str, Sequence[int]],
) -> list[int]:
    """
    The atom numbers no fragment has claimed.

    Named fragments are usually the interesting part of a structure --
    the adsorbate, the surface layer -- and what is left is the substrate
    they sit on, which is a fragment too and a tedious one to write out.
    """

    taken = {index for members in fragments.values() for index in members}

    return [
        atom.index for atom in result.atoms if atom.index not in taken
    ]


def basin_surface(
    result: BaderResult,
    atoms: Iterable[int],
) -> np.ndarray:
    """
    A field that is 1 inside the chosen atoms' volumes and 0 outside.

    Contouring it at a half gives the zero-flux surface itself, which is
    the only way to actually look at where the partition put the
    boundaries.
    """

    count = len(result.atoms)

    wanted = np.zeros(count + 1, dtype=np.float64)

    # The label grid counts atoms from zero; every index the rest of the
    # tool passes around counts from one, as the tables show them.
    for index in atoms:

        if not 1 <= int(index) <= count:
            raise BaderError(
                f"Atom {index} is outside the structure, which has "
                f"{count} atoms."
            )

        wanted[int(index) - 1] = 1.0

    # -1 is vacuum and lives in the last slot, which stays zero.
    return wanted[np.where(result.labels < 0, count, result.labels)]

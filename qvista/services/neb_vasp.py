"""
Nudged elastic band with VASP: writing the images out and reading them back.

:mod:`qvista.analyses.neb` relaxes the path here, with forces from a
machine-learning potential, and answers in minutes. This module does the
two ends of the first-principles version: it lays out the image folders
VASP expects, and it reads the barrier back out of them afterwards.

Nothing here launches VASP.

The layout VASP wants
---------------------
One numbered folder per image, ``00`` upwards, each holding a POSCAR.
The two endpoints are ``00`` and the last one, and they are *not*
relaxed by the NEB itself -- VASP only moves the intermediate images::

    <root>/
        INCAR KPOINTS POTCAR      shared by every image
        00/   POSCAR              start, fixed
        01/   POSCAR              image 1
        ...
        05/   POSCAR              image 5
        06/   POSCAR              end, fixed

``IMAGES`` in the INCAR counts the *intermediate* images only, so a
chain of five images between two endpoints is ``IMAGES = 5`` and seven
folders. Getting that off by one is the classic NEB setup mistake: VASP
either refuses to start or silently treats an endpoint as movable.

Interpolating between the endpoints
-----------------------------------
Linearly, in fractional coordinates, taking the shortest image of each
displacement across the periodic boundary. Without that minimum-image
step an atom that crosses the cell edge between the endpoints -- which
is exactly what a migrating ion does -- is interpolated the long way
round, straight through the rest of the structure. The path still looks
plausible in a viewer and the barrier is nonsense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class NebVaspError(Exception):
    """Raised when a path cannot be written or read, with the reason."""


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

#: Everything one NEB lives under, matching the phonon convention.
EXPERIMENT_DIR = "nebexp"

#: Intermediate images, not counting the two endpoints.
DEFAULT_IMAGES = 5

#: Folders are two digits: 00, 01, ... VASP requires exactly this.
FOLDER = "{:02d}"

#: Below this an interpolated displacement is treated as no movement,
#: so an atom that never moves is not reported as the migrating one.
STILL = 1e-6


def folder_name(index: int) -> str:
    """The directory one image goes in. ``index`` counts from 0."""

    return FOLDER.format(int(index))


def image_folders(root: Path | str) -> list[Path]:
    """Every numbered image folder under ``root``, in order."""

    path = Path(root)

    if not path.is_dir():
        return []

    found = [
        item
        for item in path.iterdir()
        if item.is_dir() and re.fullmatch(r"\d{2,}", item.name)
    ]

    return sorted(found, key=lambda item: int(item.name))


# ------------------------------------------------------------
# Interpolation
# ------------------------------------------------------------


def minimum_image(delta: np.ndarray) -> np.ndarray:
    """
    Reduce fractional displacements to the shorter way round the cell.

    The step that stops an ion crossing the cell boundary from being
    interpolated the long way. Results land in ``[-0.5, 0.5]``; a
    displacement of exactly half a cell is the same distance either way,
    so its sign is arbitrary and not worth defining.
    """

    return np.asarray(delta, dtype=float) - np.rint(delta)


def interpolate(start, end, images: int = DEFAULT_IMAGES) -> list:
    """
    Build the chain, endpoints included.

    Returns ``images + 2`` structures: the start, the intermediates, and
    the end. Both endpoints are returned exactly as given rather than
    re-derived, so nothing is perturbed that the user relaxed.
    """

    from pymatgen.core import Structure

    if images < 1:
        raise NebVaspError(
            "A path needs at least one intermediate image."
        )

    if len(start) != len(end):
        raise NebVaspError(
            f"The endpoints have different numbers of atoms "
            f"({len(start)} and {len(end)}). NEB interpolates between "
            f"two arrangements of the same atoms."
        )

    if [site.specie.symbol for site in start] != [
        site.specie.symbol for site in end
    ]:
        raise NebVaspError(
            "The endpoints list their atoms in a different order. VASP "
            "matches images atom by atom, so the two POSCARs must have "
            "the same species in the same order."
        )

    first = np.asarray(start.frac_coords, dtype=float)
    last = np.asarray(end.frac_coords, dtype=float)

    step = minimum_image(last - first)

    lattice = start.lattice

    chain = [start]

    for index in range(1, images + 1):

        fraction = index / (images + 1)

        chain.append(
            Structure(
                lattice,
                [site.specie for site in start],
                first + fraction * step,
                coords_are_cartesian=False,
            )
        )

    chain.append(end)

    return chain


def moving_atoms(start, end, threshold: float = 0.1) -> list[int]:
    """
    Which atoms move more than ``threshold`` angstrom between endpoints.

    Reported so the user can confirm the path is the one they meant --
    an NEB where forty atoms move is a phase transition, not an ion
    hopping between two sites.
    """

    first = np.asarray(start.frac_coords, dtype=float)
    last = np.asarray(end.frac_coords, dtype=float)

    step = minimum_image(last - first)

    cartesian = np.asarray(
        start.lattice.get_cartesian_coords(step), dtype=float
    )

    distances = np.linalg.norm(cartesian, axis=1)

    return [
        index
        for index, value in enumerate(distances)
        if value > threshold
    ]


def path_length(chain: Sequence) -> list[float]:
    """
    Cumulative distance along the chain, in angstrom.

    The reaction coordinate a barrier is plotted against. Each step is
    the norm of the whole 3N displacement between consecutive images,
    which is the usual NEB definition -- the distance travelled in
    configuration space, not by any one atom.

    Displacements use the minimum image convention, for the same reason
    the interpolation does: an ion that crosses the cell boundary would
    otherwise contribute most of a lattice vector to the path length.
    """

    coordinates = [0.0]

    for before, after in zip(chain, chain[1:]):

        step = minimum_image(
            np.asarray(after.frac_coords, dtype=float)
            - np.asarray(before.frac_coords, dtype=float)
        )

        cartesian = np.asarray(
            before.lattice.get_cartesian_coords(step), dtype=float
        )

        coordinates.append(
            coordinates[-1] + float(np.linalg.norm(cartesian))
        )

    return coordinates


# ------------------------------------------------------------
# Writing
# ------------------------------------------------------------


@dataclass
class WriteResult:
    """What was written, and where."""

    root: Path
    folders: list[Path] = field(default_factory=list)
    images: int = 0
    coordinates: list[float] = field(default_factory=list)
    moving: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_folders(self) -> int:
        """Endpoints included."""

        return len(self.folders)


def write_path(
    root: Path | str,
    start,
    end,
    images: int = DEFAULT_IMAGES,
    incar: str = "",
    kpoints: str = "",
    potcar: str = "",
) -> WriteResult:
    """
    Lay out a VASP NEB under ``root``.

    ``incar``, ``kpoints`` and ``potcar`` are written at the top level,
    shared by every image, which is where VASP looks for them. Any that
    is empty is simply not written, so a user who wants to supply their
    own is not fighting a file QVista already put there.
    """

    from pymatgen.io.vasp.inputs import Poscar

    base = Path(root)

    chain = interpolate(start, end, images)

    base.mkdir(parents=True, exist_ok=True)

    folders: list[Path] = []

    for index, structure in enumerate(chain):

        folder = base / folder_name(index)

        folder.mkdir(exist_ok=True)

        Poscar(structure).write_file(str(folder / "POSCAR"))

        folders.append(folder)

    for name, text in (
        ("INCAR", incar),
        ("KPOINTS", kpoints),
        ("POTCAR", potcar),
    ):
        if text:
            (base / name).write_text(text, encoding="utf-8")

    result = WriteResult(
        root=base,
        folders=folders,
        images=images,
        coordinates=path_length(chain),
        moving=moving_atoms(start, end),
    )

    (base / "README.txt").write_text(
        _readme(result, start), encoding="utf-8"
    )

    result.notes = _write_notes(result, start)

    return result


def incar_text(
    images: int = DEFAULT_IMAGES,
    climbing: bool = True,
    spring: float = -5.0,
    ediffg: float = -0.03,
    steps: int = 200,
) -> str:
    """
    A working NEB INCAR.

    ``IMAGES`` counts the intermediate images only. ``LCLIMB`` puts the
    highest image on the saddle point rather than leaving the barrier as
    a lower bound from whichever image landed nearest the top.
    """

    lines = [
        "# NEB, written by QVista",
        "",
        f"IMAGES = {int(images)}      # intermediate images, endpoints excluded",
        f"SPRING = {spring}     # negative: nudged elastic band",
        f"LCLIMB = {'.TRUE.' if climbing else '.FALSE.'}   "
        f"# climbing image, so the barrier is the saddle point",
        "",
        "IBRION = 3      # damped molecular dynamics, what NEB needs",
        "POTIM  = 0",
        "IOPT   = 3      # requires VTST; drop this line for plain VASP",
        f"NSW    = {int(steps)}",
        f"EDIFFG = {ediffg}   # eV/Angstrom, force convergence",
        "EDIFF  = 1E-6",
        "",
        "PREC   = Accurate",
        "ENCUT  = 520",
        "ISMEAR = 0",
        "SIGMA  = 0.05",
        "ISPIN  = 2",
        "",
        "LWAVE  = .FALSE.",
        "LCHARG = .FALSE.",
        "",
        "# One image per group of cores:",
        "#   mpirun -np <cores> vasp_std   with KPAR / NCORE set to suit",
    ]

    return "\n".join(lines) + "\n"


def _readme(result: WriteResult, start) -> str:
    """What this folder is, for whoever opens it in a month."""

    return "\n".join(
        [
            "QVista NEB",
            "==========",
            "",
            f"{result.images} intermediate images between two endpoints, "
            f"so {result.total_folders} folders.",
            "",
            f"  {folder_name(0)}/          start, held fixed",
            f"  {folder_name(1)}/ .. "
            f"{folder_name(result.images)}/   the images VASP relaxes",
            f"  {folder_name(result.total_folders - 1)}/          "
            f"end, held fixed",
            "",
            f"Path length: {result.coordinates[-1]:.3f} Angstrom",
            f"Atoms that move: "
            + (
                ", ".join(str(index + 1) for index in result.moving)
                if result.moving
                else "none above the threshold"
            ),
            "",
            "IMAGES in the INCAR counts the intermediate images only.",
            "",
            "Afterwards, point Analysis > NEB > VASP Post-Processing at this",
            "folder to read the barrier back.",
        ]
    ) + "\n"


def _write_notes(result: WriteResult, start) -> list[str]:
    """Anything worth saying about the path that was just written."""

    notes: list[str] = []

    if not result.moving:
        notes.append(
            "WARNING: no atom moves more than 0.1 Angstrom between the "
            "two endpoints. Check they are actually different "
            "structures."
        )

    elif len(result.moving) > max(4, 0.2 * len(start)):
        notes.append(
            f"WARNING: {len(result.moving)} of {len(start)} atoms move "
            f"between the endpoints. NEB assumes one process; a path "
            f"this collective usually means the endpoints were relaxed "
            f"into different structures rather than two sites of the "
            f"same one."
        )

    spacing = (
        result.coordinates[-1] / (result.total_folders - 1)
        if result.total_folders > 1
        else 0.0
    )

    if spacing > 1.0:
        notes.append(
            f"The images are {spacing:.2f} Angstrom apart. That is a "
            f"long step for NEB; more images would resolve the saddle "
            f"point better."
        )

    notes.append(
        "The endpoints must already be relaxed. NEB holds them fixed, "
        "so any force left in them is carried into every image."
    )

    return notes


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------

_ENERGY = re.compile(
    r"free  energy   TOTEN\s*=\s*([-+]?\d*\.?\d+)"
)

_OSZICAR = re.compile(
    r"E0=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)


def read_energy(folder: Path) -> float | None:
    """
    The final energy of one image.

    OUTCAR first; OSZICAR is the fallback, because a run stopped by a
    wall clock often has a complete OSZICAR and a truncated OUTCAR.
    """

    outcar = folder / "OUTCAR"

    if outcar.is_file():

        try:
            found = _ENERGY.findall(outcar.read_text(errors="replace"))

        except OSError:
            found = []

        if found:
            return float(found[-1])

    oszicar = folder / "OSZICAR"

    if oszicar.is_file():

        try:
            found = _OSZICAR.findall(oszicar.read_text(errors="replace"))

        except OSError:
            found = []

        if found:
            return float(found[-1])

    return None


#: What VTST writes for each ionic step: the spring force along the
#: tangent, then the true force along it. The second is the one that
#: matters -- it is minus the derivative of the energy with respect to
#: the reaction coordinate, which is what turns four points into a
#: curve.
_TANGENT = re.compile(
    r"projections on to tangent \(spring, REAL\)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def read_tangent_force(folder: Path) -> float | None:
    """
    The true force along the path at one image, in eV/Angstrom.

    ``None`` for an endpoint, which VASP never moves and never reports a
    projection for. Taken from the last ionic step in the OUTCAR, since
    that is the geometry the energy belongs to.
    """

    outcar = folder / "OUTCAR"

    if not outcar.is_file():
        return None

    try:
        found = _TANGENT.findall(outcar.read_text(errors="replace"))

    except OSError:
        return None

    if not found:
        return None

    try:
        return float(found[-1][1])

    except ValueError:
        return None


def neb_dat(root: Path | str) -> list[tuple[int, float, float, float]]:
    """
    The table VTST's ``nebresults.pl`` writes as ``neb.dat``.

    ``(image, distance, energy, force)`` -- the image index, the
    cumulative distance along the path in Angstrom, the energy relative
    to the first image that has one, and the force along the tangent.

    Written here rather than shelled out to the Perl scripts. They are
    not installed on most machines that run this, and everything they
    need is in the OUTCARs already: the energies, the geometries the
    distances come from, and the tangent projections.
    """

    base = Path(root)

    folders = image_folders(base)

    energies = [read_energy(folder) for folder in folders]
    structures = [read_structure(folder) for folder in folders]
    forces = [read_tangent_force(folder) for folder in folders]

    finite = [value for value in energies if value is not None]

    if not finite:
        raise NebVaspError(
            f"None of the {len(folders)} image folders has an energy in "
            f"it. Each one needs its own OUTCAR or OSZICAR."
        )

    # Referenced to the first image that has an energy, not to image 00.
    # Endpoints are relaxed in a separate run and their OUTCARs are
    # routinely not kept, which used to make every energy NaN and take
    # the whole read down with it.
    reference = finite[0]

    if all(item is not None for item in structures):
        distances = path_length(structures)

    else:
        distances = [float(index) for index in range(len(folders))]

    rows: list[tuple[int, float, float, float]] = []

    for index, folder in enumerate(folders):

        energy = energies[index]

        if energy is None:
            continue

        rows.append(
            (
                index,
                float(distances[index]),
                float(energy - reference),
                float(forces[index] or 0.0),
            )
        )

    return rows


def spline_profile(rows, points: int = 200):
    """
    The smooth curve ``nebspline.pl`` draws, as ``(distance, energy)``.

    A cubic Hermite through each interval: two energies and two slopes
    fix a cubic exactly, and the slope is the tangent force with its
    sign flipped, since force is minus the derivative of the energy.

    This is what makes a NEB profile worth splining at all. An ordinary
    spline through five energies is a guess between them; this one knows
    the gradient at every image, so the barrier it reports is the
    barrier the calculation actually found rather than the highest of
    the points that happened to be sampled.
    """

    if len(rows) < 2:
        return (
            np.asarray([row[1] for row in rows], dtype=float),
            np.asarray([row[2] for row in rows], dtype=float),
        )

    distances = np.asarray([row[1] for row in rows], dtype=float)
    energies = np.asarray([row[2] for row in rows], dtype=float)
    slopes = np.asarray([-row[3] for row in rows], dtype=float)

    grid: list[float] = []
    curve: list[float] = []

    per = max(2, points // (len(rows) - 1))

    for index in range(len(rows) - 1):

        start, end = distances[index], distances[index + 1]

        span = end - start

        if span <= 0.0:
            continue

        local = np.linspace(0.0, 1.0, per, endpoint=False)

        # Hermite basis on the unit interval.
        h00 = 2 * local**3 - 3 * local**2 + 1
        h10 = local**3 - 2 * local**2 + local
        h01 = -2 * local**3 + 3 * local**2
        h11 = local**3 - local**2

        grid.extend(start + local * span)

        curve.extend(
            h00 * energies[index]
            + h10 * span * slopes[index]
            + h01 * energies[index + 1]
            + h11 * span * slopes[index + 1]
        )

    grid.append(float(distances[-1]))
    curve.append(float(energies[-1]))

    return np.asarray(grid), np.asarray(curve)


def read_structure(folder: Path):
    """The relaxed geometry of one image, or its input if unrelaxed."""

    from pymatgen.core import Structure

    for name in ("CONTCAR", "POSCAR"):

        path = folder / name

        if path.is_file() and path.stat().st_size:

            try:
                return Structure.from_file(str(path))

            except Exception:  # pragma: no cover - depends on the file
                continue

    return None


#: Longest separation counted as a bond, in Angstrom.
#:
#: This is a bond cutoff, not a search radius, and the difference
#: matters. At 3.2 A a migrating lithium in an oxide came back
#: sixteen-coordinate: six oxygens it is genuinely bonded to, plus the
#: lithiums and cobalts of the next shell, which are neighbours in the
#: sense of being nearby and not in the sense of being bonded. The
#: coordination number that results is not one anybody would quote.
#:
#: 2.6 A keeps the first shell of a typical oxide and drops the metal
#: sublattice behind it. It is a convention rather than a fact, which is
#: why it is adjustable and why the figure and the table both say what
#: it was set to.
NEIGHBOUR_CUTOFF = 2.6

#: What the cutoff may be set to. Below ~1 A nothing is a bond; above
#: ~4 A everything is.
MIN_CUTOFF = 1.0
MAX_CUTOFF = 4.0


@dataclass
class Contact:
    """One neighbour of the migrating atom."""

    symbol: str
    index: int
    distance: float


@dataclass
class Environment:
    """What the migrating atom is surrounded by at one image."""

    label: str
    """"Initial", "Transition state" or "Final"."""

    image: int
    contacts: list

    @property
    def coordination(self) -> int:
        """How many neighbours are inside the cutoff."""

        return len(self.contacts)

    @property
    def shortest(self) -> float:

        return min(
            (contact.distance for contact in self.contacts),
            default=float("nan"),
        )

    @property
    def mean(self) -> float:

        if not self.contacts:
            return float("nan")

        return sum(
            contact.distance for contact in self.contacts
        ) / len(self.contacts)


def migrating_atom(structures) -> int | None:
    """
    The atom the path is about: the one that moves furthest.

    NEB describes one process, so there is normally one such atom and it
    moves much further than the framework relaxing around it. Returning
    the largest displacement rather than everything over a threshold
    keeps the analysis pointed at the migration itself.
    """

    usable = [item for item in structures if item is not None]

    if len(usable) < 2:
        return None

    first = np.asarray(usable[0].frac_coords, dtype=float)
    last = np.asarray(usable[-1].frac_coords, dtype=float)

    if first.shape != last.shape:
        return None

    step = minimum_image(last - first)

    cartesian = step @ np.asarray(usable[0].lattice.matrix, dtype=float)

    return int(np.argmax(np.linalg.norm(cartesian, axis=1)))


def environment_of(
    structure, index: int, cutoff: float = NEIGHBOUR_CUTOFF
) -> list:
    """
    Everything within ``cutoff`` of one atom, nearest first.

    Uses pymatgen's own neighbour search, so the periodic images are
    handled by the same code the rest of the application trusts rather
    than by a hand-rolled minimum image that would be right for a cubic
    cell and wrong for this one.
    """

    if structure is None:
        return []

    found = structure.get_neighbors(structure[index], cutoff)

    contacts = [
        Contact(
            symbol=str(neighbour.specie),
            index=int(getattr(neighbour, "index", -1)),
            distance=float(neighbour.nn_distance),
        )
        for neighbour in found
    ]

    return sorted(contacts, key=lambda contact: contact.distance)


def migration_analysis(
    root: Path | str, cutoff: float = NEIGHBOUR_CUTOFF
) -> tuple[int | None, list]:
    """
    What the migrating atom is bonded to at the start, the saddle and
    the end.

    Three images rather than all of them, because those are the three a
    barrier is explained by: what the atom sat in, what it had to pass
    through, and what it ended up in. A coordination that drops from
    four to three at the saddle and back is the barrier, written out.
    """

    base = Path(root)

    folders = image_folders(base)

    structures = [read_structure(folder) for folder in folders]

    moving = migrating_atom(structures)

    if moving is None:
        return None, []

    energies = [read_energy(folder) for folder in folders]

    values = np.asarray(
        [
            float("nan") if energy is None else energy
            for energy in energies
        ],
        dtype=float,
    )

    if np.all(np.isnan(values)):
        return moving, []

    saddle = int(np.nanargmax(values))

    wanted = [
        ("Initial", 0),
        ("Transition state", saddle),
        ("Final", len(folders) - 1),
    ]

    return moving, [
        Environment(
            label=label,
            image=index,
            contacts=environment_of(structures[index], moving, cutoff),
        )
        for label, index in wanted
        if 0 <= index < len(structures)
    ]


def tangent_forces(structures, forces) -> list[float]:
    """
    Each image's force projected onto the path, in eV/Angstrom.

    The tangent is the direction from the previous image to the next --
    the plain central estimate, which is what a relaxed band's tangent
    has converged to. Endpoints have no tangent and no force worth
    projecting, so they come back as zero.

    This is the number VASP's own NEB writes into the OUTCAR, computed
    here for the force-field route so that both produce the same
    ``neb.dat`` and both can be splined the same way.
    """

    if len(structures) < 3 or len(forces) != len(structures):
        return [0.0] * len(structures)

    projected = [0.0] * len(structures)

    for index in range(1, len(structures) - 1):

        before = np.asarray(
            structures[index - 1].cart_coords, dtype=float
        )
        after = np.asarray(structures[index + 1].cart_coords, dtype=float)

        tangent = (after - before).ravel()

        length = float(np.linalg.norm(tangent))

        if length <= 0.0:
            continue

        tangent = tangent / length

        force = np.asarray(forces[index], dtype=float).ravel()

        if force.shape != tangent.shape:
            continue

        projected[index] = float(force @ tangent)

    return projected


def enrich(result, structures, forces=None, cutoff: float = None) -> None:
    """
    Fill in everything the post-processing views read.

    Both routes end with the same question -- where is the barrier, what
    is the migrating atom doing -- so both end with the same fields, and
    one set of tables and figures serves them. The VASP route gets its
    forces from the OUTCARs; the force-field route computes them here
    from the relaxed band.
    """

    structures = [item for item in structures]

    result.structures = structures

    result.neighbour_cutoff = (
        NEIGHBOUR_CUTOFF if cutoff is None else float(cutoff)
    )

    energies = list(result.energies or [])

    if len(energies) == len(structures) and structures:

        distances = (
            path_length(structures)
            if all(item is not None for item in structures)
            else [float(index) for index in range(len(structures))]
        )

        projected = (
            tangent_forces(structures, forces)
            if forces is not None
            else [0.0] * len(structures)
        )

        result.dat = [
            (index, float(distances[index]), float(energies[index]),
             float(projected[index]))
            for index in range(len(structures))
        ]

        grid, curve = spline_profile(result.dat)

        result.spline = (grid, curve)

        if len(curve):
            result.barrier_spline = float(np.max(curve))

    moving = migrating_atom(structures)

    result.moving_atom = moving

    if moving is None or not energies:
        result.environments = []
        return

    saddle = int(np.nanargmax(np.asarray(energies, dtype=float)))

    wanted = [
        ("Initial", 0),
        ("Transition state", saddle),
        ("Final", len(structures) - 1),
    ]

    result.environments = [
        Environment(
            label=label,
            image=index,
            contacts=environment_of(
                structures[index], moving, result.neighbour_cutoff
            ),
        )
        for label, index in wanted
        if 0 <= index < len(structures)
    ]


def mep_structure(result):
    """
    The framework once, and the migrating atom at every image.

    A picture of the path rather than of a point on it: the trail of
    copies *is* the minimum energy path, and whether it runs straight
    through a bottleneck or bows around one is the thing a barrier
    number cannot tell you.

    Built from the first image, so the framework shown is the relaxed
    starting geometry rather than an average of seven that belongs to
    none of them. The copies are placed by minimum image against that
    first position, so a path crossing a cell face stays a path instead
    of jumping to the far side.

    ``None`` when there is no migration to draw.
    """

    from pymatgen.core import Structure

    structures = list(getattr(result, "structures", []) or [])

    moving = result.moving_atom

    if moving is None or not structures or structures[0] is None:
        return None

    base = structures[0]

    lattice = base.lattice

    species = [site.specie for site in base]
    coordinates = [site.frac_coords for site in base]

    start = np.asarray(base[moving].frac_coords, dtype=float)

    for structure in structures[1:]:

        if structure is None or moving >= len(structure):
            continue

        here = np.asarray(structure[moving].frac_coords, dtype=float)

        # Against the starting position, not the previous copy: errors
        # would otherwise accumulate along the chain.
        step = minimum_image(here - start)

        species.append(base[moving].specie)
        coordinates.append(start + step)

    return Structure(lattice, species, coordinates)


def recount(result, cutoff: float) -> None:
    """
    Redo the coordination at a different bond cutoff.

    In place, and without touching the disk: the structures came back
    with the read, so changing the convention is three neighbour
    searches rather than another pass over seven OUTCARs.
    """

    structures = list(getattr(result, "structures", []) or [])

    if result.moving_atom is None or not structures:
        return

    result.neighbour_cutoff = float(cutoff)

    for environment in result.environments:

        if 0 <= environment.image < len(structures):

            environment.contacts = environment_of(
                structures[environment.image],
                result.moving_atom,
                cutoff,
            )


#: What a saved run can be made of. Each is independent -- a run whose
#: trajectory failed to write should still leave its structures behind.
SAVEABLE = (
    ("contcars", "Relaxed images (CONTCAR)"),
    ("trajectory", "Trajectory (neb.traj)"),
    ("log", "Run log (neb.log)"),
    ("plots", "Profile plot (neb_profile.png)"),
)


@dataclass
class SaveResult:
    """What a save actually managed to write."""

    root: Path

    written: list[Path] = field(default_factory=list)

    #: What was asked for and did not happen, and why.
    skipped: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.written)


def save_run(
    result,
    root: Path | str,
    contcars: bool = True,
    trajectory: bool = True,
    log: bool = True,
    plots: bool = True,
    figure=None,
) -> SaveResult:
    """
    Write a finished band to disk in the layout the reader expects.

    A relaxation that took minutes should not live only in a window.
    What goes out is the numbered image folders with their relaxed
    geometry, ``neb.dat`` in the VTST format, and the profile as CSV --
    which together are enough for :func:`read_path` to reconstruct the
    result, so a saved run can be reopened through Post-Processing
    rather than run again.

    The trajectory, the log and the plot are for people rather than for
    the reader: the first to step through the path in any viewer, the
    second to show how the band converged, the third to paste into
    something.

    Each piece is written independently. A missing optional dependency
    or an unwritable file costs that piece and is reported, not the
    save.
    """

    base = Path(root)

    base.mkdir(parents=True, exist_ok=True)

    saved = SaveResult(root=base)

    structures = list(getattr(result, "structures", []) or [])

    # --------------------------------------------------------------
    # The images themselves
    # --------------------------------------------------------------

    if contcars:

        if not structures:
            saved.skipped.append(
                "No image structures to write."
            )

        for index, structure in enumerate(structures):

            if structure is None:
                continue

            folder = base / folder_name(index)

            try:
                folder.mkdir(parents=True, exist_ok=True)

                # CONTCAR is the relaxed geometry, which is what these
                # are. POSCAR beside it so the folder also opens in
                # anything that expects an input rather than an output.
                for name in ("CONTCAR", "POSCAR"):
                    structure.to(filename=str(folder / name), fmt="poscar")

                saved.written.append(folder / "CONTCAR")

            except Exception as exc:  # noqa: BLE001 - reported per image
                saved.skipped.append(
                    f"{folder_name(index)}: {type(exc).__name__}: {exc}"
                )

    # --------------------------------------------------------------
    # The profile, in the two formats that can be read back
    # --------------------------------------------------------------

    rows = list(getattr(result, "dat", []) or [])

    if rows:

        try:
            (base / "neb.dat").write_text(
                _neb_dat_text(result, rows), encoding="utf-8"
            )

            saved.written.append(base / "neb.dat")

        except OSError as exc:
            saved.skipped.append(f"neb.dat: {exc}")

    try:
        (base / "neb.csv").write_text(result.to_csv(), encoding="utf-8")

        saved.written.append(base / "neb.csv")

    except Exception as exc:  # noqa: BLE001 - one file, not the save
        saved.skipped.append(f"neb.csv: {type(exc).__name__}: {exc}")

    # --------------------------------------------------------------
    # For people
    # --------------------------------------------------------------

    if trajectory:

        images = list(getattr(result, "images", []) or [])

        if not images:
            saved.skipped.append(
                "No ASE images to write as a trajectory."
            )

        else:
            try:
                from ase.io import write as ase_write

                path = base / "neb.traj"

                ase_write(str(path), images)

                saved.written.append(path)

            except Exception as exc:  # noqa: BLE001
                saved.skipped.append(
                    f"neb.traj: {type(exc).__name__}: {exc}"
                )

    if log:

        lines = list(getattr(result, "log", []) or [])

        if not lines:
            saved.skipped.append("The run kept no log.")

        else:
            try:
                path = base / "neb.log"

                path.write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )

                saved.written.append(path)

            except OSError as exc:
                saved.skipped.append(f"neb.log: {exc}")

    if plots:

        if figure is None:
            saved.skipped.append("No figure to save.")

        else:
            try:
                path = base / "neb_profile.png"

                figure.savefig(str(path), dpi=300, bbox_inches="tight")

                saved.written.append(path)

            except Exception as exc:  # noqa: BLE001
                saved.skipped.append(
                    f"neb_profile.png: {type(exc).__name__}: {exc}"
                )

    try:
        (base / "README.txt").write_text(
            _saved_readme(result), encoding="utf-8"
        )

        saved.written.append(base / "README.txt")

    except OSError:
        pass

    return saved


def _neb_dat_text(result, rows) -> str:
    """
    ``neb.dat`` in the VTST layout: image, distance, energy, force.

    The same five columns ``nebresults.pl`` writes, so the file opens
    in whatever people already plot these with.
    """

    lines = [
        "# image   distance (A)   energy (eV)   force (eV/A)",
    ]

    for index, distance, energy, force in rows:
        lines.append(
            f"{int(index):5d} {float(distance):14.6f} "
            f"{float(energy):14.6f} {float(force):14.6f}"
        )

    return "\n".join(lines) + "\n"


def _saved_readme(result) -> str:
    """What this folder is, and what produced it."""

    forces = getattr(result, "force_field", "") or "unknown"

    lines = [
        "QVista NEB run",
        "=" * 14,
        "",
        f"Forces: {forces}",
        "",
        "  00, 01, ...    one folder per image, holding the relaxed",
        "                 geometry as CONTCAR (and the same as POSCAR)",
        "  neb.dat        image, distance, energy, force -- VTST layout",
        "  neb.csv        the energy profile",
        "  neb.traj       every image as one ASE trajectory",
        "  neb.log        how the relaxation converged, step by step",
        "  neb_profile.png the barrier plot",
        "",
        "Reopen this folder through NEB > VASP > Post-Processing to get",
        "the barrier, the coordination analysis and the structures back",
        "without relaxing it again.",
    ]

    if "not DFT" in forces:
        lines += [
            "",
            "These are machine-learning forces, not DFT. The barrier is",
            "as good as the potential, and should not be quoted as a",
            "first-principles number.",
        ]

    return "\n".join(lines) + "\n"


def read_neb_dat(root: Path | str) -> list[tuple[int, float, float, float]]:
    """
    Parse a ``neb.dat`` file, if one is there.

    Written by ``nebresults.pl``, and by :func:`save_run` for a band
    relaxed here. Empty when there is no such file -- an absent one is
    the normal case for a folder straight out of VASP, not an error.
    """

    path = Path(root) / "neb.dat"

    if not path.is_file():
        return []

    rows: list[tuple[int, float, float, float]] = []

    try:
        text = path.read_text(errors="replace")

    except OSError:
        return []

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()

        if len(parts) < 3:
            continue

        try:
            rows.append(
                (
                    int(float(parts[0])),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]) if len(parts) > 3 else 0.0,
                )
            )

        except ValueError:
            continue

    return rows


def read_path(root: Path | str):
    """
    Read a finished VASP NEB and return it as an ``NEBResult``.

    The same result type the force-field route produces, so the same
    profile figure, the same CSV and the same barrier table serve both.
    """

    from qvista.analyses.neb import NEBResult

    base = Path(root)

    if not base.is_dir():
        raise NebVaspError(f"There is no such folder: {base}")

    folders = image_folders(base)

    if len(folders) < 3:
        raise NebVaspError(
            f"{base.name} holds {len(folders)} numbered image "
            f"folder(s).\n\nA NEB needs at least three: two endpoints "
            f"and one image between them. Point this at the folder that "
            f"contains 00, 01, 02 and so on."
        )

    energies: list[float] = []
    structures: list[Any] = []
    missing: list[str] = []

    for folder in folders:

        energy = read_energy(folder)

        if energy is None:
            missing.append(folder.name)

        energies.append(float("nan") if energy is None else energy)

        structures.append(read_structure(folder))

    # A folder saved from a band relaxed here has the geometry but no
    # VASP output to read an energy from, and its profile lives in
    # neb.dat instead. Taken wholesale or not at all: those energies are
    # relative to the first image where an OUTCAR's are absolute, and a
    # folder holding some of each would mix the two scales.
    saved = read_neb_dat(base) if all(np.isnan(energies)) else []

    if saved and len(saved) == len(folders):

        energies = [float(row[2]) for row in saved]

        missing = []

    elif all(np.isnan(energies)):
        raise NebVaspError(
            f"None of the {len(folders)} image folders has an energy in "
            f"it. Each one needs its own OUTCAR or OSZICAR, or the "
            f"folder needs the neb.dat that QVista writes beside them."
        )

    values = np.asarray(energies, dtype=float)

    # The first image that has an energy, not image 00. Endpoints are
    # relaxed in a separate run and their OUTCARs are routinely not
    # kept; referencing to a missing one made every energy NaN, and the
    # whole read came down with "All-NaN slice encountered" rather than
    # saying which folder was short.
    finite = values[~np.isnan(values)]

    reference = float(finite[0])

    relative = values - reference

    usable = ~np.isnan(relative)

    saddle = int(np.nanargmax(relative))

    # The spline knows the gradient at each image, so its peak is the
    # barrier the calculation found rather than the highest image that
    # happened to be sampled.
    # Built from the folders, unless the folders had nothing to build
    # from and the file beside them is where the profile came from.
    rows_dat = saved if saved else neb_dat(base)

    spline = spline_profile(rows_dat)

    peak = float(np.max(spline[1])) if len(spline[1]) else float("nan")

    coordinates = (
        path_length([item for item in structures if item is not None])
        if all(item is not None for item in structures)
        else [float(index) for index in range(len(folders))]
    )

    result = NEBResult(
        energies=[float(value) for value in relative],
        coordinates=coordinates,
        barrier_forward=float(np.nanmax(relative)),
        barrier_spline=peak,
        barrier_reverse=float(np.nanmax(relative) - relative[-1])
        if not np.isnan(relative[-1])
        else float("nan"),
        reaction_energy=float(relative[-1]),
        saddle_index=saddle,
        n_images=len(folders) - 2,
        climbing=_climbing(base),
        converged=not missing,
        force_field="VASP nudged elastic band (first principles)",
    )

    result.dat = rows_dat
    result.spline = spline

    result.structures = structures
    result.neighbour_cutoff = NEIGHBOUR_CUTOFF

    result.moving_atom, result.environments = migration_analysis(base)

    result.rows = _rows(result, base, folders)
    result.notes = _read_notes(result, missing, folders, usable)

    return result


def _climbing(root: Path) -> bool:
    """Whether LCLIMB was on, from the INCAR if there is one."""

    incar = root / "INCAR"

    if not incar.is_file():
        return False

    try:
        text = incar.read_text(errors="replace")

    except OSError:
        return False

    text = "\n".join(
        line.split("#")[0].split("!")[0] for line in text.splitlines()
    )

    found = re.search(r"\bLCLIMB\s*=\s*\.?(TRUE|FALSE|T|F)", text, re.I)

    return bool(found and found.group(1).upper().startswith("T"))


def summary_rows(result, total: int) -> list[tuple[str, str]]:
    """
    The summary: what was run, and what came out of it.

    One barrier, not three. Forward, reverse and the spline peak were
    all reported side by side, which invites picking whichever is being
    argued for -- and on an asymmetric path the forward one is the
    barrier, so that is the number and the rest belong to the profile.

    Shared by both routes rather than written twice. A folder of VASP
    output and a band relaxed here answer the same question, so they
    get the same table, and one of them cannot quietly grow a row the
    other lacks. ``total`` counts every image including the two fixed
    endpoints, whether those came from folders or from memory.
    """

    return [
        ("Method", result.force_field),
        ("Images", str(total)),
        ("Intermediate images", str(max(total - 2, 0))),
        ("Climbing image", "on" if result.climbing else "off"),
        ("Barrier", f"{result.barrier_forward:.4f} eV"),
        ("Saddle at image", folder_name(result.saddle_index)),
    ]


def _rows(result, root: Path, folders) -> list[tuple[str, str]]:
    """The summary for a folder of VASP output."""

    return summary_rows(result, len(folders))


#: The analysis table's columns.
ANALYSIS_COLUMNS = (
    "Stage",
    "Image",
    "Coordination",
    "Atom type",
    "Bonds",
    "Bond lengths (Å)",
)


def analysis_rows(result) -> list[tuple[str, ...]]:
    """
    What the migrating atom is bonded to, split by neighbour species.

    One row per species per stage rather than one row per stage: a
    coordination number is a count of different things, and "10" says
    much less than "two oxygens at 1.78 and four cobalts at 2.55". The
    species is what makes the number mean something.
    """

    if result.moving_atom is None or not result.environments:
        return []

    rows: list[tuple[str, ...]] = []

    for environment in result.environments:

        by_species: dict[str, list[float]] = {}

        for contact in environment.contacts:
            by_species.setdefault(contact.symbol, []).append(
                contact.distance
            )

        if not by_species:

            rows.append(
                (
                    environment.label,
                    folder_name(environment.image),
                    "0",
                    "-",
                    "0",
                    "-",
                )
            )

            continue

        # Nearest species first, so the bond that defines the site is
        # the row a reader lands on.
        order = sorted(by_species, key=lambda name: min(by_species[name]))

        for position, symbol in enumerate(order):

            lengths = sorted(by_species[symbol])

            rows.append(
                (
                    environment.label if position == 0 else "",
                    folder_name(environment.image)
                    if position == 0
                    else "",
                    str(environment.coordination)
                    if position == 0
                    else "",
                    symbol,
                    str(len(lengths)),
                    ", ".join(f"{value:.3f}" for value in lengths),
                )
            )

    return rows


def _read_notes(result, missing, folders, usable) -> list[str]:
    """Diagnostics, in the order they matter."""

    notes: list[str] = []

    if missing:
        notes.append(
            f"WARNING: no energy found in image folder(s) "
            f"{', '.join(missing)}. Those points are missing from the "
            f"profile, and the barrier is whatever the remaining images "
            f"reach -- which is a lower bound, not the barrier."
        )

    if result.saddle_index in (0, len(folders) - 1):
        notes.append(
            "WARNING: the highest image is an endpoint, so this path "
            "has no barrier on it. Either the endpoints are the wrong "
            "way round, or the run has not moved yet."
        )

    if not result.climbing:
        notes.append(
            "LCLIMB was off, so no image sits exactly on the saddle "
            "point. The forward barrier below is a lower bound: the "
            "true one is at least this large."
        )

    notes.append(
        "Energies are read from each image folder as they stand. "
        "Check the run converged -- an unconverged NEB gives a smooth, "
        "plausible and wrong profile."
    )

    return notes

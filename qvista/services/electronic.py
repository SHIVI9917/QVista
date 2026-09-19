"""
Electronic structure of a finished VASP run: bands, DOS, band centres.

Everything here reads ``vasprun.xml`` and projects what is inside it.
The three things the Electrochemical Tools ask for are the same data cut
three ways:

    Bands       eigenvalues along a k-path, optionally weighted by how
                much of each state sits on chosen atoms, elements or
                orbitals.

    DOS         the same eigenvalues counted per unit energy, projected
                the same ways, and -- for the two k- and band-resolved
                entries -- restricted to chosen k-points or bands and
                broadened by hand, since there is no such thing as a
                DOSCAR for a subset of the zone.

    Band centre The first moment of a projected DOS, which is the
                d-band centre of Hammer and Norskov when the band is d.

Everything is referenced to the Fermi level, because every quantity in
this module is only meaningful relative to it.

Two things worth knowing before trusting a number out of here
-------------------------------------------------------------
A projection is not an observable.  VASP obtains it by integrating the
wavefunction against spherical harmonics inside a sphere around each
atom, and the radius of that sphere is a choice.  Projections therefore
do not sum to one, and comparing them between runs made with different
POTCARs is not meaningful.

A band centre depends on the energy window it is integrated over, and
the literature does not agree on one.  The window is therefore a
parameter here and is reported beside every number that came out of it.

References
----------
Hammer & Norskov, Nature 376, 238 (1995); Adv. Catal. 45, 71 (2000)
    the d-band model, and the first moment as its descriptor
pymatgen ``CompleteDos.get_band_center`` and friends
    the reference implementation of the moments used here
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from qvista.services.orbital import (
    GROUPS,
    ORBITALS,
    VIEW_RESOLVED,
    VIEW_SPD,
    VIEWS,
    OrbitalError,
    parse_atom_list,
    view_labels,
)


class ElectronicError(RuntimeError):
    """Raised when a run cannot be read or a request makes no sense."""


# ------------------------------------------------------------
# Choices
# ------------------------------------------------------------

#: How a band structure or a DOS can be resolved.
PROJECTIONS: dict[str, str] = {
    "total": "Everything together.",
    "atom": "One curve per atom you name.",
    "element": "One curve per species.",
    "orbital": "One curve per s, p, d shell.",
}

#: The shells, in the order they fill.
SHELLS: tuple[str, ...] = ("s", "p", "d")

#: How a band centre is resolved.
#:
#: The same three words the DOS and band windows use, and they mean the
#: same thing here: which set of projections the first moment is taken
#: over.  Orbital wise is the whole cell split by shell, which is the
#: figure to start from -- the per-element and per-atom breakdowns are
#: only worth reading once it is clear which shell carries the states.
CENTRE_MODES: dict[str, str] = {
    "atom": "Atom wise",
    "element": "Element wise",
    "orbital": "Orbital wise",
}

#: Whole-cell label used by the orbital-wise mode.
WHOLE_CELL = "all atoms"

#: How finely an orbital projection is split, shared with the Orbital
#: window so the two never drift apart in vocabulary or in ordering.
#:
#: ``VIEW_SPD`` gives three curves and answers "how much p character".
#: ``VIEW_RESOLVED`` gives nine and answers "which d orbital" -- the
#: t2g/eg split, which summing the shell erases.
ORBITAL_VIEWS = VIEWS

DEFAULT_ORBITAL_VIEW = VIEW_SPD


def orbital_series(view: str = DEFAULT_ORBITAL_VIEW) -> list[str]:
    """
    The labels an orbital-projected figure carries, under one view.

    Every one of them is accepted by :func:`orbital_columns`, so the
    weighting code does not care which view it was handed.
    """

    try:
        return list(view_labels(view))

    except OrbitalError as exc:
        raise ElectronicError(str(exc)) from exc


def _orbital_labels(
    shells: Sequence[str] | None, view: str
) -> list[str]:
    """
    The orbital series to build, honouring an explicit request.

    ``shells`` wins when the caller named one -- the shell chips in the
    dialog still narrow a resolved figure to just the five d orbitals.
    """

    if shells:

        wanted: list[str] = []

        for name in shells:

            if view == VIEW_RESOLVED and name in GROUPS:
                low, high = GROUPS[name]
                wanted.extend(ORBITALS[low:high])
            else:
                wanted.append(name)

        return wanted

    return orbital_series(view)


#: Default Gaussian width for a hand-broadened DOS, in eV.  Matches the
#: smearing a self-consistent run is usually done with.
DEFAULT_SIGMA = 0.05

#: Points on the energy axis of a hand-broadened DOS.
DEFAULT_POINTS = 1200

#: Energy window for a band centre, in eV relative to the Fermi level.
#:
#: There is no single right answer. A window that runs well below the
#: band and a little above the Fermi level is the common choice, and
#: what matters is that it is stated: the same structure gives a
#: different centre over a different window.
DEFAULT_ERANGE = (-10.0, 2.0)

#: Below this the projected weight is treated as nothing, so an empty
#: selection is reported rather than divided by.
WEIGHT_FLOOR = 1e-12

#: What the x axis of a band figure means.
#:
#: ``PATH_MODE`` is the real thing: k-points tracing a route between high
#: symmetry points, so the distance between them is meaningful and the
#: eigenvalues either side of one belong to the same branch.
#:
#: ``SCF_MODE`` is the self-consistent run's own sampling mesh plotted
#: against k-point index. Neighbouring points are unrelated -- the mesh
#: is generated in whatever order VASP happened to list it -- so the
#: eigenvalues are *states*, not a dispersion, and joining them with a
#: line would draw branches that do not exist. It is still worth
#: plotting: it shows the gap, the band edges and the spread of every
#: occupied state, which is most of what a band figure is read for, and
#: it is available from the SCF run alone with no second calculation.
PATH_MODE = "path"
SCF_MODE = "scf"

BAND_MODES: dict[str, str] = {
    PATH_MODE: "Dispersion along a k-path.",
    SCF_MODE: "Eigenvalues of the SCF mesh, against k-point index.",
}


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------


@dataclass
class Run:
    """A parsed VASP run and what it can answer."""

    vasprun: Any
    directory: Path

    fermi: float = 0.0
    spin_polarised: bool = False
    line_mode: bool = False
    """True when the k-points trace a path, so a dispersion can be drawn."""

    nkpoints: int = 0
    nbands: int = 0
    natoms: int = 0

    symbols: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def has_projections(self) -> bool:

        return bool(getattr(self.vasprun, "projected_eigenvalues", None))

    def elements(self) -> list[str]:

        return list(dict.fromkeys(self.symbols))

    def indices_of(self, element: str) -> list[int]:

        return [
            index
            for index, symbol in enumerate(self.symbols)
            if symbol == element
        ]


@dataclass
class Curve:
    """One labelled line on a DOS plot."""

    label: str
    energies: np.ndarray
    """Relative to the Fermi level, in eV."""

    values: np.ndarray
    spin: str = ""
    """``up`` or ``down`` when the run is spin polarised, else empty."""


@dataclass
class DosResult:
    """A set of DOS curves and how they were made."""

    curves: list[Curve] = field(default_factory=list)
    fermi: float = 0.0
    projection: str = "total"

    total: Curve | None = None
    """Both spin channels added together, as one trace."""

    totals: list[Curve] = field(default_factory=list)
    """The total per spin channel, which is what a figure should draw."""
    notes: list[str] = field(default_factory=list)

    #: Set when the curves were broadened here rather than read from the
    #: run's own DOS.
    sigma: float | None = None
    kpoints: tuple[int, ...] = ()
    bands: tuple[int, ...] = ()


@dataclass(frozen=True)
class BandCentre:
    """The moments of one projected band."""

    label: str
    band: str
    centre: float
    """First moment, eV relative to the Fermi level."""

    width: float
    """Second moment about the centre -- the band width."""

    skewness: float
    kurtosis: float
    filling: float
    upper_edge: float
    erange: tuple[float, float]
    count: int
    """Atoms the projection was summed over."""

    weight: float = 0.0
    """States of this shell inside the window. Zero means there are none."""

    @property
    def defined(self) -> bool:
        """
        False when the selection has no states in this shell at all.

        Asking for the d-band centre of oxygen is a reasonable thing to
        do by accident -- the menu offers one button for the whole
        structure -- and the honest answer is that there is no d band,
        not a NaN.
        """

        return self.weight > WEIGHT_FLOOR and np.isfinite(self.centre)


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


def load(
    directory: Path | str,
    projected: bool = False,
    dos: bool = False,
) -> Run:
    """
    Parse a run, saying plainly what it is and is not good for.

    ``projected`` and ``dos`` are off by default: both are slow on a
    large run, and the caller knows which it needs.
    """

    from pymatgen.io.vasp import Vasprun

    folder = Path(directory)

    path = folder / "vasprun.xml"

    if not path.is_file():
        raise ElectronicError(
            f"No vasprun.xml in {folder}. Every tool in this section "
            f"reads the run from it."
        )

    try:
        vasprun = Vasprun(
            str(path),
            parse_dos=dos,
            parse_eigen=True,
            parse_projected_eigen=projected,
            separate_spins=False,
        )

    except Exception as exc:
        raise ElectronicError(
            f"vasprun.xml could not be parsed: {exc}\n\n"
            f"A run killed by the queue leaves a file that is "
            f"well-formed up to the point it stopped."
        ) from exc

    notes: list[str] = []

    if not vasprun.converged_electronic:
        notes.append(
            "The run did not reach electronic convergence. Eigenvalues "
            "from it are not converged either, so treat everything here "
            "as provisional."
        )

    fermi = float(vasprun.efermi) if vasprun.efermi is not None else 0.0

    structure = vasprun.final_structure

    symbols = tuple(str(site.specie.symbol) for site in structure)

    kpoints = np.asarray(vasprun.actual_kpoints, dtype=float)

    line_mode = _is_line_mode(vasprun, kpoints)

    if not line_mode:
        notes.append(
            f"The {len(kpoints)} k-points are a sampling mesh, not a "
            f"path, so there is no dispersion to draw. A band structure "
            f"needs a separate non-self-consistent run over a k-path."
        )

    if projected and not getattr(vasprun, "projected_eigenvalues", None):
        notes.append(
            "The run carries no projections. Set LORBIT = 11 and rerun "
            "to resolve anything by atom, element or orbital."
        )

    return Run(
        vasprun=vasprun,
        directory=folder,
        fermi=fermi,
        spin_polarised=bool(vasprun.is_spin),
        line_mode=line_mode,
        nkpoints=len(kpoints),
        nbands=int(vasprun.parameters.get("NBANDS", 0) or 0),
        natoms=len(symbols),
        symbols=symbols,
        notes=notes,
    )


def _is_line_mode(vasprun, kpoints: np.ndarray) -> bool:
    """
    Whether the k-points trace a path rather than fill a mesh.

    Taken from what VASP recorded rather than guessed from the numbers:
    a coarse mesh and a short path can look alike.
    """

    try:
        style = str(vasprun.kpoints.style).lower()

    except Exception:  # noqa: BLE001 - fall through to the count
        style = ""

    if "line" in style:
        return True

    # A path is long and a mesh is short, in the runs this ever sees.
    return len(kpoints) >= 20 and not _looks_like_a_grid(kpoints)


def _looks_like_a_grid(kpoints: np.ndarray) -> bool:
    """True when the points sit on a regular lattice in each direction."""

    if kpoints.size == 0:
        return False

    return all(
        len(np.unique(np.round(kpoints[:, axis], 6))) ** 3 >= len(kpoints)
        for axis in range(3)
    )


# ------------------------------------------------------------
# Selecting
# ------------------------------------------------------------


def resolve_selection(
    run: Run,
    projection: str,
    atoms: str = "",
    elements: Sequence[str] | None = None,
    orbitals: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    """
    Turn what the dialog says into ``{label: atom indices}``.

    Atom indices count from zero here, as the projection array does,
    while the text the user types counts from one, as every other table
    in QVista does.
    """

    if projection == "total":
        return {"total": list(range(run.natoms))}

    if projection == "element":

        wanted = list(elements) if elements else run.elements()

        return {
            element: run.indices_of(element)
            for element in wanted
            if run.indices_of(element)
        }

    if projection == "atom":

        try:
            chosen = parse_atom_list(atoms) if atoms else []

        except OrbitalError as exc:
            raise ElectronicError(str(exc)) from exc

        if not chosen:
            raise ElectronicError(
                "Name at least one atom, as numbers or ranges such as "
                "1-6 or 3,7,11."
            )

        for index in chosen:
            if not 1 <= index <= run.natoms:
                raise ElectronicError(
                    f"Atom {index} is outside the structure, which has "
                    f"{run.natoms} atoms."
                )

        return {
            f"{index} {run.symbols[index - 1]}": [index - 1]
            for index in chosen
        }

    if projection == "orbital":
        # Every atom, split by shell instead; the shells are applied
        # downstream, so the atom set is the whole cell.
        return {"all": list(range(run.natoms))}

    raise ElectronicError(
        f"{projection!r} is not a projection. "
        f"Try one of {', '.join(PROJECTIONS)}."
    )


def orbital_columns(shell: str) -> list[int]:
    """Which columns of the projection array belong to a shell."""

    if shell == "all":
        return list(range(len(ORBITALS)))

    if shell in GROUPS:
        low, high = GROUPS[shell]
        return list(range(low, high))

    if shell in ORBITALS:
        return [ORBITALS.index(shell)]

    raise ElectronicError(
        f"{shell!r} is not an orbital. Try one of "
        f"{', '.join(SHELLS)}, or a named one such as dz2."
    )


# ------------------------------------------------------------
# Density of states, from the run's own DOS
# ------------------------------------------------------------


def _spin_label(spin) -> str:

    from pymatgen.electronic_structure.core import Spin

    return "up" if spin == Spin.up else "down"


def projected_dos(
    run: Run,
    projection: str = "element",
    atoms: str = "",
    elements: Sequence[str] | None = None,
    shells: Sequence[str] | None = None,
    orbital_view: str = DEFAULT_ORBITAL_VIEW,
) -> DosResult:
    """
    The run's own density of states, projected.

    Uses what VASP wrote rather than broadening the eigenvalues by hand:
    a self-consistent run's DOS is integrated over the whole zone with
    the tetrahedron method or the run's own smearing, and reproducing
    that from four k-points would be worse in every way.
    """

    complete = getattr(run.vasprun, "complete_dos", None)

    if complete is None:
        raise ElectronicError(
            "The run carries no density of states. Reload it with the "
            "DOS switched on, and check the run had LORBIT set."
        )

    fermi = float(complete.efermi)

    energies = np.asarray(complete.energies, dtype=float) - fermi

    curves: list[Curve] = []

    notes: list[str] = list(run.notes)

    # ----------------------------------------------------------
    # The total
    #
    # One curve per spin channel, exactly like every projected series,
    # so the minority channel mirrors below the axis when the figure is
    # drawn. Summing the two into one trace instead -- which is what
    # this used to do -- produces a total that is positive everywhere
    # and has no lower half at all, while every curve beside it does.
    # ----------------------------------------------------------

    totals = [
        Curve(
            label="total",
            energies=energies,
            values=np.asarray(values, dtype=float),
            spin=_spin_label(spin) if run.spin_polarised else "",
        )
        for spin, values in complete.densities.items()
    ]

    # The two channels added together, for anything wanting a single
    # reference trace rather than a pair.
    total = Curve(
        "total",
        energies,
        sum(
            np.asarray(values, dtype=float)
            for values in complete.densities.values()
        ),
    )

    # ----------------------------------------------------------

    if projection == "orbital":

        if orbital_view == VIEW_RESOLVED:

            curves.extend(
                _resolved_orbital_curves(complete, energies, run, shells)
            )

        else:

            wanted = list(shells) if shells else list(SHELLS)

            spd = complete.get_spd_dos()

            for shell in wanted:

                for orbital_type, dos in spd.items():

                    if str(orbital_type) != shell:
                        continue

                    curves.extend(_curves_from(dos, shell, energies, run))

    elif projection == "element":

        wanted = list(elements) if elements else run.elements()

        for element in wanted:

            dos = complete.get_element_dos().get(
                _species(complete, element)
            )

            if dos is None:
                continue

            curves.extend(_curves_from(dos, element, energies, run))

    elif projection == "atom":

        selection = resolve_selection(run, "atom", atoms=atoms)

        structure = complete.structure

        for label, indices in selection.items():

            dos = complete.get_site_dos(structure[indices[0]])

            curves.extend(_curves_from(dos, label, energies, run))

    elif projection == "total":
        curves.extend(totals)

    else:
        raise ElectronicError(
            f"{projection!r} is not a projection. "
            f"Try one of {', '.join(PROJECTIONS)}."
        )

    return DosResult(
        curves=curves,
        totals=totals,
        fermi=fermi,
        projection=projection,
        total=total,
        notes=notes,
    )


def _species(complete, element: str):
    """The Element key of a CompleteDos element map, from its symbol."""

    for candidate in complete.get_element_dos():

        if str(candidate) == element:
            return candidate

    raise ElectronicError(
        f"{element} is not in this structure. It holds "
        f"{', '.join(str(e) for e in complete.get_element_dos())}."
    )


#: How a projected figure is split up: by species, or by named atoms.
#:
#: Separate from the orbital choice on purpose. "Which element" and
#: "which shell" are independent questions, and the pairing of the two
#: is the figure most people actually want -- the d states of cobalt,
#: not cobalt's whole DOS and not every atom's d.
BASES: dict[str, str] = {
    "element": "Element wise",
    "atom": "Atom wise",
}


def projected_curves(
    run: Run,
    basis: str = "element",
    elements: Sequence[str] | None = None,
    atoms: str = "",
    shells: Sequence[str] | None = None,
    resolved: bool = False,
) -> DosResult:
    """
    A DOS projected onto a selection *and* onto orbitals at once.

    The older :func:`projected_dos` splits one way or the other:
    ``element`` gives cobalt's whole density of states, ``orbital``
    gives the d states of the entire cell. Neither answers "the d
    states of cobalt", which is the question a crystal-field or a
    band-centre argument is actually about.

    ``basis`` chooses what the selection is made of -- species, or
    atoms named by index. ``shells`` narrows it to particular shells,
    and ``resolved`` splits those into their individual orbitals, so
    the t2g and eg sets can be told apart rather than summed into one d
    curve.

    Curves come back labelled ``"Co d"`` or ``"Co d_z2"``, so a figure
    with several selections on it stays readable.
    """

    complete = getattr(run.vasprun, "complete_dos", None)

    if complete is None:
        raise ElectronicError(
            "The run carries no density of states. Reload it with the "
            "DOS switched on, and check the run had LORBIT set."
        )

    if basis not in BASES:
        raise ElectronicError(
            f"{basis!r} is not a projection basis. "
            f"Try one of {', '.join(BASES)}."
        )

    fermi = float(complete.efermi)
    energies = np.asarray(complete.energies, dtype=float) - fermi

    wanted_shells = [s for s in (shells or SHELLS) if s in SHELLS]

    if not wanted_shells:
        raise ElectronicError(
            f"No shells selected. Choose at least one of "
            f"{', '.join(SHELLS)}."
        )

    curves: list[Curve] = []
    notes: list[str] = []

    for label, sites in _selection_sites(
        complete, run, basis, elements, atoms
    ).items():

        for shell in wanted_shells:

            if resolved:
                curves.extend(
                    _resolved_for(
                        complete, sites, label, shell, energies, run
                    )
                )

            else:
                dos = _shell_for(complete, sites, shell)

                if dos is not None:
                    curves.extend(
                        _curves_from(
                            dos, f"{label} {shell}", energies, run
                        )
                    )

    if not curves:
        notes.append(
            "Nothing was projected. The run may not carry the "
            "site-resolved projections these need, which come from "
            "LORBIT = 11."
        )

    totals = [
        Curve(
            label="total",
            energies=energies,
            values=np.asarray(values, dtype=float),
            spin=_spin_label(spin) if run.spin_polarised else "",
        )
        for spin, values in complete.densities.items()
    ]

    total = Curve(
        "total",
        energies,
        sum(
            np.asarray(values, dtype=float)
            for values in complete.densities.values()
        ),
    )

    return DosResult(
        curves=curves,
        fermi=fermi,
        projection=f"{basis}+orbital",
        total=total,
        totals=totals,
        notes=notes,
    )


def _selection_sites(complete, run, basis, elements, atoms):
    """``{label: [sites]}`` for whichever basis was chosen."""

    structure = complete.structure

    if basis == "element":

        wanted = list(elements) if elements else run.elements()

        out = {}

        for element in wanted:

            sites = [
                site
                for site in structure
                if site.specie.symbol == element
            ]

            if sites:
                out[element] = sites

        return out

    # Atom wise. Indices are typed one-based, as every table in QVista
    # shows them, and used zero-based here.
    try:
        chosen = parse_atom_list(atoms) if atoms else []

    except OrbitalError as exc:
        raise ElectronicError(str(exc)) from exc

    if not chosen:
        raise ElectronicError(
            "No atoms named. Give them as indices, for example "
            "'1, 4, 7' -- counting from 1, as the structure table "
            "shows them."
        )

    out = {}

    for number in chosen:

        # parse_atom_list hands back the numbers as typed, which are
        # 1-based like every table in QVista. The structure is indexed
        # from zero, so the conversion happens here -- indexing with the
        # typed number plots the atom after the one that was asked for,
        # and says nothing about it.
        index = number - 1

        if not 0 <= index < len(structure):
            raise ElectronicError(
                f"There is no atom {number}: the structure has "
                f"{len(structure)}."
            )

        site = structure[index]

        out[f"{site.specie.symbol}{number}"] = [site]

    return out


def _shell_for(complete, sites, shell: str):
    """The summed DOS of one shell over a set of sites."""

    from pymatgen.electronic_structure.core import OrbitalType

    try:
        wanted = OrbitalType[shell]

    except KeyError:
        return None

    total = None

    for site in sites:

        spd = complete.get_site_spd_dos(site)

        dos = spd.get(wanted)

        if dos is None:
            continue

        total = dos if total is None else total + dos

    return total


def _resolved_for(complete, sites, label, shell, energies, run):
    """One curve per individual orbital within a shell."""

    from pymatgen.electronic_structure.core import Orbital

    out: list[Curve] = []

    members = [
        orbital
        for orbital in Orbital
        if orbital.name.startswith(shell) and orbital.name != shell
    ] or [orbital for orbital in Orbital if orbital.name == shell]

    for orbital in members:

        total = None

        for site in sites:

            try:
                dos = complete.get_site_orbital_dos(site, orbital)

            except Exception:  # noqa: BLE001 - orbital absent for this site
                continue

            total = dos if total is None else total + dos

        if total is not None:
            out.extend(
                _curves_from(
                    total, f"{label} {orbital.name}", energies, run
                )
            )

    return out


def _curves_from(dos, label: str, energies: np.ndarray, run: Run):
    """One curve per spin channel, or one in total."""

    out: list[Curve] = []

    for spin, values in dos.densities.items():

        out.append(
            Curve(
                label=label,
                energies=energies,
                values=np.asarray(values, dtype=float),
                spin=_spin_label(spin) if run.spin_polarised else "",
            )
        )

    return out


# ------------------------------------------------------------
# Density of states, broadened here
# ------------------------------------------------------------


def broadened_dos(
    run: Run,
    kpoints: Sequence[int] | None = None,
    bands: Sequence[int] | None = None,
    projection: str = "total",
    atoms: str = "",
    elements: Sequence[str] | None = None,
    shells: Sequence[str] | None = None,
    orbital_view: str = DEFAULT_ORBITAL_VIEW,
    sigma: float = DEFAULT_SIGMA,
    points: int = DEFAULT_POINTS,
) -> DosResult:
    """
    A density of states from chosen k-points or bands only.

    There is no such thing as a DOSCAR for part of the zone, so this
    puts a Gaussian on each eigenvalue and adds them up.  That is what
    makes "the DOS of these three bands" or "the DOS at this k-point"
    answerable at all -- and it is also why the result depends on
    ``sigma``, which the run's own DOS does not.

    Indices count from one, as the tables do.
    """

    if sigma <= 0.0:
        raise ElectronicError("The broadening width has to be positive.")

    eigenvalues = getattr(run.vasprun, "eigenvalues", None)

    if not eigenvalues:
        raise ElectronicError("The run carries no eigenvalues.")

    projections = getattr(run.vasprun, "projected_eigenvalues", None)

    if projection != "total" and not projections:
        raise ElectronicError(
            "Projecting needs LORBIT = 11 in the run, and this one has "
            "no projections in it."
        )

    chosen_k = _resolve_indices(kpoints, run.nkpoints, "k-point")
    chosen_b = _resolve_indices(bands, run.nbands, "band")

    columns = _shell_columns(shells)

    selection = (
        resolve_selection(
            run, projection, atoms=atoms, elements=elements
        )
        if projection != "orbital"
        else {"all": list(range(run.natoms))}
    )

    # ----------------------------------------------------------
    # The energy axis, from the states actually being counted
    # ----------------------------------------------------------

    sampled = []

    for spin, array in eigenvalues.items():

        values = np.asarray(array, dtype=float)

        sampled.append(values[np.ix_(chosen_k, chosen_b)][..., 0])

    stacked = np.concatenate([block.ravel() for block in sampled])

    low = float(stacked.min()) - 5.0 * sigma - run.fermi
    high = float(stacked.max()) + 5.0 * sigma - run.fermi

    grid = np.linspace(low, high, int(points))

    curves: list[Curve] = []

    notes = list(run.notes)

    for spin, array in eigenvalues.items():

        values = np.asarray(array, dtype=float)[..., 0] - run.fermi

        label_spin = _spin_label(spin) if run.spin_polarised else ""

        if projection == "orbital":

            for shell in _orbital_labels(shells, orbital_view):

                weights = _weights(
                    projections,
                    spin,
                    chosen_k,
                    chosen_b,
                    list(range(run.natoms)),
                    orbital_columns(shell),
                )

                curves.append(
                    Curve(
                        shell,
                        grid,
                        _accumulate(
                            values, chosen_k, chosen_b, weights, grid, sigma
                        ),
                        label_spin,
                    )
                )

            continue

        for label, indices in selection.items():

            weights = (
                None
                if projection == "total"
                else _weights(
                    projections, spin, chosen_k, chosen_b, indices, columns
                )
            )

            curves.append(
                Curve(
                    label,
                    grid,
                    _accumulate(
                        values, chosen_k, chosen_b, weights, grid, sigma
                    ),
                    label_spin,
                )
            )

    if len(chosen_k) < run.nkpoints or len(chosen_b) < run.nbands:
        notes.append(
            f"Built from {len(chosen_k)} of {run.nkpoints} k-points and "
            f"{len(chosen_b)} of {run.nbands} bands, broadened by "
            f"{sigma:g} eV. This is a slice of the spectrum, not the "
            f"density of states of the material."
        )

    return DosResult(
        curves=curves,
        fermi=run.fermi,
        projection=projection,
        notes=notes,
        sigma=sigma,
        kpoints=tuple(index + 1 for index in chosen_k),
        bands=tuple(index + 1 for index in chosen_b),
    )


def _resolve_indices(
    wanted: Sequence[int] | None,
    total: int,
    what: str,
) -> list[int]:
    """One-based selection to zero-based indices, or everything."""

    if not wanted:
        return list(range(total))

    out: list[int] = []

    for index in wanted:

        if not 1 <= index <= total:
            raise ElectronicError(
                f"{what.capitalize()} {index} is outside the run, which "
                f"has {total}."
            )

        if index - 1 not in out:
            out.append(index - 1)

    return out


def _resolved_orbital_curves(
    complete,
    energies: np.ndarray,
    run: Run,
    shells: Sequence[str] | None = None,
) -> list[Curve]:
    """
    One DOS curve per individual orbital, summed over every atom.

    pymatgen offers ``get_spd_dos`` for the three shells but nothing for
    the nine channels, so they are summed out of ``pdos`` directly.
    That mapping is by *position*, not by name: pymatgen calls the last
    d orbital ``dx2`` where VASP's own listing calls it ``dx2-y2``, and
    matching on the string would silently drop it.
    """

    from pymatgen.electronic_structure.dos import Dos

    wanted = _orbital_labels(shells, VIEW_RESOLVED)

    # pdos is {site: {Orbital: {Spin: densities}}}. Collect the first
    # site's orbital keys in file order, which is VASP's order.
    keys: list[Any] = []

    for orbitals in complete.pdos.values():

        keys = sorted(orbitals, key=lambda item: item.value)
        break

    if not keys:
        raise ElectronicError(
            "This run has no per-orbital DOS in it. LORBIT = 11 is "
            "needed for a resolved projection."
        )

    curves: list[Curve] = []

    for position, name in enumerate(ORBITALS):

        if name not in wanted or position >= len(keys):
            continue

        key = keys[position]

        summed: dict[Any, np.ndarray] = {}

        for orbitals in complete.pdos.values():

            for spin, values in orbitals.get(key, {}).items():

                array = np.asarray(values, dtype=float)

                summed[spin] = (
                    array if spin not in summed else summed[spin] + array
                )

        if not summed:
            continue

        curves.extend(
            _curves_from(
                Dos(complete.efermi, complete.energies, summed),
                name,
                energies,
                run,
            )
        )

    return curves


# ------------------------------------------------------------
# What each series contributes
# ------------------------------------------------------------


@dataclass(frozen=True)
class Share:
    """How much one projected series carries."""

    label: str
    spin: str

    occupied: float
    """Integral of the curve below the Fermi level."""

    whole: float
    """Integral over the entire energy range plotted."""

    percent: float
    """Share of the occupied total, across every series."""

    @property
    def empty(self) -> bool:
        """True when this series carries essentially nothing."""

        return abs(self.occupied) <= WEIGHT_FLOOR


def contributions(
    result: DosResult, upto: float = 0.0
) -> list[Share]:
    """
    Integrate each curve up to ``upto`` and express it as a share.

    The number a projected DOS is usually read for: not the shape of
    the curve but how much of the occupied manifold each series
    accounts for. Taken below the Fermi level by default, because the
    empty states above it are not what the material is made of.

    Percentages are over the occupied integrals of the curves shown, so
    they answer "of what is plotted here, how much is this" -- which is
    the only question projections can answer, since they do not sum to
    one in the first place.
    """

    rows: list[tuple[str, str, float, float]] = []

    for curve in result.curves:

        energies = np.asarray(curve.energies, dtype=float)
        values = np.asarray(curve.values, dtype=float)

        # Curves for spin down are stored mirrored below the axis on
        # some paths; integrate the magnitude so a share is positive.
        below = energies <= upto

        occupied = (
            float(abs(np.trapezoid(values[below], energies[below])))
            if below.any()
            else 0.0
        )

        whole = float(abs(np.trapezoid(values, energies)))

        rows.append((curve.label, curve.spin, occupied, whole))

    total = sum(row[2] for row in rows)

    return [
        Share(
            label=label,
            spin=spin,
            occupied=occupied,
            whole=whole,
            percent=(
                0.0 if total <= WEIGHT_FLOOR else 100.0 * occupied / total
            ),
        )
        for label, spin, occupied, whole in rows
    ]


def band_contributions(result: BandResult) -> list[Share]:
    """
    The same idea for a band figure: mean weight per series.

    A band structure carries a weight per state rather than a density,
    so "contribution" here is the average fraction of a state sitting
    on that series, over the occupied states.
    """

    rows: list[tuple[str, str, float, float]] = []

    energies = next(iter(result.energies.values()))[0]

    occupied_mask = np.asarray(energies, dtype=float) <= 0.0

    for label, weights in result.weights.items():

        array = np.asarray(weights, dtype=float)

        occupied = (
            float(array[occupied_mask].mean())
            if occupied_mask.any()
            else 0.0
        )

        rows.append((label, "", occupied, float(array.mean())))

    total = sum(row[2] for row in rows)

    return [
        Share(
            label=label,
            spin=spin,
            occupied=occupied,
            whole=whole,
            percent=(
                0.0 if total <= WEIGHT_FLOOR else 100.0 * occupied / total
            ),
        )
        for label, spin, occupied, whole in rows
    ]


def _shell_columns(shells: Sequence[str] | None) -> list[int]:

    if not shells:
        return list(range(len(ORBITALS)))

    columns: list[int] = []

    for shell in shells:
        columns.extend(orbital_columns(shell))

    return sorted(set(columns))


def _weights(
    projections,
    spin,
    chosen_k: Sequence[int],
    chosen_b: Sequence[int],
    atoms: Sequence[int],
    columns: Sequence[int],
) -> np.ndarray:
    """How much of each chosen state sits on the chosen atoms."""

    array = np.asarray(projections[spin], dtype=float)

    block = array[np.ix_(chosen_k, chosen_b, atoms, columns)]

    return block.sum(axis=(2, 3))


def _accumulate(
    values: np.ndarray,
    chosen_k: Sequence[int],
    chosen_b: Sequence[int],
    weights: np.ndarray | None,
    grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Put a Gaussian on every chosen eigenvalue and add them up."""

    energies = values[np.ix_(chosen_k, chosen_b)]

    if weights is None:
        weights = np.ones_like(energies)

    out = np.zeros_like(grid)

    scale = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

    for energy, weight in zip(energies.ravel(), np.asarray(weights).ravel()):

        if weight <= 0.0:
            continue

        out += weight * scale * np.exp(
            -0.5 * ((grid - energy) / sigma) ** 2
        )

    return out


# ------------------------------------------------------------
# Band centres
# ------------------------------------------------------------


def band_centre(
    run: Run,
    band: str = "d",
    elements: Sequence[str] | None = None,
    atoms: str = "",
    erange: tuple[float, float] | None = DEFAULT_ERANGE,
    projection: str = "",
) -> list[BandCentre]:
    """
    The moments of a projected band, one row per selection.

    The first moment is the band centre of the d-band model: the energy
    the band sits at, relative to the Fermi level, which correlates with
    how strongly the surface binds an adsorbate.  The second is its
    width, and the two together are what the model actually uses.

    ``projection`` picks what a row is, from :data:`CENTRE_MODES`:
    one per named atom, one per species, or one for the whole cell.
    Left empty it is inferred -- atoms if any were named, species
    otherwise -- which is what every caller wanted before the choice
    existed.

    Delegates the integrals to pymatgen so that the definition here is
    the same one the rest of the field's tooling uses.
    """

    from pymatgen.electronic_structure.core import OrbitalType

    complete = getattr(run.vasprun, "complete_dos", None)

    if complete is None:
        raise ElectronicError(
            "A band centre is a moment of the projected density of "
            "states, and this run carries none. Rerun with LORBIT = 11."
        )

    if band not in SHELLS:
        raise ElectronicError(
            f"{band!r} is not a shell. Try one of {', '.join(SHELLS)}."
        )

    orbital = getattr(OrbitalType, band)

    if not projection:
        projection = "atom" if atoms else "element"

    if projection not in CENTRE_MODES:
        raise ElectronicError(
            f"{projection!r} is not a way to resolve a band centre. "
            f"Try one of {', '.join(CENTRE_MODES)}."
        )

    rows: list[BandCentre] = []

    structure = complete.structure

    # (label, species, sites, atom count). The count is carried rather
    # than worked out per row, because "all atoms" is not an element and
    # looking it up as one would report zero.
    wanted: list[tuple[str, list | None, list | None, int]]

    if projection == "orbital":

        # Neither species nor sites: pymatgen then takes the moment over
        # the cell's own s/p/d decomposition, which is the whole
        # structure split by shell.
        wanted = [(WHOLE_CELL, None, None, run.natoms)]

    elif projection == "atom":

        selection = resolve_selection(run, "atom", atoms=atoms)

        wanted = [
            (
                label,
                None,
                [structure[index] for index in indices],
                len(indices),
            )
            for label, indices in selection.items()
        ]

    else:

        chosen = list(elements) if elements else run.elements()

        wanted = [
            (
                element,
                [_species(complete, element)],
                None,
                len(run.indices_of(element)),
            )
            for element in chosen
        ]

    for label, species, sites, count in wanted:

        # How much of this shell the selection has at all. pymatgen
        # divides by this to take a moment, so without the check an
        # element with no d states comes back as a row of NaN.
        weight = _shell_weight(complete, orbital, species, sites, erange)

        if weight <= WEIGHT_FLOOR:

            rows.append(
                BandCentre(
                    label=label,
                    band=band,
                    centre=float("nan"),
                    width=float("nan"),
                    skewness=float("nan"),
                    kurtosis=float("nan"),
                    filling=float("nan"),
                    upper_edge=float("nan"),
                    erange=erange or (float("-inf"), float("inf")),
                    count=count,
                    weight=0.0,
                )
            )

            continue

        try:
            centre = complete.get_band_center(
                band=orbital, elements=species, sites=sites, erange=erange
            )
            width = complete.get_band_width(
                band=orbital, elements=species, sites=sites, erange=erange
            )
            skew = complete.get_n_moment(
                3, band=orbital, elements=species, sites=sites,
                erange=erange,
            )
            kurtosis = complete.get_n_moment(
                4, band=orbital, elements=species, sites=sites,
                erange=erange,
            )
            filling = complete.get_band_filling(
                band=orbital, elements=species, sites=sites
            )
            edge = complete.get_upper_band_edge(
                band=orbital, elements=species, sites=sites, erange=erange
            )

        except Exception as exc:  # noqa: BLE001 - reported per row
            raise ElectronicError(
                f"Could not take the moments for {label}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        rows.append(
            BandCentre(
                label=label,
                band=band,
                centre=float(centre),
                width=float(width),
                skewness=float(skew),
                kurtosis=float(kurtosis),
                filling=float(filling),
                upper_edge=float(edge),
                erange=erange or (float("-inf"), float("inf")),
                count=count,
                weight=weight,
            )
        )

    return rows


#: What an atom selection can be, in the band-centre table.
ATOMS_ALL = "all"
ATOMS_INDICES = "indices"

ATOM_CHOICES: dict[str, str] = {
    ATOMS_ALL: "All atoms",
    ATOMS_INDICES: "By indices",
}

#: Stands for "every element", so a row can ask for a shell over the
#: whole cell without naming a species.
ANY_ELEMENT = "All"


def parse_indices(text: str) -> list[int]:
    """
    ``1 2 5 8-12 20`` as zero-based indices.

    Separated by spaces, commas or both -- the reference workflow shows
    spaces in its placeholder and people type commas out of habit, and
    refusing either would be a puzzle rather than a rule.

    The text counts from one, as VASP does and as every table here does;
    what comes back counts from zero, as the projection arrays do. That
    conversion is the whole reason this is a function rather than a
    call to :func:`~qvista.services.orbital.parse_atom_list`: forgetting
    it silently reads the neighbouring atom, which is the kind of wrong
    answer that looks perfectly reasonable.
    """

    found: list[int] = []

    for part in text.replace(",", " ").split():

        if "-" in part[1:]:

            head, _, tail = part.partition("-")

            try:
                first, last = int(head), int(tail)

            except ValueError:
                raise ElectronicError(
                    f"{part!r} is not a range of atoms."
                ) from None

            if last < first:
                raise ElectronicError(f"{part!r} runs backwards.")

            found.extend(range(first, last + 1))

            continue

        try:
            found.append(int(part))

        except ValueError:
            raise ElectronicError(
                f"{part!r} is not an atom number."
            ) from None

    return sorted({number - 1 for number in found})


def orbital_of(contribution: str) -> str:
    """
    The shell or orbital a contribution name refers to.

    The inverse of :func:`contribution_label`: "d-band" is the d shell,
    and anything else is already an orbital name.
    """

    return contribution[:-5] if contribution.endswith("-band") else contribution


def contribution_label(orbital: str) -> str:
    """
    How one contribution is named in the table and the key.

    A shell is written "d-band", because that is the quantity the
    d-band model is about and the name it is published under. A resolved
    orbital is written plainly -- "dz2" -- since "dz2-band" would claim
    a band where there is one component of one.
    """

    return f"{orbital}-band" if orbital in SHELLS else orbital


def orbital_choices(resolved: bool = False) -> tuple[str, ...]:
    """
    The orbitals a band-centre row may ask for.

    Shells by default -- s, p and d are what the d-band model is written
    in. Resolved gives the nine individual orbitals; a p_z centre is a
    different number from a p one, and on a surface it is often the one
    that matters.
    """

    return ORBITALS if resolved else SHELLS


@dataclass(frozen=True)
class CentreRequest:
    """One row of the band-centre table: what to take the moment of."""

    #: :data:`ATOMS_ALL` or :data:`ATOMS_INDICES`.
    atoms: str = ATOMS_ALL

    #: One-based indices, as typed. Only read for :data:`ATOMS_INDICES`.
    indices: str = ""

    #: An element symbol, or :data:`ANY_ELEMENT`.
    element: str = ANY_ELEMENT

    #: A shell or a resolved orbital.
    orbital: str = "d"

    def label(self) -> str:
        """How this row is named in the table and on the figure."""

        where = self.element if self.element != ANY_ELEMENT else "all"

        if self.atoms == ATOMS_INDICES and self.indices.strip():
            where = f"{where} [{self.indices.strip()}]"

        return f"{where} {contribution_label(self.orbital)}"

    @property
    def contribution(self) -> str:
        """The shell or orbital, named as the table reports it."""

        return contribution_label(self.orbital)


def centre_of(
    run: Run,
    request: CentreRequest,
    erange: tuple[float, float] | None = DEFAULT_ERANGE,
) -> BandCentre:
    """
    The moments of exactly one selection.

    Written alongside :func:`band_centre` rather than replacing it. That
    one answers "every shell of every species", which is a sweep; this
    answers "this element, these atoms, this orbital", which is one row
    of a table somebody built by hand.

    The integral is done here for shells and resolved orbitals alike, so
    that a p and a p_z centre are the same quantity computed the same
    way. Deferring the shell case to pymatgen and doing the resolved one
    by hand would put two definitions in one table, and the difference
    between them would read as physics.
    """

    complete = getattr(run.vasprun, "complete_dos", None)

    if complete is None:
        raise ElectronicError(
            "A band centre is a moment of the projected density of "
            "states, and this run carries none. Rerun with LORBIT = 11."
        )

    sites, count = _requested_sites(run, complete, request)

    energies, density = _requested_density(
        complete, sites, request.orbital
    )

    return _moments(
        label=request.label(),
        band=request.orbital,
        energies=energies,
        density=density,
        erange=erange,
        count=count,
    )


@dataclass
class CentreCurve:
    """
    One contribution: its density, and the centre taken from it.

    The density travels with the number because the figure draws both.
    A band centre on its own is a horizontal line with nothing to say
    where it came from; drawn against the states it is the first moment
    of, it is obviously the balance point of that curve.
    """

    element: str
    contribution: str

    atoms: int
    """How many atoms the projection was summed over."""

    energies: object
    """Fermi-referenced, inside the window."""

    dos: object
    """Projected density on the same grid."""

    centre: float
    width: float

    @property
    def defined(self) -> bool:
        """False when the selection has no states of this kind at all."""

        import math

        return not math.isnan(self.centre)


def centre_curves(
    run: Run,
    requests,
    erange: tuple[float, float] | None = DEFAULT_ERANGE,
) -> list[CentreCurve]:
    """
    A curve per request, ready to draw and to tabulate.

    The window is applied here rather than in the figure, so that the
    curve drawn and the number reported come from exactly the same
    points. Cropping afterwards would leave a centre computed over one
    range beside a curve showing another.
    """

    complete = getattr(run.vasprun, "complete_dos", None)

    if complete is None:
        raise ElectronicError(
            "A band centre is a moment of the projected density of "
            "states, and this run carries none. Rerun with LORBIT = 11."
        )

    curves: list[CentreCurve] = []

    for request in requests:

        sites, count = _requested_sites(run, complete, request)

        energies, density = _requested_density(
            complete, sites, request.orbital
        )

        inside = np.ones_like(energies, dtype=bool)

        if erange is not None and energies.size:
            inside = (energies >= erange[0]) & (energies <= erange[1])

        grid = energies[inside]
        values = density[inside]

        moments = _moments(
            label=request.label(),
            band=request.orbital,
            energies=energies,
            density=density,
            erange=erange,
            count=count,
        )

        curves.append(
            CentreCurve(
                element=(
                    request.element
                    if request.element != ANY_ELEMENT
                    else "all"
                ),
                contribution=contribution_label(request.orbital),
                atoms=count,
                energies=grid,
                dos=values,
                centre=moments.centre,
                width=moments.width,
            )
        )

    return curves


def _requested_sites(run: Run, complete, request: CentreRequest):
    """The sites a request covers, and how many atoms that is."""

    structure = complete.structure

    indices = list(range(run.natoms))

    if request.atoms == ATOMS_INDICES:

        chosen = parse_indices(request.indices) if request.indices else []

        if not chosen:
            raise ElectronicError(
                "This row asks for atoms by index but names none. Type "
                "something like 1-6, 9."
            )

        outside = [n for n in chosen if n < 0 or n >= run.natoms]

        if outside:
            raise ElectronicError(
                f"This cell has {run.natoms} atoms, so "
                f"{', '.join(str(n + 1) for n in outside)} is out of "
                f"range."
            )

        indices = chosen

    if request.element != ANY_ELEMENT:

        wanted = set(run.indices_of(request.element))

        indices = [index for index in indices if index in wanted]

        if not indices:
            raise ElectronicError(
                f"No {request.element} atoms in that selection."
            )

    return [structure[index] for index in indices], len(indices)


def _requested_density(complete, sites, orbital: str):
    """
    Energies relative to the Fermi level, and the summed density.

    Both spin channels are added: a band centre is a property of the
    states, not of one channel, and the majority channel alone would be
    a different quantity wearing the same name.
    """

    from pymatgen.electronic_structure.core import Orbital, OrbitalType

    if orbital in SHELLS:

        wanted = getattr(OrbitalType, orbital)

        parts = [
            complete.get_site_spd_dos(site).get(wanted) for site in sites
        ]

    else:

        wanted = getattr(Orbital, _pymatgen_orbital(orbital))

        parts = [
            complete.get_site_orbital_dos(site, wanted) for site in sites
        ]

    parts = [dos for dos in parts if dos is not None]

    if not parts:
        return np.zeros(0), np.zeros(0)

    energies = np.asarray(parts[0].energies, dtype=float) - float(
        complete.efermi
    )

    total = np.zeros_like(energies)

    for dos in parts:
        for values in dos.densities.values():
            total = total + np.asarray(values, dtype=float)

    return energies, total


#: QVista writes the orbitals the way VASP does; pymatgen spells one of
#: them differently.
_ORBITAL_SPELLING = {"dx2-y2": "dx2"}


def _pymatgen_orbital(name: str) -> str:

    return _ORBITAL_SPELLING.get(name, name)


def _moments(
    label: str,
    band: str,
    energies,
    density,
    erange,
    count: int,
) -> BandCentre:
    """
    First and second moments of a density over the window.

    The window is part of the answer: the same states give a different
    centre over a different range, which is why it is carried on the
    result rather than assumed.
    """

    empty = BandCentre(
        label=label,
        band=band,
        centre=float("nan"),
        width=float("nan"),
        skewness=float("nan"),
        kurtosis=float("nan"),
        filling=float("nan"),
        upper_edge=float("nan"),
        erange=erange or (float("-inf"), float("inf")),
        count=count,
        weight=0.0,
    )

    if energies.size == 0:
        return empty

    inside = np.ones_like(energies, dtype=bool)

    if erange is not None:
        inside = (energies >= erange[0]) & (energies <= erange[1])

    grid = energies[inside]
    values = density[inside]

    if grid.size < 2:
        return empty

    area = _integrate(values, grid)

    if area <= WEIGHT_FLOOR:
        return empty

    centre = _integrate(values * grid, grid) / area

    variance = _integrate(values * (grid - centre) ** 2, grid) / area

    return BandCentre(
        label=label,
        band=band,
        centre=float(centre),
        width=float(np.sqrt(max(variance, 0.0))),
        skewness=float("nan"),
        kurtosis=float("nan"),
        filling=float("nan"),
        upper_edge=float("nan"),
        erange=erange or (float("-inf"), float("inf")),
        count=count,
        weight=float(area),
    )


def _integrate(values, grid) -> float:
    """
    Trapezoidal integral, under whichever name NumPy is using.

    ``trapz`` was removed in NumPy 2 and ``trapezoid`` does not exist
    before it.
    """

    rule = getattr(np, "trapezoid", None) or np.trapz

    return float(rule(values, grid))


def _shell_weight(complete, orbital, species, sites, erange) -> float:
    """Total projected states of one shell inside the window."""

    from pymatgen.electronic_structure.dos import add_densities

    if sites is not None:
        densities = [
            complete.get_site_spd_dos(site).get(orbital) for site in sites
        ]

    elif species is not None:
        densities = [
            complete.get_element_spd_dos(element).get(orbital)
            for element in species
        ]

    else:
        densities = [complete.get_spd_dos().get(orbital)]

    densities = [dos for dos in densities if dos is not None]

    if not densities:
        return 0.0

    energies = np.asarray(densities[0].energies, dtype=float) - float(
        complete.efermi
    )

    total = np.zeros_like(energies)

    for dos in densities:
        for values in dos.densities.values():
            total = total + np.asarray(values, dtype=float)

    if erange is not None:
        inside = (energies >= erange[0]) & (energies <= erange[1])
    else:
        inside = np.ones_like(energies, dtype=bool)

    if not inside.any():
        return 0.0

    return float(np.trapezoid(total[inside], energies[inside]))


# ------------------------------------------------------------
# Bands
# ------------------------------------------------------------


@dataclass
class BandResult:
    """A dispersion, and optionally the weight of each state."""

    distances: list[np.ndarray] = field(default_factory=list)
    energies: dict[str, list[np.ndarray]] = field(default_factory=dict)
    """``spin -> [(bands, kpoints) per segment]``, relative to E_F."""

    labels: list[str] = field(default_factory=list)
    ticks: list[float] = field(default_factory=list)

    weights: dict[str, np.ndarray] = field(default_factory=dict)
    """``label -> (bands, kpoints)`` in [0, 1], when projected."""

    fermi: float = 0.0
    gap: float = 0.0
    is_metal: bool = False
    projection: str = "total"
    notes: list[str] = field(default_factory=list)

    mode: str = PATH_MODE
    """:data:`PATH_MODE` or :data:`SCF_MODE` -- what the x axis means."""

    @property
    def is_path(self) -> bool:
        """True when the x axis is a distance along a k-path."""

        return self.mode == PATH_MODE

    @property
    def xlabel(self) -> str:
        """What to write under the x axis, which differs by mode."""

        return (
            "Wave vector"
            if self.is_path
            else "k-point index (sampling mesh, not a path)"
        )


def band_structure(
    run: Run,
    projection: str = "total",
    atoms: str = "",
    elements: Sequence[str] | None = None,
    shells: Sequence[str] | None = None,
    orbital_view: str = DEFAULT_ORBITAL_VIEW,
) -> BandResult:
    """
    Eigenvalues along the k-path, with optional projected weights.

    The weights are normalised per state, so a value is the fraction of
    that state sitting on the selection -- which is what makes a fat
    band readable. Raw projections are not normalised and would make
    the marker size depend on the POTCAR as much as on the physics.
    """

    if not run.line_mode:
        raise ElectronicError(
            f"These {run.nkpoints} k-points are a sampling mesh, not a "
            f"path.\n\nA band structure needs a non-self-consistent run "
            f"over a k-path: write the path with Crystallography > High "
            f"Symmetry Pathway, run VASP with ICHARG = 11, and point "
            f"this at that folder."
        )

    from pymatgen.electronic_structure.core import Spin

    bandstructure = run.vasprun.get_band_structure(
        line_mode=True, efermi=_fermi_for(run)
    )

    fermi = float(bandstructure.efermi)

    notes = list(run.notes)

    energies: dict[str, list[np.ndarray]] = {}

    for spin in bandstructure.bands:

        label = "up" if spin == Spin.up else "down"

        energies[label] = [
            np.asarray(bandstructure.bands[spin], dtype=float) - fermi
        ]

    distances = [np.asarray(bandstructure.distance, dtype=float)]

    ticks: list[float] = []
    labels: list[str] = []

    for index, kpoint in enumerate(bandstructure.kpoints):

        if kpoint.label:
            ticks.append(float(bandstructure.distance[index]))
            labels.append(str(kpoint.label))

    gap = bandstructure.get_band_gap()

    weights: dict[str, np.ndarray] = {}

    if projection != "total":

        weights = _band_weights(
            run, projection, atoms, elements, shells, orbital_view
        )

    return BandResult(
        distances=distances,
        energies=energies,
        labels=labels,
        ticks=ticks,
        weights=weights,
        fermi=fermi,
        gap=float(gap.get("energy", 0.0) or 0.0),
        is_metal=bool(bandstructure.is_metal()),
        projection=projection,
        notes=notes,
        mode=PATH_MODE,
    )


# ------------------------------------------------------------
# The same eigenvalues, from the self-consistent run
# ------------------------------------------------------------


def scf_bands(
    run: Run,
    projection: str = "total",
    atoms: str = "",
    elements: Sequence[str] | None = None,
    shells: Sequence[str] | None = None,
    orbital_view: str = DEFAULT_ORBITAL_VIEW,
) -> BandResult:
    """
    Eigenvalues of the self-consistent run against k-point index.

    What this is for
    ----------------
    A proper band structure needs a second, non-self-consistent run over
    a k-path, and most people have not done one yet -- the SCF run they
    already have samples a mesh. Refusing to draw anything until they go
    away and run VASP again throws away a figure that answers most of the
    same questions: where the band edges are, how wide the gap is, how
    the occupied states are distributed, and which atoms they sit on.

    What it is not
    --------------
    A dispersion. The mesh points are in whatever order VASP listed them,
    so the x axis carries no geometry and adjacent points are not
    neighbours in reciprocal space. :data:`SCF_MODE` is set on the
    result so the figure draws these as points rather than joining them
    into branches that do not exist.
    """

    from pymatgen.electronic_structure.core import Spin

    eigenvalues = getattr(run.vasprun, "eigenvalues", None)

    if not eigenvalues:
        raise ElectronicError(
            "This run has no eigenvalues in its vasprun.xml, so nothing "
            "can be plotted. A run written with NWRITE = 0 omits them."
        )

    fermi = float(run.fermi)

    energies: dict[str, list[np.ndarray]] = {}

    for spin, block in eigenvalues.items():

        label = "up" if spin == Spin.up else "down"

        # (kpoints, bands, 2) out of the file, energy first and occupancy
        # second. The plot wants bands along the rows.
        array = np.asarray(block, dtype=float)[:, :, 0].T

        energies[label] = [array - fermi]

    # One unit per k-point, counting from one to match every k-point
    # table QVista shows.
    count = next(iter(energies.values()))[0].shape[1]

    distances = [np.arange(1, count + 1, dtype=float)]

    gap, is_metal = _scf_gap(run)

    notes = [note for note in run.notes if "sampling mesh" not in note]

    notes.append(
        f"These are the {count} k-points of the self-consistent run "
        f"plotted against their index, not a dispersion. The horizontal "
        f"axis has no geometric meaning, so the points are not joined."
    )

    weights: dict[str, np.ndarray] = {}

    if projection != "total":
        weights = _band_weights(
            run, projection, atoms, elements, shells, orbital_view
        )

    return BandResult(
        distances=distances,
        energies=energies,
        labels=[],
        ticks=[],
        weights=weights,
        fermi=fermi,
        gap=gap,
        is_metal=is_metal,
        projection=projection,
        notes=notes,
        mode=SCF_MODE,
    )


def _scf_gap(run: Run) -> tuple[float, bool]:
    """
    The band gap from the mesh, and whether the run came out metallic.

    Delegated to pymatgen, which finds the valence and conduction edges
    from the occupancies rather than from a Fermi level that VASP places
    inside the gap at an arbitrary height.
    """

    try:
        gap, _cbm, _vbm, _direct = run.vasprun.eigenvalue_band_properties

        return float(gap), bool(gap <= 0.0)

    except Exception:  # pragma: no cover - depends on the run
        return 0.0, False


#: What VASPKIT calls the file it writes the Fermi level into.
FERMI_FILE = "FERMI_ENERGY"


def _fermi_for(run: Run) -> float | None:
    """
    The Fermi level for a band run, from wherever it survives.

    A non-self-consistent band run (``ICHARG = 11``) does not always
    write one into vasprun.xml -- it read the density rather than
    converging it, so there was no Fermi level to report. pymatgen then
    refuses the whole band structure with "e_fermi is None", which
    reads as the run being unusable when only one number is missing.

    VASPKIT writes it beside the run in FERMI_ENERGY, and that is the
    one the accompanying self-consistent run found, so it is the right
    reference. Falling back to it turns a refusal into a figure.
    """

    fermi = getattr(run.vasprun, "efermi", None)

    if fermi is not None:
        return float(fermi)

    directory = getattr(run, "directory", None)

    if directory is not None:

        path = Path(directory) / FERMI_FILE

        if path.is_file():

            try:
                for line in path.read_text(errors="replace").splitlines():

                    stripped = line.strip()

                    if not stripped or stripped.startswith("#"):
                        continue

                    return float(stripped.split()[0])

            except (OSError, ValueError, IndexError):
                pass

    # Nothing found. pymatgen's own "smart" guess reads it off the
    # eigenvalues, which is better than refusing outright.
    return "smart"


def _band_weights(
    run: Run,
    projection: str,
    atoms: str,
    elements: Sequence[str] | None,
    shells: Sequence[str] | None,
    orbital_view: str = DEFAULT_ORBITAL_VIEW,
) -> dict[str, np.ndarray]:
    """Fraction of each state on each selection, in [0, 1]."""

    from pymatgen.electronic_structure.core import Spin

    projections = getattr(run.vasprun, "projected_eigenvalues", None)

    if not projections:
        raise ElectronicError(
            "Projecting a band structure needs LORBIT = 11 in the run."
        )

    array = np.asarray(projections[Spin.up], dtype=float)

    columns = _shell_columns(shells)

    selection = (
        resolve_selection(run, projection, atoms=atoms, elements=elements)
        if projection != "orbital"
        else {
            name: list(range(run.natoms))
            for name in _orbital_labels(shells, orbital_view)
        }
    )

    # Which orbital groupings each selection is split into. One entry
    # normally -- the chosen shells summed together -- but the resolved
    # view splits them, so a selection becomes one series per orbital
    # and the t2g and eg sets can be told apart on the bands as well as
    # in the DOS.
    if projection != "orbital" and orbital_view == VIEW_RESOLVED:

        wanted = [s for s in (shells or SHELLS) if s in SHELLS]

        parts = [
            (name, orbital_columns(name))
            for shell in wanted
            for name in ORBITALS
            if name == shell or name.startswith(shell) and name != shell
        ]

    else:
        parts = [("", columns)]

    # Everything, so a fraction can be taken against it.
    everything = array.sum(axis=(2, 3))

    everything[everything < WEIGHT_FLOOR] = np.nan

    out: dict[str, np.ndarray] = {}

    for label, indices in selection.items():

        if projection == "orbital":

            block = array[np.ix_(
                range(array.shape[0]),
                range(array.shape[1]),
                indices,
                orbital_columns(label),
            )].sum(axis=(2, 3))

            # (kpoints, bands) out of the file; the plot wants bands
            # first.
            out[label] = np.nan_to_num(block / everything).T
            continue

        for suffix, picked in parts:

            block = array[np.ix_(
                range(array.shape[0]),
                range(array.shape[1]),
                indices,
                picked,
            )].sum(axis=(2, 3))

            # Name the orbitals when only some were taken. The weight is
            # already the intersection of the selection and the shells,
            # so a series labelled "Co" while carrying only Co's d
            # character is a legend describing the wrong quantity.
            if suffix:
                named = f"{label} {suffix}"

            elif shells and len(shells) < len(SHELLS):
                named = f"{label} {''.join(shells)}"

            else:
                named = label

            out[named] = np.nan_to_num(block / everything).T

    return out


def selected_atoms(
    run: Run,
    atoms: str = "",
    elements: Sequence[str] | None = None,
) -> list[int]:
    """
    Atoms named directly and by element, as one set.

    Behind the menu entry that asks for the sum over a chosen set rather
    than one curve per member: an adsorbate, a surface layer, a
    sublattice.  Indices count from zero, as the projection array does.
    """

    chosen: list[int] = []

    if atoms:
        for indices in resolve_selection(run, "atom", atoms=atoms).values():
            chosen.extend(indices)

    for element in elements or ():
        chosen.extend(run.indices_of(element))

    if not chosen:
        raise ElectronicError(
            "Nothing selected. Name some atoms, or tick an element, or "
            "both -- the point of this one is that it adds them up."
        )

    return sorted(set(chosen))

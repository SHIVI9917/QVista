"""
Orbital projections and real-space wavefunctions from a finished VASP run.

This is the calculation half of the Orbital tools.  It reads a directory
that VASP has already written and answers two questions about a single
electronic state:

* how much of that state sits on each element, atom and orbital, and
* what the wavefunction looks like in real space.

Nothing here imports Qt.  Every function returns data; the dialog decides
how to show it.

What the run must contain
-------------------------
``vasprun.xml``
    Always.  Eigenvalues and occupations come from here, and so do the
    projections -- but only when the run set ``LORBIT >= 10``.  Without
    that tag VASP writes no per-ion decomposition at all and there is
    nothing to tabulate; :func:`projection_for_state` says so plainly
    rather than returning zeros.
``POSCAR`` or ``CONTCAR``
    For the wavefunction, which needs a lattice to be written against.
``WAVECAR``
    For the wavefunction only.  It exists only when the run set
    ``LWAVE = .TRUE.``; the usual failure is a zero-byte file, which is
    VASP creating it and then writing nothing.
``OUTCAR``
    Optional, and used only as a fallback source for the Fermi energy.

On getting a wavefunction out of a WAVECAR
------------------------------------------
``fft_mesh`` places a band's coefficients at their G-vector indices, so
transforming it gives the cell-periodic part ``u_nk`` rather than the
wavefunction ``psi_nk = exp(i k . r) u_nk``.  At Gamma the two agree.
Anywhere else they differ by a phase that winds across the cell, and
drawing ``u`` in place of ``psi`` moves lobes and nodes.  On top of that,
a wavefunction is only defined up to a global phase, so "the real part"
means nothing until that phase is pinned.  :func:`psi_grid` does both --
the Bloch factor and the phase -- and everything drawn or written goes
through it.

On VBM and CBM in a metal
-------------------------
The band edges here are the highest occupied and lowest unoccupied
states, found by occupation rather than by assuming a gap.  In a metal
those two straddle the Fermi level and are essentially the same state,
which is the correct answer to "which state sits at the edge" for a
system that has no gap.  :func:`partially_occupied` reports the smeared
states so a caller can say as much instead of implying a band gap that
does not exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pymatgen.electronic_structure.core import Spin


# ------------------------------------------------------------
# Orbital ordering
# ------------------------------------------------------------

#: The nine channels VASP writes under ``LORBIT = 11``, in file order.
#: The order matters: the projection array is indexed by it, so a
#: rearranged list here would silently mislabel every column.
ORBITALS: tuple[str, ...] = (
    "s",
    "py",
    "pz",
    "px",
    "dxy",
    "dyz",
    "dz2",
    "dxz",
    "dx2-y2",
)

#: Where each shell sits inside :data:`ORBITALS`, as a half-open range.
GROUPS: dict[str, tuple[int, int]] = {
    "s": (0, 1),
    "p": (1, 4),
    "d": (4, 9),
}

#: Offered in the dialog alongside the three shells.
ALL_ORBITALS = "all"

#: The shells, in the order they fill.
SHELL_NAMES: tuple[str, ...] = ("s", "p", "d")

#: How finely a contribution table is broken down.
#:
#: ``VIEW_SPD`` sums each shell into one column, which is what a
#: chemist reads: "this state is 70% O p". ``VIEW_RESOLVED`` keeps all
#: nine channels apart, which is what says *which* d orbital -- the
#: difference between a t2g and an eg state, and the thing a crystal
#: field argument turns on.
VIEW_SPD = "spd"
VIEW_RESOLVED = "resolved"

VIEWS: dict[str, str] = {
    VIEW_SPD: "Projected - s, p, d",
    VIEW_RESOLVED: "Resolved - s(1), p(3), d(5)",
}


def to_shells(values: np.ndarray) -> np.ndarray:
    """
    Sum the nine channels into three, one per shell.

    Works on a single row of nine or on a stack of them; the last axis
    is the one collapsed.
    """

    array = np.asarray(values, dtype=float)

    if array.shape[-1] != len(ORBITALS):
        raise OrbitalError(
            f"Expected {len(ORBITALS)} orbital channels, got "
            f"{array.shape[-1]}."
        )

    return np.stack(
        [
            array[..., start:stop].sum(axis=-1)
            for start, stop in (GROUPS[name] for name in SHELL_NAMES)
        ],
        axis=-1,
    )


def view_labels(view: str) -> tuple[str, ...]:
    """The column names for a view."""

    if view == VIEW_SPD:
        return SHELL_NAMES

    if view == VIEW_RESOLVED:
        return ORBITALS

    raise OrbitalError(f"Unknown view {view!r}.")


def view_values(values: np.ndarray, view: str) -> np.ndarray:
    """The numbers under :func:`view_labels`, for the same view."""

    if view == VIEW_SPD:
        return to_shells(values)

    if view == VIEW_RESOLVED:
        return np.asarray(values, dtype=float)

    raise OrbitalError(f"Unknown view {view!r}.")


def shell_of(orbital: str) -> str:
    """Which shell a named orbital belongs to."""

    for name in SHELL_NAMES:

        start, stop = GROUPS[name]

        if orbital in ORBITALS[start:stop]:
            return name

    raise OrbitalError(f"Unknown orbital {orbital!r}.")

#: Occupations below this count as empty.  VASP writes exact zeros for
#: states well above the Fermi level, so the threshold only has to
#: separate "numerically zero" from "genuinely smeared".
OCCUPATION_THRESHOLD = 1e-8

#: Above this an occupation is treated as full.  Between the two a state
#: is partially occupied, which is the signature of a metal or of heavy
#: smearing.
FULL_OCCUPATION = 0.999999


class OrbitalError(Exception):
    """A run cannot answer the question being asked of it."""


# ------------------------------------------------------------
# Spin
# ------------------------------------------------------------


def spin_name(spin: Spin) -> str:
    """``UP`` or ``DOWN``, for display."""

    if spin == Spin.up:
        return "UP"

    if spin == Spin.down:
        return "DOWN"

    return str(spin)


def spin_short(spin: Spin) -> str:
    """``UP`` or ``DW``, for filenames."""

    if spin == Spin.up:
        return "UP"

    if spin == Spin.down:
        return "DW"

    return str(spin)


def spin_index(spin: Spin) -> int:
    """
    Translate to the integer :class:`~pymatgen.io.vasp.outputs.Wavecar`
    wants, where up is 0 and down is 1.
    """

    if spin == Spin.up:
        return 0

    if spin == Spin.down:
        return 1

    raise OrbitalError(f"Unsupported spin: {spin}")


def parse_spin(value: str) -> Spin | None:
    """Read ``up``/``down`` in any of the usual spellings."""

    value = value.strip().lower()

    if value in ("up", "u"):
        return Spin.up

    if value in ("down", "dw", "d"):
        return Spin.down

    return None


def get_spin_keys(mapping) -> list[Spin]:
    """
    The spin channels present in a pymatgen eigenvalue mapping.

    Normally the keys are :class:`Spin` members.  Some versions key by
    ``(Spin, index)`` tuples instead, so both are accepted rather than
    letting a version difference read as "this run is not spin
    polarised".
    """

    found: list[Spin] = []

    for key in mapping:

        if isinstance(key, Spin):
            if key not in found:
                found.append(key)

        elif isinstance(key, tuple) and key and isinstance(key[0], Spin):
            if key[0] not in found:
                found.append(key[0])

    return found


def get_spin_array(mapping, spin: Spin) -> np.ndarray:
    """The array for one spin channel, for either key convention."""

    if spin in mapping:
        return np.asarray(mapping[spin])

    matches = [
        (key, value)
        for key, value in mapping.items()
        if isinstance(key, tuple) and key and key[0] == spin
    ]

    if not matches:
        raise OrbitalError(f"Spin {spin_name(spin)} is not in this run.")

    matches.sort(key=lambda item: item[0][1])

    return np.asarray([np.asarray(value) for _, value in matches])


# ------------------------------------------------------------
# Reading the run
# ------------------------------------------------------------


def read_fermi_from_outcar(path: Path | str) -> float | None:
    """
    The last ``E-fermi`` written to an OUTCAR, or ``None``.

    The last one is the one that matters: a relaxation writes one per
    ionic step and only the final geometry is the converged answer.
    """

    path = Path(path)

    if not path.exists():
        return None

    pattern = re.compile(
        r"E-fermi\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
    )

    last: float | None = None

    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                last = float(match.group(1))

    return last


def load_run(directory: Path | str, projected: bool = False):
    """
    Parse ``vasprun.xml``, falling back to OUTCAR for the Fermi energy.

    ``projected`` is off by default because parsing the projections on a
    large run is slow and most callers only want eigenvalues.
    """

    from pymatgen.io.vasp import Vasprun

    directory = Path(directory)
    path = directory / "vasprun.xml"

    if not path.exists():
        raise OrbitalError(f"No vasprun.xml in {directory}")

    try:
        run = Vasprun(
            str(path),
            parse_dos=False,
            parse_eigen=True,
            parse_projected_eigen=projected,
            separate_spins=False,
        )

    except Exception as exc:
        # A truncated or null-padded vasprun.xml is common enough after
        # an interrupted job that the raw XML error is worth translating.
        raise OrbitalError(
            f"{path.name} could not be parsed: {exc}"
        ) from exc

    fermi: float | None = None

    if run.efermi is not None:
        try:
            fermi = float(run.efermi)
        except (TypeError, ValueError):
            fermi = None

    if fermi is None:
        fermi = read_fermi_from_outcar(directory / "OUTCAR")

    if fermi is not None:
        run.efermi = fermi

    return run


def load_poscar(directory: Path | str):
    """The run's structure, from POSCAR or failing that CONTCAR."""

    from pymatgen.io.vasp import Poscar

    directory = Path(directory)

    for name in ("POSCAR", "CONTCAR"):

        path = directory / name

        if path.exists():
            return Poscar.from_file(str(path))

    raise OrbitalError(f"Neither POSCAR nor CONTCAR is in {directory}")


def load_wavecar(directory: Path | str, precision: str = "normal"):
    """
    Open the WAVECAR.

    A zero-byte file gets its own message because that is what a run
    with ``LWAVE = .FALSE.`` leaves behind, and "cannot parse" would
    send the user looking for corruption instead of a tag.
    """

    from pymatgen.io.vasp import Wavecar

    directory = Path(directory)
    path = directory / "WAVECAR"

    if not path.exists():
        raise OrbitalError(f"No WAVECAR in {directory}")

    if path.stat().st_size == 0:
        raise OrbitalError(
            "WAVECAR is empty, so the run was made with "
            "LWAVE = .FALSE. and kept no wavefunctions. Re-run with "
            "LWAVE = .TRUE. to visualise them."
        )

    try:
        return Wavecar(str(path), precision=precision)

    except Exception as exc:
        raise OrbitalError(f"WAVECAR could not be read: {exc}") from exc


# ------------------------------------------------------------
# States
# ------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """
    One electronic state: a band at a k-point in a spin channel.

    ``kpoint`` and ``band`` are 1-based, matching how VASP numbers them
    everywhere the user will have seen them.
    """

    spin: Spin
    kpoint: int
    band: int
    energy: float
    occupation: float
    kcoord: tuple[float, float, float]

    def label(self) -> str:
        """Short description for a dialog heading."""

        return (
            f"k={self.kpoint}  band={self.band}  "
            f"spin={spin_name(self.spin)}"
        )

    def filename_stem(self) -> str:
        """``B0130_K0001_DW``, as the reference analyser names files."""

        return (
            f"B{self.band:04d}_K{self.kpoint:04d}_{spin_short(self.spin)}"
        )


def _state(run, spin: Spin, array: np.ndarray, k: int, b: int) -> State:
    """Build a :class:`State` from zero-based indices into ``array``."""

    coords = np.asarray(run.actual_kpoints[k], dtype=float)

    return State(
        spin=spin,
        kpoint=k + 1,
        band=b + 1,
        energy=float(array[k, b, 0]),
        occupation=float(array[k, b, 1]),
        kcoord=(float(coords[0]), float(coords[1]), float(coords[2])),
    )


def collect_states(run) -> list[State]:
    """Every state in the run, across all spins, k-points and bands."""

    states: list[State] = []

    for spin in get_spin_keys(run.eigenvalues):

        array = get_spin_array(run.eigenvalues, spin)

        if array.ndim != 3:
            raise OrbitalError(
                f"Unexpected eigenvalue shape {array.shape}"
            )

        for k in range(array.shape[0]):
            for b in range(array.shape[1]):
                states.append(_state(run, spin, array, k, b))

    return states


def _edges(
    states: Sequence[State],
    threshold: float,
) -> tuple[State | None, State | None]:
    """Split by occupation and take the two innermost states."""

    occupied = [s for s in states if s.occupation > threshold]
    empty = [s for s in states if s.occupation <= threshold]

    highest = max(occupied, key=lambda s: s.energy) if occupied else None
    lowest = min(empty, key=lambda s: s.energy) if empty else None

    return highest, lowest


def find_vbm_cbm(
    run,
    threshold: float = OCCUPATION_THRESHOLD,
) -> tuple[State | None, State | None]:
    """
    The highest occupied and lowest unoccupied states in the run.

    For a semiconductor these are the valence and conduction band edges.
    For a metal they are the two states either side of the Fermi level,
    which is the honest answer rather than a fabricated gap.
    """

    return _edges(collect_states(run), threshold)


def find_spin_vbm_cbm(
    run,
    spin: Spin,
    threshold: float = OCCUPATION_THRESHOLD,
) -> tuple[State | None, State | None]:
    """The same two edges, restricted to one spin channel."""

    states = [s for s in collect_states(run) if s.spin == spin]

    return _edges(states, threshold)


def find_fermi_state(run, spin: Spin | None = None) -> State:
    """The state closest in energy to the Fermi level."""

    if run.efermi is None:
        raise OrbitalError(
            "This run has no Fermi energy in vasprun.xml or OUTCAR."
        )

    fermi = float(run.efermi)

    states = collect_states(run)

    if spin is not None:
        states = [s for s in states if s.spin == spin]

    if not states:
        raise OrbitalError("The run contains no states.")

    return min(states, key=lambda s: abs(s.energy - fermi))


def find_state(run, name: str, spin: Spin | None = None) -> State:
    """Resolve ``vbm``, ``cbm`` or ``fermi`` to an actual state."""

    name = name.strip().lower()

    if name == "fermi":
        return find_fermi_state(run, spin=spin)

    if name not in ("vbm", "cbm"):
        raise OrbitalError("State must be vbm, cbm or fermi.")

    if spin is None:
        valence, conduction = find_vbm_cbm(run)
    else:
        valence, conduction = find_spin_vbm_cbm(run, spin)

    state = valence if name == "vbm" else conduction

    if state is None:
        missing = "occupied" if name == "vbm" else "unoccupied"
        raise OrbitalError(f"The run has no {missing} states.")

    return state


def state_at(run, kpoint: int, band: int, spin: Spin) -> State:
    """
    A state named explicitly by 1-based k-point and band.

    This is what the Custom Band section uses, so the bounds are checked
    against the run rather than left to raise an IndexError deeper in.
    """

    array = get_spin_array(run.eigenvalues, spin)

    if not 1 <= kpoint <= array.shape[0]:
        raise OrbitalError(
            f"k-point must be between 1 and {array.shape[0]}."
        )

    if not 1 <= band <= array.shape[1]:
        raise OrbitalError(
            f"Band must be between 1 and {array.shape[1]}."
        )

    return _state(run, spin, array, kpoint - 1, band - 1)


def partially_occupied(
    run,
    threshold: float = OCCUPATION_THRESHOLD,
    full: float = FULL_OCCUPATION,
) -> list[State]:
    """
    States that are neither empty nor full, nearest the Fermi level first.

    A non-empty result means the band edges should be read as
    highest-occupied and lowest-unoccupied rather than as a band gap.
    """

    smeared = [
        s
        for s in collect_states(run)
        if threshold < s.occupation < full
    ]

    if run.efermi is None:
        return smeared

    fermi = float(run.efermi)

    smeared.sort(key=lambda s: abs(s.energy - fermi))

    return smeared


# ------------------------------------------------------------
# Atom selection
# ------------------------------------------------------------


def parse_atom_list(text: str) -> list[int]:
    """
    Read ``9``, ``9,11``, ``9-20`` or ``9-12,18,20`` as 1-based numbers.
    """

    atoms: list[int] = []

    for part in text.split(","):

        part = part.strip()

        if not part:
            continue

        if "-" in part:

            first, last = part.split("-", maxsplit=1)

            try:
                start, stop = int(first), int(last)
            except ValueError:
                raise OrbitalError(f"{part!r} is not a range of atoms.")

            if stop < start:
                raise OrbitalError(f"{part!r} runs backwards.")

            atoms.extend(range(start, stop + 1))

            continue

        try:
            atoms.append(int(part))
        except ValueError:
            raise OrbitalError(f"{part!r} is not an atom number.")

    return sorted(set(atoms))


def select_atoms(
    structure,
    element: str | None = None,
    atoms: str | None = None,
) -> list[int]:
    """
    Zero-based indices of the atoms to report on.

    ``element`` selects every atom of that species, ``atoms`` takes an
    explicit 1-based list, and giving neither means the whole cell.
    """

    if element is not None:

        chosen = [
            index
            for index, site in enumerate(structure)
            if site.specie.symbol.lower() == element.lower()
        ]

        if not chosen:
            raise OrbitalError(f"No {element} atoms in this structure.")

        return chosen

    if atoms is not None:

        numbers = parse_atom_list(atoms)

        for number in numbers:
            if not 1 <= number <= len(structure):
                raise OrbitalError(
                    f"Atom {number} is outside 1-{len(structure)}."
                )

        return [number - 1 for number in numbers]

    return list(range(len(structure)))


def elements_of(structure) -> list[str]:
    """
    The distinct species, in the order they first appear.

    First-appearance order rather than alphabetical so the buttons in
    the dialog line up with the POSCAR the user is looking at.
    """

    seen: list[str] = []

    for site in structure:

        symbol = site.specie.symbol

        if symbol not in seen:
            seen.append(symbol)

    return seen


# ------------------------------------------------------------
# Projections
# ------------------------------------------------------------


def projection_for_state(run, state: State) -> np.ndarray:
    """
    The ``(atom, orbital)`` projection array for one state.

    Only the first nine channels are returned.  A run with f electrons
    writes sixteen, and the extra seven are dropped rather than shown,
    because everything downstream is labelled by :data:`ORBITALS`.
    """

    if run.projected_eigenvalues is None:
        raise OrbitalError(
            "This run has no orbital projections. Set LORBIT = 11 in "
            "the INCAR and re-run to write them."
        )

    projected = get_spin_array(run.projected_eigenvalues, state.spin)

    values = np.asarray(
        projected[state.kpoint - 1, state.band - 1],
        dtype=float,
    )

    if values.ndim != 2:
        raise OrbitalError(
            f"Unexpected projection shape {values.shape}"
        )

    if values.shape[1] < len(ORBITALS):
        raise OrbitalError(
            f"Expected {len(ORBITALS)} orbital channels, found "
            f"{values.shape[1]}. LORBIT = 11 writes the full set."
        )

    return values[:, : len(ORBITALS)]


@dataclass(frozen=True)
class Contribution:
    """
    What one selection of atoms contributes to one state.

    ``per_atom`` keeps the rows separate so the dialog can show the
    per-atom table; ``totals`` is their sum, which is what the summary
    and the percentages are built from.
    """

    label: str
    atoms: tuple[int, ...]
    per_atom: np.ndarray
    totals: np.ndarray

    def group(self, name: str) -> np.ndarray:
        """The s, p or d slice of :attr:`totals`, or all nine."""

        if name == ALL_ORBITALS:
            return self.totals

        if name not in GROUPS:
            raise OrbitalError(f"Unknown orbital group {name!r}.")

        start, stop = GROUPS[name]

        return self.totals[start:stop]

    def group_names(self, name: str) -> tuple[str, ...]:
        """The orbital labels matching :meth:`group`."""

        if name == ALL_ORBITALS:
            return ORBITALS

        start, stop = GROUPS[name]

        return ORBITALS[start:stop]


def contribution(
    run,
    state: State,
    structure,
    atoms: Iterable[int],
    label: str,
) -> Contribution:
    """Sum the projection over ``atoms`` (zero-based)."""

    projected = projection_for_state(run, state)

    chosen = tuple(int(index) for index in atoms)

    if not chosen:
        raise OrbitalError("No atoms were selected.")

    for index in chosen:
        if not 0 <= index < projected.shape[0]:
            raise OrbitalError(
                f"Atom {index + 1} is outside the projection array."
            )

    rows = projected[list(chosen), :]

    return Contribution(
        label=label,
        atoms=chosen,
        per_atom=rows,
        totals=rows.sum(axis=0),
    )


def percentages(values: np.ndarray) -> np.ndarray:
    """
    Normalise to percent, returning zeros when there is nothing there.

    An all-but-zero group is common -- a d state has no p character --
    and dividing by its sum would give NaNs across a whole table row.
    """

    values = np.asarray(values, dtype=float)

    total = float(values.sum())

    if abs(total) <= 1e-14:
        return np.zeros_like(values)

    return 100.0 * values / total


def element_table(
    run,
    state: State,
    structure,
) -> list[Contribution]:
    """
    One :class:`Contribution` per element, in structure order.

    This is the table the VBM and CBM sections open on: every element
    against all nine orbitals, before any element has been picked.
    """

    return [
        contribution(
            run,
            state,
            structure,
            select_atoms(structure, element=symbol),
            symbol,
        )
        for symbol in elements_of(structure)
    ]


@dataclass(frozen=True)
class Breakdown:
    """
    Every orbital of a state against every element, as one table.

    The element table answers "which atoms is this state on" and the
    shell buttons answer "which shell", but neither puts the whole state
    on one screen. This does: nine rows, one column per element, and a
    percentage of the whole state on each -- so the sentence "the VBM is
    O p with some Co d" can be read off directly instead of assembled
    from three clicks.
    """

    #: Row labels: the nine orbitals, then the three shell subtotals.
    orbitals: tuple[str, ...]

    #: Which shell each row belongs to. Empty for a subtotal row.
    shells: tuple[str, ...]

    #: True on the three shell-subtotal rows, which are drawn in bold.
    is_subtotal: tuple[bool, ...]

    #: Column labels: one per element.
    elements: tuple[str, ...]

    #: ``(rows, elements)`` raw values.
    values: np.ndarray

    #: ``(rows,)`` summed over the elements.
    totals: np.ndarray

    #: ``(rows,)`` as a percentage of the whole state.
    percent: np.ndarray

    @property
    def total(self) -> float:
        """Everything, over every orbital and element."""

        # Only the nine real rows; the subtotals would double-count.
        return float(
            self.totals[
                [not flag for flag in self.is_subtotal]
            ].sum()
        )


def breakdown(rows: Sequence[Contribution]) -> Breakdown:
    """
    Build the whole-state table from the per-element contributions.

    Percentages are taken against the sum over the nine orbitals only.
    Including the shell subtotals in the denominator would halve every
    number while leaving the table looking entirely reasonable.
    """

    if not rows:
        raise OrbitalError("There are no contributions to break down.")

    # (elements, 9) -> (9, elements)
    matrix = np.vstack([entry.totals for entry in rows]).T

    labels = list(ORBITALS)
    shells = [shell_of(name) for name in ORBITALS]
    subtotal = [False] * len(ORBITALS)

    blocks = [matrix]

    for name in SHELL_NAMES:

        start, stop = GROUPS[name]

        blocks.append(matrix[start:stop].sum(axis=0, keepdims=True))

        labels.append(f"{name} total")
        shells.append(name)
        subtotal.append(True)

    values = np.vstack(blocks)

    totals = values.sum(axis=1)

    whole = float(matrix.sum())

    percent = (
        np.zeros_like(totals)
        if abs(whole) <= 1e-14
        else 100.0 * totals / whole
    )

    return Breakdown(
        orbitals=tuple(labels),
        shells=tuple(shells),
        is_subtotal=tuple(subtotal),
        elements=tuple(entry.label for entry in rows),
        values=values,
        totals=totals,
        percent=percent,
    )


# ------------------------------------------------------------
# Wavefunctions
# ------------------------------------------------------------

#: The components that can be written, and how each is labelled.
WAVE_PARTS: dict[str, str] = {
    "real": "Real part of psi",
    "imag": "Imaginary part of psi",
    "square": "psi squared",
}


def wavefunction_filename(part: str, state: State) -> str:
    """``WF_REAL_B0130_K0001_DW.vasp``, matching the reference analyser."""

    if part not in WAVE_PARTS:
        raise OrbitalError(f"Unknown wavefunction part {part!r}.")

    return f"WF_{part.upper()}_{state.filename_stem()}.vasp"


def bloch_phase(shape: tuple[int, int, int], kpoint) -> np.ndarray:
    """
    The ``exp(2i pi k . f)`` factor on a grid of fractional coordinates.

    ``kpoint`` is fractional, as VASP and the WAVECAR store it, so the
    dot product is taken against fractional coordinates and no
    reciprocal lattice is needed.
    """

    k = np.asarray(kpoint, dtype=float)

    if k.shape != (3,):
        raise OrbitalError(f"A k-point needs three components, got {k.shape}")

    nx, ny, nz = shape

    axes = [
        np.arange(n, dtype=float) / float(n) for n in (nx, ny, nz)
    ]

    fx, fy, fz = np.meshgrid(*axes, indexing="ij")

    return np.exp(
        2j * np.pi * (k[0] * fx + k[1] * fy + k[2] * fz)
    )


def is_time_reversal_invariant(kpoint) -> bool:
    """
    Whether ``-k`` is ``k`` plus a reciprocal lattice vector.

    At such a point -- Gamma, a zone-boundary point, and every k-point
    of a Gamma-centred even mesh -- a band and its complex conjugate are
    the same state, so the wavefunction can be made real by a single
    global phase.  Away from one it cannot, and the real part is a
    genuine choice of gauge rather than the whole wavefunction.
    """

    k = np.asarray(kpoint, dtype=float)

    return bool(np.allclose(2.0 * k, np.round(2.0 * k), atol=1e-6))


def realify(psi: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Rotate a wavefunction's global phase until it is as real as it can be.

    ``psi`` and ``exp(i theta) psi`` are the same state, but their real
    parts are different functions -- so without fixing the phase, "the
    real part" means whatever arbitrary rotation the diagonaliser
    happened to leave behind, and can be almost nothing at all.

    The angle that maximises the real part is ``arg(sum psi^2) / 2``.
    The overall sign is then fixed by making the largest value positive,
    which is not physics -- a wavefunction and its negative are the same
    state -- but does mean the same band always comes out the same way
    up.

    Returns the rotated array and the fraction of the norm still left in
    the imaginary part.  At a time-reversal invariant k-point that
    fraction should be ~0; a large one means the real part genuinely is
    only a part.
    """

    psi = np.asarray(psi)

    total = float(np.sum(np.abs(psi) ** 2))

    if total <= 0.0:
        return psi, 0.0

    theta = 0.5 * float(np.angle(np.sum(psi ** 2)))

    rotated = psi * np.exp(-1j * theta)

    imaginary = float(np.sum(np.imag(rotated) ** 2)) / total

    real = np.real(rotated)

    if real.size and real.flat[int(np.argmax(np.abs(real)))] < 0:
        rotated = -rotated

    return rotated, imaginary


def psi_grid(wavecar, state: State) -> tuple[np.ndarray, float]:
    """
    The complex Bloch function of one state on the real-space grid.

    Two things happen here that ``ifftn(fft_mesh(...))`` alone does not
    do, and both of them matter for what gets drawn:

    ``exp(i k . r)``
        ``fft_mesh`` places the coefficients at their G-vector indices
        and nothing else, so the transform returns the cell-periodic
        part ``u_nk``, not the wavefunction ``psi_nk = exp(i k . r)
        u_nk``.  Away from Gamma the two differ by a phase that winds
        across the cell, which puts lobes and nodes in the wrong places.

    the global phase
        See :func:`realify`.  Without it the real part is an arbitrary
        slice through a complex number.

    Returns the grid and the fraction of the norm left in the imaginary
    part after the rotation.
    """

    mesh = wavecar.fft_mesh(
        state.kpoint - 1,
        state.band - 1,
        spin=spin_index(state.spin),
    )

    psi = np.fft.ifftn(mesh)

    kpoint = np.asarray(wavecar.kpoints[state.kpoint - 1], dtype=float)

    psi = psi * bloch_phase(psi.shape, kpoint)

    return realify(psi)


def _realspace(wavecar, state: State, part: str) -> np.ndarray:
    """
    One component of a state's wavefunction on the real-space grid.

    ``square`` is unaffected by everything :func:`psi_grid` does -- a
    phase has modulus one -- so the density is the same either way and
    only the signed parts change.
    """

    if part not in WAVE_PARTS:
        raise OrbitalError(f"Unknown wavefunction part {part!r}.")

    psi, _ = psi_grid(wavecar, state)

    if part == "real":
        data = np.real(psi)

    elif part == "imag":
        data = np.imag(psi)

    else:
        data = np.abs(psi) ** 2

    return np.asarray(data, dtype=float)


def _check_range(wavecar, state: State) -> None:
    """Refuse a state the WAVECAR does not hold."""

    if state.kpoint - 1 >= wavecar.nk:
        raise OrbitalError(
            f"The WAVECAR holds {wavecar.nk} k-points, so k-point "
            f"{state.kpoint} is not in it."
        )

    if state.band - 1 >= wavecar.nb:
        raise OrbitalError(
            f"The WAVECAR holds {wavecar.nb} bands, so band "
            f"{state.band} is not in it."
        )


def wavefunction_data(
    wavecar,
    poscar,
    state: State,
    part: str,
) -> np.ndarray:
    """
    The real-space grid for one component, without writing a file.

    This is what the 3D viewer contours.  It shares ``_realspace`` with
    :func:`write_wavefunction`, so what is drawn on screen and what
    lands in the ``.vasp`` file are the same numbers.
    """

    _check_range(wavecar, state)

    if part == "square":

        parchg = wavecar.get_parchg(
            poscar,
            kpoint=state.kpoint - 1,
            band=state.band - 1,
            spin=spin_index(state.spin),
            phase=False,
            scale=2,
        )

        return np.asarray(parchg.data["total"], dtype=float)

    return _realspace(wavecar, state, part)


def wavefunction_note(wavecar, state: State) -> str | None:
    """
    What to say about the real part of this state, if anything.

    At a time-reversal invariant k-point the wavefunction is real once
    its global phase is fixed, and there is nothing to warn about.
    Anywhere else the real and imaginary parts are two halves of one
    complex function and the caller should know that before reading a
    node off the picture.
    """

    _, imaginary = psi_grid(wavecar, state)

    kpoint = np.asarray(wavecar.kpoints[state.kpoint - 1], dtype=float)

    if is_time_reversal_invariant(kpoint) and imaginary < 1e-3:
        return None

    return (
        f"k = {np.round(kpoint, 4).tolist()} is not time-reversal "
        f"invariant, so this state is genuinely complex: "
        f"{imaginary:.0%} of it is in the imaginary part. The real part "
        "is one choice of phase, not the whole wavefunction."
    )


def write_wavefunction(
    wavecar,
    poscar,
    state: State,
    part: str,
    path: Path | str,
) -> Path:
    """
    Write one component of a state's wavefunction as a VASP volumetric file.

    ``square`` goes through pymatgen's ``get_parchg``, which is the
    supported route to a charge density and returns it on a finer grid.
    ``real`` and ``imag`` come from :func:`psi_grid`, which is the only
    place the Bloch factor and the global phase are applied -- and which
    the viewer contours too, so the picture and the file are the same
    numbers.
    """

    from pymatgen.io.vasp import VolumetricData

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    _check_range(wavecar, state)

    if part == "square":

        parchg = wavecar.get_parchg(
            poscar,
            kpoint=state.kpoint - 1,
            band=state.band - 1,
            spin=spin_index(state.spin),
            phase=False,
            scale=2,
        )

        parchg.write_file(str(path))

        return path

    data = _realspace(wavecar, state, part)

    volumetric = VolumetricData(poscar.structure, {"total": data})

    volumetric.write_file(str(path))

    return path

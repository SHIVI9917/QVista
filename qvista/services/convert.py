"""
Structure format conversion.

The named conversions people actually ask for, each as one entry rather
than a source format and a target format to pair up themselves.

Two of them are not format conversions at all
---------------------------------------------
``Direct to Cartesian`` and its inverse rewrite a POSCAR in place: same
file format, same structure, different coordinate convention. They are
here because that is where a user looks for them, and because getting
one by hand means multiplying every position by the lattice matrix,
which is exactly the sort of thing to let a machine do.

The molecule problem
--------------------
A Gaussian input file describes a molecule floating in nothing. VASP
needs a periodic cell, so ``gjf -> POSCAR`` cannot be a conversion in
the way ``CIF -> POSCAR`` is: a cell has to come from somewhere.

Two honest answers, and this offers both. If the file carries ``Tv``
translation vectors -- Gaussian's own way of saying periodic -- they are
the cell. Otherwise a box is built around the molecule with a stated
amount of vacuum on each side, and the result is flagged as invented
rather than read. Silently picking a box size is the one thing not on
offer: the vacuum spacing decides how strongly the molecule sees its own
periodic images, and that is the user's call, not this module's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ConversionError(Exception):
    """Raised when a conversion cannot be done, with a reason."""


# ------------------------------------------------------------
# Coordinate conventions
# ------------------------------------------------------------

DIRECT = "direct"
CARTESIAN = "cartesian"

COORDINATE_MODES: dict[str, str] = {
    DIRECT: "Fractional coordinates, in units of the lattice vectors.",
    CARTESIAN: "Absolute coordinates, in angstrom.",
}

#: Vacuum on each side of a molecule placed in a box, in angstrom.
#:
#: Ten is the usual starting point for an isolated molecule: it puts
#: about twenty angstrom between periodic images, which is far enough
#: for most things and not so far that the plane-wave basis becomes
#: absurd. It is a default, not a right answer.
DEFAULT_VACUUM = 10.0

#: What Gaussian calls a translation vector.
TV = "Tv"


# ------------------------------------------------------------
# The conversions on offer
# ------------------------------------------------------------


@dataclass(frozen=True)
class Conversion:
    """One named conversion, as it appears in the menu."""

    key: str
    label: str

    #: What it reads and what it writes, for the file dialogs.
    source_filter: str
    target_name: str
    target_filter: str

    #: True when the result needs a lattice the source may not have.
    needs_cell: bool = False

    #: Set for the two that rewrite coordinates rather than convert.
    coordinates: str = ""

    note: str = ""


CONVERSIONS: tuple[Conversion, ...] = (
    Conversion(
        key="gjf_to_vasp",
        label="Gaussian .gjf  ->  .vasp",
        source_filter="Gaussian input (*.gjf *.com);;All Files (*)",
        target_name="structure.vasp",
        target_filter="VASP structure (*.vasp);;All Files (*)",
        needs_cell=True,
    ),
    Conversion(
        key="gjf_to_poscar",
        label="Gaussian .gjf  ->  POSCAR",
        source_filter="Gaussian input (*.gjf *.com);;All Files (*)",
        target_name="POSCAR",
        target_filter="VASP POSCAR (POSCAR *.vasp);;All Files (*)",
        needs_cell=True,
    ),
    Conversion(
        key="cif_to_poscar",
        label="CIF  ->  POSCAR",
        source_filter="CIF file (*.cif);;All Files (*)",
        target_name="POSCAR",
        target_filter="VASP POSCAR (POSCAR *.vasp);;All Files (*)",
    ),
    Conversion(
        key="poscar_to_cif",
        label="POSCAR  ->  CIF",
        source_filter="VASP structure (POSCAR CONTCAR *.vasp);;All Files (*)",
        target_name="structure.cif",
        target_filter="CIF file (*.cif);;All Files (*)",
    ),
    Conversion(
        key="poscar_to_xyz",
        label="POSCAR  ->  XYZ",
        source_filter="VASP structure (POSCAR CONTCAR *.vasp);;All Files (*)",
        target_name="structure.xyz",
        target_filter="XYZ file (*.xyz);;All Files (*)",
        note=(
            "Written as extended XYZ, so the lattice survives and the "
            "file reads back as a crystal rather than a molecule."
        ),
    ),
    Conversion(
        key="to_cartesian",
        label="POSCAR: Direct  ->  Cartesian",
        source_filter="VASP structure (POSCAR CONTCAR *.vasp);;All Files (*)",
        target_name="POSCAR_cartesian",
        target_filter="VASP POSCAR (POSCAR *.vasp);;All Files (*)",
        coordinates=CARTESIAN,
        note="Same structure and same format; only the coordinates change.",
    ),
    Conversion(
        key="to_direct",
        label="POSCAR: Cartesian  ->  Direct",
        source_filter="VASP structure (POSCAR CONTCAR *.vasp);;All Files (*)",
        target_name="POSCAR_direct",
        target_filter="VASP POSCAR (POSCAR *.vasp);;All Files (*)",
        coordinates=DIRECT,
        note="Same structure and same format; only the coordinates change.",
    ),
)


def find(key: str) -> Conversion:
    """Look one up, or say which exist."""

    for item in CONVERSIONS:
        if item.key == key:
            return item

    raise ConversionError(
        f"Unknown conversion {key!r}. Try one of "
        f"{', '.join(item.key for item in CONVERSIONS)}."
    )


def source_patterns(conversion: Conversion) -> tuple[str, ...]:
    """
    The filename patterns a conversion reads, as ``POSCAR``, ``*.cif``.

    Taken from the first group of the Qt file filter rather than kept
    beside it, so the list offered in the browser and the list used to
    recognise a file can never drift apart. Only the first group is
    read: the last one is always "All Files (*)", which matches
    everything and would make the answer meaningless.
    """

    head = conversion.source_filter.split(";;")[0]

    opened = head.find("(")
    closed = head.rfind(")")

    if opened == -1 or closed <= opened:
        return ()

    return tuple(piece for piece in head[opened + 1:closed].split() if piece)


def accepts(conversion: Conversion, path: Path | str) -> bool:
    """Whether ``path`` looks like something this conversion can read."""

    import fnmatch

    name = Path(path).name

    return any(
        fnmatch.fnmatch(name, pattern)
        for pattern in source_patterns(conversion)
    )


def conversion_for(path: Path | str) -> Conversion | None:
    """
    The first conversion that reads files like ``path``, if any.

    Used to open the dialog already pointing at the structure that is
    loaded, rather than at whichever conversion happens to be first in
    the list. ``None`` when nothing on offer reads that format, which is
    the honest answer -- guessing would silently pair a CIF with a
    Gaussian parser.
    """

    for item in CONVERSIONS:

        if accepts(item, path):
            return item

    return None


# ------------------------------------------------------------
# Gaussian input
# ------------------------------------------------------------


def _looks_like(text: str) -> str | None:
    """
    Name the format a file actually is, when it is a familiar one.

    Only used to explain a parse failure. Picking the wrong conversion
    off the list is an easy thing to do, and "it looks like a CIF" is
    the whole of what the user needs to hear -- whereas the parser's
    own complaint is about the line it happened to stop on.
    """

    lines = [line.strip() for line in text.splitlines()]

    body = [line for line in lines if line]

    if not body:
        return None

    # A CIF is the one format that names itself: every file opens with
    # a data block header.
    if any(line.startswith("data_") for line in body[:20]):
        return "a CIF"

    # An XYZ opens with a bare atom count and a comment, then that many
    # rows. Checking the count matches is what keeps this from claiming
    # every file whose first line is a number.
    if body[0].isdigit():

        expected = int(body[0])

        if expected and len(lines) >= expected + 2:
            return "an XYZ file"

    # A POSCAR has a comment, a scale factor on its own, and three
    # lattice rows of three numbers each. The scale plus the rows is
    # specific enough; the comment line above them can say anything.
    if len(body) >= 5 and _is_number(body[1]):

        if all(_row_of_numbers(line, 3) for line in body[2:5]):
            return "a POSCAR"

    return None


def _is_number(line: str) -> bool:

    try:
        float(line)

    except ValueError:
        return False

    return True


def _row_of_numbers(line: str, count: int) -> bool:

    parts = line.split()

    return len(parts) == count and all(_is_number(part) for part in parts)


def read_gjf(path: Path | str) -> tuple[Any, np.ndarray | None]:
    """
    Read a Gaussian input file.

    Returns the molecule and its translation vectors, or ``None`` when
    it has none -- which is the usual case, a molecule in vacuum.

    ``Tv`` rows are Gaussian's own periodic notation: up to three of
    them, written as if they were atoms of element ``Tv``. pymatgen
    reads them back as sites of an unknown species, so they are pulled
    out here before they can be mistaken for atoms and end up in the
    POSCAR as a made-up element.
    """

    path = Path(path)

    if not path.is_file():
        raise ConversionError(f"There is no such file: {path}")

    from pymatgen.io.gaussian import GaussianInput

    try:
        text = path.read_text(errors="replace")

    except OSError as exc:
        raise ConversionError(f"Could not read {path.name}: {exc}") from exc

    # Tv rows have to come out before pymatgen sees them. They are
    # written as if they were atoms of element "Tv", which is not an
    # element, and the parser raises on it rather than skipping it --
    # so a periodic Gaussian file is unreadable unless they are removed
    # here first.
    body, vectors = _split_translation_vectors(text)

    try:
        parsed = GaussianInput.from_str(body)

    except Exception as exc:

        # The parser fails on the line it happened to reach, so it
        # reports an IndexError on a file that is simply a CIF. Lead
        # with what is wrong with the file, and keep the parser's own
        # words at the end for the case where it is a gjf after all.
        instead = _looks_like(text)

        raise ConversionError(
            f"{path.name} is not a Gaussian input file"
            + (f" -- it looks like {instead}." if instead else ".")
            + "\n\nA .gjf has a route line, a title, a charge and "
            "multiplicity line, and then the atoms."
            + (
                f"\n\n({type(exc).__name__}: {exc})"
                if instead is None
                else ""
            )
        ) from exc

    molecule = parsed.molecule

    if molecule is None or not len(molecule):
        raise ConversionError(
            f"{path.name} carries no atoms."
            + (
                " It has translation vectors, but nothing to put in "
                "them."
                if vectors
                else ""
            )
        )

    trimmed = molecule

    lattice = np.asarray(vectors, dtype=float) if vectors else None

    if lattice is not None and lattice.shape != (3, 3):
        # One or two Tv rows means a chain or a slab. VASP wants three
        # vectors, and inventing the missing ones is the box case.
        lattice = None

    return trimmed, lattice


#: A ``Tv`` row: the marker, then three numbers.
_TV_ROW = re.compile(
    r"^\s*" + TV + r"\s+([-+]?[\d.eEdD+-]+)\s+([-+]?[\d.eEdD+-]+)"
    r"\s+([-+]?[\d.eEdD+-]+)\s*$",
    re.IGNORECASE,
)


def _split_translation_vectors(text: str) -> tuple[str, list[list[float]]]:
    """
    Pull the ``Tv`` rows out of a Gaussian input.

    Returns the file without them, and the vectors they carried. Done
    on the text rather than on the parsed molecule because pymatgen
    refuses ``Tv`` as an element and never gets far enough to hand one
    back.
    """

    kept: list[str] = []
    vectors: list[list[float]] = []

    for line in text.splitlines():

        found = _TV_ROW.match(line)

        if found is None:
            kept.append(line)
            continue

        vectors.append(
            [
                float(value.replace("D", "E").replace("d", "e"))
                for value in found.groups()
            ]
        )

    return "\n".join(kept), vectors


def box_around(
    molecule, vacuum: float = DEFAULT_VACUUM
) -> np.ndarray:
    """
    An orthorhombic cell enclosing ``molecule`` with vacuum around it.

    Sized from the molecule's own extent plus ``vacuum`` on *each* side,
    so the gap between one periodic image and the next is twice the
    number asked for.
    """

    if vacuum < 0.0:
        raise ConversionError("Vacuum cannot be negative.")

    coords = np.asarray(molecule.cart_coords, dtype=float)

    if coords.size == 0:
        raise ConversionError("There are no atoms to put in a box.")

    extent = coords.max(axis=0) - coords.min(axis=0)

    edges = extent + 2.0 * vacuum

    # A single atom has no extent at all, and a zero-length lattice
    # vector is not a cell.
    edges = np.maximum(edges, 2.0 * max(vacuum, 1.0))

    return np.diag(edges)


def molecule_to_structure(
    molecule,
    lattice: np.ndarray | None = None,
    vacuum: float = DEFAULT_VACUUM,
) -> tuple[Any, bool]:
    """
    Put a molecule in a cell.

    Returns the structure and whether the cell was invented rather than
    read from the file. The caller is expected to pass that on: a
    fabricated cell is a modelling decision, and it should not be
    possible to end up with one without having been told.
    """

    from pymatgen.core import Lattice, Structure

    invented = lattice is None

    matrix = box_around(molecule, vacuum) if invented else lattice

    coords = np.asarray(molecule.cart_coords, dtype=float)

    if invented:
        # Centre it, so the vacuum is even on all sides rather than all
        # of it on one.
        centre = 0.5 * (coords.max(axis=0) + coords.min(axis=0))

        coords = coords - centre + 0.5 * np.diag(matrix)

    structure = Structure(
        Lattice(matrix),
        [site.specie for site in molecule],
        coords,
        coords_are_cartesian=True,
    )

    return structure, invented


# ------------------------------------------------------------
# Coordinate mode
# ------------------------------------------------------------


def rewrite_coordinates(
    source: Path | str, mode: str
) -> str:
    """
    Re-render a POSCAR in the other coordinate convention.

    The structure is unchanged: the same atoms in the same places,
    described differently. Selective-dynamics flags survive, because
    they belong to the atoms and not to the convention.
    """

    if mode not in COORDINATE_MODES:
        raise ConversionError(
            f"Unknown coordinate mode {mode!r}. "
            f"Try {DIRECT} or {CARTESIAN}."
        )

    from pymatgen.io.vasp.inputs import Poscar

    path = Path(source)

    if not path.is_file():
        raise ConversionError(f"There is no such file: {path}")

    try:
        poscar = Poscar.from_file(str(path), check_for_potcar=False)

    except Exception as exc:
        raise ConversionError(
            f"Could not read {path.name} as a POSCAR: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return poscar.get_str(direct=(mode == DIRECT))


def coordinate_mode(source: Path | str) -> str:
    """
    Which convention a POSCAR is written in.

    Read off the file rather than the parsed object, because pymatgen
    normalises to fractional on the way in and the original wording is
    the only record of what the file said.
    """

    path = Path(source)

    try:
        lines = path.read_text(errors="replace").splitlines()

    except OSError as exc:
        raise ConversionError(f"Could not read {path.name}: {exc}") from exc

    for line in lines[5:10]:

        word = line.strip().lower()

        if not word:
            continue

        if word.startswith("s"):
            # Selective dynamics; the mode is on the next line.
            continue

        if word.startswith(("d", "c", "k")):
            # VASP accepts anything beginning with C or K as Cartesian.
            return DIRECT if word.startswith("d") else CARTESIAN

    raise ConversionError(
        f"{path.name} has no coordinate mode line where a POSCAR "
        f"should have one. Is it really a POSCAR?"
    )


# ------------------------------------------------------------
# Running one
# ------------------------------------------------------------


@dataclass
class Result:
    """What a conversion did."""

    source: Path
    target: Path
    conversion: str

    natoms: int = 0
    formula: str = ""

    #: True when a cell was built rather than read.
    invented_cell: bool = False
    vacuum: float = 0.0

    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:

        if self.notes is None:
            self.notes = []


def run(
    key: str,
    source: Path | str,
    target: Path | str,
    vacuum: float = DEFAULT_VACUUM,
) -> Result:
    """Do one named conversion."""

    conversion = find(key)

    source = Path(source)
    target = Path(target)

    if not source.is_file():
        raise ConversionError(f"There is no such file: {source}")

    result = Result(
        source=source, target=target, conversion=conversion.label
    )

    # ----------------------------------------------------------
    # The two that only change coordinates
    # ----------------------------------------------------------

    if conversion.coordinates:

        was = coordinate_mode(source)

        if was == conversion.coordinates:
            result.notes.append(
                f"{source.name} is already written in "
                f"{conversion.coordinates} coordinates. The copy is "
                f"identical apart from formatting."
            )

        text = rewrite_coordinates(source, conversion.coordinates)

        _write(target, text)

        from pymatgen.io.vasp.inputs import Poscar

        poscar = Poscar.from_file(str(source), check_for_potcar=False)

        result.natoms = len(poscar.structure)
        result.formula = poscar.structure.composition.reduced_formula

        return result

    # ----------------------------------------------------------
    # Gaussian in
    # ----------------------------------------------------------

    if key.startswith("gjf"):

        molecule, lattice = read_gjf(source)

        structure, invented = molecule_to_structure(
            molecule, lattice, vacuum
        )

        result.invented_cell = invented
        result.natoms = len(structure)
        result.formula = structure.composition.reduced_formula

        if invented:
            result.vacuum = vacuum
            result.notes.append(
                f"WARNING: {source.name} carries no lattice, so one was "
                f"built: an orthorhombic box with {vacuum:g} A of vacuum "
                f"on each side, giving {2 * vacuum:g} A between periodic "
                f"images. That spacing is a modelling decision and it "
                f"has not been converged -- check it before running "
                f"anything on this cell."
            )
        else:
            result.notes.append(
                "The cell came from the Tv translation vectors in the "
                "file."
            )

        from pymatgen.io.vasp.inputs import Poscar

        _write(target, str(Poscar(structure)))

        return result

    # ----------------------------------------------------------
    # Everything else goes through the writer
    # ----------------------------------------------------------

    from qvista.services import structure_writer

    fmt = {
        "cif_to_poscar": "poscar",
        "poscar_to_cif": "cif",
        "poscar_to_xyz": "xyz",
    }[key]

    try:
        written = structure_writer.convert(source, target, fmt)

    except structure_writer.StructureWriteError as exc:
        raise ConversionError(str(exc)) from exc

    result.target = written

    from qvista.services.structure_analyzer import StructureAnalyzer

    parsed = StructureAnalyzer.parse(source)

    result.natoms = len(parsed.structure)
    result.formula = parsed.structure.composition.reduced_formula

    if conversion.note:
        result.notes.append(conversion.note)

    return result


def _write(target: Path, text: str) -> None:
    """Write text out, reporting a failure the way a user needs it."""

    try:
        target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(text, encoding="utf-8")

    except OSError as exc:
        raise ConversionError(
            f"Could not write {target.name}: {exc}"
        ) from exc

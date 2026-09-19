"""
Tests for structure format conversion.

Most of these are round trips, because a conversion has exactly one job
and a round trip is the only check that it did it: a file that parses,
has the right number of atoms and the right formula can still have every
atom in the wrong place.

The two that need more care are the ones that are not really format
conversions. ``Direct <-> Cartesian`` changes only how coordinates are
written, so the structure must come back bit for bit. And a Gaussian
input has no cell at all, so the conversion to POSCAR has to invent one
-- and say that it did.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvista.services import convert as cv

WATER = """%chk=water.chk
#p B3LYP/6-31G(d) Opt

water molecule

0 1
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200

"""

#: Gaussian's own periodic notation: Tv rows among the atoms.
#:
#: H2 rather than anything larger, because pymatgen checks that the
#: charge and multiplicity are possible for the electron count -- an
#: invalid pair is refused before the Tv handling is ever reached.
PERIODIC_GJF = """#p PBEPBE/6-31G(d)

hydrogen in a box

0 1
H   0.000000   0.000000   0.000000
H   0.740000   0.000000   0.000000
Tv  8.000000   0.000000   0.000000
Tv  0.000000   9.000000   0.000000
Tv  0.000000   0.000000  10.000000

"""


def crystal():
    """A small periodic structure with a non-orthogonal cell."""

    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.from_parameters(4.0, 4.2, 6.1, 90.0, 90.0, 120.0),
        ["Li", "Co", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.31],
            [0.75, 0.75, 0.69],
        ],
    )


def write_poscar(tmp_path: Path, name: str = "POSCAR") -> Path:

    from pymatgen.io.vasp.inputs import Poscar

    path = tmp_path / name

    Poscar(crystal()).write_file(str(path))

    return path


def write_gjf(tmp_path: Path, text: str = WATER, name="in.gjf") -> Path:

    path = tmp_path / name

    path.write_text(text, encoding="utf-8")

    return path


def read_back(path: Path):

    from pymatgen.io.vasp.inputs import Poscar

    return Poscar.from_file(str(path), check_for_potcar=False).structure


# ------------------------------------------------------------
# The list itself
# ------------------------------------------------------------


def test_every_conversion_asked_for_is_there() -> None:

    keys = {item.key for item in cv.CONVERSIONS}

    assert keys == {
        "gjf_to_vasp",
        "gjf_to_poscar",
        "cif_to_poscar",
        "poscar_to_cif",
        "poscar_to_xyz",
        "to_cartesian",
        "to_direct",
    }


def test_each_one_can_be_looked_up() -> None:

    for item in cv.CONVERSIONS:
        assert cv.find(item.key) is item


def test_an_unknown_key_lists_the_real_ones() -> None:

    with pytest.raises(cv.ConversionError, match="Unknown conversion"):
        cv.find("poscar_to_pdf")


def test_only_the_gaussian_ones_need_a_cell() -> None:
    """
    Everything else reads a format that already carries one.

    The flag drives whether the vacuum control is shown, so a wrong one
    either hides a decision that matters or offers an irrelevant knob.
    """

    needing = {item.key for item in cv.CONVERSIONS if item.needs_cell}

    assert needing == {"gjf_to_vasp", "gjf_to_poscar"}


# ------------------------------------------------------------
# Coordinate conventions
# ------------------------------------------------------------


def test_a_direct_poscar_is_recognised(tmp_path) -> None:

    assert cv.coordinate_mode(write_poscar(tmp_path)) == cv.DIRECT


def test_a_cartesian_poscar_is_recognised(tmp_path) -> None:

    path = tmp_path / "POSCAR_cart"

    path.write_text(
        cv.rewrite_coordinates(write_poscar(tmp_path), cv.CARTESIAN),
        encoding="utf-8",
    )

    assert cv.coordinate_mode(path) == cv.CARTESIAN


def test_the_mode_line_is_found_past_selective_dynamics(
    tmp_path,
) -> None:
    """
    Selective dynamics inserts a line before the mode.

    Reading a fixed line number instead of skipping it finds the word
    "Selective" where the convention should be, and every such POSCAR
    is then reported as neither.
    """

    from pymatgen.io.vasp.inputs import Poscar

    path = tmp_path / "POSCAR_sd"

    Poscar(
        crystal(), selective_dynamics=[[True] * 3] * 4
    ).write_file(str(path))

    assert cv.coordinate_mode(path) == cv.DIRECT


def test_vasp_accepts_k_for_cartesian(tmp_path) -> None:
    """
    A mode line beginning with C *or* K means Cartesian.

    VASP's own rule, and files written by other tools use both.
    """

    path = write_poscar(tmp_path)

    lines = path.read_text().splitlines()
    lines[7] = "Kartesian"

    path.write_text("\n".join(lines), encoding="utf-8")

    assert cv.coordinate_mode(path) == cv.CARTESIAN


def test_a_file_with_no_mode_line_is_refused(tmp_path) -> None:

    path = tmp_path / "not_a_poscar"
    path.write_text("just\nsome\ntext\n", encoding="utf-8")

    with pytest.raises(cv.ConversionError, match="no coordinate mode"):
        cv.coordinate_mode(path)


def test_the_round_trip_moves_nothing(tmp_path) -> None:
    """
    Direct to Cartesian and back must be exact.

    This is the whole content of those two conversions: the same atoms
    in the same places, written differently. A non-orthogonal cell is
    used deliberately -- for a cubic one the transformation is a scalar
    multiply and a wrong lattice matrix would still round-trip.
    """

    source = write_poscar(tmp_path)

    cartesian = tmp_path / "POSCAR_cart"
    back = tmp_path / "POSCAR_back"

    cv.run("to_cartesian", source, cartesian)
    cv.run("to_direct", cartesian, back)

    before = read_back(source)
    after = read_back(back)

    assert np.abs(
        before.frac_coords - after.frac_coords
    ).max() == pytest.approx(0.0, abs=1e-12)

    assert np.abs(
        np.asarray(before.lattice.matrix)
        - np.asarray(after.lattice.matrix)
    ).max() == pytest.approx(0.0, abs=1e-12)


def test_converting_to_the_mode_it_is_already_in_says_so(
    tmp_path,
) -> None:
    """Not an error -- just worth knowing nothing happened."""

    result = cv.run(
        "to_direct", write_poscar(tmp_path), tmp_path / "out"
    )

    assert any("already written" in note for note in result.notes)


def test_an_unknown_mode_is_refused(tmp_path) -> None:

    with pytest.raises(cv.ConversionError, match="Unknown coordinate"):
        cv.rewrite_coordinates(write_poscar(tmp_path), "polar")


# ------------------------------------------------------------
# Gaussian input
# ------------------------------------------------------------


def test_a_gjf_is_read(tmp_path) -> None:

    molecule, lattice = cv.read_gjf(write_gjf(tmp_path))

    assert len(molecule) == 3

    assert lattice is None


def test_translation_vectors_become_the_cell(tmp_path) -> None:
    """
    ``Tv`` is Gaussian's periodic notation.

    Read as an element it becomes two extra "atoms" of an unknown
    species, which then appear in the POSCAR.
    """

    molecule, lattice = cv.read_gjf(
        write_gjf(tmp_path, PERIODIC_GJF, "chain.gjf")
    )

    assert len(molecule) == 2, "Tv rows were counted as atoms"

    assert lattice is not None

    assert np.allclose(np.diag(lattice), [8.0, 9.0, 10.0])


def test_a_gjf_with_a_cell_does_not_invent_one(tmp_path) -> None:

    result = cv.run(
        "gjf_to_poscar",
        write_gjf(tmp_path, PERIODIC_GJF, "chain.gjf"),
        tmp_path / "POSCAR",
    )

    assert not result.invented_cell

    assert any("Tv" in note for note in result.notes)


def test_a_molecule_gets_a_box_and_is_told_so(tmp_path) -> None:
    """
    The vacuum spacing is a modelling decision, not a detail.

    It decides how strongly the molecule sees its own periodic images,
    and a silently chosen one is the kind of thing that ends up in a
    paper.
    """

    result = cv.run(
        "gjf_to_poscar",
        write_gjf(tmp_path),
        tmp_path / "POSCAR",
        vacuum=8.0,
    )

    assert result.invented_cell

    assert result.vacuum == 8.0

    assert any(note.startswith("WARNING") for note in result.notes)


def test_the_box_has_the_vacuum_asked_for(tmp_path) -> None:
    """Water is about 1.5 A across; 8 A each side makes roughly 17."""

    cv.run(
        "gjf_to_poscar",
        write_gjf(tmp_path),
        tmp_path / "POSCAR",
        vacuum=8.0,
    )

    structure = read_back(tmp_path / "POSCAR")

    edges = structure.lattice.abc

    assert min(edges) >= 16.0

    assert max(edges) < 20.0


def test_the_molecule_is_centred_in_its_box(tmp_path) -> None:
    """
    Otherwise the vacuum is all on one side.

    An off-centre molecule in a box is still periodic and still runs,
    and the image separation is then half what was asked for.
    """

    cv.run(
        "gjf_to_poscar",
        write_gjf(tmp_path),
        tmp_path / "POSCAR",
        vacuum=8.0,
    )

    structure = read_back(tmp_path / "POSCAR")

    centre = structure.cart_coords.mean(axis=0)

    assert np.allclose(
        centre, 0.5 * np.array(structure.lattice.abc), atol=1.0
    )


def test_a_bigger_vacuum_gives_a_bigger_box(tmp_path) -> None:

    volumes = []

    for vacuum in (5.0, 12.0):

        cv.run(
            "gjf_to_poscar",
            write_gjf(tmp_path),
            tmp_path / f"POSCAR_{vacuum:g}",
            vacuum=vacuum,
        )

        volumes.append(read_back(tmp_path / f"POSCAR_{vacuum:g}").volume)

    assert volumes[1] > volumes[0]


def test_negative_vacuum_is_refused(tmp_path) -> None:

    molecule, _ = cv.read_gjf(write_gjf(tmp_path))

    with pytest.raises(cv.ConversionError, match="cannot be negative"):
        cv.box_around(molecule, vacuum=-1.0)


def test_a_single_atom_still_gets_a_real_cell(tmp_path) -> None:
    """
    One atom has no extent, and a zero-length lattice vector is not a
    cell -- it is a structure VASP refuses and pymatgen cannot even
    build.
    """

    from pymatgen.core import Molecule

    matrix = cv.box_around(Molecule(["Ne"], [[0.0, 0.0, 0.0]]), 6.0)

    assert np.all(np.diag(matrix) > 0.0)


def test_a_file_that_is_not_gaussian_is_refused(tmp_path) -> None:

    path = tmp_path / "broken.gjf"
    path.write_text("this is not a Gaussian input\n", encoding="utf-8")

    with pytest.raises(cv.ConversionError):
        cv.read_gjf(path)


def test_a_missing_file_is_refused(tmp_path) -> None:

    with pytest.raises(cv.ConversionError, match="no such file"):
        cv.read_gjf(tmp_path / "nowhere.gjf")


# ------------------------------------------------------------
# The format conversions
# ------------------------------------------------------------


def test_poscar_to_cif_and_back(tmp_path) -> None:
    """A CIF round trip should preserve the structure."""

    source = write_poscar(tmp_path)

    cv.run("poscar_to_cif", source, tmp_path / "out.cif")

    result = cv.run(
        "cif_to_poscar", tmp_path / "out.cif", tmp_path / "POSCAR_2"
    )

    before = read_back(source)

    # Read the path the conversion reports, not the one asked for: a
    # name that is not exactly POSCAR or CONTCAR gains ".vasp" so the
    # file can be detected when it is reopened.
    after = read_back(result.target)

    assert before.composition == after.composition

    assert before.volume == pytest.approx(after.volume, rel=1e-4)


def test_poscar_to_xyz_keeps_the_lattice(tmp_path) -> None:
    """
    Extended XYZ, not plain.

    A plain XYZ drops the cell, and the file then reads back as a
    molecule -- so a round trip through it silently turns a crystal
    into one.
    """

    cv.run(
        "poscar_to_xyz", write_poscar(tmp_path), tmp_path / "out.xyz"
    )

    text = (tmp_path / "out.xyz").read_text()

    assert "Lattice" in text


def test_the_result_reports_what_it_wrote(tmp_path) -> None:

    result = cv.run(
        "poscar_to_cif", write_poscar(tmp_path), tmp_path / "out.cif"
    )

    assert result.natoms == 4

    assert result.formula

    assert result.target.name == "out.cif"


def test_a_missing_source_is_refused(tmp_path) -> None:

    with pytest.raises(cv.ConversionError, match="no such file"):
        cv.run("poscar_to_cif", tmp_path / "nowhere", tmp_path / "out")


def test_an_extension_is_added_when_it_is_missing(tmp_path) -> None:
    """
    So the file QVista wrote can be reopened by QVista.

    Format detection works off the name, and a CIF called "out" is
    written correctly and cannot be read back.
    """

    result = cv.run(
        "poscar_to_cif", write_poscar(tmp_path), tmp_path / "out"
    )

    assert result.target.suffix == ".cif"


# ------------------------------------------------------------
# Choosing the wrong conversion
#
# The easiest mistake on that menu: seven entries, and a source box that
# accepts whatever is typed into it. Handing a CIF to the Gaussian
# reader is what happens, and pymatgen fails on the line it happened to
# reach -- an IndexError, which says nothing about the file. What the
# user needs is the name of the format they actually picked.
# ------------------------------------------------------------


def test_a_cif_given_to_the_gaussian_reader_is_named(tmp_path) -> None:

    path = tmp_path / "out.cif"
    path.write_text(
        "data_LiCoO2\n_cell_length_a 2.8\nloop_\n_atom_site_label\nLi1\n",
        encoding="utf-8",
    )

    with pytest.raises(cv.ConversionError, match="looks like a CIF"):
        cv.read_gjf(path)


def test_a_poscar_given_to_the_gaussian_reader_is_named(tmp_path) -> None:

    with pytest.raises(cv.ConversionError, match="looks like a POSCAR"):
        cv.read_gjf(write_poscar(tmp_path))


def test_an_xyz_given_to_the_gaussian_reader_is_named(tmp_path) -> None:

    path = tmp_path / "water.xyz"
    path.write_text(
        "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\n"
        "H 0 -0.7572 -0.4692\n",
        encoding="utf-8",
    )

    with pytest.raises(cv.ConversionError, match="looks like an XYZ"):
        cv.read_gjf(path)


def test_the_parser_is_still_quoted_when_the_format_is_unknown(
    tmp_path,
) -> None:
    """
    A truncated .gjf really is a .gjf, and there is nothing better to
    say about it than what the parser said.
    """

    path = tmp_path / "cut.gjf"
    path.write_text("%mem=1GB\n#p b3lyp\n", encoding="utf-8")

    with pytest.raises(cv.ConversionError) as caught:
        cv.read_gjf(path)

    message = str(caught.value)

    assert "looks like" not in message

    assert "IndexError" in message


def test_a_real_gjf_is_not_mistaken_for_anything(tmp_path) -> None:
    """The recogniser must not fire on the format it is defending."""

    assert cv._looks_like(WATER) is None

    assert cv._looks_like(PERIODIC_GJF) is None


def test_an_empty_file_is_not_guessed_at(tmp_path) -> None:

    assert cv._looks_like("") is None
    assert cv._looks_like("\n\n   \n") is None


def test_an_xyz_whose_count_is_wrong_is_not_claimed() -> None:
    """
    A leading integer is not enough.

    Plenty of files open with a number, and calling one of them an XYZ
    would send the user to look for a problem that is not there.
    """

    assert cv._looks_like("12\ntitle\nH 0 0 0\n") is None


def test_a_poscar_needs_its_lattice_rows() -> None:
    """A scale factor alone is a number on a line, nothing more."""

    assert cv._looks_like(
        "title\n1.0\nnot a lattice\nnor this\nnor this\nmore\n"
    ) is None


# ------------------------------------------------------------
# Recognising what a conversion reads
# ------------------------------------------------------------
#
# The dialog opens with the loaded structure already filled in, which
# means it has to pick the conversion that reads it. Getting that wrong
# pairs a CIF with the Gaussian parser and reports a parse failure for a
# file that is perfectly good.


def test_the_patterns_come_from_the_browser_filter() -> None:
    """
    One list, not two.

    The patterns offered in the file browser and the patterns used to
    recognise a loaded file are the same list read from the same place,
    so they cannot drift apart as conversions are added.
    """

    assert cv.source_patterns(cv.find("cif_to_poscar")) == ("*.cif",)

    assert cv.source_patterns(cv.find("poscar_to_cif")) == (
        "POSCAR",
        "CONTCAR",
        "*.vasp",
    )


def test_the_catch_all_group_is_not_read() -> None:
    """
    Every filter ends in "All Files (*)", which matches everything.

    Reading it would make every conversion claim every file, and the
    first one in the list would silently win.
    """

    for conversion in cv.CONVERSIONS:
        assert "*" not in cv.source_patterns(conversion)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("POSCAR", "poscar_to_cif"),
        ("CONTCAR", "poscar_to_cif"),
        ("LiMnO2.vasp", "poscar_to_cif"),
        ("structure.cif", "cif_to_poscar"),
        ("water.gjf", "gjf_to_vasp"),
        ("water.com", "gjf_to_vasp"),
    ],
)
def test_a_loaded_file_finds_the_conversion_that_reads_it(
    name: str, expected: str
) -> None:

    match = cv.conversion_for(name)

    assert match is not None
    assert match.key == expected


def test_a_format_nothing_reads_is_said_to_be_nothing() -> None:
    """
    ``None`` rather than a guess.

    Falling back to the first conversion would fill the dialog in with
    a pairing that cannot work, and the failure would arrive out of a
    parser rather than out of the choice that caused it.
    """

    assert cv.conversion_for("trajectory.pdb") is None
    assert cv.conversion_for("INCAR") is None


def test_a_full_path_is_matched_on_its_name(tmp_path) -> None:
    """A conversion reads a file, not the folder it happens to sit in."""

    poscar = tmp_path / "run" / "POSCAR"

    match = cv.conversion_for(poscar)

    assert match is not None
    assert match.key == "poscar_to_cif"

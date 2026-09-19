"""
Tests for the DDEC6 tools.

Chargemol output is plain text, so the fixtures here are the real thing
written out verbatim -- a rocksalt NaCl cell, which is small enough to
write by hand and has bond lengths that are known exactly: the first
neighbour sits at a/2 and the second at a/sqrt(2).  That makes the bond
length reconstruction, which is the part this module adds on top of what
Chargemol prints, checkable against arithmetic rather than against
itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services import ddec as dd


#: Half the conventional cell edge; the fcc primitive vectors are this
#: long along two axes each.
HALF = 2.829447

HEADER = (
    'jmolscript: load "" {1 1 1} spacegroup "x,y,z" unitcell '
    "[{      0.000000     2.829447     2.829447 }, "
    "{      2.829447     0.000000     2.829447 }, "
    "{      2.829447     2.829447     0.000000 }]"
)

CHARGES = f"""    2
{HEADER}
Na     0.000000     0.000000     0.000000     0.843200
Cl     2.829447     2.829447     2.829447    -0.843200

 Chargemol version 3.5 September 26, 2017.

The following XYZ coordinates are in angstroms. The atomic dipoles and quadrupoles are in atomic units.
atom number, atomic symbol, x, y, z, net_charge, dipole_x, dipole_y, dipole_z, dipole_mag, Qxy
    1 Na     0.000000     0.000000     0.000000     0.843200     0.000000     0.000000    -0.000000     0.111000    -0.000003
    2 Cl     2.829447     2.829447     2.829447    -0.843200    -0.000000    -0.000000    -0.000000     0.222000    -0.000020
"""

BONDS = f"""    2
{HEADER}
Na      0.000000      0.000000      0.000000       0.539920
Cl      2.829447      2.829447      2.829447       0.901058

 The sum of bond orders (SBO) for each atom in the unit cell are listed above.

 ===================================================================================
 Printing BOs for ATOM #      1 ( Na ) in the reference unit cell.

 Bonded to the ( -1,  -1,   0) translated image of atom number     2 ( Cl ) with bond order =     0.0882    The average spin polarization of this bonding =     0.0000
 Bonded to the (  0,   0,  -1) translated image of atom number     2 ( Cl ) with bond order =     0.0882    The average spin polarization of this bonding =     0.0000
 The sum of bond orders for this atom is SBO =        0.539920

 ===================================================================================
 Printing BOs for ATOM #      2 ( Cl ) in the reference unit cell.

 Bonded to the (  1,   1,   0) translated image of atom number     1 ( Na ) with bond order =     0.0882    The average spin polarization of this bonding =     0.0000
 Bonded to the (  0,   0,   1) translated image of atom number     1 ( Na ) with bond order =     0.0882    The average spin polarization of this bonding =     0.0000
 Bonded to the (  0,   0,   1) translated image of atom number     2 ( Cl ) with bond order =     0.0306    The average spin polarization of this bonding =     0.0000
 The sum of bond orders for this atom is SBO =        0.901058
"""

SPINS = f"""    2
{HEADER}
Na     0.000000     0.000000     0.000000     0.150000
Cl     2.829447     2.829447     2.829447    -0.050000
"""

MOLECULE = """    2
Nonperiodic system
O    0.000000   0.000000   0.000000   -0.400000
H    0.000000   0.000000   0.980000    0.400000
"""


def write(folder, charges=CHARGES, bonds=BONDS, spins=None, molecule=False):
    """Lay out a folder the way Chargemol leaves one."""

    if charges is not None:
        (folder / "DDEC6_even_tempered_net_atomic_charges.xyz").write_text(
            MOLECULE if molecule else charges, encoding="utf-8"
        )

    if bonds is not None and not molecule:
        (folder / "DDEC6_even_tempered_bond_orders.xyz").write_text(
            bonds, encoding="utf-8"
        )

    if spins is not None:
        (folder / "DDEC6_even_tempered_atomic_spin_moments.xyz").write_text(
            spins, encoding="utf-8"
        )

    return folder


# ------------------------------------------------------------
# Finding files
# ------------------------------------------------------------


def test_the_files_present_are_reported(tmp_path):

    write(tmp_path)

    found = dd.available(tmp_path)

    assert set(found) == {"charges", "bonds"}
    assert found["charges"].name.endswith("net_atomic_charges.xyz")


def test_an_empty_file_does_not_count(tmp_path):
    """A run killed part way leaves the name behind with nothing in it."""

    write(tmp_path, bonds=None)

    (tmp_path / "DDEC6_even_tempered_bond_orders.xyz").write_text("")

    assert "bonds" not in dd.available(tmp_path)


def test_ddec3_output_is_read_as_a_fallback(tmp_path):
    """The older method writes the same layout under another name."""

    (tmp_path / "DDEC3_net_atomic_charges.xyz").write_text(
        CHARGES, encoding="utf-8"
    )

    result = dd.read(tmp_path)

    assert result.method == "DDEC3"
    assert len(result.atoms) == 2


def test_a_folder_with_nothing_in_it_says_what_is_missing(tmp_path):

    with pytest.raises(dd.DdecError, match="net atomic charges"):
        dd.read(tmp_path)


def test_a_folder_that_is_not_there_says_so(tmp_path):

    with pytest.raises(dd.DdecError, match="not a folder"):
        dd.read(tmp_path / "nowhere")


# ------------------------------------------------------------
# Charges
# ------------------------------------------------------------


def test_charges_come_out_of_the_xyz_block(tmp_path):

    result = dd.read(write(tmp_path))

    assert [atom.symbol for atom in result.atoms] == ["Na", "Cl"]
    assert result.atoms[0].charge == pytest.approx(0.8432)
    assert result.atoms[1].charge == pytest.approx(-0.8432)
    assert result.total_charge == pytest.approx(0.0)


def test_the_lattice_survives_the_supercell_block(tmp_path):
    """
    The comment line opens with ``{1 1 1}`` before it gets to the cell.

    Reading every brace pair on that line finds four vectors, not three,
    which is how a lattice that is perfectly well formed ends up being
    rejected.
    """

    result = dd.read(write(tmp_path))

    assert result.periodic
    assert np.allclose(
        result.lattice,
        [[0, HALF, HALF], [HALF, 0, HALF], [HALF, HALF, 0]],
    )


def test_a_molecule_has_no_lattice(tmp_path):

    result = dd.read(write(tmp_path, molecule=True))

    assert not result.periodic
    assert result.lattice is None
    assert len(result.atoms) == 2


def test_dipole_magnitudes_come_from_the_wide_table(tmp_path):

    result = dd.read(write(tmp_path))

    assert result.atoms[0].dipole == pytest.approx(0.111)
    assert result.atoms[1].dipole == pytest.approx(0.222)


def test_spin_moments_are_read_when_the_run_was_polarised(tmp_path):

    result = dd.read(write(tmp_path, spins=SPINS))

    assert result.atoms[0].spin == pytest.approx(0.15)
    assert result.atoms[1].spin == pytest.approx(-0.05)


def test_without_a_spin_file_there_is_no_spin(tmp_path):

    result = dd.read(write(tmp_path))

    assert all(atom.spin is None for atom in result.atoms)


def test_a_mismatched_spin_file_is_ignored_not_trusted(tmp_path):
    """Pairing the wrong file up atom by atom would be worse than none."""

    write(tmp_path, spins=SPINS.replace("Cl ", "Br "))

    result = dd.read(tmp_path)

    assert all(atom.spin is None for atom in result.atoms)
    assert any("spin moments file" in note for note in result.notes)


def test_a_charged_cell_is_pointed_out(tmp_path):

    write(tmp_path, charges=CHARGES.replace("-0.843200", "-0.343200"))

    result = dd.read(tmp_path)

    assert any("add up to" in note for note in result.notes)


def test_a_short_atom_line_is_refused(tmp_path):

    write(tmp_path, charges=f"    2\n{HEADER}\nNa 0.0 0.0\nCl 1.0 1.0 1.0 -0.5\n")

    with pytest.raises(dd.DdecError, match="short atom line"):
        dd.read(tmp_path)


def test_a_wrong_atom_count_is_refused(tmp_path):

    write(tmp_path, charges=CHARGES.replace("    2\n", "    3\n", 1))

    with pytest.raises(dd.DdecError, match="promises 3"):
        dd.read(tmp_path)


# ------------------------------------------------------------
# Bonds
# ------------------------------------------------------------


def test_bond_orders_and_sums_are_read(tmp_path):

    result = dd.read(write(tmp_path))

    assert len(result.bonds) == 5
    assert result.atoms[0].sbo == pytest.approx(0.539920)
    assert result.atoms[1].sbo == pytest.approx(0.901058)
    assert sorted({round(bond.order, 6) for bond in result.bonds}) == [
        0.0306,
        0.0882,
    ]


def test_the_bond_length_is_rebuilt_from_the_image(tmp_path):
    """
    The number Chargemol does not print.

    In rocksalt the first neighbour is at a/2 and the second at
    a/sqrt(2).  Both bonds here cross a cell face, so ignoring the image
    vector would give a completely different -- and wrong -- distance.
    """

    result = dd.read(write(tmp_path))

    first = [bond for bond in result.bonds if bond.pair == "Cl-Na"]
    second = [bond for bond in result.bonds if bond.pair == "Cl-Cl"]

    assert first
    assert all(bond.length == pytest.approx(HALF, abs=1e-4) for bond in first)

    assert second
    assert all(
        bond.length == pytest.approx(HALF * np.sqrt(2), abs=1e-4)
        for bond in second
    )


def test_ignoring_the_image_would_give_the_wrong_length():
    """The guard on the test above: the two answers really do differ."""

    a = np.zeros(3)
    b = np.array([HALF, HALF, HALF])
    lattice = np.array([[0, HALF, HALF], [HALF, 0, HALF], [HALF, HALF, 0]])

    with_image = dd.bond_length(a, b, (-1, -1, 0), lattice)
    without = dd.bond_length(a, b, (0, 0, 0), lattice)

    assert with_image == pytest.approx(HALF, abs=1e-6)
    assert without == pytest.approx(HALF * np.sqrt(3), abs=1e-6)
    assert not np.isclose(with_image, without)


def test_a_molecule_needs_no_image():

    a = np.zeros(3)
    b = np.array([0.0, 0.0, 0.98])

    assert dd.bond_length(a, b, (0, 0, 0), None) == pytest.approx(0.98)


def test_the_columns_may_be_padded_or_not():
    """
    Chargemol pads its columns to fit the numbers in them.

    An atom number that needs an extra digit eats the space before it, so
    a parser that splits on whitespace and counts fields reads the wrong
    column as soon as a structure gets big enough. Both spacings have to
    give the same answer.
    """

    spaced = (
        " Bonded to the ( -1,  -1,   0) translated image of atom number"
        "     2 ( Cl ) with bond order =     0.0882    The average spin"
        " polarization of this bonding =     0.0000"
    )
    tight = (
        " Bonded to the (-11,-1,0) translated image of atom number"
        " 1234 (Cl) with bond order = 0.0882    The average spin"
        " polarization of this bonding = 0.0000"
    )

    first = dd._BONDED.search(spaced)
    second = dd._BONDED.search(tight)

    assert first and second

    assert first.groups()[:3] == ("-1", "-1", "0")
    assert first.group(4) == "2"
    assert float(first.group(6)) == pytest.approx(0.0882)

    assert second.groups()[:3] == ("-11", "-1", "0")
    assert second.group(4) == "1234"
    assert float(second.group(6)) == pytest.approx(0.0882)


def test_a_bond_naming_an_atom_that_is_not_there_is_refused(tmp_path):

    write(tmp_path, bonds=BONDS.replace("atom number     2", "atom number    99"))

    with pytest.raises(dd.DdecError, match="99"):
        dd.read(tmp_path)


def test_without_a_bonds_file_the_charges_still_load(tmp_path):

    result = dd.read(write(tmp_path, bonds=None))

    assert len(result.atoms) == 2
    assert result.bonds == []
    assert all(atom.sbo is None for atom in result.atoms)
    assert any("No bond orders file" in note for note in result.notes)


def test_each_bond_is_listed_once_from_each_end(tmp_path):
    """That is how Chargemol prints them, and it double counts."""

    result = dd.read(write(tmp_path))

    na_cl = [bond for bond in result.bonds if bond.pair == "Cl-Na"]

    assert len(na_cl) == 4
    assert len(dd.unique_bonds(na_cl)) == 2


def test_deduplication_matches_a_bond_to_its_reverse():
    """
    Seen from the other end the atoms swap and the image negates.

    Without negating the image, the reverse of a bond looks like a
    different bond and nothing is ever deduplicated.
    """

    forward = dd.Bond(
        a=1, a_symbol="Na", b=2, b_symbol="Cl", image=(0, 0, 1),
        order=0.5, length=2.8,
    )
    backward = dd.Bond(
        a=2, a_symbol="Cl", b=1, b_symbol="Na", image=(0, 0, -1),
        order=0.5, length=2.8,
    )

    assert len(dd.unique_bonds([forward, backward])) == 1


def test_two_genuinely_different_bonds_both_survive():

    first = dd.Bond(
        a=1, a_symbol="Na", b=2, b_symbol="Cl", image=(0, 0, 1),
        order=0.5, length=2.8,
    )
    second = dd.Bond(
        a=1, a_symbol="Na", b=2, b_symbol="Cl", image=(0, 1, 0),
        order=0.5, length=2.8,
    )

    assert len(dd.unique_bonds([first, second])) == 2


def test_weak_bonds_can_be_cut_off(tmp_path):

    result = dd.read(write(tmp_path))

    assert len(dd.bonds_above(result, 0.0)) == 5
    assert len(dd.bonds_above(result, 0.05)) == 4
    assert dd.bonds_above(result, 1.0) == []


def test_bonds_come_back_strongest_first(tmp_path):

    result = dd.read(write(tmp_path))

    orders = [bond.order for bond in dd.bonds_above(result, 0.0)]

    assert orders == sorted(orders, reverse=True)


def test_a_pair_is_named_the_same_way_from_either_end():
    """Or the same kind of bond lands in two different rows."""

    forward = dd.Bond(
        a=1, a_symbol="Na", b=2, b_symbol="Cl", image=(0, 0, 0),
        order=0.5, length=2.8,
    )
    backward = dd.Bond(
        a=2, a_symbol="Cl", b=1, b_symbol="Na", image=(0, 0, 0),
        order=0.5, length=2.8,
    )

    assert forward.pair == backward.pair == "Cl-Na"


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------


def test_elements_are_summed(tmp_path):

    rows = dd.by_element(dd.read(write(tmp_path, spins=SPINS)))

    assert [row.label for row in rows] == ["Na", "Cl"]

    sodium = rows[0]

    assert sodium.count == 1
    assert sodium.charge == pytest.approx(0.8432)
    assert sodium.charge_per_atom == pytest.approx(0.8432)
    assert sodium.spin == pytest.approx(0.15)
    assert sodium.sbo == pytest.approx(0.539920)


def test_bond_types_summarise_order_and_length(tmp_path):

    result = dd.read(write(tmp_path))

    rows = dd.by_bond_type(dd.unique_bonds(dd.bonds_above(result, 0.0)))

    assert [row.pair for row in rows] == ["Cl-Na", "Cl-Cl"]

    strong = rows[0]

    assert strong.count == 2
    assert strong.order_mean == pytest.approx(0.0882)
    assert strong.order_min == strong.order_max == pytest.approx(0.0882)
    assert strong.length_mean == pytest.approx(HALF, abs=1e-4)

    assert rows[1].length_mean == pytest.approx(HALF * np.sqrt(2), abs=1e-4)


def test_bond_types_come_back_strongest_first(tmp_path):

    result = dd.read(write(tmp_path))

    rows = dd.by_bond_type(dd.unique_bonds(dd.bonds_above(result, 0.0)))

    assert [row.order_mean for row in rows] == sorted(
        [row.order_mean for row in rows], reverse=True
    )


# ------------------------------------------------------------
# The structure the viewer draws
# ------------------------------------------------------------


def test_a_periodic_run_gives_a_structure(tmp_path):

    structure = dd.read(write(tmp_path)).structure()

    assert len(structure) == 2
    assert structure.lattice.volume == pytest.approx(
        abs(np.linalg.det([[0, HALF, HALF], [HALF, 0, HALF], [HALF, HALF, 0]]))
    )


def test_a_molecule_gives_a_molecule(tmp_path):

    from pymatgen.core import Molecule

    assert isinstance(
        dd.read(write(tmp_path, molecule=True)).structure(), Molecule
    )


# ------------------------------------------------------------
# Running Chargemol
# ------------------------------------------------------------


def test_job_control_carries_what_chargemol_needs(tmp_path):

    text = dd.write_job_control(tmp_path, tmp_path / "densities").read_text()

    assert "<net charge>" in text
    assert "<atomic densities directory complete path>" in text
    assert "DDEC6" in text
    assert "<compute BOs>" in text


def test_the_densities_path_gets_its_trailing_separator(tmp_path):
    """Chargemol fails obscurely without one."""

    text = dd.write_job_control(tmp_path, "/home/me/atomic_densities").read_text()

    assert "/home/me/atomic_densities/" in text


def test_periodicity_is_written_per_axis(tmp_path):

    text = dd.write_job_control(
        tmp_path, tmp_path, periodic=(True, True, False)
    ).read_text()

    block = text.split("<periodicity along A, B, and C vectors>")[1]
    block = block.split("</periodicity")[0]

    assert block.split() == [".true.", ".true.", ".false."]


def test_bond_orders_can_be_turned_off(tmp_path):

    text = dd.write_job_control(
        tmp_path, tmp_path, bond_orders=False
    ).read_text()

    assert ".false." in text.split("<compute BOs>")[1]


def test_running_without_chargemol_says_so(tmp_path):

    with pytest.raises(dd.DdecError, match="Chargemol is not at"):
        dd.run(tmp_path, str(tmp_path / "nothing"), tmp_path)


def test_running_without_the_densities_says_so(tmp_path):

    exe = tmp_path / "chargemol"
    exe.write_text("")

    with pytest.raises(dd.DdecError, match="atomic densities"):
        dd.run(tmp_path, str(exe), tmp_path / "nowhere")


def test_running_without_the_densities_files_names_them(tmp_path):

    exe = tmp_path / "chargemol"
    exe.write_text("")

    with pytest.raises(dd.DdecError, match="CHGCAR"):
        dd.run(tmp_path, str(exe), tmp_path)

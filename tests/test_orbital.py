"""
Tests for the Orbital tools.

A real VASP run with ``LORBIT = 11`` and ``LWAVE = .TRUE.`` is a large
binary thing, so the run here is synthesised: :func:`write_vasprun`
emits the smallest ``vasprun.xml`` pymatgen will accept that still
carries eigenvalues, occupations and a full nine-channel projection.
That keeps the file-reading path under test instead of stubbed, which
is where a pymatgen upgrade would break things.

The WAVECAR is faked outright -- its format is a Fortran binary that
nothing here needs to exercise -- but the transform applied to it is
real, so the sign of a real wavefunction and the positivity of a
squared one are genuine assertions.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pymatgen.core import Lattice, Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp import Poscar

from qvista.services import orbital as ob


ORB = list(ob.ORBITALS)


# ------------------------------------------------------------
# A synthetic run
# ------------------------------------------------------------


def _i(name, val, typ=None):
    kind = f' type="{typ}"' if typ else ""
    return f'<i{kind} name="{name}">{val}</i>'


def _structure_block(name, positions, a=4.0):
    out = [f'<structure name="{name}"><crystal><varray name="basis">']

    for vector in np.eye(3) * a:
        out.append("<v> %s </v>" % " ".join(f"{x:.8f}" for x in vector))

    out.append(
        f'</varray><i name="volume">{a ** 3:.8f}</i>'
        '<varray name="rec_basis">'
    )

    for vector in np.eye(3) / a:
        out.append("<v> %s </v>" % " ".join(f"{x:.8f}" for x in vector))

    out.append('</varray></crystal><varray name="positions">')

    for point in positions:
        out.append("<v> %s </v>" % " ".join(f"{x:.8f}" for x in point))

    return "\n".join(out + ["</varray></structure>"])


def write_vasprun(
    path: Path,
    elements: tuple[str, ...] = ("Co", "O"),
    nk: int = 2,
    nb: int = 3,
    projected: bool = True,
    efermi: float = 0.0,
    seed: int = 0,
    metal: bool = False,
) -> dict[int, np.ndarray] | None:
    """
    Write a parseable ``vasprun.xml`` and return the projections written.

    Returning the array is what lets a test assert the file round-trips
    rather than merely that something was parsed.
    """

    nion = len(elements)

    positions = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
    ][:nion]

    rows = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>',
        "<modeling>",
        "<generator>",
        _i("program", " vasp ", "string"),
        _i("version", " 6.3.2 ", "string"),
        "</generator>",
        "<incar>",
        _i("LORBIT", " 11", "int"),
        "</incar>",
        "<parameters>",
        '<separator name="electronic">',
        _i("NBANDS", f" {nb}", "int"),
        _i("ISPIN", " 2", "int"),
        _i("NELM", " 60", "int"),
        _i("NSW", " 0", "int"),
        _i("IBRION", " -1", "int"),
        _i("LCHIMAG", " F ", "logical"),
        _i("LNONCOLLINEAR", " F ", "logical"),
        _i("LSORBIT", " F ", "logical"),
        "</separator>",
        "</parameters>",
        "<kpoints>",
        '<varray name="kpointlist">',
    ]

    for k in range(nk):
        rows.append(f"<v> {0.5 * k:.8f} 0.00000000 0.00000000 </v>")

    rows += ["</varray>", '<varray name="weights">']
    rows += [f"<v> {1.0 / nk:.8f} </v>"] * nk
    rows += ["</varray>", "</kpoints>"]

    rows.append(f"<atominfo><atoms>{nion}</atoms><types>{nion}</types>")
    rows.append(
        '<array name="atoms"><dimension dim="1">ion</dimension>'
        '<field type="string">element</field>'
        '<field type="int">atomtype</field><set>'
    )

    for index, element in enumerate(elements, start=1):
        rows.append(f"<rc><c>{element}</c><c>{index}</c></rc>")

    rows.append("</set></array>")
    rows.append(
        '<array name="atomtypes"><dimension dim="1">type</dimension>'
        '<field type="int">atomspertype</field>'
        '<field type="string">element</field>'
        "<field>mass</field><field>valence</field>"
        '<field type="string">pseudopotential</field><set>'
    )

    for element in elements:
        rows.append(
            f"<rc><c>   1</c><c>{element}</c><c>  50.00</c>"
            f"<c>   6.00</c><c> PAW_PBE {element} 01Jan2000</c></rc>"
        )

    rows.append("</set></array></atominfo>")

    rows.append(_structure_block("initialpos", positions))
    rows.append("<calculation>")

    for energy in (-10.0, -11.0, -11.5):
        rows.append(
            f'<scstep><energy><i name="e_fr_energy"> {energy:.8f} </i>'
            f'<i name="e_wo_entrp"> {energy:.8f} </i>'
            f'<i name="e_0_energy"> {energy:.8f} </i></energy></scstep>'
        )

    rows.append(_structure_block("finalpos_inner", positions))
    rows.append(
        '<energy><i name="e_fr_energy"> -11.50000000 </i>'
        '<i name="e_wo_entrp"> -11.50000000 </i>'
        '<i name="e_0_energy"> -11.50000000 </i></energy>'
    )
    rows.append(f'<dos><i name="efermi"> {efermi:.8f} </i></dos>')

    rows.append(
        '<eigenvalues><array><dimension dim="1">band</dimension>'
        '<dimension dim="2">kpoint</dimension>'
        '<dimension dim="3">spin</dimension>'
        "<field>eigene</field><field>occ</field><set>"
    )

    for spin in range(2):

        rows.append(f'<set comment="spin {spin + 1}">')

        for k in range(nk):

            rows.append(f'<set comment="kpoint {k + 1}">')

            for b in range(nb):

                energy = -2.0 + b + 0.5 * k + 0.3 * spin
                occupation = 1.0 if energy < 0 else 0.0

                if metal and spin == 0 and k == 0 and b == 1:
                    occupation = 0.5058

                rows.append(f"<r> {energy:.8f} {occupation:.8f} </r>")

            rows.append("</set>")

        rows.append("</set>")

    rows.append("</set></array></eigenvalues>")

    written: dict[int, np.ndarray] | None = None

    if projected:

        generator = np.random.default_rng(seed)
        written = {}

        rows.append(
            '<projected><array><dimension dim="1">ion</dimension>'
            '<dimension dim="2">band</dimension>'
            '<dimension dim="3">kpoint</dimension>'
            '<dimension dim="4">spin</dimension>'
        )
        rows += [f"<field>{name}</field>" for name in ORB]
        rows.append("<set>")

        for spin in range(2):

            rows.append(f'<set comment="spin{spin + 1}">')

            values = generator.random((nk, nb, nion, len(ORB))).round(4)
            written[spin] = values

            for k in range(nk):

                rows.append(f'<set comment="kpoint {k + 1}">')

                for b in range(nb):

                    rows.append(f'<set comment="band {b + 1}">')

                    for ion in range(nion):
                        rows.append(
                            "<r> %s </r>"
                            % " ".join(
                                f"{x:.4f}" for x in values[k, b, ion]
                            )
                        )

                    rows.append("</set>")

                rows.append("</set>")

            rows.append("</set>")

        rows.append("</set></array></projected>")

    rows += [
        "</calculation>",
        _structure_block("finalpos", positions),
        "</modeling>",
    ]

    path.write_text("\n".join(rows), encoding="utf-8")

    return written


@pytest.fixture
def run_dir(tmp_path):
    """A folder holding a parseable run with projections."""

    warnings.simplefilter("ignore")

    write_vasprun(tmp_path / "vasprun.xml")

    return tmp_path


@pytest.fixture
def run(run_dir):
    """The parsed run itself."""

    return ob.load_run(run_dir, projected=True)


# ------------------------------------------------------------
# Reading the run
# ------------------------------------------------------------


def test_a_run_is_read_with_its_projections(run):

    assert run.projected_eigenvalues is not None
    assert len(run.final_structure) == 2
    assert len(run.actual_kpoints) == 2
    assert ob.get_spin_keys(run.eigenvalues) == [Spin.up, Spin.down]


def test_the_projection_round_trips_through_the_file(tmp_path):
    """What comes back out is what was written in."""

    warnings.simplefilter("ignore")

    written = write_vasprun(tmp_path / "vasprun.xml")

    parsed = ob.load_run(tmp_path, projected=True)

    state = ob.find_state(parsed, "vbm")

    channel = 0 if state.spin == Spin.up else 1

    expected = written[channel][state.kpoint - 1, state.band - 1]

    assert np.allclose(ob.projection_for_state(parsed, state), expected)


def test_a_missing_vasprun_names_the_folder(tmp_path):

    with pytest.raises(ob.OrbitalError, match="No vasprun.xml"):
        ob.load_run(tmp_path)


def test_an_unparseable_vasprun_is_reported_rather_than_raised(tmp_path):
    """
    A truncated run should say so, not surface an XML exception.

    An interrupted job leaves exactly this behind, so it is a normal
    thing to point the dialog at by accident.
    """

    (tmp_path / "vasprun.xml").write_text(
        "<modeling><generator>", encoding="utf-8"
    )

    with pytest.raises(ob.OrbitalError, match="could not be parsed"):
        ob.load_run(tmp_path)


def test_the_fermi_energy_falls_back_to_the_outcar(run_dir):
    """
    Parsing skips the DOS for speed, which is where vasprun keeps the
    Fermi energy, so OUTCAR is the usual source.
    """

    (run_dir / "OUTCAR").write_text(
        " E-fermi :   4.9450   XC(G=0)\n", encoding="utf-8"
    )

    assert ob.load_run(run_dir).efermi == pytest.approx(4.9450)


def test_the_last_fermi_energy_in_an_outcar_wins(tmp_path):
    """A relaxation writes one per ionic step; only the last converged."""

    (tmp_path / "OUTCAR").write_text(
        " E-fermi :   4.9450\n E-fermi :   5.1234\n", encoding="utf-8"
    )

    assert ob.read_fermi_from_outcar(
        tmp_path / "OUTCAR"
    ) == pytest.approx(5.1234)


def test_a_missing_outcar_is_not_an_error(tmp_path):

    assert ob.read_fermi_from_outcar(tmp_path / "OUTCAR") is None


def test_a_structure_is_read_from_contcar_when_poscar_is_absent(tmp_path):

    structure = Structure(
        Lattice.cubic(4.0), ["Co", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )

    Poscar(structure).write_file(str(tmp_path / "CONTCAR"))

    assert len(ob.load_poscar(tmp_path).structure) == 2


def test_neither_poscar_nor_contcar_is_refused(tmp_path):

    with pytest.raises(ob.OrbitalError, match="Neither POSCAR nor CONTCAR"):
        ob.load_poscar(tmp_path)


# ------------------------------------------------------------
# States
# ------------------------------------------------------------


def test_the_valence_edge_is_the_highest_occupied_state(run):

    state = ob.find_state(run, "vbm")

    assert state.occupation > 0

    energies = [
        s.energy for s in ob.collect_states(run) if s.occupation > 0
    ]

    assert state.energy == pytest.approx(max(energies))


def test_the_conduction_edge_is_the_lowest_unoccupied_state(run):

    state = ob.find_state(run, "cbm")

    assert state.occupation == pytest.approx(0.0)

    energies = [
        s.energy for s in ob.collect_states(run) if s.occupation == 0
    ]

    assert state.energy == pytest.approx(min(energies))


def test_the_edges_can_be_taken_within_one_spin_channel(run):
    """
    The spin-up valence edge need not be the spin-down one, which is
    why the wavefunction resolves a state per channel.
    """

    up, _ = ob.find_spin_vbm_cbm(run, Spin.up)
    down, _ = ob.find_spin_vbm_cbm(run, Spin.down)

    assert up.spin == Spin.up
    assert down.spin == Spin.down


def test_a_named_band_is_returned_as_asked(run):

    state = ob.state_at(run, kpoint=2, band=3, spin=Spin.down)

    assert (state.kpoint, state.band, state.spin) == (2, 3, Spin.down)


def test_a_band_outside_the_run_is_refused(run):

    with pytest.raises(ob.OrbitalError, match="Band must be between"):
        ob.state_at(run, kpoint=1, band=999, spin=Spin.up)


def test_a_kpoint_outside_the_run_is_refused(run):

    with pytest.raises(ob.OrbitalError, match="k-point must be between"):
        ob.state_at(run, kpoint=999, band=1, spin=Spin.up)


def test_an_unknown_state_name_is_refused(run):

    with pytest.raises(ob.OrbitalError, match="vbm, cbm or fermi"):
        ob.find_state(run, "middle")


def test_a_metal_reports_its_smeared_states(tmp_path):
    """
    Partial occupation is how a metal shows up, and saying so is what
    stops the two edge energies being read as a band gap.
    """

    warnings.simplefilter("ignore")

    write_vasprun(tmp_path / "vasprun.xml", metal=True)

    assert ob.partially_occupied(ob.load_run(tmp_path)) != []


def test_a_gapped_run_reports_no_smearing(run):

    assert ob.partially_occupied(run) == []


def test_a_state_labels_itself_for_the_heading(run):

    state = ob.state_at(run, kpoint=1, band=2, spin=Spin.down)

    assert state.label() == "k=1  band=2  spin=DOWN"


def test_a_state_names_its_own_file(run):
    """The reference analyser's convention, which VESTA users expect."""

    state = ob.state_at(run, kpoint=1, band=130 % 3 + 1, spin=Spin.down)

    assert state.filename_stem().startswith("B000")
    assert state.filename_stem().endswith("_K0001_DW")


# ------------------------------------------------------------
# Projections
# ------------------------------------------------------------


def test_projections_are_absent_without_lorbit(tmp_path):
    """
    The message must name LORBIT.  Without it the user sees an empty
    table and no reason, and the tag is not on by default.
    """

    warnings.simplefilter("ignore")

    write_vasprun(tmp_path / "vasprun.xml", projected=False)

    parsed = ob.load_run(tmp_path, projected=True)

    with pytest.raises(ob.OrbitalError, match="LORBIT"):
        ob.projection_for_state(parsed, ob.find_state(parsed, "vbm"))


def test_a_contribution_sums_over_the_chosen_atoms(run):

    structure = run.final_structure
    state = ob.find_state(run, "vbm")

    whole = ob.contribution(
        run, state, structure, range(len(structure)), "All"
    )

    parts = [
        ob.contribution(run, state, structure, [index], str(index))
        for index in range(len(structure))
    ]

    assert np.allclose(
        whole.totals, sum(part.totals for part in parts)
    )


def test_a_contribution_keeps_one_row_per_atom(run):

    structure = run.final_structure

    entry = ob.contribution(
        run, ob.find_state(run, "vbm"), structure, [0, 1], "both"
    )

    assert entry.per_atom.shape == (2, len(ob.ORBITALS))


def test_an_empty_selection_is_refused(run):

    with pytest.raises(ob.OrbitalError, match="No atoms"):
        ob.contribution(
            run, ob.find_state(run, "vbm"), run.final_structure, [], "none"
        )


def test_the_element_table_has_one_row_per_species(run):

    rows = ob.element_table(
        run, ob.find_state(run, "vbm"), run.final_structure
    )

    assert [row.label for row in rows] == ["Co", "O"]


def test_elements_keep_structure_order(tmp_path):
    """
    First-appearance order, so the buttons match the POSCAR rather than
    being alphabetised behind the user's back.
    """

    structure = Structure(
        Lattice.cubic(6.0),
        ["O", "Co", "O", "Mn"],
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    )

    assert ob.elements_of(structure) == ["O", "Co", "Mn"]


def test_a_shell_selects_its_own_columns(run):

    entry = ob.element_table(
        run, ob.find_state(run, "vbm"), run.final_structure
    )[0]

    assert entry.group_names("s") == ("s",)
    assert entry.group_names("p") == ("py", "pz", "px")
    assert entry.group_names("d") == (
        "dxy",
        "dyz",
        "dz2",
        "dxz",
        "dx2-y2",
    )
    assert entry.group_names(ob.ALL_ORBITALS) == ob.ORBITALS
    assert len(entry.group("d")) == 5


def test_an_unknown_shell_is_refused(run):

    entry = ob.element_table(
        run, ob.find_state(run, "vbm"), run.final_structure
    )[0]

    with pytest.raises(ob.OrbitalError, match="Unknown orbital group"):
        entry.group("f")


def test_percentages_within_a_shell_sum_to_one_hundred(run):

    entry = ob.element_table(
        run, ob.find_state(run, "vbm"), run.final_structure
    )[0]

    assert ob.percentages(entry.group("d")).sum() == pytest.approx(100.0)


def test_percentages_of_an_empty_shell_are_zero_not_nan():
    """
    A d state has no p character.  Dividing by that sum would put NaN
    across a whole row of the table.
    """

    result = ob.percentages(np.zeros(3))

    assert np.all(result == 0.0)
    assert not np.isnan(result).any()


# ------------------------------------------------------------
# Atom selection
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("9", [9]),
        ("9,11", [9, 11]),
        ("9-12", [9, 10, 11, 12]),
        ("9-12,18,20", [9, 10, 11, 12, 18, 20]),
        ("20,9", [9, 20]),
        ("9,9", [9]),
    ],
)
def test_atom_ranges_are_parsed(text, expected):

    assert ob.parse_atom_list(text) == expected


def test_a_backwards_range_is_refused():

    with pytest.raises(ob.OrbitalError, match="runs backwards"):
        ob.parse_atom_list("20-9")


def test_nonsense_in_an_atom_list_is_refused():

    with pytest.raises(ob.OrbitalError, match="not an atom number"):
        ob.parse_atom_list("nine")


def test_selecting_an_element_gives_zero_based_indices(run):

    assert ob.select_atoms(run.final_structure, element="O") == [1]


def test_selecting_an_absent_element_is_refused(run):

    with pytest.raises(ob.OrbitalError, match="No Fe atoms"):
        ob.select_atoms(run.final_structure, element="Fe")


def test_an_atom_outside_the_cell_is_refused(run):

    with pytest.raises(ob.OrbitalError, match="outside 1-2"):
        ob.select_atoms(run.final_structure, atoms="7")


def test_selecting_nothing_takes_every_atom(run):

    assert ob.select_atoms(run.final_structure) == [0, 1]


# ------------------------------------------------------------
# Wavefunctions
# ------------------------------------------------------------


class FakeWavecar:
    """A WAVECAR that yields one known plane wave."""

    nk = 2
    nb = 3

    #: Fractional k-points, as pymatgen's Wavecar carries them. The
    #: second is a zone-boundary point, so the Bloch factor is not 1
    #: there and anything that drops it shows up.
    kpoints = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
    ]

    def fft_mesh(self, kpoint, band, spin=0, spinor=0, shift=True):

        mesh = np.zeros((8, 8, 8), dtype=complex)
        mesh[1, 0, 0] = 1 + 2j

        return mesh


@pytest.fixture
def poscar():

    return Poscar(
        Structure(
            Lattice.cubic(4.0),
            ["Co", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
    )


@pytest.fixture
def state():

    return ob.State(
        spin=Spin.down,
        kpoint=1,
        band=2,
        energy=-0.1,
        occupation=1.0,
        kcoord=(0.0, 0.0, 0.0),
    )


@pytest.mark.parametrize(
    "part, stem",
    [
        ("real", "WF_REAL_B0002_K0001_DW.vasp"),
        ("imag", "WF_IMAG_B0002_K0001_DW.vasp"),
        ("square", "WF_SQUARE_B0002_K0001_DW.vasp"),
    ],
)
def test_wavefunction_filenames_follow_the_reference(part, stem, state):

    assert ob.wavefunction_filename(part, state) == stem


def test_an_unknown_component_is_refused(state):

    with pytest.raises(ob.OrbitalError, match="Unknown wavefunction part"):
        ob.wavefunction_filename("gradient", state)


def test_a_real_wavefunction_changes_sign(tmp_path, poscar, state):
    """
    The point of writing the real part rather than the density: it has
    a sign, and the lobes either side of a node are what the picture is
    for.  A file of positive numbers would be a charge density.
    """

    path = ob.write_wavefunction(
        FakeWavecar(), poscar, state, "real", tmp_path / "real.vasp"
    )

    values = np.fft.ifftn(FakeWavecar().fft_mesh(0, 1)).real

    assert path.exists()
    assert values.min() < 0 < values.max()


def test_the_squared_wavefunction_is_never_negative():

    values = np.abs(np.fft.ifftn(FakeWavecar().fft_mesh(0, 1))) ** 2

    assert values.min() >= 0


def test_a_written_wavefunction_is_a_vasp_volumetric_file(
    tmp_path, poscar, state
):
    """It has to be openable by whatever reads CHGCAR, VESTA included."""

    from pymatgen.io.vasp import Chgcar

    path = ob.write_wavefunction(
        FakeWavecar(), poscar, state, "real", tmp_path / "real.vasp"
    )

    assert Chgcar.from_file(str(path)).data["total"].shape == (8, 8, 8)


def test_the_output_folder_is_created(tmp_path, poscar, state):

    path = ob.write_wavefunction(
        FakeWavecar(),
        poscar,
        state,
        "real",
        tmp_path / "made" / "up" / "real.vasp",
    )

    assert path.exists()


def test_a_band_beyond_the_wavecar_is_refused(tmp_path, poscar):

    beyond = ob.State(Spin.up, 1, 99, 0.0, 0.0, (0.0, 0.0, 0.0))

    with pytest.raises(ob.OrbitalError, match="holds 3 bands"):
        ob.write_wavefunction(
            FakeWavecar(), poscar, beyond, "real", tmp_path / "x.vasp"
        )


def test_a_kpoint_beyond_the_wavecar_is_refused(tmp_path, poscar):

    beyond = ob.State(Spin.up, 9, 1, 0.0, 0.0, (0.0, 0.0, 0.0))

    with pytest.raises(ob.OrbitalError, match="holds 2 k-points"):
        ob.write_wavefunction(
            FakeWavecar(), poscar, beyond, "real", tmp_path / "x.vasp"
        )


def test_an_empty_wavecar_names_the_tag_that_causes_it(tmp_path):
    """
    A zero-byte WAVECAR means LWAVE was left false.  Saying "cannot
    parse" would send the user hunting for corruption instead.
    """

    (tmp_path / "WAVECAR").write_bytes(b"")

    with pytest.raises(ob.OrbitalError, match="LWAVE"):
        ob.load_wavecar(tmp_path)


def test_a_missing_wavecar_says_so(tmp_path):

    with pytest.raises(ob.OrbitalError, match="No WAVECAR"):
        ob.load_wavecar(tmp_path)


def test_the_grid_shown_is_the_grid_written(tmp_path, poscar, state):
    """
    The viewer contours the same numbers the file gets.

    They come from one function precisely so a picture cannot show
    something the exported file does not contain.
    """

    from pymatgen.io.vasp import Chgcar

    shown = ob.wavefunction_data(FakeWavecar(), poscar, state, "real")

    path = ob.write_wavefunction(
        FakeWavecar(), poscar, state, "real", tmp_path / "real.vasp"
    )

    assert np.allclose(shown, Chgcar.from_file(str(path)).data["total"])


def test_grid_data_is_range_checked_like_the_writer(poscar):

    beyond = ob.State(Spin.up, 1, 99, 0.0, 0.0, (0.0, 0.0, 0.0))

    with pytest.raises(ob.OrbitalError, match="holds 3 bands"):
        ob.wavefunction_data(FakeWavecar(), poscar, beyond, "real")


class StubRun:
    """
    A run with the occupations stated outright.

    The generated fixture shifts both spin channels by the same amount,
    so their edges land on the same band by coincidence.  A test about
    the channels differing has to say so explicitly rather than rely on
    that.
    """

    def __init__(self, occupied_up: int, occupied_down: int, nb: int = 5):

        energies = np.linspace(-4.0, -0.5, nb)

        self.eigenvalues = {}

        for spin, filled in (
            (Spin.up, occupied_up),
            (Spin.down, occupied_down),
        ):
            array = np.zeros((1, nb, 2))
            array[0, :, 0] = energies
            array[0, :, 1] = [
                1.0 if band < filled else 0.0 for band in range(nb)
            ]
            self.eigenvalues[spin] = array

        self.actual_kpoints = [[0.0, 0.0, 0.0]]
        self.efermi = 0.0
        self.projected_eigenvalues = None


def test_the_spin_channels_of_an_edge_can_be_different_bands():
    """
    Why the viewer pins the band to the heading.

    Resolving the edge again inside each spin channel lands on a
    different band, which is a state the table is not describing --
    the fault behind a table reading 78% dz2 next to a picture with no
    dz2 in it.
    """

    run = StubRun(occupied_up=2, occupied_down=4)

    up, _ = ob.find_spin_vbm_cbm(run, Spin.up)
    down, _ = ob.find_spin_vbm_cbm(run, Spin.down)

    assert up.band == 2
    assert down.band == 4

    # The global edge can only be one of them.
    assert ob.find_state(run, "vbm").band == 4


def test_pinning_the_band_keeps_both_channels_on_the_same_state():
    """
    What the viewer does instead: hold k-point and band, vary only the
    channel, so every surface belongs to the state in the heading.
    """

    run = StubRun(occupied_up=2, occupied_down=4)

    heading = ob.find_state(run, "vbm")

    twin = ob.state_at(run, heading.kpoint, heading.band, Spin.up)

    assert (twin.kpoint, twin.band) == (heading.kpoint, heading.band)
    assert twin.spin == Spin.up
    assert twin.spin != heading.spin


def test_the_same_band_exists_in_both_channels(run):
    """
    What the viewer relies on instead: a band index is valid in either
    channel, so the heading's band can be drawn spin-up or spin-down.
    """

    state = ob.find_state(run, "vbm")

    other = Spin.up if state.spin == Spin.down else Spin.down

    twin = ob.state_at(run, state.kpoint, state.band, other)

    assert (twin.kpoint, twin.band) == (state.kpoint, state.band)
    assert twin.spin == other


# ------------------------------------------------------------
# The grid the viewer contours
# ------------------------------------------------------------


def test_the_grid_gains_one_plane_per_axis():
    """
    VASP stores one period.  Without repeating the first plane at the
    far face every lobe crossing a boundary is sliced flat.
    """

    from qvista.gui.dialogs.wavefunction_viewer import wrapped_grid

    grid = wrapped_grid(np.zeros((6, 6, 6)), np.eye(3) * 6.0)

    assert tuple(grid.dimensions) == (7, 7, 7)


def test_the_wrapped_plane_repeats_the_first():

    from qvista.gui.dialogs.wavefunction_viewer import wrapped_grid

    data = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)

    grid = wrapped_grid(data, np.eye(3) * 2.0)

    values = grid["psi"].reshape((3, 3, 3), order="F")

    assert np.allclose(values[0], values[-1])
    assert np.allclose(values[:, 0], values[:, -1])


def test_the_grid_spans_the_whole_cell():

    from qvista.gui.dialogs.wavefunction_viewer import wrapped_grid

    grid = wrapped_grid(np.zeros((4, 4, 4)), np.eye(3) * 5.0)

    assert np.allclose(grid.points.max(axis=0), [5.0, 5.0, 5.0])


def test_a_triclinic_cell_is_sheared_not_boxed():
    """
    The points are put through the lattice, so an oblique cell comes
    out oblique rather than forced into an axis-aligned box.
    """

    from qvista.gui.dialogs.wavefunction_viewer import wrapped_grid

    lattice = np.array([[4.0, 0.0, 0.0], [2.0, 3.5, 0.0], [0.0, 0.0, 5.0]])

    grid = wrapped_grid(np.zeros((4, 4, 4)), lattice)

    assert np.allclose(grid.points.max(axis=0), [6.0, 3.5, 5.0])


def test_a_signed_grid_contours_into_two_lobes():
    """The point of drawing psi rather than its square."""

    from qvista.gui.dialogs.wavefunction_viewer import wrapped_grid

    axis = (np.arange(20) / 20 - 0.5) * 2

    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")

    psi = x * np.exp(-4 * (x**2 + y**2 + z**2))

    grid = wrapped_grid(psi, np.eye(3) * 8.0)

    level = float(np.abs(psi).max()) * 0.3

    assert grid.contour([level], scalars="psi").n_points > 0
    assert grid.contour([-level], scalars="psi").n_points > 0


def test_a_huge_grid_is_strided_down_for_drawing():
    """
    Contouring cost grows with the cube of the grid.  Only the picture
    is thinned; the exported file is never touched.
    """

    from qvista.gui.dialogs.wavefunction_viewer import _stride_for

    assert _stride_for((64, 64, 64)) == 1
    assert _stride_for((300, 300, 300)) > 1


# ------------------------------------------------------------
# Spin
# ------------------------------------------------------------


def test_spin_maps_to_the_wavecar_index():
    """pymatgen's Wavecar takes 0 and 1, not Spin members."""

    assert ob.spin_index(Spin.up) == 0
    assert ob.spin_index(Spin.down) == 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ("up", Spin.up),
        ("U", Spin.up),
        ("down", Spin.down),
        ("dw", Spin.down),
        ("d", Spin.down),
        ("sideways", None),
    ],
)
def test_spin_is_read_in_the_usual_spellings(text, expected):

    assert ob.parse_spin(text) == expected


def test_spin_names_are_stable():
    """These end up in filenames, so they are part of the contract."""

    assert (ob.spin_name(Spin.up), ob.spin_short(Spin.up)) == ("UP", "UP")
    assert (
        ob.spin_name(Spin.down),
        ob.spin_short(Spin.down),
    ) == ("DOWN", "DW")


def test_tuple_keyed_eigenvalues_are_still_readable():
    """
    Some pymatgen versions key by ``(Spin, index)``.  Accepting both
    stops a version difference reading as "not spin polarised".
    """

    array = np.zeros((2, 3, 2))

    mapping = {(Spin.up, 0): array, (Spin.down, 0): array}

    assert ob.get_spin_keys(mapping) == [Spin.up, Spin.down]
    assert ob.get_spin_array(mapping, Spin.up).shape == (1, 2, 3, 2)


def test_an_absent_spin_channel_is_refused():

    with pytest.raises(ob.OrbitalError, match="not in this run"):
        ob.get_spin_array({Spin.up: np.zeros((1, 1, 2))}, Spin.down)


# ------------------------------------------------------------
# The Bloch factor and the global phase
#
# fft_mesh puts a band's coefficients at their G-vector indices and
# nothing else, so transforming it returns the cell-periodic part u_nk
# rather than the wavefunction.  Drawing u in place of psi is the kind
# of error that leaves the angular character intact -- so a table still
# reads 78% dz2 -- while moving lobes and nodes around the cell.
# ------------------------------------------------------------


def test_the_bloch_phase_is_one_at_gamma():

    phase = ob.bloch_phase((4, 4, 4), [0.0, 0.0, 0.0])

    assert np.allclose(phase, 1.0)


def test_the_bloch_phase_winds_once_across_the_cell():
    """At k = (1, 0, 0) the phase must come back to itself exactly once."""

    phase = ob.bloch_phase((8, 1, 1), [1.0, 0.0, 0.0])

    angles = np.unwrap(np.angle(phase[:, 0, 0]))

    assert np.isclose(angles[-1] - angles[0], 2 * np.pi * 7 / 8)


def test_the_bloch_phase_turns_through_half_a_cycle_at_the_zone_boundary():
    """
    k = (0.5, 0, 0) means the wavefunction comes back as its own
    negative one cell along.  Inside the cell the phase sweeps from 1 to
    i, so its real part falls to zero halfway -- and a picture drawn
    from u alone has a node plane there that the state does not have.
    """

    phase = ob.bloch_phase((4, 1, 1), [0.5, 0.0, 0.0])[:, 0, 0]

    assert np.isclose(phase[0], 1.0)
    assert np.isclose(phase[2], 1j)

    # the spurious node: Re falls monotonically to zero at f = 0.5
    real = np.real(phase)

    assert np.all(np.diff(real) < 0)
    assert np.isclose(real[2], 0.0, atol=1e-12)

    # and one cell along it really is the negative
    assert np.isclose(phase[2] ** 2, -1.0)


def test_a_bad_kpoint_is_refused():

    with pytest.raises(ob.OrbitalError, match="three components"):
        ob.bloch_phase((2, 2, 2), [0.0, 0.0])


@pytest.mark.parametrize(
    "kpoint, invariant",
    [
        ([0.0, 0.0, 0.0], True),
        ([0.5, 0.0, 0.0], True),
        ([0.5, 0.5, 0.5], True),
        ([0.25, 0.0, 0.0], False),
        ([1.0 / 3.0, 0.0, 0.0], False),
    ],
)
def test_time_reversal_invariant_points_are_recognised(kpoint, invariant):
    """
    Only at these can a wavefunction be made real, which is what decides
    whether "the real part" is the whole state or half of it.
    """

    assert ob.is_time_reversal_invariant(kpoint) is invariant


def test_realify_finds_the_phase_that_makes_a_state_real():
    """
    A real function rotated by an arbitrary phase must come back real,
    with nothing left in the imaginary part.
    """

    truth = np.array([1.0, -2.0, 3.0, -0.5])

    hidden = truth * np.exp(1j * 0.9)

    rotated, imaginary = ob.realify(hidden)

    assert imaginary < 1e-12
    assert np.allclose(np.real(rotated), truth)


def test_realify_recovers_a_state_the_real_part_would_have_lost():
    """
    The failure this guards against: a purely imaginary array has a real
    part of exactly zero, so drawing Re() without fixing the phase draws
    nothing at all.
    """

    truth = np.array([1.0, -2.0, 3.0, -0.5])

    hidden = truth * 1j

    assert np.allclose(np.real(hidden), 0.0)

    rotated, imaginary = ob.realify(hidden)

    assert imaginary < 1e-12
    assert np.allclose(np.abs(np.real(rotated)), np.abs(truth))


def test_realify_is_sign_stable():
    """Same state twice, same way up -- the picture must not flip."""

    truth = np.array([1.0, -2.0, 3.0, -0.5])

    first, _ = ob.realify(truth * np.exp(1j * 0.3))
    second, _ = ob.realify(truth * np.exp(1j * 2.7))

    assert np.allclose(first, second)


def test_realify_reports_a_genuinely_complex_state():
    """
    Two components in quadrature cannot both be real at once, and the
    reported residue is what tells a caller so.
    """

    psi = np.array([1.0 + 0.0j, 0.0 + 1.0j])

    _, imaginary = ob.realify(psi)

    assert imaginary > 0.4


def test_realify_survives_an_empty_state():

    rotated, imaginary = ob.realify(np.zeros(4, dtype=complex))

    assert imaginary == 0.0
    assert np.allclose(rotated, 0.0)


def test_psi_grid_applies_the_bloch_factor():
    """
    The regression itself.

    Comparing magnitudes would prove nothing -- a phase has modulus one,
    so ``abs`` is exactly the quantity it cannot change, and that is why
    dropping the factor leaves a table looking right while the picture
    is wrong.  What must change is how the phase varies from point to
    point, so that is what is measured, divided through by the value at
    the origin to cancel the arbitrary global rotation.
    """

    wavecar = FakeWavecar()

    bare = np.fft.ifftn(wavecar.fft_mesh(1, 0))

    at_gamma, _ = ob.psi_grid(
        wavecar, ob.State(Spin.up, 1, 1, 0.0, 1.0, (0.0, 0.0, 0.0))
    )
    at_boundary, _ = ob.psi_grid(
        wavecar, ob.State(Spin.up, 2, 1, 0.0, 1.0, (0.5, 0.0, 0.0))
    )

    def relative(grid):
        ratio = grid / bare
        return ratio / ratio.flat[0]

    # At Gamma the factor is 1, so nothing varies across the cell.
    assert np.allclose(relative(at_gamma), 1.0)

    # At the boundary it must vary, exactly as the Bloch factor does.
    phase = ob.bloch_phase(bare.shape, [0.5, 0.0, 0.0])

    assert not np.allclose(relative(at_boundary), 1.0)
    assert np.allclose(relative(at_boundary), phase / phase.flat[0])

    # And the real part -- what is actually drawn -- differs.
    assert not np.allclose(np.real(at_gamma), np.real(at_boundary))


def test_the_density_is_untouched_by_the_phase(poscar):
    """A phase has modulus one, so psi squared must not move."""

    wavecar = FakeWavecar()

    state = ob.State(Spin.up, 2, 1, 0.0, 1.0, (0.5, 0.0, 0.0))

    bare = np.abs(np.fft.ifftn(wavecar.fft_mesh(1, 0))) ** 2

    assert np.allclose(ob._realspace(wavecar, state, "square"), bare)


class RealStateWavecar:
    """
    A WAVECAR holding one known real wavefunction, hidden as VASP hides it.

    ``FakeWavecar`` returns a single plane wave, which is genuinely
    complex and can never be made real -- fine for checking a transform
    runs, useless for checking the answer.  Here a real dz2-shaped
    function is chosen first, multiplied by ``exp(-i k . r)`` to make
    the cell-periodic part VASP would have stored, and transformed to
    coefficients.  Recovering ``truth`` from that is the whole job:
    Bloch factor, global phase and all.
    """

    nk = 2
    nb = 1

    kpoints = [np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])]

    def __init__(self, n: int = 12) -> None:

        self.n = n

        axis = np.arange(n) / n
        fx, fy, fz = np.meshgrid(axis, axis, axis, indexing="ij")

        x, y, z = (fx - 0.5), (fy - 0.5), (fz - 0.5)
        r2 = x ** 2 + y ** 2 + z ** 2 + 1e-9

        # dz2: two lobes along z, a torus around the waist.
        self.truth = (3 * z ** 2 - r2) / r2 * np.exp(-12 * r2)

        self.frac = (fx, fy, fz)

    def fft_mesh(self, kpoint, band, spin=0, spinor=0, shift=True):

        k = self.kpoints[kpoint]
        fx, fy, fz = self.frac

        periodic = self.truth * np.exp(
            -2j * np.pi * (k[0] * fx + k[1] * fy + k[2] * fz)
        )

        return np.fft.fftn(periodic)


@pytest.mark.parametrize("kpoint", [1, 2])
def test_a_known_real_wavefunction_comes_back_intact(kpoint):
    """
    The end-to-end answer to "is the orbital in the picture".

    A dz2 goes in; the same dz2 must come out, at the zone boundary as
    well as at Gamma.  Without the Bloch factor the k=2 case comes back
    modulated by cos(pi x) instead -- lobes shrunk, a node plane added.
    """

    wavecar = RealStateWavecar()

    state = ob.State(
        Spin.up, kpoint, 1, 0.0, 1.0, tuple(wavecar.kpoints[kpoint - 1])
    )

    psi, imaginary = ob.psi_grid(wavecar, state)

    # A real state at a TRIM point must leave nothing behind.
    assert imaginary < 1e-12

    recovered = np.real(psi)

    # Same shape, same sign structure, same everything.
    assert np.allclose(recovered, wavecar.truth, atol=1e-10)

    # And what gets drawn is that, not the bare transform.
    assert np.allclose(
        ob._realspace(wavecar, state, "real"), wavecar.truth, atol=1e-10
    )


def test_dropping_the_bloch_factor_would_be_caught():
    """
    Guard the guard: the bare transform really is a different picture,
    so the test above is not passing by accident.
    """

    wavecar = RealStateWavecar()

    bare = np.real(np.fft.ifftn(wavecar.fft_mesh(1, 0)))

    assert not np.allclose(bare, wavecar.truth, atol=1e-6)

    # It is the true state times cos(2 pi k . f) -- the spurious
    # envelope, with a node plane the state does not have.
    fx = wavecar.frac[0]

    assert np.allclose(bare, wavecar.truth * np.cos(np.pi * fx), atol=1e-10)


@pytest.mark.parametrize("kpoint", [1, 2])
def test_no_note_is_shown_for_a_state_that_is_really_real(kpoint):

    wavecar = RealStateWavecar()

    state = ob.State(
        Spin.up, kpoint, 1, 0.0, 1.0, tuple(wavecar.kpoints[kpoint - 1])
    )

    assert ob.wavefunction_note(wavecar, state) is None


def test_the_square_is_the_real_part_squared_at_a_trim_point():
    """
    Nothing is left in the imaginary part there, so the two must agree
    -- which is also what says the drawn sign structure is complete.
    """

    wavecar = RealStateWavecar()

    state = ob.State(Spin.up, 2, 1, 0.0, 1.0, (0.5, 0.0, 0.0))

    real = ob._realspace(wavecar, state, "real")
    square = ob._realspace(wavecar, state, "square")

    assert np.allclose(real ** 2, square, atol=1e-12)


def test_a_complex_state_is_flagged():
    """Away from a TRIM point the real part is a choice, and says so."""

    class Odd(FakeWavecar):
        kpoints = [np.array([0.25, 0.0, 0.0]), np.array([0.25, 0.0, 0.0])]

        def fft_mesh(self, kpoint, band, spin=0, spinor=0, shift=True):
            mesh = np.zeros((8, 8, 8), dtype=complex)
            mesh[1, 0, 0] = 1.0
            mesh[0, 1, 0] = 1.0
            return mesh

    note = ob.wavefunction_note(Odd(), ob.State(Spin.up, 1, 1, 0.0, 1.0, (0.25, 0, 0)))

    assert note is not None
    assert "imaginary part" in note


# ------------------------------------------------------------
# How finely a contribution is broken down
#
# Two views over the same nine numbers. "70% O p" is what a chemist
# reads; "62% dz2" is what a crystal-field argument needs, and summing
# the shell throws it away. Both come from the same projection, so
# switching between them must never mean reloading the run.
# ------------------------------------------------------------


def test_the_two_views_are_offered() -> None:

    assert set(ob.VIEWS) == {ob.VIEW_SPD, ob.VIEW_RESOLVED}


def test_the_spd_view_has_three_columns() -> None:

    assert ob.view_labels(ob.VIEW_SPD) == ("s", "p", "d")


def test_the_resolved_view_has_all_nine() -> None:

    assert ob.view_labels(ob.VIEW_RESOLVED) == ob.ORBITALS

    assert len(ob.view_labels(ob.VIEW_RESOLVED)) == 9


def test_an_unknown_view_is_refused() -> None:

    with pytest.raises(ob.OrbitalError, match="Unknown view"):
        ob.view_labels("f")

    with pytest.raises(ob.OrbitalError, match="Unknown view"):
        ob.view_values(np.zeros(9), "f")


def test_collapsing_to_shells_conserves_the_total() -> None:
    """
    The two views describe the same state.

    A view that silently dropped or double-counted a channel would
    change the answer depending on which button was pressed.
    """

    values = np.arange(1.0, 10.0)

    shells = ob.to_shells(values)

    assert shells.shape == (3,)

    assert float(shells.sum()) == pytest.approx(float(values.sum()))


def test_each_shell_takes_its_own_channels() -> None:
    """s is one, p is three, d is five, in VASP's order."""

    values = np.array([1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6, 1e7, 1e8])

    shells = ob.to_shells(values)

    assert shells[0] == pytest.approx(1.0)
    assert shells[1] == pytest.approx(1110.0)
    assert shells[2] == pytest.approx(1e4 + 1e5 + 1e6 + 1e7 + 1e8)


def test_collapsing_works_on_a_stack_of_rows() -> None:
    """The per-atom table is a stack, and is collapsed the same way."""

    values = np.ones((4, 9))

    shells = ob.to_shells(values)

    assert shells.shape == (4, 3)

    assert np.allclose(shells, [[1.0, 3.0, 5.0]] * 4)


def test_a_row_of_the_wrong_width_is_refused() -> None:

    with pytest.raises(ob.OrbitalError, match="channels"):
        ob.to_shells(np.zeros(5))


def test_the_resolved_view_changes_nothing() -> None:

    values = np.arange(9.0)

    assert np.allclose(ob.view_values(values, ob.VIEW_RESOLVED), values)


def test_every_orbital_knows_its_shell() -> None:

    assert ob.shell_of("s") == "s"
    assert ob.shell_of("px") == "p"
    assert ob.shell_of("dz2") == "d"

    for name in ob.ORBITALS:
        assert ob.shell_of(name) in ob.SHELL_NAMES


def test_an_unknown_orbital_has_no_shell() -> None:

    with pytest.raises(ob.OrbitalError, match="Unknown orbital"):
        ob.shell_of("fz3")


# ------------------------------------------------------------
# The Contribution table
# ------------------------------------------------------------


def contributions(*rows):
    """Build Contributions from raw nine-channel rows."""

    return [
        ob.Contribution(
            label=label,
            atoms=(index,),
            per_atom=np.asarray([values], dtype=float),
            totals=np.asarray(values, dtype=float),
        )
        for index, (label, values) in enumerate(rows)
    ]


def test_the_breakdown_has_a_row_per_orbital_plus_subtotals() -> None:

    data = ob.breakdown(
        contributions(("Li", np.ones(9)), ("O", np.ones(9)))
    )

    assert len(data.orbitals) == 12

    assert data.orbitals[:9] == ob.ORBITALS
    assert data.orbitals[9:] == ("s total", "p total", "d total")

    assert list(data.is_subtotal[9:]) == [True, True, True]
    assert not any(data.is_subtotal[:9])


def test_the_columns_are_the_elements() -> None:

    data = ob.breakdown(
        contributions(("Li", np.ones(9)), ("O", np.ones(9)))
    )

    assert data.elements == ("Li", "O")

    assert data.values.shape == (12, 2)


def test_the_percentages_add_to_a_hundred() -> None:
    """
    Over the nine real rows only.

    Including the subtotals in the denominator would halve every number
    while leaving a table that still looks entirely reasonable.
    """

    generator = np.random.default_rng(0)

    data = ob.breakdown(
        contributions(
            ("Li", generator.random(9)),
            ("Co", generator.random(9)),
            ("O", generator.random(9)),
        )
    )

    real = [
        value
        for value, flag in zip(data.percent, data.is_subtotal)
        if not flag
    ]

    assert sum(real) == pytest.approx(100.0)


def test_a_subtotal_is_the_sum_of_its_shell() -> None:

    values = np.arange(1.0, 10.0)

    data = ob.breakdown(contributions(("X", values)))

    assert data.totals[9] == pytest.approx(values[0])
    assert data.totals[10] == pytest.approx(values[1:4].sum())
    assert data.totals[11] == pytest.approx(values[4:].sum())


def test_the_total_counts_each_channel_once() -> None:
    """Not once as itself and again inside its shell subtotal."""

    data = ob.breakdown(contributions(("X", np.ones(9))))

    assert data.total == pytest.approx(9.0)


def test_an_empty_state_gives_zeros_not_nans() -> None:
    """
    A state with nothing on it is common above the Fermi level.

    Dividing by its total would put NaN through the whole table.
    """

    data = ob.breakdown(contributions(("X", np.zeros(9))))

    assert np.all(np.isfinite(data.percent))
    assert np.allclose(data.percent, 0.0)


def test_a_breakdown_of_nothing_is_refused() -> None:

    with pytest.raises(ob.OrbitalError, match="no contributions"):
        ob.breakdown([])

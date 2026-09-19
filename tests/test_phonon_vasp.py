"""
Tests for the VASP phonon workflow.

The interesting part is that VASP is never involved. Preprocessing is
file writing and postprocessing is file reading, so the whole round trip
-- configuration out, vasprun.xml in, force constants, band, DOS -- runs
here with a Lennard-Jones potential standing in for the DFT run.

That stand-in is the point rather than a shortcut: it produces real
forces on the real displaced supercells phonopy generated, written into
real vasprun.xml files parsed by phonopy's own reader, so everything
between QVista and the answer is exercised. Only the electrons are fake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qvista.analyses.base import AnalysisError
from qvista.analyses.phonon_vasp import VaspPhononCollect
from qvista.services import phonopy_vasp
from qvista.services.phonopy_vasp import (
    DFPT,
    FINITE_DISPLACEMENT,
    METHODS,
    method_by_name,
)


# ------------------------------------------------------------
# Structures
# ------------------------------------------------------------


@pytest.fixture
def silicon():
    """Two-atom primitive silicon: small, and highly symmetric."""

    from pymatgen.core import Lattice, Structure

    half = 5.43 / 2.0

    return Structure(
        Lattice([[0.0, half, half], [half, 0.0, half], [half, half, 0.0]]),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


@pytest.fixture
def triclinic():
    """
    Three species in general positions: genuinely P1.

    Silicon at 2x2x2 needs exactly one displacement, which is fine for
    a round trip and useless for testing a half-finished set.
    """

    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.from_parameters(4.0, 4.5, 5.0, 71.0, 82.0, 95.0),
        ["Si", "O", "N"],
        [[0.01, 0.02, 0.03], [0.31, 0.27, 0.23], [0.62, 0.11, 0.55]],
    )


@pytest.fixture
def rocksalt():
    """Two species, so the per-species lists have something to line up."""

    from pymatgen.core import Lattice, Structure

    half = 4.2 / 2.0

    return Structure(
        Lattice([[0.0, half, half], [half, 0.0, half], [half, half, 0.0]]),
        ["Mg", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def method(name: str):
    """One route, by name."""

    found = method_by_name(name)

    assert found is not None

    return found


# ------------------------------------------------------------
# Fake VASP output
#
# The minimum vasprun.xml phonopy's parser accepts. The <generator>
# block is not decoration: phonopy scans for the version line before
# parsing and, not finding one, reads to end of file and hands an
# exhausted stream to expat.
# ------------------------------------------------------------


def _varray(name, rows) -> str:

    lines = [f'   <varray name="{name}" >']

    for row in rows:
        lines.append(
            "    <v>"
            + " ".join(f"{float(value):20.8f}" for value in row)
            + " </v>"
        )

    lines.append("   </varray>")

    return "\n".join(lines)


def write_vasprun(folder: Path, cell, forces) -> Path:
    """A vasprun.xml holding one ionic step's forces."""

    import numpy as np

    basis = np.asarray(cell.cell, dtype=float)

    text = "\n".join(
        [
            '<?xml version="1.0" encoding="ISO-8859-1"?>',
            "<modeling>",
            " <generator>",
            '  <i name="program" type="string">vasp </i>',
            '  <i name="version" type="string">6.4.2</i>',
            " </generator>",
            " <calculation>",
            "  <structure>",
            "   <crystal>",
            _varray("basis", basis),
            f'    <i name="volume">{abs(np.linalg.det(basis)):.8f}</i>',
            _varray("rec_basis", np.linalg.inv(basis).T),
            "   </crystal>",
            _varray("positions", cell.scaled_positions),
            "  </structure>",
            _varray("forces", forces),
            "  <energy>",
            '   <i name="e_fr_energy">-1.00000000</i>',
            '   <i name="e_wo_entrp">-1.00000000</i>',
            '   <i name="e_0_energy">-1.00000000</i>',
            "  </energy>",
            " </calculation>",
            "</modeling>",
        ]
    )

    target = folder / "vasprun.xml"
    target.write_text(text, encoding="utf-8")

    return target


def write_dfpt_vasprun(folder: Path, cell, hessian) -> Path:
    """A vasprun.xml holding a <dynmat> block, as IBRION = 8 writes."""

    import numpy as np

    from pymatgen.core.periodic_table import Element

    symbols = [str(symbol) for symbol in cell.symbols]

    blocks: list[tuple[str, int]] = []

    for symbol in symbols:

        if blocks and blocks[-1][0] == symbol:
            blocks[-1] = (symbol, blocks[-1][1] + 1)
        else:
            blocks.append((symbol, 1))

    types = "\n".join(
        f"    <rc><c>{count:4d}</c><c>{symbol}</c>"
        f"<c>{Element(symbol).atomic_mass.real:12.4f}</c>"
        f"<c>{4.0:12.4f}</c><c> PAW_PBE {symbol}</c></rc>"
        for symbol, count in blocks
    )

    text = "\n".join(
        [
            '<?xml version="1.0" encoding="ISO-8859-1"?>',
            "<modeling>",
            " <generator>",
            '  <i name="program" type="string">vasp </i>',
            '  <i name="version" type="string">6.4.2</i>',
            " </generator>",
            " <atominfo>",
            '  <array name="atomtypes" >',
            "   <set>",
            types,
            "   </set>",
            "  </array>",
            " </atominfo>",
            " <calculation>",
            "  <dynmat>",
            _varray("hessian", np.asarray(hessian, dtype=float)),
            '   <i name="unit">eV/Angst^2</i>',
            "  </dynmat>",
            " </calculation>",
            "</modeling>",
        ]
    )

    target = folder / "vasprun.xml"
    target.write_text(text, encoding="utf-8")

    return target


def lennard_jones():
    """
    A cheap, real pair potential standing in for the DFT run.

    A spring pulling each atom back to its own site couples nothing to
    anything, so every mode would come out degenerate and the DOS would
    have zero width for phonopy to divide by. A genuine pair potential
    gives real dispersion, which is what the pipeline has to be tested
    against. Sigma sits near silicon's nearest-neighbour distance so the
    atoms are inside the cutoff.
    """

    from ase.calculators.lj import LennardJones

    return LennardJones(sigma=2.35, epsilon=1.0, rc=6.0)


def config_folders(root: Path) -> list[Path]:
    """The Config_N folders of ``root``, in numerical order."""

    series = root / phonopy_vasp.FD_DIR

    if not series.is_dir():
        return []

    numbered = [
        (phonopy_vasp.config_index(folder.name), folder)
        for folder in series.iterdir()
        if folder.is_dir()
    ]

    return [
        folder
        for index, folder in sorted(
            (item for item in numbered if item[0] is not None),
            key=lambda item: item[0],
        )
    ]


def run_fake_vasp(root: Path) -> None:
    """Fill every force run in a configuration with computed forces."""

    import numpy as np
    from ase import Atoms
    from phonopy.interface.vasp import read_vasp

    calculator = lennard_jones()

    for folder in config_folders(root):

        cell = read_vasp(folder / "POSCAR")

        atoms = Atoms(
            symbols=[str(symbol) for symbol in cell.symbols],
            scaled_positions=cell.scaled_positions,
            cell=cell.cell,
            pbc=True,
        )

        atoms.calc = calculator

        write_vasprun(
            folder,
            cell,
            np.asarray(atoms.get_forces(), dtype=float),
        )


# ------------------------------------------------------------
# The two routes
# ------------------------------------------------------------


def test_there_are_exactly_two_routes():
    """Finite displacement and DFPT, and nothing else."""

    assert sorted(entry.name for entry in METHODS) == sorted(
        [DFPT, FINITE_DISPLACEMENT]
    )


def test_each_route_is_defined_by_its_ibrion():
    """The one tag that says which calculation VASP is being asked for."""

    assert method(FINITE_DISPLACEMENT).ibrion == "-1"
    assert method(DFPT).ibrion == "8"


# ------------------------------------------------------------
# The INCAR
#
# One fixed file. These tests are here to keep it fixed: it is short
# enough to read, and every line in it was chosen on purpose.
# ------------------------------------------------------------


EXPECTED = """SYSTEM = Phonopy - Finite Displacement

# ==========================================================
# Electronic Structure
# ==========================================================
ENCUT   = 520
PREC    = Accurate
ALGO    = Normal

# ==========================================================
# SCF Convergence
# ==========================================================
EDIFF   = 1E-8

# ==========================================================
# Ionic / Force Calculation
# ==========================================================
IBRION  = -1
NSW     = 0
ISIF    = 2

# ==========================================================
# Smearing
# ==========================================================
ISMEAR  = 0
SIGMA   = 0.05

# ==========================================================
# Spin
# ==========================================================
ISPIN   = 1

# ==========================================================
# Precision / Numerical Accuracy
# ==========================================================
LREAL   = .FALSE.
ADDGRID = .TRUE.

# ==========================================================
# Output
# ==========================================================
LWAVE   = .FALSE.
LCHARG  = .FALSE.
"""


def test_the_finite_displacement_incar_is_exactly_this():
    """Character for character. Anything else is a regression."""

    assert phonopy_vasp.phonon_incar(
        method(FINITE_DISPLACEMENT)
    ) == EXPECTED


def test_dfpt_differs_by_two_lines_and_no_others():
    """
    SYSTEM names the route and IBRION is what makes it that route.

    Everything else about a force calculation is the same either way,
    and the test says so by diffing rather than by listing.
    """

    fd = phonopy_vasp.phonon_incar(method(FINITE_DISPLACEMENT))
    dfpt = phonopy_vasp.phonon_incar(method(DFPT))

    differing = [
        (a, b)
        for a, b in zip(fd.splitlines(), dfpt.splitlines())
        if a != b
    ]

    assert differing == [
        ("SYSTEM = Phonopy - Finite Displacement", "SYSTEM = Phonopy - DFPT"),
        ("IBRION  = -1", "IBRION  = 8"),
    ]


def test_the_cutoff_can_be_filled_in():
    """Derived from the POTCARs when there are any, silently."""

    text = phonopy_vasp.phonon_incar(method(DFPT), encut=650)

    assert "ENCUT   = 650" in text
    assert "ENCUT   = 520" not in text


def test_the_incar_carries_nothing_beyond_the_file():
    """
    No MAGMOM, no +U, no dispersion, no parallelisation.

    The file is the specification; a tag that is not in it must not
    appear because something upstream thought it would be helpful.
    """

    for entry in METHODS:

        tags = {
            line.split("=")[0].strip()
            for line in phonopy_vasp.phonon_incar(entry).splitlines()
            if "=" in line and not line.startswith("#")
        }

        assert tags == {
            "SYSTEM",
            "ENCUT",
            "PREC",
            "ALGO",
            "EDIFF",
            "IBRION",
            "NSW",
            "ISIF",
            "ISMEAR",
            "SIGMA",
            "ISPIN",
            "LREAL",
            "ADDGRID",
            "LWAVE",
            "LCHARG",
        }


def test_the_incar_has_no_inline_comments():
    """
    Comments are section banners only.

    A tag-by-tag commentary doubles the length of a file whose whole
    point is that it is short enough to read at a glance.
    """

    for line in phonopy_vasp.phonon_incar(method(DFPT)).splitlines():

        if line.startswith("#") or not line.strip():
            continue

        assert "!" not in line
        assert "#" not in line


def test_pymatgen_reads_the_incar_back():
    """Whatever is written has to survive being parsed as an INCAR."""

    from pymatgen.io.vasp.inputs import Incar

    for entry in METHODS:

        parsed = Incar.from_str(phonopy_vasp.phonon_incar(entry, encut=520))

        assert parsed["ENCUT"] == 520
        assert parsed["IBRION"] == int(entry.ibrion)
        assert parsed["EDIFF"] == pytest.approx(1e-8)
        assert parsed["LREAL"] is False


# ------------------------------------------------------------
# Supercells and meshes
# ------------------------------------------------------------


def test_a_bigger_supercell_gets_a_coarser_mesh(silicon):
    """
    The same physical k-point resolution, not the same grid.

    A supercell is smaller in reciprocal space, so keeping the unit
    cell's mesh would oversample it by the repetition factor and make
    every run needlessly expensive.
    """

    unit = phonopy_vasp.mesh_for(silicon, 0.3)
    super_ = phonopy_vasp.mesh_for(silicon * (2, 2, 2), 0.3)

    assert all(a > b for a, b in zip(unit, super_))


def test_a_mesh_never_drops_below_one(silicon):

    grid = phonopy_vasp.mesh_for(silicon * (8, 8, 8), 0.9)

    assert all(value >= 1 for value in grid)


def test_species_of_counts_a_supercell(silicon):
    """The counts multiply; the species order does not change."""

    phonon = phonopy_vasp.build_phonon(silicon, (2, 2, 2))

    unit_elements, unit_counts = phonopy_vasp.species_of(phonon.unitcell)
    elements, counts = phonopy_vasp.species_of(phonon.supercell)

    assert unit_elements == elements == ["Si"]
    assert unit_counts == [2]
    assert counts == [16]


# ------------------------------------------------------------
# Writing a configuration
# ------------------------------------------------------------


def test_finite_displacement_writes_one_folder_per_displacement(
    silicon, tmp_path
):

    root = tmp_path / "fd"

    config = phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
        potcar="FAKE POTCAR\nEnd of Dataset\n",
    )

    folders = config_folders(root)

    assert len(folders) == config.n_displacements
    assert len(folders) == len(config.force_runs)

    for folder in folders:
        for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
            assert (folder / name).is_file(), f"{folder.name}/{name}"


def test_the_top_level_carries_what_postprocessing_needs(
    silicon, tmp_path
):
    """
    Everything needed to read the folder back is in the folder.

    That is what lets the runs happen on another machine, in another
    month, with no QVista session in between.
    """

    root = tmp_path / "fd"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    assert (root / phonopy_vasp.DISP_YAML).is_file()
    assert (root / phonopy_vasp.SPOSCAR).is_file()
    assert (root / phonopy_vasp.UNITCELL).is_file()
    assert (root / "README.txt").is_file()


def test_dfpt_writes_a_single_run_on_the_perfect_supercell(
    silicon, tmp_path
):
    """No displaced cells: the POSCAR is SPOSCAR itself."""

    root = tmp_path / "dfpt"

    config = phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(DFPT),
        supercell=(2, 2, 2),
        encut=520,
    )

    assert not config_folders(root)
    assert not (root / phonopy_vasp.FD_DIR).exists()
    assert len(config.force_runs) == 1

    folder = root / phonopy_vasp.DFPT_DIR

    assert folder.is_dir()

    assert (folder / "POSCAR").read_text() == (
        root / phonopy_vasp.SPOSCAR
    ).read_text()

    from pymatgen.io.vasp.inputs import Incar

    parsed = Incar.from_str((folder / "INCAR").read_text())

    assert parsed["IBRION"] == 8
    assert "NCORE" not in parsed


def test_no_potcar_is_a_warning_rather_than_a_failure(silicon, tmp_path):
    """
    The configuration is still worth having without one.

    A POTCAR can be dropped into the folders afterwards; refusing to
    write anything would throw away the part that took the thinking.
    """

    root = tmp_path / "nopotcar"

    config = phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(DFPT),
        supercell=(2, 2, 2),
        encut=520,
        potcar="",
    )

    assert not config.has_potcar
    assert config.warnings

    assert not (root / phonopy_vasp.DFPT_DIR / "POTCAR").exists()
    assert (root / phonopy_vasp.DFPT_DIR / "INCAR").is_file()


def test_every_run_gets_the_supercell_mesh(silicon, tmp_path):
    """Not the unit cell's, which would oversample by the repetition."""

    root = tmp_path / "mesh"

    config = phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(DFPT),
        supercell=(2, 2, 2),
        encut=520,
        kspacing=0.3,
    )

    text = (root / phonopy_vasp.DFPT_DIR / "KPOINTS").read_text()

    nx, ny, nz = config.mesh

    assert f"{nx} {ny} {nz}" in text
    assert config.mesh < phonopy_vasp.mesh_for(silicon, 0.3)


# ------------------------------------------------------------
# The two write passes
# ------------------------------------------------------------


def test_the_first_pass_writes_cells_and_nothing_else(silicon, tmp_path):
    """
    Choosing a route and generating its inputs are two clicks.

    The first one has to leave the folders in a state the second can
    fill in, which means POSCARs and no INCAR.
    """

    root = tmp_path / "cells"

    phonon = phonopy_vasp.build_phonon(silicon, (2, 2, 2))

    config = phonopy_vasp.write_structures(
        root, phonon, method(FINITE_DISPLACEMENT)
    )

    for run in config.force_runs:
        assert (run.path / "POSCAR").is_file()
        assert not (run.path / "INCAR").exists()
        assert not (run.path / "KPOINTS").exists()
        assert not (run.path / "POTCAR").exists()

    assert (root / phonopy_vasp.DISP_YAML).is_file()
    assert (root / phonopy_vasp.SPOSCAR).is_file()


def test_the_second_pass_fills_the_folders_in(silicon, tmp_path):

    root = tmp_path / "inputs"

    phonon = phonopy_vasp.build_phonon(silicon, (2, 2, 2))

    config = phonopy_vasp.write_structures(
        root, phonon, method(DFPT)
    )

    phonopy_vasp.write_vasp_inputs(
        config,
        phonon,
        encut=480,
        potcar="FAKE POTCAR\nEnd of Dataset\n",
    )

    folder = root / phonopy_vasp.DFPT_DIR

    for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
        assert (folder / name).is_file(), name

    from pymatgen.io.vasp.inputs import Incar

    assert Incar.from_str((folder / "INCAR").read_text())["ENCUT"] == 480

    assert config.encut == 480
    assert config.has_potcar


def test_a_supplied_incar_goes_in_verbatim(silicon, tmp_path):
    """
    The INCAR dialog's output is not re-derived on the way to disk.

    Whatever the user built there is the file, tag for tag; deriving one
    "close to" it would silently drop their edits.
    """

    root = tmp_path / "byhand"

    phonon = phonopy_vasp.build_phonon(silicon, (2, 2, 2))

    config = phonopy_vasp.write_structures(root, phonon, method(DFPT))

    mine = "ENCUT = 700\nIBRION = 8\nEDIFF = 1E-09\nNSW = 1\n"

    phonopy_vasp.write_vasp_inputs(
        config, phonon, encut=300, incar=mine
    )

    assert (
        root / phonopy_vasp.DFPT_DIR / "INCAR"
    ).read_text() == mine


def test_a_supplied_mesh_goes_in_verbatim(silicon, tmp_path):
    """The k-points dialog's grid, not the one KSPACING would give."""

    root = tmp_path / "mymesh"

    phonon = phonopy_vasp.build_phonon(silicon, (2, 2, 2))

    config = phonopy_vasp.write_structures(root, phonon, method(DFPT))

    phonopy_vasp.write_vasp_inputs(
        config, phonon, encut=520, mesh=(5, 6, 7), scheme="Gamma"
    )

    text = (root / phonopy_vasp.DFPT_DIR / "KPOINTS").read_text()

    assert "5 6 7" in text
    assert config.mesh == (5, 6, 7)


def test_the_second_pass_can_be_run_without_a_potcar(silicon, tmp_path):
    """A missing library is a warning, not a refusal to write anything."""

    root = tmp_path / "nopotcar2"

    phonon = phonopy_vasp.build_phonon(silicon, (2, 2, 2))

    config = phonopy_vasp.write_structures(root, phonon, method(DFPT))

    phonopy_vasp.write_vasp_inputs(config, phonon, encut=520)

    folder = root / phonopy_vasp.DFPT_DIR

    assert (folder / "INCAR").is_file()
    assert not (folder / "POTCAR").exists()
    assert config.warnings


# ------------------------------------------------------------
# Folder naming
# ------------------------------------------------------------


def test_config_folders_are_named_and_numbered_from_one():

    assert phonopy_vasp.config_name(1) == "Config_1"
    assert phonopy_vasp.config_name(12) == "Config_12"
    assert phonopy_vasp.DFPT_DIR == "DFPT"
    assert phonopy_vasp.FD_DIR == "Finite Displacement"


def test_the_two_routes_sit_side_by_side_under_one_root(
    silicon, tmp_path
):
    """
    Each route gets a folder of its own, in the same place.

    Writing both into one root is how the two are compared, so neither
    may land on top of the other or inside it.
    """

    root = tmp_path / "both"

    phonon = phonopy_vasp.build_phonon(silicon, (2, 2, 2))

    phonopy_vasp.write_structures(root, phonon, method(DFPT))
    phonopy_vasp.write_structures(
        root, phonon, method(FINITE_DISPLACEMENT)
    )

    assert (root / phonopy_vasp.DFPT_DIR / "POSCAR").is_file()

    series = root / phonopy_vasp.FD_DIR

    assert series.is_dir()
    assert (series / "Config_1" / "POSCAR").is_file()

    # ...and nothing loose at the top beside them.
    assert not (root / "Config_1").exists()


def test_config_index_reads_the_number_back():

    assert phonopy_vasp.config_index("Config_7") == 7
    assert phonopy_vasp.config_index("Config_10") == 10

    assert phonopy_vasp.config_index("DFPT") is None
    assert phonopy_vasp.config_index("Config_x") is None
    assert phonopy_vasp.config_index("scratch") is None


def test_runs_come_back_in_numerical_order_past_ten(
    triclinic, tmp_path
):
    """
    Config_10 must not sort before Config_2.

    Force sets handed to phonopy in a different order from its
    displacements is not an error anywhere -- it is a dispersion for a
    material that does not exist.
    """

    root = tmp_path / "ordering"

    config = phonopy_vasp.write_configuration(
        root=root,
        structure=triclinic,
        method=method(FINITE_DISPLACEMENT),
        supercell=(1, 1, 1),
        encut=520,
    )

    assert config.n_displacements >= 10, "need a two-digit set"

    run_fake_vasp(root)

    found = phonopy_vasp.discover(root)

    numbers = [
        phonopy_vasp.config_index(path.parent.name)
        for path in found.found
    ]

    assert numbers == sorted(numbers)
    assert numbers == list(range(1, config.n_displacements + 1))


# ------------------------------------------------------------
# Reading a configuration back
# ------------------------------------------------------------


def test_a_folder_that_is_not_a_configuration_says_so(tmp_path):

    found = phonopy_vasp.discover(tmp_path)

    assert found.problems
    assert not found.usable
    assert phonopy_vasp.DISP_YAML in found.problems[0]


def test_a_missing_folder_says_so(tmp_path):

    found = phonopy_vasp.discover(tmp_path / "nothing here")

    assert found.problems
    assert not found.usable


def test_a_fresh_configuration_is_found_but_not_usable(silicon, tmp_path):
    """Nothing has been run yet, and that is not an error."""

    root = tmp_path / "fresh"

    config = phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    found = phonopy_vasp.discover(root)

    assert found.method == FINITE_DISPLACEMENT
    assert found.expected == config.n_displacements
    assert len(found.missing) == config.n_displacements
    assert not found.usable


def test_the_supercell_is_read_back_out_of_the_yaml(silicon, tmp_path):
    """
    Parsed by hand, because it runs on every keystroke in a folder
    picker and the file can be megabytes long.
    """

    root = tmp_path / "header"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(DFPT),
        supercell=(2, 3, 4),
        encut=520,
    )

    found = phonopy_vasp.discover(root)

    assert found.supercell == (2, 3, 4)
    assert found.unitcell_natoms == 2
    assert found.supercell_natoms == 2 * 2 * 3 * 4


def test_a_half_finished_set_names_what_is_missing(
    triclinic, tmp_path
):
    """A run still in the queue is named, not glossed over."""

    root = tmp_path / "half"

    config = phonopy_vasp.write_configuration(
        root=root,
        structure=triclinic,
        method=method(FINITE_DISPLACEMENT),
        supercell=(1, 1, 1),
        encut=520,
    )

    assert config.n_displacements > 1, "need more than one to remove one"

    run_fake_vasp(root)

    # Take one back out again.
    (
        root
        / phonopy_vasp.FD_DIR
        / phonopy_vasp.config_name(1)
        / "vasprun.xml"
    ).unlink()

    found = phonopy_vasp.discover(root)

    assert found.missing == [phonopy_vasp.config_name(1)]
    assert len(found.found) == config.n_displacements - 1
    assert not found.usable
    assert phonopy_vasp.config_name(1) in found.summary()


def test_a_finished_set_is_usable(silicon, tmp_path):

    root = tmp_path / "done"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(root)

    found = phonopy_vasp.discover(root)

    assert found.usable
    assert found.complete
    assert not found.missing


# ------------------------------------------------------------
# The whole round trip
# ------------------------------------------------------------


def test_finite_displacement_round_trip(silicon, tmp_path):
    """
    Configuration out, forces in, dispersion and DOS back.

    Everything between QVista and the answer runs for real here: the
    displacements are phonopy's, the vasprun.xml files go through
    phonopy's own parser, and the band structure is evaluated on the
    seekpath path.
    """

    root = tmp_path / "roundtrip"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(8, 8, 8)).run()

    assert result.band is not None
    assert result.dos is not None
    assert result.band["frequencies"]
    assert len(result.dos["frequency_points"]) > 0

    assert (root / phonopy_vasp.FORCE_SETS).is_file()
    assert (root / phonopy_vasp.FORCE_CONSTANTS).is_file()
    assert (root / phonopy_vasp.BAND_YAML).is_file()
    assert (root / phonopy_vasp.TOTAL_DOS).is_file()


def test_the_round_trip_says_which_code_produced_the_forces(
    silicon, tmp_path
):
    """
    The label is stamped on every figure.

    Screening numbers and first-principles numbers look identical on a
    plot, so the one thing that must never be lost is which is which.
    """

    root = tmp_path / "labelled"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(6, 6, 6)).run()

    assert "VASP" in result.phonon_data.force_field
    assert "finite displacement" in result.phonon_data.force_field


def test_the_round_trip_reports_on_stability(silicon, tmp_path):
    """The verdict is the reason anyone ran this."""

    root = tmp_path / "verdict"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(6, 6, 6)).run()

    assert result.lowest is not None

    verdict = " ".join(result.notes)

    assert "STABLE" in verdict or "UNSTABLE" in verdict


def test_dfpt_round_trip(silicon, tmp_path):
    """
    The Hessian route: no FORCE_SETS, force constants straight off.

    The Hessian written here is the one the finite-displacement path
    produces for the same cell, so the two are being asked to agree.
    """

    import numpy as np

    from phonopy import load as phonopy_load

    # First get a genuine set of force constants the honest way.
    fd_root = tmp_path / "fd_reference"

    phonopy_vasp.write_configuration(
        root=fd_root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(fd_root)

    reference = VaspPhononCollect(None, fd_root, mesh=(6, 6, 6)).run()

    constants = np.asarray(
        reference.phonon_data.phonon.force_constants, dtype=float
    )

    # ...then write them out the way VASP writes a Hessian, which is
    # mass-normalised and sign-flipped, and check they come back.
    root = tmp_path / "dfpt_roundtrip"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(DFPT),
        supercell=(2, 2, 2),
        encut=520,
    )

    phonon = phonopy_load(
        root / phonopy_vasp.DISP_YAML, produce_fc=False, log_level=0
    )

    masses = np.asarray(phonon.supercell.masses, dtype=float)

    natoms = len(masses)

    hessian = np.zeros((natoms * 3, natoms * 3))

    for i in range(natoms):
        for j in range(natoms):
            hessian[
                i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3
            ] = -constants[i, j] / np.sqrt(masses[i] * masses[j])

    write_dfpt_vasprun(
        root / phonopy_vasp.DFPT_DIR, phonon.supercell, hessian
    )

    result = VaspPhononCollect(None, root, mesh=(6, 6, 6)).run()

    assert (root / phonopy_vasp.FORCE_CONSTANTS).is_file()
    assert not (root / phonopy_vasp.FORCE_SETS).exists()

    recovered = np.asarray(
        result.phonon_data.phonon.force_constants, dtype=float
    )

    assert np.abs(recovered - constants).max() < 1e-6


def test_collecting_before_the_runs_finish_explains_itself(
    silicon, tmp_path
):
    """
    A partial set is refused with the names of what is outstanding.

    phonopy cannot fill in a missing displacement, so the alternative to
    stopping here is a wrong answer.
    """

    root = tmp_path / "partial"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    with pytest.raises(AnalysisError) as excinfo:
        VaspPhononCollect(None, root).run()

    assert "vasprun.xml" in str(excinfo.value)


def test_collecting_from_a_folder_that_is_not_one_explains_itself(
    tmp_path
):

    with pytest.raises(AnalysisError) as excinfo:
        VaspPhononCollect(None, tmp_path).run()

    assert phonopy_vasp.DISP_YAML in str(excinfo.value)


def test_a_conventional_cell_is_used_as_given(tmp_path):
    """
    The cell the user opened is the cell phonopy gets.

    phonopy's own workflow: the unit cell is whatever is in the POSCAR,
    the supercell repeats it, and primitive_matrix="auto" leaves phonopy
    to reduce it. Nothing standardises the cell on the way in, and a
    conventional cell is not a problem to be refused -- it is the
    ordinary case.
    """

    from pymatgen.core import Lattice, Structure

    conventional = Structure(
        Lattice.cubic(5.43),
        ["Si"] * 8,
        [
            [0.00, 0.00, 0.00],
            [0.00, 0.50, 0.50],
            [0.50, 0.00, 0.50],
            [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ],
    )

    root = tmp_path / "conventional"

    phonopy_vasp.write_configuration(
        root=root,
        structure=conventional,
        method=method(FINITE_DISPLACEMENT),
        supercell=(1, 1, 1),
        encut=520,
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(6, 6, 6)).run()

    assert result.band is not None

    # phonopy found the 2-atom primitive cell for itself, so the bands
    # are 3N of *that* cell rather than of the 8-atom one handed in.
    primitive = result.phonon_data.phonon.primitive

    assert len(primitive) == 2

    import numpy as np

    assert np.asarray(result.band["frequencies"][0]).shape[1] == 6


def test_the_unit_cell_written_out_is_the_one_supplied(silicon, tmp_path):
    """
    POSCAR-unitcell is the user's cell, not a standardisation of it.

    They have to be able to recognise what they are about to run.
    """

    from phonopy.interface.vasp import read_vasp

    root = tmp_path / "asgiven"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(DFPT),
        supercell=(1, 1, 1),
        encut=520,
    )

    written = read_vasp(root / phonopy_vasp.UNITCELL)

    import numpy as np

    assert len(written) == len(silicon)
    assert np.allclose(
        np.asarray(written.cell, dtype=float),
        silicon.lattice.matrix,
        atol=1e-8,
    )


def test_the_band_path_comes_from_phonopys_own_primitive_cell(
    silicon, tmp_path
):
    """
    Which is what makes the two share a basis without any arranging.

    The q-points are generated from phonon.primitive and the dynamical
    matrix is evaluated on phonon.primitive, so a mismatch is not
    something the caller can cause.
    """

    root = tmp_path / "path"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(6, 6, 6)).run()

    import numpy as np

    primitive = result.phonon_data.phonon.primitive

    assert np.asarray(
        result.band["frequencies"][0]
    ).shape[1] == 3 * len(primitive)

    # Labels and breaks come back from phonopy, one label per end point.
    assert result.band["labels"]
    assert len(result.band["path_connections"]) == len(
        result.band["distances"]
    )


# ------------------------------------------------------------
# The README
# ------------------------------------------------------------


def test_the_readme_says_what_to_run(silicon, tmp_path):
    """
    Days pass between writing these folders and having the output.

    By then the only thing left is the folder, so it has to say what to
    do with itself.
    """

    root = tmp_path / "readme"

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(DFPT),
        supercell=(2, 2, 2),
        encut=520,
    )

    text = (root / "README.txt").read_text()

    assert phonopy_vasp.DFPT_DIR in text
    assert "NCORE" in text
    assert "vasprun.xml" in text
    # The whole path, not just the word: the instruction is only
    # useful if the place it names still exists in the menu.
    assert "Phonon Analysis > VASP > Post-Processing" in text


# ------------------------------------------------------------
# The experiment folder
#
# Pre-processing writes everything under one `phonopyexp` folder, and
# post-processing has to find it from whichever of the three folders a
# user would think to pick: the one they saved into, the experiment
# folder itself, or the configuration inside it.
# ------------------------------------------------------------


def _configuration(silicon, root, supercell=(2, 2, 2)):
    """Write one configuration, the way pre-processing does."""

    phonopy_vasp.write_configuration(
        root=root,
        structure=silicon,
        method=method(FINITE_DISPLACEMENT),
        supercell=supercell,
        encut=520,
    )

    return root


def test_a_configuration_is_recognised_by_its_disp_yaml(silicon, tmp_path):

    root = _configuration(
        silicon, tmp_path / phonopy_vasp.EXPERIMENT_DIR / "Si_phonon_222"
    )

    assert phonopy_vasp.is_configuration(root)
    assert not phonopy_vasp.is_configuration(root.parent)


def test_the_configuration_is_found_from_any_of_three_folders(
    silicon,
    tmp_path,
):
    """
    All three name the same thing to whoever is using this.

    Refusing two of them is how post-processing ends up feeling broken
    on a folder that is perfectly complete.
    """

    experiment = tmp_path / phonopy_vasp.EXPERIMENT_DIR

    root = _configuration(silicon, experiment / "Si_phonon_222")

    for start in (tmp_path, experiment, root):

        found = phonopy_vasp.configurations_under(start)

        assert found, f"nothing found from {start}"
        assert found[0] == root


def test_a_run_folder_is_not_a_configuration(silicon, tmp_path):
    """Pointing at one displacement must not look like the whole set."""

    root = _configuration(
        silicon, tmp_path / phonopy_vasp.EXPERIMENT_DIR / "Si_phonon_222"
    )

    assert phonopy_vasp.configurations_under(config_folders(root)[0]) == []


def test_discovery_walks_down_into_the_experiment_folder(
    silicon,
    tmp_path,
):

    root = _configuration(
        silicon, tmp_path / phonopy_vasp.EXPERIMENT_DIR / "Si_phonon_222"
    )

    run_fake_vasp(root)

    found = phonopy_vasp.discover(tmp_path)

    assert found.root == root
    assert not found.problems
    assert found.usable


def test_two_configurations_are_both_reported(silicon, tmp_path):
    """
    A second supercell beside the first is a normal thing to have.

    One is taken and the other named, rather than the folder being
    called ambiguous and refused outright.
    """

    experiment = tmp_path / phonopy_vasp.EXPERIMENT_DIR

    for supercell in ((2, 2, 2), (3, 3, 3)):

        nx, ny, nz = supercell

        _configuration(
            silicon,
            experiment / f"Si_phonon_{nx}{ny}{nz}",
            supercell=supercell,
        )

    assert len(phonopy_vasp.configurations_under(tmp_path)) == 2

    assert len(phonopy_vasp.discover(tmp_path).siblings) == 2


def test_an_empty_folder_names_the_experiment_folder(tmp_path):
    """The error has to say where the thing it wants would be."""

    found = phonopy_vasp.discover(tmp_path)

    assert found.problems
    assert phonopy_vasp.EXPERIMENT_DIR in found.problems[0]


def test_post_processing_accepts_the_parent_folder(silicon, tmp_path):
    """
    The round trip, entered one level too high on purpose.

    The analysis has to follow discovery down to the configuration, or
    it looks for phonopy_disp.yaml in the wrong place and writes its
    output outside the experiment.
    """

    root = _configuration(
        silicon, tmp_path / phonopy_vasp.EXPERIMENT_DIR / "Si_phonon_222"
    )

    run_fake_vasp(root)

    # Handed tmp_path, not root.
    result = VaspPhononCollect(None, tmp_path, mesh=(8, 8, 8)).run()

    assert result.root == root

    assert result.band is not None
    assert result.dos is not None

    # The outputs land beside the runs, inside the experiment.
    assert (root / phonopy_vasp.FORCE_SETS).is_file()
    assert (root / phonopy_vasp.BAND_YAML).is_file()


# ------------------------------------------------------------
# Thermal properties
# ------------------------------------------------------------


def test_post_processing_computes_thermal_properties(silicon, tmp_path):
    """
    The same harmonic thermodynamics the force-field route offers.

    Taken off the mesh the density of states already sampled, so they
    cost nothing beyond the integration itself.
    """

    root = _configuration(
        silicon, tmp_path / phonopy_vasp.EXPERIMENT_DIR / "Si_phonon_222"
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(
        None, root, mesh=(8, 8, 8), thermal=True, t_max=600.0, t_step=100.0
    ).run()

    assert result.thermal is not None

    thermal = result.thermal

    # Entropy and heat capacity vanish at absolute zero; the free
    # energy does not, and what is left is the zero-point energy.
    assert thermal.at(0.0)[1] == pytest.approx(0.0, abs=1e-9)
    assert thermal.at(0.0)[2] == pytest.approx(0.0, abs=1e-9)

    assert thermal.zero_point == pytest.approx(thermal.at(0.0)[0])

    assert (root / phonopy_vasp.THERMAL_YAML).is_file()


def test_thermal_properties_can_be_turned_off(silicon, tmp_path):

    root = _configuration(
        silicon, tmp_path / phonopy_vasp.EXPERIMENT_DIR / "Si_phonon_222"
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(
        None, root, mesh=(8, 8, 8), thermal=False
    ).run()

    assert result.thermal is None

    assert not (root / phonopy_vasp.THERMAL_YAML).is_file()


def test_the_thermal_numbers_reach_the_console(silicon, tmp_path):

    root = _configuration(
        silicon, tmp_path / phonopy_vasp.EXPERIMENT_DIR / "Si_phonon_222"
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(
        None, root, mesh=(8, 8, 8), thermal=True, t_max=600.0, t_step=100.0
    ).run()

    rows = dict(result.rows)

    assert "Zero-point energy" in rows
    assert "Cv at 300 K" in rows

    assert "kJ/mol" in rows["Zero-point energy"]
    assert "3NR" in rows["Cv at 300 K"]


def test_the_dos_panel_runs_on_vasp_force_constants(silicon, tmp_path):
    """
    One density of states for both routes.

    The panel takes a phonopy object and nothing else, so force
    constants from VASP go through it exactly as force-field ones do --
    which is the point of not giving the VASP branch its own half-built
    copy of the same thing.
    """

    from qvista.services import phonon_dos

    root = _configuration(
        silicon, tmp_path / phonopy_vasp.EXPERIMENT_DIR / "Si_phonon_222"
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(8, 8, 8)).run()

    dos = phonon_dos.run_dos(
        result.phonon_data.phonon, mesh=(10, 10, 10), projection="element"
    )

    assert dos.integral == pytest.approx(dos.expected_integral, rel=0.05)

    assert set(dos.curves) == {"Si"}


# ------------------------------------------------------------
# All four post-processing figures, off one pass
#
# The VASP branch used to end at band-with-DOS while the CHGNet branch
# offered four figures. Nothing about the forces justified the
# difference: reading a configuration back already builds the
# dispersion, and the other three readings are the same dispersion
# looked at differently. What was actually missing was the eigenvectors
# -- without them there is no element projection to draw -- so they are
# now asked for on the one pass that computes the bands.
# ------------------------------------------------------------


def test_the_collected_band_carries_element_projections(rocksalt, tmp_path):
    """
    Two species, so a projection has something to separate.

    Without this the VASP branch cannot draw a projected band at all,
    which is the one of the four figures that needs more than the
    frequencies.
    """

    root = tmp_path / phonopy_vasp.EXPERIMENT_DIR / "MgO_phonon_222"

    phonopy_vasp.write_configuration(
        root=root,
        structure=rocksalt,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(6, 6, 6)).run()

    projections = result.band.get("projections")

    assert projections, "no projections came back off the VASP route"

    assert set(projections) == {"Mg", "O"}

    assert result.band["symbols"] == ["Mg", "O"]


def test_the_projections_partition_each_mode(rocksalt, tmp_path):
    """
    Every mode is shared out between the species and nothing is lost.

    The weights are squared eigenvector components, so they sum to one
    per mode by construction -- which is what makes a marker area on the
    projected figure readable as a fraction.
    """

    import numpy as np

    root = tmp_path / phonopy_vasp.EXPERIMENT_DIR / "MgO_phonon_sum"

    phonopy_vasp.write_configuration(
        root=root,
        structure=rocksalt,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(6, 6, 6)).run()

    total = sum(
        np.asarray(weights, dtype=float)
        for weights in result.band["projections"].values()
    )

    assert np.allclose(total, 1.0, atol=1e-6)


def test_every_one_of_the_four_figures_draws(rocksalt, tmp_path):
    """
    One collected result, four figures, no second pass over the folder.

    Drawn rather than inspected: the projected figure in particular
    reaches for keys the band-with-DOS one never touches, and a missing
    one is only an error at drawing time.
    """

    import matplotlib

    matplotlib.use("Agg")

    from matplotlib.figure import Figure

    from qvista.visualization import phonon_plots

    root = tmp_path / phonopy_vasp.EXPERIMENT_DIR / "MgO_phonon_figs"

    phonopy_vasp.write_configuration(
        root=root,
        structure=rocksalt,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(6, 6, 6)).run()

    band_only = Figure()
    phonon_plots.draw_band(band_only, result.band)

    assert band_only.axes[0].lines

    with_dos = Figure()
    phonon_plots.draw_band_with_dos(with_dos, result.band, result.dos)

    assert len(with_dos.axes) == 2

    projected = Figure()
    phonon_plots.draw_projected_band(projected, result.band)

    assert projected.axes[0].collections, "nothing was projected"

    # The fourth is mode animation, which needs the phonopy object
    # rather than a figure.
    assert result.phonon_data.phonon is not None


def test_losing_the_projection_does_not_lose_the_bands(
    rocksalt, tmp_path, monkeypatch
):
    """
    A projection is an extra reading, not a precondition.

    If it fails the dispersion is already computed, and throwing it away
    would cost three working figures to protect one.
    """

    from qvista.analyses import phonon_vasp

    def explode(*_args, **_kwargs):
        raise RuntimeError("no eigenvectors today")

    monkeypatch.setattr(phonon_vasp, "project_eigenvectors", explode)

    root = tmp_path / phonopy_vasp.EXPERIMENT_DIR / "MgO_phonon_noproj"

    phonopy_vasp.write_configuration(
        root=root,
        structure=rocksalt,
        method=method(FINITE_DISPLACEMENT),
        supercell=(2, 2, 2),
        encut=520,
    )

    run_fake_vasp(root)

    result = VaspPhononCollect(None, root, mesh=(6, 6, 6)).run()

    assert result.band["frequencies"]

    assert not result.band.get("projections")

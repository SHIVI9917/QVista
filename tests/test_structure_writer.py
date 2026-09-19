"""
Tests for structure format conversion.

Output format is independent of input format, so every input must be
writable as every format its physics allows -- and the conversion has to
preserve the structure, not merely produce a file that parses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qvista.services.structure_analyzer import StructureAnalyzer
from qvista.services.structure_writer import (
    OUTPUT_FORMATS,
    StructureWriteError,
    convert,
    find_format,
    formats_for,
)

PERIODIC_FORMATS = ("poscar", "cif", "xyz")


def _poscar(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)

    return directory / "POSCAR"


# ------------------------------------------------------------
# Inputs, one per supported format
# ------------------------------------------------------------


@pytest.fixture
def source_poscar(tmp_path: Path) -> Path:
    """LiMnO2-like orthorhombic cell, written as POSCAR."""

    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.orthorhombic(2.869, 4.634, 5.833),
        ["Li", "Li", "O", "O", "Mn", "Mn"],
        [
            [0.75, 0.75, 0.88],
            [0.25, 0.25, 0.12],
            [0.75, 0.25, 0.86],
            [0.25, 0.75, 0.60],
            [0.75, 0.75, 0.36],
            [0.25, 0.25, 0.64],
        ],
    )

    path = _poscar(tmp_path, "source")
    structure.to(filename=str(path), fmt="poscar")

    return path


@pytest.fixture
def source_cif(tmp_path: Path, source_poscar: Path) -> Path:
    """The same structure as a CIF."""

    target = tmp_path / "source.cif"

    return convert(source_poscar, target, "cif")


@pytest.fixture
def source_xyz(tmp_path: Path, source_poscar: Path) -> Path:
    """The same structure as an extended XYZ."""

    target = tmp_path / "source.xyz"

    return convert(source_poscar, target, "xyz")


@pytest.fixture
def molecule_xyz(tmp_path: Path) -> Path:
    """A plain XYZ -- no lattice."""

    path = tmp_path / "water.xyz"

    path.write_text(
        "3\nwater\n"
        "O  0.000  0.000  0.000\n"
        "H  0.758  0.586  0.000\n"
        "H -0.758  0.586  0.000\n"
    )

    return path


# ------------------------------------------------------------
# Every input to every format
# ------------------------------------------------------------


@pytest.mark.parametrize("target_format", PERIODIC_FORMATS)
@pytest.mark.parametrize(
    "source_name",
    ["source_poscar", "source_cif", "source_xyz"],
)
def test_any_input_converts_to_any_format(
    request,
    tmp_path: Path,
    source_name: str,
    target_format: str,
) -> None:
    """
    The full conversion matrix: 3 inputs x 3 outputs.

    This is the requirement in one test -- output format does not depend
    on input format.
    """

    source = request.getfixturevalue(source_name)

    target = tmp_path / f"out_{source_name}_{target_format}"

    written = convert(source, target, target_format)

    assert written.exists()
    assert written.stat().st_size > 0

    # The result must be readable back as a periodic structure.
    parsed = StructureAnalyzer.parse(written)

    assert parsed.periodic


# ------------------------------------------------------------
# Conversion preserves the structure
# ------------------------------------------------------------


@pytest.mark.parametrize("target_format", PERIODIC_FORMATS)
def test_conversion_preserves_composition_and_cell(
    tmp_path: Path,
    source_poscar: Path,
    target_format: str,
) -> None:
    """Round-tripping must not change what the structure is."""

    import numpy as np

    original = StructureAnalyzer.parse(source_poscar).structure

    written = convert(
        source_poscar,
        tmp_path / f"round_{target_format}",
        target_format,
    )

    result = StructureAnalyzer.parse(written).structure

    assert result.composition.reduced_formula == (
        original.composition.reduced_formula
    )

    assert len(result) == len(original)

    assert result.volume == pytest.approx(original.volume, rel=1e-4)

    assert np.allclose(
        sorted(result.lattice.abc),
        sorted(original.lattice.abc),
        atol=1e-4,
    )


def test_xyz_output_keeps_the_lattice(
    tmp_path: Path,
    source_poscar: Path,
) -> None:
    """
    A periodic structure saved as XYZ must stay periodic.

    Plain XYZ would silently drop the cell, so extended XYZ is written and
    QVista reads it back as a crystal rather than a molecule.
    """

    written = convert(source_poscar, tmp_path / "kept.xyz", "xyz")

    text = written.read_text()

    assert "Lattice=" in text

    parsed = StructureAnalyzer.parse(written)

    assert parsed.periodic
    assert parsed.source_format == "extended XYZ"


# ------------------------------------------------------------
# Molecules
# ------------------------------------------------------------


def test_molecule_can_be_written_as_xyz(
    tmp_path: Path,
    molecule_xyz: Path,
) -> None:
    """A molecule has a valid XYZ representation."""

    written = convert(molecule_xyz, tmp_path / "mol.xyz", "xyz")

    parsed = StructureAnalyzer.parse(written)

    assert not parsed.periodic
    assert len(parsed.structure) == 3


@pytest.mark.parametrize("target_format", ["poscar", "cif"])
def test_molecule_to_periodic_format_is_refused(
    tmp_path: Path,
    molecule_xyz: Path,
    target_format: str,
) -> None:
    """
    A molecule has no cell, so it cannot become a POSCAR or CIF.

    Refusing is the point: inventing a box would produce a file that looks
    valid and describes something the user never supplied.
    """

    with pytest.raises(StructureWriteError) as excinfo:
        convert(molecule_xyz, tmp_path / "bad", target_format)

    message = str(excinfo.value)

    assert "lattice" in message.lower()
    assert "XYZ" in message

    assert not (tmp_path / "bad").exists()


def test_formats_offered_depend_on_periodicity() -> None:
    """The UI only offers formats that can actually be written."""

    periodic = {fmt.key for fmt in formats_for(True)}
    molecular = {fmt.key for fmt in formats_for(False)}

    assert periodic == {"poscar", "cif", "xyz"}
    assert molecular == {"xyz"}


# ------------------------------------------------------------
# Errors
# ------------------------------------------------------------


def test_unknown_format_is_rejected() -> None:
    """An unsupported key fails clearly."""

    with pytest.raises(StructureWriteError) as excinfo:
        find_format("pdb")

    assert "pdb" in str(excinfo.value)


def test_missing_source_is_reported(tmp_path: Path) -> None:
    """A missing input produces a readable message, not a traceback."""

    with pytest.raises(StructureWriteError) as excinfo:
        convert(tmp_path / "nope.cif", tmp_path / "out.cif", "cif")

    assert "not found" in str(excinfo.value).lower()


def test_every_format_has_ui_metadata() -> None:
    """Each format can be presented in a file dialog."""

    for fmt in OUTPUT_FORMATS:
        assert fmt.label
        assert fmt.default_name
        assert "(" in fmt.file_filter


# ------------------------------------------------------------
# Written files must be reopenable
# ------------------------------------------------------------


@pytest.mark.parametrize("target_format", PERIODIC_FORMATS)
def test_extensionless_target_gains_a_readable_name(
    tmp_path: Path,
    source_poscar: Path,
    target_format: str,
) -> None:
    """
    A file QVista writes must be a file QVista can reopen.

    Format detection reads the filename, so saving a CIF as ``mystruct``
    would produce a valid file the application could not load. The
    canonical suffix is added when it is missing.
    """

    written = convert(
        source_poscar,
        tmp_path / "mystruct",
        target_format,
    )

    assert written.exists()

    # The round trip is the actual requirement.
    parsed = StructureAnalyzer.parse(written)

    assert parsed.periodic


def test_conventional_poscar_name_is_left_alone(
    tmp_path: Path,
    source_poscar: Path,
) -> None:
    """A file named POSCAR is already detectable and keeps its name."""

    target = tmp_path / "written" / "POSCAR"
    target.parent.mkdir(parents=True, exist_ok=True)

    written = convert(source_poscar, target, "poscar")

    assert written.name == "POSCAR"

    assert StructureAnalyzer.parse(written).periodic


def test_correct_extension_is_not_doubled(
    tmp_path: Path,
    source_poscar: Path,
) -> None:
    """A name that already ends correctly is untouched."""

    written = convert(source_poscar, tmp_path / "good.cif", "cif")

    assert written.name == "good.cif"


@pytest.mark.parametrize(
    "filename",
    ["my_poscar_export.xyz", "CONTCAR_relaxed.xyz", "poscar.xyz"],
)
def test_xyz_named_after_a_vasp_file_keeps_its_lattice(
    tmp_path: Path,
    source_poscar: Path,
    filename: str,
) -> None:
    """
    An XYZ whose name contains "poscar" or "contcar" is still an XYZ.

    ASE's format detection matches those words anywhere in the basename
    and hands the file to the VASP parser, which fails. The failure was
    being swallowed, silently degrading a periodic structure to a molecule
    and losing the cell -- valid-looking output describing the wrong thing.
    """

    written = convert(source_poscar, tmp_path / filename, "xyz")

    parsed = StructureAnalyzer.parse(written)

    assert parsed.periodic, f"{filename} lost its lattice"
    assert parsed.source_format == "extended XYZ"

    assert parsed.structure.volume == pytest.approx(
        StructureAnalyzer.parse(source_poscar).structure.volume,
        rel=1e-4,
    )

"""
Tests for the elastic tensor workflow.

The deformation geometry and the tensor fit are pure numerics, so they are
tested against an analytic case: synthetic stresses generated from a known
cubic tensor must fit back to that same tensor. Nothing here invokes
MatterSim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvista.analyses.base import AnalysisError
from qvista.analyses.elastic import (
    STRAIN_AMPLITUDES,
    STRAIN_MODES,
    VOIGT_INDEX,
    ElasticAnalysis,
    born_stability,
    compliance_tensor,
    deformation_gradient,
    derive_moduli,
    fit_elastic_tensor,
    strain_lattice,
)
from qvista.services.project_service import ProjectService

# A physically sensible cubic tensor, roughly silicon-like (GPa).
C11, C12, C44 = 166.0, 64.0, 80.0


def cubic_tensor(c11: float = C11, c12: float = C12, c44: float = C44):
    """Build a cubic elastic tensor in Voigt form."""

    tensor = np.zeros((6, 6))

    for i in range(3):
        for j in range(3):
            tensor[i, j] = c11 if i == j else c12

    for i in range(3, 6):
        tensor[i, i] = c44

    return tensor


def _poscar(tmp_path: Path, name: str) -> Path:
    """A POSCAR in its own directory, so fixtures cannot collide."""

    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)

    return directory / "POSCAR"


# ------------------------------------------------------------
# Deformation gradient
# ------------------------------------------------------------


def test_normal_strain_is_diagonal() -> None:
    """A normal mode adds the strain to one diagonal entry only."""

    gradient = deformation_gradient("xx", 0.01)

    expected = np.eye(3)
    expected[0, 0] = 1.01

    assert np.allclose(gradient, expected)


def test_shear_strain_is_symmetric_and_halved() -> None:
    """
    Engineering shear puts eps/2 on both off-diagonal entries.

    That is what makes the fitted slope equal C44 directly: the Voigt
    component is the engineering strain, which is twice the tensor strain.
    """

    gradient = deformation_gradient("xy", 0.01)

    assert gradient[0, 1] == pytest.approx(0.005)
    assert gradient[1, 0] == pytest.approx(0.005)

    assert np.allclose(gradient, gradient.T)

    assert gradient[0, 0] == pytest.approx(1.0)
    assert gradient[2, 2] == pytest.approx(1.0)


def test_unknown_mode_is_rejected() -> None:
    """A bad mode name fails as AnalysisError, not KeyError."""

    with pytest.raises(AnalysisError):
        deformation_gradient("qq", 0.01)


def test_strain_lattice_uses_row_vector_convention() -> None:
    """
    Each lattice vector must be transformed individually.

    Rows are lattice vectors, so ``F`` must contract against the Cartesian
    index: ``M @ F.T``. This checks the transformed rows really are
    ``F @ a_i``.
    """

    half = 5.43 / 2.0

    # An FCC primitive cell -- deliberately non-diagonal.
    matrix = np.array(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]]
    )

    gradient = deformation_gradient("xy", 0.02)

    strained = strain_lattice(matrix, gradient)

    for index in range(3):
        assert np.allclose(strained[index], gradient @ matrix[index])


def test_row_and_column_conventions_differ_on_a_nondiagonal_cell() -> None:
    """
    Guard against reverting to ``F @ M``.

    For a diagonal cell the two forms agree, which is why the mistake
    survives on cubic conventional cells. On a primitive FCC cell they do
    not, and the wrong one applies a different strain.
    """

    half = 5.43 / 2.0

    matrix = np.array(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]]
    )

    gradient = deformation_gradient("xy", 0.02)

    correct = strain_lattice(matrix, gradient)
    reference_form = gradient @ matrix

    assert not np.allclose(correct, reference_form)

    # ...and they do agree for a diagonal cell, which is the trap.
    diagonal = np.diag([5.43, 5.43, 5.43])

    assert np.allclose(
        strain_lattice(diagonal, gradient),
        gradient @ diagonal,
    )


# ------------------------------------------------------------
# Tensor fit
# ------------------------------------------------------------


def synthetic_measurements(tensor, amplitudes=STRAIN_AMPLITUDES):
    """
    Generate the stresses a material with ``tensor`` would produce.

    sigma = C @ epsilon, with epsilon a unit Voigt strain in one mode.
    """

    measurements = {}

    for mode in STRAIN_MODES:

        rows = []

        for amplitude in amplitudes:

            strain = np.zeros(6)
            strain[VOIGT_INDEX[mode]] = amplitude

            rows.append((amplitude, tensor @ strain))

        measurements[mode] = rows

    return measurements


def test_known_cubic_tensor_is_recovered() -> None:
    """The analytic case: synthetic stresses must fit back to the tensor."""

    expected = cubic_tensor()

    fitted = fit_elastic_tensor(synthetic_measurements(expected))

    assert np.allclose(fitted, expected, atol=1e-8)

    assert fitted[0, 0] == pytest.approx(C11)
    assert fitted[0, 1] == pytest.approx(C12)
    assert fitted[3, 3] == pytest.approx(C44)


def test_fit_is_symmetric() -> None:
    """The fitted tensor is symmetrised, as a real tensor must be."""

    expected = cubic_tensor()

    measurements = synthetic_measurements(expected)

    # Perturb one direction only, so the raw fit would be asymmetric.
    mode = "xx"
    measurements[mode] = [
        (amplitude, stress + np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0]) * amplitude)
        for amplitude, stress in measurements[mode]
    ]

    fitted = fit_elastic_tensor(measurements)

    assert np.allclose(fitted, fitted.T)


def test_fit_is_robust_to_noise() -> None:
    """A linear fit over four amplitudes should absorb small noise."""

    expected = cubic_tensor()

    generator = np.random.default_rng(0)

    measurements = {
        mode: [
            (
                amplitude,
                stress + generator.normal(0.0, 1e-4, size=6),
            )
            for amplitude, stress in rows
        ]
        for mode, rows in synthetic_measurements(expected).items()
    }

    fitted = fit_elastic_tensor(measurements)

    assert np.allclose(fitted, expected, atol=1.0)


def test_fit_rejects_a_single_point() -> None:
    """One strain amplitude cannot define a slope."""

    measurements = synthetic_measurements(cubic_tensor(), amplitudes=(0.01,))

    with pytest.raises(AnalysisError) as excinfo:
        fit_elastic_tensor(measurements)

    assert "two" in str(excinfo.value)


# ------------------------------------------------------------
# Derived moduli
# ------------------------------------------------------------


def test_cubic_moduli_match_closed_form() -> None:
    """
    For a cubic crystal the Voigt/Reuss averages have closed forms.

    K is exact in both averages: KV = KR = (C11 + 2*C12)/3.
    """

    moduli = derive_moduli(cubic_tensor())

    expected_bulk = (C11 + 2.0 * C12) / 3.0

    assert moduli.bulk_voigt == pytest.approx(expected_bulk)
    assert moduli.bulk_reuss == pytest.approx(expected_bulk)
    assert moduli.bulk_hill == pytest.approx(expected_bulk)

    expected_shear_voigt = (C11 - C12 + 3.0 * C44) / 5.0

    assert moduli.shear_voigt == pytest.approx(expected_shear_voigt)

    # Hill sits between the two bounds.
    assert (
        min(moduli.shear_voigt, moduli.shear_reuss)
        <= moduli.shear_hill
        <= max(moduli.shear_voigt, moduli.shear_reuss)
    )


def test_isotropic_tensor_has_zero_anisotropy() -> None:
    """
    A_U vanishes for an isotropic material.

    Isotropy requires C44 = (C11 - C12)/2.
    """

    c11, c12 = 166.0, 64.0
    c44 = (c11 - c12) / 2.0

    moduli = derive_moduli(cubic_tensor(c11, c12, c44))

    assert moduli.anisotropy == pytest.approx(0.0, abs=1e-9)

    assert moduli.shear_voigt == pytest.approx(moduli.shear_reuss)


def test_poisson_and_pugh_are_consistent() -> None:
    """Derived ratios follow from the Hill averages."""

    moduli = derive_moduli(cubic_tensor())

    assert moduli.pugh == pytest.approx(
        moduli.bulk_hill / moduli.shear_hill
    )

    expected_youngs = (
        9.0
        * moduli.bulk_hill
        * moduli.shear_hill
        / (3.0 * moduli.bulk_hill + moduli.shear_hill)
    )

    assert moduli.youngs == pytest.approx(expected_youngs)

    # A physically sensible material.
    assert -1.0 < moduli.poisson < 0.5


def test_singular_tensor_is_reported_clearly() -> None:
    """An uninvertible tensor must not raise a raw LinAlgError."""

    with pytest.raises(AnalysisError) as excinfo:
        derive_moduli(np.zeros((6, 6)))

    assert "singular" in str(excinfo.value).lower()


# ------------------------------------------------------------
# Compliance
# ------------------------------------------------------------


def test_the_compliance_is_the_inverse_of_the_stiffness() -> None:

    tensor = cubic_tensor()

    product = compliance_tensor(tensor) @ tensor

    assert np.allclose(product, np.eye(6), atol=1e-10)


def test_the_compliance_gives_the_directional_youngs_modulus() -> None:
    """
    ``1 / S_11`` is Young's modulus along x.

    This is the reason for showing S at all: the number is sitting on
    its diagonal, and there is no way to read it off C.
    """

    tensor = cubic_tensor()

    compliance = compliance_tensor(tensor)

    # For a cubic crystal the closed form is known exactly.
    expected = (
        (C11 - C12) * (C11 + 2.0 * C12) / (C11 + C12)
    )

    assert 1.0 / compliance[0, 0] == pytest.approx(expected)


def test_a_singular_stiffness_has_no_compliance() -> None:

    with pytest.raises(AnalysisError) as excinfo:
        compliance_tensor(np.zeros((6, 6)))

    assert "singular" in str(excinfo.value).lower()


# ------------------------------------------------------------
# Table rows
# ------------------------------------------------------------


def test_the_moduli_table_carries_symbols_and_units_apart() -> None:
    """
    Five fields per row, so the panel can column them.

    Running the unit into the value, as the console rows do, makes the
    value column unsortable and unreadable down the page.
    """

    rows = derive_moduli(cubic_tensor()).as_table()

    assert len(rows) == 10

    for section, name, symbol, value, unit in rows:

        assert section and name
        assert unit in ("GPa", "")

        # The value is a bare number: no unit, no symbol.
        float(value)


def test_the_moduli_table_and_the_console_rows_agree() -> None:

    moduli = derive_moduli(cubic_tensor())

    # Keyed on the section too: "Hill average" is a row under both the
    # bulk and the shear modulus, and they are different numbers.
    table = {
        (section, name): value
        for section, name, _, value, _ in moduli.as_table()
    }

    assert table[("Bulk modulus", "Hill average")] == f"{moduli.bulk_hill:.2f}"
    assert (
        table[("Shear modulus", "Hill average")]
        == f"{moduli.shear_hill:.2f}"
    )
    assert table[("Derived", "Young's modulus")] == f"{moduli.youngs:.2f}"
    assert table[("Derived", "Poisson ratio")] == f"{moduli.poisson:.4f}"


def test_the_moduli_table_groups_bulk_shear_and_derived() -> None:

    sections = []

    for section, *_ in derive_moduli(cubic_tensor()).as_table():

        if section not in sections:
            sections.append(section)

    assert sections == ["Bulk modulus", "Shear modulus", "Derived"]


def test_the_stability_table_says_which_way_it_went() -> None:

    good = born_stability(cubic_tensor()).as_table()

    assert good[0][3] == "satisfied"

    soft = cubic_tensor()
    soft[3][3] = soft[4][4] = soft[5][5] = -10.0

    assert born_stability(soft).as_table()[0][3] == "VIOLATED"


# ------------------------------------------------------------
# Born stability
# ------------------------------------------------------------


def test_sensible_tensor_is_stable() -> None:
    """A real cubic tensor satisfies the Born criteria."""

    report = born_stability(cubic_tensor())

    assert report.stable
    assert not report.failures
    assert min(report.eigenvalues) > 0.0


def test_unstable_tensor_is_flagged() -> None:
    """A negative shear constant must be caught."""

    report = born_stability(cubic_tensor(c44=-10.0))

    assert not report.stable
    assert report.failures

    assert "born" in " ".join(report.failures).lower()


def test_soft_tensor_violating_positive_definiteness() -> None:
    """C12 larger than C11 makes the tensor indefinite."""

    report = born_stability(cubic_tensor(c11=50.0, c12=120.0))

    assert not report.stable


# ------------------------------------------------------------
# Guards
# ------------------------------------------------------------


@pytest.fixture
def plain_xyz(tmp_path: Path) -> Path:
    """A molecule -- no lattice, no elastic tensor."""

    path = tmp_path / "water.xyz"

    path.write_text(
        "3\nwater\n"
        "O  0.000  0.000  0.000\n"
        "H  0.758  0.586  0.000\n"
        "H -0.758  0.586  0.000\n"
    )

    return path


@pytest.fixture
def triclinic_p1(tmp_path: Path) -> Path:
    """Three species in general positions -- genuinely P1."""

    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.from_parameters(4.0, 4.5, 5.0, 71.0, 82.0, 95.0),
        ["Si", "O", "N"],
        [[0.01, 0.02, 0.03], [0.31, 0.27, 0.23], [0.62, 0.11, 0.55]],
    )

    path = _poscar(tmp_path, "triclinic")
    structure.to(filename=str(path), fmt="poscar")

    return path


def test_plain_xyz_is_refused(plain_xyz: Path) -> None:
    """A molecule has no elastic tensor; refuse before any relaxation."""

    analysis = ElasticAnalysis(ProjectService.load(plain_xyz))

    with pytest.raises(AnalysisError) as excinfo:
        analysis.run()

    message = str(excinfo.value)

    assert "lattice" in message.lower()
    assert "Traceback" not in message


def test_p1_refinement_is_a_no_op(triclinic_p1: Path) -> None:
    """
    Refinement cannot reorient a P1 structure.

    This is the condition the workflow must warn about: the tensor axes stay
    arbitrary, so the shear constants are not what their labels claim.
    """

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from qvista.services.structure_analyzer import StructureAnalyzer

    structure = StructureAnalyzer.parse(triclinic_p1).structure

    analyzer = SpacegroupAnalyzer(structure, symprec=0.01)

    assert analyzer.get_space_group_number() == 1

    refined = analyzer.get_refined_structure()

    after = SpacegroupAnalyzer(refined, symprec=0.01)

    assert after.get_space_group_number() == 1


def test_p1_result_is_marked_arbitrary() -> None:
    """A P1 refinement sets the flag the panel and console warn on."""

    from qvista.analyses.elastic import ElasticResult

    result = ElasticResult(
        tensor=cubic_tensor(),
        spacegroup_after="P1",
        spacegroup_number_after=1,
        orientation_is_arbitrary=True,
    )

    csv = result.to_csv()

    assert "WARNING" in csv
    assert "arbitrary" in csv


def test_csv_export_contains_tensor_and_moduli() -> None:
    """CSV export carries the numbers, not just a summary."""

    from qvista.analyses.elastic import ElasticResult

    tensor = cubic_tensor()

    result = ElasticResult(
        tensor=tensor,
        moduli=derive_moduli(tensor),
        stability=born_stability(tensor),
        force_field="test",
    )

    csv = result.to_csv()

    assert "166.000000" in csv
    assert "Bulk K (Hill)" in csv
    assert "Born stability" in csv

    # Six rows of six numbers.
    numeric_rows = [
        line
        for line in csv.splitlines()
        if line and not line.startswith("#") and line.count(",") == 5
    ]

    assert len(numeric_rows) == 6

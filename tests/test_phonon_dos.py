"""
Tests for the phonon density of states.

Force constants come from Lennard-Jones rather than a force field: it is
analytic, it needs no download, and fcc is its ground state, so a cell
built at the LJ minimum is dynamically stable by construction.  That
matters more than it sounds -- the sum rule, the projection weights and
the approach of Cv to Dulong-Petit are only meaningful checks on a
spectrum with no imaginary modes, and an arbitrary test potential does
not give one.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services import phonon_dos as pd

#: The LJ parameters used throughout. Sigma sets the length scale.
SIGMA = 3.4
EPSILON = 0.3


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------


def _force_constants(cell, supercell=(2, 2, 2), distance=0.02):
    """Run LJ over phonopy's displacements and produce force constants."""

    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from phonopy import Phonopy

    phonon = Phonopy(
        cell,
        supercell_matrix=np.diag(supercell),
        primitive_matrix=np.eye(3),
    )

    phonon.generate_displacements(distance=distance)

    calculator = LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=8.5)

    forces = []

    for displaced in phonon.supercells_with_displacements:

        atoms = Atoms(
            symbols=displaced.symbols,
            cell=displaced.cell,
            scaled_positions=displaced.scaled_positions,
            pbc=True,
        )

        atoms.calc = calculator

        forces.append(atoms.get_forces())

    phonon.forces = np.array(forces)
    phonon.produce_force_constants()

    return phonon


@pytest.fixture(scope="module")
def stable_phonon():
    """
    Four atoms of fcc at the Lennard-Jones minimum.

    The species labels alternate, which LJ ignores -- so the lattice is
    plain fcc and stable -- while the two masses differ, which is what
    makes an element projection say anything.
    """

    from phonopy.structure.atoms import PhonopyAtoms

    a = 2 ** (1 / 6) * SIGMA * np.sqrt(2.0)

    cell = PhonopyAtoms(
        symbols=["Na", "Cl", "Na", "Cl"],
        cell=np.eye(3) * a,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
    )

    return _force_constants(cell)


@pytest.fixture(scope="module")
def unstable_phonon():
    """Rocksalt at a spacing Lennard-Jones dislikes: imaginary modes."""

    from phonopy.structure.atoms import PhonopyAtoms

    a = 5.6

    cell = PhonopyAtoms(
        symbols=["Na", "Cl"],
        cell=[
            [0.0, a / 2, a / 2],
            [a / 2, 0.0, a / 2],
            [a / 2, a / 2, 0.0],
        ],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from phonopy import Phonopy

    phonon = Phonopy(
        cell,
        supercell_matrix=np.diag((2, 2, 2)),
        primitive_matrix=np.eye(3),
    )

    phonon.generate_displacements(distance=0.02)

    calculator = LennardJones(sigma=2.6, epsilon=0.05, rc=8.5)

    forces = []

    for displaced in phonon.supercells_with_displacements:

        atoms = Atoms(
            symbols=displaced.symbols,
            cell=displaced.cell,
            scaled_positions=displaced.scaled_positions,
            pbc=True,
        )

        atoms.calc = calculator

        forces.append(atoms.get_forces())

    phonon.forces = np.array(forces)
    phonon.produce_force_constants()

    return phonon


MESH = (12, 12, 12)


# ------------------------------------------------------------
# The sum rule
# ------------------------------------------------------------


def test_the_dos_integrates_to_three_modes_per_atom(stable_phonon) -> None:
    """
    The one identity a density of states cannot get wrong.

    A cell of N atoms has exactly 3N vibrational modes, so the area
    under the curve is fixed. A departure means the mesh was too coarse
    or the frequency range clipped the spectrum, and there is no other
    cheap way to notice either.
    """

    result = pd.run_dos(stable_phonon, mesh=MESH)

    assert result.natoms == 4
    assert result.expected_integral == pytest.approx(12.0)

    assert result.integral == pytest.approx(12.0, abs=0.05)


def test_the_stable_cell_has_no_imaginary_modes(stable_phonon) -> None:
    """If this fails, every other check on this fixture means nothing."""

    result = pd.run_dos(stable_phonon, mesh=MESH)

    assert not result.has_imaginary
    assert result.lowest >= pd.IMAGINARY_TOLERANCE


# ------------------------------------------------------------
# Projections
# ------------------------------------------------------------


def test_a_total_dos_has_no_projections(stable_phonon) -> None:

    result = pd.run_dos(stable_phonon, mesh=MESH)

    assert result.curves == {}
    assert result.projection == "total"


def test_a_total_dos_uses_the_symmetry_reduced_mesh(stable_phonon) -> None:
    """The cheap path, taken whenever no projection was asked for."""

    result = pd.run_dos(stable_phonon, mesh=MESH)

    assert result.qpoints < MESH[0] * MESH[1] * MESH[2]


def test_a_projection_samples_the_whole_zone(stable_phonon) -> None:
    """
    phonopy will not give eigenvectors off a symmetry-reduced mesh.

    It raises rather than projecting onto the wrong atoms, so the whole
    zone has to be sampled -- which is the real cost of a projected DOS
    and is worth being explicit about.
    """

    result = pd.run_dos(stable_phonon, mesh=MESH, projection="element")

    assert result.qpoints == MESH[0] * MESH[1] * MESH[2]

    assert any("eigenvectors" in note for note in result.notes)


def test_the_projections_sum_back_to_the_total(stable_phonon) -> None:
    """Projecting must partition the modes, not invent or lose any."""

    result = pd.run_dos(stable_phonon, mesh=MESH, projection="element")

    stacked = sum(result.curves.values())

    assert np.abs(stacked - result.total).max() < 1e-9


def test_each_element_holds_three_modes_per_atom(stable_phonon) -> None:

    result = pd.run_dos(stable_phonon, mesh=MESH, projection="element")

    weights = dict(pd.band_weights(result))

    assert set(weights) == {"Na", "Cl"}

    # Two atoms of each species, so six modes each.
    for element, modes in weights.items():
        assert modes == pytest.approx(6.0, abs=0.05)


def test_an_atom_projection_keeps_the_atoms_apart(stable_phonon) -> None:

    result = pd.run_dos(stable_phonon, mesh=MESH, projection="atom")

    assert list(result.curves) == ["1 Na", "2 Cl", "3 Na", "4 Cl"]

    for modes in dict(pd.band_weights(result)).values():
        assert modes == pytest.approx(3.0, abs=0.05)


def test_the_lighter_element_sits_at_higher_frequency(
    stable_phonon,
) -> None:
    """
    The physical content of an element projection.

    Both sublattices are the same fcc lattice here and feel the same
    potential, so the only thing separating them is mass -- and the
    lighter one has to carry the higher frequencies.
    """

    result = pd.run_dos(stable_phonon, mesh=MESH, projection="element")

    def centre(values):
        weight = np.trapezoid(values, result.frequencies)
        return float(
            np.trapezoid(values * result.frequencies, result.frequencies)
            / weight
        )

    # Na is 22.99, Cl is 35.45.
    assert centre(result.curves["Na"]) > centre(result.curves["Cl"])


# ------------------------------------------------------------
# Grouping, without phonopy
# ------------------------------------------------------------


def test_grouping_by_element_sums_the_atoms_of_a_species() -> None:

    projected = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ]
    )

    curves = pd.group_projections(projected, ["Si", "O", "Si"], "element")

    assert list(curves) == ["Si", "O"]

    assert curves["Si"] == pytest.approx([2.0, 3.0, 4.0])
    assert curves["O"] == pytest.approx([0.5, 0.5, 0.5])


def test_grouping_by_atom_numbers_from_one() -> None:

    curves = pd.group_projections(np.zeros((2, 3)), ["Si", "O"], "atom")

    assert list(curves) == ["1 Si", "2 O"]


def test_grouping_refuses_a_mismatched_projection() -> None:

    with pytest.raises(pd.PhononDosError, match="3 projections"):
        pd.group_projections(np.zeros((3, 4)), ["Si", "O"], "element")


def test_grouping_refuses_an_unknown_projection() -> None:

    with pytest.raises(pd.PhononDosError):
        pd.group_projections(np.zeros((2, 4)), ["Si", "O"], "spin")


# ------------------------------------------------------------
# Units
# ------------------------------------------------------------


def test_converting_the_axis_preserves_the_area() -> None:
    """
    The mistake this function exists to prevent.

    A DOS is a density per unit frequency, so stretching the axis by 33
    without dividing the curve by 33 multiplies the number of modes by
    33 -- and the sum rule is the only thing that would have caught it.
    """

    frequencies = np.linspace(0.0, 10.0, 501)
    dos = np.exp(-((frequencies - 5.0) ** 2))

    before = np.trapezoid(dos, frequencies)

    for unit in pd.UNITS:

        moved, scaled = pd.convert(frequencies, dos, unit)

        assert np.trapezoid(scaled, moved) == pytest.approx(before)


def test_the_conversion_factors_are_the_expected_ones() -> None:

    assert pd.UNITS["THz"].factor == 1.0
    assert pd.UNITS["cm^-1"].factor == pytest.approx(33.35641, abs=1e-4)
    assert pd.UNITS["meV"].factor == pytest.approx(4.13567, abs=1e-4)


def test_an_unknown_unit_is_refused() -> None:

    with pytest.raises(pd.PhononDosError, match="Hartree"):
        pd.convert(np.zeros(3), np.zeros(3), "Hartree")


def test_the_sum_rule_survives_every_unit(stable_phonon) -> None:

    for unit in pd.UNITS:

        result = pd.run_dos(stable_phonon, mesh=MESH, unit=unit)

        area = float(np.trapezoid(result.total, result.frequencies))

        assert area == pytest.approx(12.0, abs=0.05)


# ------------------------------------------------------------
# Broadening
# ------------------------------------------------------------


def test_smearing_is_recorded_and_flagged(stable_phonon) -> None:

    result = pd.run_dos(
        stable_phonon, mesh=MESH, method="smearing", sigma=0.2
    )

    assert result.method == "smearing"
    assert result.sigma == 0.2

    assert any("smearing" in note.lower() for note in result.notes)


def test_the_tetrahedron_leaves_no_sigma_behind(stable_phonon) -> None:
    """A width that was never used must not be reported as if it was."""

    result = pd.run_dos(stable_phonon, mesh=MESH, sigma=0.5)

    assert result.method == "tetrahedron"
    assert result.sigma is None


def test_smearing_broadens_the_curve(stable_phonon) -> None:
    """Wider smearing must flatten the peaks, not sharpen them."""

    narrow = pd.run_dos(
        stable_phonon, mesh=MESH, method="smearing", sigma=0.1
    )
    wide = pd.run_dos(
        stable_phonon, mesh=MESH, method="smearing", sigma=0.5
    )

    assert wide.total.max() < narrow.total.max()


# ------------------------------------------------------------
# Refusals
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"projection": "spin"},
        {"method": "lorentzian"},
        {"unit": "Hartree"},
        {"mesh": (0, 4, 4)},
        {"mesh": (4, 4)},
        {"method": "smearing", "sigma": 0.0},
        {"method": "smearing", "sigma": -1.0},
    ],
)
def test_bad_settings_are_refused_before_anything_runs(
    stable_phonon,
    kwargs,
) -> None:

    with pytest.raises(pd.PhononDosError):
        pd.run_dos(stable_phonon, **kwargs)


# ------------------------------------------------------------
# Thermal properties
# ------------------------------------------------------------


def test_thermal_properties_vanish_at_absolute_zero(
    stable_phonon,
) -> None:
    """
    Entropy and heat capacity go to zero; the free energy does not.

    What is left at 0 K is the zero-point energy, which is the whole
    reason a harmonic free energy is worth computing.
    """

    pd.run_dos(stable_phonon, mesh=MESH)

    thermal = pd.run_thermal(stable_phonon, t_max=1000, t_step=100)

    free, entropy, capacity = thermal.at(0.0)

    assert entropy == pytest.approx(0.0, abs=1e-9)
    assert capacity == pytest.approx(0.0, abs=1e-9)

    assert free > 0.0
    assert thermal.zero_point == pytest.approx(free)


def test_heat_capacity_approaches_dulong_petit(stable_phonon) -> None:
    """
    Cv -> 3NR as the temperature rises, from below.

    This is the check that the mesh, the frequencies and the integration
    all hang together: it fails if any of them is wrong, and it cannot
    be satisfied by accident.
    """

    pd.run_dos(stable_phonon, mesh=MESH)

    thermal = pd.run_thermal(stable_phonon, t_max=2000, t_step=100)

    hot = thermal.at(2000.0)[2]

    assert hot == pytest.approx(thermal.dulong_petit, rel=0.01)

    assert hot <= thermal.dulong_petit


def test_free_energy_falls_and_entropy_rises(stable_phonon) -> None:

    pd.run_dos(stable_phonon, mesh=MESH)

    thermal = pd.run_thermal(stable_phonon, t_max=1000, t_step=100)

    assert np.all(np.diff(thermal.free_energy) < 0.0)
    assert np.all(np.diff(thermal.entropy) > 0.0)


def test_a_stable_cell_gives_trustworthy_thermal_properties(
    stable_phonon,
) -> None:

    pd.run_dos(stable_phonon, mesh=MESH)

    thermal = pd.run_thermal(stable_phonon, t_max=500, t_step=100)

    assert thermal.trustworthy
    assert thermal.notes == []


def test_an_empty_temperature_range_is_refused(stable_phonon) -> None:

    pd.run_dos(stable_phonon, mesh=MESH)

    with pytest.raises(pd.PhononDosError):
        pd.run_thermal(stable_phonon, t_min=500, t_max=100)

    with pytest.raises(pd.PhononDosError):
        pd.run_thermal(stable_phonon, t_max=500, t_step=0.0)


# ------------------------------------------------------------
# Imaginary modes
# ------------------------------------------------------------


def test_imaginary_modes_are_reported(unstable_phonon) -> None:

    result = pd.run_dos(unstable_phonon, mesh=MESH)

    assert result.has_imaginary
    assert result.lowest < pd.IMAGINARY_TOLERANCE

    assert any("maginary" in note for note in result.notes)


def test_thermal_properties_are_disowned_when_modes_are_imaginary(
    unstable_phonon,
) -> None:
    """
    Harmonic thermodynamics assumes every mode is a real oscillator.

    Phonopy will still return numbers for an unstable structure, and
    they look perfectly plausible, so the warning is the only thing
    standing between the user and a meaningless free energy.
    """

    pd.run_dos(unstable_phonon, mesh=MESH)

    thermal = pd.run_thermal(unstable_phonon, t_max=500, t_step=100)

    assert not thermal.trustworthy
    assert thermal.notes


# ------------------------------------------------------------
# Reading the curve
# ------------------------------------------------------------


def test_peaks_come_back_tallest_first(stable_phonon) -> None:

    result = pd.run_dos(stable_phonon, mesh=MESH)

    peaks = pd.peak_frequencies(result, count=5)

    assert len(peaks) >= 2

    heights = [height for _, height in peaks]

    assert heights == sorted(heights, reverse=True)


def test_peaks_are_kept_apart() -> None:
    """
    Two samples either side of one maximum are not two peaks.

    Without a separation the list fills up with neighbouring points off
    the same feature and says nothing about the rest of the spectrum.
    """

    frequencies = np.linspace(0.0, 10.0, 1001)

    dos = (
        np.exp(-((frequencies - 2.0) ** 2) / 0.02)
        + np.exp(-((frequencies - 7.0) ** 2) / 0.02)
    )

    result = pd.DosResult(
        frequencies=frequencies, total=dos, natoms=1
    )

    peaks = pd.peak_frequencies(result, count=5)

    assert len(peaks) == 2

    assert {round(frequency, 1) for frequency, _ in peaks} == {2.0, 7.0}


def test_a_curve_too_short_to_have_peaks_has_none() -> None:

    result = pd.DosResult(
        frequencies=np.array([0.0, 1.0]),
        total=np.array([1.0, 2.0]),
        natoms=1,
    )

    assert pd.peak_frequencies(result) == []


def test_band_weights_are_empty_without_a_projection() -> None:

    result = pd.DosResult(
        frequencies=np.linspace(0.0, 1.0, 10),
        total=np.ones(10),
        natoms=1,
    )

    assert pd.band_weights(result) == []

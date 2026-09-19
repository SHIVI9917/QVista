"""
Tests for the Bader tools.

The densities here are analytic -- sums of Gaussians on a grid -- which
is what makes them worth testing against: where the zero-flux surface
between two identical Gaussians falls is known exactly, so the partition
can be checked against the right answer rather than against itself.

The end-to-end tests write real ``CHGCAR``/``AECCAR`` files with
pymatgen and read them back through :func:`qvista.services.bader.analyse`,
so the file-reading path is exercised rather than stubbed.  They are
small: a handful of atoms on a coarse grid, which runs in about a second.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Chgcar, Poscar

from qvista.services import bader as bd


# ------------------------------------------------------------
# Analytic densities
# ------------------------------------------------------------


#: Gaussian width in angstrom, not in fractional coordinates.  Wide
#: enough to be resolved: a peak narrower than the grid spacing lands on
#: one or two points and underflows to zero everywhere else, which makes
#: the whole cell a plateau and the partition meaningless.
DEFAULT_WIDTH = 0.7


def gaussians(shape, lattice, centres, widths=None, heights=None):
    """
    A sum of periodic Gaussians, in VASP's ``rho * V`` convention.

    Centres are fractional; ``widths`` are in angstrom.  The nearest
    periodic image is used for each one so a Gaussian near a face does
    not get cut in half.
    """

    centres = np.atleast_2d(np.asarray(centres, dtype=float))

    widths = (
        np.full(len(centres), DEFAULT_WIDTH)
        if widths is None
        else np.broadcast_to(
            np.asarray(widths, dtype=float), (len(centres),)
        )
    )
    heights = (
        np.ones(len(centres))
        if heights is None
        else np.broadcast_to(
            np.asarray(heights, dtype=float), (len(centres),)
        )
    )

    axes = [np.arange(n) / float(n) for n in shape]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    total = np.zeros(shape, dtype=float)

    for centre, width, height in zip(centres, widths, heights):

        offset = grid - centre
        offset -= np.round(offset)

        distance = np.linalg.norm(offset @ np.asarray(lattice, float), axis=-1)

        total += height * np.exp(-(distance ** 2) / (2.0 * width ** 2))

    return total


def symmetric_pair(shape, lattice, centre=(0.25, 0.25, 0.25)):
    """
    Two identical peaks, symmetric to the last bit.

    Evaluating a Gaussian at 0.25 and another at 0.75 gives a density
    that is symmetric only to about 5e-16, because the two coordinate
    subtractions round differently.  That is not good enough here: the
    plane between two identical peaks is ill conditioned -- the density
    gradient there runs *along* the surface, so which side a grid point
    falls on is decided by the last bits -- and a 5e-16 asymmetry in the
    input moves about a tenth of a percent of the charge across.

    Rolling one peak onto its own translate instead only permutes
    values, so the result is exactly invariant under that translation and
    any asymmetry that comes out is the partitioner's own.
    """

    half = tuple(n // 2 for n in shape)

    peak = gaussians(shape, lattice, [centre])

    return peak + np.roll(peak, half, axis=(0, 1, 2))


def two_atom_cell(a=6.0, symbols=("Na", "Cl")):
    return Structure(
        Lattice.cubic(a),
        list(symbols),
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )


# ------------------------------------------------------------
# Grid geometry
# ------------------------------------------------------------


def test_there_are_twenty_six_neighbours():
    """The 3x3x3 block around a point, minus the point itself."""

    assert bd.OFFSETS.shape == (26, 3)
    assert not (bd.OFFSETS == 0).all(axis=1).any()


def test_step_lengths_are_the_real_distances():
    """A face, an edge and a body diagonal of one grid cell."""

    lengths = bd.step_lengths((10, 10, 10), np.eye(3) * 10.0)

    steps = np.abs(bd.OFFSETS).sum(axis=1)

    assert np.allclose(lengths[steps == 1], 1.0)
    assert np.allclose(lengths[steps == 2], np.sqrt(2.0))
    assert np.allclose(lengths[steps == 3], np.sqrt(3.0))


def test_step_lengths_follow_a_stretched_cell():
    """Doubling one axis doubles the step along it, not the others."""

    lengths = bd.step_lengths((10, 10, 10), np.diag([10.0, 10.0, 20.0]))

    along_z = np.abs(bd.OFFSETS) @ np.array([0, 0, 1]) == 1
    only_z = along_z & (np.abs(bd.OFFSETS).sum(axis=1) == 1)

    assert np.allclose(lengths[only_z], 2.0)


def test_the_gradient_matrix_is_one_over_spacing_squared():
    """
    For an orthogonal cell the general form has to collapse to Tang's.

    Tang 2009 equations 4 to 6 divide each gradient component by its own
    grid spacing squared; the metric-tensor form used here has to agree
    with that wherever the cell is orthogonal, or it is wrong everywhere.
    """

    shape = (8, 10, 20)
    lattice = np.diag([4.0, 5.0, 30.0])

    spacing = np.array([4.0 / 8, 5.0 / 10, 30.0 / 20])

    matrix = bd.gradient_matrix(shape, lattice)

    assert np.allclose(matrix, np.diag(1.0 / spacing ** 2))


def test_the_gradient_matrix_is_symmetric_for_a_sheared_cell():
    """It is built from a metric tensor, so it cannot be anything else."""

    lattice = np.array([[4.0, 0.0, 0.0], [2.0, 3.5, 0.0], [0.0, 0.0, 5.0]])

    matrix = bd.gradient_matrix((12, 12, 12), lattice)

    assert np.allclose(matrix, matrix.T)


# ------------------------------------------------------------
# Ascent
# ------------------------------------------------------------


def test_a_single_peak_swallows_the_cell():
    """One maximum means one basin, and it holds every point."""

    shape = (16, 16, 16)
    lattice = np.eye(3) * 6.0

    rho = gaussians(shape, lattice, [[0.5, 0.5, 0.5]], widths=[0.8])

    roots = bd.resolve_maxima(bd.ascent_parents(rho, lattice))

    assert len(np.unique(roots)) == 1

    peak = np.unravel_index(int(roots[0]), shape)

    assert peak == (8, 8, 8)


def test_two_identical_peaks_split_the_charge_exactly():
    """
    Symmetry is the one case where the right answer is known.

    Two identical peaks related by a translation must end up with the
    same charge to the last decimal, because every step of the
    partitioner is built from array shifts and so commutes with that
    translation. Anything else is a real asymmetry in the algorithm.
    """

    shape = (24, 24, 24)
    lattice = np.eye(3) * 6.0

    structure = two_atom_cell(6.0, ("Na", "Na"))

    rho = symmetric_pair(shape, lattice)

    assert np.array_equal(rho, np.roll(rho, (12, 12, 12), axis=(0, 1, 2)))

    labels, _, _, _ = bd.partition(rho, structure, method="neargrid")

    charge, volume = bd.integrate(labels, rho, structure)

    assert charge[0] == pytest.approx(charge[1], rel=1e-12)
    assert volume[0] == pytest.approx(volume[1], rel=1e-12)


def test_nothing_is_lost_in_the_partition():
    """Every grid point belongs to exactly one atom."""

    shape = (20, 20, 20)
    lattice = np.eye(3) * 6.0

    structure = two_atom_cell()

    rho = gaussians(
        shape, lattice, [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        heights=[1.0, 2.0],
    )

    labels, _, _, _ = bd.partition(rho, structure)

    charge, volume = bd.integrate(labels, rho, structure)

    assert charge.sum() == pytest.approx(rho.sum() / rho.size, rel=1e-12)
    assert volume.sum() == pytest.approx(structure.volume, rel=1e-12)


def test_a_bigger_peak_takes_the_bigger_share():
    """The deeper well pulls the boundary towards its neighbour."""

    shape = (24, 24, 24)
    lattice = np.eye(3) * 6.0

    structure = two_atom_cell()

    rho = gaussians(
        shape, lattice, [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        widths=[1.0, 0.5],
    )

    labels, _, _, _ = bd.partition(rho, structure)

    charge, volume = bd.integrate(labels, rho, structure)

    assert charge[0] > charge[1]
    assert volume[0] > volume[1]


def test_the_boundary_between_equal_peaks_is_the_plane_between_them():
    """
    Two Gaussians on the z axis divide at the planes halfway between.

    This is the assertion that says the surface is in the *right* place
    rather than merely in a consistent one.
    """

    shape = (16, 16, 32)
    lattice = np.eye(3) * 8.0

    structure = Structure(
        Lattice.cubic(8.0), ["Na", "Na"], [[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]]
    )

    rho = gaussians(
        shape, lattice, [[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]], widths=0.9
    )

    labels, _, _, _ = bd.partition(rho, structure)

    # Down the line through both peaks, the first half belongs to the
    # first atom and the second half to the second.
    line = labels[8, 8, :]

    assert set(np.unique(line)) == {0, 1}
    assert (line[2:14] == line[8]).all()
    assert (line[18:30] == line[24]).all()
    assert line[8] != line[24]


def test_the_labels_name_atoms_not_maxima():
    """
    A grid finds more maxima than there are atoms, and each is given to
    the nearest nucleus, so the label grid only ever holds atom numbers.
    """

    shape = (20, 20, 20)
    lattice = np.eye(3) * 6.0

    structure = two_atom_cell()

    rho = gaussians(
        shape, lattice, [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
    )

    labels, maxima, merged, _ = bd.partition(rho, structure)

    assert set(np.unique(labels)) <= {0, 1}
    assert merged.sum() == maxima
    assert len(merged) == len(structure)


def test_ongrid_and_neargrid_agree_on_a_symmetric_cell():
    """
    Where the surfaces run along the grid there is no bias to remove.

    This is the control for
    :func:`test_neargrid_beats_ongrid_on_a_tilted_surface`: if the two
    methods differed here as well, the difference there would say
    nothing about lattice bias.
    """

    shape = (24, 24, 24)
    lattice = np.eye(3) * 6.0

    structure = two_atom_cell(6.0, ("Na", "Na"))

    rho = symmetric_pair(shape, lattice)

    near, _, _, _ = bd.partition(rho, structure, method="neargrid")
    on, _, _, _ = bd.partition(rho, structure, method="ongrid")

    near_charge, _ = bd.integrate(near, rho, structure)
    on_charge, _ = bd.integrate(on, rho, structure)

    assert near_charge == pytest.approx(on_charge, rel=1e-12)


def nearest_centre(shape, lattice, centres):
    """
    Which centre each grid point is closest to, through the boundary.

    For a sum of identical, well separated isotropic Gaussians this is
    the exact Bader partition: the density is symmetric under the
    reflection that swaps any two of them, so the zero-flux surface has
    to be the plane halfway between -- which is the Voronoi boundary.
    That makes it a ground truth to measure a partitioner against rather
    than another partitioner.
    """

    axes = [np.arange(n) / float(n) for n in shape]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    distances = []

    for centre in np.atleast_2d(np.asarray(centres, dtype=float)):

        offset = grid - centre
        offset -= np.round(offset)

        distances.append(
            np.linalg.norm(offset @ np.asarray(lattice, float), axis=-1)
        )

    return np.argmin(np.stack(distances, axis=-1), axis=-1)


def test_neargrid_beats_ongrid_on_a_tilted_surface():
    """
    The case Tang 2009 was written for.

    Two identical Gaussians are placed so the plane between them runs at
    an angle to the grid. The on-grid method cannot follow it -- its
    paths are stuck to the 26 lattice directions, so the surface comes
    out faceted along the grid, and refining the grid does not help.
    The near-grid method is the fix, and it has to put measurably more
    grid points on the correct side.
    """

    shape = (32, 32, 32)
    lattice = np.eye(3) * 8.0

    first = [0.27, 0.21, 0.16]
    second = [0.73, 0.79, 0.84]

    structure = Structure(Lattice.cubic(8.0), ["Na", "Na"], [first, second])

    rho = gaussians(shape, lattice, [first, second], widths=0.6)

    truth = nearest_centre(shape, lattice, [first, second])

    near, _, _, _ = bd.partition(rho, structure, method="neargrid")
    on, _, _, _ = bd.partition(rho, structure, method="ongrid")

    near_wrong = int((near != truth).sum())
    on_wrong = int((on != truth).sum())

    assert near_wrong < on_wrong
    assert near_wrong / truth.size < 0.01


def test_an_unknown_method_is_refused():

    structure = two_atom_cell()

    rho = gaussians((8, 8, 8), np.eye(3) * 6.0, [[0.5, 0.5, 0.5]])

    with pytest.raises(bd.BaderError, match="method"):
        bd.partition(rho, structure, method="wishful")


def test_partition_reports_progress():
    """The dialog runs this on a thread and drives a bar from it."""

    structure = two_atom_cell()

    rho = gaussians((16, 16, 16), np.eye(3) * 6.0,
                    [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])

    seen: list[tuple[str, int]] = []

    bd.partition(rho, structure, progress=lambda m, p: seen.append((m, p)))

    assert seen
    assert all(0 <= percent <= 100 for _, percent in seen)
    assert any("uphill" in message for message, _ in seen)


# ------------------------------------------------------------
# Measuring
# ------------------------------------------------------------


def test_the_surface_distance_is_a_radius():
    """
    Halfway between two atoms 3 A apart, measured from either one.

    The atoms sit at z = 1/4 and z = 3/4 of an 8 A cell, so the surface
    is 2 A from each -- give or take the half grid spacing the surface
    is resolved to.
    """

    shape = (16, 16, 32)
    lattice = np.eye(3) * 8.0

    structure = Structure(
        Lattice.cubic(8.0), ["Na", "Na"], [[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]]
    )

    rho = gaussians(
        shape, lattice, [[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]], widths=0.9
    )

    labels, _, _, _ = bd.partition(rho, structure)

    radii = bd.surface_distances(labels, structure, 2)

    assert radii[0] == pytest.approx(2.0, abs=0.3)
    assert radii[1] == pytest.approx(2.0, abs=0.3)


def test_an_atom_with_no_volume_has_no_radius():
    """NaN rather than zero, which would read as a radius of nothing."""

    labels = np.zeros((6, 6, 6), dtype=np.int32)

    structure = two_atom_cell()

    radii = bd.surface_distances(labels, structure, 2)

    assert np.isnan(radii[1])


def test_the_reference_quality_check_passes_a_good_density():
    """A density that integrates to the atomic numbers raises nothing."""

    structure = Structure(Lattice.cubic(4.0), ["H", "H"],
                          [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    rho = np.full((8, 8, 8), 2.0)

    integrated, expected, warning = bd.reference_quality(rho, structure)

    assert integrated == pytest.approx(2.0)
    assert expected == 2.0
    assert warning is None


def test_a_ringing_reference_is_reported():
    """
    The failure mode that matters: an FFT grid too coarse for the core.

    The all-electron density then goes negative between the nuclei and
    integrates short, and the numbers that come out of it look perfectly
    plausible, so this has to be said out loud.
    """

    structure = Structure(Lattice.cubic(4.0), ["Co", "Co"],
                          [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    rho = np.full((8, 8, 8), 20.0)
    rho[::2, ::2, ::2] = -5.0

    _, expected, warning = bd.reference_quality(rho, structure)

    assert expected == 54.0
    assert warning is not None
    assert "NGXF" in warning
    assert "negative" in warning


# ------------------------------------------------------------
# Selections
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1", [1]),
        ("1-4", [1, 2, 3, 4]),
        ("1..4", [1, 2, 3, 4]),
        ("4-1", [1, 2, 3, 4]),
        ("1,3,5", [1, 3, 5]),
        ("1-3 7", [1, 2, 3, 7]),
        ("2,2,2", [2]),
        ("  ", []),
    ],
)
def test_selections_are_read_as_written(text, expected):

    assert bd.parse_selection(text, 10) == expected


@pytest.mark.parametrize("text", ["0", "11", "-3", "x", "1-", "1-99"])
def test_a_bad_selection_says_so(text):

    with pytest.raises(bd.BaderError):
        bd.parse_selection(text, 10)


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------


def sample_result():
    """A result with the numbers filled in by hand."""

    atoms = [
        bd.AtomCharge(1, "Li", np.zeros(3), 0.2, 1.0, 8.0, 0.5, 1),
        bd.AtomCharge(2, "Li", np.zeros(3), 0.3, 1.0, 9.0, 0.5, 1),
        bd.AtomCharge(3, "O", np.zeros(3), 7.5, 6.0, 12.0, 0.8, 1),
    ]

    return bd.BaderResult(
        atoms=atoms,
        structure=two_atom_cell(),
        labels=np.zeros((2, 2, 2), dtype=np.int32),
        reference="all-electron",
        method="neargrid",
        grid=(2, 2, 2),
        basins=3,
    )


def test_a_net_charge_is_zval_minus_the_population():
    """The sign that catches people out: charge counts electrons."""

    atom = bd.AtomCharge(1, "Li", np.zeros(3), 0.2, 1.0, 8.0, 0.5, 1)

    assert atom.charge == 0.2
    assert atom.net == pytest.approx(0.8)


def test_elements_are_summed_and_averaged():

    rows = bd.by_element(sample_result())

    assert [row.label for row in rows] == ["Li", "O"]

    lithium, oxygen = rows

    assert lithium.count == 2
    assert lithium.charge == pytest.approx(0.5)
    assert lithium.zval == pytest.approx(2.0)
    assert lithium.net == pytest.approx(1.5)
    assert lithium.net_per_atom == pytest.approx(0.75)
    assert lithium.volume_per_atom == pytest.approx(8.5)

    assert oxygen.net == pytest.approx(-1.5)


def test_the_element_rows_add_back_up():
    """Grouping must not create or destroy electrons."""

    result = sample_result()

    rows = bd.by_element(result)

    assert sum(row.charge for row in rows) == pytest.approx(result.electrons)
    assert sum(row.net for row in rows) == pytest.approx(0.0, abs=1e-12)


def test_a_fragment_is_the_charge_that_moved_into_it():

    rows = bd.by_fragment(
        sample_result(), {"cation": [1, 2], "anion": [3]}
    )

    cation, anion = rows

    assert cation.members == (1, 2)
    assert cation.net == pytest.approx(1.5)
    assert anion.net == pytest.approx(-1.5)


def test_fragments_may_overlap_or_leave_atoms_out():
    """
    They are questions, not a partition.

    An adsorbate and the surface layer it sits on genuinely do share
    atoms in some analyses, and refusing that would be wrong.
    """

    rows = bd.by_fragment(sample_result(), {"a": [1, 2], "b": [2, 3]})

    assert rows[0].count == 2
    assert rows[1].count == 2
    assert rows[1].charge == pytest.approx(7.8)


def test_a_fragment_naming_a_missing_atom_says_so():

    with pytest.raises(bd.BaderError, match="42"):
        bd.by_fragment(sample_result(), {"nope": [42]})


def stacked_result():
    """
    Six atoms at known heights, for cutting the cell up.

    They are deliberately not in height order: an axis cut has to look
    at the coordinates rather than at the atom numbering, and a sorted
    structure would hide the difference.
    """

    heights = [8.0, 1.0, 9.0, 2.0, 3.0, 7.0]

    atoms = [
        bd.AtomCharge(
            number,
            "Li",
            np.array([0.0, 0.0, z]),
            1.0,
            1.0,
            8.0,
            0.5,
            1,
        )
        for number, z in enumerate(heights, start=1)
    ]

    return bd.BaderResult(
        atoms=atoms,
        structure=two_atom_cell(),
        labels=np.zeros((2, 2, 2), dtype=np.int32),
        reference="all-electron",
        method="neargrid",
        grid=(2, 2, 2),
        basins=6,
    )


def test_an_axis_span_is_the_reach_of_the_atoms():

    assert bd.axis_span(stacked_result(), 2) == pytest.approx((1.0, 9.0))
    assert bd.axis_span(stacked_result(), 0) == pytest.approx((0.0, 0.0))


def test_a_cut_sorts_atoms_by_where_they_are_not_by_their_number():

    below, above = bd.split_on_axis(stacked_result(), 2, 5.0)

    assert below == [2, 4, 5]
    assert above == [1, 3, 6]


def test_a_cut_always_partitions_the_structure():
    """
    Every atom lands on exactly one side, wherever the plane goes.

    Fragments in general may overlap, but the two halves of a cut are
    the one case where that would be a bug: the charge either side would
    stop adding up to the cell's.
    """

    result = stacked_result()

    for cut in (-5.0, 1.0, 5.0, 9.0, 50.0):

        below, above = bd.split_on_axis(result, 2, cut)

        assert sorted(below + above) == [1, 2, 3, 4, 5, 6]
        assert set(below).isdisjoint(above)


def test_an_atom_on_the_plane_goes_above_it():
    """Which way it falls is arbitrary, but it must be decided."""

    below, above = bd.split_on_axis(stacked_result(), 2, 3.0)

    assert 5 in above
    assert 5 not in below


def test_a_cut_outside_the_cell_leaves_one_side_empty():
    """The caller is expected to notice and say so."""

    below, above = bd.split_on_axis(stacked_result(), 2, -1.0)

    assert below == []
    assert len(above) == 6


def test_a_fourth_axis_is_refused():

    with pytest.raises(bd.BaderError):
        bd.split_on_axis(stacked_result(), 3, 0.0)

    with pytest.raises(bd.BaderError):
        bd.axis_span(stacked_result(), -1)


def test_the_rest_is_what_no_fragment_claimed():

    rest = bd.remaining(stacked_result(), {"top": [1, 3], "odd": [5]})

    assert rest == [2, 4, 6]


def test_the_rest_counts_an_atom_once_however_often_it_was_claimed():
    """Overlapping fragments must not leave an atom out twice over."""

    rest = bd.remaining(
        stacked_result(), {"a": [1, 2], "b": [2, 3]}
    )

    assert rest == [4, 5, 6]


def test_there_is_no_rest_once_everything_is_spoken_for():

    result = stacked_result()

    below, above = bd.split_on_axis(result, 2, 5.0)

    assert bd.remaining(result, {"below": below, "above": above}) == []


def test_the_rest_of_nothing_is_everything():

    assert bd.remaining(stacked_result(), {}) == [1, 2, 3, 4, 5, 6]


def test_a_cut_and_its_rest_describe_the_same_two_groups():
    """
    The two ways into a fragment agree.

    Cutting gives both halves at once; naming one and taking the rest
    is the other route to the same partition, and a user who mixes them
    should not get different atoms.
    """

    result = stacked_result()

    below, above = bd.split_on_axis(result, 2, 5.0)

    assert bd.remaining(result, {"below": below}) == above


def test_the_elements_keep_the_order_they_appear_in():

    assert sample_result().elements() == ["Li", "O"]


# ------------------------------------------------------------
# The field the viewer draws
# ------------------------------------------------------------


def test_the_basin_field_marks_only_the_chosen_atoms():
    """One inside, zero outside -- contoured at a half by the viewer."""

    result = sample_result()
    result.labels = np.array([0, 0, 1, 2, 2, 2, -1, -1], dtype=np.int32).reshape(
        (2, 2, 2)
    )

    field = bd.basin_surface(result, [1])

    assert field.shape == (2, 2, 2)
    assert field.ravel().tolist() == [1, 1, 0, 0, 0, 0, 0, 0]


def test_the_basin_field_takes_several_atoms():

    result = sample_result()
    result.labels = np.array([0, 0, 1, 2, 2, 2, -1, -1], dtype=np.int32).reshape(
        (2, 2, 2)
    )

    field = bd.basin_surface(result, [1, 3])

    assert field.ravel().tolist() == [1, 1, 0, 1, 1, 1, 0, 0]


def test_vacuum_is_never_inside_a_basin():
    """-1 means nobody's, and it has to stay that way."""

    result = sample_result()
    result.labels = np.full((2, 2, 2), -1, dtype=np.int32)

    assert bd.basin_surface(result, [1, 2, 3]).sum() == 0.0


def test_the_basin_field_refuses_an_atom_that_is_not_there():

    with pytest.raises(bd.BaderError):
        bd.basin_surface(sample_result(), [9])


# ------------------------------------------------------------
# Reading a run
# ------------------------------------------------------------


def write_run(folder, structure, shape=(20, 20, 20), zval=None):
    """
    A run with the three densities a Bader analysis reads.

    The reference gets a sharp, tall core peak on each nucleus and the
    valence density a broad low one, which is the relationship the real
    files have and the reason the reference is the one that draws the
    boundaries.

    ``zval`` is one value per *species*, as POTCAR and OUTCAR carry it --
    not one per atom.
    """

    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    centres = np.asarray(structure.frac_coords, dtype=float)

    core = gaussians(shape, lattice, centres, widths=0.45, heights=400.0)
    smooth = gaussians(shape, lattice, centres, widths=1.0)
    valence = gaussians(shape, lattice, centres, widths=1.1, heights=10.0)

    poscar = Poscar(structure)

    Chgcar(poscar, {"total": core}).write_file(str(folder / "AECCAR0"))
    Chgcar(poscar, {"total": smooth}).write_file(str(folder / "AECCAR2"))
    Chgcar(poscar, {"total": valence}).write_file(str(folder / "CHGCAR"))

    if zval is not None:
        (folder / "OUTCAR").write_text(
            "\n".join(
                f"   POMASS =   22.990; ZVAL   =  {value:.3f}    "
                "mass and valenz"
                for value in zval
            )
            + "\n Iteration    1\n",
            encoding="utf-8",
        )


def test_only_the_references_the_folder_can_support_are_offered(tmp_path):

    structure = two_atom_cell()

    write_run(tmp_path, structure, shape=(12, 12, 12))

    assert bd.available_references(tmp_path) == ["all-electron", "valence"]

    (tmp_path / "AECCAR0").unlink()

    assert bd.available_references(tmp_path) == ["valence"]

    (tmp_path / "CHGCAR").unlink()

    assert bd.available_references(tmp_path) == []


def test_a_missing_density_names_the_file(tmp_path):

    with pytest.raises(bd.BaderError, match="CHGCAR"):
        bd.load_densities(tmp_path)


def test_an_unknown_reference_is_refused(tmp_path):

    with pytest.raises(bd.BaderError, match="reference"):
        bd.load_densities(tmp_path, reference="guesswork")


def test_zval_is_read_from_the_outcar(tmp_path):
    """The fallback for a folder the POTCAR was not copied into."""

    structure = two_atom_cell()

    write_run(tmp_path, structure, shape=(12, 12, 12), zval=[9.0, 7.0])

    assert bd.load_zval(tmp_path, structure) == {"Na": 9.0, "Cl": 7.0}


def test_without_zval_the_failure_explains_itself(tmp_path):

    structure = two_atom_cell()

    write_run(tmp_path, structure, shape=(12, 12, 12))

    with pytest.raises(bd.BaderError, match="POTCAR"):
        bd.load_zval(tmp_path, structure)


def test_densities_on_different_grids_are_refused(tmp_path):

    structure = two_atom_cell()

    write_run(tmp_path, structure, shape=(12, 12, 12), zval=[9.0, 7.0])

    # Rewrite the valence density on a coarser grid than the reference.
    coarse = gaussians(
        (8, 8, 8),
        np.asarray(structure.lattice.matrix, float),
        np.asarray(structure.frac_coords, float),
    )
    Chgcar(Poscar(structure), {"total": coarse}).write_file(
        str(tmp_path / "CHGCAR")
    )

    with pytest.raises(bd.BaderError, match="different grids"):
        bd.load_densities(tmp_path)


# ------------------------------------------------------------
# End to end
# ------------------------------------------------------------


def test_a_whole_run_comes_out_conserved(tmp_path):
    """
    The one invariant the whole tool rests on.

    Whatever the partition does, the electrons in it have to add up to
    the electrons that went in, and the volumes to the cell.
    """

    structure = two_atom_cell()

    write_run(tmp_path, structure, shape=(24, 24, 24), zval=[9.0, 7.0])

    result = bd.analyse(tmp_path)

    valence, _, _ = bd.load_densities(tmp_path, "valence")

    assert result.electrons == pytest.approx(
        valence.sum() / valence.size, rel=1e-10
    )
    assert result.volume == pytest.approx(structure.volume, rel=1e-10)

    assert len(result.atoms) == 2
    assert result.grid == (24, 24, 24)
    assert [atom.symbol for atom in result.atoms] == ["Na", "Cl"]
    assert [atom.zval for atom in result.atoms] == [9.0, 7.0]


def test_equivalent_atoms_come_out_equal_end_to_end(tmp_path):
    """Two of the same element, placed symmetrically, in real files."""

    structure = two_atom_cell(6.0, ("Na", "Na"))

    # One species, so one ZVAL, however many atoms there are.
    write_run(tmp_path, structure, shape=(24, 24, 24), zval=[9.0])

    result = bd.analyse(tmp_path)

    first, second = result.atoms

    assert first.charge == pytest.approx(second.charge, abs=1e-9)
    assert first.volume == pytest.approx(second.volume, abs=1e-9)
    assert first.min_dist == pytest.approx(second.min_dist, abs=1e-9)


def test_the_valence_reference_still_runs(tmp_path):
    """A run made without LAECHG is worse, not unusable."""

    structure = two_atom_cell()

    write_run(tmp_path, structure, shape=(20, 20, 20), zval=[9.0, 7.0])

    (tmp_path / "AECCAR0").unlink()

    result = bd.analyse(tmp_path, reference="valence")

    assert result.reference == "valence"
    assert len(result.atoms) == 2

    valence, _, _ = bd.load_densities(tmp_path, "valence")

    assert result.electrons == pytest.approx(
        valence.sum() / valence.size, rel=1e-10
    )


def test_setting_the_vacuum_aside_moves_charge_out_of_the_atoms(tmp_path):
    """
    A molecule in a box: most of the cell is empty and should say so.

    The vacuum charge and volume have to come out of the atoms, not be
    added on top, or the totals stop meaning anything.
    """

    structure = Structure(
        Lattice.cubic(10.0), ["Na", "Cl"],
        [[0.45, 0.5, 0.5], [0.55, 0.5, 0.5]],
    )

    write_run(tmp_path, structure, shape=(24, 24, 24), zval=[9.0, 7.0])

    plain = bd.analyse(tmp_path)
    empty = bd.analyse(tmp_path, vacuum=True)

    assert empty.vacuum_volume > 0.0
    assert empty.volume == pytest.approx(plain.volume, rel=1e-10)
    assert empty.electrons == pytest.approx(plain.electrons, rel=1e-10)

    assert sum(a.volume for a in empty.atoms) < sum(
        a.volume for a in plain.atoms
    )

    assert any("vacuum" in note for note in empty.notes)


def test_the_run_reports_what_it_did(tmp_path):
    """The notes are shown to the user and saved into the CSV."""

    structure = two_atom_cell()

    write_run(tmp_path, structure, shape=(20, 20, 20), zval=[9.0, 7.0])

    result = bd.analyse(tmp_path)

    assert result.notes
    assert any("electrons were accounted for" in note for note in result.notes)


def test_every_definition_has_wording(tmp_path):
    """
    The dialog shows these above the table; an empty one would be a hole
    in the explanation rather than a cosmetic problem.
    """

    assert len(bd.DEFINITIONS) >= 8

    for term, text in bd.DEFINITIONS:
        assert term.strip()
        assert len(text.strip()) > 40

    for title, url in bd.SOURCES:
        assert title.strip()
        assert url == "" or url.startswith("https://")


def test_the_choices_are_all_described():
    """Every entry the combo boxes offer has a line saying what it is."""

    assert set(bd.REFERENCES) == {"all-electron", "valence"}
    assert set(bd.METHODS) == {"neargrid", "ongrid"}

    for text in list(bd.REFERENCES.values()) + list(bd.METHODS.values()):
        assert len(text.strip()) > 40

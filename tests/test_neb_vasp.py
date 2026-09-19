"""
Tests for writing and reading a VASP nudged elastic band.

Three things here are wrong-but-plausible if unguarded:

* **The periodic boundary.** A migrating ion usually crosses it. Without
  the minimum-image convention it is interpolated the long way round,
  straight through the rest of the cell. Every image is still a valid
  structure, the path still plots as a smooth curve, and the barrier is
  meaningless.

* **``IMAGES``.** VASP counts intermediate images, not folders. Off by
  one and VASP either refuses to start or treats an endpoint as movable,
  which quietly relaxes away the thing whose energy the barrier is
  measured from.

* **The barrier when an image is missing.** An image folder with no
  energy makes the maximum a lower bound rather than the barrier, and a
  profile drawn through the gap looks entirely normal.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services import neb_vasp as nv


# ------------------------------------------------------------
# Structures to move between
# ------------------------------------------------------------


def cell(positions, species=None, length: float = 8.0):
    """A cubic cell with the given fractional coordinates."""

    from pymatgen.core import Lattice, Structure

    if species is None:
        species = ["Li"] + ["O"] * (len(positions) - 1)

    return Structure(Lattice.cubic(length), species, positions)


def hop_across_the_boundary():
    """A Li at x = 0.05 moving to x = 0.95: 0.8 A the short way."""

    fixed = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]]

    start = cell([[0.05, 0.5, 0.5], *fixed])
    end = cell([[0.95, 0.5, 0.5], *fixed])

    return start, end


# ------------------------------------------------------------
# The periodic boundary
# ------------------------------------------------------------


def test_minimum_image_takes_the_shorter_displacement() -> None:

    wrapped = nv.minimum_image(np.array([0.9, -0.9, 0.4, -0.4, 1.3]))

    assert np.allclose(wrapped, [-0.1, 0.1, 0.4, -0.4, 0.3])


def test_a_half_cell_displacement_is_a_tie() -> None:
    """
    Exactly 0.5 is the same distance either way round.

    Whichever sign comes back is correct, so the guarantee is on the
    magnitude, not the direction. Asserting a sign here would be
    pinning ``np.rint``'s round-half-to-even rather than the physics.
    """

    wrapped = nv.minimum_image(np.array([0.5, -0.5, 1.5]))

    assert np.all(np.abs(wrapped) <= 0.5 + 1e-12)


def test_an_ion_crossing_the_boundary_takes_the_short_way() -> None:
    """
    The test this module exists for.

    0.05 to 0.95 is 0.1 of a cell the short way and 0.9 the long way.
    An 8 A cell makes that 0.8 A against 7.2 A -- an order of magnitude,
    and entirely invisible in the shape of the resulting profile.
    """

    start, end = hop_across_the_boundary()

    chain = nv.interpolate(start, end, images=3)

    length = nv.path_length(chain)[-1]

    assert length == pytest.approx(0.8, abs=1e-6)

    assert length < 1.0, "the ion went the long way round"


def test_the_intermediate_images_stay_near_the_boundary() -> None:
    """Not strung out across the middle of the cell."""

    start, end = hop_across_the_boundary()

    chain = nv.interpolate(start, end, images=3)

    # Distance from x = 0, wrapped -- every image should be within
    # 0.05 of the boundary it is hopping across.
    for image in chain:

        x = float(image.frac_coords[0][0]) % 1.0

        assert min(x, 1.0 - x) <= 0.0501


def test_path_length_measures_the_whole_configuration_step() -> None:
    """
    Not one atom's displacement.

    Two atoms each moving 0.3 A is a step of sqrt(2) * 0.3, which is
    the NEB reaction coordinate; taking one of them would understate it.
    """

    start = cell([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], species=["Li", "Li"])
    end = cell([[0.05, 0.0, 0.0], [0.55, 0.5, 0.5]], species=["Li", "Li"])

    chain = [start, end]

    each = 0.05 * 8.0

    assert nv.path_length(chain)[-1] == pytest.approx(
        np.sqrt(2.0) * each
    )


# ------------------------------------------------------------
# Interpolation
# ------------------------------------------------------------


def test_the_chain_has_the_endpoints_on_it() -> None:

    start, end = hop_across_the_boundary()

    chain = nv.interpolate(start, end, images=5)

    assert len(chain) == 7

    assert np.allclose(chain[0].frac_coords, start.frac_coords)
    assert np.allclose(chain[-1].frac_coords, end.frac_coords)


def test_the_endpoints_are_handed_back_untouched() -> None:
    """
    They were relaxed; nothing here may perturb them.

    Re-deriving them from the interpolation would move them by rounding,
    and NEB holds them fixed, so the error would sit in the reference
    energy the whole barrier is measured against.
    """

    start, end = hop_across_the_boundary()

    chain = nv.interpolate(start, end, images=3)

    assert chain[0] is start
    assert chain[-1] is end


def test_the_images_are_evenly_spaced() -> None:

    start = cell([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    end = cell([[0.4, 0.0, 0.0], [0.5, 0.5, 0.5]])

    chain = nv.interpolate(start, end, images=3)

    steps = np.diff(nv.path_length(chain))

    assert np.allclose(steps, steps[0])


def test_zero_images_is_refused() -> None:

    start, end = hop_across_the_boundary()

    with pytest.raises(nv.NebVaspError, match="at least one"):
        nv.interpolate(start, end, images=0)


def test_endpoints_with_different_atom_counts_are_refused() -> None:

    start = cell([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    end = cell([[0.1, 0.0, 0.0]])

    with pytest.raises(nv.NebVaspError, match="different numbers"):
        nv.interpolate(start, end)


def test_endpoints_with_a_different_atom_order_are_refused() -> None:
    """
    VASP matches images atom by atom.

    Two POSCARs with the same composition in a different order give a
    path in which every atom transmutes into its neighbour, which is
    both physically absurd and perfectly runnable.
    """

    start = cell([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], species=["Li", "O"])
    end = cell([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], species=["O", "Li"])

    with pytest.raises(nv.NebVaspError, match="different order"):
        nv.interpolate(start, end)


def test_only_the_moving_atom_is_reported() -> None:

    start, end = hop_across_the_boundary()

    assert nv.moving_atoms(start, end) == [0]


def test_an_atom_that_barely_moves_is_not_counted() -> None:

    fixed = [[0, 0, 0], [0.5, 0, 0]]

    start = cell([[0.5, 0.5, 0.5], *fixed])
    end = cell([[0.5005, 0.5, 0.5], *fixed])

    # 0.0005 of 8 A is 0.004 A, well under the threshold.
    assert nv.moving_atoms(start, end) == []


# ------------------------------------------------------------
# Writing
# ------------------------------------------------------------


def test_the_folders_are_two_digit_and_start_at_zero() -> None:

    assert nv.folder_name(0) == "00"
    assert nv.folder_name(7) == "07"
    assert nv.folder_name(12) == "12"


def test_a_written_path_has_images_plus_two_folders(tmp_path) -> None:
    """Endpoints included, which is what confuses people about IMAGES."""

    start, end = hop_across_the_boundary()

    result = nv.write_path(tmp_path / "nebexp", start, end, images=5)

    assert result.total_folders == 7

    assert [folder.name for folder in result.folders] == [
        "00", "01", "02", "03", "04", "05", "06",
    ]

    for folder in result.folders:
        assert (folder / "POSCAR").is_file()


def test_incar_images_counts_the_intermediates_only(tmp_path) -> None:
    """
    Seven folders, ``IMAGES = 5``.

    The off-by-one that stops a NEB dead, or worse, lets VASP relax an
    endpoint.
    """

    start, end = hop_across_the_boundary()

    nv.write_path(
        tmp_path / "nebexp",
        start,
        end,
        images=5,
        incar=nv.incar_text(images=5),
    )

    text = (tmp_path / "nebexp" / "INCAR").read_text()

    assert "IMAGES = 5" in text

    assert len(nv.image_folders(tmp_path / "nebexp")) == 7


def test_the_climbing_image_tag_follows_the_request() -> None:

    assert "LCLIMB = .TRUE." in nv.incar_text(climbing=True)
    assert "LCLIMB = .FALSE." in nv.incar_text(climbing=False)


def test_a_readme_is_left_behind(tmp_path) -> None:

    start, end = hop_across_the_boundary()

    nv.write_path(tmp_path / "nebexp", start, end, images=3)

    assert (tmp_path / "nebexp" / "README.txt").is_file()


def test_identical_endpoints_are_called_out(tmp_path) -> None:

    start, _ = hop_across_the_boundary()

    result = nv.write_path(tmp_path / "nebexp", start, start, images=3)

    assert any("no atom moves" in note for note in result.notes)


def test_a_collective_rearrangement_is_called_out(tmp_path) -> None:
    """NEB assumes one process; forty atoms moving is not one."""

    generator = np.random.default_rng(0)

    positions = generator.random((20, 3))

    start = cell(positions, species=["Li"] * 20)
    end = cell(positions + 0.05, species=["Li"] * 20)

    result = nv.write_path(tmp_path / "nebexp", start, end, images=3)

    assert any("atoms move" in note for note in result.notes)


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


def finished_run(tmp_path, energies, climbing=True, skip=()):
    """Write a NEB and fill in an OUTCAR for each image."""

    start, end = hop_across_the_boundary()

    result = nv.write_path(
        tmp_path / "nebexp",
        start,
        end,
        images=len(energies) - 2,
        incar=nv.incar_text(
            images=len(energies) - 2, climbing=climbing
        ),
    )

    for index, (folder, energy) in enumerate(
        zip(result.folders, energies)
    ):

        if index in skip:
            continue

        (folder / "OUTCAR").write_text(
            f"  free  energy   TOTEN  =      {energy:.8f} eV\n",
            encoding="utf-8",
        )

    return result.root


def test_the_barrier_is_measured_from_the_first_image(tmp_path) -> None:

    root = finished_run(
        tmp_path, [-100.0, -99.85, -99.58, -99.80, -99.98]
    )

    result = nv.read_path(root)

    assert result.barrier_forward == pytest.approx(0.42)
    assert result.reaction_energy == pytest.approx(0.02)

    # From the far side: up from the end to the same saddle.
    assert result.barrier_reverse == pytest.approx(0.40)

    assert result.saddle_index == 2


def test_energies_come_back_relative_to_the_start(tmp_path) -> None:
    """So the profile starts at zero, whatever VASP's absolute scale."""

    root = finished_run(tmp_path, [-500.0, -499.7, -499.9])

    result = nv.read_path(root)

    assert result.energies[0] == pytest.approx(0.0)


def test_oszicar_is_used_when_there_is_no_outcar(tmp_path) -> None:
    """A run killed by a wall clock often has only this."""

    start, end = hop_across_the_boundary()

    written = nv.write_path(tmp_path / "nebexp", start, end, images=1)

    for folder, energy in zip(written.folders, [-100.0, -99.5, -99.9]):
        (folder / "OSZICAR").write_text(
            f"   1 F= {energy:.8E} E0= {energy:.8E}  d E =0\n",
            encoding="utf-8",
        )

    result = nv.read_path(written.root)

    assert result.barrier_forward == pytest.approx(0.5)


def test_a_missing_image_energy_is_flagged(tmp_path) -> None:
    """
    The barrier becomes a lower bound, and the profile still draws.

    Silence here would let somebody quote a barrier that skipped the
    saddle point entirely.
    """

    root = finished_run(
        tmp_path, [-100.0, -99.85, -99.58, -99.80, -99.98], skip=(2,)
    )

    result = nv.read_path(root)

    assert any("no energy found" in note for note in result.notes)

    assert not result.converged

    # The remaining maximum, not the one that was skipped.
    assert result.barrier_forward == pytest.approx(0.20)


def test_a_saddle_at_an_endpoint_is_flagged(tmp_path) -> None:
    """A monotonic path has no barrier on it."""

    root = finished_run(tmp_path, [-100.0, -100.2, -100.5])

    result = nv.read_path(root)

    assert any("no barrier" in note for note in result.notes)


def test_a_run_without_the_climbing_image_says_so(tmp_path) -> None:

    root = finished_run(
        tmp_path, [-100.0, -99.5, -99.9], climbing=False
    )

    result = nv.read_path(root)

    assert not result.climbing

    assert any("lower bound" in note for note in result.notes)


def test_the_climbing_tag_is_read_back(tmp_path) -> None:

    root = finished_run(tmp_path, [-100.0, -99.5, -99.9], climbing=True)

    assert nv.read_path(root).climbing


def test_a_folder_with_too_few_images_is_refused(tmp_path) -> None:

    (tmp_path / "00").mkdir()
    (tmp_path / "01").mkdir()

    with pytest.raises(nv.NebVaspError, match="at least three"):
        nv.read_path(tmp_path)


def test_a_missing_folder_is_refused(tmp_path) -> None:

    with pytest.raises(nv.NebVaspError, match="no such folder"):
        nv.read_path(tmp_path / "nowhere")


def test_a_run_with_no_energies_at_all_is_refused(tmp_path) -> None:

    for name in ("00", "01", "02"):
        (tmp_path / name).mkdir()

    with pytest.raises(nv.NebVaspError, match="has an energy"):
        nv.read_path(tmp_path)


def test_image_folders_sort_numerically(tmp_path) -> None:
    """``10`` after ``09``, which a string sort also gets right, and
    ``100`` after ``99``, which it does not."""

    for index in range(12):
        (tmp_path / nv.folder_name(index)).mkdir()

    names = [folder.name for folder in nv.image_folders(tmp_path)]

    assert names == [nv.folder_name(index) for index in range(12)]


def test_non_numeric_folders_are_ignored(tmp_path) -> None:
    """A ``vasp_logs`` folder beside the images must not become one."""

    for name in ("00", "01", "02", "logs", "backup"):
        (tmp_path / name).mkdir()

    assert len(nv.image_folders(tmp_path)) == 3


# ------------------------------------------------------------
# Into the shared result
# ------------------------------------------------------------


def test_the_result_is_an_neb_result(tmp_path) -> None:
    """So the profile figure and CSV are shared with the CHGNet route."""

    from qvista.analyses.neb import NEBResult

    root = finished_run(tmp_path, [-100.0, -99.5, -99.9])

    result = nv.read_path(root)

    assert isinstance(result, NEBResult)

    assert "VASP" in result.force_field


def test_the_result_exports_csv(tmp_path) -> None:

    root = finished_run(tmp_path, [-100.0, -99.5, -99.9])

    text = nv.read_path(root).to_csv()

    assert "forward barrier" in text
    assert text.count("\n") > 3


# ------------------------------------------------------------
# Endpoints relaxed elsewhere
# ------------------------------------------------------------


def chain_without_endpoint_energies(tmp_path):
    """
    The usual layout: 00 and the last folder hold a POSCAR and nothing
    else, because the endpoints were relaxed in a separate run and their
    OUTCARs were not kept.
    """

    from pymatgen.core import Lattice, Structure

    energies = [None, -302.58, -302.45, -302.30, -302.45, -302.58, None]

    for index, energy in enumerate(energies):

        folder = tmp_path / f"{index:02d}"
        folder.mkdir()

        # Close enough to be neighbours: the point of the fixture is a
        # migrating atom with an environment, and two atoms five
        # Angstrom apart have none.
        structure = Structure(
            Lattice.cubic(4.0),
            ["Li", "O", "O"],
            [
                [0.1 * index, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
        )

        structure.to(filename=str(folder / "POSCAR"), fmt="poscar")

        if energy is None:
            continue

        (folder / "OUTCAR").write_text(
            f"  free  energy   TOTEN  =     {energy:.8f} eV\n"
            f" NEB: projections on to tangent (spring, REAL)"
            f"     0.001000   -0.010000\n",
            encoding="utf-8",
        )

    return tmp_path


def test_a_missing_endpoint_energy_does_not_take_the_read_down(tmp_path):
    """
    Referencing to image 00 made every energy NaN when 00 had no OUTCAR,
    and the read failed with "All-NaN slice encountered" rather than
    saying which folder was short.
    """

    result = nv.read_path(chain_without_endpoint_energies(tmp_path))

    assert result.barrier_forward > 0.0


def test_energies_are_referenced_to_the_first_one_there_is(tmp_path):

    rows = nv.neb_dat(chain_without_endpoint_energies(tmp_path))

    assert rows[0][2] == pytest.approx(0.0)


def test_images_without_an_energy_are_left_out_of_the_table(tmp_path):

    rows = nv.neb_dat(chain_without_endpoint_energies(tmp_path))

    assert [row[0] for row in rows] == [1, 2, 3, 4, 5]


# ------------------------------------------------------------
# neb.dat and the spline
# ------------------------------------------------------------


def test_the_tangent_force_is_read_from_the_outcar(tmp_path):
    """
    The second number VTST prints, which is the true force along the
    path -- the spring force is the first and is not what a spline
    wants.
    """

    folder = tmp_path / "01"
    folder.mkdir()

    (folder / "OUTCAR").write_text(
        " NEB: projections on to tangent (spring, REAL)"
        "     0.005851   -0.001644\n",
        encoding="utf-8",
    )

    assert nv.read_tangent_force(folder) == pytest.approx(-0.001644)


def test_an_endpoint_has_no_tangent_force(tmp_path):
    """VASP never moves it and never reports a projection for it."""

    folder = tmp_path / "00"
    folder.mkdir()

    assert nv.read_tangent_force(folder) is None


def test_the_last_step_is_the_one_that_counts(tmp_path):
    """The geometry the energy belongs to is the one it finished at."""

    folder = tmp_path / "01"
    folder.mkdir()

    (folder / "OUTCAR").write_text(
        " NEB: projections on to tangent (spring, REAL)"
        "     0.1   -0.5\n"
        " NEB: projections on to tangent (spring, REAL)"
        "     0.2   -0.02\n",
        encoding="utf-8",
    )

    assert nv.read_tangent_force(folder) == pytest.approx(-0.02)


def test_the_spline_passes_through_every_image():
    """
    It interpolates the calculated energies, so the barrier read off the
    curve is still the barrier in the table.
    """

    rows = [
        (1, 0.0, 0.0, 0.0),
        (2, 1.0, 0.20, -0.1),
        (3, 2.0, 0.28, 0.0),
        (4, 3.0, 0.20, 0.1),
        (5, 4.0, 0.0, 0.0),
    ]

    grid, curve = nv.spline_profile(rows)

    for _, distance, energy, _ in rows:

        nearest = int(np.argmin(np.abs(grid - distance)))

        assert curve[nearest] == pytest.approx(energy, abs=1e-6)


def test_the_spline_uses_the_gradient_at_each_image():
    """
    Which is the whole point of it. Two paths through the same energies
    with different forces are different curves, and the one with the
    steeper approach has the higher saddle.
    """

    flat = [(1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0)]
    steep = [(1, 0.0, 0.0, -0.5), (2, 1.0, 0.0, 0.5)]

    assert max(nv.spline_profile(steep)[1]) > max(
        nv.spline_profile(flat)[1]
    )


def test_a_single_image_is_not_splined():

    grid, curve = nv.spline_profile([(1, 0.0, 0.0, 0.0)])

    assert len(grid) == 1


# ------------------------------------------------------------
# What the migrating atom is doing
# ------------------------------------------------------------


def test_the_migrating_atom_is_the_one_that_moves_furthest(tmp_path):
    """
    NEB describes one process. The framework relaxes around it, so the
    largest displacement is the migration itself.
    """

    result = nv.read_path(chain_without_endpoint_energies(tmp_path))

    assert result.moving_atom == 0


def test_the_environment_is_reported_at_three_images(tmp_path):
    """
    What the atom sat in, what it had to pass through, and what it
    ended up in -- which is what a barrier is explained by.
    """

    result = nv.read_path(chain_without_endpoint_energies(tmp_path))

    assert [item.label for item in result.environments] == [
        "Initial",
        "Transition state",
        "Final",
    ]


def test_the_transition_state_is_the_highest_image(tmp_path):

    result = nv.read_path(chain_without_endpoint_energies(tmp_path))

    saddle = next(
        item
        for item in result.environments
        if item.label == "Transition state"
    )

    assert saddle.image == result.saddle_index


def test_neighbours_come_back_nearest_first(tmp_path):

    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(8.0),
        ["Li", "O", "O"],
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.35, 0.0, 0.0]],
    )

    contacts = nv.environment_of(structure, 0, cutoff=4.0)

    distances = [contact.distance for contact in contacts]

    assert distances == sorted(distances)


def test_an_empty_environment_reports_no_coordination():

    environment = nv.Environment(label="Initial", image=0, contacts=[])

    assert environment.coordination == 0

    assert environment.shortest != environment.shortest  # NaN


# ------------------------------------------------------------
# What the two tables say
# ------------------------------------------------------------


def read_chain(tmp_path):

    return nv.read_path(chain_without_endpoint_energies(tmp_path))


def test_the_summary_reports_one_barrier(tmp_path):
    """
    Forward, reverse and the spline peak side by side invites picking
    whichever is being argued for. On an asymmetric path the forward one
    is the barrier; the rest belong to the profile.
    """

    labels = [label for label, _ in read_chain(tmp_path).rows]

    assert "Barrier" in labels

    assert not any("Reverse" in label for label in labels)
    assert not any("spline" in label.lower() for label in labels)


def test_the_summary_says_what_was_run(tmp_path):

    rows = dict(read_chain(tmp_path).rows)

    assert set(rows) == {
        "Method",
        "Images",
        "Intermediate images",
        "Climbing image",
        "Barrier",
        "Saddle at image",
    }


def test_the_summary_names_the_saddle_by_its_folder(tmp_path):
    """``03`` is what is on disk; ``3`` is an index into a list."""

    rows = dict(read_chain(tmp_path).rows)

    assert rows["Saddle at image"] == "03"


def test_the_analysis_is_a_table_of_its_own(tmp_path):

    rows = nv.analysis_rows(read_chain(tmp_path))

    assert rows

    assert len(rows[0]) == len(nv.ANALYSIS_COLUMNS)


def test_the_analysis_splits_the_coordination_by_species(tmp_path):
    """
    A coordination number counts different things, and "10" says much
    less than "two oxygens at 1.78 and four cobalts at 2.55".
    """

    rows = nv.analysis_rows(read_chain(tmp_path))

    species = {row[3] for row in rows}

    assert species
    assert "-" not in species


def test_the_analysis_covers_the_three_stages(tmp_path):

    rows = nv.analysis_rows(read_chain(tmp_path))

    stages = [row[0] for row in rows if row[0]]

    assert stages == ["Initial", "Transition state", "Final"]


def test_the_nearest_species_is_listed_first(tmp_path):
    """
    The bond that defines the site is the row a reader lands on.
    """

    result = read_chain(tmp_path)

    rows = nv.analysis_rows(result)

    first = next(row for row in rows if row[0] == "Initial")

    nearest = min(
        contact.distance
        for contact in result.environments[0].contacts
    )

    assert f"{nearest:.3f}" in first[5]


def test_a_path_with_no_migrating_atom_has_no_analysis():

    from qvista.analyses.neb import NEBResult

    assert nv.analysis_rows(NEBResult()) == []


def test_the_analysis_table_stays_hidden_until_there_is_one():

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.neb_vasp_dialog import NebPostprocessDialog

    QApplication.instance() or QApplication([])

    dialog = NebPostprocessDialog()

    assert not dialog.analysis.isVisible()
    assert not dialog.analysis_label.isVisible()


def test_the_preprocess_dialog_still_builds():
    """
    It has a table too, and an edit meant for the other one landed in it
    once already.
    """

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.neb_vasp_dialog import NebPreprocessDialog

    QApplication.instance() or QApplication([])

    dialog = NebPreprocessDialog()

    assert not hasattr(dialog, "analysis")


# ------------------------------------------------------------
# The cutoff is a bond cutoff, not a search radius
# ------------------------------------------------------------


def crowded_chain(tmp_path):
    """
    A migrating atom with a first shell and a second one.

    Two oxygens close enough to be bonds, and two more atoms beyond any
    reasonable bond length -- which is the case the default has to get
    right.
    """

    from pymatgen.core import Lattice, Structure

    for index, energy in enumerate([-1.0, -0.9, -0.8, -0.9, -1.0]):

        folder = tmp_path / f"{index:02d}"
        folder.mkdir()

        structure = Structure(
            Lattice.cubic(10.0),
            ["Li", "O", "O", "Co", "Co"],
            [
                [0.0, 0.0, 0.02 * index],
                [0.20, 0.0, 0.0],
                [0.0, 0.20, 0.0],
                [0.30, 0.0, 0.0],
                [0.0, 0.30, 0.0],
            ],
        )

        structure.to(filename=str(folder / "POSCAR"), fmt="poscar")

        (folder / "OUTCAR").write_text(
            f"  free  energy   TOTEN  =     {energy:.8f} eV\n",
            encoding="utf-8",
        )

    return tmp_path


def test_the_default_cutoff_keeps_the_first_shell_only(tmp_path):
    """
    At 3.2 A a six-coordinate lithium came back sixteen-coordinate: the
    oxygens it is bonded to, plus the metal sublattice behind them.
    """

    result = nv.read_path(crowded_chain(tmp_path))

    initial = result.environments[0]

    assert {contact.symbol for contact in initial.contacts} == {"O"}


def test_the_cutoff_can_be_widened_after_the_read(tmp_path):
    """
    Without touching the disk: the structures came back with the read.
    """

    result = nv.read_path(crowded_chain(tmp_path))

    before = result.environments[0].coordination

    nv.recount(result, 3.2)

    assert result.environments[0].coordination > before


def test_recounting_records_the_cutoff_it_used(tmp_path):
    """
    So the table can say what convention it is reporting, which is the
    whole reason a coordination number means anything.
    """

    result = nv.read_path(crowded_chain(tmp_path))

    nv.recount(result, 3.0)

    assert result.neighbour_cutoff == pytest.approx(3.0)


def test_narrowing_the_cutoff_drops_the_longer_bonds(tmp_path):

    result = nv.read_path(crowded_chain(tmp_path))

    nv.recount(result, 1.5)

    assert result.environments[0].coordination == 0


def test_the_structures_travel_with_the_result(tmp_path):
    """What the viewer draws, and what a recount is done against."""

    result = nv.read_path(crowded_chain(tmp_path))

    assert len(result.structures) == 5

    assert all(item is not None for item in result.structures)


def test_recounting_a_result_with_no_migration_is_harmless():

    from qvista.analyses.neb import NEBResult

    nv.recount(NEBResult(), 2.5)


# ------------------------------------------------------------
# Showing the structure
# ------------------------------------------------------------


def test_the_viewer_can_be_told_which_atom_matters():
    """
    A tool that already knows which atom is interesting should not make
    the user hunt for it.
    """

    pytest.importorskip("PySide6.QtWidgets")

    import inspect

    from qvista.visualization.viewer import CrystalViewer

    assert hasattr(CrystalViewer, "select_index")

    source = inspect.getsource(CrystalViewer.select_index)

    # The home cell's copy, not whichever periodic image comes first.
    assert "is_original_cell" in source


def test_showing_a_stage_loads_it_and_highlights_the_atom(tmp_path):

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication, QWidget

    from qvista.gui.dialogs.neb_vasp_dialog import NebPostprocessDialog

    QApplication.instance() or QApplication([])

    class Viewer:
        def __init__(self):
            self.picked = None

        def select_index(self, index):
            self.picked = index
            return True

    class Workspace:
        def __init__(self):
            self.viewer = Viewer()
            self.loaded = None

        def load_structure(self, structure):
            self.loaded = structure

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.workspace = Workspace()

    window = Window()

    dialog = NebPostprocessDialog(parent=window)
    dialog._loaded(nv.read_path(crowded_chain(tmp_path)))

    dialog._show_structure()

    assert window.workspace.loaded is not None
    assert window.workspace.viewer.picked == dialog.result.moving_atom


def test_every_stage_is_offered(tmp_path):

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.neb_vasp_dialog import NebPostprocessDialog

    QApplication.instance() or QApplication([])

    dialog = NebPostprocessDialog()
    dialog._loaded(nv.read_path(crowded_chain(tmp_path)))

    stages = [
        dialog.stage.itemText(index)
        for index in range(dialog.stage.count())
    ]

    # Three points on the path, and the path itself.
    assert len(stages) == 4

    assert "Transition state" in stages[1]
    assert "MEP" in stages[3]


# ------------------------------------------------------------
# The whole path in one structure
# ------------------------------------------------------------


def test_the_path_carries_one_copy_per_image(tmp_path):
    """
    The trail of copies *is* the minimum energy path. Whether it runs
    straight through a bottleneck or bows around one is the thing a
    barrier number cannot tell you.
    """

    result = nv.read_path(crowded_chain(tmp_path))

    base = result.structures[0]

    path = nv.mep_structure(result)

    assert len(path) == len(base) + len(result.structures) - 1


def test_the_framework_is_the_first_image(tmp_path):
    """
    Not an average of all of them, which belongs to none.
    """

    result = nv.read_path(crowded_chain(tmp_path))

    base = result.structures[0]

    path = nv.mep_structure(result)

    for index in range(len(base)):
        assert np.allclose(path[index].coords, base[index].coords)


def test_every_copy_is_the_migrating_species(tmp_path):

    result = nv.read_path(crowded_chain(tmp_path))

    base = result.structures[0]

    path = nv.mep_structure(result)

    symbol = str(base[result.moving_atom].specie)

    for index in range(len(base), len(path)):
        assert str(path[index].specie) == symbol


def test_the_copies_trace_the_migration(tmp_path):
    """
    They have to move. A path whose copies all land on one another means
    the placement dropped the displacement somewhere.
    """

    result = nv.read_path(crowded_chain(tmp_path))

    base = result.structures[0]

    path = nv.mep_structure(result)

    trail = np.array(
        [path[index].coords for index in range(len(base), len(path))]
    )

    assert np.ptp(trail, axis=0).max() > 0.1


def test_a_path_across_a_cell_face_stays_a_path(tmp_path):
    """
    Placed by minimum image against the starting position, so a
    migration that crosses a boundary does not jump to the far side of
    the cell halfway along.
    """

    from pymatgen.core import Lattice, Structure

    for index in range(3):

        folder = tmp_path / f"{index:02d}"
        folder.mkdir()

        # Walking backwards through zero: 0.02, 0.00, 0.98.
        z = (0.02 - 0.02 * index) % 1.0

        Structure(
            Lattice.cubic(10.0),
            ["Li", "O"],
            [[0.0, 0.0, z], [0.2, 0.0, 0.0]],
        ).to(filename=str(folder / "POSCAR"), fmt="poscar")

        (folder / "OUTCAR").write_text(
            f"  free  energy   TOTEN  =     {-1.0 + 0.1 * index:.8f} eV\n",
            encoding="utf-8",
        )

    result = nv.read_path(tmp_path)

    path = nv.mep_structure(result)

    base = result.structures[0]

    trail = np.array(
        [path[index].frac_coords[2] for index in range(len(base), len(path))]
    )

    # Continuous: no step of half a cell.
    assert np.abs(np.diff(trail)).max() < 0.5


def test_there_is_no_path_without_a_migration():

    from qvista.analyses.neb import NEBResult

    assert nv.mep_structure(NEBResult()) is None


def test_the_dialog_offers_the_path_beside_the_stages(tmp_path):

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.neb_vasp_dialog import (
        MEP_STAGE,
        NebPostprocessDialog,
    )

    QApplication.instance() or QApplication([])

    dialog = NebPostprocessDialog()
    dialog._loaded(nv.read_path(crowded_chain(tmp_path)))

    assert dialog.stage.count() == 4

    assert dialog.stage.itemData(3) == MEP_STAGE

    # Not an image index, and out of range of one.
    assert MEP_STAGE < 0


# ------------------------------------------------------------
# The barrier is in the key, not across the curve
# ------------------------------------------------------------


def profile_figure(tmp_path):

    import matplotlib

    matplotlib.use("Agg")

    from matplotlib.figure import Figure

    from qvista.visualization import dynamics_plots

    figure = Figure()

    dynamics_plots.draw_neb_profile(
        figure, nv.read_path(crowded_chain(tmp_path))
    )

    return figure


def test_nothing_is_written_across_the_path(tmp_path):
    """
    The annotation sat over the very curve it was measuring, and it
    repeated what the title already says.
    """

    assert list(profile_figure(tmp_path).axes[0].texts) == []


def test_the_barrier_is_named_in_the_key(tmp_path):

    legend = profile_figure(tmp_path).axes[0].get_legend()

    assert legend is not None

    entries = [text.get_text() for text in legend.get_texts()]

    assert len(entries) == 1
    assert "barrier" in entries[0]
    assert "eV" in entries[0]


# ------------------------------------------------------------
# MEP is the path, not a point on it
# ------------------------------------------------------------


def test_the_path_view_does_not_highlight_one_copy(tmp_path):
    """
    A selection cage around one copy of the trail points at the least
    interesting thing in the picture.
    """

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication, QWidget

    from qvista.gui.dialogs.neb_vasp_dialog import (
        MEP_STAGE,
        NebPostprocessDialog,
    )

    QApplication.instance() or QApplication([])

    class Viewer:
        def __init__(self):
            self.picked = None

        def select_index(self, index):
            self.picked = index
            return True

    class Workspace:
        def __init__(self):
            self.viewer = Viewer()
            self.loaded = None

        def load_structure(self, structure):
            self.loaded = structure

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.workspace = Workspace()

    window = Window()

    dialog = NebPostprocessDialog(parent=window)
    dialog._loaded(nv.read_path(crowded_chain(tmp_path)))

    dialog.stage.setCurrentIndex(dialog.stage.findData(MEP_STAGE))
    dialog._show_structure()

    assert window.workspace.loaded is not None
    assert window.workspace.viewer.picked is None


def test_a_single_image_still_highlights_it(tmp_path):
    """The fix must not take the highlight off the other views."""

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication, QWidget

    from qvista.gui.dialogs.neb_vasp_dialog import NebPostprocessDialog

    QApplication.instance() or QApplication([])

    class Viewer:
        def __init__(self):
            self.picked = None

        def select_index(self, index):
            self.picked = index
            return True

    class Workspace:
        def __init__(self):
            self.viewer = Viewer()
            self.loaded = None

        def load_structure(self, structure):
            self.loaded = structure

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.workspace = Workspace()

    window = Window()

    dialog = NebPostprocessDialog(parent=window)
    dialog._loaded(nv.read_path(crowded_chain(tmp_path)))

    dialog.stage.setCurrentIndex(0)
    dialog._show_structure()

    assert window.workspace.viewer.picked == dialog.result.moving_atom


# ------------------------------------------------------------
# Both routes end in the same place
# ------------------------------------------------------------


def test_the_force_field_route_fills_the_same_fields():
    """
    A relaxed path is a relaxed path. If the two routes did not end with
    the same fields, the shared tables and figure would work for one and
    silently do nothing for the other.
    """

    from pymatgen.core import Lattice, Structure

    from qvista.analyses.neb import NEBResult

    structures = [
        Structure(
            Lattice.cubic(10.0),
            ["Li", "O", "O"],
            [
                [0.0, 0.0, 0.05 * index],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
            ],
        )
        for index in range(5)
    ]

    result = NEBResult(
        energies=[0.0, 0.2, 0.3, 0.2, 0.0], saddle_index=2, n_images=5
    )

    nv.enrich(result, structures, [np.zeros((3, 3))] * 5)

    assert len(result.dat) == 5
    assert len(result.spline[0]) > 5
    assert result.moving_atom == 0
    assert len(result.environments) == 3
    assert result.neighbour_cutoff == nv.NEIGHBOUR_CUTOFF


def test_the_tangent_force_is_zero_at_the_endpoints():
    """VASP never moves them, and neither does the force-field route."""

    from pymatgen.core import Lattice, Structure

    structures = [
        Structure(
            Lattice.cubic(10.0), ["Li"], [[0.0, 0.0, 0.05 * index]]
        )
        for index in range(4)
    ]

    forces = [np.array([[0.0, 0.0, 0.3]]) for _ in structures]

    projected = nv.tangent_forces(structures, forces)

    assert projected[0] == 0.0
    assert projected[-1] == 0.0

    assert any(value != 0.0 for value in projected[1:-1])


def test_the_tangent_uses_the_neighbouring_images():
    """
    The central estimate, which is what a relaxed band's tangent has
    converged to.
    """

    from pymatgen.core import Lattice, Structure

    structures = [
        Structure(
            Lattice.cubic(10.0), ["Li"], [[0.0, 0.0, 0.1 * index]]
        )
        for index in range(3)
    ]

    # A force straight along the path: the projection is its magnitude.
    forces = [np.array([[0.0, 0.0, 0.5]]) for _ in structures]

    projected = nv.tangent_forces(structures, forces)

    assert projected[1] == pytest.approx(0.5)


# ------------------------------------------------------------
# The combined CHGNet window
# ------------------------------------------------------------


def test_the_force_field_route_is_one_window():
    """
    The VASP split exists because the calculation happens elsewhere. A
    force field has no such gap.
    """

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.neb_chgnet_dialog import NebChgnetDialog

    QApplication.instance() or QApplication([])

    dialog = NebChgnetDialog()

    # Asks for everything.
    assert dialog.start is not None
    assert dialog.end is not None
    assert dialog.climb.isChecked()
    assert dialog.fmax.value() > 0
    assert dialog.images.value() >= 1

    # ...and shows everything.
    assert dialog.table is not None
    assert dialog.analysis is not None
    assert dialog.stage is not None


def test_both_routes_share_the_showing_half():

    pytest.importorskip("PySide6.QtWidgets")

    from qvista.gui.dialogs.neb_chgnet_dialog import NebChgnetDialog
    from qvista.gui.dialogs.neb_vasp_dialog import (
        NebPostprocessDialog,
        NebResultsView,
    )

    for dialog in (NebChgnetDialog, NebPostprocessDialog):
        assert issubclass(dialog, NebResultsView)

    for name in ("_draw", "_fill", "_fill_analysis", "_show_structure"):
        assert hasattr(NebResultsView, name)


def test_the_force_field_window_needs_both_endpoints():

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.neb_chgnet_dialog import NebChgnetDialog

    QApplication.instance() or QApplication([])

    dialog = NebChgnetDialog()

    dialog._run()

    assert "endpoint" in dialog.status.text().lower()


# ------------------------------------------------------------
# Endpoints relaxed separately
# ------------------------------------------------------------


def endpoint_pair(tmp_path, cell_shift=0.0, wrap=False):
    """
    Two endpoints written as separate POSCARs.

    ``cell_shift`` nudges the final cell, as relaxing the two separately
    with the cell free always does. ``wrap`` puts one atom just across a
    cell face, which is what makes an interpolation take the long way.
    """

    from ase import Atoms
    from ase.io import write

    cell = np.diag([6.0, 6.0, 6.0])

    start = Atoms(
        "LiO2",
        scaled_positions=[[0.05, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        cell=cell,
        pbc=True,
    )

    moved = 0.95 if wrap else 0.30

    end = Atoms(
        "LiO2",
        scaled_positions=[[moved, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        cell=cell + np.diag([cell_shift, 0.0, 0.0]),
        pbc=True,
    )

    first = tmp_path / "initial.vasp"
    last = tmp_path / "final.vasp"

    write(str(first), start, format="vasp")
    write(str(last), end, format="vasp")

    return first, last


def analysis_for(first, last):

    from qvista.analyses.neb import NEBAnalysis
    from qvista.models.project import Project
    from qvista.models.project_info import ProjectInfo
    from qvista.services.structure_loader import StructureLoader

    project = Project(
        info=ProjectInfo(
            name=first.stem, directory=first.parent, source_file=first
        ),
        structure=StructureLoader.load(first),
    )

    return NEBAnalysis(project, last, n_images=2, fmax=0.1, steps=1)


def test_endpoints_relaxed_apart_still_build_a_band(tmp_path):
    """
    ASE refuses a band whose cells differ by more than 1e-8 Angstrom,
    and two endpoints relaxed separately with the cell free never agree
    to that -- the pair this was found on differed by 1.3e-3.
    """

    first, last = endpoint_pair(tmp_path, cell_shift=1.3e-3)

    initial, final = analysis_for(first, last).endpoints()

    assert np.allclose(initial.cell.array, final.cell.array, atol=0.0)


def test_the_shared_cell_is_the_initial_one(tmp_path):
    """
    Not an average of the two, which belongs to neither endpoint.
    """

    from ase.io import read

    first, last = endpoint_pair(tmp_path, cell_shift=1.3e-3)

    initial, _ = analysis_for(first, last).endpoints()

    assert np.allclose(
        initial.cell.array, read(str(first)).cell.array, atol=1e-9
    )


def test_moving_onto_the_shared_cell_keeps_the_site(tmp_path):
    """
    By fractional coordinates, not Cartesian. The atom is in a site, and
    carrying Cartesian positions across would shift every atom by the
    cell difference and count that as migration.
    """

    first, last = endpoint_pair(tmp_path, cell_shift=0.5)

    initial, final = analysis_for(first, last).endpoints()

    displacement = np.linalg.norm(
        final.positions - initial.positions, axis=1
    )

    # Only the atom that actually moved.
    assert int((displacement > 0.1).sum()) == 1


def test_an_atom_across_a_cell_face_takes_the_short_way(tmp_path):
    """
    Written at 0.95 against 0.05, a straight interpolation drags the
    atom the long way round -- right across the cell, through every
    atom in between -- and relaxes a path nobody asked about.
    """

    first, last = endpoint_pair(tmp_path, wrap=True)

    initial, final = analysis_for(first, last).endpoints()

    displacement = np.linalg.norm(
        final.positions - initial.positions, axis=1
    )

    # The short way is 0.6 Angstrom; the long way is 5.4.
    assert displacement.max() < 1.0


def test_a_wrap_is_not_mistaken_for_a_migration(tmp_path):
    """
    An atom whose fractional coordinate goes 0.001 to 0.999 has not
    moved. Uncorrected it looks like the largest displacement in the
    cell, and the migrating atom is picked as the wrong one.
    """

    from ase import Atoms
    from ase.io import write

    cell = np.diag([6.0, 6.0, 6.0])

    write(
        str(tmp_path / "a.vasp"),
        Atoms(
            "LiO",
            scaled_positions=[[0.2, 0.0, 0.0], [0.001, 0.0, 0.0]],
            cell=cell,
            pbc=True,
        ),
        format="vasp",
    )

    write(
        str(tmp_path / "b.vasp"),
        Atoms(
            "LiO",
            scaled_positions=[[0.6, 0.0, 0.0], [0.999, 0.0, 0.0]],
            cell=cell,
            pbc=True,
        ),
        format="vasp",
    )

    initial, final = analysis_for(
        tmp_path / "a.vasp", tmp_path / "b.vasp"
    ).endpoints()

    displacement = np.linalg.norm(
        final.positions - initial.positions, axis=1
    )

    # The lithium moves; the oxygen only wrapped.
    assert displacement[0] > displacement[1]

    assert displacement[1] < 0.1


def test_identical_cells_are_left_alone(tmp_path):
    """Nothing to correct, and nothing should be touched."""

    first, last = endpoint_pair(tmp_path, cell_shift=0.0)

    initial, final = analysis_for(first, last).endpoints()

    assert np.allclose(initial.cell.array, final.cell.array, atol=0.0)


def test_the_recommended_tangent_is_asked_for():
    """
    ASE's old default is unpublished and, in its own words, "frequently
    results in very poor bands".
    """

    import inspect

    from qvista.analyses.neb import NEBAnalysis

    source = inspect.getsource(NEBAnalysis.execute)

    assert "improvedtangent" in source


# ------------------------------------------------------------------
# The two routes share one results view
# ------------------------------------------------------------------


def test_each_fill_belongs_to_its_own_dialog() -> None:
    """
    The chain table and the summary table are not the same method.

    They share a name and nothing else: one lists the folders being
    written before a run, the other summarises a finished one. Pulling
    the shared results view out of the postprocess dialog once matched
    the wrong one by name and gave every consumer of the mixin the
    preprocess signature, which failed only at the point of use.
    """

    import inspect

    from qvista.gui.dialogs.neb_vasp_dialog import (
        NebPostprocessDialog,
        NebPreprocessDialog,
        NebResultsView,
    )

    # The shared one summarises a result and needs nothing passed in.
    assert list(inspect.signature(NebResultsView._fill).parameters) == [
        "self"
    ]

    # The preprocess dialog owns its own, and it takes the chain.
    assert "_fill" in NebPreprocessDialog.__dict__

    assert list(
        inspect.signature(NebPreprocessDialog._fill).parameters
    ) == ["self", "chain", "coordinates", "moving", "written"]

    # The postprocess dialog no longer keeps a copy of the shared one.
    assert "_fill" not in NebPostprocessDialog.__dict__


def test_the_chgnet_dialog_can_show_a_result() -> None:
    """
    Every method the shared view calls resolves on the CHGNet dialog.

    The force-field route reaches the same view by a different door,
    and a mismatch there surfaces only after a relaxation that takes
    minutes -- which is the worst possible moment to find it.
    """

    from qvista.gui.dialogs.neb_chgnet_dialog import NebChgnetDialog
    from qvista.gui.dialogs.neb_vasp_dialog import NebResultsView

    for name in vars(NebResultsView):

        if name.startswith("__"):
            continue

        assert hasattr(NebChgnetDialog, name), name

    # And the widgets those methods read, on a real instance: they are
    # built in helpers rather than assigned in __init__, so only a
    # constructed dialog answers the question.
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    dialog = NebChgnetDialog(parent=None)

    for name in (
        "result",
        "table",
        "analysis",
        "analysis_label",
        "analysis_row",
        "cutoff",
        "stage",
        "status",
        "panel",
        "figures",
    ):
        assert hasattr(dialog, name), name


def test_both_routes_summarise_the_same_way() -> None:
    """
    One barrier and one set of rows, whichever route produced them.

    Reported separately these drifted: the force-field route still
    listed a forward and a reverse barrier side by side after the VASP
    route had been cut down to one.
    """

    from qvista.services import neb_vasp

    class Fake:
        force_field = "CHGNet"
        climbing = True
        barrier_forward = 0.2958
        saddle_index = 1

    rows = neb_vasp.summary_rows(Fake(), 4)

    assert [label for label, _ in rows] == [
        "Method",
        "Images",
        "Intermediate images",
        "Climbing image",
        "Barrier",
        "Saddle at image",
    ]

    values = dict(rows)

    assert values["Images"] == "4"
    assert values["Intermediate images"] == "2"
    assert values["Barrier"] == "0.2958 eV"

    # Nothing that invites picking a favourite number.
    assert "Forward barrier" not in values
    assert "Reverse barrier" not in values

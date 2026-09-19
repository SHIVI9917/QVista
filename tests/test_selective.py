"""
Tests for selective dynamics.

The flags are three characters per atom and easy to get subtly wrong in
ways that still run. Three things are guarded here above the rest.

**Which way round.** A selection names the atoms that are *free*. Both
conventions are in use, and the inverted one produces a POSCAR that runs
perfectly and relaxes exactly the atoms that were meant to be held.

**The partition.** An atom exactly on a cutting plane has to land on one
side and only one. Falling on both double-counts it; falling on neither
drops it, and a POSCAR with the wrong number of flag rows is rejected by
VASP with a message about nothing in particular.

**The complement.** "These, and not those" is what almost every real
selection is, and the complement of a selection of seventy atoms is
where the mistakes happen when it is written out by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvista.services import selective as sd


# ------------------------------------------------------------
# A slab to select from
# ------------------------------------------------------------


def slab(layers: int = 4, per_layer: int = 3):
    """
    A stack of layers along z, alternating species.

    Layer heights are 0, 3, 6, 9 ... so a cut at 4.5 takes exactly the
    bottom two, and a cut at 6.0 lands one layer exactly on the plane.
    """

    from pymatgen.core import Lattice, Structure

    species: list[str] = []
    coords: list[list[float]] = []

    height = 3.0 * layers + 6.0

    for layer in range(layers):
        for atom in range(per_layer):

            species.append("Pt" if layer % 2 == 0 else "O")

            coords.append(
                [
                    (atom + 0.5) / per_layer,
                    0.5,
                    (3.0 * layer) / height,
                ]
            )

    return Structure(
        Lattice.from_parameters(6.0, 6.0, height, 90.0, 90.0, 90.0),
        species,
        coords,
    )


def write(tmp_path: Path, structure=None, name: str = "POSCAR") -> Path:

    from pymatgen.io.vasp.inputs import Poscar

    path = tmp_path / name

    Poscar(structure if structure is not None else slab()).write_file(
        str(path)
    )

    return path


def cell(tmp_path) -> sd.Cell:

    return sd.load(write(tmp_path))


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


def test_a_structure_loads(tmp_path) -> None:

    loaded = cell(tmp_path)

    assert loaded.natoms == 12

    assert loaded.elements() == ["Pt", "O"]


def test_atom_numbers_count_from_one(tmp_path) -> None:
    """
    Matching every table in QVista and VASP's own output.

    Off by one here silently freezes the neighbouring atom.
    """

    loaded = cell(tmp_path)

    assert min(loaded.indices_of("Pt")) == 1

    assert max(loaded.indices_of("O")) == loaded.natoms


def test_the_span_says_where_the_atoms_are(tmp_path) -> None:
    """
    A cutting plane is only choosable against the span it cuts.

    "5 A" means nothing until you know z runs from 0 to 14.
    """

    low, high = cell(tmp_path).span(2)

    assert low == pytest.approx(0.0)
    assert high == pytest.approx(9.0)


def test_an_axis_outside_three_dimensions_is_refused(tmp_path) -> None:

    with pytest.raises(sd.SelectiveError, match="three axes"):
        cell(tmp_path).span(3)


def test_a_molecule_is_refused(tmp_path) -> None:
    """
    Selective dynamics is a POSCAR feature.

    An XYZ has no lattice, so there is no POSCAR to write the flags in.
    """

    path = tmp_path / "water.xyz"

    path.write_text(
        "3\nwater\nO 0 0 0\nH 0 0.76 0.59\nH 0 -0.76 0.59\n",
        encoding="utf-8",
    )

    with pytest.raises(sd.SelectiveError, match="no lattice"):
        sd.load(path)


def test_a_missing_file_is_refused(tmp_path) -> None:

    with pytest.raises(sd.SelectiveError, match="no such file"):
        sd.load(tmp_path / "nowhere")


# ------------------------------------------------------------
# Axis cutoff
# ------------------------------------------------------------


def test_a_cut_takes_the_layers_below_it(tmp_path) -> None:

    below = sd.by_axis(cell(tmp_path), 2, 4.5, sd.BELOW)

    # Layers at z = 0 and 3, three atoms each.
    assert len(below) == 6


def test_the_two_sides_partition_the_structure(tmp_path) -> None:
    """
    Nothing claimed twice, nothing dropped.

    A POSCAR needs exactly one flag row per atom; either mistake makes
    VASP reject the file for reasons it does not explain well.
    """

    loaded = cell(tmp_path)

    below = set(sd.by_axis(loaded, 2, 4.5, sd.BELOW))
    above = set(sd.by_axis(loaded, 2, 4.5, sd.ABOVE))

    assert not (below & above)

    assert below | above == set(range(1, loaded.natoms + 1))


def test_an_atom_exactly_on_the_plane_goes_above(tmp_path) -> None:
    """
    It has to go somewhere, and only one somewhere.

    Pinned because "on the plane" is the case a slab cut written by
    hand lands on constantly -- layer heights are round numbers.
    """

    loaded = cell(tmp_path)

    below = sd.by_axis(loaded, 2, 6.0, sd.BELOW)
    above = sd.by_axis(loaded, 2, 6.0, sd.ABOVE)

    # Layers at 0 and 3 below; the layer at exactly 6.0 goes above.
    assert len(below) == 6
    assert len(above) == 6


def test_the_other_axes_work_too(tmp_path) -> None:

    loaded = cell(tmp_path)

    for axis in (0, 1):
        chosen = sd.by_axis(loaded, axis, 3.0, sd.BELOW)

        assert 0 <= len(chosen) <= loaded.natoms


def test_an_unknown_side_is_refused(tmp_path) -> None:

    with pytest.raises(sd.SelectiveError, match="Unknown side"):
        sd.by_axis(cell(tmp_path), 2, 4.5, "middle")


# ------------------------------------------------------------
# Element and by hand
# ------------------------------------------------------------


def test_an_element_takes_all_of_it(tmp_path) -> None:

    chosen = sd.by_element(cell(tmp_path), ["Pt"])

    assert len(chosen) == 6


def test_several_elements_combine(tmp_path) -> None:

    loaded = cell(tmp_path)

    chosen = sd.by_element(loaded, ["Pt", "O"])

    assert len(chosen) == loaded.natoms


def test_an_element_that_is_not_there_says_what_is(tmp_path) -> None:

    with pytest.raises(sd.SelectiveError, match="no Fe in this"):
        sd.by_element(cell(tmp_path), ["Fe"])


def test_no_element_at_all_is_refused(tmp_path) -> None:

    with pytest.raises(sd.SelectiveError, match="at least one element"):
        sd.by_element(cell(tmp_path), [])


def test_atom_numbers_accept_ranges(tmp_path) -> None:

    assert sd.by_hand(cell(tmp_path), "1-4, 7") == [1, 2, 3, 4, 7]


def test_atom_numbers_accept_spaces(tmp_path) -> None:
    """
    A list pasted out of a terminal arrives space separated.

    Refusing it is pedantry, and the failure looks like a parse bug
    rather than a formatting rule.
    """

    assert sd.by_hand(cell(tmp_path), "1 2 3") == [1, 2, 3]

    assert sd.by_hand(cell(tmp_path), "1-3\n8\n9") == [1, 2, 3, 8, 9]


def test_an_atom_past_the_end_is_refused(tmp_path) -> None:

    with pytest.raises(sd.SelectiveError, match="outside it"):
        sd.by_hand(cell(tmp_path), "1, 99")


def test_naming_nothing_is_refused(tmp_path) -> None:

    with pytest.raises(sd.SelectiveError, match="at least one atom"):
        sd.by_hand(cell(tmp_path), "   ")


# ------------------------------------------------------------
# The rest
# ------------------------------------------------------------


def test_the_rest_is_everything_else(tmp_path) -> None:

    loaded = cell(tmp_path)

    chosen = sd.by_element(loaded, ["Pt"])

    complement = sd.rest(loaded, chosen)

    assert len(complement) == loaded.natoms - len(chosen)

    assert not set(complement) & set(chosen)

    assert set(complement) | set(chosen) == set(
        range(1, loaded.natoms + 1)
    )


def test_the_rest_of_everything_is_nothing(tmp_path) -> None:

    loaded = cell(tmp_path)

    everything = list(range(1, loaded.natoms + 1))

    assert sd.rest(loaded, everything) == []


def test_the_rest_of_nothing_is_everything(tmp_path) -> None:

    loaded = cell(tmp_path)

    assert len(sd.rest(loaded, [])) == loaded.natoms


def test_inverting_twice_comes_back(tmp_path) -> None:

    loaded = cell(tmp_path)

    chosen = sd.by_axis(loaded, 2, 4.5, sd.BELOW)

    assert sd.rest(loaded, sd.rest(loaded, chosen)) == sorted(chosen)


# ------------------------------------------------------------
# The flags
# ------------------------------------------------------------


def test_the_selection_names_the_free_atoms(tmp_path) -> None:
    """
    The direction the whole feature turns on.

    Inverted, the POSCAR runs perfectly and relaxes exactly the atoms
    that were meant to be held.
    """

    loaded = cell(tmp_path)

    free = sd.by_axis(loaded, 2, 4.5, sd.ABOVE)

    selection = sd.build(loaded, free=free)

    flags = selection.flags

    for number in free:
        assert flags[number - 1] == (True, True, True)

    for number in selection.frozen:
        assert flags[number - 1] == (False, False, False)


def test_there_is_one_flag_row_per_atom(tmp_path) -> None:

    loaded = cell(tmp_path)

    selection = sd.build(loaded, free=[1, 2, 3])

    assert len(selection.flags) == loaded.natoms


def test_the_axes_restrict_a_free_atom(tmp_path) -> None:
    """
    Free along z only is a common slab constraint.

    It is a constraint, not a symmetry -- the forces along x and y are
    still there and are simply not acted on.
    """

    loaded = cell(tmp_path)

    selection = sd.build(
        loaded, free=[1, 2], axes=(False, False, True)
    )

    assert selection.flags[0] == (False, False, True)

    # And a frozen atom is still frozen in every direction.
    assert selection.flags[-1] == (False, False, False)


def test_an_atom_outside_the_structure_is_refused(tmp_path) -> None:

    with pytest.raises(sd.SelectiveError, match="outside this"):
        sd.build(cell(tmp_path), free=[99])


def test_freezing_everything_is_called_out(tmp_path) -> None:
    """
    VASP runs and every ionic step changes nothing.

    That looks exactly like instant convergence.
    """

    selection = sd.build(cell(tmp_path), free=[])

    assert any(
        "nothing is free" in note for note in selection.notes
    )


def test_freeing_everything_is_called_out(tmp_path) -> None:

    loaded = cell(tmp_path)

    selection = sd.build(
        loaded, free=list(range(1, loaded.natoms + 1))
    )

    assert any("Every atom is free" in note for note in selection.notes)


def test_no_direction_at_all_is_called_out(tmp_path) -> None:

    selection = sd.build(
        cell(tmp_path), free=[1, 2], axes=(False, False, False)
    )

    assert any(
        "no direction is allowed" in note for note in selection.notes
    )


def test_the_convention_is_always_stated(tmp_path) -> None:
    """Because both are in use and the file itself does not say."""

    selection = sd.build(cell(tmp_path), free=[1])

    assert any("T means free" in note for note in selection.notes)


# ------------------------------------------------------------
# Writing
# ------------------------------------------------------------


def test_the_poscar_carries_selective_dynamics(tmp_path) -> None:

    text = sd.build(cell(tmp_path), free=[1, 2]).to_poscar()

    assert "elective" in text


def test_the_flags_survive_a_round_trip(tmp_path) -> None:
    """
    Written and read back, the same atoms are free.

    The one check that the flags landed on the atoms they were meant
    for rather than in a different order.
    """

    from pymatgen.io.vasp.inputs import Poscar

    loaded = cell(tmp_path)

    free = sd.by_axis(loaded, 2, 4.5, sd.ABOVE)

    selection = sd.build(loaded, free=free)

    target = sd.write(selection, tmp_path / "POSCAR_sd")

    back = Poscar.from_file(str(target), check_for_potcar=False)

    recovered = [
        index + 1
        for index, flags in enumerate(back.selective_dynamics)
        if all(flags)
    ]

    assert recovered == sorted(free)


def test_partial_axes_survive_the_round_trip(tmp_path) -> None:

    from pymatgen.io.vasp.inputs import Poscar

    loaded = cell(tmp_path)

    selection = sd.build(
        loaded, free=[1, 2], axes=(True, False, False)
    )

    target = sd.write(selection, tmp_path / "POSCAR_partial")

    back = Poscar.from_file(str(target), check_for_potcar=False)

    assert list(back.selective_dynamics[0]) == [True, False, False]


def test_the_structure_is_unchanged_by_the_flags(tmp_path) -> None:
    """Selective dynamics says what may move, not where anything is."""

    from pymatgen.io.vasp.inputs import Poscar

    source = write(tmp_path)

    loaded = sd.load(source)

    target = sd.write(
        sd.build(loaded, free=[1, 2, 3]), tmp_path / "POSCAR_sd"
    )

    before = Poscar.from_file(
        str(source), check_for_potcar=False
    ).structure

    after = Poscar.from_file(
        str(target), check_for_potcar=False
    ).structure

    assert np.abs(
        before.frac_coords - after.frac_coords
    ).max() == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------
# What it says about itself
# ------------------------------------------------------------


def test_the_summary_counts_both_sides(tmp_path) -> None:

    loaded = cell(tmp_path)

    rows = dict(sd.build(loaded, free=[1, 2, 3]).summary())

    assert rows["Atoms"] == "12"

    assert rows["Free to move"].startswith("3")
    assert rows["Held fixed"] == "9"


def test_the_per_element_split_adds_up(tmp_path) -> None:

    loaded = cell(tmp_path)

    selection = sd.build(loaded, free=sd.by_element(loaded, ["Pt"]))

    split = dict(
        (element, (free, frozen))
        for element, free, frozen in selection.per_element()
    )

    assert split["Pt"] == (6, 0)
    assert split["O"] == (0, 6)


# ------------------------------------------------------------
# Writing a set of picked atoms back out
# ------------------------------------------------------------
#
# Atoms can be chosen by clicking them in the viewer, and what comes
# back is a set of indices that has to go into the same text box a user
# types into. It has to survive that round trip: the parser above reads
# what this writes, and anything it cannot read would turn a selection
# made by clicking into an error message.


def _ranges(numbers):
    """The dialog's collapser, imported without building a window."""

    from qvista.gui.dialogs.selective_dialog import _ranges as collapse

    return collapse(numbers)


def test_picked_atoms_collapse_into_ranges() -> None:
    """
    A complement is usually most of the structure, and forty numbers
    in a text box is unreadable and uneditable.
    """

    assert _ranges([1, 2, 3, 4, 5, 6, 9, 12, 13, 14]) == "1-6, 9, 12-14"


def test_a_single_atom_is_not_written_as_a_range() -> None:

    assert _ranges([7]) == "7"
    assert _ranges([3, 9]) == "3, 9"


def test_a_pair_stays_a_pair() -> None:
    """Two in a row is shorter written out than hyphenated is clear."""

    assert _ranges([4, 5]) == "4-5"


def test_picks_arrive_in_any_order() -> None:
    """
    Clicking atom 9 and then atom 2 is a normal thing to do, and the
    viewer hands them back in the order they were clicked.
    """

    assert _ranges([9, 2, 3, 1]) == "1-3, 9"


def test_nothing_picked_is_an_empty_box() -> None:

    assert _ranges([]) == ""


def test_what_clicking_writes_is_what_typing_reads(tmp_path) -> None:
    """
    The round trip that matters.

    ``by_hand`` parses what a user types; the viewer writes into the
    same box. A set that came back out different would freeze atoms
    nobody chose.
    """

    loaded = cell(tmp_path)

    picked = [1, 2, 3, 5, 8]

    assert sd.by_hand(loaded, _ranges(picked)) == picked


def test_a_whole_structure_survives_the_round_trip(tmp_path) -> None:
    """Select-all by clicking is one range, and reads back as all."""

    loaded = cell(tmp_path)

    every = list(range(1, loaded.natoms + 1))

    assert _ranges(every) == f"1-{loaded.natoms}"

    assert sd.by_hand(loaded, _ranges(every)) == every

"""
Choosing the path a phonon dispersion is drawn along.

Distinct from the Crystallography k-path, which describes a structure
and writes KPOINTS. This one changes a figure whose force constants
already exist, so the thing to get right is that a break in the path
stays a break -- a pair spanning one would draw a line between two
points the path never travels between, and it would look like a real
branch.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("seekpath")
pytest.importorskip("phonopy")

from qvista.services import phonon_path as pp  # noqa: E402


@pytest.fixture(scope="module")
def phonon():
    """Silicon with fabricated force constants -- enough to diagonalise."""

    from ase.build import bulk
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    atoms = bulk("Si", "diamond", a=5.43)

    cell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )

    built = Phonopy(cell, supercell_matrix=np.eye(3) * 2)

    size = len(built.supercell)

    built.force_constants = (
        np.tile(np.eye(3), (size, size, 1, 1)) * 0.05
    )

    return built


@pytest.fixture(scope="module")
def options(phonon):

    return pp.options_for(phonon.primitive)


# ----------------------------------------------------------------
# What is on offer
# ----------------------------------------------------------------


def test_the_points_come_from_the_cell_the_matrix_is_built_on(options):
    """
    Not from whatever structure is open. Post-processing runs on folders
    produced elsewhere, and a path in another cell's reciprocal basis
    would be wrong rather than rejected.
    """

    assert "GAMMA" in options.points
    assert len(options.points) > 3


def test_gamma_is_offered_first(options):
    """A path essentially always starts there."""

    assert options.labels()[0] == "GAMMA"


def test_the_rest_are_alphabetical(options):

    rest = options.labels()[1:]

    assert rest == sorted(rest)


def test_a_recommended_path_comes_with_them(options):

    assert options.recommended
    assert all(len(pair) == 2 for pair in options.recommended)


def test_every_recommended_label_has_a_coordinate(options):

    for start, end in options.recommended:
        assert start in options.points
        assert end in options.points


# ----------------------------------------------------------------
# Breaks
# ----------------------------------------------------------------


def test_a_continuous_path_is_one_stretch():

    assert pp.group_segments([("A", "B"), ("B", "C")]) == [["A", "B", "C"]]


def test_a_jump_starts_a_new_stretch():
    """
    The whole point. Drawn as one stretch, the jump becomes a line
    between two points the path never travels between -- and it looks
    like a branch.
    """

    assert pp.group_segments([("A", "B"), ("C", "D")]) == [
        ["A", "B"],
        ["C", "D"],
    ]


def test_the_label_marks_a_break_the_way_the_axis_does():

    assert pp.path_label([("G", "X"), ("X", "W")]) == "G-X-W"

    assert pp.path_label([("G", "X"), ("K", "G")]) == "G-X | K-G"


def test_an_empty_path_says_so():

    assert pp.path_label([]) == "(empty)"


# ----------------------------------------------------------------
# Building q-points
# ----------------------------------------------------------------


def test_each_stretch_becomes_its_own_block(options):

    qpoints, labels, connections = pp.build_qpoints(
        options.points,
        [("GAMMA", "X"), ("X", "W"), ("L", "GAMMA")],
        npoints=21,
    )

    # Three segments in two stretches: G-X, X-W, then L-G.
    assert len(qpoints) == 3

    # The connection flags say where the path continues and where it
    # breaks, which is what puts the bar on the axis.
    assert connections[-1] is False or connections[-1] == False  # noqa: E712


def test_the_labels_cover_every_point_visited(options):

    _, labels, _ = pp.build_qpoints(
        options.points, [("GAMMA", "X"), ("X", "W")], npoints=11
    )

    assert labels == ["GAMMA", "X", "W"]


def test_the_sampling_is_honoured(options):

    qpoints, _, _ = pp.build_qpoints(
        options.points, [("GAMMA", "X")], npoints=31
    )

    assert np.asarray(qpoints[0]).shape[0] == 31


def test_an_unknown_point_is_named_rather_than_guessed(options):

    with pytest.raises(pp.PhononPathError) as raised:
        pp.build_qpoints(
            options.points, [("GAMMA", "NOWHERE")], npoints=11
        )

    assert "NOWHERE" in str(raised.value)


def test_an_empty_path_is_refused(options):

    with pytest.raises(pp.PhononPathError):
        pp.build_qpoints(options.points, [], npoints=11)


def test_the_sampling_is_held_inside_something_drawable():

    assert pp.clamp_npoints(2) == pp.MIN_NPOINTS
    assert pp.clamp_npoints(100000) == pp.MAX_NPOINTS
    assert pp.clamp_npoints("many") == pp.DEFAULT_NPOINTS


# ----------------------------------------------------------------
# Re-running
# ----------------------------------------------------------------


def test_a_custom_path_produces_a_band(phonon, options):

    band = pp.run_band(
        phonon,
        options.points,
        [("GAMMA", "X"), ("X", "W")],
        npoints=21,
    )

    assert "frequencies" in band
    assert "distances" in band


def test_the_band_follows_the_path_it_was_given(phonon, options):
    """Two segments in, two blocks of frequencies out."""

    band = pp.run_band(
        phonon, options.points, [("GAMMA", "X"), ("X", "W")], npoints=21
    )

    assert len(band["frequencies"]) == 2


def test_a_different_path_gives_a_different_band(phonon, options):

    first = pp.run_band(
        phonon, options.points, [("GAMMA", "X")], npoints=21
    )
    second = pp.run_band(
        phonon, options.points, [("GAMMA", "L")], npoints=21
    )

    assert not np.allclose(
        np.asarray(first["frequencies"][0]),
        np.asarray(second["frequencies"][0]),
    )


def test_eigenvectors_come_back_by_default(phonon, options):
    """
    They cost nothing extra -- the diagonalisation happens anyway -- and
    the element-projected figure is built from them. Without them,
    switching to that view after changing the path would need a second
    pass over the whole path.
    """

    band = pp.run_band(
        phonon, options.points, [("GAMMA", "X")], npoints=11
    )

    assert "eigenvectors" in band


def test_the_band_carries_what_the_axis_needs(phonon, options):
    """
    Labels and connections, not just numbers. Without them the figure
    has a dispersion with no ticks on its axis and no idea where the
    path breaks -- which is what ``get_band_structure_dict()`` alone
    would have given.
    """

    band = pp.run_band(
        phonon,
        options.points,
        [("GAMMA", "X"), ("L", "GAMMA")],
        npoints=11,
    )

    for key in (
        "qpoints",
        "distances",
        "frequencies",
        "labels",
        "path_connections",
        "symbols",
    ):
        assert key in band, key


def test_the_band_matches_the_shape_the_analyses_produce(phonon, options):
    """
    One dictionary shape for every route, so the figures do not need to
    know whether the path was the recommended one or chosen by hand.
    """

    band = pp.run_band(
        phonon, options.points, [("GAMMA", "X")], npoints=11
    )

    assert band["labels"] == ["GAMMA", "X"]
    assert len(band["symbols"]) == len(phonon.primitive)


def test_a_break_is_recorded_in_the_connections(phonon, options):

    band = pp.run_band(
        phonon,
        options.points,
        [("GAMMA", "X"), ("L", "GAMMA")],
        npoints=11,
    )

    # Two stretches, so the flag between them says "do not join".
    assert False in list(band["path_connections"])

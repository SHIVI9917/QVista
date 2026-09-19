"""
Element-pair bond rules, and the projection the camera draws with.

Bonds in a viewer are a drawing convention. The tests that matter are
the ones where the convention and the geometry disagree: an intercalated
alkali metal that clears the covalent criterion and should not be drawn
bonded anyway, and a rule written for one direction of a pair that has
to hold for the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.models.visible_atom import VisibleAtom
from qvista.visualization.bonding import (
    DEFAULT_SCALE,
    FROM_RADII,
    MAX_CUTOFF,
    BondRules,
    element_pairs,
    pair_key,
)
from qvista.visualization.settings import Projection, ViewerSettings
from qvista.visualization.visible_bond_builder import (
    VisibleBondBuilder,
    bond_pairs,
)
from qvista.visualization.viewer import _only_projection_changed


# ----------------------------------------------------------------
# A layered cathode
# ----------------------------------------------------------------


def slab() -> list[VisibleAtom]:
    """
    Four atoms of a layered oxide: a Co, the two O it is bonded to, and
    a Li sitting 2 A off the upper O.

    That Li-O separation clears the covalent criterion, so the default
    convention draws a bond there. It is the bond an intercalation
    figure has to be able to remove.
    """

    # The indices have to differ: an atom is keyed by its index and
    # periodic image, and four atoms sharing index 0 are one atom as far
    # as a frozen bond list is concerned.
    return [
        VisibleAtom(
            symbol="Co", position=np.array([0.0, 0.0, 0.0]), original_index=0
        ),
        VisibleAtom(
            symbol="O", position=np.array([0.0, 0.0, 2.0]), original_index=1
        ),
        VisibleAtom(
            symbol="O", position=np.array([0.0, 0.0, -2.0]), original_index=2
        ),
        VisibleAtom(
            symbol="Li", position=np.array([0.0, 0.0, 4.0]), original_index=3
        ),
    ]


def kinds(graph) -> set[tuple[str, str]]:
    """Which element pairs the graph actually drew."""

    return {
        pair_key(bond.atom_i.symbol, bond.atom_j.symbol)
        for bond in graph.bonds
    }


# ----------------------------------------------------------------
# Pair keys
# ----------------------------------------------------------------


def test_a_pair_has_one_key_whichever_way_round_it_is_given():
    """
    A bond has no direction. Two keys for one pair would mean a rule
    written from the dialog silently missing the bonds the builder
    happened to reach from the other atom first.
    """

    assert pair_key("Li", "O") == pair_key("O", "Li")


def test_a_rule_holds_in_both_directions():

    rules = BondRules()
    rules.set_enabled("Li", "O", False)

    assert not rules.is_enabled("Li", "O")
    assert not rules.is_enabled("O", "Li")


def test_every_pair_is_offered_homonuclear_ones_included():
    """
    Three elements give six pairs, not three. O-O is a real contact in a
    dense oxide and is the one a user reaches for after widening
    something else.
    """

    pairs = element_pairs(["Li", "Co", "O"])

    assert len(pairs) == 6
    assert ("O", "O") in pairs
    assert ("Co", "Li") in pairs


def test_the_pairs_do_not_depend_on_the_order_the_elements_arrive_in():

    assert element_pairs(["O", "Li", "Co"]) == element_pairs(
        ["Li", "Co", "O"]
    )


def test_a_repeated_element_is_only_offered_once():

    assert element_pairs(["Li", "Li", "O"]) == element_pairs(["Li", "O"])


# ----------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------


def test_everything_is_drawn_until_something_says_otherwise():

    rules = BondRules()

    assert rules.is_enabled("Li", "O")
    assert rules.is_default("Li", "O")


def test_the_default_cutoff_is_the_covalent_radii_with_slack():

    from qvista.visualization.radii import covalent_radius

    rules = BondRules()

    expected = DEFAULT_SCALE * (
        covalent_radius("Co") + covalent_radius("O")
    )

    assert rules.cutoff("Co", "O") == pytest.approx(expected)


def test_an_untouched_pair_is_not_stored():
    """
    Only departures from the default are kept, so that a pair left alone
    goes on tracking its radii rather than being pinned to whatever they
    gave at the moment the rules were made.
    """

    rules = BondRules()

    rules.cutoff("Li", "O")
    rules.is_enabled("Li", "O")

    assert rules.rules == {}


def test_a_cutoff_of_zero_means_ask_the_radii():

    rules = BondRules()
    rules.set_enabled("Li", "O", False)

    assert rules.rule_for("Li", "O").cutoff == FROM_RADII
    assert rules.cutoff("Li", "O") == rules.default_cutoff("Li", "O")


# ----------------------------------------------------------------
# Editing
# ----------------------------------------------------------------


def test_switching_a_pair_off_leaves_its_cutoff_alone():
    """
    The two settings are independent. Switching a pair back on should
    bring back the distance it had, not a default.
    """

    rules = BondRules()

    rules.set_cutoff("Li", "O", 2.6)
    rules.set_enabled("Li", "O", False)
    rules.set_enabled("Li", "O", True)

    assert rules.cutoff("Li", "O") == pytest.approx(2.6)


def test_a_cutoff_is_held_inside_something_drawable():
    """
    Past a few Angstroms every atom in the cell is within range and the
    picture is a solid mass of cylinders.
    """

    rules = BondRules()

    rules.set_cutoff("Li", "O", 500.0)

    assert rules.cutoff("Li", "O") == MAX_CUTOFF

    rules.set_cutoff("Li", "O", -3.0)

    assert rules.cutoff("Li", "O") >= 0.0


def test_clearing_a_cutoff_puts_the_pair_back_on_its_radii():

    rules = BondRules()

    rules.set_cutoff("Co", "O", 3.0)
    rules.clear_cutoff("Co", "O")

    assert rules.cutoff("Co", "O") == rules.default_cutoff("Co", "O")
    assert rules.is_default("Co", "O")


def test_reset_forgets_everything():

    rules = BondRules()

    rules.set_enabled("Li", "O", False)
    rules.set_cutoff("Co", "O", 3.0)

    rules.reset()

    assert rules.rules == {}
    assert rules.is_default("Li", "O")


def test_a_copy_is_independent():
    """
    The dialog edits a copy, so that a change it does not emit cannot
    reach the scene by accident.
    """

    rules = BondRules()
    rules.set_cutoff("Li", "O", 2.5)

    other = rules.copy()
    other.set_cutoff("Li", "O", 3.0)
    other.set_enabled("Co", "O", False)

    assert rules.cutoff("Li", "O") == pytest.approx(2.5)
    assert rules.is_enabled("Co", "O")


# ----------------------------------------------------------------
# The cutoff table the builder reads
# ----------------------------------------------------------------


def test_a_disabled_pair_is_absent_from_the_table():
    """
    Absent rather than zero, so the builder skips it on a dictionary
    miss instead of needing a second test per candidate pair.
    """

    rules = BondRules()
    rules.set_enabled("Li", "O", False)

    table = rules.cutoff_table(["Li", "Co", "O"])

    assert ("Li", "O") not in table
    assert ("Co", "O") in table


def test_the_table_holds_squared_cutoffs():
    """
    Squared, so the inner loop never takes a square root. On a wide
    boundary that loop runs to hundreds of thousands of candidates.
    """

    rules = BondRules()

    table = rules.cutoff_table(["Co", "O"])

    assert table[("Co", "O")] == pytest.approx(
        rules.cutoff("Co", "O") ** 2
    )


# ----------------------------------------------------------------
# The bonds actually drawn
# ----------------------------------------------------------------


def test_the_default_convention_bonds_the_intercalated_lithium():
    """
    Not a bug -- it is what the covalent criterion says, and it is the
    reason the rules exist.
    """

    graph = VisibleBondBuilder().build(slab())

    assert ("Li", "O") in kinds(graph)


def test_switching_a_pair_off_removes_exactly_those_bonds():

    rules = BondRules()
    rules.set_enabled("Li", "O", False)

    graph = VisibleBondBuilder().build(slab(), rules=rules)

    assert ("Li", "O") not in kinds(graph)
    assert ("Co", "O") in kinds(graph)


def test_narrowing_a_cutoff_removes_the_longer_bonds():

    rules = BondRules()
    rules.set_cutoff("Co", "O", 1.5)

    graph = VisibleBondBuilder().build(slab(), rules=rules)

    # The Co-O separation in the slab is 2.0 A.
    assert ("Co", "O") not in kinds(graph)


def test_widening_a_cutoff_brings_bonds_in():

    far = [
        VisibleAtom(symbol="Li", position=np.array([0.0, 0.0, 0.0])),
        VisibleAtom(symbol="O", position=np.array([0.0, 0.0, 3.2])),
    ]

    assert not VisibleBondBuilder().build(far).bonds

    rules = BondRules()
    rules.set_cutoff("Li", "O", 3.5)

    assert VisibleBondBuilder().build(far, rules=rules).bonds


def test_no_rules_draws_what_it_always_drew():
    """
    The rules are an addition, not a change of default. A caller that
    passes none has to get the covalent criterion it got before.
    """

    plain = VisibleBondBuilder().build(slab())
    empty = VisibleBondBuilder().build(slab(), rules=BondRules())

    assert len(plain.bonds) == len(empty.bonds)
    assert kinds(plain) == kinds(empty)


def test_the_distance_recorded_is_the_real_one():
    """
    The builder compares squared distances for speed; the bond still has
    to carry the distance itself, since a panel reports it.
    """

    graph = VisibleBondBuilder().build(slab())

    for bond in graph.bonds:

        expected = float(
            np.linalg.norm(bond.atom_j.position - bond.atom_i.position)
        )

        assert bond.distance == pytest.approx(expected)


# ----------------------------------------------------------------
# Frozen topology
# ----------------------------------------------------------------


def test_a_frozen_bond_list_still_respects_a_disabled_pair():
    """
    Freezing exists so that a vibrating bond does not blink as it
    crosses the cutoff. A pair the user switched off is a different
    matter and should be gone whether or not something is animating.
    """

    atoms = slab()

    frozen = bond_pairs(VisibleBondBuilder().build(atoms))

    rules = BondRules()
    rules.set_enabled("Li", "O", False)

    graph = VisibleBondBuilder().build(atoms, pairs=frozen, rules=rules)

    assert ("Li", "O") not in kinds(graph)
    assert ("Co", "O") in kinds(graph)


def test_a_frozen_bond_ignores_the_cutoff():
    """
    The whole point of freezing. A narrowed cutoff must not start
    dropping frames' worth of bonds mid-animation.
    """

    atoms = slab()

    frozen = bond_pairs(VisibleBondBuilder().build(atoms))

    rules = BondRules()
    rules.set_cutoff("Co", "O", 0.1)

    graph = VisibleBondBuilder().build(atoms, pairs=frozen, rules=rules)

    assert ("Co", "O") in kinds(graph)


# ----------------------------------------------------------------
# Projection
# ----------------------------------------------------------------


def test_the_camera_starts_parallel():
    """
    A crystal is looked at to judge whether things line up, and
    perspective makes that unreliable by drawing the far end of the cell
    smaller than the near end.
    """

    assert ViewerSettings().projection is Projection.PARALLEL


def test_a_projection_change_alone_is_recognised():
    """
    The panel rebuilds its settings from scratch and emits the lot, so
    the only way to know a click did not also change geometry is to
    compare. Getting this wrong costs a full rebuild of every sphere and
    cylinder on a camera change.
    """

    before = ViewerSettings()
    after = ViewerSettings(projection=Projection.PERSPECTIVE)

    assert _only_projection_changed(before, after)


def test_a_projection_change_with_anything_else_is_not():

    before = ViewerSettings()

    after = ViewerSettings(
        projection=Projection.PERSPECTIVE,
        show_bonds=False,
    )

    assert not _only_projection_changed(before, after)


def test_changed_bond_rules_are_not_mistaken_for_a_camera_move():
    """
    The rules live in the settings like everything else, and they need
    geometry rebuilt. Comparing only the fields somebody remembered to
    name is how that would be missed.
    """

    before = ViewerSettings()

    after = ViewerSettings(projection=Projection.PERSPECTIVE)
    after.bond_rules.set_enabled("Li", "O", False)

    assert not _only_projection_changed(before, after)


def test_identical_settings_are_not_a_projection_change():

    settings = ViewerSettings()

    assert not _only_projection_changed(settings, settings)


class FakeViewer:
    """
    Enough of a viewer to check which PyVista call the projection makes.

    The real one cannot be built in a test: VTK needs an OpenGL context
    and there is none on a headless machine. Binding the method onto a
    stub tests the part that is ours -- which call is made, and that the
    clipping range is recomputed after it.
    """

    def __init__(self, projection: Projection) -> None:

        self.settings = ViewerSettings(projection=projection)
        self.calls: list[str] = []

    def enable_parallel_projection(self) -> None:
        self.calls.append("parallel")

    def disable_parallel_projection(self) -> None:
        self.calls.append("perspective")

    def reset_camera_clipping_range(self) -> None:
        self.calls.append("clip")


@pytest.mark.parametrize(
    "projection,expected",
    [
        (Projection.PARALLEL, "parallel"),
        (Projection.PERSPECTIVE, "perspective"),
    ],
)
def test_the_projection_setting_reaches_the_camera(projection, expected):

    from qvista.visualization.viewer import CrystalViewer

    viewer = FakeViewer(projection)

    CrystalViewer.apply_projection(viewer)

    assert viewer.calls == [expected, "clip"]


def test_the_clipping_range_is_recomputed_after_a_switch():
    """
    The near plane VTK computed for one projection slices the structure
    open under the other, which looks like the model is broken rather
    than like the camera changed.
    """

    viewer = FakeViewer(Projection.PARALLEL)

    from qvista.visualization.viewer import CrystalViewer

    CrystalViewer.apply_projection(viewer)

    assert viewer.calls[-1] == "clip"


def test_the_comparison_covers_every_field():
    """
    A field added later must not be silently exempt. This is why the
    check walks the dataclass rather than naming what matters.
    """

    for name in ViewerSettings.__slots__:

        if name == "projection":
            continue

        before = ViewerSettings()
        after = ViewerSettings(projection=Projection.PERSPECTIVE)

        current = getattr(after, name)

        # Something guaranteed to differ, whatever the field holds.
        setattr(
            after,
            name,
            not current if isinstance(current, bool) else None,
        )

        assert not _only_projection_changed(before, after), name


# ----------------------------------------------------------------
# Space-filling radii
# ----------------------------------------------------------------


def test_a_transition_metal_has_a_real_van_der_waals_radius():
    """
    ASE's table is NaN for Sc through Ru, the lanthanides, Hf through Ir
    and most actinides. Falling back to the covalent radius for those
    answers a different question -- covalent runs about 0.6 of van der
    Waals -- and mixes two scales in one space-filling picture.
    """

    from qvista.visualization.radii import covalent_radius, vdw_radius

    for symbol in ("Sc", "Ti", "Cr", "Fe", "Co", "Y", "Mo", "La", "W"):

        # Not the covalent fallback, and in the range a metal's van der
        # Waals radius actually occupies.
        assert vdw_radius(symbol) != covalent_radius(symbol), symbol
        assert vdw_radius(symbol) >= 2.0, symbol


def test_boron_is_not_drawn_larger_than_cobalt():
    """
    The case that showed the bug. MgCoB rendered boron at ASE's 1.92 A
    and cobalt at its covalent 1.26 A, so the metalloid came out half
    again the size of the metal.
    """

    from qvista.visualization.radii import vdw_radius

    assert vdw_radius("B") < vdw_radius("Co")


def test_every_element_gets_a_usable_space_filling_radius():

    from ase.data import chemical_symbols

    from qvista.visualization.radii import vdw_radius

    for symbol in chemical_symbols[1:104]:

        radius = vdw_radius(symbol)

        assert 0.2 < radius < 4.0, symbol


def test_the_radii_of_one_period_are_ordered_sensibly():
    """
    Across a period the radius contracts. Two scales mixed together
    breaks that ordering, which is what made the picture look wrong
    before anyone checked a number.
    """

    from qvista.visualization.radii import vdw_radius

    period = ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"]

    radii = [vdw_radius(symbol) for symbol in period]

    assert radii == sorted(radii, reverse=True)


def test_a_missing_radius_is_looked_up_once():
    """
    Guards the cache sentinel. A plain ``.get()`` returns None for a
    missing key, which is indistinguishable from a cached "no value" and
    reports every element as unknown forever.
    """

    from qvista.visualization import radii

    radii._VDW_CACHE.pop("Co", None)

    first = radii.vdw_radius("Co")

    assert "Co" in radii._VDW_CACHE
    assert radii.vdw_radius("Co") == first


# ----------------------------------------------------------------
# Origin shift
# ----------------------------------------------------------------


def slab_structure():
    """A slab against the bottom of a tall cell, vacuum above."""

    from ase import Atoms

    from qvista.models.structure import Structure

    return Structure(
        atoms=Atoms(
            "H5",
            scaled_positions=[[0.5, 0.5, z] for z in (0.0, 0.1, 0.2, 0.3, 0.4)],
            cell=np.diag([4.0, 4.0, 30.0]),
            pbc=True,
        )
    )


def fractional_z(visible) -> list[float]:

    return sorted(round(float(atom.fractional[2]), 2) for atom in visible)


def test_no_shift_leaves_the_contents_alone():

    from qvista.visualization.periodic_image_generator import (
        PeriodicImageGenerator,
    )
    from qvista.visualization.settings import Boundary, Origin

    generator = PeriodicImageGenerator()

    plain = generator.generate(slab_structure(), Boundary())
    zero = generator.generate(slab_structure(), Boundary(), Origin())

    assert fractional_z(plain) == fractional_z(zero)


def test_a_shift_moves_the_contents():

    from qvista.visualization.periodic_image_generator import (
        PeriodicImageGenerator,
    )
    from qvista.visualization.settings import Boundary, Origin

    visible = PeriodicImageGenerator().generate(
        slab_structure(), Boundary(), Origin(z=0.2)
    )

    assert fractional_z(visible) == [0.2, 0.3, 0.4, 0.5, 0.6]


def test_an_atom_pushed_out_comes_back_the_other_side():
    """
    The wrap is what makes the control useful, and it comes free: the
    atom leaving the top falls outside the boundary while its periodic
    image one cell below falls inside.
    """

    from qvista.visualization.periodic_image_generator import (
        PeriodicImageGenerator,
    )
    from qvista.visualization.settings import Boundary, Origin

    visible = PeriodicImageGenerator().generate(
        slab_structure(), Boundary(), Origin(z=-0.1)
    )

    # The atom at 0.0 leaves the bottom and reappears at 0.9.
    assert 0.9 in fractional_z(visible)


def test_a_shift_never_loses_an_atom():
    """
    Sliding the contents is a change of viewpoint, not an edit. Whatever
    the shift, the same atoms have to be on screen.
    """

    from qvista.visualization.periodic_image_generator import (
        PeriodicImageGenerator,
    )
    from qvista.visualization.settings import Boundary, Origin

    generator = PeriodicImageGenerator()
    structure = slab_structure()

    for step in range(0, 21):

        visible = generator.generate(
            structure, Boundary(), Origin(z=step / 20.0)
        )

        # Five atoms, plus any sitting exactly on a boundary face and so
        # drawn at both ends.
        assert len(visible) >= 5, step

        assert len({atom.original_index for atom in visible}) == 5, step


def test_an_origin_starts_at_nothing():

    from qvista.visualization.settings import Origin, ViewerSettings

    assert Origin().is_zero
    assert ViewerSettings().origin.is_zero


def test_a_shift_reaches_the_polyhedron_neighbour_pool():
    """
    The padded pool has to be shifted with everything else, or the
    polyhedra are built from atoms that are no longer where the drawn
    ones are.
    """

    from qvista.visualization.periodic_image_generator import (
        PeriodicImageGenerator,
    )
    from qvista.visualization.settings import Boundary, Origin

    generator = PeriodicImageGenerator()

    plain = generator.generate_padded(slab_structure(), Boundary())

    shifted = generator.generate_padded(
        slab_structure(), Boundary(), origin=Origin(z=0.2)
    )

    assert fractional_z(plain) != fractional_z(shifted)

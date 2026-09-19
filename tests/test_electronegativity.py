"""
Tests for the electronegativity tool.

Two things here are easy to get subtly wrong and impossible to notice from
the number alone:

* **Mixing scales.** Pauling has no value for argon, so a pair containing
  it must move *both* atoms onto the Mulliken scale. Subtracting a
  Mulliken eV from a Pauling unit would produce a plausible-looking
  difference that means nothing.
* **Stoichiometry in the cell mean.** Sanderson's equalization is a
  geometric mean over every atom, not over the distinct elements, so
  Li2O and LiO2 must not come out the same.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from qvista.models.visible_atom import VisibleAtom
from qvista.services.electronegativity import (
    IONIC_MIN,
    NONPOLAR_MAX,
    Scale,
    cell_electronegativity,
    chi,
    classify,
    common_scale,
    compare,
    ionic_character,
    mulliken,
    pauling,
    polyhedron,
    symmetric_limits,
)
from qvista.visualization.selection import (
    AtomSelection,
    atom_key,
    nearest_atom,
)


# ------------------------------------------------------------
# The tables
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,value",
    [("Li", 0.98), ("O", 3.44), ("F", 3.98), ("Cs", 0.79), ("Mn", 1.55)],
)
def test_pauling_values(symbol: str, value: float) -> None:
    """Spot-check against the published scale."""

    assert pauling(symbol) == pytest.approx(value)


@pytest.mark.parametrize("symbol", ["He", "Ne", "Ar"])
def test_noble_gases_have_no_pauling_value(symbol: str) -> None:
    """
    pymatgen returns NaN rather than raising for these.

    A NaN that reaches the arithmetic propagates silently to every
    derived number, so it has to be caught at the lookup.
    """

    assert pauling(symbol) is None


@pytest.mark.parametrize("symbol", ["He", "Ne", "Ar", "Li", "O"])
def test_mulliken_covers_the_gaps(symbol: str) -> None:
    """The fallback scale is defined where Pauling is not."""

    value = mulliken(symbol)

    assert value is not None
    assert not math.isnan(value)


def test_unknown_symbol_is_not_an_error() -> None:
    """A junk symbol reports as undefined rather than raising."""

    assert pauling("Xx") is None
    assert mulliken("Xx") is None


# ------------------------------------------------------------
# Scale selection
# ------------------------------------------------------------


def test_pauling_is_preferred() -> None:
    """Where it covers everything, Pauling wins."""

    assert common_scale(["Li", "O"]) is Scale.PAULING


def test_one_gap_moves_everything_to_mulliken() -> None:
    """Both atoms move, not just the one that was missing."""

    assert common_scale(["Be", "Ar"]) is Scale.MULLIKEN


def test_no_scale_covers_the_superheavies() -> None:
    """Neither table has oganesson; nothing is invented for it."""

    assert common_scale(["Og", "O"]) is None


def test_a_mixed_pair_is_never_mixed_in_the_numbers(
) -> None:
    """
    Be against Ar must report Be's *Mulliken* value.

    The failure this guards against looks entirely reasonable on screen:
    Be 1.57 against Ar 7.37, difference 5.80, which is two different
    scales subtracted from each other.
    """

    result = compare("Be", "Ar")

    assert result.scale is Scale.MULLIKEN
    assert result.chi_a == pytest.approx(mulliken("Be"))
    assert result.chi_b == pytest.approx(mulliken("Ar"))
    assert result.chi_a != pytest.approx(pauling("Be"))


def test_undefined_pair_reports_nothing(
) -> None:
    """No scale, no numbers."""

    result = compare("Og", "O")

    assert not result.defined
    assert result.delta is None
    assert result.ionic_percent is None


# ------------------------------------------------------------
# Pairs
# ------------------------------------------------------------


def test_sodium_chloride_is_about_seventy_percent_ionic() -> None:
    """The textbook worked example."""

    result = compare("Na", "Cl")

    assert result.delta == pytest.approx(2.23, abs=0.01)
    assert result.ionic_percent == pytest.approx(71.2, abs=0.5)
    assert result.character == "ionic"


def test_carbon_hydrogen_is_nonpolar() -> None:
    """Δχ 0.35, below the 0.4 convention."""

    result = compare("C", "H")

    assert result.delta == pytest.approx(0.35, abs=0.01)
    assert result.character == "nonpolar covalent"


def test_delta_is_symmetric_but_direction_is_not() -> None:
    """
    Swapping the click order must not change the physics.

    The magnitude is symmetric; which atom pulls is the half of the
    answer that would be lost by taking the absolute value alone.
    """

    forward = compare("Li", "O")
    backward = compare("O", "Li")

    assert forward.delta == pytest.approx(backward.delta)
    assert forward.more_electronegative == "O"
    assert backward.more_electronegative == "O"


def test_identical_elements_have_no_polarity() -> None:
    """An O-O contact is nonpolar and has no direction."""

    result = compare("O", "O")

    assert result.delta == pytest.approx(0.0)
    assert result.more_electronegative is None
    assert result.character == "nonpolar covalent"


def test_ionic_character_is_withheld_on_the_mulliken_scale() -> None:
    """
    Pauling's relation is fitted to Pauling differences.

    Feeding it an eV difference returns a number, which is exactly why it
    has to be suppressed rather than shown.
    """

    result = compare("Be", "Ar")

    assert result.delta is not None
    assert result.ionic_percent is None
    assert result.character is None


@pytest.mark.parametrize(
    "delta,expected",
    [
        (0.0, "nonpolar covalent"),
        (NONPOLAR_MAX - 0.01, "nonpolar covalent"),
        (NONPOLAR_MAX, "polar covalent"),
        (IONIC_MIN - 0.01, "polar covalent"),
        (IONIC_MIN, "ionic"),
        (3.0, "ionic"),
    ],
)
def test_classification_boundaries(delta: float, expected: str) -> None:
    """The conventional 0.4 and 1.7 cuts, inclusive at the lower edge."""

    assert classify(delta) == expected


def test_ionic_character_spans_the_range() -> None:
    """Zero difference is covalent; a large one approaches fully ionic."""

    assert ionic_character(0.0) == pytest.approx(0.0)
    assert ionic_character(1.7) == pytest.approx(0.514, abs=0.005)
    assert ionic_character(4.0) > 0.98


# ------------------------------------------------------------
# The cell
# ------------------------------------------------------------


def test_cell_is_the_geometric_mean() -> None:
    """Sanderson equalization for NaCl is √(χ_Na · χ_Cl)."""

    result = cell_electronegativity(["Na", "Cl"])

    assert result.equalized == pytest.approx(
        math.sqrt(0.93 * 3.16), abs=1e-6
    )
    assert result.scale is Scale.PAULING


def test_cell_counts_every_atom_not_every_element() -> None:
    """
    Stoichiometry has to move the mean.

    Averaging the distinct elements instead would give Li2O and LiO2 the
    same electronegativity, which is the whole point of the quantity.
    """

    lithium_rich = cell_electronegativity(["Li", "Li", "O"])
    oxygen_rich = cell_electronegativity(["Li", "O", "O"])

    assert lithium_rich.equalized < oxygen_rich.equalized


def test_cell_deviation_signs_the_direction() -> None:
    """Oxygen sits above the mean, lithium below it."""

    result = cell_electronegativity(["Li"] * 2 + ["Mn"] * 2 + ["O"] * 4)

    assert result.deviation("O") > 0
    assert result.deviation("Li") < 0


def test_cell_falls_back_for_a_noble_gas() -> None:
    """A cell containing argon is reported on the Mulliken scale."""

    result = cell_electronegativity(["Be", "Ar"])

    assert result.defined
    assert result.scale is Scale.MULLIKEN


def test_empty_cell_is_undefined() -> None:
    """No atoms, no mean -- and no ZeroDivisionError."""

    result = cell_electronegativity([])

    assert not result.defined
    assert result.deviation("O") is None


def test_cell_with_an_uncovered_element_is_undefined() -> None:
    """One superheavy is enough to leave no usable scale."""

    result = cell_electronegativity(["Og", "O"])

    assert not result.defined


# ------------------------------------------------------------
# The coordination shell
# ------------------------------------------------------------

#: The six directions of a regular octahedron.
OCTAHEDRON = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def shell(symbol: str, directions, distance: float = 1.95):
    """A coordination shell of one element in the given directions."""

    return [(symbol, distance, direction) for direction in directions]


def test_regular_octahedron_cancels() -> None:
    """
    Six equal pulls in opposite directions sum to nothing.

    This is the whole reason for taking a vector sum: every bond is
    strongly polar, and the site as a whole is not.
    """

    result = polyhedron("Mn", shell("O", OCTAHEDRON))

    assert result.coordination == 6
    assert result.composition == "O6"
    assert result.resultant == pytest.approx(0.0, abs=1e-9)
    assert result.asymmetry == pytest.approx(0.0, abs=1e-9)
    assert "balanced" in result.balance


def test_one_sided_shell_does_not_cancel() -> None:
    """The same composition, all on one side, is maximally lopsided."""

    result = polyhedron("Mn", shell("O", [(1, 0, 0)] * 6))

    assert result.composition == "O6"
    assert result.asymmetry == pytest.approx(1.0)
    assert "one-sided" in result.balance


def test_mean_is_over_the_ligands_not_the_whole_shell() -> None:
    """Six oxygens around manganese average to oxygen."""

    result = polyhedron("Mn", shell("O", OCTAHEDRON))

    assert result.mean_ligand_chi == pytest.approx(3.44)
    assert result.net_delta == pytest.approx(3.44 - 1.55)


def test_net_delta_signs_the_direction() -> None:
    """
    An oxygen surrounded by cations is on the receiving end.

    The sign is what says whether the site gains or loses density, and
    an absolute value would throw it away.
    """

    centre = polyhedron("O", shell("Mn", OCTAHEDRON))

    assert centre.net_delta < 0

    assert polyhedron("Mn", shell("O", OCTAHEDRON)).net_delta > 0


def test_directions_need_not_be_normalised() -> None:
    """
    Callers hand over raw bond vectors.

    Leaving them unnormalised would weight long bonds more heavily, by
    accident rather than by design.
    """

    long_bonds = [("O", 4.0, (4, 0, 0)), ("O", 1.0, (-1, 0, 0))]

    assert polyhedron("Mn", long_bonds).resultant == pytest.approx(
        0.0, abs=1e-9
    )


def test_ligands_come_back_nearest_first() -> None:
    """The panel lists them in that order."""

    mixed = [
        ("O", 2.3, (1, 0, 0)),
        ("O", 1.9, (0, 1, 0)),
        ("Li", 2.8, (0, 0, 1)),
    ]

    distances = [
        ligand.distance for ligand in polyhedron("Mn", mixed).ligands
    ]

    assert distances == sorted(distances)


def test_composition_counts_each_element() -> None:
    """Mixed shells report a formula, commonest element first."""

    mixed = shell("O", OCTAHEDRON) + shell("Li", [(1, 1, 0), (-1, -1, 0)])

    assert polyhedron("Mn", mixed).composition == "O6 Li2"


def test_attracting_drops_the_cation_contacts() -> None:
    """
    Metal-metal contacts are bonded but are not the polyhedron.

    The drawing rule joins Mn to Mn at under 3 Å in a layered oxide.
    Those contacts contribute nothing to the resultant and pull the mean
    down towards manganese's own value, so the coordination polyhedron
    proper has to be recoverable.
    """

    full = polyhedron(
        "Mn",
        shell("O", OCTAHEDRON) + shell("Mn", [(1, 1, 0), (-1, -1, 0)], 2.9),
    )

    assert full.coordination == 8
    assert full.mean_ligand_chi < 3.44

    only_ligands = full.attracting()

    assert only_ligands.coordination == 6
    assert only_ligands.composition == "O6"
    assert only_ligands.mean_ligand_chi == pytest.approx(3.44)


def test_empty_shell_is_undefined() -> None:
    """An isolated atom has no environment to report."""

    result = polyhedron("Mn", [])

    assert result.coordination == 0
    assert not result.defined
    assert result.net_delta is None
    assert result.balance is None


def test_shell_falls_back_to_mulliken() -> None:
    """One argon neighbour moves the whole shell onto Mulliken."""

    result = polyhedron("Be", [("Ar", 2.0, (1, 0, 0))])

    assert result.scale is Scale.MULLIKEN
    assert result.defined


def test_shell_with_an_uncovered_element_is_undefined() -> None:
    """No scale covers oganesson, so nothing is reported."""

    assert not polyhedron("Mn", [("Og", 2.0, (1, 0, 0))]).defined


def test_same_element_shell_has_no_pull() -> None:
    """
    A metal surrounded by its own kind faces nothing.

    Every Δχ is zero, so there is no resultant and no ratio to report --
    and the ratio must not be a division by zero.
    """

    result = polyhedron("Mn", shell("Mn", OCTAHEDRON))

    assert result.net_delta == pytest.approx(0.0)
    assert result.scalar_total == pytest.approx(0.0)
    assert result.asymmetry is None
    assert result.balance is None


# ------------------------------------------------------------
# Colour limits for the map
# ------------------------------------------------------------


def test_limits_are_centred_on_zero() -> None:
    """
    The midpoint of the colour scale has to mean "no shift".

    Running the scale from the smallest value to the largest would put
    zero wherever it happened to fall, so the same colour would mean
    "gives up density" in one structure and "neutral" in the next.
    """

    low, high = symmetric_limits([-1.08, -0.51, 1.38])

    assert low == pytest.approx(-1.38)
    assert high == pytest.approx(1.38)
    assert low + high == pytest.approx(0.0)


def test_limits_survive_a_flat_structure() -> None:
    """
    An elemental structure has no shift anywhere.

    Zero-width limits would make the colour bar degenerate, so a floor
    is applied.
    """

    low, high = symmetric_limits([0.0, 0.0, 0.0])

    assert low < high


def test_limits_ignore_undefined_atoms() -> None:
    """Atoms drawn grey must not drag the scale."""

    assert symmetric_limits([None, 2.0, None]) == pytest.approx((-2.0, 2.0))


def test_limits_of_nothing_do_not_raise() -> None:
    """A structure where no atom has a value still needs a range."""

    low, high = symmetric_limits([])

    assert low < 0 < high


# ------------------------------------------------------------
# Selection
# ------------------------------------------------------------


def make_atom(symbol: str, position, index: int = 0, image=(0, 0, 0)):
    """A visible atom at a given place."""

    return VisibleAtom(
        symbol=symbol,
        position=np.asarray(position, dtype=float),
        original_index=index,
        tx=image[0],
        ty=image[1],
        tz=image[2],
    )


def test_periodic_images_are_distinct_atoms() -> None:
    """
    Two images of one site are separate selections.

    They are different points on screen, so clicking one must not read as
    clicking the other -- but they share a site index, which is why the
    key carries the translation too.
    """

    home = make_atom("Li", (0, 0, 0), index=0)
    image = make_atom("Li", (4, 0, 0), index=0, image=(1, 0, 0))

    assert atom_key(home) != atom_key(image)

    selection = AtomSelection()
    selection.pick(home)
    selection.pick(image)

    assert len(selection) == 2


def test_nearest_atom_resolves_a_surface_click() -> None:
    """
    A click lands on the sphere's surface, not its centre.

    The point handed back by the picker is one radius away from the atom
    it belongs to, so the lookup has to be nearest-centre.
    """

    atoms = [
        make_atom("Li", (0, 0, 0), index=0),
        make_atom("O", (3, 0, 0), index=1),
    ]

    # A point on the near face of the oxygen.
    assert nearest_atom(atoms, (2.3, 0, 0)) is atoms[1]


def test_nearest_atom_honours_the_element() -> None:
    """
    Restricting to the mesh that was hit disambiguates overlap.

    Without it, a click on an oxygen that slightly overlaps a nearer
    lithium centre would select the lithium.
    """

    atoms = [
        make_atom("Li", (0, 0, 0), index=0),
        make_atom("O", (1.2, 0, 0), index=1),
    ]

    assert nearest_atom(atoms, (0.5, 0, 0), symbol="O") is atoms[1]
    assert nearest_atom(atoms, (0.9, 0, 0), symbol="Li") is atoms[0]


def test_nearest_atom_rejects_a_distant_click() -> None:
    """A pick that landed nowhere near an atom selects nothing."""

    atoms = [make_atom("Li", (0, 0, 0))]

    assert nearest_atom(atoms, (50, 0, 0), max_distance=2.0) is None


def test_nearest_atom_on_an_empty_scene() -> None:
    """Nothing drawn, nothing picked."""

    assert nearest_atom([], (0, 0, 0)) is None


def test_selection_holds_two_and_rolls() -> None:
    """
    A third click drops the oldest rather than clearing.

    Walking along a chain of bonds then keeps showing the last two atoms
    clicked, which is the way the panel is meant to be used.
    """

    selection = AtomSelection()

    first = make_atom("Li", (0, 0, 0), index=0)
    second = make_atom("O", (2, 0, 0), index=1)
    third = make_atom("Mn", (4, 0, 0), index=2)

    selection.pick(first)
    selection.pick(second)
    selection.pick(third)

    assert [atom.symbol for atom in selection.atoms] == ["O", "Mn"]


def test_clicking_a_selected_atom_deselects_it() -> None:
    """The standard toggle affordance."""

    selection = AtomSelection()

    atom = make_atom("Li", (0, 0, 0))

    selection.pick(atom)
    selection.pick(atom)

    assert len(selection) == 0


def test_resync_drops_atoms_that_left_the_view() -> None:
    """
    Shrinking the display boundary removes images from the scene.

    A selection still holding one would report a bond length to an atom
    that is no longer there.
    """

    home = make_atom("Li", (0, 0, 0), index=0)
    image = make_atom("Li", (4, 0, 0), index=0, image=(1, 0, 0))

    selection = AtomSelection()
    selection.pick(home)
    selection.pick(image)

    changed = selection.resync([home])

    assert changed
    assert [atom_key(a) for a in selection.atoms] == [atom_key(home)]


def test_resync_reports_no_change_when_everything_survives() -> None:
    """A redraw that keeps the atoms must not churn the panel."""

    atom = make_atom("Li", (0, 0, 0))

    selection = AtomSelection()
    selection.pick(atom)

    # A rebuilt scene holds equal-but-not-identical atom objects.
    rebuilt = make_atom("Li", (0, 0, 0))

    assert not selection.resync([rebuilt])
    assert len(selection) == 1


def test_resync_rebinds_to_the_new_objects() -> None:
    """
    The selection must point at the *new* atoms after a redraw.

    Holding the old objects would keep reporting stale positions once a
    setting moved them.
    """

    atom = make_atom("Li", (0, 0, 0))

    selection = AtomSelection()
    selection.pick(atom)

    moved = make_atom("Li", (1, 1, 1))

    selection.resync([moved])

    assert selection.atoms[0] is moved


# ------------------------------------------------------------
# The two menu entries and the one panel
# ------------------------------------------------------------

# Select Atoms and Polyhedral Environment are separate menu entries onto
# the same window: one compares the pair you clicked, the other sums over
# the whole coordination shell of one atom. The entry is the only place
# the choice is stated, so it has to arrive at the panel.
#
# It did not. MainWindow.electronegativity_panel took a ``polyhedral``
# argument and never passed it on, so Polyhedral Environment built the
# panel with the box unchecked -- identical to Select Atoms, and looking
# like the menu entry had done nothing. These are pure: the real window
# needs an OpenGL viewport, so the shipped methods are bound onto a stub
# and the dialog class is replaced by a recorder, which checks the part
# that is ours -- what gets asked for, and of what.


class RecordingPanel:
    """Stands in for ElectronegativityDialog and remembers the call."""

    def __init__(self, viewer, parent=None, polyhedral: bool = False) -> None:

        self.viewer = viewer
        self.polyhedral = polyhedral
        self.visible = False
        self.raised = 0

    # -- the panel's side of the menu action

    def set_polyhedral(self, polyhedral: bool) -> None:
        self.polyhedral = bool(polyhedral)

    # -- enough QDialog to be opened and re-opened

    def show(self) -> None:
        self.visible = True

    def isVisible(self) -> bool:  # noqa: N802 - Qt's name
        return self.visible

    def raise_(self) -> None:
        self.raised += 1

    def activateWindow(self) -> None:  # noqa: N802 - Qt's name
        pass


@pytest.fixture
def window(monkeypatch):
    """A stand-in main window with the two menu actions bound onto it."""

    from qvista.gui import main_window as module

    monkeypatch.setattr(module, "ElectronegativityDialog", RecordingPanel)

    class FakeWindow:

        electronegativity_panel = module.MainWindow.electronegativity_panel
        electronegativity_polyhedral = (
            module.MainWindow.electronegativity_polyhedral
        )

        def __init__(self) -> None:
            # Only ``structure is None`` is read, and it must not be.
            self.workspace = SimpleNamespace(
                viewer=SimpleNamespace(structure=object())
            )

    return FakeWindow()


def test_select_atoms_opens_on_the_pair_view(window) -> None:
    """The default, and the common case."""

    window.electronegativity_panel()

    assert window._electronegativity_dialog.polyhedral is False


def test_polyhedral_environment_opens_on_the_shell_view(window) -> None:
    """The menu entry has to reach the checkbox it names."""

    window.electronegativity_polyhedral()

    assert window._electronegativity_dialog.polyhedral is True


def test_polyhedral_environment_switches_a_panel_already_open(
    window,
) -> None:
    """
    Raising a window that is already up is invisible.

    With the panel open on the pair view, the entry has nothing to show
    for itself unless it also changes the view -- which is what made it
    look like the action did nothing.
    """

    window.electronegativity_panel()

    panel = window._electronegativity_dialog

    window.electronegativity_polyhedral()

    assert panel.polyhedral is True

    # And it is still the one panel: a second would arm picking twice.
    assert window._electronegativity_dialog is panel


def test_select_atoms_switches_back(window) -> None:
    """The other direction, for the same reason."""

    window.electronegativity_polyhedral()

    panel = window._electronegativity_dialog

    window.electronegativity_panel()

    assert panel.polyhedral is False
    assert window._electronegativity_dialog is panel

"""
Phonon post-processing: thermal properties, and the menu around them.

The thermal panels were computed on the VASP route and drawn nowhere,
and not computed at all on the machine-learning one. These pin the
wiring rather than the physics, which lives in
:mod:`qvista.services.phonon_dos`.
"""

from __future__ import annotations

import pytest

from qvista.analyses.phonon import (
    MODE_BAND,
    MODE_BOTH,
    MODE_DOS,
    MODE_PROJECTED,
    MODE_THERMAL,
    PhononResult,
)
from qvista.gui.menu_bar import Action, Menu, MENU_STRUCTURE, Separator


def entries(items=MENU_STRUCTURE, path=""):
    """Every leaf in the menu, as ``(path, slot)``."""

    for item in items:

        if isinstance(item, Menu):
            yield from entries(item.items, f"{path}/{item.label}")

        elif isinstance(item, Action):
            yield f"{path}/{item.label}", item.slot


def phonon_entries() -> dict[str, str]:

    return {
        path: slot
        for path, slot in entries()
        if "Phonon Analysis" in path
    }


# ----------------------------------------------------------------
# The mode
# ----------------------------------------------------------------


def test_thermal_is_a_mode_of_its_own():

    assert MODE_THERMAL not in {
        MODE_BAND,
        MODE_DOS,
        MODE_BOTH,
        MODE_PROJECTED,
    }


def test_a_result_can_carry_thermal_properties():

    result = PhononResult(mode=MODE_THERMAL, thermal={"any": "object"})

    assert result.thermal == {"any": "object"}


def test_thermal_travels_in_the_stored_result():
    """
    Cached on the project like the band and the DOS, so asking for the
    figure twice does not integrate the mesh twice.
    """

    result = PhononResult(mode=MODE_THERMAL, thermal=object())

    assert "thermal" in result.as_storage()


def test_a_result_without_thermal_defaults_to_none():

    assert PhononResult().thermal is None


# ----------------------------------------------------------------
# The menu
# ----------------------------------------------------------------


def test_both_branches_offer_thermal_properties():

    found = phonon_entries()

    assert "/Analysis/Phonon Analysis/CHGNet/Thermal Properties" in found
    assert (
        "/Analysis/Phonon Analysis/VASP/Post-Processing/"
        "Thermal Properties" in found
    )


def test_the_standalone_mode_visualiser_is_gone():
    """
    Both branches already end in Band Visualization, which animates the
    force constants that branch just produced. A third entry animating
    "whichever were computed last" gives no way to tell which set is
    loaded.
    """

    assert not any("Visualize Modes" in path for path, _ in entries())


def test_no_menu_entry_names_a_slot_that_does_not_exist():
    """
    A missing slot falls back to a placeholder rather than failing, so
    a typo here is invisible until someone clicks it.
    """

    pytest.importorskip("PySide6.QtWidgets")

    from qvista.gui.main_window import MainWindow

    missing = [
        f"{path} -> {slot}"
        for path, slot in entries()
        if not hasattr(MainWindow, slot)
    ]

    assert not missing


@pytest.mark.parametrize(
    "slot",
    ["phonon_thermal", "vasp_phonon_thermal"],
)
def test_each_thermal_entry_asks_for_the_thermal_mode(slot):

    pytest.importorskip("PySide6.QtWidgets")

    import inspect

    from qvista.gui.main_window import MainWindow

    source = inspect.getsource(getattr(MainWindow, slot))

    assert "MODE_THERMAL" in source


# ----------------------------------------------------------------
# Tiling during an animation
# ----------------------------------------------------------------


def slab_structure():

    import numpy as np
    from ase import Atoms

    from qvista.models.structure import Structure

    return Structure(
        atoms=Atoms(
            "H4",
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            cell=np.diag([4.0, 4.0, 4.0]),
            pbc=True,
        )
    )


def test_tiling_is_on_by_default():

    from qvista.visualization.periodic_image_generator import (
        PeriodicImageGenerator,
    )
    from qvista.visualization.settings import Boundary

    visible = PeriodicImageGenerator().generate(
        slab_structure(), Boundary()
    )

    # Four atoms, plus the face duplicates the boundary brings in.
    assert len(visible) > 4


def test_tiling_can_be_switched_off_for_an_animation():
    """
    A mode animation builds its own repeated cell carrying the
    exp(i q.R) phase. Tiling that again gives face duplicates that are
    the home atom's displaced position plus a lattice vector, so they
    move in phase with the home cell rather than their own -- and at the
    zone boundary, where neighbouring cells are in antiphase, that is
    exactly backwards.
    """

    from qvista.visualization.periodic_image_generator import (
        PeriodicImageGenerator,
    )
    from qvista.visualization.settings import Boundary

    visible = PeriodicImageGenerator().generate(
        slab_structure(), Boundary(), tile=False
    )

    assert len(visible) == 4


def test_untiled_atoms_stay_where_they_actually_are():

    import numpy as np

    from qvista.visualization.periodic_image_generator import (
        PeriodicImageGenerator,
    )
    from qvista.visualization.settings import Boundary

    structure = slab_structure()

    visible = PeriodicImageGenerator().generate(
        structure, Boundary(), tile=False
    )

    drawn = np.array([atom.position for atom in visible])

    assert np.allclose(drawn, structure.atoms.positions)


def test_the_mode_explorer_switches_tiling_off():

    pytest.importorskip("PySide6.QtWidgets")

    import inspect

    from qvista.gui.dialogs import mode_explorer

    source = inspect.getsource(mode_explorer.ModeExplorer)

    assert "tile_images = False" in source


# ----------------------------------------------------------------
# What the first panel actually is
# ----------------------------------------------------------------


class FakeThermal:
    """Just enough of a ThermalResult to draw."""

    def __init__(self) -> None:

        import numpy as np

        self.temperatures = np.linspace(0.0, 1000.0, 51)
        self.free_energy = 29.0 - 0.09 * self.temperatures
        self.entropy = 0.17 * self.temperatures
        self.heat_capacity = 99.8 * (
            1.0 - np.exp(-self.temperatures / 180.0)
        )
        self.dulong_petit = 99.8
        self.zero_point = 29.0


def thermal_figure():

    pytest.importorskip("matplotlib")

    from matplotlib.figure import Figure

    from qvista.visualization import phonon_plots

    figure = Figure()

    phonon_plots.draw_thermal(figure, FakeThermal())

    return figure


def test_the_first_panel_is_named_as_vibrational_only():
    """
    phonopy returns the Helmholtz *vibrational* free energy. The static
    electronic energy is absent -- no total-energy calculation happens
    here -- so calling it plain "free energy" invites it being quoted on
    its own, which is wrong by tens of eV per cell and wrong in a
    direction that looks plausible. Naming it F_vib is the whole fix.
    """

    label = thermal_figure().axes[0].get_ylabel()

    assert "vib" in label
    assert "kJ/mol" in label


def test_the_panel_carries_the_name_and_nothing_else():
    """
    The subscript says what it is. Spelling the caveat out on the axis
    as well made a figure that reads as an argument rather than a plot.
    """

    figure = thermal_figure()

    # matplotlib hands back an ArtistList, not a list, so this counts.
    assert len(figure.axes[0].texts) == 0

    assert "excludes" not in figure.axes[0].get_ylabel()


def test_the_other_two_panels_are_unchanged():

    figure = thermal_figure()

    assert "Entropy" in figure.axes[1].get_ylabel()
    assert "C" in figure.axes[2].get_ylabel()


def test_the_dulong_petit_guide_is_still_drawn():

    texts = [t.get_text() for t in thermal_figure().axes[2].texts]

    assert any("3NR" in text for text in texts)


# ----------------------------------------------------------------
# The path the dispersion is drawn along
# ----------------------------------------------------------------


def test_a_continuous_path_reads_as_one_run():

    pytest.importorskip("PySide6.QtWidgets")

    from qvista.gui.main_window import _path_label

    assert _path_label([("GAMMA", "X"), ("X", "M")]) == "GAMMA-X-M"


def test_a_break_in_the_path_is_marked():
    """
    The same vertical bar the figure's own axis uses, so the log line
    and the plot agree about where the path jumps.
    """

    pytest.importorskip("PySide6.QtWidgets")

    from qvista.gui.main_window import _path_label

    label = _path_label(
        [("GAMMA", "T"), ("T", "H2"), ("H0", "L"), ("L", "GAMMA")]
    )

    assert label == "GAMMA-T-H2|H0-L-GAMMA"


def test_an_empty_path_says_so_rather_than_being_blank():

    pytest.importorskip("PySide6.QtWidgets")

    from qvista.gui.main_window import _path_label

    assert _path_label([]) == "(empty)"


def test_no_band_path_entry_sits_in_the_menu():
    """
    A phonon path is a property of the figure, not of the structure, so
    it is chosen on the dispersion itself. The Crystallography path
    builder is a different thing: it describes the structure, draws the
    Brillouin zone and writes KPOINTS.
    """

    assert not any("Band Path" in path for path, _ in entries())


def test_the_chooser_is_offered_on_the_dispersion_itself():

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.widgets.band_controls import BandControls

    QApplication.instance() or QApplication([])

    controls = BandControls()

    assert "Custom Path" in controls.path_button.text()


def test_pressing_it_asks_the_window_rather_than_acting_itself():
    """
    The widget does not hold the force constants. It says it was asked,
    and whatever owns the figure re-runs the band.
    """

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.widgets.band_controls import BandControls

    QApplication.instance() or QApplication([])

    controls = BandControls()

    asked: list[int] = []

    controls.path_requested.connect(lambda: asked.append(1))

    controls.path_button.click()

    assert asked == [1]


# ----------------------------------------------------------------
# Tiling during the animation that is actually used
# ----------------------------------------------------------------


def test_the_band_visualisation_switches_tiling_off():
    """
    Band Visualization drives the main viewer through
    ``ModeAnimationDialog``, which animates a single primitive cell with
    no exp(i q.R) phase in it. Every tiled image therefore carries the
    home cell's displacement exactly, and away from Gamma that is wrong.
    """

    pytest.importorskip("PySide6.QtWidgets")

    import inspect

    from qvista.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._animate_modes)

    assert "tile_images = False" in source


def test_tiling_comes_back_when_the_animation_stops():

    pytest.importorskip("PySide6.QtWidgets")

    import inspect

    from qvista.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._restore_structure)

    assert "tile_images = True" in source


def test_the_projected_figure_can_change_its_path_too():
    """
    It colours by element and legends itself, so its branch controls do
    nothing -- but which points the dispersion runs through is a
    question it has exactly as much as the plain band.
    """

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.widgets.band_controls import BandControls

    QApplication.instance() or QApplication([])

    controls = BandControls()
    controls.show()

    controls.set_colouring_offered(False)

    assert controls.path_button.isVisible()
    assert not controls.scheme.isVisible()
    assert not controls.colour_button.isVisible()
    assert not controls.legend.isVisible()


def test_the_branch_controls_come_back_for_a_plain_band():

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.widgets.band_controls import BandControls

    QApplication.instance() or QApplication([])

    controls = BandControls()
    controls.show()

    controls.set_colouring_offered(False)
    controls.set_colouring_offered(True)

    assert controls.scheme.isVisible()
    assert controls.path_button.isVisible()

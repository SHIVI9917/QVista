"""
Tests for the band-centre window: what a row is, and how it is drawn.

A band centre used to be one thing -- the first moment per species --
and the window offered no say in it. It now resolves the same three ways
the DOS and band windows do, because which selection the moment is taken
over is the whole question. The d-band centre of a slab is not the
d-band centre of the atom at the top of it, and both are worth asking
for.

The figure is checked for orientation as much as for content. Energy has
to run *up*: it is the axis every other energy figure in QVista uses, a
centre above the Fermi level has to be drawn above the line, and a band
that moves up in energy has to move up on the page. Laid on its side --
which is how this used to be drawn -- "higher" meant nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from qvista.services import electronic as el
from qvista.visualization import electronic_plots as ep


# ------------------------------------------------------------
# A run with a known answer
# ------------------------------------------------------------

#: Where each species' states are put, in eV relative to E_F.
NICKEL_D = -2.0
OXYGEN_P = -5.0


class _Vasprun:
    """Only the attribute :func:`el.band_centre` reaches for."""

    def __init__(self, complete_dos) -> None:
        self.complete_dos = complete_dos


def two_species_run() -> el.Run:
    """
    One Ni with d states at -2 eV, one O with p states at -5 eV.

    Nothing anywhere else, so a shell with no states is a real case
    rather than a rounding accident: Ni has no p, O has no d, and
    neither has s.
    """

    from pymatgen.core import Lattice, Structure
    from pymatgen.electronic_structure.core import Orbital, Spin
    from pymatgen.electronic_structure.dos import CompleteDos, Dos

    energies = np.linspace(-15.0, 15.0, 3001)

    def peak(at: float) -> np.ndarray:
        return np.exp(-0.5 * ((energies - at) / 0.4) ** 2)

    nickel = peak(NICKEL_D)
    oxygen = peak(OXYGEN_P)

    structure = Structure(
        Lattice.cubic(4.0),
        ["Ni", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    empty = np.zeros_like(energies)

    pdos = {
        structure[0]: {
            orbital: {
                Spin.up: nickel if orbital.name.startswith("d") else empty
            }
            for orbital in Orbital
        },
        structure[1]: {
            orbital: {
                Spin.up: oxygen
                if orbital.name.startswith("p")
                else empty
            }
            for orbital in Orbital
        },
    }

    total = Dos(0.0, energies, {Spin.up: nickel + oxygen})

    return el.Run(
        vasprun=_Vasprun(CompleteDos(structure, total, pdos)),
        directory=Path("."),
        fermi=0.0,
        natoms=2,
        symbols=("Ni", "O"),
    )


@pytest.fixture(scope="module")
def run() -> el.Run:
    return two_species_run()


WINDOW = (-15.0, 15.0)


# ------------------------------------------------------------
# The three ways a row can be resolved
# ------------------------------------------------------------


def test_all_three_projections_are_offered() -> None:
    """The same words the DOS and band windows use."""

    assert set(el.CENTRE_MODES) == {"atom", "element", "orbital"}


def test_orbital_wise_is_one_row_for_the_whole_cell(run) -> None:
    """
    The figure to start from: which shell carries the states.

    Per-element and per-atom breakdowns are only worth reading once
    that is known, so the window opens on this one.
    """

    rows = el.band_centre(
        run, band="d", erange=WINDOW, projection="orbital"
    )

    assert len(rows) == 1

    assert rows[0].label == el.WHOLE_CELL

    # Every atom in the cell, not the zero an element lookup would give
    # for a label that is not a species.
    assert rows[0].count == run.natoms

    assert rows[0].centre == pytest.approx(NICKEL_D, abs=0.05)


def test_element_wise_is_one_row_per_species(run) -> None:

    rows = el.band_centre(
        run, band="d", erange=WINDOW, projection="element"
    )

    assert [row.label for row in rows] == ["Ni", "O"]

    assert [row.count for row in rows] == [1, 1]


def test_element_wise_can_be_narrowed_to_a_chosen_species(run) -> None:
    """The tick boxes, which is why the elements argument exists."""

    rows = el.band_centre(
        run,
        band="d",
        elements=["Ni"],
        erange=WINDOW,
        projection="element",
    )

    assert [row.label for row in rows] == ["Ni"]


def test_atom_wise_is_one_row_per_named_atom(run) -> None:
    """Numbered from one, as every other table in QVista is."""

    rows = el.band_centre(
        run, band="d", atoms="1", erange=WINDOW, projection="atom"
    )

    assert [row.label for row in rows] == ["1 Ni"]

    assert rows[0].count == 1

    assert rows[0].centre == pytest.approx(NICKEL_D, abs=0.05)


def test_atom_wise_with_nothing_named_says_so(run) -> None:

    with pytest.raises(el.ElectronicError, match="at least one atom"):
        el.band_centre(run, band="d", projection="atom")


def test_an_unknown_projection_is_refused(run) -> None:

    with pytest.raises(el.ElectronicError, match="not a way to resolve"):
        el.band_centre(run, band="d", projection="spin")


def test_the_projection_is_inferred_when_it_is_not_given(run) -> None:
    """
    So every caller that predates the choice still works.

    Atoms named means atom wise; nothing named means species.
    """

    named = el.band_centre(run, band="d", atoms="1", erange=WINDOW)
    unnamed = el.band_centre(run, band="d", erange=WINDOW)

    assert [row.label for row in named] == ["1 Ni"]
    assert [row.label for row in unnamed] == ["Ni", "O"]


# ------------------------------------------------------------
# A shell a selection has no states in
# ------------------------------------------------------------


def test_a_species_with_no_states_in_a_shell_is_undefined(run) -> None:
    """
    Oxygen has no d band. The honest answer is an absence, not a NaN
    dressed up as a number.
    """

    rows = {
        row.label: row
        for row in el.band_centre(
            run, band="d", erange=WINDOW, projection="element"
        )
    }

    assert rows["Ni"].defined
    assert not rows["O"].defined


def test_each_shell_finds_its_own_species(run) -> None:
    """p is oxygen's, d is nickel's, and neither has any s."""

    def centres(band: str) -> dict[str, float]:

        return {
            row.label: row.centre
            for row in el.band_centre(
                run, band=band, erange=WINDOW, projection="element"
            )
            if row.defined
        }

    assert centres("d") == pytest.approx({"Ni": NICKEL_D}, abs=0.05)
    assert centres("p") == pytest.approx({"O": OXYGEN_P}, abs=0.05)

    assert centres("s") == {}


# ------------------------------------------------------------
# The figure
# ------------------------------------------------------------


def curves_for(*specs) -> list:
    """
    ``(element, contribution, centre, width)`` as drawable curves.

    Each gets a Gaussian of states at its centre, so the figure has
    something real to draw the balance point across.
    """

    energies = np.linspace(-8.0, 3.0, 600)

    return [
        el.CentreCurve(
            element=element,
            contribution=contribution,
            atoms=4,
            energies=energies,
            dos=np.exp(-0.5 * ((energies - centre) / max(width, 0.2)) ** 2),
            centre=centre,
            width=width,
        )
        for element, contribution, centre, width in specs
    ]


def drawn(*specs) -> Figure:

    figure = Figure()

    ep.draw_band_centre(figure, curves_for(*specs))

    return figure


def test_energy_runs_up_the_vertical_axis() -> None:
    """
    A centre above the Fermi level has to be above the line on the page,
    which means energy on y and y not inverted.
    """

    figure = drawn(
        ("Ni", "d-band", -2.0, 1.0),
        ("O", "p-band", +1.5, 0.8),
    )

    axes = figure.axes[0]

    assert "E" in axes.get_ylabel()

    bottom, top = axes.get_ylim()

    assert bottom < top, "the energy axis is still upside down"


def test_the_density_runs_across_the_page() -> None:
    """
    The curve is the band and the dashed line through it is its centre.
    Drawn together, the centre is visibly the balance point of the
    states beside it rather than a bare number.
    """

    figure = drawn(("Ni", "d-band", -2.0, 1.0))

    assert "PDOS" in figure.axes[0].get_xlabel()


def test_a_centre_is_marked_across_its_own_curve() -> None:

    figure = drawn(("Ni", "d-band", -2.0, 1.0))

    dashed = [
        line
        for line in figure.axes[0].lines
        if line.get_linestyle() in ("--", (0, (6.4, 1.6)))
    ]

    assert dashed

    assert dashed[0].get_ydata()[0] == pytest.approx(-2.0)


def test_a_higher_centre_is_drawn_higher() -> None:
    """Stated directly, because reading it off the axis is the point."""

    figure = drawn(
        ("low", "d-band", -4.0, 0.5),
        ("high", "d-band", +2.0, 0.5),
    )

    heights = [
        line.get_ydata()[0]
        for line in figure.axes[0].lines
        if line.get_linestyle() not in ("-", "solid")
        and len(line.get_ydata())
    ]

    assert max(heights) == pytest.approx(2.0)
    assert min(heights) == pytest.approx(-4.0)


def test_each_curve_is_labelled_with_its_centre() -> None:
    """
    Element, contribution and the number, beside the line -- which is
    how the reference reports it.
    """

    figure = drawn(("Ni", "d-band", -2.0, 1.0))

    labels = [
        text.get_text()
        for text in figure.axes[0].texts
        if "d-band" in text.get_text()
    ]

    assert labels

    # The name only. The number is in the table beside the figure, and
    # repeating it on every curve crowded the panel.
    assert labels[0] == "Ni d-band"


def test_the_fermi_level_is_marked_and_named() -> None:

    figure = drawn(("Ni", "d-band", -2.0, 1.0))

    assert any(
        "E_F" in text.get_text() for text in figure.axes[0].texts
    )


def test_every_curve_is_normalised_to_the_same_height() -> None:
    """
    The absolute height of a projected DOS depends on how many atoms
    were summed and on the PAW potential, neither of which is being
    compared. Where each band sits is.
    """

    figure = drawn(
        ("Ni", "d-band", -2.0, 1.0),
        ("O", "p-band", -5.0, 0.8),
    )

    filled = [
        float(np.max(collection.get_paths()[0].vertices[:, 0]))
        for collection in figure.axes[0].collections
    ]

    assert filled

    assert max(filled) == pytest.approx(min(filled), abs=1e-6)


def test_the_scale_sets_how_far_a_curve_reaches() -> None:

    figure = Figure()

    ep.draw_band_centre(
        figure, curves_for(("Ni", "d-band", -2.0, 1.0)), scale=1.5
    )

    reach = float(
        np.max(figure.axes[0].collections[0].get_paths()[0].vertices[:, 0])
    )

    assert reach == pytest.approx(1.5, abs=1e-6)


def test_two_elements_sharing_a_band_are_told_apart() -> None:
    """
    A figure comparing the d states of two metals has to draw them in
    two colours, or there is nothing to read it by.
    """

    figure = drawn(
        ("Ni", "d-band", -2.0, 1.0),
        ("Fe", "d-band", -3.0, 1.0),
    )

    colours = {
        line.get_color()
        for line in figure.axes[0].lines
        if line.get_linestyle() not in ("-", "solid")
    }

    assert len(colours) == 2


def test_a_selection_with_no_states_is_named_rather_than_dropped() -> None:
    """
    Asking for the d band of oxygen is a reasonable thing to do by
    accident, and silence would read as "zero" rather than "none".
    """

    energies = np.linspace(-8.0, 3.0, 200)

    figure = Figure()

    ep.draw_band_centre(
        figure,
        [
            *curves_for(("Ni", "d-band", -2.0, 1.0)),
            el.CentreCurve(
                element="O",
                contribution="d-band",
                atoms=4,
                energies=energies,
                dos=np.zeros_like(energies),
                centre=float("nan"),
                width=float("nan"),
            ),
        ],
    )

    notes = [text.get_text() for text in figure.axes[0].texts]

    assert any("no states" in note and "O" in note for note in notes)


def test_nothing_to_draw_says_so_rather_than_drawing_empty_axes() -> None:

    energies = np.linspace(-8.0, 3.0, 200)

    figure = Figure()

    ep.draw_band_centre(
        figure,
        [
            el.CentreCurve(
                element="O",
                contribution="d-band",
                atoms=4,
                energies=energies,
                dos=np.zeros_like(energies),
                centre=float("nan"),
                width=float("nan"),
            )
        ],
    )

    notes = [text.get_text() for text in figure.axes[0].texts]

    assert any("no states" in note for note in notes)


# ------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------


def test_smoothing_leaves_a_short_curve_alone() -> None:
    """
    The filter needs a window longer than the polynomial order and no
    longer than the data. Violating either raises, so a curve too short
    to filter is returned as it is.
    """

    values = np.array([1.0, 2.0, 3.0])

    assert np.allclose(ep._smoothed(values, 21, 3), values)


def test_an_even_window_is_made_odd() -> None:
    """Savitzky-Golay refuses an even window rather than rounding it."""

    values = np.abs(np.sin(np.linspace(0.0, 6.0, 101)))

    assert ep._smoothed(values, 20, 3).shape == values.shape


def test_a_window_longer_than_the_data_is_brought_in() -> None:

    values = np.abs(np.sin(np.linspace(0.0, 6.0, 31)))

    assert ep._smoothed(values, 101, 3).shape == values.shape


def test_smoothing_does_not_move_a_peak() -> None:
    """
    Which is why the filter is used rather than a running mean: the
    centre is read off the curve, so a filter that shifted it would
    change the number being reported.
    """

    grid = np.linspace(-5.0, 5.0, 401)

    values = np.exp(-0.5 * (grid / 0.6) ** 2)

    smooth = ep._smoothed(values, 21, 3)

    assert grid[int(np.argmax(smooth))] == pytest.approx(
        grid[int(np.argmax(values))], abs=0.05
    )


# ------------------------------------------------------------
# Against the published definition
# ------------------------------------------------------------


def norskov(energy, dos) -> float:
    """
    The band centre as Hammer and Norskov define it, written out.

    Kept here in full rather than imported, so the implementation is
    checked against the equation instead of against itself.
    """

    energy = np.asarray(energy, dtype=float)
    dos = np.asarray(dos, dtype=float)

    keep = np.isfinite(energy) & np.isfinite(dos)

    energy, dos = energy[keep], dos[keep]

    if len(energy) < 2:
        return float("nan")

    denominator = np.trapezoid(dos, energy)

    if denominator <= 0:
        return float("nan")

    return float(np.trapezoid(energy * dos, energy) / denominator)


@pytest.mark.parametrize(
    "window",
    [(-15.0, 15.0), (-10.0, 2.0), (-6.0, 0.0), (-4.0, -1.0)],
)
def test_the_centre_is_the_norskov_ratio(window) -> None:
    """
    First moment over zeroth, on Fermi-referenced energies, inside the
    window. Any other reading of "band centre" is a different number.
    """

    energies = np.linspace(-15.0, 15.0, 3001)

    density = np.exp(
        -0.5 * ((energies + 2.0) / 0.4) ** 2
    ) + 0.3 * np.exp(-0.5 * ((energies - 1.0) / 0.6) ** 2)

    inside = (energies >= window[0]) & (energies <= window[1])

    expected = norskov(energies[inside], density[inside])

    got = el._moments(
        label="x",
        band="d",
        energies=energies,
        density=density,
        erange=window,
        count=1,
    )

    assert got.centre == pytest.approx(expected, abs=1e-12)


def test_the_window_changes_the_answer() -> None:
    """
    Which is why it is carried on the result rather than assumed. The
    same states give a different centre over a different range.
    """

    energies = np.linspace(-15.0, 15.0, 3001)

    density = np.exp(
        -0.5 * ((energies + 2.0) / 0.4) ** 2
    ) + 0.3 * np.exp(-0.5 * ((energies - 1.0) / 0.6) ** 2)

    whole = el._moments("x", "d", energies, density, (-15.0, 15.0), 1)
    below = el._moments("x", "d", energies, density, (-6.0, 0.0), 1)

    assert whole.centre != pytest.approx(below.centre, abs=1e-3)


def test_a_density_of_zero_has_no_centre() -> None:
    """
    The denominator is the states there are. None means there is no
    band, which is an absence rather than a NaN dressed as a number.
    """

    energies = np.linspace(-10.0, 10.0, 201)

    got = el._moments(
        "x", "d", energies, np.zeros_like(energies), (-5.0, 5.0), 1
    )

    assert not got.defined


# ------------------------------------------------------------
# How a contribution is named
# ------------------------------------------------------------


def test_a_shell_is_named_as_a_band() -> None:
    """The quantity the d-band model is about, and its published name."""

    assert el.contribution_label("d") == "d-band"
    assert el.contribution_label("p") == "p-band"
    assert el.contribution_label("s") == "s-band"


def test_a_resolved_orbital_is_named_plainly() -> None:
    """
    "dz2-band" would claim a band where there is one component of one.
    """

    assert el.contribution_label("dz2") == "dz2"
    assert el.contribution_label("dx2-y2") == "dx2-y2"


def test_a_request_names_itself_by_where_and_what() -> None:

    assert (
        el.CentreRequest(element="Co", orbital="d").label() == "Co d-band"
    )

    assert (
        el.CentreRequest(
            atoms=el.ATOMS_INDICES,
            indices="1-4",
            element="B",
            orbital="p",
        ).label()
        == "B [1-4] p-band"
    )


# ------------------------------------------------------------
# Atom indices, as the reference accepts them
# ------------------------------------------------------------


def test_indices_may_be_separated_by_spaces_or_commas() -> None:
    """
    The reference's own placeholder is "1 2 5 8-12 20"; people type
    commas out of habit. Refusing either would be a puzzle, not a rule.
    """

    assert el.parse_indices("1 2 5") == el.parse_indices("1,2,5")


def test_a_range_is_inclusive_at_both_ends() -> None:

    assert el.parse_indices("8-12") == [7, 8, 9, 10, 11]


def test_the_text_counts_from_one_and_the_answer_from_zero() -> None:
    """
    The whole reason this exists rather than a call to the orbital
    parser: forgetting the conversion silently reads the neighbouring
    atom, which is a wrong answer that looks perfectly reasonable.
    """

    assert el.parse_indices("1") == [0]


def test_repeats_and_order_do_not_matter() -> None:

    assert el.parse_indices("5 1 5 3") == [0, 2, 4]


def test_a_backwards_range_is_refused() -> None:

    with pytest.raises(el.ElectronicError, match="backwards"):
        el.parse_indices("12-8")


def test_something_that_is_not_a_number_is_refused() -> None:

    with pytest.raises(el.ElectronicError, match="not an atom number"):
        el.parse_indices("first")


# ------------------------------------------------------------
# A contribution and the orbital behind it
# ------------------------------------------------------------


def test_a_band_name_maps_back_to_its_shell() -> None:

    assert el.orbital_of("d-band") == "d"
    assert el.orbital_of("p-band") == "p"


def test_a_resolved_orbital_maps_to_itself() -> None:

    assert el.orbital_of("dz2") == "dz2"
    assert el.orbital_of("dx2-y2") == "dx2-y2"


def test_naming_round_trips() -> None:
    """A contribution named from an orbital has to name it back."""

    for orbital in ("s", "p", "d", "dz2", "px", "dx2-y2"):
        assert el.orbital_of(el.contribution_label(orbital)) == orbital


# ------------------------------------------------------------
# The two diagrams
# ------------------------------------------------------------


def test_both_plot_types_are_offered() -> None:

    assert list(ep.CENTRE_PLOTS) == [ep.PLOT_PDOS, ep.PLOT_SCHEMATIC]


def test_the_pdos_diagram_draws_the_real_states() -> None:

    figure = Figure()

    ep.draw_band_centre(
        figure, curves_for(("Ni", "d-band", -2.0, 1.0)), plot=ep.PLOT_PDOS
    )

    axes = figure.axes[0]

    assert "PDOS" in axes.get_xlabel()
    assert "DOSCAR" in axes.get_title()


def test_the_schematic_keeps_only_the_energies() -> None:
    """
    Every band the same set width at its own centre, so the only thing
    that varies down the page is where it sits.
    """

    figure = Figure()

    ep.draw_band_centre(
        figure,
        curves_for(("Ni", "d-band", -2.0, 1.0)),
        plot=ep.PLOT_SCHEMATIC,
        window=(-6.0, 1.0),
    )

    axes = figure.axes[0]

    assert list(axes.get_xticks()) == []
    assert "Band-Center" in axes.get_title()


def test_the_schematic_width_is_a_setting_not_the_second_moment() -> None:
    """
    It is a drawing of the numbers, not of the states, so two bands of
    different real widths are drawn the same.
    """

    narrow = Figure()
    wide = Figure()

    curves = curves_for(("Ni", "d-band", -2.0, 1.0))

    ep.draw_band_centre(
        narrow, curves, plot=ep.PLOT_SCHEMATIC,
        window=(-6.0, 1.0), band_width=0.1,
    )
    ep.draw_band_centre(
        wide, curves, plot=ep.PLOT_SCHEMATIC,
        window=(-6.0, 1.0), band_width=0.9,
    )

    def spread(figure):
        path = figure.axes[0].collections[0].get_paths()[0]
        inside = path.vertices[path.vertices[:, 0] > 0.4]
        return float(inside[:, 1].max() - inside[:, 1].min())

    assert spread(wide) > spread(narrow)


def test_a_series_keeps_its_colour_in_both_diagrams() -> None:
    """
    Reading one diagram against the other should be a matter of
    matching colours, not of matching labels.
    """

    curves = three_series()

    colours = ep.centre_colours(curves)

    for curve in curves:

        assert ep._centre_colour(curve, colours) == colours[
            (curve.element, curve.contribution)
        ]


# ------------------------------------------------------------
# Calculating and plotting are separate questions
# ------------------------------------------------------------


@pytest.fixture
def dialog():

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.electronic_dialog import BandCentreDialog

    QApplication.instance() or QApplication([])

    window = BandCentreDialog()

    window.run = two_species_run()
    window._offer(window.run)

    window.element_boxes["Ni"].setChecked(True)
    window.element_boxes["O"].setChecked(True)

    window.per_element["Ni"]["bands"]["d-band"].setChecked(True)
    window.per_element["O"]["bands"]["p-band"].setChecked(True)

    window.low.setValue(-15.0)
    window.high.setValue(15.0)

    return window


def calculated(dialog):

    dialog._on_result(
        el.centre_curves(dialog.run, dialog._requests(), erange=WINDOW)
    )

    return dialog


def test_calculating_shows_the_numbers(dialog) -> None:
    """
    The band centres are the result. The diagram is a way of looking at
    them, and it is not what was asked for.
    """

    calculated(dialog)

    assert dialog.results_window().isVisible()


def test_calculating_does_not_open_the_plot(dialog) -> None:

    calculated(dialog)

    assert not dialog.figures.isVisible()


def test_the_numbers_window_holds_the_reference_columns(dialog) -> None:

    calculated(dialog)

    table = dialog.results_window().table

    headers = [
        table.horizontalHeaderItem(column).text()
        for column in range(table.columnCount())
    ]

    assert headers == list(dialog.COLUMNS)

    assert table.rowCount() == 2


def test_the_numbers_can_be_saved(dialog) -> None:

    calculated(dialog)

    assert dialog.results_window().save_button.isEnabled()


def test_changing_a_plot_setting_does_not_raise_the_plot(dialog) -> None:
    """
    A setting changed is a request to change the plot, not to be shown
    it. A window jumping to the front on every nudge of a spin box is
    unusable.
    """

    calculated(dialog)

    dialog.pdos_scale.setValue(1.2)
    dialog.band_width.setValue(0.5)
    dialog.smooth_window.setValue(31)

    assert not dialog.figures.isVisible()


def test_the_figure_is_still_drawn_while_hidden(dialog) -> None:
    """So that Show plots has something to show, without recomputing."""

    calculated(dialog)

    assert dialog.panel.figure.axes


# ------------------------------------------------------------
# The per-element blocks follow the selection
# ------------------------------------------------------------


def blocks(dialog) -> list[str]:

    from PySide6.QtWidgets import QGroupBox

    return [
        box.title() for box in dialog.bands_host.findChildren(QGroupBox)
    ]


def test_one_block_per_selected_element(dialog) -> None:

    assert blocks(dialog) == ["Ni", "O"]


def test_deselecting_an_element_removes_its_block(dialog) -> None:
    """
    Guards a leak: deleteLater only queues the delete, so a rebuild
    found the old blocks still parented and stacked new ones under them.
    """

    dialog.element_boxes["Ni"].setChecked(False)

    assert blocks(dialog) == ["O"]


def test_reselecting_does_not_stack_duplicates(dialog) -> None:

    dialog.element_boxes["Ni"].setChecked(False)
    dialog.element_boxes["Ni"].setChecked(True)

    assert blocks(dialog) == ["Ni", "O"]


def test_ticks_survive_a_rebuild(dialog) -> None:
    """
    Changing which atoms are selected should not silently undo the
    bands already chosen for an element that is still there.
    """

    dialog.element_boxes["O"].setChecked(False)

    assert dialog.per_element["Ni"]["bands"]["d-band"].isChecked()


def test_the_configuration_shows_what_will_be_calculated(dialog) -> None:

    table = dialog.configuration

    rows = {
        table.item(row, 0).text(): table.item(row, 2).text()
        for row in range(table.rowCount())
    }

    assert rows == {"Ni": "d-band", "O": "p-band"}


# ------------------------------------------------------------
# One colour per series
# ------------------------------------------------------------


def three_series():

    energies = np.linspace(-8.0, 2.0, 400)

    def peak(at, width):
        return np.exp(-0.5 * ((energies - at) / width) ** 2)

    return [
        el.CentreCurve("Co", "p-band", 16, energies, peak(-2.1, 1.1),
                       -2.10, 1.10),
        el.CentreCurve("Co", "d-band", 16, energies, peak(-1.3, 0.8),
                       -1.30, 0.80),
        el.CentreCurve("B", "p-band", 16, energies, peak(-3.7, 1.3),
                       -3.68, 1.30),
    ]


def test_two_elements_sharing_a_band_get_different_colours() -> None:
    """
    Keyed on the contribution alone, Co p-band and B p-band came out the
    same green -- two curves, one colour, nothing to tell them apart by.
    """

    colours = ep.centre_colours(three_series())

    assert colours[("Co", "p-band")] != colours[("B", "p-band")]


def test_every_series_gets_its_own_colour() -> None:

    colours = ep.centre_colours(three_series())

    assert len(set(colours.values())) == len(colours)


def test_a_series_keeps_its_colour_between_the_two_diagrams() -> None:
    """
    The same selection has to be the same colour in both, or reading one
    against the other means matching labels instead of colours.
    """

    curves = three_series()

    assert ep.centre_colours(curves) == ep.centre_colours(curves)


def test_a_repeated_series_is_not_given_a_second_colour() -> None:

    curves = three_series()

    colours = ep.centre_colours(curves + curves)

    assert len(colours) == 3


# ------------------------------------------------------------
# Labels name the series and nothing else
# ------------------------------------------------------------


@pytest.mark.parametrize("plot", [ep.PLOT_PDOS, ep.PLOT_SCHEMATIC])
def test_a_label_is_the_series_name(plot) -> None:
    """
    The number is in the table beside the figure. Repeating it on every
    curve crowded the panel without adding anything.
    """

    figure = Figure()

    ep.draw_band_centre(
        figure, three_series(), plot=plot, window=(-8.0, 2.0)
    )

    labels = [
        text.get_text()
        for text in figure.axes[0].texts
        if "band" in text.get_text()
    ]

    assert labels == ["Co p-band", "Co d-band", "B p-band"]

    for label in labels:
        assert "eV" not in label


# ------------------------------------------------------------
# The results window fits what it holds
# ------------------------------------------------------------


def test_the_results_window_grows_with_its_rows(dialog) -> None:
    """
    A fixed height left most of the window empty under three rows and
    scrolled when there were thirty.
    """

    from qvista.gui.dialogs import electronic_dialog as ed

    window = dialog.results_window()

    columns = ("A", "B", "C", "D")

    window.show_rows(columns, [("x", "y", "z", "w")] * 2)
    small = window.table.minimumHeight()

    window.show_rows(columns, [("x", "y", "z", "w")] * 8)
    larger = window.table.minimumHeight()

    assert larger > small


def test_the_results_window_stops_growing_at_a_screenful(dialog) -> None:

    from qvista.gui.dialogs import electronic_dialog as ed

    window = dialog.results_window()

    window.show_rows(
        ("A", "B", "C", "D"), [("x", "y", "z", "w")] * 200
    )

    assert window.table.minimumHeight() <= ed.MAX_TABLE_HEIGHT


def test_an_empty_result_still_leaves_a_usable_window(dialog) -> None:

    from qvista.gui.dialogs import electronic_dialog as ed

    window = dialog.results_window()

    window.show_rows(("A", "B", "C", "D"), [])

    assert window.table.minimumHeight() >= ed.MIN_TABLE_HEIGHT

    assert not window.save_button.isEnabled()


def test_the_footer_save_is_gone_where_the_window_carries_one(
    dialog,
) -> None:
    """
    Two Save buttons for one table, one of them further from it.
    """

    dialog.show()

    assert not dialog.save_button.isVisible()


# ------------------------------------------------------------
# Labels sit clear of the line they name
# ------------------------------------------------------------


@pytest.mark.parametrize("plot", [ep.PLOT_PDOS, ep.PLOT_SCHEMATIC])
def test_a_label_sits_below_its_centre_line(plot) -> None:
    """
    Centred on the centre, the text sat on the very line it names -- and
    two selections close in energy put one label through the other's
    rule.
    """

    figure = Figure()

    ep.draw_band_centre(
        figure, three_series(), plot=plot, window=(-8.0, 2.0)
    )

    labels = [
        text
        for text in figure.axes[0].texts
        if "band" in text.get_text()
    ]

    assert labels

    for label in labels:
        assert label.get_va() == "top"


def test_the_drop_is_measured_in_points_not_energy() -> None:
    """
    So it stays clear at any zoom. In data units it would grow and
    shrink with the window.
    """

    figure = Figure()

    ep.draw_band_centre(
        figure, three_series(), plot=ep.PLOT_PDOS, window=(-8.0, 2.0)
    )

    label = next(
        text
        for text in figure.axes[0].texts
        if "band" in text.get_text()
    )

    # The anchor is the centre itself, in data coordinates; the drop is
    # the offset, in points. get_position() reports the offset.
    assert label.xy[1] == pytest.approx(-2.10)

    assert label.get_position()[1] == -ep.LABEL_DROP


# ------------------------------------------------------------
# Tables fit what they hold
# ------------------------------------------------------------


def test_a_table_is_sized_to_its_rows() -> None:

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication, QTableWidget

    from qvista.gui.dialogs import electronic_dialog as ed

    QApplication.instance() or QApplication([])

    table = QTableWidget(0, 3)

    heights = []

    for rows in (1, 4, 10):

        table.setRowCount(rows)

        ed.fit_table(table, rows)

        heights.append(table.maximumHeight())

    assert heights == sorted(heights)
    assert len(set(heights)) > 1


def test_a_table_stops_growing_rather_than_filling_the_window() -> None:

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication, QTableWidget

    from qvista.gui.dialogs import electronic_dialog as ed

    QApplication.instance() or QApplication([])

    table = QTableWidget(400, 3)

    ed.fit_table(table, 400)

    assert table.maximumHeight() <= ed.MAX_TABLE_HEIGHT


def test_an_empty_table_keeps_a_usable_height() -> None:

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication, QTableWidget

    from qvista.gui.dialogs import electronic_dialog as ed

    QApplication.instance() or QApplication([])

    table = QTableWidget(0, 3)

    ed.fit_table(table, 0)

    assert table.maximumHeight() >= ed.MIN_TABLE_HEIGHT


def test_the_configuration_table_fits_its_rows(dialog) -> None:
    """
    It showed five rows of empty grid under two real ones.
    """

    from qvista.gui.dialogs import electronic_dialog as ed

    assert dialog.configuration.rowCount() == 2

    assert dialog.configuration.maximumHeight() <= ed.CONFIG_MOST


def test_the_summary_tab_fits_its_rows(dialog) -> None:

    calculated(dialog)

    from qvista.gui.dialogs import electronic_dialog as ed

    assert dialog.table.maximumHeight() <= ed.MAX_TABLE_HEIGHT

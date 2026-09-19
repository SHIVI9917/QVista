"""
Tests for the electronic-structure tools.

The band centre is checked against a density of states built by hand:
a Gaussian centred at a known energy has that energy as its first
moment, so the pipeline can be asked to recover a number decided in
advance rather than merely to produce a plausible one.

The selection and index handling is tested on stub runs, because the
part that goes wrong is off-by-one between the one-based numbering every
table in QVista uses and the zero-based projection array.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services import electronic as el


# ------------------------------------------------------------
# Stubs
# ------------------------------------------------------------


def stub_run(symbols=("Li", "Li", "Co", "O", "O"), nbands=8, nkpoints=4):
    """A Run with only the fields the selection helpers touch."""

    return el.Run(
        vasprun=None,
        directory=None,
        symbols=tuple(symbols),
        natoms=len(symbols),
        nbands=nbands,
        nkpoints=nkpoints,
    )


# ------------------------------------------------------------
# Selections
# ------------------------------------------------------------


def test_elements_keep_the_order_they_appear_in() -> None:

    assert stub_run().elements() == ["Li", "Co", "O"]


def test_element_selection_gathers_every_atom_of_a_species() -> None:

    selection = el.resolve_selection(stub_run(), "element")

    assert selection == {"Li": [0, 1], "Co": [2], "O": [3, 4]}


def test_atom_selection_is_one_based_going_in_and_zero_based_coming_out(
) -> None:
    """
    The off-by-one this module is most likely to get wrong.

    Every table in QVista numbers atoms from one; the projection array
    is indexed from zero. Getting it backwards silently projects onto
    the neighbouring atom, which looks entirely reasonable.
    """

    selection = el.resolve_selection(stub_run(), "atom", atoms="1,3")

    assert selection == {"1 Li": [0], "3 Co": [2]}


def test_an_atom_outside_the_structure_is_refused() -> None:

    for text in ("0", "6", "1-9"):

        with pytest.raises(el.ElectronicError):
            el.resolve_selection(stub_run(), "atom", atoms=text)


def test_an_empty_atom_selection_is_refused() -> None:

    with pytest.raises(el.ElectronicError, match="at least one"):
        el.resolve_selection(stub_run(), "atom", atoms="")


def test_an_unknown_projection_is_refused() -> None:

    with pytest.raises(el.ElectronicError):
        el.resolve_selection(stub_run(), "spin")


def test_the_orbital_columns_match_the_vasp_ordering() -> None:
    """
    LORBIT = 11 writes nine channels in a fixed order.

    A wrong slice here mislabels every projected curve without ever
    failing, so the boundaries are pinned.
    """

    assert el.orbital_columns("s") == [0]
    assert el.orbital_columns("p") == [1, 2, 3]
    assert el.orbital_columns("d") == [4, 5, 6, 7, 8]

    assert el.orbital_columns("all") == list(range(9))

    # A single named orbital, not just a shell.
    assert el.orbital_columns("dz2") == [6]


def test_an_unknown_orbital_is_refused() -> None:

    with pytest.raises(el.ElectronicError):
        el.orbital_columns("f")


def test_indices_default_to_everything() -> None:

    assert el._resolve_indices(None, 4, "band") == [0, 1, 2, 3]
    assert el._resolve_indices([], 3, "band") == [0, 1, 2]


def test_indices_come_back_zero_based_and_deduplicated() -> None:

    assert el._resolve_indices([1, 3, 3], 5, "band") == [0, 2]


def test_an_index_past_the_end_is_refused() -> None:

    with pytest.raises(el.ElectronicError, match="outside the run"):
        el._resolve_indices([9], 4, "k-point")


def test_selected_atoms_merges_names_and_elements() -> None:

    run = stub_run()

    chosen = el.selected_atoms(run, atoms="1", elements=["O"])

    assert chosen == [0, 3, 4]


def test_selecting_nothing_at_all_is_refused() -> None:

    with pytest.raises(el.ElectronicError, match="Nothing selected"):
        el.selected_atoms(stub_run())


# ------------------------------------------------------------
# Broadening
# ------------------------------------------------------------


def test_a_gaussian_lands_where_the_eigenvalue_is() -> None:

    grid = np.linspace(-5.0, 5.0, 2001)

    values = np.array([[1.5]])

    curve = el._accumulate(values, [0], [0], None, grid, sigma=0.05)

    assert grid[int(np.argmax(curve))] == pytest.approx(1.5, abs=0.01)


def test_broadening_conserves_the_number_of_states() -> None:
    """
    One state in, one state of area out.

    The normalisation is what makes the area of a hand-broadened DOS
    mean something, and it is invisible until it is wrong.
    """

    grid = np.linspace(-5.0, 5.0, 4001)

    values = np.array([[0.0, 1.0, 2.0]])

    curve = el._accumulate(values, [0], [0, 1, 2], None, grid, sigma=0.1)

    assert np.trapezoid(curve, grid) == pytest.approx(3.0, rel=1e-3)


def test_weights_scale_the_area() -> None:

    grid = np.linspace(-5.0, 5.0, 4001)

    values = np.array([[0.0]])

    curve = el._accumulate(
        values, [0], [0], np.array([[0.25]]), grid, sigma=0.1
    )

    assert np.trapezoid(curve, grid) == pytest.approx(0.25, rel=1e-3)


def test_a_wider_gaussian_is_shorter() -> None:

    grid = np.linspace(-5.0, 5.0, 2001)

    values = np.array([[0.0]])

    narrow = el._accumulate(values, [0], [0], None, grid, sigma=0.05)
    wide = el._accumulate(values, [0], [0], None, grid, sigma=0.5)

    assert wide.max() < narrow.max()

    assert np.trapezoid(narrow, grid) == pytest.approx(
        np.trapezoid(wide, grid), rel=1e-3
    )


# ------------------------------------------------------------
# Band centres, against a DOS with a known answer
# ------------------------------------------------------------


def gaussian_dos(centre: float, width: float = 0.5, fermi: float = 0.0):
    """
    A CompleteDos whose d channel is one Gaussian at a known energy.

    Its first moment is the Gaussian's centre by construction, so the
    band centre that comes back can be compared with a number decided
    before the code ran.
    """

    from pymatgen.core import Lattice, Structure
    from pymatgen.electronic_structure.core import Orbital, Spin
    from pymatgen.electronic_structure.dos import CompleteDos, Dos

    energies = np.linspace(-15.0, 15.0, 6001)

    shape = np.exp(-0.5 * ((energies - centre) / width) ** 2)

    structure = Structure(
        Lattice.cubic(3.0), ["Ni"], [[0.0, 0.0, 0.0]]
    )

    total = Dos(fermi, energies, {Spin.up: shape})

    # Everything in the d shell, nothing anywhere else.
    site_dos = {
        structure[0]: {
            orbital: {Spin.up: shape if orbital.name.startswith("d")
                      else np.zeros_like(shape)}
            for orbital in Orbital
        }
    }

    return CompleteDos(structure, total, site_dos)


def test_the_band_centre_is_the_first_moment() -> None:
    """In at minus two, out at minus two."""

    dos = gaussian_dos(centre=-2.0)

    from pymatgen.electronic_structure.core import OrbitalType

    got = dos.get_band_center(band=OrbitalType.d, erange=(-15.0, 15.0))

    assert got == pytest.approx(-2.0, abs=0.01)


def test_the_band_width_is_the_second_moment() -> None:
    """A Gaussian of width w has a second moment of w."""

    from pymatgen.electronic_structure.core import OrbitalType

    dos = gaussian_dos(centre=-2.0, width=0.8)

    got = dos.get_band_width(band=OrbitalType.d, erange=(-15.0, 15.0))

    assert got == pytest.approx(0.8, rel=0.05)


def test_the_window_changes_the_centre() -> None:
    """
    Which is why the window is a parameter and is reported.

    Two Gaussians, one either side of the Fermi level: including the
    empty one or not moves the centre by electron-volts, and neither
    answer is wrong -- only unstated.
    """

    from pymatgen.core import Lattice, Structure
    from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
    from pymatgen.electronic_structure.dos import CompleteDos, Dos

    energies = np.linspace(-15.0, 15.0, 6001)

    shape = (
        np.exp(-0.5 * ((energies + 3.0) / 0.5) ** 2)
        + np.exp(-0.5 * ((energies - 3.0) / 0.5) ** 2)
    )

    structure = Structure(Lattice.cubic(3.0), ["Ni"], [[0.0, 0.0, 0.0]])

    site_dos = {
        structure[0]: {
            orbital: {Spin.up: shape if orbital.name.startswith("d")
                      else np.zeros_like(shape)}
            for orbital in Orbital
        }
    }

    dos = CompleteDos(structure, Dos(0.0, energies, {Spin.up: shape}), site_dos)

    both = dos.get_band_center(band=OrbitalType.d, erange=(-15.0, 15.0))
    occupied = dos.get_band_center(band=OrbitalType.d, erange=(-15.0, 0.0))

    assert both == pytest.approx(0.0, abs=0.05)
    assert occupied == pytest.approx(-3.0, abs=0.05)

    assert abs(both - occupied) > 2.0


def test_the_default_window_is_stated_and_finite() -> None:
    """
    Not None, on purpose.

    With no window the empty states far above the Fermi level dominate
    the first moment and the number stops describing the band anyone
    means by "the d-band centre".
    """

    low, high = el.DEFAULT_ERANGE

    assert low < 0.0 < high
    assert np.isfinite(low) and np.isfinite(high)


# ------------------------------------------------------------
# What a BandCentre row says about itself
# ------------------------------------------------------------


def test_a_shell_with_no_states_is_not_defined() -> None:
    """An oxygen d-band centre is not a number; it is an absence."""

    row = el.BandCentre(
        label="O", band="d", centre=float("nan"), width=float("nan"),
        skewness=float("nan"), kurtosis=float("nan"), filling=float("nan"),
        upper_edge=float("nan"), erange=(-10.0, 2.0), count=24, weight=0.0,
    )

    assert not row.defined


def test_a_shell_with_states_is_defined() -> None:

    row = el.BandCentre(
        label="Co", band="d", centre=-1.45, width=2.35, skewness=-7.5,
        kurtosis=69.5, filling=0.58, upper_edge=-0.25,
        erange=(-10.0, 2.0), count=12, weight=108.6,
    )

    assert row.defined


def test_a_nan_centre_is_not_defined_even_with_weight() -> None:
    """Both conditions, because either alone has been seen to slip."""

    row = el.BandCentre(
        label="X", band="d", centre=float("nan"), width=1.0,
        skewness=0.0, kurtosis=0.0, filling=0.5, upper_edge=0.0,
        erange=(-10.0, 2.0), count=1, weight=5.0,
    )

    assert not row.defined


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


def test_a_folder_with_no_vasprun_says_so(tmp_path) -> None:

    with pytest.raises(el.ElectronicError, match="No vasprun.xml"):
        el.load(tmp_path)


def test_a_broken_vasprun_is_translated(tmp_path) -> None:
    """A truncated run is common; the raw XML error is not helpful."""

    (tmp_path / "vasprun.xml").write_text("<modeling>", encoding="utf-8")

    with pytest.raises(el.ElectronicError, match="could not be parsed"):
        el.load(tmp_path)


def test_a_short_k_list_is_not_a_path() -> None:
    """Four points are a mesh, whatever else they are."""

    assert not el._is_line_mode(
        _stub_vasprun(), np.zeros((4, 3))
    )


def test_a_long_irregular_k_list_is_a_path() -> None:

    path = np.linspace(0.0, 0.5, 60).reshape(-1, 1) * np.array([1.0, 1.0, 0.0])

    assert el._is_line_mode(_stub_vasprun(), path)


class _stub_vasprun:
    """Just enough to ask about the k-point style."""

    kpoints = None


# ------------------------------------------------------------
# Choices
# ------------------------------------------------------------


def test_the_projections_offered_are_the_ones_implemented() -> None:

    assert set(el.PROJECTIONS) == {"total", "atom", "element", "orbital"}


def test_the_shells_are_s_p_d_in_filling_order() -> None:

    assert el.SHELLS == ("s", "p", "d")


# ------------------------------------------------------------
# Which eigenvalues a band figure is showing
#
# A self-consistent run samples a mesh; a band structure needs a path.
# Plotting the mesh is useful -- it shows the gap and the band edges
# without a second calculation -- but it is not a dispersion, and the
# distinction has to survive into the figure rather than being lost the
# moment somebody screenshots it.
# ------------------------------------------------------------


def test_the_two_band_modes_are_distinct() -> None:

    assert el.PATH_MODE != el.SCF_MODE

    assert set(el.BAND_MODES) == {el.PATH_MODE, el.SCF_MODE}


def test_a_path_result_says_it_is_one() -> None:

    result = el.BandResult(mode=el.PATH_MODE)

    assert result.is_path
    assert "Wave vector" in result.xlabel


def test_an_scf_result_says_it_is_not_a_path() -> None:
    """
    The x-axis label carries the caveat.

    It is the only part of the figure that survives being pasted into a
    slide, so "not a path" belongs there rather than only in the notes.
    """

    result = el.BandResult(mode=el.SCF_MODE)

    assert not result.is_path

    assert "not a path" in result.xlabel


def test_the_default_mode_is_a_path() -> None:
    """So the existing dispersion route is unaffected by the addition."""

    assert el.BandResult().mode == el.PATH_MODE


def test_a_mesh_run_is_still_refused_for_a_dispersion() -> None:
    """
    ``band_structure`` must not quietly fall back to the mesh.

    The fallback belongs in the dialog, where the user can see which of
    the two they were given; doing it here would make a mesh silently
    answer a question about a dispersion.
    """

    run = stub_run()
    run.line_mode = False

    with pytest.raises(el.ElectronicError, match="not a path"):
        el.band_structure(run)


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------


def test_the_two_draw_styles_are_offered() -> None:

    from qvista.visualization import electronic_plots as plots

    assert set(plots.BAND_STYLES) == {
        plots.LINE_STYLE, plots.POINT_STYLE
    }


def test_a_mesh_is_drawn_as_points_by_default() -> None:
    """
    Joining mesh k-points would draw branches that do not exist.

    The mesh is listed in whatever order VASP generated it, so a line
    between consecutive points connects two unrelated corners of the
    Brillouin zone. It looks exactly like a dispersion and is not one.
    """

    from matplotlib.figure import Figure

    from qvista.visualization import electronic_plots as plots

    result = el.BandResult(
        distances=[np.arange(1.0, 5.0)],
        energies={"up": [np.array([[-1.0, -1.1, -0.9, -1.2],
                                   [1.0, 1.2, 0.8, 1.1]])]},
        mode=el.SCF_MODE,
    )

    figure = Figure()

    plots.draw_band(figure, result)

    axes = figure.axes[0]

    assert not axes.lines or all(
        # The only line allowed is the Fermi level marker.
        line.get_linestyle() == "--" for line in axes.lines
    )

    assert axes.collections, "the states should have been scattered"


def test_a_path_is_drawn_as_lines_by_default() -> None:

    from matplotlib.figure import Figure

    from qvista.visualization import electronic_plots as plots

    result = el.BandResult(
        distances=[np.linspace(0.0, 1.0, 4)],
        energies={"up": [np.array([[-1.0, -1.1, -0.9, -1.2],
                                   [1.0, 1.2, 0.8, 1.1]])]},
        mode=el.PATH_MODE,
    )

    figure = Figure()

    plots.draw_band(figure, result)

    # Two bands plus the Fermi line.
    assert len(figure.axes[0].lines) >= 2


def test_the_style_can_be_forced_either_way() -> None:
    """The automatic choice is a default, not a lock."""

    from matplotlib.figure import Figure

    from qvista.visualization import electronic_plots as plots

    result = el.BandResult(
        distances=[np.arange(1.0, 5.0)],
        energies={"up": [np.array([[-1.0, -1.1, -0.9, -1.2]])]},
        mode=el.SCF_MODE,
    )

    figure = Figure()

    plots.draw_band(figure, result, style=plots.LINE_STYLE)

    solid = [
        line for line in figure.axes[0].lines
        if line.get_linestyle() == "-"
    ]

    assert solid, "a forced line style should draw a line"


# ------------------------------------------------------------
# The orbital projection, at two levels of detail
#
# The same vocabulary as the Orbital window, deliberately: "resolved"
# has to mean the same nine channels in the same order wherever it is
# offered, or a figure and a table of the same state disagree.
# ------------------------------------------------------------


def test_the_orbital_views_come_from_one_place() -> None:
    """Shared with the Orbital window rather than redeclared."""

    from qvista.services import orbital as ob

    assert el.ORBITAL_VIEWS is ob.VIEWS

    assert el.DEFAULT_ORBITAL_VIEW == ob.VIEW_SPD


def test_the_spd_series_is_three_shells() -> None:

    assert el.orbital_series(el.VIEW_SPD) == ["s", "p", "d"]


def test_the_resolved_series_is_all_nine() -> None:

    from qvista.services import orbital as ob

    assert el.orbital_series(el.VIEW_RESOLVED) == list(ob.ORBITALS)


def test_every_series_name_is_a_valid_column() -> None:
    """
    The weighting code takes a label and asks for its columns.

    If a view ever produced a name ``orbital_columns`` did not know,
    the resolved figure would fail at run time rather than at import.
    """

    for view in el.ORBITAL_VIEWS:
        for name in el.orbital_series(view):
            assert el.orbital_columns(name)


def test_an_unknown_view_is_refused() -> None:

    with pytest.raises(el.ElectronicError):
        el.orbital_series("f")


def test_a_named_shell_expands_in_the_resolved_view() -> None:
    """
    The shell chips still narrow a resolved figure.

    Asking for "d" resolved means the five d orbitals, not one lump
    called d -- otherwise the chips and the view contradict each other.
    """

    assert el._orbital_labels(["d"], el.VIEW_RESOLVED) == [
        "dxy", "dyz", "dz2", "dxz", "dx2-y2"
    ]

    assert el._orbital_labels(["d"], el.VIEW_SPD) == ["d"]


def test_a_named_orbital_survives_either_view() -> None:

    assert el._orbital_labels(["dz2"], el.VIEW_RESOLVED) == ["dz2"]


def test_no_shells_means_the_whole_series() -> None:

    assert el._orbital_labels(None, el.VIEW_SPD) == ["s", "p", "d"]

    assert len(el._orbital_labels([], el.VIEW_RESOLVED)) == 9


# ------------------------------------------------------------
# Contributions
# ------------------------------------------------------------


def flat_curves():
    """Curves whose integrals are known by construction."""

    energies = np.linspace(-2.0, 2.0, 401)

    # Half of each is below zero, so the occupied integral is half the
    # whole one and the arithmetic is checkable by hand.
    return el.DosResult(
        curves=[
            el.Curve("s", energies, np.ones_like(energies)),
            el.Curve("p", energies, 3.0 * np.ones_like(energies)),
        ]
    )


def test_a_contribution_is_the_integral_below_the_fermi_level() -> None:

    shares = el.contributions(flat_curves())

    # 2 eV wide, height 1 and 3.
    assert shares[0].occupied == pytest.approx(2.0, rel=1e-3)
    assert shares[1].occupied == pytest.approx(6.0, rel=1e-3)


def test_the_whole_integral_is_reported_alongside() -> None:
    """So a series that is mostly empty states is visible as such."""

    shares = el.contributions(flat_curves())

    assert shares[0].whole == pytest.approx(4.0, rel=1e-3)


def test_contributions_add_to_a_hundred_percent() -> None:

    shares = el.contributions(flat_curves())

    assert sum(share.percent for share in shares) == pytest.approx(100.0)

    assert shares[0].percent == pytest.approx(25.0, rel=1e-3)
    assert shares[1].percent == pytest.approx(75.0, rel=1e-3)


def test_the_window_can_be_moved() -> None:
    """The Fermi level is the default, not the only choice."""

    everything = el.contributions(flat_curves(), upto=2.0)

    assert everything[0].occupied == pytest.approx(4.0, rel=1e-3)


def test_a_mirrored_spin_channel_still_counts_positive() -> None:
    """
    Spin down is drawn below the axis on some paths.

    Integrating it signed would give a negative share, and a percentage
    that sums to something other than a hundred.
    """

    energies = np.linspace(-2.0, 2.0, 401)

    result = el.DosResult(
        curves=[
            el.Curve("s", energies, np.ones_like(energies), "up"),
            el.Curve("s", energies, -np.ones_like(energies), "down"),
        ]
    )

    shares = el.contributions(result)

    assert all(share.occupied > 0 for share in shares)

    assert sum(share.percent for share in shares) == pytest.approx(100.0)


def test_an_empty_series_is_marked_rather_than_dividing_by_zero() -> None:

    energies = np.linspace(-2.0, 2.0, 401)

    result = el.DosResult(
        curves=[el.Curve("d", energies, np.zeros_like(energies))]
    )

    shares = el.contributions(result)

    assert shares[0].empty
    assert shares[0].percent == 0.0


def test_band_contributions_average_the_occupied_weights() -> None:

    result = el.BandResult(
        distances=[np.arange(1.0, 5.0)],
        energies={"up": [np.array([[-1.0, -1.0, -1.0, -1.0],
                                   [1.0, 1.0, 1.0, 1.0]])]},
        weights={
            "s": np.array([[0.8, 0.8, 0.8, 0.8], [0.0, 0.0, 0.0, 0.0]]),
            "d": np.array([[0.2, 0.2, 0.2, 0.2], [1.0, 1.0, 1.0, 1.0]]),
        },
    )

    shares = {row.label: row for row in el.band_contributions(result)}

    # Only the lower band is occupied.
    assert shares["s"].occupied == pytest.approx(0.8)
    assert shares["d"].occupied == pytest.approx(0.2)

    assert shares["s"].percent == pytest.approx(80.0)


# ------------------------------------------------------------
# The total density of states
#
# It used to be summed over the spin channels into a single positive
# trace and laid over every projected figure. Both halves of that were
# wrong: the sum has no lower half where every curve beside it does,
# and the total on a forty-atom cell is an order of magnitude larger
# than any projection, so overlaying it flattens what it is drawn on.
# ------------------------------------------------------------


def test_the_total_is_kept_per_spin() -> None:

    result = el.DosResult(
        totals=[
            el.Curve("total", np.zeros(3), np.ones(3), "up"),
            el.Curve("total", np.zeros(3), np.ones(3), "down"),
        ]
    )

    assert {curve.spin for curve in result.totals} == {"up", "down"}


def test_a_dos_result_starts_with_no_totals() -> None:
    """So a result built by hand does not claim to have one."""

    assert el.DosResult().totals == []


def test_the_total_is_not_drawn_by_default() -> None:
    """
    The figure has to be asked for it now.

    Left on, every projected chart would still be flattened by a curve
    an order of magnitude taller than the ones being looked at.
    """

    import inspect

    from qvista.visualization import electronic_plots

    signature = inspect.signature(electronic_plots.draw_dos)

    assert signature.parameters["with_total"].default is False


def test_the_total_is_mirrored_when_it_is_drawn() -> None:
    """Like every other curve, or the two halves stop comparing."""

    from matplotlib.figure import Figure

    from qvista.visualization import electronic_plots

    energies = np.linspace(-1.0, 1.0, 20)

    result = el.DosResult(
        curves=[],
        totals=[
            el.Curve("total", energies, np.ones(20), "up"),
            el.Curve("total", energies, np.ones(20), "down"),
        ],
    )

    figure = Figure()

    electronic_plots.draw_dos(figure, result, with_total=True)

    # The Fermi line aside -- it is the one drawn at a single x -- every
    # line here is a total. They come off a B-spline, so they carry more
    # points than the twenty they were built from.
    drawn = [
        line.get_ydata()
        for line in figure.axes[0].lines
        if len(line.get_ydata()) > 2
    ]

    assert any(np.all(values > 0) for values in drawn)
    assert any(np.all(values < 0) for values in drawn), (
        "the minority channel was not mirrored below the axis"
    )


def test_the_total_is_named_once_in_the_legend() -> None:
    """Two channels, one entry."""

    from matplotlib.figure import Figure

    from qvista.visualization import electronic_plots

    energies = np.linspace(-1.0, 1.0, 20)

    result = el.DosResult(
        totals=[
            el.Curve("total", energies, np.ones(20), "up"),
            el.Curve("total", energies, np.ones(20), "down"),
        ]
    )

    figure = Figure()

    electronic_plots.draw_dos(figure, result, with_total=True)

    labels = figure.axes[0].get_legend_handles_labels()[1]

    assert labels.count("total") == 1


# ------------------------------------------------------------
# A projected band figure has to name its series
# ------------------------------------------------------------


def _legend_labels(figure) -> list[str]:
    """
    What the legend actually says.

    Read off the drawn legend rather than off the artists, because the
    band figure builds its key from proxy handles: a fat-band marker's
    size is its weight, so a swatch taken from a real one would be
    whatever that band happened to weigh.
    """

    legend = figure.axes[0].get_legend()

    if legend is None:
        return []

    return [text.get_text() for text in legend.get_texts()]


def band_result(weights):
    """A two-band result carrying the given per-series weights."""

    return el.BandResult(
        distances=[np.linspace(0.0, 1.0, 4)],
        energies={
            "up": [
                np.array([[-3.0, -3.0, -3.0, -3.0],
                          [1.0, 1.0, 1.0, 1.0]])
            ]
        },
        weights=weights,
    )


def test_every_series_reaches_the_legend() -> None:
    """
    The label used to go on band 0 and nowhere else.

    Band 0 is the deepest, most core-like state, and for almost any
    selection its weight is under the drawing threshold -- so it was
    skipped, taking that series' only legend entry with it. Every
    atom-wise and orbital-resolved band figure came out with no legend.
    """

    from matplotlib.figure import Figure

    from qvista.visualization import electronic_plots

    # Nothing on band 0; everything on band 1.
    weights = {
        "Co": np.array([[0.0, 0.0, 0.0, 0.0], [0.9, 0.9, 0.9, 0.9]]),
        "O": np.array([[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]),
    }

    figure = Figure()

    electronic_plots.draw_band(figure, band_result(weights))

    labels = _legend_labels(figure)

    assert "Co" in labels
    assert "O" in labels


def test_a_series_is_named_once_however_many_bands_carry_it() -> None:

    from matplotlib.figure import Figure

    from qvista.visualization import electronic_plots

    weights = {
        "Co": np.array([[0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]])
    }

    figure = Figure()

    electronic_plots.draw_band(figure, band_result(weights))

    labels = _legend_labels(figure)

    assert labels.count("Co") == 1


def test_a_series_with_no_weight_anywhere_is_still_named() -> None:
    """
    So an empty selection reads as empty.

    Silently omitting it makes "I asked for Li and got nothing" look
    identical to "I forgot to ask for Li".
    """

    from matplotlib.figure import Figure

    from qvista.visualization import electronic_plots

    weights = {"Li": np.zeros((2, 4))}

    figure = Figure()

    electronic_plots.draw_band(figure, band_result(weights))

    labels = _legend_labels(figure)

    assert any(label.startswith("Li") for label in labels)
    assert any("below" in label for label in labels)


# ----------------------------------------------------------------
# How a DOS curve is drawn between its points
# ----------------------------------------------------------------


def dos_result(points: int = 21):
    """
    A single Gaussian peak on a deliberately coarse grid.

    Coarse on purpose: that is what a low NEDOS gives, and it is the
    case the two curve styles actually differ on.
    """

    import numpy as np

    class Curve:
        def __init__(self, label, energies, values, spin="up"):
            self.label = label
            self.energies = energies
            self.values = values
            self.spin = spin

    class Result:
        def __init__(self, curves):
            self.curves = curves
            self.totals = []

    energies = np.linspace(-5.0, 5.0, points)

    return Result([Curve("Ni d", energies, np.exp(-(energies**2)))])


def drawn_curve(figure):
    """The DOS line, which is the only one with more than two points."""

    return [
        line
        for line in figure.axes[0].lines
        if len(line.get_xdata()) > 2
    ][0]


def dos_figure(style):

    import matplotlib

    matplotlib.use("Agg")

    from matplotlib.figure import Figure

    from qvista.visualization import electronic_plots

    figure = Figure()

    electronic_plots.draw_dos(figure, dos_result(), curve_style=style)

    return figure


def test_both_curve_styles_are_offered():

    from qvista.visualization import electronic_plots

    assert set(electronic_plots.CURVE_STYLES) == {
        electronic_plots.CURVE_SPLINE,
        electronic_plots.CURVE_LINE,
    }


def test_the_smooth_curve_is_the_default():
    """
    A DOSCAR grid is coarse enough that a sharp peak reads as a corner,
    and the smooth curve is what nearly every figure wants.
    """

    from qvista.visualization import electronic_plots

    assert (
        next(iter(electronic_plots.CURVE_STYLES))
        == electronic_plots.CURVE_SPLINE
    )


def test_a_line_plot_draws_exactly_the_calculated_points():
    """
    Nothing added and nothing taken away. This is the style to use when
    the question is whether the energy grid is fine enough -- smoothing
    hides the corners that say it is not.
    """

    from qvista.visualization import electronic_plots

    line = drawn_curve(dos_figure(electronic_plots.CURVE_LINE))

    assert len(line.get_xdata()) == 21


def test_a_spline_plot_draws_more_points_than_were_calculated():

    from qvista.visualization import electronic_plots

    line = drawn_curve(dos_figure(electronic_plots.CURVE_SPLINE))

    assert len(line.get_xdata()) > 21


def test_the_spline_keeps_the_peak_it_was_given():
    """
    It interpolates rather than approximating, so a peak keeps its
    height and its position -- and the area under it, which is the
    number the table beside the figure reports.
    """

    import numpy as np

    from qvista.visualization import electronic_plots

    for style in (
        electronic_plots.CURVE_LINE,
        electronic_plots.CURVE_SPLINE,
    ):

        line = drawn_curve(dos_figure(style))

        assert float(np.max(line.get_ydata())) == pytest.approx(1.0)


def test_the_spline_passes_through_every_calculated_point():

    import numpy as np

    from qvista.visualization import electronic_plots

    plain = drawn_curve(dos_figure(electronic_plots.CURVE_LINE))
    smooth = drawn_curve(dos_figure(electronic_plots.CURVE_SPLINE))

    grid = np.asarray(smooth.get_xdata())
    values = np.asarray(smooth.get_ydata())

    for x, y in zip(plain.get_xdata(), plain.get_ydata()):

        nearest = int(np.argmin(np.abs(grid - x)))

        assert values[nearest] == pytest.approx(y, abs=1e-6)


def test_the_dialog_offers_the_choice_and_defaults_to_smooth():

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.electronic_dialog import DosDialog
    from qvista.visualization import electronic_plots

    QApplication.instance() or QApplication([])

    dialog = DosDialog()

    offered = [
        dialog.curve_style.itemData(index)
        for index in range(dialog.curve_style.count())
    ]

    assert offered == list(electronic_plots.CURVE_STYLES)
    assert dialog.curve_style.currentData() == electronic_plots.CURVE_SPLINE


def test_switching_style_before_a_run_does_nothing():
    """
    It is a redraw, not a re-run -- and before the first run there is
    nothing to redraw, which should be quiet rather than an error.
    """

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.electronic_dialog import DosDialog

    QApplication.instance() or QApplication([])

    dialog = DosDialog()

    assert dialog.result is None

    dialog.curve_style.setCurrentIndex(1)

    assert dialog.curve_style.currentData() == "line"


# ------------------------------------------------------------------
# Selection and orbital together
# ------------------------------------------------------------------


def test_the_projection_bases_are_named() -> None:
    """
    Two bases, and they are the two the dialog offers.

    The shells are a separate choice on purpose: "which element" and
    "which shell" are independent questions, and offering them as
    alternatives is what made the d states of one species impossible to
    ask for.
    """

    from qvista.services import electronic as el

    assert set(el.BASES) == {"element", "atom"}


def test_a_bad_basis_says_what_the_bases_are() -> None:
    """An unknown basis is refused by name, not by traceback."""

    from qvista.services import electronic as el

    class Fake:
        vasprun = type("V", (), {"complete_dos": object()})()

    with pytest.raises(el.ElectronicError) as caught:
        el.projected_curves(Fake(), basis="sideways")

    assert "sideways" in str(caught.value)
    assert "element" in str(caught.value)


def test_atom_numbers_are_one_based() -> None:
    """
    Atom 1 means the first atom.

    parse_atom_list hands back the numbers as typed, which are 1-based
    like every table in QVista, while the structure is indexed from
    zero. Indexing with the typed number plots the atom *after* the one
    that was asked for and says nothing about it.
    """

    import inspect

    from qvista.services import electronic as el

    source = inspect.getsource(el._selection_sites)

    assert "number - 1" in source


def test_band_labels_name_the_shell_when_it_is_a_subset() -> None:
    """
    A series carrying only d character is not labelled "Co".

    The weighting already intersects the selection with the shells, so
    the label has to say so -- otherwise the legend describes a
    different quantity from the one plotted.
    """

    import inspect

    from qvista.services import electronic as el

    source = inspect.getsource(el._band_weights)

    assert "len(shells) < len(SHELLS)" in source
    assert "out[named]" in source


def test_both_windows_offer_plain_and_projected() -> None:
    """
    The top-level choice is two questions, not six.

    Element wise and Orbital wise as alternatives on one row is what
    made a fat band of one species' d character unreachable.
    """

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from qvista.gui.dialogs.electronic_dialog import BandDialog, DosDialog

    for cls in (DosDialog, BandDialog):

        assert "plain" in cls.MODES, cls.__name__
        assert "projected" in cls.MODES, cls.__name__

        # The old alternatives are gone from the top row.
        assert "element" not in cls.MODES, cls.__name__
        assert "orbital" not in cls.MODES, cls.__name__

        dialog = cls()

        assert dialog.mode == "projected"
        assert dialog.basis.count() == 2
        assert not dialog.resolved.isChecked()


def test_the_basis_disables_the_selector_it_does_not_use() -> None:
    """
    Element wise ignores the atom field, so the field says so.

    Leaving both live invites filling in one and wondering why the
    other was used.
    """

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from qvista.gui.dialogs.electronic_dialog import DosDialog

    dialog = DosDialog()

    dialog.basis.setCurrentIndex(dialog.basis.findData("element"))
    assert not dialog.atoms.isEnabled()

    dialog.basis.setCurrentIndex(dialog.basis.findData("atom"))
    assert dialog.atoms.isEnabled()


def test_a_band_run_without_a_fermi_level_still_plots() -> None:
    """
    One missing number is not a reason to refuse the whole band.

    A non-self-consistent band run (ICHARG = 11) does not always write
    a Fermi level into vasprun.xml -- it read the density rather than
    converging one. pymatgen then refuses with "e_fermi is None", which
    reads as the run being unusable. VASPKIT writes the value beside it
    in FERMI_ENERGY, and that is the one the accompanying SCF run
    found.
    """

    import inspect

    from qvista.services import electronic as el

    source = inspect.getsource(el._fermi_for)

    assert el.FERMI_FILE == "FERMI_ENERGY"

    # Falls back to the file, then to pymatgen's own estimate rather
    # than raising.
    assert "FERMI_FILE" in source
    assert "smart" in source

    # And the band reader actually passes it through.
    assert "efermi=_fermi_for(run)" in inspect.getsource(el.band_structure)


def test_the_fermi_file_is_parsed_past_its_comments(tmp_path) -> None:
    """VASPKIT writes a comment line above the number."""

    from qvista.services import electronic as el

    (tmp_path / el.FERMI_FILE).write_text(
        "# Read Fermi-Energy from the FERMI_ENERGY.in file if it exists.\n"
        "    6.304098        # e.g., Set the value of VBM\n",
        encoding="utf-8",
    )

    class Fake:
        vasprun = type("V", (), {"efermi": None})()
        directory = tmp_path

    assert el._fermi_for(Fake()) == pytest.approx(6.304098)


def test_resolved_splits_a_band_selection_into_orbitals() -> None:
    """
    Resolved gives one series per orbital, not one per shell.

    Without this the checkbox changed nothing on a band plot: the
    weighting summed the whole shell whatever the view said, so the
    t2g/eg distinction the option exists for never appeared.
    """

    import inspect

    from qvista.services import electronic as el

    source = inspect.getsource(el._band_weights)

    assert "VIEW_RESOLVED" in source
    assert "parts" in source


def test_the_resolved_checkbox_uses_a_key_not_a_position() -> None:
    """
    ORBITAL_VIEWS is a mapping, and indexing it by position raised
    KeyError the moment Resolved was ticked -- so the band never drew.
    """

    from qvista.services import electronic as el

    assert isinstance(el.ORBITAL_VIEWS, dict)
    assert el.VIEW_RESOLVED in el.ORBITAL_VIEWS

    import inspect

    from qvista.gui.dialogs import electronic_dialog

    source = inspect.getsource(electronic_dialog.BandDialog._run)

    assert "el.VIEW_RESOLVED" in source
    assert "ORBITAL_VIEWS[1]" not in source


def test_figure_dialogs_are_not_opened_modally() -> None:
    """
    A dialog that draws into its own window must not be modal.

    These put their plot in a separate top-level window, and an
    application-modal dialog sits above every other window in the
    application -- so the figure opened behind the dialog that made it.
    The table filled and the plot drew, and neither could be seen,
    which reads as the analysis having returned nothing.
    """

    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent

    # Every dialog class that owns a FigureWindow.
    owners = set()

    for path in (root / "qvista" / "gui" / "dialogs").glob("*.py"):

        text = path.read_text(encoding="utf-8")

        if "FigureWindow(" not in text:
            continue

        current = None

        for line in text.splitlines():

            found = re.match(r"class ([A-Za-z_]+)", line)

            if found:
                current = found.group(1)

            if "FigureWindow(" in line and current:
                owners.add(current)

    assert owners, "no FigureWindow owners found -- has the layout moved?"

    main = (root / "qvista" / "gui" / "main_window.py").read_text(
        encoding="utf-8"
    )

    offenders = []

    for name in sorted(owners):

        for match in re.finditer(r"(?<![\w.])" + name + r"\(", main):

            tail = main[match.end(): match.end() + 600]

            closes = tail.find(".exec()")
            opens = tail.find("self._open(")

            if closes != -1 and (opens == -1 or closes < opens):

                if "self._open(dialog)" not in tail[: closes + 400]:
                    offenders.append(name)

    assert not offenders, (
        f"opened modally, so their figure hides behind them: "
        f"{sorted(set(offenders))}"
    )

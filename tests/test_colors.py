"""
Tests for the element colour schemes.

The bug these guard against is silent: two of the three palettes were empty
dicts, so ``get_color`` fell through to the third and every scheme in the
menu rendered identically. Nothing failed, the dropdown just did nothing.
"""

from __future__ import annotations

import pytest

from qvista.visualization.colors import (
    COLOR_SCHEME_NAMES,
    COLOR_SCHEMES,
    CPK_COLORS,
    JMOL_COLORS,
    UNKNOWN_ELEMENT_COLOR,
    VESTA_COLORS,
    available_color_schemes,
    get_color,
)
from qvista.visualization.settings import ColorScheme

#: Present in the user's own test structures.
COMMON = ("H", "C", "N", "O", "Li", "Mn", "Fe", "Si", "Pb", "Br")


# ------------------------------------------------------------
# Every table is populated
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "scheme",
    [ColorScheme.CPK, ColorScheme.JMOL, ColorScheme.VESTA],
)
def test_every_scheme_has_colours(scheme: ColorScheme) -> None:
    """No scheme may be an empty dict."""

    table = COLOR_SCHEMES[scheme]

    assert len(table) > 90, f"{scheme.name} defines only {len(table)} elements"


@pytest.mark.parametrize(
    "scheme",
    [ColorScheme.CPK, ColorScheme.JMOL, ColorScheme.VESTA],
)
def test_common_elements_are_defined(scheme: ColorScheme) -> None:
    """Elements the user actually opens must be covered directly."""

    table = COLOR_SCHEMES[scheme]

    for element in COMMON:
        assert element in table, f"{element} missing from {scheme.name}"


# ------------------------------------------------------------
# The schemes are actually different
# ------------------------------------------------------------


def test_schemes_are_not_identical() -> None:
    """
    Selecting a different scheme must change what is drawn.

    This is the regression test for the original defect: three menu
    entries that all produced the same picture.
    """

    for element in COMMON:

        colours = {
            get_color(element, scheme)
            for scheme in (
                ColorScheme.CPK,
                ColorScheme.JMOL,
                ColorScheme.VESTA,
            )
        }

        assert len(colours) > 1, (
            f"all schemes render {element} identically"
        )


def test_carbon_differs_across_all_three() -> None:
    """Carbon is the classic point of difference between the palettes."""

    cpk = get_color("C", ColorScheme.CPK)
    jmol = get_color("C", ColorScheme.JMOL)
    vesta = get_color("C", ColorScheme.VESTA)

    assert len({cpk, jmol, vesta}) == 3


def test_cpk_is_not_jmol_data() -> None:
    """
    CPK must be the CPK palette, not Jmol wearing its label.

    The original table was 99% identical to Jmol.
    """

    shared = [
        element
        for element in CPK_COLORS
        if element in JMOL_COLORS
    ]

    identical = sum(
        1
        for element in shared
        if CPK_COLORS[element] == JMOL_COLORS[element]
    )

    assert identical < len(shared) * 0.2, (
        "CPK table looks like a copy of Jmol"
    )


# ------------------------------------------------------------
# Values are well formed
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "table,name",
    [
        (CPK_COLORS, "CPK"),
        (JMOL_COLORS, "Jmol"),
        (VESTA_COLORS, "VESTA"),
    ],
)
def test_colours_are_rgb_triples_in_range(table, name: str) -> None:
    """Every entry is an RGB triple normalised to [0, 1]."""

    for element, colour in table.items():

        assert len(colour) == 3, f"{name} {element} is not a triple"

        for channel in colour:
            assert 0.0 <= channel <= 1.0, (
                f"{name} {element} has an out-of-range channel: {colour}"
            )


# ------------------------------------------------------------
# Lookup behaviour
# ------------------------------------------------------------


def test_unknown_symbol_falls_back_to_grey() -> None:
    """An unrecognised symbol must not raise."""

    assert get_color("Xx", ColorScheme.CPK) == UNKNOWN_ELEMENT_COLOR


def test_element_missing_from_one_scheme_falls_back() -> None:
    """
    A scheme that lacks an element borrows from Jmol, not from grey.

    CPK covers 103 elements; Jmol and VESTA reach further. An element past
    CPK's range should still draw in a sensible colour.
    """

    beyond = [
        element
        for element in JMOL_COLORS
        if element not in CPK_COLORS
    ]

    if not beyond:
        pytest.skip("CPK covers every element the other schemes define")

    element = beyond[0]

    assert get_color(element, ColorScheme.CPK) == JMOL_COLORS[element]
    assert get_color(element, ColorScheme.CPK) != UNKNOWN_ELEMENT_COLOR


# ------------------------------------------------------------
# Registry
# ------------------------------------------------------------


def test_every_scheme_is_selectable_and_named() -> None:
    """The dropdown can offer, and label, every registered scheme."""

    schemes = available_color_schemes()

    assert set(schemes) == set(COLOR_SCHEMES)

    for scheme in schemes:
        assert COLOR_SCHEME_NAMES[scheme]


# ------------------------------------------------------------
# The viewport, and everything drawn against it
#
# The background used to be near-black, and the cell box, the atom
# labels and the orientation-marker letters were all chosen pale to
# show up against it. It is white now. Nothing here checks that white
# is the right choice -- that is taste -- but each of those three has
# to contrast with whatever the background is, and flipping one without
# the others produces a viewport that renders perfectly and shows
# nothing.
# ------------------------------------------------------------


def _luminance(colour) -> float:
    """Perceived brightness of an RGB triple in 0-1, or a hex string."""

    if isinstance(colour, str):
        text = colour.lstrip("#")
        parts = [int(text[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    else:
        parts = list(colour)[:3]

    red, green, blue = parts

    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def test_the_viewport_is_light() -> None:
    """A figure lifted out of here lands in a document, not in the app."""

    from qvista.visualization.settings import ViewerSettings

    settings = ViewerSettings()

    assert _luminance(settings.background_color) > 0.5
    assert _luminance(settings.background_top) > 0.5


def test_the_cell_box_contrasts_with_the_background() -> None:

    from qvista.visualization.settings import ViewerSettings

    settings = ViewerSettings()

    difference = abs(
        _luminance(settings.cell_color)
        - _luminance(settings.background_color)
    )

    assert difference > 0.3, "the cell box would be invisible"


def test_the_atom_labels_contrast_with_the_background() -> None:

    from qvista.visualization.settings import ViewerSettings

    settings = ViewerSettings()

    difference = abs(
        _luminance(settings.label_color)
        - _luminance(settings.background_color)
    )

    assert difference > 0.3, "labels would be invisible"


def test_the_hint_and_axes_contrast_with_the_background() -> None:

    from qvista.visualization import viewer
    from qvista.visualization.settings import ViewerSettings

    ground = _luminance(ViewerSettings().background_color)

    for name, colour in (
        ("hint", viewer.HINT_COLOR),
        ("axes labels", viewer.AXES_LABEL_COLOR),
    ):
        assert abs(_luminance(colour) - ground) > 0.3, (
            f"{name} would be invisible"
        )


def test_the_hint_is_the_theme_blue() -> None:
    """
    The one thing on an empty viewport should belong to the app.

    Pinned to the theme constant rather than to a literal, so a change
    to the palette carries here rather than leaving one orphan colour.
    """

    from qvista.gui import theme
    from qvista.visualization import viewer

    assert viewer.HINT_COLOR == theme.IRIS_DEEP

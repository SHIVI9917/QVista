"""
Tests for screen-aware window sizing.

The interesting case is a screen smaller than the one these tests run
on, so the available size is stubbed rather than read from the display:
otherwise the clamp would never fire and the tests would pass on a large
monitor while the behaviour they describe was broken.
"""

from __future__ import annotations

import pytest

from qvista.gui import sizing


class FakeWidget:
    """Just enough of QWidget to record what minimum was applied."""

    def __init__(self) -> None:
        self.minimum: tuple[int, int] | None = None
        self.size: tuple[int, int] | None = None

    def setMinimumSize(self, width, height) -> None:  # noqa: N802
        self.minimum = (int(width), int(height))

    def resize(self, width, height) -> None:
        self.size = (int(width), int(height))

    def screen(self):
        return None


@pytest.fixture
def small_screen(monkeypatch):
    """A 1366x768 laptop, where the big dialogs do not fit."""

    monkeypatch.setattr(sizing, "available_size", lambda widget=None: (1366, 768))


@pytest.fixture
def large_screen(monkeypatch):

    monkeypatch.setattr(sizing, "available_size", lambda widget=None: (2560, 1440))


def test_a_big_dialog_is_cut_down_to_the_screen(small_screen) -> None:
    """
    The case this module exists for.

    A 880-pixel floor on a 768-pixel screen puts part of the dialog
    permanently out of reach: a minimum size cannot be dragged smaller,
    so whatever falls off the bottom stays off.
    """

    widget = FakeWidget()

    applied = sizing.fit_to_screen(widget, 1120, 880)

    assert applied[1] < 768
    assert widget.minimum == applied


def test_the_clamp_leaves_room_for_the_title_bar(small_screen) -> None:

    widget = FakeWidget()

    _, height = sizing.fit_to_screen(widget, 4000, 4000)

    assert height < 768


def test_a_dialog_that_already_fits_is_left_alone(large_screen) -> None:
    """Never enlarge, never shrink what does not need shrinking."""

    widget = FakeWidget()

    assert sizing.fit_to_screen(widget, 1120, 880) == (1120, 880)


def test_width_and_height_are_clamped_independently(monkeypatch) -> None:
    """A wide, short screen must not shrink the width too."""

    monkeypatch.setattr(
        sizing, "available_size", lambda widget=None: (3440, 700)
    )

    widget = FakeWidget()

    width, height = sizing.fit_to_screen(widget, 1120, 880)

    assert width == 1120
    assert height < 700


def test_an_absurdly_small_screen_still_leaves_a_usable_floor(
    monkeypatch,
) -> None:
    """
    Below a point, shrinking further helps nobody.

    A virtual display can report a few hundred pixels; clamping a dialog
    to that would produce a window with no room for its own controls,
    which is not an improvement on one that overflows.
    """

    monkeypatch.setattr(
        sizing, "available_size", lambda widget=None: (200, 150)
    )

    widget = FakeWidget()

    assert sizing.fit_to_screen(widget, 1120, 880) == sizing.FLOOR


def test_opening_size_also_respects_the_screen(small_screen) -> None:

    widget = FakeWidget()

    sizing.open_at(widget, 1600, 1200)

    assert widget.size is not None
    assert widget.size[0] <= 1366
    assert widget.size[1] <= 768


def test_a_modest_opening_size_is_honoured(large_screen) -> None:

    widget = FakeWidget()

    sizing.open_at(widget, 1200, 800)

    assert widget.size == (1200, 800)


def test_available_size_falls_back_when_there_is_no_screen(
    monkeypatch,
) -> None:
    """Headless runs must not crash on a missing primary screen."""

    monkeypatch.setattr(
        sizing.QGuiApplication, "primaryScreen", staticmethod(lambda: None)
    )

    assert sizing.available_size(None) == (1920, 1080)

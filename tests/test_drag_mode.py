"""
The drag mode: left button slides the scene instead of turning it.

Panning already lives on the middle button and on shift-drag, but a
trackpad has neither, and sliding a structure across the viewport is
constant work while a figure is being framed.

These are pure. The real viewer cannot be built here -- VTK wants an
OpenGL context -- so the shipped method is bound onto a stub, which
checks the part that is ours: which mapping is asked for, and what has
to happen alongside it. The rendering half is covered in
``test_navigation.py``.
"""

from __future__ import annotations

import pytest

from qvista.visualization.viewer import CrystalViewer


class FakeViewer:
    """Enough of a viewer to record what the drag mode does to it."""

    def __init__(self) -> None:

        self.drag_mode = False
        self.styles: list[dict] = []
        self.muted = 0
        self.cursors: list = []

    def enable_custom_trackball_style(self, **buttons) -> None:
        self.styles.append(buttons)

    def setCursor(self, cursor) -> None:  # noqa: N802 - Qt's name
        self.cursors.append(cursor)


@pytest.fixture
def fake(monkeypatch):

    viewer = FakeViewer()

    monkeypatch.setattr(
        "qvista.visualization.viewer.mute_style_wheel",
        lambda _: setattr(viewer, "muted", viewer.muted + 1),
    )

    return viewer


def test_drag_mode_puts_pan_on_the_left_button(fake):

    CrystalViewer.set_drag_mode(fake, True)

    assert fake.drag_mode
    assert fake.styles == [{"left": "pan"}]


def test_leaving_drag_mode_restores_the_default_mapping(fake):
    """
    Back to rotate on the left, by asking for the style's own defaults
    rather than naming them here. Naming them would be a second copy of
    the mapping to keep in step with PyVista's.
    """

    CrystalViewer.set_drag_mode(fake, True)
    CrystalViewer.set_drag_mode(fake, False)

    assert not fake.drag_mode
    assert fake.styles[-1] == {}


def test_every_style_change_mutes_the_new_wheel(fake):
    """
    The bug this guards: a fresh style arrives at VTK's default wheel
    factor, and the wheel then zooms twice per notch -- once from the
    style, once from the zoom-to-cursor observers, which live on the
    interactor and survive the swap.
    """

    CrystalViewer.set_drag_mode(fake, True)
    CrystalViewer.set_drag_mode(fake, False)

    assert fake.muted == len(fake.styles)


def test_the_cursor_says_which_mode_is_on(fake):

    from PySide6.QtCore import Qt

    CrystalViewer.set_drag_mode(fake, True)
    CrystalViewer.set_drag_mode(fake, False)

    assert fake.cursors == [Qt.OpenHandCursor, Qt.ArrowCursor]


def test_the_flag_is_a_plain_bool(fake):
    """A Qt checked state is not always one, and this is read as one."""

    CrystalViewer.set_drag_mode(fake, 1)

    assert fake.drag_mode is True


def test_toggling_it_repeatedly_settles_correctly(fake):

    for _ in range(3):
        CrystalViewer.set_drag_mode(fake, True)
        CrystalViewer.set_drag_mode(fake, False)

    CrystalViewer.set_drag_mode(fake, True)

    assert fake.drag_mode
    assert fake.styles[-1] == {"left": "pan"}


def test_drag_mode_draws_nothing(fake):
    """
    Panning is the camera sliding sideways, so toggling the mode must
    not rebuild the scene -- on a large cell that would put a visible
    stall on a button whose whole job is to feel immediate.

    The stub has no ``redraw`` and no ``render``, so reaching for either
    raises rather than passing quietly. That the calls below succeed is
    the assertion.
    """

    assert not hasattr(fake, "redraw")
    assert not hasattr(fake, "render")

    CrystalViewer.set_drag_mode(fake, True)
    CrystalViewer.set_drag_mode(fake, False)


# ----------------------------------------------------------------
# The button
# ----------------------------------------------------------------


def test_the_panel_offers_a_drag_toggle():

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.widgets.visualization import VisualizationPanel

    app = QApplication.instance() or QApplication([])

    panel = VisualizationPanel()

    assert panel.drag_button.isCheckable()
    assert not panel.drag_button.isChecked()
    assert not panel.drag_button.icon().isNull()


def test_the_drag_icon_is_drawn_rather_than_a_font_glyph():
    """
    The obvious characters for a move mark are missing from enough UI
    fonts that the button would show a replacement box on some machines.
    """

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.icons import move_icon

    QApplication.instance() or QApplication([])

    icon = move_icon()

    assert not icon.isNull()
    assert icon.availableSizes()

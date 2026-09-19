"""
Window sizing that respects the screen it is on.

A dialog that declares ``setMinimumSize(1120, 880)`` is making a promise
it cannot keep on a laptop: this display reports 1536x960 of usable
space, so an 880-pixel floor leaves eighty pixels for the title bar and
the taskbar, and the parts of the dialog that do not fit are simply
unreachable -- a minimum size is a floor, so the user cannot shrink the
window to bring them back either.

:func:`fit_to_screen` takes the size a dialog would like and clamps it
to what the screen can actually show, so the layout stays scrollable and
resizable instead of overflowing.  Nothing here enlarges a dialog; it
only ever gives back a size that fits.
"""

from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

#: Left for the title bar, the taskbar and a little breathing room, as a
#: fraction of the usable screen.
SCREEN_MARGIN = 0.92

#: Never clamp below this, however small the screen claims to be: a
#: dialog squeezed past it is not usable anyway.
FLOOR = (480, 360)


def available_size(widget: QWidget | None = None) -> tuple[int, int]:
    """
    Usable width and height of the screen ``widget`` is on.

    Falls back to the primary screen when the widget has not been shown
    yet and so does not know which screen it belongs to.
    """

    screen = None

    if widget is not None:
        handle = widget.screen()

        if handle is not None:
            screen = handle

    if screen is None:
        screen = QGuiApplication.primaryScreen()

    if screen is None:
        return (1920, 1080)

    geometry = screen.availableGeometry()

    return (geometry.width(), geometry.height())


def fit_to_screen(
    widget: QWidget,
    width: int,
    height: int,
) -> tuple[int, int]:
    """
    Give ``widget`` a minimum size that the screen can actually show.

    Returns the size that was applied, which is the requested one on a
    large display and a clamped one on a small display.
    """

    screen_width, screen_height = available_size(widget)

    wanted_width = min(width, max(FLOOR[0], int(screen_width * SCREEN_MARGIN)))
    wanted_height = min(
        height, max(FLOOR[1], int(screen_height * SCREEN_MARGIN))
    )

    widget.setMinimumSize(wanted_width, wanted_height)

    return (wanted_width, wanted_height)


def open_at(
    widget: QWidget,
    width: int,
    height: int,
) -> None:
    """
    Size a window to ``width`` x ``height``, or to the screen if smaller.

    For the initial size rather than the floor: a dialog should open
    roomy on a big monitor without opening off the edge of a small one.
    """

    screen_width, screen_height = available_size(widget)

    widget.resize(
        min(width, int(screen_width * SCREEN_MARGIN)),
        min(height, int(screen_height * SCREEN_MARGIN)),
    )

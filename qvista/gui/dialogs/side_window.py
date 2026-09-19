"""
Base class for QVista's floating side windows.

Results that want room -- a Brillouin zone, a band structure, a 6x6 tensor
-- get their own top-level window rather than a dock. Docks in the bottom
strip are short and wide, which is the wrong shape for all of them, and the
console owns that strip anyway.

Windows are modeless, sized square, and placed beside the main window so
both stay visible.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog

#: Preferred edge length. Clamped to whatever the screen allows.
PREFERRED_SIZE = 900

#: Gap between the main window and this one.
MARGIN = 12


class SideWindow(QDialog):
    """A modeless, resizable window placed beside the main window."""

    def __init__(
        self,
        title: str,
        parent=None,
        preferred_size: int = PREFERRED_SIZE,
    ) -> None:

        super().__init__(parent)

        self.setWindowTitle(title)

        # A real window rather than a dialog box: minimise and maximise
        # buttons, and it can be left open alongside the main window.
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )

        self.setModal(False)

        self._preferred_size = preferred_size

        self.resize(preferred_size, preferred_size)

    # ------------------------------------------------------------

    def present(self, reference=None) -> None:
        """
        Show the window beside ``reference``, raised and focused.

        Only positioned the first time, so a window the user has moved or
        resized stays where they put it.
        """

        if not self.isVisible():
            self._place_beside(reference or self.parent())

        self.show()

        # A minimised window must be restored before raising, or it stays
        # in the taskbar.
        if self.isMinimized():
            self.showNormal()

        self.raise_()
        self.activateWindow()

    def _place_beside(self, reference) -> None:
        """Size the window square and put it to the right of ``reference``."""

        screen = None

        if reference is not None:
            screen = QGuiApplication.screenAt(
                reference.frameGeometry().center()
            )

        if screen is None:
            screen = QGuiApplication.primaryScreen()

        if screen is None:
            return

        available = screen.availableGeometry()

        size = max(
            400,
            min(self._preferred_size, available.height() - 2 * MARGIN),
        )

        self.resize(size, size)

        # Prefer the right of the main window; fall back to the right edge
        # of the screen when there is not enough room.
        if reference is not None:
            geometry = reference.frameGeometry()
            x = geometry.right() + MARGIN
            y = geometry.top()
        else:
            x = available.right() - size
            y = available.top()

        if x + size > available.right():
            x = available.right() - size

        x = max(available.left(), x)

        if y + size > available.bottom():
            y = available.bottom() - size

        y = max(available.top(), y)

        self.move(x, y)

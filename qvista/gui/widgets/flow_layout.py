"""
A layout that wraps its children like words in a paragraph.

Qt ships nothing of the sort: a grid needs a column count decided in
advance, and a horizontal box refuses to wrap at all. Rows of toggles
need neither -- a row of nine chips should become three rows when the
pane is narrow and one when it is wide, without anyone choosing a number.

The geometry is the standard flow: walk the items, start a new line
whenever the next one would cross the right edge, and report the height
that took so a scroll area can size itself. ``heightForWidth`` is what
makes that work inside one.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy


class FlowLayout(QLayout):
    """Left-to-right, wrapping onto a new line when it runs out of room."""

    def __init__(self, parent=None, spacing: int = 6) -> None:

        super().__init__(parent)

        self._items: list = []

        self.setSpacing(spacing)

    # ------------------------------------------------------------
    # The list of items, which QLayout keeps in the subclass
    # ------------------------------------------------------------

    def addItem(self, item) -> None:  # noqa: N802 - Qt's name

        self._items.append(item)

    def count(self) -> int:

        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt's name

        if 0 <= index < len(self._items):
            return self._items[index]

        return None

    def takeAt(self, index: int):  # noqa: N802 - Qt's name

        if 0 <= index < len(self._items):
            return self._items.pop(index)

        return None

    # ------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------

    def expandingDirections(self):  # noqa: N802 - Qt's name
        """Nothing here wants extra space; the wrapping absorbs it."""

        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt's name

        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt's name

        return self._arrange(QRect(0, 0, width, 0), place=False)

    def setGeometry(self, rect) -> None:  # noqa: N802 - Qt's name

        super().setGeometry(rect)

        self._arrange(rect, place=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's name

        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt's name

        size = QSize()

        for item in self._items:
            size = size.expandedTo(item.minimumSize())

        margins = self.contentsMargins()

        return size + QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )

    # ------------------------------------------------------------

    def _arrange(self, rect: QRect, place: bool) -> int:
        """
        Lay the items out inside ``rect``, and return the height used.

        Called twice for every resize -- once by ``heightForWidth`` to
        ask how tall this would be, once by ``setGeometry`` to actually
        do it -- so ``place`` decides whether anything moves.
        """

        margins = self.contentsMargins()

        area = rect.adjusted(
            margins.left(),
            margins.top(),
            -margins.right(),
            -margins.bottom(),
        )

        x = area.x()
        y = area.y()

        tallest = 0

        for item in self._items:

            hint = item.sizeHint()

            # Wrap, unless this is the first thing on the line: an item
            # wider than the whole pane has to go somewhere.
            if tallest > 0 and x + hint.width() > area.right():

                x = area.x()
                y += tallest + self.spacing()
                tallest = 0

            if place:
                item.setGeometry(QRect(QPoint(x, y), hint))

            x += hint.width() + self.spacing()

            tallest = max(tallest, hint.height())

        return y + tallest - rect.y() + margins.bottom()


def flow_widget(spacing: int = 6):
    """A plain widget carrying a :class:`FlowLayout`, ready to fill."""

    from PySide6.QtWidgets import QWidget

    host = QWidget()

    layout = FlowLayout(host, spacing=spacing)
    layout.setContentsMargins(0, 0, 0, 0)

    # Without this the widget above is asked only for a size *hint*, gets
    # the height of a single row back, and everything below it is drawn
    # on top of the rows that wrapped. A parent layout only asks the
    # height-for-width question when the size policy says there is one.
    policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    policy.setHeightForWidth(True)

    host.setSizePolicy(policy)

    return host, layout

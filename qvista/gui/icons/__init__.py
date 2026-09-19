"""
Small vector marks used by the stylesheet.

Qt draws a combo box's arrow itself only while the drop-down is
unstyled; the moment the sub-control is given a background the arrow
becomes the stylesheet's responsibility, and there is no way to draw
one in QSS -- the CSS border-triangle trick renders as a filled box.
So the marks are files. They are two-colour by necessity (a
stylesheet cannot recolour an image), which is why each comes with an
``_active`` twin for hover; keep both in step with the palette in
:mod:`qvista.gui.theme`.
"""

from __future__ import annotations

from pathlib import Path

#: Where the marks live, as a POSIX path -- Qt stylesheets take
#: forward slashes on every platform, including Windows.
DIRECTORY = Path(__file__).resolve().parent


def url(name: str) -> str:
    """The ``url(...)`` a stylesheet needs for one mark."""

    return f"url({(DIRECTORY / name).as_posix()})"


def move_icon(color: str | None = None, size: int = 40):
    """
    The four-arrow mark that means "drag this".

    Drawn rather than set as a glyph. The obvious characters for it --
    U+2725 and friends -- are missing from enough UI fonts that the
    button would show a replacement box on some machines, and a button
    whose label is a box is worse than one with no icon at all.

    ``color`` defaults to the panel's text colour. A checkable button
    keeps one icon in both states, so this is drawn in a shade that
    holds up against the chip's checked background as well as its
    normal one.
    """

    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

    from qvista.gui import theme

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    ink = QColor(color or theme.TEXT)

    pen = QPen(ink, max(2.0, size * 0.075))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)

    painter.setPen(pen)

    middle = size / 2.0

    # The shaft ends stop short of the rim so the heads have room.
    reach = size * 0.36
    head = size * 0.13

    painter.drawLine(
        QPointF(middle - reach, middle), QPointF(middle + reach, middle)
    )
    painter.drawLine(
        QPointF(middle, middle - reach), QPointF(middle, middle + reach)
    )

    # One chevron per arm, pointing outwards.
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):

        tip = QPointF(middle + dx * reach, middle + dy * reach)

        # Perpendicular to the arm, so the chevron opens the right way.
        for side in (1, -1):

            painter.drawLine(
                tip,
                QPointF(
                    tip.x() - dx * head + dy * head * side,
                    tip.y() - dy * head + dx * head * side,
                ),
            )

    painter.end()

    return QIcon(pixmap)


def app_icon():
    """
    The window and taskbar mark, drawn rather than shipped.

    A rounded tile with a crystal motif on it -- one atom bonded to
    three, which is the smallest thing that reads as a structure at
    16 pixels. Drawing it here keeps it in step with the palette
    automatically and avoids carrying a binary asset for one image.
    """

    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import (
        QBrush,
        QColor,
        QIcon,
        QLinearGradient,
        QPainter,
        QPen,
        QPixmap,
    )

    from qvista.gui import theme

    size = 256

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # The tile.
    ground = QLinearGradient(0, 0, size, size)
    ground.setColorAt(0.0, QColor(theme.IRIS))
    ground.setColorAt(1.0, QColor(theme.AQUA_DEEP))

    painter.setBrush(QBrush(ground))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(QRectF(8, 8, size - 16, size - 16), 56, 56)

    centre = QPointF(size / 2, size / 2 + 6)

    corners = [
        QPointF(size * 0.26, size * 0.32),
        QPointF(size * 0.76, size * 0.36),
        QPointF(size * 0.50, size * 0.80),
    ]

    # Bonds first, so the atoms sit on top of them.
    bonds = QPen(QColor(255, 255, 255, 200), 11)
    bonds.setCapStyle(Qt.RoundCap)

    painter.setPen(bonds)

    for corner in corners:
        painter.drawLine(centre, corner)

    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor("#ffffff")))
    painter.drawEllipse(centre, 34, 34)

    painter.setBrush(QBrush(QColor(255, 255, 255, 225)))

    for corner in corners:
        painter.drawEllipse(corner, 20, 20)

    painter.end()

    return QIcon(pixmap)

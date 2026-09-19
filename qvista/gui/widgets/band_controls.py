"""
Colour and legend controls for a phonon dispersion.

A strip that sits above the figure and says how the branches are to be
drawn. It holds the choice and nothing else: pressing anything here
emits :attr:`BandControls.changed`, and whoever owns the figure redraws
it. That keeps the widget usable from any window that draws a
dispersion, and keeps the redrawing where the data is.

The general Design panel on every plot can already recolour a figure,
but only series a legend would name -- and a plain band structure has
none, so the whole of it took one colour and no more could be said. The
schemes here give the branches identities first; the Design panel's
palettes then apply to those, which is why the two do not overlap.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from qvista.visualization import phonon_plots as pp


class BandControls(QWidget):
    """How the branches of a dispersion are coloured and legended."""

    #: A control moved. Carries nothing: the reader asks for the values.
    changed = Signal()

    #: The custom-path button was pressed. The window that owns the
    #: figure holds the force constants, so it opens the chooser and
    #: re-runs the band; this widget only says that it was asked for.
    path_requested = Signal()

    def __init__(self, parent=None) -> None:

        super().__init__(parent)

        #: Used by the plain scheme only. The other two colour by what
        #: the branch *is*, so a single colour would mean nothing there.
        self._colour = pp.DEFAULT_BAND_COLOUR

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 0)
        row.setSpacing(8)

        self.scheme_label = QLabel("Bands")

        row.addWidget(self.scheme_label)

        self.scheme = QComboBox()

        for key, label in pp.BAND_SCHEMES.items():
            self.scheme.addItem(label, key)

        self.scheme.setToolTip(
            "Acoustic / optical separates the three lowest branches, "
            "which go to zero at Gamma and carry the sound velocities, "
            "from the rest. One colour per branch is for following a "
            "single branch through a crossing."
        )
        self.scheme.currentIndexChanged.connect(self._scheme_changed)

        row.addWidget(self.scheme)

        self.colour_button = QPushButton("Colour...")
        self.colour_button.setToolTip(
            "The colour the plain scheme draws every branch in."
        )
        self.colour_button.clicked.connect(self._choose_colour)

        row.addWidget(self.colour_button)

        self.legend = QCheckBox("Legend")
        self.legend.setChecked(True)
        self.legend.setToolTip(
            "Dropped automatically where there are more branches than "
            "a legend could usefully name."
        )
        self.legend.toggled.connect(lambda _: self.changed.emit())

        row.addWidget(self.legend)

        row.addSpacing(12)

        # On the figure rather than in a menu. The force constants are
        # already in hand at this point, so changing the path is seconds
        # of work against the hours they took -- and the question "is
        # this the right path" is one you only ask once the dispersion
        # is in front of you.
        self.path_button = QPushButton("Select Custom Path...")
        self.path_button.setToolTip(
            "Choose which high-symmetry points the dispersion runs "
            "through, and redraw along them. The force constants are "
            "reused, so this does not re-run the displacements."
        )
        self.path_button.clicked.connect(self.path_requested.emit)

        row.addWidget(self.path_button)

        row.addStretch(1)

        self._show_colour()
        self._follow_scheme()

    # ------------------------------------------------------------
    # What is on offer
    # ------------------------------------------------------------

    def set_colouring_offered(self, offered: bool) -> None:
        """
        Show or hide the branch-colouring controls.

        The element-projected figure colours by element and legends
        itself, so a branch scheme there would be three controls that do
        nothing. The path button stays either way: which points the
        dispersion runs through is a question the projected figure has
        exactly as much as the plain one.
        """

        for widget in (
            self.scheme_label,
            self.scheme,
            self.colour_button,
            self.legend,
        ):
            widget.setVisible(bool(offered))

    # ------------------------------------------------------------
    # What was chosen
    # ------------------------------------------------------------

    @property
    def values(self) -> dict[str, object]:
        """The arguments the drawing functions take, as keywords."""

        return {
            "scheme": self.scheme.currentData(),
            "colour": self._colour,
            "legend": self.legend.isChecked(),
        }

    # ------------------------------------------------------------

    def _follow_scheme(self) -> None:
        """Only the plain scheme has a colour to pick."""

        self.colour_button.setEnabled(
            self.scheme.currentData() == pp.BAND_SINGLE
        )

    def _scheme_changed(self) -> None:

        self._follow_scheme()

        self.changed.emit()

    def _show_colour(self) -> None:
        """Put the chosen colour on the button, so it is visible."""

        colour = QColor(self._colour)

        if not colour.isValid():
            return

        # Dark swatches need light text on them and light ones dark, or
        # the button's own label disappears into it.
        text = "#ffffff" if colour.lightness() < 140 else "#101010"

        self.colour_button.setStyleSheet(
            f"background-color: {colour.name()}; color: {text};"
        )

    def _choose_colour(self) -> None:

        chosen = QColorDialog.getColor(
            QColor(self._colour), self, "Band colour"
        )

        if not chosen.isValid():
            return

        self._colour = chosen.name()

        self._show_colour()

        self.changed.emit()

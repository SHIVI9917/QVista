"""
Floating plot window.

Plots used to live in a dock tabbed with the console, which forced them
into a short, wide strip -- the worst shape for a Brillouin zone or a
band structure, both of which want roughly square space.

This hosts the same :class:`~qvista.gui.widgets.plot_panel.PlotPanel` in
its own top-level window: large, square, and placed beside the main window
so both are visible at once. It is deliberately modeless, so the
application stays usable while a figure is open.
"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from qvista.gui.dialogs.side_window import PREFERRED_SIZE, SideWindow
from qvista.gui.widgets.plot_panel import PlotPanel


class PlotWindow(SideWindow):
    """A resizable, modeless window containing a plot panel."""

    def __init__(self, parent=None) -> None:

        super().__init__("QVista - Plots", parent, PREFERRED_SIZE)

        self.panel = PlotPanel(self)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Stretch 1, so every spare pixel goes to the figure. Without
        # it the control strip above shares the extra height equally
        # and opens a band of empty window between the controls and the
        # plot -- on a tall window that gap was most of the screen.
        self._layout.addWidget(self.panel, 1)

        #: A strip of controls above the figure, when the figure has
        #: any. See :meth:`set_controls`.
        self.controls = None

    # ------------------------------------------------------------

    @property
    def figure(self):
        """The matplotlib figure being displayed."""

        return self.panel.figure

    def take(self, controls=None):
        """
        Claim the window for a new figure, and hand the figure back.

        One window serves every analysis that draws a single figure, so
        drawing into it is a handover: whatever the last one put above
        the figure has to go with it. Controls left behind would be
        wired to a result that is no longer on screen, and pressing one
        would redraw the previous analysis over the current one.

        Clearing is therefore the default and keeping is the thing that
        has to be asked for, which is the way round that fails safely.
        """

        self.set_controls(controls)

        return self.figure

    def refresh(self) -> None:
        """Repaint after a drawing function has written to the figure."""

        self.panel.refresh()

    # ------------------------------------------------------------

    def set_controls(self, widget) -> None:
        """
        Put a strip of controls above the figure, or clear it.

        This window is reused by every analysis that draws one figure,
        so whatever the last one left behind has to go: controls for a
        phonon dispersion still sitting over a Brillouin zone would be
        wired to a figure that is no longer there.

        Passing the same widget again is a no-op, which is what lets a
        redraw keep the controls the user is currently holding.

        The widget stays the caller's: it is taken out of the layout
        and unparented, never destroyed. Callers keep one and hand it
        back on every redraw, so that the scheme chosen for a phonon
        dispersion survives running something else in between -- and a
        widget deleted here would be a dangling reference the next time
        they did.
        """

        if widget is self.controls:
            return

        if self.controls is not None:
            self._layout.removeWidget(self.controls)
            self.controls.setParent(None)

        self.controls = widget

        if widget is not None:

            # Reparent before inserting: setParent on a widget already
            # in a layout takes it back out again.
            widget.setParent(self)

            # A strip of controls wants its natural height and no more.
            # Left at the default Preferred policy it grows with the
            # window and pushes the figure down.
            widget.setSizePolicy(
                widget.sizePolicy().horizontalPolicy(),
                QSizePolicy.Fixed,
            )

            self._layout.insertWidget(0, widget)

            # setParent(None) hid it on the way out, and Qt remembers
            # an explicit hide across a reparent.
            widget.show()

    # ------------------------------------------------------------


# ------------------------------------------------------------
# Several figures at once
# ------------------------------------------------------------


class FigureWindow(SideWindow):
    """
    A popup holding one or more named figures, one per tab.

    Figures used to sit inside the dialog that produced them, sharing a
    splitter with the controls and the tables. That works until the
    dialog is the size of a laptop screen, at which point the figure gets
    whatever is left over -- typically a strip a couple of hundred pixels
    high, which is unreadable for a band structure and useless for
    anything meant to be exported.

    Moving them out solves it at the root: the figure gets a window of
    its own that can be maximised, moved to a second screen, or left open
    while the dialog behind it is driven again with different settings.
    The dialog keeps the controls and the numbers, which are what it is
    actually shaped for.

    The window is modeless and parented to the dialog. Qt does not block
    a modal dialog's own children, so this stays usable even though every
    analysis dialog in QVista is opened with ``exec()``.
    """

    def __init__(
        self,
        title: str,
        names: Sequence[str] = ("Figure",),
        parent=None,
        preferred_size: int = PREFERRED_SIZE,
    ) -> None:

        super().__init__(title, parent, preferred_size)

        self.names = tuple(names) or ("Figure",)

        self.panels: dict[str, PlotPanel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # One figure needs no tab bar; several do.
        if len(self.names) == 1:

            self.tabs = None

            panel = PlotPanel(self)

            self.panels[self.names[0]] = panel

            layout.addWidget(panel)

        else:

            self.tabs = QTabWidget(self)

            for name in self.names:

                panel = PlotPanel(self)

                self.panels[name] = panel

                self.tabs.addTab(panel, name)

            layout.addWidget(self.tabs)

    # ------------------------------------------------------------
    # Access
    # ------------------------------------------------------------

    def panel_for(self, name: str | None = None) -> PlotPanel:
        """The panel called ``name``, or the first one."""

        if name is None:
            return self.panels[self.names[0]]

        return self.panels[name]

    def figure_for(self, name: str | None = None):
        """The matplotlib figure called ``name``, or the first one."""

        return self.panel_for(name).figure

    def refresh(self, name: str | None = None) -> None:
        """Repaint one figure, or every one of them."""

        if name is None:
            for panel in self.panels.values():
                panel.refresh()
            return

        self.panel_for(name).refresh()

    # ------------------------------------------------------------
    # Showing
    # ------------------------------------------------------------

    def show_figure(self, name: str | None = None, reference=None) -> None:
        """Bring the window up with ``name`` selected."""

        if name is not None and self.tabs is not None:

            index = self.tabs.indexOf(self.panels[name])

            if index >= 0:
                self.tabs.setCurrentIndex(index)

        self.present(reference)


# ------------------------------------------------------------
# The stand-in left behind in the dialog
# ------------------------------------------------------------


class FigurePlaceholder(QWidget):
    """
    What sits in the dialog where the figure used to be.

    A line saying where the figure went and a button that brings it
    back, so a window closed by accident is one click away rather than a
    reason to re-run the analysis.
    """

    def __init__(self, window: FigureWindow, parent=None) -> None:

        super().__init__(parent)

        self.window_ref = window

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.label = QLabel("Plots open in their own window.")
        self.label.setStyleSheet("color: #7f8c8d;")

        self.button = QPushButton("Show plots")
        self.button.clicked.connect(self._show)

        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.button)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    # ------------------------------------------------------------

    def _show(self) -> None:

        self.window_ref.show_figure(reference=self.parent())

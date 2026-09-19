"""
Embedded matplotlib canvas.

QVista's existing :mod:`qvista.visualization` stack is VTK/PyVista and draws
3D scenes only, so there was no 2D plotting surface to reuse. matplotlib is
used here rather than pyqtgraph because phonon figures are a publication
output: the reference exports at dpi 600 with tight bounding boxes, which is
matplotlib's own idiom, and the same figure object serves both the on-screen
canvas and the export.

Each panel carries its own design window, opened from the Design button.
Because the styling is applied in :meth:`PlotPanel.refresh` -- which every
drawing function already calls when it has finished writing to the figure --
a chosen design survives re-running the analysis behind it, and every plot
in the application gets one without its drawing code knowing.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from qvista.visualization import plot_style as ps
from qvista.visualization.phonon_plots import EXPORT_DPI


class PlotPanel(QWidget):
    """
    A matplotlib figure with a navigation toolbar and an export button.

    The figure is owned here and handed to the drawing functions in
    :mod:`qvista.visualization.phonon_plots`, so what is exported is exactly
    what is on screen.
    """

    def __init__(self, parent=None) -> None:

        super().__init__(parent)

        # Imported here so that merely importing the GUI does not pull in
        # matplotlib's Qt backend.
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(7.0, 4.5))

        self.canvas = FigureCanvasQTAgg(self.figure)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        #: How this figure is to look. Starts as no change at all.
        self.style = ps.PlotStyle()

        #: Built the first time the Design button is pressed.
        self.style_window = None

        layout = QVBoxLayout(self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # ----------------------------------------------------------
        # Buttons
        # ----------------------------------------------------------

        buttons = QHBoxLayout()

        self.design_button = QPushButton("Design...")
        self.design_button.setToolTip(
            "Line width, colours, ranges, labels, legend and export "
            "resolution for this figure."
        )
        self.design_button.clicked.connect(self.open_design)

        buttons.addWidget(self.design_button)

        buttons.addStretch(1)

        self.export_button = QPushButton("Export figure...")
        self.export_button.clicked.connect(self.export)

        buttons.addWidget(self.export_button)

        layout.addLayout(buttons)

        self.clear()

    # ------------------------------------------------------------

    def clear(self) -> None:
        """Empty the figure and repaint."""

        self.figure.clear()

        self.canvas.draw_idle()

    def refresh(self) -> None:
        """
        Repaint after a drawing function has written to the figure.

        The design is imposed here rather than at the point it is
        chosen, so it is re-applied every time the figure is redrawn --
        which is what makes a chosen style survive re-running the
        analysis behind it.
        """

        ps.apply(self.figure, self.style)

        self.canvas.draw_idle()

    # ------------------------------------------------------------
    # Design
    # ------------------------------------------------------------

    def open_design(self) -> None:
        """Show the design window for this figure, building it once."""

        if self.style_window is None:

            from qvista.gui.dialogs.style_window import StyleWindow

            title = self.window().windowTitle() or "Plot"

            self.style_window = StyleWindow(
                f"Design - {title.removeprefix('QVista - ')}", self
            )

            self.style_window.changed.connect(self._restyle)
            self.style_window.cleared.connect(self._unstyle)

        self.style_window.present(self.window())

    def _restyle(self, style) -> None:
        """A control moved: re-impose the whole design."""

        # From the original every time rather than on top of the last
        # one, so dragging a slider back and forth cannot accumulate.
        ps.reset(self.figure)

        self.style = style

        self.refresh()

    def _unstyle(self) -> None:
        """Reset was pressed: put the figure back as it was drawn."""

        ps.reset(self.figure)

        self.style = ps.PlotStyle()

        self.canvas.draw_idle()

    # ------------------------------------------------------------

    def export(self) -> None:
        """
        Save the current figure.

        Defaults to dpi 600 with a tight bounding box, which is what the
        reference used for publication figures; the design window can
        change the resolution.
        """

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export figure",
            "phonon.png",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg)",
        )

        if not filename:
            return

        dpi = self.style.export_dpi or EXPORT_DPI

        try:
            self.figure.savefig(
                filename,
                dpi=dpi,
                bbox_inches="tight",
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export failed",
                f"Could not write {Path(filename).name}: {exc}",
            )
            return

        QMessageBox.information(
            self,
            "Figure exported",
            f"Saved to {filename} at {dpi} dpi.",
        )

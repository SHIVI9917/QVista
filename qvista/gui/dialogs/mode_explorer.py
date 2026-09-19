"""
Click the dispersion, watch the mode.

The band structure on the left, the cell it belongs to on the right.
Clicking anywhere on the chart snaps to the nearest computed point,
diagonalises the dynamical matrix there, and plays that normal mode in
the viewer beside it. A dispersion curve says a mode exists at 8.2 THz;
this says what the atoms are doing.

Two details make the difference between a wave and a shuffle.

The cell is repeated. A phonon at the zone boundary moves neighbouring
cells in antiphase, and in a single cell that is invisible -- every mode
looks like a rigid shuffle of the same few atoms. The repetition carries
the ``exp(i q.R)`` factor, so what is on screen is the wave.

The picked point is snapped, not interpolated. Frequencies exist only at
the q-points the band was evaluated at, and inventing one between them
would animate a mode that was never computed.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from qvista.analyses.phonon_modes import (
    DEFAULT_AMPLITUDE,
    DEFAULT_FRAMES,
    modes_at_qpoint,
    travelling_frames,
)
from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import INFO_BUTTON_STYLE
from qvista.models.structure import Structure

#: Milliseconds between frames. Twenty a second is smooth enough for an
#: oscillation and slow enough that a full scene rebuild keeps up.
FRAME_INTERVAL = 50

#: The picked point on the chart.
MARKER_COLOR = "#ff3b6b"
MARKER_SIZE = 90

#: Displacement arrows: bright against both the dark background and the
#: element colours, and long enough to read at a glance. The scale
#: multiplies an excursion of well under an Angstrom up to something
#: comparable with the spheres it comes out of.
ARROW_COLOR = "#ffe14d"
ARROW_SCALE = 5.0

#: Beyond this many atoms the repeated cell stops being a visualisation
#: and starts being a slideshow.
MAX_ANIMATED_ATOMS = 600

#: The readout under the two panes.
READOUT_STYLE = f"{theme.MONO_STYLE} color: {theme.TEXT};"


class ModeExplorer(QDialog):
    """
    Band structure on one side, the mode it describes on the other.

    ``phonon`` is a phonopy object carrying force constants, and
    ``band`` is the dict :mod:`qvista.analyses.phonon` produces --
    distances, frequencies and q-points, segment by segment.
    """

    def __init__(self, phonon, band, force_field: str = "", parent=None):

        super().__init__(parent)

        self.phonon = phonon
        self.band = band
        self.force_field = force_field

        #: The mode being animated, and its frames.
        self.mode = None
        self.frames: list = []
        self.equilibrium = None
        self._frame = 0

        #: Where the picked point sits on the chart.
        self._marker = None

        self.setWindowTitle("Phonon modes")
        fit_to_screen(self, 1100, 640)
        self.resize(1360, 780)

        # Not modal: the main window stays usable, and closing this is
        # the only thing that stops playback.
        self.setModal(False)

        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_INTERVAL)
        self.timer.timeout.connect(self._advance)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ----------------------------------------------------------
        # The two panes
        # ----------------------------------------------------------

        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self._build_chart())
        splitter.addWidget(self._build_viewer())

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 560])

        layout.addWidget(splitter, 1)

        # ----------------------------------------------------------

        self.readout = QLabel()
        self.readout.setStyleSheet(READOUT_STYLE)
        self.readout.setTextFormat(Qt.PlainText)

        layout.addWidget(self.readout)

        layout.addLayout(self._build_controls())

        # ----------------------------------------------------------

        self._draw_band()

        self._show_equilibrium()

        self.readout.setText("Click a point on the dispersion.")

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_chart(self) -> QWidget:
        """
        The clickable dispersion, in a window of its own.

        It stays clickable there -- picking a mode off it is the whole
        point of this window -- and putting it beside the animating cell
        rather than above it means both get a usable amount of space on
        a laptop screen.
        """

        from qvista.gui.dialogs.plot_window import (
            FigurePlaceholder,
            FigureWindow,
        )

        self.figures = FigureWindow(
            "QVista - Phonon Dispersion", ("Dispersion",), self
        )

        self.plot = self.figures.panel_for()

        self.plot.canvas.mpl_connect("button_press_event", self._on_click)

        return FigurePlaceholder(self.figures, self)

    def _build_viewer(self) -> QWidget:
        """The cell, animating."""

        from qvista.visualization.viewer import CrystalViewer

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)

        self.viewer = CrystalViewer(host)

        # No periodic images. This window builds its own repetition,
        # carrying the exp(i q.R) phase each cell is supposed to have,
        # and the boundary tiling on top of that does not: a face
        # duplicate is the home atom's displaced position plus a lattice
        # vector, so it moves in phase with the home cell instead of its
        # own. At the zone boundary neighbouring cells are in antiphase,
        # which is exactly the thing being shown, and the duplicates
        # snapped about while the atoms they copied moved smoothly.
        self.viewer.tile_images = False

        # No cell box. The repeated cell is a viewing aid, not the
        # structure, and its edges cut through the very displacements
        # this window exists to show.
        settings = self.viewer.settings

        settings.show_cell = False

        self.viewer.apply_settings(settings)

        column.addWidget(self.viewer)

        return host

    def _build_controls(self) -> QHBoxLayout:
        """One row: play, how big, how far, and out."""

        row = QHBoxLayout()
        row.setSpacing(10)

        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle)

        row.addWidget(self.play_button)

        row.addWidget(QLabel("Amplitude"))

        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(0.02, 2.0)
        self.amplitude.setSingleStep(0.05)
        self.amplitude.setValue(DEFAULT_AMPLITUDE)
        self.amplitude.setSuffix(" A")
        self.amplitude.valueChanged.connect(self._rebuild)

        row.addWidget(self.amplitude)

        row.addWidget(QLabel("Repeat"))

        self.repeat: list[QSpinBox] = []

        for value in (2, 2, 2):

            box = QSpinBox()
            box.setRange(1, 6)
            box.setValue(value)
            box.setMaximumWidth(56)
            box.valueChanged.connect(self._rebuild)

            row.addWidget(box)

            self.repeat.append(box)

        self.arrows = QCheckBox("Arrows")
        self.arrows.setChecked(True)
        self.arrows.toggled.connect(self._rebuild)

        row.addWidget(self.arrows)

        row.addStretch(1)

        close = QPushButton("Close")
        close.setStyleSheet(INFO_BUTTON_STYLE)
        close.clicked.connect(self.close)

        row.addWidget(close)

        return row

    # ------------------------------------------------------------
    # The chart
    # ------------------------------------------------------------

    def _draw_band(self) -> None:
        """Draw the dispersion, once."""

        from qvista.visualization import phonon_plots

        phonon_plots.draw_band(
            self.plot.figure,
            self.band,
            force_field=self.force_field,
        )

        self.axes = self.plot.figure.axes[0]

        self.figures.show_figure(reference=self)

        self.plot.refresh()

    def _points(self):
        """
        Every computed point, flattened.

        Returns ``(x, y, segment, index, band)`` as parallel arrays. Built
        once and reused, because a click has to search all of them.
        """

        if getattr(self, "_flat", None) is not None:
            return self._flat

        xs, ys, segments, indices, bands = [], [], [], [], []

        for segment, (distances, frequencies) in enumerate(
            zip(self.band["distances"], self.band["frequencies"])
        ):

            distances = np.asarray(distances, dtype=float)
            frequencies = np.asarray(frequencies, dtype=float)

            for band_index in range(frequencies.shape[1]):

                xs.append(distances)
                ys.append(frequencies[:, band_index])

                segments.append(np.full(distances.shape, segment))
                indices.append(np.arange(len(distances)))
                bands.append(np.full(distances.shape, band_index))

        self._flat = (
            np.concatenate(xs),
            np.concatenate(ys),
            np.concatenate(segments),
            np.concatenate(indices),
            np.concatenate(bands),
        )

        return self._flat

    def _on_click(self, event) -> None:
        """Snap to the nearest computed point and animate it."""

        if event.inaxes is not self.axes or event.xdata is None:
            return

        # The toolbar owns the click while zoom or pan is armed.
        if self.plot.toolbar.mode:
            return

        xs, ys, segments, indices, bands = self._points()

        # Compared in pixels, not in data units: the axes are THz against
        # a path length in inverse Angstrom, so "nearest" in data units
        # would mean whichever axis happens to have smaller numbers.
        here = np.column_stack([xs, ys])

        screen = self.axes.transData.transform(here)

        offset = screen - np.array([event.x, event.y])

        nearest = int(np.argmin((offset ** 2).sum(axis=1)))

        self._pick(
            int(segments[nearest]),
            int(indices[nearest]),
            int(bands[nearest]),
        )

    def _pick(self, segment: int, index: int, band_index: int) -> None:
        """Diagonalise at one point of the path and load the mode."""

        qpoint = np.asarray(
            self.band["qpoints"][segment], dtype=float
        )[index]

        try:
            modes = modes_at_qpoint(self.phonon, qpoint)

        except Exception as exc:
            self.readout.setText(f"Could not compute modes here: {exc}")
            return

        if not 0 <= band_index < len(modes):
            return

        self.mode = modes[band_index]

        self._mark(segment, index, band_index)

        self._rebuild()

        self.play_button.setEnabled(True)

        self.play()

    def _mark(self, segment: int, index: int, band_index: int) -> None:
        """Put a dot on the point that was picked."""

        x = float(np.asarray(self.band["distances"][segment])[index])
        y = float(
            np.asarray(self.band["frequencies"][segment])[index, band_index]
        )

        if self._marker is not None:

            try:
                self._marker.remove()
            except Exception:
                pass

        self._marker = self.axes.scatter(
            [x],
            [y],
            s=MARKER_SIZE,
            facecolors="none",
            edgecolors=MARKER_COLOR,
            linewidths=2.0,
            zorder=5,
        )

        self.plot.refresh()

    # ------------------------------------------------------------
    # The animation
    # ------------------------------------------------------------

    def _rebuild(self) -> None:
        """Regenerate the frames for the current mode and settings."""

        if self.mode is None:
            return

        primitive = self.phonon.primitive

        repeat = [box.value() for box in self.repeat]

        natoms = len(primitive) * repeat[0] * repeat[1] * repeat[2]

        if natoms > MAX_ANIMATED_ATOMS:
            self.readout.setText(
                f"{natoms} atoms is too many to animate smoothly. "
                f"Reduce Repeat."
            )
            self.stop()
            return

        frames, source, lattice, equilibrium = travelling_frames(
            np.asarray(primitive.cell, dtype=float),
            np.asarray(primitive.positions, dtype=float),
            primitive.masses,
            self.mode.eigenvector,
            self.mode.qpoint,
            repeat=repeat,
            frames=DEFAULT_FRAMES,
            amplitude=self.amplitude.value(),
        )

        self.frames = frames

        #: Where the atoms sit at rest, which is what the arrows are
        #: measured from -- not frame zero, which is already displaced.
        self.equilibrium = np.asarray(equilibrium, dtype=float)

        self._frame = 0

        symbols = [str(s) for s in primitive.symbols]

        # Loaded at rest, and the bonds fixed there. Every frame after
        # this one moves those same bonds instead of asking again which
        # atoms are close enough to be bonded -- otherwise the structure
        # comes apart and reassembles once a period.
        self._load_cell(
            [symbols[int(i)] for i in source],
            lattice,
            self.equilibrium,
        )

        self.viewer.freeze_bonds()

        self._describe()

        self._show_frame(frames[0])

    def _load_cell(self, symbols, lattice, positions) -> None:
        """
        Put the repeated cell into the viewer, and frame it.

        The camera is refitted only when the cell itself changed --
        picking a different mode or nudging the amplitude keeps whatever
        view the user had rotated to, which they would otherwise lose on
        every click.
        """

        from ase import Atoms

        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=lattice,
            pbc=True,
        )

        changed = getattr(self, "_loaded", None) != len(atoms)

        self.viewer.load_structure(Structure(atoms))

        if changed:
            self._loaded = len(atoms)

            self.viewer.reset_camera()
            self.viewer.render()

    def _show_equilibrium(self) -> None:
        """Show the cell at rest, before anything has been picked."""

        primitive = self.phonon.primitive

        repeat = [box.value() for box in self.repeat]

        # A zero eigenvector displaces nothing, so this is the same code
        # path as an animated mode with the motion switched off.
        _frames, source, lattice, equilibrium = travelling_frames(
            np.asarray(primitive.cell, dtype=float),
            np.asarray(primitive.positions, dtype=float),
            primitive.masses,
            np.zeros(3 * len(primitive), dtype=complex),
            (0.0, 0.0, 0.0),
            repeat=repeat,
            frames=1,
            amplitude=0.0,
        )

        symbols = [str(s) for s in primitive.symbols]

        self._load_cell(
            [symbols[int(i)] for i in source], lattice, equilibrium
        )

        self.viewer.freeze_bonds()

    def _describe(self) -> None:
        """The one line under the panes."""

        if self.mode is None:
            return

        qx, qy, qz = self.mode.qpoint

        marker = "  imaginary" if self.mode.is_imaginary else ""

        self.readout.setText(
            f"q = ({qx:+.3f} {qy:+.3f} {qz:+.3f})     "
            f"band {self.mode.band_index + 1:>2}     "
            f"{self.mode.frequency:8.3f} THz{marker}"
        )

    # ------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------

    def _toggle(self) -> None:

        if self.timer.isActive():
            self.stop()
        else:
            self.play()

    def play(self) -> None:
        """Start animating."""

        if not self.frames:
            return

        self.play_button.setText("Pause")
        self.timer.start()

    def stop(self) -> None:
        """Stop animating, leaving the cell where it is."""

        if self.timer.isActive():
            self.timer.stop()

        self.play_button.setText("Play")

    def _advance(self) -> None:
        """Push the next frame."""

        if not self.frames:
            return

        self._show_frame(self.frames[self._frame])

        self._frame = (self._frame + 1) % len(self.frames)

    def _show_frame(self, positions) -> None:
        """Move the atoms, and redraw the arrows if they are on."""

        self.viewer.update_positions(positions)

        self._draw_arrows(positions)

    def _draw_arrows(self, positions) -> None:
        """
        An arrow per atom, along its instantaneous displacement.

        The direction is what the eye reads a mode by -- which atoms
        move together, which move against each other -- and it is much
        harder to see from the motion alone at small amplitude.
        """

        name = "mode_arrows"

        try:
            self.viewer.remove_actor(name, render=False)
        except Exception:
            pass

        if not self.arrows.isChecked() or self.equilibrium is None:
            self.viewer.render()
            return

        import pyvista as pv

        equilibrium = self.equilibrium

        offsets = np.asarray(positions, dtype=float) - equilibrium

        largest = float(np.abs(offsets).max())

        if largest < 1e-9:
            self.viewer.render()
            return

        cloud = pv.PolyData(equilibrium)
        cloud["vectors"] = offsets

        try:
            glyphs = cloud.glyph(
                orient="vectors",
                scale="vectors",
                factor=ARROW_SCALE,
                geom=pv.Arrow(
                    tip_length=0.3,
                    tip_radius=0.12,
                    shaft_radius=0.04,
                ),
            )

            # glyph() carries the displacement through as scalars, and a
            # mesh with active scalars is coloured by them -- which is
            # why an arrow asked to be yellow comes out green.
            glyphs.set_active_scalars(None)

            self.viewer.add_mesh(
                glyphs,
                color=ARROW_COLOR,
                name=name,
                pickable=False,
                show_scalar_bar=False,
                render=False,
            )

        except Exception:
            pass

        self.viewer.render()

    # ------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Stop the timer, and let go of the embedded render window."""

        self.stop()

        try:
            self.viewer.close()
        except Exception:
            pass

        super().closeEvent(event)

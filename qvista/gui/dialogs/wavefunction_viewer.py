"""
A wavefunction drawn over the structure it belongs to.

The window VESTA gives you for a PARCHG: the cell with its atoms, and an
isosurface of the wavefunction through it, with the level adjustable.

The one thing worth saying about the drawing is the sign.  A real
wavefunction is positive in some lobes and negative in others, and the
node between them is most of what the picture is for -- so a signed
component is contoured twice, at ``+level`` and ``-level``, and the two
surfaces are given different colours.  ``psi squared`` has no sign and
gets a single surface.

Grid handling
-------------
The grid is wrapped by one layer before contouring, repeating the first
plane at the far face.  VASP's grid is periodic and stores only one
period, so without the wrap every lobe crossing a cell boundary is
sliced flat where it meets the face.  The extra plane costs one row of
points and closes those surfaces.

The points are placed by multiplying fractional coordinates into the
lattice rather than using an axis-aligned grid, so a triclinic cell
comes out sheared as it should be rather than forced into a box.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.models.structure import Structure

#: Lobe colours.  Warm for the positive lobe and cool for the negative,
#: which is the convention every wavefunction figure uses.
POSITIVE_COLOR = "#ffd23f"
NEGATIVE_COLOR = "#4d8bff"

#: A single unsigned surface -- a density, not a wavefunction.
DENSITY_COLOR = "#ffd23f"

#: Actor names, so a redraw replaces its own surfaces and nothing else.
POSITIVE_ACTOR = "psi-positive"
NEGATIVE_ACTOR = "psi-negative"

#: Where the level slider starts, as a percentage of the largest value.
#:
#: Low levels look generous and are almost useless in a cell with more
#: than a few atoms: at 15% of the peak in a 42-atom cell, only about a
#: sixth of the drawn surface sits on the atom the state is actually on,
#: and the orbital being looked for is buried in everything else.
DEFAULT_LEVEL_PERCENT = 25

#: Shown in the isolation box when the whole cell is drawn.
WHOLE_CELL = "Whole cell"

#: How far from a selected atom to keep the surface, in angstrom.
DEFAULT_RADIUS = 2.0

#: Contouring cost grows with the cube of the grid, and a WAVECAR from a
#: large cell can be 200 points a side.  Beyond this the grid is strided
#: down for drawing; the file that gets written is never touched.
MAX_GRID_POINTS = 4_000_000


def _stride_for(shape: tuple[int, ...]) -> int:
    """How much to thin a grid by so it stays quick to contour."""

    stride = 1

    while np.prod([max(1, n // stride) for n in shape]) > MAX_GRID_POINTS:
        stride += 1

    return stride


def distance_field(
    points: np.ndarray,
    centres: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """
    Distance from each point to the nearest of ``centres``, through the
    periodic boundary.

    An atom near a face is closer to points on the far side of the cell
    than a straight subtraction suggests, so the difference is taken in
    fractional coordinates and wrapped to the nearest image before being
    measured.
    """

    lattice = np.asarray(lattice, dtype=float)
    inverse = np.linalg.inv(lattice)

    nearest = None

    for centre in np.atleast_2d(np.asarray(centres, dtype=float)):

        offset = (points - centre) @ inverse
        offset -= np.round(offset)

        distance = np.linalg.norm(offset @ lattice, axis=1)

        nearest = (
            distance if nearest is None else np.minimum(nearest, distance)
        )

    if nearest is None:
        return np.zeros(len(points))

    return nearest


def wrapped_grid(data: np.ndarray, lattice: np.ndarray, stride: int = 1):
    """
    Build the PyVista grid a contour can be taken from.

    ``data`` is indexed ``[x][y][z]`` as VASP writes it.  The returned
    grid carries one extra plane on each axis, holding a copy of the
    first, so surfaces close across the periodic boundary.
    """

    import pyvista as pv

    if stride > 1:
        data = data[::stride, ::stride, ::stride]

    # Repeat the first plane at the far face.  np.take with mode="wrap"
    # does the three axes without special-casing the last index.
    for axis in range(3):
        data = np.take(
            data,
            range(data.shape[axis] + 1),
            axis=axis,
            mode="wrap",
        )

    nx, ny, nz = data.shape

    # Fractional coordinates run 0 to 1 inclusive now that the far face
    # is present, so the divisor is the original count, not the padded
    # one.
    axes = [
        np.arange(n) / float(n - 1) for n in (nx, ny, nz)
    ]

    fx, fy, fz = np.meshgrid(*axes, indexing="ij")

    fractional = np.column_stack(
        [
            fx.ravel(order="F"),
            fy.ravel(order="F"),
            fz.ravel(order="F"),
        ]
    )

    grid = pv.StructuredGrid()

    grid.points = fractional @ np.asarray(lattice, dtype=float)
    grid.dimensions = [nx, ny, nz]

    # Fortran order to match how the points were laid out above.
    grid["psi"] = data.ravel(order="F")

    return grid


class WavefunctionViewer(QDialog):
    """
    One window per set of components, switchable from inside.

    ``components`` maps a label to ``(data, signed)``.  Passing the
    whole set rather than one array means the three things the Orbital
    dialog can write are all reachable without reopening the window.
    """

    def __init__(
        self,
        components: dict[str, tuple[np.ndarray, bool]],
        structure,
        heading: str = "",
        isolate: str | None = None,
        note: str | None = None,
        parent=None,
    ) -> None:

        super().__init__(parent)

        if not components:
            raise ValueError("No wavefunction components to show.")

        self.components = components
        self.structure = structure

        self.symbols = [site.specie.symbol for site in structure]

        self.elements: list[str] = []

        for symbol in self.symbols:
            if symbol not in self.elements:
                self.elements.append(symbol)

        self.wanted_element = (
            isolate if isolate in self.elements else None
        )

        self.setWindowTitle("Wavefunction")
        fit_to_screen(self, 900, 680)

        layout = QVBoxLayout(self)

        if heading:

            label = QLabel(heading)
            label.setStyleSheet(
                f"color: {theme.TEXT}; font-weight: 600;"
            )

            layout.addWidget(label)

        if note:

            warning = QLabel(note)
            warning.setStyleSheet(theme.WARNING_STYLE)
            warning.setWordWrap(True)

            layout.addWidget(warning)

        layout.addWidget(self._build_viewer(), 1)
        layout.addLayout(self._build_controls())

        self._load_structure()

        self._redraw()

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_viewer(self) -> QWidget:

        from qvista.visualization.viewer import CrystalViewer

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)

        self.viewer = CrystalViewer(host)

        column.addWidget(self.viewer)

        return host

    def _build_controls(self) -> QVBoxLayout:
        """Component, level, opacity, and what else to draw."""

        outer = QVBoxLayout()

        top = QHBoxLayout()
        top.setSpacing(10)

        top.addWidget(QLabel("Component"))

        self.component = QComboBox()
        self.component.addItems(list(self.components))
        self.component.currentTextChanged.connect(self._on_component)

        top.addWidget(self.component)

        top.addWidget(QLabel("Isolevel"))

        self.level = QSlider(Qt.Horizontal)
        self.level.setRange(1, 95)
        self.level.setValue(DEFAULT_LEVEL_PERCENT)
        self.level.valueChanged.connect(self._on_level)

        top.addWidget(self.level, 1)

        self.level_value = QLabel("")
        self.level_value.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.level_value.setMinimumWidth(150)

        top.addWidget(self.level_value)

        outer.addLayout(top)

        # Isolating the element the table is about is the difference
        # between seeing an orbital and seeing a cell full of lobes. In
        # anything bigger than a handful of atoms most of the surface
        # belongs to atoms the question was not about.
        middle = QHBoxLayout()
        middle.setSpacing(10)

        middle.addWidget(QLabel("Show"))

        self.isolate = QComboBox()
        self.isolate.addItem(WHOLE_CELL)
        self.isolate.addItems(self.elements)

        if self.wanted_element is not None:
            self.isolate.setCurrentText(self.wanted_element)

        self.isolate.currentTextChanged.connect(lambda _: self._redraw())

        middle.addWidget(self.isolate)

        middle.addWidget(QLabel("within"))

        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.5, 12.0)
        self.radius.setSingleStep(0.25)
        self.radius.setValue(DEFAULT_RADIUS)
        self.radius.setSuffix(" A")
        self.radius.valueChanged.connect(lambda _: self._redraw())

        middle.addWidget(self.radius)

        self.kept = QLabel("")
        self.kept.setStyleSheet(f"color: {theme.TEXT_DIM};")

        middle.addWidget(self.kept)

        middle.addStretch(1)

        outer.addLayout(middle)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        bottom.addWidget(QLabel("Opacity"))

        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.05, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(0.6)
        self.opacity.valueChanged.connect(lambda _: self._redraw())

        bottom.addWidget(self.opacity)

        self.show_bonds = QCheckBox("Bonds")
        self.show_bonds.setChecked(True)
        self.show_bonds.toggled.connect(self._on_bonds)

        bottom.addWidget(self.show_bonds)

        self.show_cell = QCheckBox("Cell")
        self.show_cell.setChecked(True)
        self.show_cell.toggled.connect(self._on_cell)

        bottom.addWidget(self.show_cell)

        self.legend = QLabel("")
        self.legend.setStyleSheet(f"color: {theme.TEXT_DIM};")

        bottom.addWidget(self.legend)

        bottom.addStretch(1)

        reset = QPushButton("Reset view")
        reset.clicked.connect(self._reset_view)

        bottom.addWidget(reset)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        bottom.addWidget(close)

        outer.addLayout(bottom)

        return outer

    # ------------------------------------------------------------
    # The scene
    # ------------------------------------------------------------

    def _load_structure(self) -> None:
        """Put the atoms in, once."""

        from ase import Atoms

        atoms = Atoms(
            symbols=[site.specie.symbol for site in self.structure],
            positions=[site.coords for site in self.structure],
            cell=np.asarray(self.structure.lattice.matrix, dtype=float),
            pbc=True,
        )

        self.viewer.load_structure(Structure(atoms))

        self.viewer.reset_camera()

    def _current(self) -> tuple[np.ndarray, bool]:

        return self.components[self.component.currentText()]

    def _on_component(self, _text: str) -> None:

        self._redraw()

    def _on_level(self, _value: int) -> None:

        self._redraw()

    def _on_bonds(self, shown: bool) -> None:

        settings = self.viewer.settings
        settings.show_bonds = shown

        self.viewer.apply_settings(settings)
        self.viewer.render()

    def _on_cell(self, shown: bool) -> None:

        settings = self.viewer.settings
        settings.show_cell = shown

        self.viewer.apply_settings(settings)
        self.viewer.render()

    def _reset_view(self) -> None:

        self.viewer.reset_camera()
        self.viewer.render()

    def _redraw(self) -> None:
        """Recontour at the current level and put the surfaces back."""

        data, signed = self._current()

        for name in (POSITIVE_ACTOR, NEGATIVE_ACTOR):
            self.viewer.remove_actor(name, render=False)

        peak = float(np.abs(data).max())

        if peak <= 0.0:
            self.level_value.setText("the grid is flat")
            self.viewer.render()
            return

        level = peak * self.level.value() / 100.0

        self.level_value.setText(
            f"{level:.3e}  of  {peak:.3e}"
        )

        stride = _stride_for(data.shape)

        lattice = np.asarray(self.structure.lattice.matrix, dtype=float)

        grid = wrapped_grid(data, lattice, stride=stride)

        self._attach_distance(grid, lattice)

        opacity = self.opacity.value()

        if signed:

            self._add(grid, level, POSITIVE_COLOR, POSITIVE_ACTOR, opacity)
            self._add(grid, -level, NEGATIVE_COLOR, NEGATIVE_ACTOR, opacity)

            self.legend.setText("yellow +   blue -")

        else:

            self._add(grid, level, DENSITY_COLOR, POSITIVE_ACTOR, opacity)

            self.legend.setText("")

        self.viewer.render()

    def _attach_distance(self, grid, lattice: np.ndarray) -> None:
        """
        Tag every grid point with its distance to the chosen element.

        Carried as a scalar rather than applied to the volume, because
        blanking the data before contouring would cut a surface across
        the sphere's own boundary.  Contouring first and discarding the
        far parts afterwards leaves the orbital's own shape intact.
        """

        chosen = self.isolate.currentText()

        if chosen == WHOLE_CELL or chosen not in self.elements:
            self.kept.setText("")
            return

        centres = np.array(
            [
                site.coords
                for site, symbol in zip(self.structure, self.symbols)
                if symbol == chosen
            ],
            dtype=float,
        )

        grid["dist"] = distance_field(
            np.asarray(grid.points, dtype=float), centres, lattice
        )

        # The scalar the contour is cut from has to stay the active one.
        grid.set_active_scalars("psi")

        self.kept.setText(f"{len(centres)} {chosen} atom(s)")

    def _add(
        self,
        grid,
        level: float,
        color: str,
        name: str,
        opacity: float,
    ) -> None:
        """Contour at one level and show it, if anything is there."""

        surface = grid.contour([level], scalars="psi")

        if surface.n_points == 0:
            return

        # The contour interpolates every array it was cut from, so the
        # distance tag is on it and the far parts can be dropped.
        if "dist" in surface.point_data:

            surface = surface.threshold(
                self.radius.value(),
                scalars="dist",
                invert=True,
            )

            if surface.n_points == 0:
                return

        # The contour carries the scalar it was cut from, and a mesh
        # with active scalars is coloured by them -- which would ignore
        # the colour asked for here.
        surface.set_active_scalars(None)

        self.viewer.add_mesh(
            surface,
            color=color,
            opacity=opacity,
            name=name,
            smooth_shading=True,
            show_scalar_bar=False,
            reset_camera=False,
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name

        try:
            self.viewer.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass

        super().closeEvent(event)

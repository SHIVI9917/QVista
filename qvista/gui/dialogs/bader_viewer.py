"""
A Bader analysis drawn over the structure it came from.

Two things are worth looking at rather than reading off a table, and this
window shows both.

The first is *which* atoms took the charge.  Every atom is drawn coloured
by its net charge on a scale that is symmetric about zero, so red and
blue mean gained and lost rather than large and small, and a row of
equivalent atoms that is not actually equivalent stands out at a glance.

The second is *where the boundaries fell*.  A Bader volume is bounded by
a zero-flux surface, and that surface is a real object with a shape; the
volume box takes any set of atoms and draws the surface around them.
Seeing it is the only way to tell an atom whose charge is odd because the
chemistry is odd from one whose charge is odd because a noisy reference
density put the boundary in the wrong place.

The surface is drawn by contouring the label grid at a half.  That grid
holds one integer per point, so the contour lands exactly between the
points on either side of the boundary, and the result is faceted at the
resolution of the FFT grid.  That is honest: it is where the partition
actually put the surface, not a smoothed illustration of it.
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
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.services import bader as bd

#: What an atom's colour can mean, and the label shown for it.
COLOUR_BY: dict[str, str] = {
    "net": "Net charge  (ZVAL - electrons)",
    "charge": "Charge  (electrons in the volume)",
    "volume": "Bader volume",
    "min_dist": "Distance to the surface",
}

#: Net charge is signed and its zero is meaningful, so it gets a
#: diverging map centred on zero.  The others are magnitudes.
DIVERGING = "coolwarm"
SEQUENTIAL = "viridis"

#: Actor names, so each redraw replaces its own and nothing else.
ATOM_ACTOR = "bader-atoms"
SURFACE_ACTOR = "bader-surface"
CELL_ACTOR = "bader-cell"
LABEL_ACTOR = "bader-values"

#: ``add_point_labels`` registers what it adds under the given name with
#: a suffix, one actor for the text and another for the points, so taking
#: the labels down again means naming both.
LABEL_ACTORS = (
    LABEL_ACTOR,
    f"{LABEL_ACTOR}-labels",
    f"{LABEL_ACTOR}-points",
)

#: Sphere radius as a fraction of the covalent radius.  Small enough that
#: the colours read as separate atoms rather than a solid block.
DEFAULT_SCALE = 0.45

#: The colour of the Bader surface.  Deliberately neutral: the atoms are
#: carrying the colour that means something.
SURFACE_COLOR = "#8fe3d8"


def cell_edges(lattice: np.ndarray):
    """The twelve edges of the cell, as one line mesh."""

    import pyvista as pv

    lattice = np.asarray(lattice, dtype=float)

    corners = np.array(
        [
            [x, y, z]
            for x in (0.0, 1.0)
            for y in (0.0, 1.0)
            for z in (0.0, 1.0)
        ]
    ) @ lattice

    # Corner n differs from corner m in exactly one bit when they are
    # joined by an edge, which is cheaper to say than to enumerate.
    lines = []

    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:
                lines.extend([2, a, b])

    mesh = pv.PolyData(corners)
    mesh.lines = np.array(lines)

    return mesh


class BaderViewer(QDialog):
    """The picture half of the Bader tool."""

    def __init__(self, result: bd.BaderResult, heading: str = "", parent=None):

        super().__init__(parent)

        if not result.atoms:
            raise ValueError("The Bader result has no atoms to draw.")

        self.result = result
        self.structure = result.structure
        self.lattice = np.asarray(
            result.structure.lattice.matrix, dtype=float
        )

        self.setWindowTitle("Bader - visualization")
        fit_to_screen(self, 940, 720)

        layout = QVBoxLayout(self)

        if heading:

            title = QLabel(heading)
            title.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600;")

            layout.addWidget(title)

        layout.addWidget(self._build_viewer(), 1)
        layout.addLayout(self._build_controls())

        self._draw_cell()
        self._draw_atoms()

        self.viewer.reset_camera()
        self.viewer.render()

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_viewer(self) -> QWidget:

        from pyvistaqt import QtInteractor

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)

        self.viewer = QtInteractor(host)
        self.viewer.set_background(theme.BG_BOTTOM, top=theme.BG_TOP)
        self.viewer.add_axes()

        column.addWidget(self.viewer)

        return host

    def _build_controls(self) -> QVBoxLayout:

        outer = QVBoxLayout()

        top = QHBoxLayout()
        top.setSpacing(10)

        top.addWidget(QLabel("Colour by"))

        self.colour_by = QComboBox()

        for key, text in COLOUR_BY.items():
            self.colour_by.addItem(text, key)

        self.colour_by.currentIndexChanged.connect(
            lambda _: self._draw_atoms()
        )

        top.addWidget(self.colour_by, 1)

        top.addWidget(QLabel("Atom size"))

        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.1, 1.5)
        self.scale.setSingleStep(0.05)
        self.scale.setValue(DEFAULT_SCALE)
        self.scale.valueChanged.connect(lambda _: self._draw_atoms())

        top.addWidget(self.scale)

        self.show_values = QCheckBox("Values")
        self.show_values.setChecked(False)
        self.show_values.toggled.connect(lambda _: self._draw_atoms())

        top.addWidget(self.show_values)

        self.show_cell = QCheckBox("Cell")
        self.show_cell.setChecked(True)
        self.show_cell.toggled.connect(self._on_cell)

        top.addWidget(self.show_cell)

        outer.addLayout(top)

        middle = QHBoxLayout()
        middle.setSpacing(10)

        middle.addWidget(QLabel("Bader volume of"))

        self.wanted = QLineEdit()
        self.wanted.setPlaceholderText(
            f"atom numbers, e.g. 1 or 1-6 or 3,7 "
            f"(1 to {len(self.result.atoms)})"
        )
        self.wanted.returnPressed.connect(self._draw_surface)

        middle.addWidget(self.wanted, 1)

        draw = QPushButton("Draw")
        draw.clicked.connect(self._draw_surface)

        middle.addWidget(draw)

        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear_surface)

        middle.addWidget(clear)

        middle.addWidget(QLabel("Opacity"))

        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.05, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(0.45)
        self.opacity.valueChanged.connect(lambda _: self._redraw_surface())

        middle.addWidget(self.opacity)

        outer.addLayout(middle)

        bottom = QHBoxLayout()

        self.legend = QLabel("")
        self.legend.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.legend.setWordWrap(True)

        bottom.addWidget(self.legend, 1)

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

    def _values(self) -> tuple[np.ndarray, str, bool]:
        """The number each atom is coloured by, its name, and its sign."""

        key = self.colour_by.currentData() or "net"

        values = np.array(
            [float(getattr(atom, key)) for atom in self.result.atoms]
        )

        return values, COLOUR_BY[key], key == "net"

    def _draw_atoms(self) -> None:
        """One sphere per atom, coloured by the chosen number."""

        import pyvista as pv

        from qvista.visualization.radii import covalent_radius

        values, name, signed = self._values()

        points = np.array(
            [atom.coords for atom in self.result.atoms], dtype=float
        )

        cloud = pv.PolyData(points)
        cloud["value"] = values
        cloud["radius"] = np.array(
            [
                covalent_radius(atom.symbol) * self.scale.value()
                for atom in self.result.atoms
            ]
        )

        # Scaling by an explicit array rather than by the active scalars,
        # which would size the atoms by their charge and make the picture
        # say two things at once.
        cloud.set_active_scalars("radius")

        spheres = cloud.glyph(
            geom=pv.Sphere(theta_resolution=24, phi_resolution=24),
            scale="radius",
            orient=False,
            factor=1.0,
        )

        spheres.set_active_scalars("value")

        if signed:
            reach = float(np.abs(values).max()) or 1.0
            limits = (-reach, reach)
            palette = DIVERGING
        else:
            low, high = float(values.min()), float(values.max())
            limits = (low, high if high > low else low + 1.0)
            palette = SEQUENTIAL

        self.viewer.add_mesh(
            spheres,
            scalars="value",
            cmap=palette,
            clim=limits,
            name=ATOM_ACTOR,
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": name,
                "color": theme.TEXT,
                "n_labels": 5,
            },
            reset_camera=False,
        )

        self._draw_values(points, values)

        if signed:
            self.legend.setText(
                f"Red gives electrons away, blue takes them. "
                f"Scale runs +/-{limits[1]:.3f} electrons, centred on "
                "neutral."
            )
        else:
            self.legend.setText(
                f"{name}: {limits[0]:.3f} to {limits[1]:.3f}."
            )

        self.viewer.render()

    def _draw_values(self, points: np.ndarray, values: np.ndarray) -> None:
        """The number itself, over each atom, when asked for."""

        for name in LABEL_ACTORS:
            self.viewer.remove_actor(name, render=False)

        if not self.show_values.isChecked():
            return

        self.viewer.add_point_labels(
            points,
            [f"{value:+.2f}" for value in values],
            name=LABEL_ACTOR,
            font_size=11,
            text_color=theme.TEXT,
            shape=None,
            show_points=False,
            always_visible=True,
            reset_camera=False,
        )

    def _draw_cell(self) -> None:

        self.viewer.add_mesh(
            cell_edges(self.lattice),
            color=theme.BORDER_STRONG,
            line_width=2,
            name=CELL_ACTOR,
            show_scalar_bar=False,
            reset_camera=False,
        )

    def _on_cell(self, shown: bool) -> None:

        if shown:
            self._draw_cell()
        else:
            self.viewer.remove_actor(CELL_ACTOR, render=False)

        self.viewer.render()

    # ------------------------------------------------------------
    # Bader volumes
    # ------------------------------------------------------------

    def _clear_surface(self) -> None:

        self.viewer.remove_actor(SURFACE_ACTOR, render=False)
        self.viewer.render()

    def _redraw_surface(self) -> None:
        """Reapply the opacity without recontouring."""

        if self.wanted.text().strip():
            self._draw_surface()

    def _draw_surface(self) -> None:
        """Contour the label grid around the atoms named in the box."""

        from qvista.gui.dialogs.wavefunction_viewer import wrapped_grid

        text = self.wanted.text().strip()

        if not text:
            self._clear_surface()
            return

        try:
            chosen = bd.parse_selection(text, len(self.result.atoms))
            field = bd.basin_surface(self.result, chosen)

        except bd.BaderError as exc:
            QMessageBox.warning(self, "Bader volume", str(exc))
            return

        grid = wrapped_grid(field, self.lattice)

        surface = grid.contour([0.5], scalars="psi")

        if surface.n_points == 0:
            QMessageBox.information(
                self,
                "Bader volume",
                "Those atoms have no grid points between them and the "
                "rest of the cell, so there is no surface to draw.",
            )
            return

        surface.set_active_scalars(None)

        self.viewer.add_mesh(
            surface,
            color=SURFACE_COLOR,
            opacity=self.opacity.value(),
            name=SURFACE_ACTOR,
            smooth_shading=False,
            show_scalar_bar=False,
            reset_camera=False,
        )

        held = sum(
            self.result.atoms[index - 1].volume for index in chosen
        )

        self.legend.setText(
            f"Zero-flux surface around {len(chosen)} atom(s), enclosing "
            f"{held:.2f} A^3. Faceted at the FFT grid, which is where the "
            "partition actually put it."
        )

        self.viewer.render()

    # ------------------------------------------------------------

    def _reset_view(self) -> None:

        self.viewer.reset_camera()
        self.viewer.render()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name

        try:
            self.viewer.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass

        super().closeEvent(event)

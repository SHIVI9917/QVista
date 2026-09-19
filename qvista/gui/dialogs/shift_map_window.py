"""
Electronegativity shift map.

The panel answers "what about this atom"; this answers "what about all of
them at once". Every atom in the displayed cell is drawn coloured by how
far its electronegativity sits from its surroundings, so the places where
electron density is being pulled or given up show up as colour rather
than as a number to be read one site at a time.

Two readings of "shift" are offered, because they answer different
questions:

* **Against the cell** -- χ(atom) − χ_eq, the Sanderson mean. Depends
  only on the element, so it maps composition.
* **Against its own shell** -- the net Δχ the site faces from the atoms
  bonded to it. Depends on where the atom actually sits, so two atoms of
  the same element in different environments come out differently. This
  is the one worth looking at in a defect, a surface or a mixed site.

Colour runs blue → red through the rainbow, anchored so that the middle
is always zero shift: blue gives electron density up, red pulls it in --
the same sense as an electrostatic potential map.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from pyvistaqt import QtInteractor

from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import INFO_BUTTON_STYLE
from qvista.services.electronegativity import (
    cell_electronegativity,
    polyhedron,
    symmetric_limits,
)
from qvista.visualization.coordination import (
    coordination_shell,
    neighbour_pool,
)
from qvista.visualization.navigation import enable_zoom_to_cursor
from qvista.visualization.radii import covalent_radius, vdw_radius
from qvista.visualization.settings import Representation

#: Blue through green to red. The user asked for a rainbow; the limits
#: are pinned symmetrically about zero so the green middle always means
#: "no shift" rather than "middle of whatever range this structure had".
COLORMAP = "jet"

#: Atoms with no electronegativity on any scale.
UNDEFINED_COLOR = "#79839c"

#: Context geometry, light enough to read against the dark viewport.
CELL_COLOR = "#5c6785"
BOND_COLOR = "#7d879e"

#: The scalar bar draws its text in black by default, which disappears.
BAR_TEXT_COLOR = "#e9edf8"

#: Shift measured against the whole cell, or against the atom's own
#: coordination shell.
MODE_CELL = "Against the cell mean (element)"
MODE_SHELL = "Against its own coordination shell (site)"

#: Monospace, for the per-element table.
REPORT_STYLE = theme.MONO_STYLE

#: The value written on each atom. Drawn on a dark plate because the
#: colours underneath it run the whole rainbow, and no single text colour
#: is legible against all of them.
LABEL_FONT_SIZE = 12
LABEL_TEXT_COLOR = "white"
LABEL_PLATE_COLOR = "#101010"
LABEL_PLATE_OPACITY = 0.65


def cell_wireframe(structure) -> pv.PolyData:
    """The unit cell box as line segments, for orientation."""

    vectors = np.asarray(structure.cell.array, dtype=float)

    corners = np.array(
        [
            i * vectors[0] + j * vectors[1] + k * vectors[2]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    )

    # Corner n has bits (i, j, k); an edge joins corners differing in one.
    edges = [
        (a, b)
        for a in range(8)
        for b in range(a + 1, 8)
        if bin(a ^ b).count("1") == 1
    ]

    lines = np.hstack([[2, a, b] for a, b in edges])

    return pv.PolyData(corners, lines=lines)


def bond_lines(bonds) -> pv.PolyData | None:
    """The drawn bonds as line segments, so the structure stays readable."""

    if not bonds:
        return None

    points = []
    lines = []

    for index, bond in enumerate(bonds):

        points.append(bond.atom_i.position)
        points.append(bond.atom_j.position)

        lines.extend([2, 2 * index, 2 * index + 1])

    return pv.PolyData(np.asarray(points, dtype=float), lines=np.asarray(lines))


class ShiftMapWindow(QDialog):
    """A 3D map of the electronegativity shift across the structure."""

    def __init__(self, viewer, parent=None) -> None:

        super().__init__(parent)

        self.viewer = viewer

        self.setWindowTitle("Electronegativity shift map")
        fit_to_screen(self, 820, 640)
        self.resize(940, 800)

        layout = QVBoxLayout(self)

        # ----------------------------------------------------------
        # Controls
        # ----------------------------------------------------------

        controls = QHBoxLayout()

        controls.addWidget(QLabel("Shift measured:"))

        self.mode_box = QComboBox()
        self.mode_box.addItems([MODE_CELL, MODE_SHELL])
        self.mode_box.currentTextChanged.connect(lambda _t: self.rebuild())

        controls.addWidget(self.mode_box, 1)

        # The colour says which way and roughly how far; the number says
        # how far exactly, sign included, which is the part a colour bar
        # cannot be read off precisely.
        self.values_box = QCheckBox("write the value on each atom")
        self.values_box.setChecked(True)
        self.values_box.toggled.connect(lambda _c: self.rebuild())

        controls.addWidget(self.values_box)

        refresh = QPushButton("Refresh")
        refresh.setStyleSheet(INFO_BUTTON_STYLE)
        refresh.clicked.connect(self.rebuild)

        controls.addWidget(refresh)

        layout.addLayout(controls)

        # ----------------------------------------------------------
        # Its own plotter -- the main viewer keeps its element colours
        # ----------------------------------------------------------

        self.plotter = QtInteractor(self)
        self.plotter.set_background(
            viewer.settings.background_color,
            top=viewer.settings.background_top,
        )
        self.plotter.setMinimumHeight(380)

        layout.addWidget(self.plotter, 1)

        # The legend must keep its room: given a stretch of 1 the plotter
        # takes every spare pixel and pushes the legend and the buttons
        # off the bottom of the window.
        self.summary = QLabel()
        self.summary.setStyleSheet(REPORT_STYLE)
        self.summary.setWordWrap(True)
        self.summary.setMinimumHeight(120)
        self.summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout.addWidget(self.summary, 0)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        buttons.addWidget(close_button)

        layout.addLayout(buttons)

        self._built = False

    # ------------------------------------------------------------

    def showEvent(self, event) -> None:
        """
        Build on first show, not in the constructor.

        A second VTK render window has no usable GL context until its
        widget is realised. Drawing into it before that fails every
        ``wglMakeCurrent`` and fills the log with errors, because the
        main viewer's context is the current one.
        """

        super().showEvent(event)

        if not self._built:
            self._built = True

            # Same wheel behaviour as the main viewer: this window is
            # the one people zoom into, and two 3D views that navigate
            # differently is worse than either behaviour on its own.
            enable_zoom_to_cursor(self.plotter)

            self.rebuild()

    def closeEvent(self, event) -> None:
        """Release the render window; a leaked one keeps a GL context."""

        try:
            self.plotter.close()
        except Exception:
            pass

        super().closeEvent(event)

    # ------------------------------------------------------------

    def _atoms(self) -> list:
        """The atoms currently displayed in the main viewer."""

        scene_data = getattr(self.viewer.scene, "scene_data", None)

        if scene_data is None:
            return []

        return list(scene_data.visible_atoms)

    def _radius(self, symbol: str) -> float:
        """Match the main viewer's drawn size, so the two look alike."""

        settings = self.viewer.settings

        if settings.representation == Representation.SPACE_FILL:
            base = vdw_radius(symbol)
        else:
            base = covalent_radius(symbol)

        return settings.atom_radius_scale * base

    def _shifts(self, atoms) -> tuple[list, object]:
        """
        The shift for every atom, and the scale it is on.

        ``None`` for an atom with no electronegativity on any scale; those
        are drawn grey rather than dropped, so the picture still shows
        where they are.
        """

        symbols = self.viewer.structure.atoms.get_chemical_symbols()

        cell = cell_electronegativity(symbols)

        if self.mode_box.currentText() == MODE_CELL:
            return [cell.deviation(atom.symbol) for atom in atoms], cell.scale

        pool = neighbour_pool(
            self.viewer.scene,
            self.viewer.structure,
            self.viewer.settings.boundary,
        )

        bond_scale = self.viewer.scene.visible_bond_builder.scale

        values = []
        scale = cell.scale

        for atom in atoms:

            result = polyhedron(
                atom.symbol,
                coordination_shell(atom, pool, bond_scale),
            )

            values.append(result.net_delta)

            if result.scale is not None:
                scale = result.scale

        return values, scale

    # ------------------------------------------------------------

    def rebuild(self) -> None:
        """Redraw the map for the current mode."""

        self.plotter.clear()

        atoms = self._atoms()

        if not atoms:
            self.summary.setText("No structure displayed.")
            self.plotter.render()
            return

        values, scale = self._shifts(atoms)

        defined = [
            (atom, value)
            for atom, value in zip(atoms, values)
            if value is not None
        ]

        unknown = [
            atom
            for atom, value in zip(atoms, values)
            if value is None
        ]

        # Context first, so the spheres draw over it.
        self.plotter.add_mesh(
            cell_wireframe(self.viewer.structure),
            color=CELL_COLOR,
            line_width=1,
        )

        bonds = bond_lines(
            self.viewer.scene.scene_data.graph.bonds
            if self.viewer.scene.scene_data.graph is not None
            else []
        )

        if bonds is not None:
            self.plotter.add_mesh(bonds, color=BOND_COLOR, line_width=2)

        if defined:
            self._add_atoms(defined, scale)

        if unknown:
            self._add_unknown(unknown)

        if defined and self.values_box.isChecked():
            self._add_values(defined)

        self.plotter.reset_camera()
        self.plotter.render()

        self.summary.setText(self._summary(defined, unknown, scale))

    def _add_atoms(self, defined, scale) -> None:
        """One glyphed mesh, coloured by the shift."""

        cloud = pv.PolyData(
            np.asarray([atom.position for atom, _ in defined], dtype=float)
        )

        cloud["shift"] = np.asarray(
            [value for _, value in defined], dtype=float
        )

        cloud["radius"] = np.asarray(
            [self._radius(atom.symbol) for atom, _ in defined], dtype=float
        )

        glyphs = cloud.glyph(
            geom=pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20),
            scale="radius",
            orient=False,
        )

        unit = f" ({scale.unit})" if scale is not None and scale.unit else ""

        self.plotter.add_mesh(
            glyphs,
            scalars="shift",
            cmap=COLORMAP,
            clim=symmetric_limits([value for _, value in defined]),
            smooth_shading=True,
            # Vertical, down the right-hand edge. The default horizontal
            # bar runs the width of the viewport and its end labels fall
            # off it.
            scalar_bar_args={
                "title": f"shift{unit}",
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.10,
                "width": 0.05,
                "height": 0.80,
                "n_labels": 5,
                "fmt": "%+.2f",
                "color": BAR_TEXT_COLOR,
                "title_font_size": 14,
                "label_font_size": 12,
            },
        )

    def _add_values(self, defined) -> None:
        """
        The signed shift, written on every atom that has one.

        Always with its sign: the interesting half of the map is the
        atoms giving density up, and a bare ``1.08`` beside a bare
        ``1.38`` says nothing about which is which.

        ``always_visible`` keeps a label in front of its own sphere.
        Without it the text sits at the sphere's centre and is buried by
        the geometry it belongs to.
        """

        self.plotter.add_point_labels(
            np.asarray(
                [atom.position for atom, _ in defined], dtype=float
            ),
            [f"{value:+.2f}" for _, value in defined],
            font_size=LABEL_FONT_SIZE,
            text_color=LABEL_TEXT_COLOR,
            shape="rounded_rect",
            shape_color=LABEL_PLATE_COLOR,
            shape_opacity=LABEL_PLATE_OPACITY,
            fill_shape=True,
            always_visible=True,
            show_points=False,
            name="shift_values",
        )

    def _add_unknown(self, atoms) -> None:
        """Atoms with no value at all, drawn grey and out of the scale."""

        cloud = pv.PolyData(
            np.asarray([atom.position for atom in atoms], dtype=float)
        )

        cloud["radius"] = np.asarray(
            [self._radius(atom.symbol) for atom in atoms], dtype=float
        )

        glyphs = cloud.glyph(
            geom=pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=16),
            scale="radius",
            orient=False,
        )

        self.plotter.add_mesh(
            glyphs, color=UNDEFINED_COLOR, smooth_shading=True
        )

    # ------------------------------------------------------------

    def _summary(self, defined, unknown, scale) -> str:
        """Legend, range, and the per-element averages."""

        if not defined:
            return "No atom here has an electronegativity on any scale."

        values = [value for _, value in defined]

        unit = f" {scale.unit}" if scale is not None and scale.unit else ""

        by_element: dict[str, list[float]] = {}

        for atom, value in defined:
            by_element.setdefault(atom.symbol, []).append(value)

        rows = []

        for symbol in sorted(
            by_element, key=lambda s: -sum(by_element[s]) / len(by_element[s])
        ):

            group = by_element[symbol]
            mean = sum(group) / len(group)

            spread = (
                ""
                if max(group) - min(group) < 5e-3
                else f"   ({min(group):+.2f} to {max(group):+.2f})"
            )

            rows.append(
                f"  {symbol:<3} x{len(group):<3} {mean:+.2f}{unit}{spread}"
            )

        lines = [
            f"blue = gives electron density up      "
            f"red = pulls electron density in       "
            f"({scale} scale)",
            f"range {min(values):+.2f} to {max(values):+.2f}{unit}, "
            f"colour bar centred on zero",
            "",
        ] + rows

        if unknown:
            lines.append(
                f"  grey: {len(unknown)} atom(s) with no value on any scale"
            )

        return "\n".join(lines)

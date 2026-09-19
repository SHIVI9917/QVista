"""
DDEC6 charges and bond orders drawn over the structure.

Atoms are coloured by net charge on a scale symmetric about neutral;
bonds are drawn as tubes whose thickness and colour both follow the bond
order, so a strong bond is thick and bright and the exchange-hole tail
that Chargemol prints down to 0.001 stays thin and dark until the
threshold cuts it off.

Bonds to a periodic image are drawn to where that image actually is,
which is why a bond can leave the cell.
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
    QVBoxLayout,
    QWidget,
)

from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.bader_viewer import cell_edges
from qvista.services import ddec as dd

#: What an atom's colour can mean.
COLOUR_BY: dict[str, str] = {
    "charge": "Net charge",
    "spin": "Spin moment",
    "sbo": "Sum of bond orders",
}

DIVERGING = "coolwarm"
SEQUENTIAL = "viridis"
BOND_MAP = "plasma"

ATOM_ACTOR = "ddec-atoms"
BOND_ACTOR = "ddec-bonds"
CELL_ACTOR = "ddec-cell"
LABEL_ACTOR = "ddec-values"

LABEL_ACTORS = (
    LABEL_ACTOR,
    f"{LABEL_ACTOR}-labels",
    f"{LABEL_ACTOR}-points",
)

DEFAULT_SCALE = 0.4

#: Tube radius of the weakest drawn bond, in angstrom, and how much
#: thicker the strongest one gets.
BOND_RADIUS = 0.04
BOND_RADIUS_FACTOR = 3.0


class DdecViewer(QDialog):
    """The picture half of the DDEC6 tool."""

    def __init__(self, result: dd.DdecResult, heading: str = "", parent=None):

        super().__init__(parent)

        if not result.atoms:
            raise ValueError("The DDEC result has no atoms to draw.")

        self.result = result
        self.lattice = result.lattice

        self.setWindowTitle("DDEC6 - visualization")
        fit_to_screen(self, 940, 720)

        layout = QVBoxLayout(self)

        if heading:

            title = QLabel(heading)
            title.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600;")

            layout.addWidget(title)

        layout.addWidget(self._build_viewer(), 1)
        layout.addLayout(self._build_controls())

        if self.lattice is not None:
            self._draw_cell()

        self._draw_atoms()
        self._draw_bonds()

        self.viewer.reset_camera()
        self.viewer.render()

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

        top.addWidget(QLabel("Colour"))

        self.colour_by = QComboBox()

        for key, text in COLOUR_BY.items():

            if key == "spin" and all(a.spin is None for a in self.result.atoms):
                continue

            if key == "sbo" and all(a.sbo is None for a in self.result.atoms):
                continue

            self.colour_by.addItem(text, key)

        self.colour_by.currentIndexChanged.connect(lambda _: self._draw_atoms())

        top.addWidget(self.colour_by)

        top.addWidget(QLabel("Atom size"))

        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.1, 1.5)
        self.scale.setSingleStep(0.05)
        self.scale.setValue(DEFAULT_SCALE)
        self.scale.valueChanged.connect(lambda _: self._draw_atoms())

        top.addWidget(self.scale)

        top.addWidget(QLabel("Bond order >="))

        self.minimum = QDoubleSpinBox()
        self.minimum.setRange(0.0, 3.0)
        self.minimum.setSingleStep(0.05)
        self.minimum.setDecimals(3)
        self.minimum.setValue(dd.DEFAULT_MIN_ORDER)
        self.minimum.valueChanged.connect(lambda _: self._draw_bonds())

        top.addWidget(self.minimum)

        self.show_values = QCheckBox("Values")
        self.show_values.toggled.connect(lambda _: self._draw_atoms())

        top.addWidget(self.show_values)

        self.show_cell = QCheckBox("Cell")
        self.show_cell.setChecked(True)
        self.show_cell.setEnabled(self.lattice is not None)
        self.show_cell.toggled.connect(self._on_cell)

        top.addWidget(self.show_cell)

        top.addStretch(1)

        outer.addLayout(top)

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
    # Atoms
    # ------------------------------------------------------------

    def _draw_atoms(self) -> None:

        import pyvista as pv

        from qvista.visualization.radii import covalent_radius

        key = self.colour_by.currentData() or "charge"

        values = np.array(
            [
                float(getattr(atom, key) or 0.0)
                for atom in self.result.atoms
            ]
        )

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
        cloud.set_active_scalars("radius")

        spheres = cloud.glyph(
            geom=pv.Sphere(theta_resolution=24, phi_resolution=24),
            scale="radius",
            orient=False,
            factor=1.0,
        )
        spheres.set_active_scalars("value")

        # Charge and spin are signed and their zero means something;
        # a sum of bond orders is a magnitude.
        signed = key in ("charge", "spin")

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
                "title": COLOUR_BY[key],
                "color": theme.TEXT,
                "n_labels": 5,
            },
            reset_camera=False,
        )

        for name in LABEL_ACTORS:
            self.viewer.remove_actor(name, render=False)

        if self.show_values.isChecked():

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

        self.viewer.render()

    # ------------------------------------------------------------
    # Bonds
    # ------------------------------------------------------------

    def _draw_bonds(self) -> None:
        """One tube per bond, thickness and colour following the order."""

        import pyvista as pv

        self.viewer.remove_actor(BOND_ACTOR, render=False)

        bonds = dd.unique_bonds(
            dd.bonds_above(self.result, self.minimum.value())
        )

        if not bonds:
            self.legend.setText("No bonds above that order.")
            self.viewer.render()
            return

        coords = np.array(
            [atom.coords for atom in self.result.atoms], dtype=float
        )

        points: list[np.ndarray] = []
        lines: list[int] = []
        orders: list[float] = []

        for bond in bonds:

            start = coords[bond.a - 1]
            end = coords[bond.b - 1]

            # The bond Chargemol means runs to the image it named, not
            # to the atom sitting in the reference cell.
            if self.lattice is not None:
                end = end + np.asarray(bond.image, dtype=float) @ self.lattice

            lines.extend([2, len(points), len(points) + 1])

            points.extend([start, end])
            orders.extend([bond.order, bond.order])

        mesh = pv.PolyData(np.array(points))
        mesh.lines = np.array(lines)
        mesh["order"] = np.array(orders)

        lowest = float(min(orders))
        highest = float(max(orders))

        # Varying the radius by the scalars needs the scalars to vary.
        # In a symmetric crystal every bond of a kind has the same order,
        # and asking vtk to scale by a range of zero warns and gives back
        # a degenerate tube.
        if highest - lowest > 1e-9:
            tubes = mesh.tube(
                radius=BOND_RADIUS,
                scalars="order",
                radius_factor=BOND_RADIUS_FACTOR,
                n_sides=12,
            )
        else:
            tubes = mesh.tube(radius=BOND_RADIUS, n_sides=12)

        self.viewer.add_mesh(
            tubes,
            scalars="order",
            cmap=BOND_MAP,
            clim=(lowest, highest if highest > lowest else lowest + 1e-6),
            name=BOND_ACTOR,
            smooth_shading=True,
            show_scalar_bar=False,
            reset_camera=False,
        )

        self.legend.setText(
            f"{len(bonds)} bonds at order >= {self.minimum.value():.3f}   "
            f"strongest {highest:.3f}   "
            f"shortest {min(b.length for b in bonds):.3f} A"
        )

        self.viewer.render()

    # ------------------------------------------------------------

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

        if self.lattice is None:
            return

        if shown:
            self._draw_cell()
        else:
            self.viewer.remove_actor(CELL_ACTOR, render=False)

        self.viewer.render()

    def _reset_view(self) -> None:

        self.viewer.reset_camera()
        self.viewer.render()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name

        try:
            self.viewer.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass

        super().closeEvent(event)

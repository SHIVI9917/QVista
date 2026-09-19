"""
Bond strengths drawn over the structure.

Every interaction LOBSTER reported becomes a tube between the two atoms,
with its thickness and colour following the bond strength, so which
bonds hold the structure together is a thing you can see rather than
sort a table by.  Atoms are coloured by the total strength of the bonds
they take part in, which is what separates a well bound site from an
under-coordinated one.

Bonds to a periodic image are drawn to where that image is, so a bond
that crosses a face leaves the cell.
"""

from __future__ import annotations

import numpy as np

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
from qvista.services import cohp as ch

ATOM_ACTOR = "cohp-atoms"
BOND_ACTOR = "cohp-bonds"
CELL_ACTOR = "cohp-cell"
LABEL_ACTOR = "cohp-values"

LABEL_ACTORS = (
    LABEL_ACTOR,
    f"{LABEL_ACTOR}-labels",
    f"{LABEL_ACTOR}-points",
)

BOND_MAP = "plasma"
ATOM_MAP = "viridis"

DEFAULT_SCALE = 0.35
BOND_RADIUS = 0.05
BOND_RADIUS_FACTOR = 3.0


class CohpViewer(QDialog):
    """The picture half of the bonding tool."""

    def __init__(self, result: ch.CohpResult, heading: str = "", parent=None):

        super().__init__(parent)

        if result.structure is None:
            raise ValueError(
                "There is no structure in the run folder, so the bonds "
                "cannot be placed. LOBSTER writes POSCAR.lobster.vasp "
                "beside its output."
            )

        drawable = [
            bond
            for bond in result.bonds
            if bond.a_site is not None and bond.b_site is not None
        ]

        if not drawable:
            raise ValueError(
                "None of the interactions carries site numbers, which come "
                f"from {result.kind.curves}. Without that file the bonds "
                "cannot be matched to atoms."
            )

        self.result = result
        self.drawable = drawable
        self.structure = result.structure
        self.lattice = np.asarray(result.structure.lattice.matrix, dtype=float)

        self.setWindowTitle(f"{result.kind.name} - visualization")
        fit_to_screen(self, 940, 720)

        layout = QVBoxLayout(self)

        if heading:

            title = QLabel(heading)
            title.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600;")

            layout.addWidget(title)

        layout.addWidget(self._build_viewer(), 1)
        layout.addLayout(self._build_controls())

        self._draw_cell()
        self._draw()

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

        strengths = [bond.strength for bond in self.drawable]

        top.addWidget(QLabel(f"{self.result.kind.strength_label} >="))

        self.minimum = QDoubleSpinBox()
        self.minimum.setRange(min(strengths) - 1.0, max(strengths) + 1.0)
        self.minimum.setSingleStep(0.05)
        self.minimum.setDecimals(3)
        # Opens on the strong half: LOBSTER reports every pair inside its
        # cutoff, and drawing all of them is a ball of string.
        self.minimum.setValue(
            round(float(np.median(strengths)), 3)
        )
        self.minimum.valueChanged.connect(lambda _: self._draw())

        top.addWidget(self.minimum)

        top.addWidget(QLabel("length <="))

        self.max_length = QDoubleSpinBox()
        self.max_length.setRange(0.5, 20.0)
        self.max_length.setSingleStep(0.1)
        self.max_length.setValue(
            round(max(bond.length for bond in self.drawable) + 0.1, 2)
        )
        self.max_length.setSuffix(" A")
        self.max_length.valueChanged.connect(lambda _: self._draw())

        top.addWidget(self.max_length)

        top.addWidget(QLabel("Pair"))

        self.pair = QComboBox()
        self.pair.addItem("All", None)

        for pair in self.result.pairs():
            self.pair.addItem(pair, pair)

        self.pair.currentIndexChanged.connect(lambda _: self._draw())

        top.addWidget(self.pair)

        top.addWidget(QLabel("Atom size"))

        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.1, 1.5)
        self.scale.setSingleStep(0.05)
        self.scale.setValue(DEFAULT_SCALE)
        self.scale.valueChanged.connect(lambda _: self._draw())

        top.addWidget(self.scale)

        self.show_cell = QCheckBox("Cell")
        self.show_cell.setChecked(True)
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

    def _selected(self) -> list[ch.Bond]:

        return ch.select(
            self.drawable,
            max_length=self.max_length.value(),
            min_strength=self.minimum.value(),
            pairs=[self.pair.currentData()] if self.pair.currentData() else None,
        )

    def _draw(self) -> None:

        bonds = self._selected()

        self._draw_atoms(bonds)
        self._draw_bonds(bonds)

        self.viewer.render()

    def _draw_atoms(self, bonds: list[ch.Bond]) -> None:
        """Spheres coloured by the total strength of the bonds shown."""

        import pyvista as pv

        from qvista.visualization.radii import covalent_radius

        totals = np.zeros(len(self.structure))

        for bond in bonds:
            totals[bond.a_site] += bond.strength
            totals[bond.b_site] += bond.strength

        points = np.array(
            [site.coords for site in self.structure], dtype=float
        )

        cloud = pv.PolyData(points)
        cloud["value"] = totals
        cloud["radius"] = np.array(
            [
                covalent_radius(site.specie.symbol) * self.scale.value()
                for site in self.structure
            ]
        )
        cloud.set_active_scalars("radius")

        spheres = cloud.glyph(
            geom=pv.Sphere(theta_resolution=22, phi_resolution=22),
            scale="radius",
            orient=False,
            factor=1.0,
        )
        spheres.set_active_scalars("value")

        high = float(totals.max()) or 1.0

        self.viewer.add_mesh(
            spheres,
            scalars="value",
            cmap=ATOM_MAP,
            clim=(0.0, high),
            name=ATOM_ACTOR,
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": f"total {self.result.kind.strength_label} per atom",
                "color": theme.TEXT,
                "n_labels": 5,
            },
            reset_camera=False,
        )

    def _draw_bonds(self, bonds: list[ch.Bond]) -> None:

        import pyvista as pv

        self.viewer.remove_actor(BOND_ACTOR, render=False)

        if not bonds:
            self.legend.setText("No bonds pass those filters.")
            return

        coords = np.array(
            [site.coords for site in self.structure], dtype=float
        )

        points: list[np.ndarray] = []
        lines: list[int] = []
        values: list[float] = []

        for bond in bonds:

            start = coords[bond.a_site]
            end = (
                coords[bond.b_site]
                + np.asarray(bond.translation, dtype=float) @ self.lattice
            )

            lines.extend([2, len(points), len(points) + 1])

            points.extend([start, end])
            values.extend([bond.strength, bond.strength])

        mesh = pv.PolyData(np.array(points))
        mesh.lines = np.array(lines)
        mesh["strength"] = np.array(values)

        low, high = float(min(values)), float(max(values))

        # vtk warns and gives a degenerate tube if asked to scale a radius
        # by a range of zero, which happens whenever one bond is drawn.
        if high - low > 1e-9:
            tubes = mesh.tube(
                radius=BOND_RADIUS,
                scalars="strength",
                radius_factor=BOND_RADIUS_FACTOR,
                n_sides=12,
            )
        else:
            tubes = mesh.tube(radius=BOND_RADIUS, n_sides=12)

        self.viewer.add_mesh(
            tubes,
            scalars="strength",
            cmap=BOND_MAP,
            clim=(low, high if high > low else low + 1e-6),
            name=BOND_ACTOR,
            smooth_shading=True,
            show_scalar_bar=False,
            reset_camera=False,
        )

        unit = f" {self.result.kind.unit}" if self.result.kind.unit else ""

        self.legend.setText(
            f"{len(bonds)} bonds   "
            f"{self.result.kind.strength_label} {low:.3f} to {high:.3f}{unit}   "
            f"length {min(b.length for b in bonds):.3f} to "
            f"{max(b.length for b in bonds):.3f} A"
        )

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

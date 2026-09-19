"""
Scene manager for QVista.

Coordinates all renderers used to display a crystal structure.
"""


from __future__ import annotations


import time
from qvista.visualization.scene_data import SceneData

from qvista.visualization.atom_renderer import AtomRenderer
from qvista.visualization.bond_renderer import BondRenderer
from qvista.visualization.cell_renderer import CellRenderer
from qvista.visualization.label_renderer import LabelRenderer
from qvista.visualization.periodic_image_generator import (
    PeriodicImageGenerator,
)
from qvista.visualization.polyhedron_builder import (
    PolyhedronBuilder,
)
from qvista.visualization.polyhedron_renderer import (
    PolyhedronRenderer,
)
from qvista.visualization.visible_bond_builder import (
    VisibleBondBuilder,
)


#: Per-frame timing output. Off by default: these prints sit in the redraw
#: hot path and each one is a synchronous write, which is measurable when
#: dragging a control that redraws continuously. Set True while profiling.
DEBUG_TIMING = False


def report_timing(label: str, t0: float) -> None:
    """Print a timing line, when timing output is enabled."""

    if DEBUG_TIMING:
        print(f"{label:<16}: {(time.perf_counter() - t0) * 1000:.1f} ms")


class Scene:
    """
    Scene manager.
    """

    def __init__(self) -> None:

        # ----------------------------------------------------------
        # Renderers
        # ----------------------------------------------------------

        self.atom_renderer = AtomRenderer()
        self.cell_renderer = CellRenderer()
        self.bond_renderer = BondRenderer()
        self.polyhedron_builder = PolyhedronBuilder()
        self.polyhedron_renderer = PolyhedronRenderer()
        self.label_renderer = LabelRenderer()

        # ----------------------------------------------------------
        # Utilities
        # ----------------------------------------------------------

        self.periodic_generator = PeriodicImageGenerator()
        self.visible_bond_builder = VisibleBondBuilder()

        #: When set, the bonds to draw regardless of distance. See
        #: :mod:`qvista.visualization.visible_bond_builder`: an animation
        #: needs its connectivity to stay put while the atoms move.
        self.fixed_bonds: list | None = None

        # ----------------------------------------------------------
        # SceneData
        # ----------------------------------------------------------

        self.scene_data: SceneData | None = None

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(
        self,
        viewer,
        structure,
    ) -> None:

        settings = viewer.settings

        # ----------------------------------------------------------
        # Build SceneData
        # ----------------------------------------------------------

        self.scene_data = SceneData(
            structure=structure,
            boundary=settings.boundary,
            settings=settings,
        )

        # ----------------------------------------------------------
        # Unit cell
        # ----------------------------------------------------------

        if settings.show_cell:

            self.cell_renderer.draw(
                viewer,
                structure,
            )

        # ----------------------------------------------------------
        # Visible atoms
        # ----------------------------------------------------------

        t0 = time.perf_counter()

        self.scene_data.visible_atoms = (
            self.periodic_generator.generate(
                structure,
                settings.boundary,
                settings.origin,
                tile=getattr(viewer, "tile_images", True),
            )
        )
        
        report_timing("Periodic images", t0)
        # ----------------------------------------------------------
        # Bond graph
        # ----------------------------------------------------------
        t0 = time.perf_counter()

        self.scene_data.graph = (
            self.visible_bond_builder.build(
                self.scene_data.visible_atoms,
                pairs=self.fixed_bonds,
                rules=settings.bond_rules,
            )
        )

        report_timing("Bond graph", t0)

        # ----------------------------------------------------------
        # Polyhedra
        # ----------------------------------------------------------

        t0 = time.perf_counter()

        if settings.show_polyhedra:

            neighbor_pool = self.periodic_generator.generate_padded(
                structure,
                settings.boundary,
                origin=settings.origin,
            )

            self.scene_data.polyhedra = (
                self.polyhedron_builder.build(
                    self.scene_data.visible_atoms,
                    settings,
                    neighbor_pool=neighbor_pool,
                )
            )

        report_timing("Polyhedra build", t0)

        # ----------------------------------------------------------
        # Render
        # ----------------------------------------------------------

        if settings.show_bonds:

            t0 = time.perf_counter()

            self.bond_renderer.draw(
                viewer,
                self.scene_data.graph,
            )

            report_timing("Bonds render", t0)



        t0 = time.perf_counter()

        self.atom_renderer.draw(
            viewer,
            self.scene_data,
        )

        report_timing("Atoms render", t0)

        # The timing print used to sit in the else branch while t0 was set
        # in the if branch, so it reported the atom-render time under the
        # "Poly render" label and never timed the polyhedra at all.
        t0 = time.perf_counter()

        if settings.show_polyhedra:

            self.polyhedron_renderer.draw(
                viewer,
                self.scene_data.polyhedra,
            )

        else:

            self.polyhedron_renderer.hide(viewer)

        report_timing("Poly render", t0)


        if settings.show_labels:

            t0 = time.perf_counter()

            self.label_renderer.draw(
                viewer,
                self.scene_data.visible_atoms,
            )

            report_timing("Labels render", t0)


    # ------------------------------------------------------------------
    # Rendering Engine V2 API
    # ------------------------------------------------------------------

    def update_geometry(
        self,
        viewer,
        structure,
    ) -> None:
        """
        Update all geometry.

        Currently this falls back to a complete redraw.
        Later it will update only modified meshes.
        """

        # Temporary implementation
        viewer.cache.remove_atoms(viewer)
        viewer.cache.remove_bonds(viewer)
        viewer.cache.remove_labels(viewer)
        viewer.cache.remove_cell(viewer)

# Do NOT remove polyhedron actors
        self.draw(
            viewer,
            structure,
        )

    # ------------------------------------------------------------------

    def update_visibility(
        self,
        viewer,
    ) -> None:
        """
        Update actor visibility only.

        Placeholder for Rendering Engine V2.
        """

        pass

    # ------------------------------------------------------------------

    def update_colors(
        self,
        viewer,
    ) -> None:
        """
        Update colors only.

        Placeholder for Rendering Engine V2.
        """

        pass

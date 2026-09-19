"""
Unit cell renderer.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from qvista.models.structure import Structure


class CellRenderer:
    """Draw the crystallographic unit cell."""

    def draw(self, viewer, structure: Structure) -> None:
        """
        Draw the crystallographic unit cell.
        """

        if not viewer.settings.show_cell:
            return

        # ----------------------------------------------------------
        # Cell vectors
        # ----------------------------------------------------------

        cell = structure.atoms.cell.array

        a = cell[0]
        b = cell[1]
        c = cell[2]

        origin = np.zeros(3)

        # ----------------------------------------------------------
        # Eight cell vertices
        # ----------------------------------------------------------

        p0 = origin
        p1 = a
        p2 = b
        p3 = c

        p4 = a + b
        p5 = a + c
        p6 = b + c
        p7 = a + b + c

        points = np.array(
            [
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
            ]
        )

        # ----------------------------------------------------------
        # Twelve edges
        #
        # PolyData line format:
        #
        # [2, point1, point2]
        # ----------------------------------------------------------

        lines = np.array(
            [
                2, 0, 1,
                2, 0, 2,
                2, 0, 3,

                2, 1, 4,
                2, 1, 5,

                2, 2, 4,
                2, 2, 6,

                2, 3, 5,
                2, 3, 6,

                2, 4, 7,
                2, 5, 7,
                2, 6, 7,
            ],
            dtype=np.int64,
        )

        mesh = pv.PolyData()
        mesh.points = points
        mesh.lines = lines
        
        actor = viewer.add_mesh(
            mesh,
            color=viewer.settings.cell_color,
            line_width=viewer.settings.cell_line_width,
            render_lines_as_tubes=False,
            name="unit_cell",
            render=False,
        )

        viewer.cache.set_cell_actor(
            actor,
        )

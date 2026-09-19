"""
Bond renderer for QVista.

Renders all visible bonds as a single line mesh.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv


class BondRenderer:
    """
    Render all visible bonds.
    """

    def draw(
        self,
        viewer,
        graph,
    ) -> None:

        if not graph.bonds:
            return

        points = []
        lines = []

        point_id = 0

        # ----------------------------------------------------------
        # Build PolyData
        # ----------------------------------------------------------

        for bond in graph.bonds:

            p1 = np.asarray(
                bond.atom_i.position,
                dtype=float,
            )

            p2 = np.asarray(
                bond.atom_j.position,
                dtype=float,
            )

            if np.linalg.norm(p2 - p1) < 1e-8:
                continue

            points.append(p1)
            points.append(p2)

            lines.extend(
                [
                    2,
                    point_id,
                    point_id + 1,
                ]
            )

            point_id += 2

        if point_id == 0:
            return

        mesh = pv.PolyData()

        mesh.points = np.asarray(points)

        mesh.lines = np.asarray(
            lines,
            dtype=np.int64,
        )

        actor = viewer.add_mesh(
            mesh,
            color="lightgray",
            line_width=3,
            render_lines_as_tubes=True,
            smooth_shading=True,
            name="bonds",
            render=False,
        )

        viewer.cache.add_bond_actor(
            "bonds",
            actor,
        )

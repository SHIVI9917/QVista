"""
Render coordination polyhedra.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

from qvista.visualization.colors import get_color


class PolyhedronRenderer:
    """
    Render coordination polyhedra.

    Actors are persistent. They are created only once and their
    mesh geometry is updated on subsequent redraws.
    """

    def __init__(self) -> None:

        self._actors: dict[str, object] = {}

        self._meshes: dict[str, pv.PolyData] = {}

    # --------------------------------------------------------------
    # Draw
    # --------------------------------------------------------------

    def draw(
        self,
        viewer,
        polyhedra,
    ) -> None:

        settings = viewer.settings

        # ----------------------------------------------------------
        # Hide everything first.
        # Active polyhedra will be re-enabled below.
        # ----------------------------------------------------------

        for actor in self._actors.values():
            actor.SetVisibility(False)

        active = set()

        for poly in polyhedra:

            points = np.asarray(
                [a.position for a in poly.neighbors],
                dtype=float,
            )

            if len(points) < 4:
                continue

            try:
                hull = ConvexHull(points)

            except Exception:
                continue

            faces = []

            for tri in hull.simplices:

                faces.extend(
                    (
                        3,
                        tri[0],
                        tri[1],
                        tri[2],
                    )
                )

            mesh = pv.PolyData(
                points,
                np.asarray(
                    faces,
                    dtype=np.int64,
                ),
            )

            center = poly.center

            actor_name = (
                f"poly_"
                f"{center.original_index}_"
                f"{center.tx}_"
                f"{center.ty}_"
                f"{center.tz}"
            )

            active.add(actor_name)

            color = get_color(
                center.symbol,
                settings.color_scheme,
            )

            # --------------------------------------------------
            # Existing actor
            # --------------------------------------------------

            if actor_name in self._actors:

                existing = self._meshes[actor_name]

                existing.copy_from(mesh)

                existing.Modified()

                self._actors[actor_name].SetVisibility(True)

                continue
            # --------------------------------------------------
            # New actor
            # --------------------------------------------------

            actor = viewer.add_mesh(

                mesh,

                color=color,

                opacity=settings.polyhedron_opacity,

                smooth_shading=True,

                show_edges=settings.polyhedron_show_edges,

                edge_color="black",

                name=actor_name,

                render=False,

            )

            self._actors[actor_name] = actor

            self._meshes[actor_name] = mesh

            actor.SetVisibility(True)

        # ----------------------------------------------------------
        # Remove obsolete actors
        # ----------------------------------------------------------

        obsolete = set(self._actors) - active

        for name in obsolete:

            try:
                viewer.remove_actor(
                    self._actors[name],
                    render=False,
                )
            except Exception:
                pass

            del self._actors[name]

            del self._meshes[name]

    # --------------------------------------------------------------
    # Hide
    # --------------------------------------------------------------

    def hide(self, viewer) -> None:
        """
        Hide all polyhedra without deleting them.
        """

        for actor in self._actors.values():
            actor.SetVisibility(False)


    # --------------------------------------------------------------
    # Clear
    # --------------------------------------------------------------

    def clear(self, viewer) -> None:
        """
        Remove all persistent actors.
        """

        for actor in self._actors.values():

            try:
                viewer.remove_actor(actor)
            except Exception:
                pass

        self._actors.clear()
        self._meshes.clear()

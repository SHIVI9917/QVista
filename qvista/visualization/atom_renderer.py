"""
High-performance atom renderer using PyVista glyphs.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pyvista as pv

from qvista.visualization.base_renderer import BaseRenderer
from qvista.visualization.colors import get_color
from qvista.visualization.radii import (
    covalent_radius,
    vdw_radius,
)
from qvista.visualization.settings import Representation

class AtomRenderer(BaseRenderer):
    """
    Render atoms using one glyph actor per element.
    """

    def __init__(self) -> None:

        # Cached sphere meshes
        self._sphere_cache: dict[
            tuple[float, int],
            pv.PolyData,
        ] = {}

        self._sphere_resolution = 32

    # ------------------------------------------------------------------
    # Sphere cache
    # ------------------------------------------------------------------

    def _get_sphere(
        self,
        radius: float,
    ) -> pv.PolyData:

        key = (
            round(radius, 6),
            self._sphere_resolution,
        )

        sphere = self._sphere_cache.get(key)

        if sphere is None:

            sphere = pv.Sphere(
                radius=radius,
                theta_resolution=self._sphere_resolution,
                phi_resolution=self._sphere_resolution,
            )

            self._sphere_cache[key] = sphere

        return sphere

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(
        self,
        viewer,
        scene_data,
    ) -> None:
        """
        Render atoms from SceneData.
        """

        atoms = scene_data.visible_atoms

        if not atoms:
            return

        settings = scene_data.settings

        if settings is None:
            return

        # ----------------------------------------------------------
        # In polyhedral mode, hide cations that are polyhedron
        # centers - they sit inside their shaded polyhedron and
        # would otherwise be drawn as opaque spheres on top of it.
        # ----------------------------------------------------------

        hidden_keys = set()

        if settings.representation == Representation.POLYHEDRAL:

            for poly in scene_data.polyhedra:

                center = poly.center

                hidden_keys.add(
                    (
                        center.original_index,
                        center.tx,
                        center.ty,
                        center.tz,
                    )
                )

        # ----------------------------------------------------------
        # Group atoms by element
        # ----------------------------------------------------------

        positions_by_symbol = defaultdict(list)

        for atom in atoms:

            if (
                atom.original_index,
                atom.tx,
                atom.ty,
                atom.tz,
            ) in hidden_keys:
                continue

            positions_by_symbol[
                atom.symbol
            ].append(
                atom.position
            )

        # ----------------------------------------------------------
        # One glyph actor per element
        # ----------------------------------------------------------

        for symbol, positions in positions_by_symbol.items():

            positions = np.asarray(
                positions,
                dtype=float,
            )

            cloud = pv.PolyData(
                positions,
            )

            # Representation

            if (
                settings.representation
                == Representation.SPACE_FILL
            ):

                radius = (
                    settings.atom_radius_scale
                    * vdw_radius(symbol)
                )

            else:

                radius = (
                    settings.atom_radius_scale
                    * covalent_radius(symbol)
                )

            sphere = self._get_sphere(
                radius,
            )

            glyphs = cloud.glyph(
                geom=sphere,
                orient=False,
                scale=False,
            )

            actor = viewer.add_mesh(

                glyphs,

                color=get_color(
                    symbol,
                    settings.color_scheme,
                ),

                smooth_shading=True,

                specular=0.25,

                specular_power=20,

                ambient=0.20,

                diffuse=0.80,

                name=f"atoms_{symbol}",

                render=False,
            )

            viewer.cache.add_atom_actor(
                f"atoms_{symbol}",
                actor,
            )

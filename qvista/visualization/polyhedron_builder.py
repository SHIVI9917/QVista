"""
Build coordination polyhedra from visible atoms.
"""

from __future__ import annotations

import numpy as np

from qvista.models.visible_polyhedron import VisiblePolyhedron

from qvista.visualization.radii import covalent_radius

from qvista.visualization.polyhedron_rules import (
    is_default_cation,
    is_default_anion,
)


class PolyhedronBuilder:
    """
    Build coordination polyhedra directly from visible atoms.
    """

    def __init__(
        self,
        scale: float = 1.15,
    ):

        self.scale = scale

    # ------------------------------------------------------------------

    def build(
        self,
        centers_pool,
        settings,
        neighbor_pool=None,
    ):
        """
        Build coordination polyhedra.

        `centers_pool` is the (boundary-clipped) set of atoms that may
        act as polyhedron centers. `neighbor_pool` is searched for
        coordinating ligands and may extend beyond the display
        boundary, so centers near the edge of the box still get their
        full coordination shell. Defaults to `centers_pool` if not
        given.
        """

        if neighbor_pool is None:
            neighbor_pool = centers_pool

        polyhedra = []

        # ----------------------------------------------------------
        # Loop over all possible central atoms
        # ----------------------------------------------------------

        for center in centers_pool:

            # Only default cations can be centers
            if not is_default_cation(center.symbol):
                continue

            center_key = (
                center.original_index,
                center.tx,
                center.ty,
                center.tz,
            )

            neighbors = []

            # ------------------------------------------------------
            # Find coordinating anions
            # ------------------------------------------------------

            for atom in neighbor_pool:

                atom_key = (
                    atom.original_index,
                    atom.tx,
                    atom.ty,
                    atom.tz,
                )

                if atom_key == center_key:
                    continue

                # Ligands must be anions
                if not is_default_anion(atom.symbol):
                    continue

                cutoff = self.scale * (
                    covalent_radius(center.symbol)
                    + covalent_radius(atom.symbol)
                )

                distance = np.linalg.norm(
                    atom.position - center.position
                )

                if distance <= cutoff:

                    neighbors.append(atom)

            # ------------------------------------------------------
            # Need at least four neighbors
            # ------------------------------------------------------

            if len(neighbors) < 4:
                continue

            polyhedra.append(

                VisiblePolyhedron(

                    center=center,

                    neighbors=neighbors,

                )

            )

        return polyhedra


"""
Boundary clipping utilities.
"""

from __future__ import annotations

import numpy as np

from qvista.models.visible_atom import VisibleAtom


class BoundaryClipper:

    def clip(
        self,
        expanded_atoms,
        structure,
        settings,
    ):

        cell = structure.atoms.cell.array

        inv_cell = np.linalg.inv(cell.T)

        b = settings.boundary

        xmin = -b.x_minus
        xmax = b.x_plus

        ymin = -b.y_minus
        ymax = b.y_plus

        zmin = -b.z_minus
        zmax = b.z_plus

        tol = 1e-8

        visible = []

        for atom in expanded_atoms:

            # --------------------------------------------------
            # Always keep atoms belonging to the original cell
            # --------------------------------------------------

            if atom.is_original_cell:
                visible.append(atom)
                continue

            frac = inv_cell @ atom.position

            x, y, z = frac

            if (
                xmin - tol <= x <= xmax + tol
                and ymin - tol <= y <= ymax + tol
                and zmin - tol <= z <= zmax + tol
            ):

                visible.append(atom)

        return visible

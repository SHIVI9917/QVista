"""
Boundary expansion utilities.
"""

from __future__ import annotations

import numpy as np

from qvista.models.visible_atom import VisibleAtom


class BoundaryExpander:
    """
    Expand the structure according to the visualization boundary.

    The original ASE Atoms object is never modified.
    """

    def expand(
        self,
        structure,
        settings,
    ):

        atoms = structure.atoms
        cell = atoms.cell.array

        b = settings.boundary

        # ----------------------------------------------------------
        # Translation range
        #
        # Examples
        #
        # -X = 0, +X = 1   -> cell 0
        # -X = 1, +X = 1   -> cells -1,0
        # -X = 0, +X = 2   -> cells 0,1
        # -X = 1, +X = 2   -> cells -1,0,1
        # ----------------------------------------------------------

        xmin = -int(np.ceil(b.x_minus))
        xmax = int(np.ceil(b.x_plus))

        ymin = -int(np.ceil(b.y_minus))
        ymax = int(np.ceil(b.y_plus))

        zmin = -int(np.ceil(b.z_minus))
        zmax = int(np.ceil(b.z_plus))

        expanded: list[VisibleAtom] = []

        for ix in range(xmin, xmax):
            for iy in range(ymin, ymax):
                for iz in range(zmin, zmax):

                    shift = (
                        ix * cell[0]
                        + iy * cell[1]
                        + iz * cell[2]
                    )

                    for index, atom in enumerate(atoms):

                        expanded.append(

                            VisibleAtom(

                                symbol=atom.symbol,

                                position=np.asarray(
                                    atom.position + shift,
                                    dtype=float,
                                ),

                                original_index=index,

                                tx=ix,
                                ty=iy,
                                tz=iz,

                            )

                        )

        return expanded

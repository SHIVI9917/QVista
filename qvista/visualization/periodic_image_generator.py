"""
Periodic image generator.

Generates only the periodic images required inside the user-defined
fractional display boundary.

This replaces the old BoundaryExpander + BoundaryClipper pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import numpy as np

from qvista.models.visible_atom import VisibleAtom

logger = logging.getLogger(__name__)

#: Below this cell volume, in cubic Angstrom, there is no lattice to
#: repeat -- an XYZ molecule arrives with a cell of all zeros.
MIN_CELL_VOLUME = 1.0e-8


def _has_lattice(cell) -> bool:
    """True when the cell can actually be tiled."""

    return abs(float(np.linalg.det(np.asarray(cell, dtype=float)))) > (
        MIN_CELL_VOLUME
    )


class PeriodicImageGenerator:
    """
    Generate visible periodic images.

    A molecule has no cell, and every step below divides by one: the
    fractional coordinates are undefined, and multiplying them back by a
    zero cell puts every atom on the origin. Structures without a
    lattice are therefore passed straight through, one visible atom per
    atom, which is what "no periodic images" means.
    """

    def generate_padded(
        self,
        structure,
        boundary,
        padding: float = 5.0,
        origin=None,
    ) -> list[VisibleAtom]:
        """
        Generate periodic images within the display boundary, expanded
        by `padding` Angstrom on every side.

        Used to find neighbor atoms (e.g. for coordination polyhedra)
        that lie just outside the displayed boundary - without this,
        atoms near the edge of the display box would appear to be
        missing neighbors that are only just out of view.
        """

        cell = structure.atoms.cell.array

        if not _has_lattice(cell):
            return self.generate(structure, boundary, origin)

        margins = [
            padding / np.linalg.norm(cell[axis])
            for axis in range(3)
        ]

        padded_boundary = replace(
            boundary,
            x_minus=boundary.x_minus + margins[0],
            x_plus=boundary.x_plus + margins[0],
            y_minus=boundary.y_minus + margins[1],
            y_plus=boundary.y_plus + margins[1],
            z_minus=boundary.z_minus + margins[2],
            z_plus=boundary.z_plus + margins[2],
        )

        return self.generate(structure, padded_boundary, origin)

    def generate(
        self,
        structure,
        boundary,
        origin=None,
        tile: bool = True,
    ) -> list[VisibleAtom]:
        """
        The atoms to draw, tiled through the display boundary.

        ``tile=False`` draws each atom exactly once, where it actually
        is, and ignores the boundary. That is what an animated normal
        mode needs. The animation already builds its own repeated cell
        carrying the ``exp(i q.R)`` phase, and tiling that again is not
        merely redundant: a face duplicate is the home atom's displaced
        position plus a lattice vector, so it moves in phase with the
        home cell rather than with its own. At the zone boundary, where
        neighbouring cells are in antiphase, that is the whole of the
        physics being shown -- and the duplicates were visibly wrong,
        snapping about while the atoms they copy moved smoothly.
        """

        atoms = structure.atoms

        cell = atoms.cell.array

        if not tile or not _has_lattice(cell):
            return self._molecule(atoms)

        scaled = atoms.get_scaled_positions()

        # The origin shift is added here, before anything else looks at a
        # fractional coordinate, so that the rest of this method needs to
        # know nothing about it. Wrapping comes free: an atom pushed past
        # the top of the boundary falls outside it and is dropped, while
        # the periodic image one cell below is now inside and is drawn.
        # That is the same machinery that tiles the cell, so a shifted
        # atom reappears at the far side without a special case.
        if origin is not None:

            shift = (
                origin.as_tuple()
                if hasattr(origin, "as_tuple")
                else tuple(origin)
            )

            scaled = scaled + np.asarray(shift, dtype=float)

        # ----------------------------------------------------------
        # Convert GUI boundary to actual fractional limits
        #
        # GUI:
        #     -X = 0.3
        # means
        #     xmin = -0.3
        # ----------------------------------------------------------

        xmin = -boundary.x_minus
        xmax = boundary.x_plus

        ymin = -boundary.y_minus
        ymax = boundary.y_plus

        zmin = -boundary.z_minus
        zmax = boundary.z_plus

        visible: list[VisibleAtom] = []

        seen: set[tuple[int, int, int, int]] = set()

        tol = 1.0e-8

        # ----------------------------------------------------------
        # Loop over original atoms
        # ----------------------------------------------------------

        for index, atom in enumerate(atoms):

            frac = scaled[index]

            # ------------------------------------------------------
            # Translation range
            # ------------------------------------------------------

            tx_min = math.floor(xmin - frac[0])
            tx_max = math.ceil(xmax - frac[0])

            ty_min = math.floor(ymin - frac[1])
            ty_max = math.ceil(ymax - frac[1])

            tz_min = math.floor(zmin - frac[2])
            tz_max = math.ceil(zmax - frac[2])

            # ------------------------------------------------------
            # Generate visible periodic copies
            # ------------------------------------------------------

            for tx in range(tx_min, tx_max + 1):

                for ty in range(ty_min, ty_max + 1):

                    for tz in range(tz_min, tz_max + 1):

                        frac_image = frac + np.array(
                            [tx, ty, tz],
                            dtype=float,
                        )

                        # --------------------------------------------------
                        # Check whether image lies inside boundary
                        # --------------------------------------------------

                        if (
                            frac_image[0] < xmin - tol
                            or frac_image[0] > xmax + tol
                            or frac_image[1] < ymin - tol
                            or frac_image[1] > ymax + tol
                            or frac_image[2] < zmin - tol
                            or frac_image[2] > zmax + tol
                        ):
                            continue

                        key = (
                            index,
                            tx,
                            ty,
                            tz,
                        )

                        if key in seen:
                            continue

                        seen.add(key)

                        # Fractional → Cartesian
                        cart = frac_image @ cell

                        visible.append(

                            VisibleAtom(

                                symbol=atom.symbol,

                                position=np.asarray(
                                    cart,
                                    dtype=float,
                                ),

                                fractional=np.asarray(
                                    frac_image,
                                    dtype=float,
                                ),

                                original_index=index,

                                tx=tx,
                                ty=ty,
                                tz=tz,

                            )

                        )

        # Debug rather than print: a mode animation redraws twenty times
        # a second, and this used to put twenty lines a second on stdout.
        logger.debug("Generated %d visible atoms", len(visible))

        return visible

    # ------------------------------------------------------------

    def _molecule(self, atoms) -> list[VisibleAtom]:
        """
        Every atom once, where it actually is.

        No cell means no fractional coordinates and nothing to repeat,
        so the boundary does not apply either -- there is no direction
        along which to trim.
        """

        return [
            VisibleAtom(
                symbol=atom.symbol,
                position=np.asarray(atom.position, dtype=float),
                fractional=np.zeros(3, dtype=float),
                original_index=index,
                tx=0,
                ty=0,
                tz=0,
            )
            for index, atom in enumerate(atoms)
        ]

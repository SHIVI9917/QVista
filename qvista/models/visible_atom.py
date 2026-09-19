"""
Visible atom used by the visualization pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class VisibleAtom:
    """
    One visible atom in the rendered scene.

    Attributes
    ----------
    symbol
        Chemical symbol.

    position
        Cartesian coordinates (Å).

    fractional
        Fractional coordinates of this visible periodic image.

    original_index
        Index of the atom in the original ASE Atoms object.

    tx, ty, tz
        Periodic translation indices.
        (0,0,0) corresponds to the original unit cell.
    """

    symbol: str

    # Cartesian coordinates
    position: np.ndarray

    # Fractional coordinates
    fractional: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    original_index: int = 0

    tx: int = 0
    ty: int = 0
    tz: int = 0

    @property
    def translation(self) -> tuple[int, int, int]:
        """Periodic image indices."""
        return (self.tx, self.ty, self.tz)

    @property
    def is_original_cell(self) -> bool:
        """True if this atom belongs to the original unit cell."""
        return (
            self.tx == 0
            and self.ty == 0
            and self.tz == 0
        )

"""
Visible coordination polyhedron.
"""

from __future__ import annotations

from dataclasses import dataclass

from qvista.models.visible_atom import VisibleAtom


@dataclass(slots=True)
class VisiblePolyhedron:
    """
    One coordination polyhedron.
    """

    center: VisibleAtom

    neighbors: list[VisibleAtom]
